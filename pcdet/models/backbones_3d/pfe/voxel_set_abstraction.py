import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils
import open3d as o3d
import random
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
import cv2
import math
import matplotlib.pyplot as plt
from plyfile import PlyData
import pandas as pd

from pcdet.models.backbones_3d.completion_models.model import PMPNet as CompletionModel
from pcdet.models.backbones_3d.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()

plydata_car = PlyData.read("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/target_dir/car_2048.ply")
data_car = plydata_car.elements[0].data
target_car = pd.DataFrame(data_car)
plydata_pede = PlyData.read("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/target_dir/pedestrian_512.ply")
data_pede = plydata_pede.elements[0].data
target_pede = pd.DataFrame(data_pede)
plydata_cycl = PlyData.read("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/target_dir/cyclist_512.ply")
data_cycl = plydata_cycl.elements[0].data
target_cycl = pd.DataFrame(data_cycl)


def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)

def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


def sample_points_with_roi(rois, points, points_xyzr, sample_radius_with_roi, num_max_points_of_part=200000):
    """
    Args:
        rois: (M, 7 + C)
        points: (N, 3), points_xyzr: (N, 4)
        sample_radius_with_roi:
        num_max_points_of_part:

    Returns:
        sampled_points: (N_out, 3)
    """
    if points.shape[0] < num_max_points_of_part:
        # print(points.shape, rois.shape)
        distance = (points[:, None, :] - rois[None, :, 0:3]).norm(dim=-1)
        min_dis, min_dis_roi_idx = distance.min(dim=-1)
        roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
        point_mask = min_dis < roi_max_dim + sample_radius_with_roi
        # print(point_mask.detach().cpu().numpy().tolist())
    else:
        start_idx = 0
        point_mask_list = []
        while start_idx < points.shape[0]:
            distance = (points[start_idx:start_idx + num_max_points_of_part, None, :] - rois[None, :, 0:3]).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            cur_point_mask = min_dis < roi_max_dim + sample_radius_with_roi
            point_mask_list.append(cur_point_mask)
            start_idx += num_max_points_of_part
        point_mask = torch.cat(point_mask_list, dim=0)

    sampled_points = points[:1] if point_mask.sum() == 0 else points[point_mask, :]
    if points_xyzr != None:
        sampled_points_xyzr = points_xyzr[:1] if point_mask.sum() == 0 else points_xyzr[point_mask, :]
        return sampled_points, sampled_points_xyzr, point_mask
    return sampled_points, point_mask

def sector_fps(points, points_xyzr, num_sampled_points, num_sectors):
    """
    Args:
        points: (N, 3), points_xyzr: (N, 4)
        num_sampled_points: int
        num_sectors: int

    Returns:
        sampled_points: (N_out, 3)
    """
    sector_size = np.pi * 2 / num_sectors
    point_angles = torch.atan2(points[:, 1], points[:, 0]) + np.pi
    sector_idx = (point_angles / sector_size).floor().clamp(min=0, max=num_sectors)
    xyz_points_list, xyzr_points_list = [], []
    xyz_batch_cnt = []
    num_sampled_points_list = []
    for k in range(num_sectors):
        mask = (sector_idx == k)
        cur_num_points = mask.sum().item()
        if cur_num_points > 0:
            xyz_points_list.append(points[mask])
            xyzr_points_list.append(points_xyzr[mask])
            xyz_batch_cnt.append(cur_num_points)
            ratio = cur_num_points / points.shape[0]
            num_sampled_points_list.append(
                min(cur_num_points, math.ceil(ratio * num_sampled_points))
            )

    if len(xyz_batch_cnt) == 0:
        xyz_points_list.append(points)
        xyzr_points_list.append(points_xyzr)
        xyz_batch_cnt.append(len(points))
        num_sampled_points_list.append(num_sampled_points)
        print(f'Warning: empty sector points detected in SectorFPS: points.shape={points.shape}')

    xyz = torch.cat(xyz_points_list, dim=0)
    xyzr = torch.cat(xyzr_points_list, dim=0)
    xyz_batch_cnt = torch.tensor(xyz_batch_cnt, device=points.device).int()
    sampled_points_batch_cnt = torch.tensor(num_sampled_points_list, device=points.device).int()

    sampled_pt_idxs = pointnet2_stack_utils.stack_farthest_point_sample(
        xyz.contiguous(), xyz_batch_cnt, sampled_points_batch_cnt
    ).long()

    sampled_points = xyz[sampled_pt_idxs]
    sampled_points_xyzr = xyzr[sampled_pt_idxs]

    return sampled_points, sampled_points_xyzr

def in_convex_polyhedron(points_set, test_points):
    """
    检测点是否在凸包内
    :param points_set: 凸包，需要对分区的点进行凸包生成 具体见conv_hull函数
    :param test_points: 需要检测的点 可以是多个点
    :return: bool类型
    """
    assert type(points_set) == np.ndarray
    assert type(points_set) == np.ndarray
    bol = np.zeros((test_points.shape[0], 1), dtype=np.bool)
    ori_set = points_set
    ori_edge_index = conv_hull(ori_set)
    ori_edge_index = np.sort(np.unique(ori_edge_index))
    for i in range(test_points.shape[0]):
        new_set = np.concatenate((points_set, test_points[i, np.newaxis]), axis=0)
        new_edge_index = conv_hull(new_set)
        new_edge_index = np.sort(np.unique(new_edge_index))
        bol[i] = (new_edge_index.tolist() == ori_edge_index.tolist())
    return bol

def conv_hull(points):
    """
    生成凸包 参考文档：https://blog.csdn.net/io569417668/article/details/106274172
    :param points: 待生成凸包的点集
    :return: 索引 list
    """
    pcl = array_to_pointcloud(points)
    hull, lst = pcl.compute_convex_hull()
    return lst

# 这里会返回列表类型
def load_data_txt(path):
    file = open(path, 'r')
    data = file.read().split('\n')
    lst = _data_trans(data)
    return lst
    
def array_to_pointcloud(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd
    
def _data_trans(data):
    lst = []
    for num in data:
        num_list = num.split()
        lst.append([eval(i) for i in num_list])
    lst.pop()
    return lst

def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    # 计算旋转矩阵
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])    # 8个顶点的xyz
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    # 旋转矩阵点乘(3，8)顶点矩阵
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2 += np.vstack([x,y,z])    # (3,8)
    return corners_3d_cam2.T

def normalization_x(x, type='Torch'):
    if type == 'Torch':
        min_x = torch.min(x)
        max_x = torch.max(x)
        norm_x = (x - min_x) / (max_x - min_x + 1e-4)
    elif type == 'Numpy':
        min_x = np.min(x)
        max_x = np.max(x)
        norm_x = (x - min_x) / (max_x - min_x + 1e-4)
    return norm_x

def SelectClosedProposal(rois_b, rois_labels_b, rois_scores_selected):
    roi_local_label, rois_b_new, rois_scores_new, rois_labels_new, count = [], [], [], [], 0
    rois_b_center, rois_labels_center, rois_scores_selected_center = rois_b.numpy(), rois_labels_b.numpy(), rois_scores_selected.numpy()
    while rois_b_center.shape[0] != 0:
        put_index = []
        if rois_b_center.shape[0] > -1:
            # print(rois_b_center.shape, rois_scores_selected_center.shape)
            rois_b_center_part1, rois_b_center_part2, rois_scores_part1, rois_scores_part2, rois_labels_part1, rois_labels_part2 = rois_b_center[0], rois_b_center[1:, :], rois_scores_selected_center[0], rois_scores_selected_center[1:], rois_labels_center[0], rois_labels_center[1:]
            for j in range(0, rois_b_center_part2.shape[0]):
                if np.absolute(np.linalg.norm(rois_b_center_part1[:3] - rois_b_center_part2[j][:3])) < 0.6:
                    rois_b_new.append(rois_b_center_part2[j])
                    rois_scores_new.append(rois_scores_part2[j])
                    rois_labels_new.append(rois_labels_part2[j])
                    roi_local_label.append(count)
                    put_index.append(j + 1)
            rois_b_new.append(rois_b_center_part1)
            rois_scores_new.append(rois_scores_part1)
            rois_labels_new.append(rois_labels_part1)
            roi_local_label.append(count)
            count += 1
            put_index.append(0)
        else:
            rois_b_new.append(rois_b_center[0])
            rois_scores_new.append(rois_scores_selected_center[0])
            rois_labels_new.append(rois_labels_center[0])
            roi_local_label.append(count)
            put_index = [0]
        # 删除筛选出的numpy行
        put_index_sort = sorted(put_index, reverse=True)
        for item in put_index_sort:
            rois_b_center = np.delete(rois_b_center, item, axis=0)
            rois_scores_selected_center = np.delete(rois_scores_selected_center, item, axis=0)
            rois_labels_center = np.delete(rois_labels_center, item, axis=0)
    return rois_b_new, rois_scores_new, rois_labels_new, roi_local_label

def SphereDistance(point1, point2, C):
    distance = (point1[:, None, :] - point2[None, :, 0:3]).norm(dim=-1)
    min_dis, min_dis_idx = distance.min(dim=-1)
    roi_max_dim = (point2[min_dis_idx, 3:6] / 3).norm(dim=-1)
    point_mask = min_dis < roi_max_dim + C
    return distance, min_dis, min_dis_idx, point_mask

def single_rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """

    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros_like(angle)
    ones = np.ones_like(angle)
    rot_matrix = np.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=0).reshape(3, 3)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    return points_rot

def pc_normalize(pc):
    """
    对点云数据进行归一化
    :param pc: 需要归一化的点云数据
    :return: 归一化后的点云数据
    """
    # 求质心，也就是一个平移量，实际上就是求均值
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # 对点云进行缩放
    pc = pc / m
    return pc

def pc_denormalize(pc_nor, pc_raw):
    centroid = np.mean(pc_raw, axis=0)
    pc_raw = pc_raw - centroid
    m = np.max(np.sqrt(np.sum(pc_raw ** 2, axis=1)))
    pc_denor = pc_nor * m + centroid
    return pc_denor

class VoxelSetAbstraction(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR

            if SA_cfg[src_name].get('INPUT_CHANNELS', None) is None:
                input_channels = SA_cfg[src_name].MLPS[0][0] \
                    if isinstance(SA_cfg[src_name].MLPS[0], list) else SA_cfg[src_name].MLPS[0]
            else:
                input_channels = SA_cfg[src_name]['INPUT_CHANNELS']

            cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=input_channels, config=SA_cfg[src_name]
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += cur_num_c_out

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            self.SA_rawpoints, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=num_rawpoint_features - 3, config=SA_cfg['raw_points']
            )

            c_in += cur_num_c_out

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in
        self.completion_model = CompletionModel()
        self.completion_dict = {}


    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride, weights=None):
        """
        Args:
            keypoints: (N1 + N2 + ..., 4)
            bev_features: (B, C, H, W)
            batch_size:
            bev_stride:

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]

        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            bs_mask = (keypoints[:, 0] == k)

            cur_x_idxs = x_idxs[bs_mask]
            cur_y_idxs = y_idxs[bs_mask]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
        
        # if weights != None:
        #     point_bev_features = point_bev_features.permute(1, 0).unsqueeze(0)
        #     point_bev_features = self.BEV_ReshapeWeight1(point_bev_features)
        #     weights_torch = torch.cat((weights), 1).unsqueeze(1).expand(1, point_bev_features.shape[1], point_bev_features.shape[2]).cuda()
        #     point_bev_features = point_bev_features * weights_torch
        #     point_bev_features = self.BEV_ReshapeWeight2(point_bev_features)
        #     point_bev_features = point_bev_features.squeeze(dim=0).permute(1, 0)
            
        return point_bev_features

    def sectorized_proposal_centric_sampling(self, roi_boxes, points, points_xyzr):
        """
        Args:
            roi_boxes: (M, 7 + C)
            points: (N, 3)

        Returns:
            sampled_points: (N_out, 3)
        """

        sampled_points, sampled_points_xyzr, _ = sample_points_with_roi(
            rois=roi_boxes, points=points, points_xyzr=points_xyzr,
            sample_radius_with_roi=self.model_cfg.SPC_SAMPLING.SAMPLE_RADIUS_WITH_ROI,
            num_max_points_of_part=self.model_cfg.SPC_SAMPLING.get('NUM_POINTS_OF_EACH_SAMPLE_PART', 200000)
        )
        sampled_points, sampled_points_xyzr = sector_fps(
            points=sampled_points, points_xyzr=sampled_points_xyzr, num_sampled_points=self.model_cfg.NUM_KEYPOINTS,
            num_sectors=self.model_cfg.SPC_SAMPLING.NUM_SECTORS
        )
        return sampled_points, sampled_points_xyzr

    def get_sampled_points(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:
            keypoints: (N1 + N2 + ..., 4), where 4 indicates [bs_idx, x, y, z]
        """
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            src_points_xyzr = batch_dict['points'][:, 1:5]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list, keypoints_xyzr_list = [], []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            sampled_points_xyzr = src_points_xyzr[bs_mask].unsqueeze(dim=0) # (1, N, 4)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    times = int(self.model_cfg.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0) # torch.Size([1, 2048, 3])

            elif self.model_cfg.SAMPLE_METHOD == 'SPC':
                # print(sampled_points)
                cur_keypoints, cur_keypoints_xyzr = self.sectorized_proposal_centric_sampling(
                    roi_boxes=batch_dict['rois'][bs_idx], points=sampled_points[0], points_xyzr=sampled_points_xyzr[0]
                )
                bs_idxs = cur_keypoints.new_ones(cur_keypoints.shape[0]) * bs_idx
                bs_idxs_xyzr = cur_keypoints_xyzr.new_ones(cur_keypoints_xyzr.shape[0]) * bs_idx
                keypoints = torch.cat((bs_idxs[:, None], cur_keypoints), dim=1)
                keypoints_xyzr = torch.cat((bs_idxs_xyzr[:, None], cur_keypoints_xyzr), dim=1)
            else:
                raise NotImplementedError
            keypoints_list.append(keypoints)
            keypoints_xyzr_list.append(keypoints_xyzr.detach().cpu())

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3) or (N1 + N2 + ..., 4)
        if len(keypoints.shape) == 3:
            batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1, 1)
            keypoints = torch.cat((batch_idx.float(), keypoints.view(-1, 3)), dim=1)
            keypoints_xyzr = torch.cat((batch_idx.float(), keypoints_xyzr.view(-1, 3)), dim=1)

        return keypoints, keypoints_list, keypoints_xyzr_list

    @staticmethod
    def aggregate_keypoint_features_from_one_source(
            batch_size, aggregate_func, xyz, xyz_features, xyz_bs_idxs, new_xyz, new_xyz_batch_cnt, potential_occluded_points_list, weights=None, 
            filter_neighbors_with_roi=False, radius_of_neighbor=None, num_max_points_of_part=200000, rois=None
    ):
        """

        Args:
            aggregate_func:
            xyz: (N, 3)
            xyz_features: (N, C)
            xyz_bs_idxs: (N)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size), [N1, N2, ...]

            filter_neighbors_with_roi: True/False
            radius_of_neighbor: float
            num_max_points_of_part: int
            rois: (batch_size, num_rois, 7 + C)
        Returns:

        """
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        if filter_neighbors_with_roi: # PV RCNN++
            point_features = torch.cat((xyz, xyz_features), dim=-1) if xyz_features is not None else xyz
            point_features_list = []
            if potential_occluded_points_list != None and weights == None:
                for bs_idx in range(batch_size):
                    bs_mask = (xyz_bs_idxs == bs_idx)
                    points_xyzr = None
                    _, valid_mask = sample_points_with_roi(
                        rois=rois[bs_idx], points=xyz[bs_mask], points_xyzr=points_xyzr,
                        sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part,
                    )
                    if potential_occluded_points_list[bs_idx] != [0]:
                        point_features_new = torch.cat((point_features[bs_mask][valid_mask], potential_occluded_points_list[bs_idx][:, 1:].cuda()))
                        point_features_list.append(point_features_new)
                    else:
                        point_features_list.append(point_features[bs_mask][valid_mask])
                    xyz_batch_cnt[bs_idx] = valid_mask.sum()
            else:
                for bs_idx in range(batch_size):
                    bs_mask = (xyz_bs_idxs == bs_idx)
                    points_xyzr = None
                    _, valid_mask = sample_points_with_roi(
                        rois=rois[bs_idx], points=xyz[bs_mask], points_xyzr=points_xyzr,
                        sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part,
                    )
                    point_features_list.append(point_features[bs_mask][valid_mask])
                    xyz_batch_cnt[bs_idx] = valid_mask.sum()

            valid_point_features = torch.cat(point_features_list, dim=0)
            xyz = valid_point_features[:, 0:3]
            xyz_features = valid_point_features[:, 3:] if xyz_features is not None else None
        else: # PV RCNN
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (xyz_bs_idxs == bs_idx).sum()
        pooled_points, pooled_features = aggregate_func(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=xyz_features.contiguous(),
            weights=weights,
        )
        return pooled_features
      
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints, keypoints_list, keypoints_xyzr_list = self.get_sampled_points(batch_dict)
        ###############################################
        # print(batch_dict['rois'].shape, keypoints.shape) # [bs, 128, 7], [2049*bs, 4]
        # print(keypoints_list[0].shape, keypoints_list[1].shape) # [2049, 4], [2049*bs, 4]
        # print(batch_dict)
        epoch_index, iteration_index = batch_dict['epoch_index'], batch_dict['iteration_index']
        potential_occluded_points_list, weights_all_points_all_batch = None, None
        batch_dict['weights_all_points_all_batch'] = None
        if self.training:
            roi_targets_dict = batch_dict['roi_targets_dict'] # dict_keys(['rois', 'gt_of_rois', 'gt_iou_of_rois', 'roi_scores', 'roi_labels', 'reg_valid_mask', 'rcnn_cls_labels', 'gt_of_rois_src'])
            roi_labels = roi_targets_dict['roi_labels'].detach().cpu() # [bs, 128]
            roi_scores = roi_targets_dict['roi_scores'].detach().cpu() # [bs, 128]
            rois = roi_targets_dict['rois'].detach().cpu()
            gt_of_rois = roi_targets_dict['gt_of_rois_src'].detach().cpu()
            # print(roi_targets_dict['roi_scores'].shape, roi_targets_dict['roi_labels'].shape, roi_targets_dict['rcnn_cls_labels'])
            # print('\n')
        else:
            roi_scores = batch_dict['roi_scores'].detach().cpu()
            roi_labels = batch_dict['roi_labels'].detach().cpu()
            rois = batch_dict['rois'].detach().cpu()
            # print(batch_dict['roi_labels'].shape)
        # print(batch_dict['points'].shape)
        if (epoch_index > batch_dict['start_training_epoch'] or epoch_index == -1):
        # if epoch_index > 80000:
            keypoints_list_new = []
            potential_occluded_points_list = []
            weights_all_points_all_batch = []
            points_completion_list, deltas_list, dir_list, roi_label_list = [], [], [], []
            for b in range(batch_dict['batch_size']):
                # 选取score分数topk个proposal
                roi_scores_sorted, idx_roi_scores = torch.sort(roi_scores[b], descending=True)
                # # score 归一化
                min_a = torch.min(roi_scores_sorted)
                max_a = torch.max(roi_scores_sorted)
                roi_scores_sorted = (roi_scores_sorted - min_a) / (max_a - min_a)
                # selected_index = [index0 for (index0, value0) in enumerate(roi_scores_sorted) if value0 > 0.2 and value0 < 0.8]
                selected_index = [index0 for (index0, value0) in enumerate(roi_scores_sorted) if value0 > 0.5]
                rois_scores_selected = roi_scores_sorted[selected_index]
                idx = idx_roi_scores[selected_index]
                if idx.shape[0] == 0:
                    topk = 10
                    rois_scores_selected = roi_scores_sorted[:topk]
                    idx = idx_roi_scores[:topk]
                    
                keypoints_b, rois_b, rois_labels_b = keypoints_xyzr_list[b], rois[b][idx], roi_labels[b][idx] # 每个batch的keypoints和topk个rois [2049, 4] [20, 7] [20]
                if self.training:
                    gt_of_rois_b = gt_of_rois[b]
                # 找到距离相近的proposal为一类
                rois_b_new, rois_scores_new, rois_labels_new, roi_local_label = SelectClosedProposal(rois_b, rois_labels_b, rois_scores_selected) # input: cpu() # print(np.array(rois_b_new).shape, np.array(rois_scores_new).shape, np.array(rois_labels_new).shape)
                
                # 利用所有roi进行筛选
                keypoints_b_selected = keypoints_b[:, 1:4]
                rois_b_new_torch = torch.tensor(np.array(rois_b_new))
                _, _, min_dis_roi_idx_1, point_mask_all_roi = SphereDistance(keypoints_b_selected, rois_b_new_torch, 0.4)
                
                # 利用去平均后的roi进行筛选
                static_roi_local_label = {}
                for item in roi_local_label:
                    static_roi_local_label.update({item:roi_local_label.count(item)})
                rois_b_selected, rois_scores_new_selected, rois_labels_new_selected = [], [], []
                for i in range(roi_local_label[-1] + 1):
                    if static_roi_local_label[i] == 1: # 只有一个
                        sigle_roi_index = roi_local_label.index(i)
                        rois_b_selected.append(rois_b_new[sigle_roi_index])
                        rois_scores_new_selected.append(rois_scores_new[sigle_roi_index])
                        rois_labels_new_selected.append(rois_labels_new[sigle_roi_index])
                    else: # 距离相近的proposal，取平均
                        multi_roi_index_start = roi_local_label.index(i)
                        multi_roi_index = []
                        for m in range(static_roi_local_label[i]):
                            multi_roi_index.append(multi_roi_index_start + m)
                        rois_b_tmp = np.array(rois_b_new)[multi_roi_index] # [multi_roi_index, 7]
                        rois_scores_tmp = np.array(rois_scores_new)[multi_roi_index] # [multi_roi_index]
                        rois_labels_temp = np.array(rois_labels_new)[multi_roi_index]
                        rois_b_mean = np.mean(rois_b_tmp, axis=0)
                        rois_scores_mean = np.mean(rois_scores_tmp, axis=0)
                        rois_scores_max = np.max(rois_labels_temp)
                        rois_b_selected.append(rois_b_mean)
                        rois_scores_new_selected.append(rois_scores_mean)
                        rois_labels_new_selected.append(rois_scores_max)
                rois_b_selected_torch = torch.from_numpy(np.array(rois_b_selected))
                keypoints_b_selected = keypoints_b[:, 1:4]
                _, _, min_dis_roi_idx_2, point_mask_mean_roi = SphereDistance(keypoints_b_selected, rois_b_selected_torch, 0.4)
                
                fg_points_selected_falsepos, fg_points_selected_index_basedAllRoi = keypoints_b[point_mask_all_roi], min_dis_roi_idx_1[point_mask_all_roi]
                fg_points_selected, fg_points_selected_index_basedMeanRoi = keypoints_b[point_mask_mean_roi], min_dis_roi_idx_2[point_mask_mean_roi] 
                bg_points_selected = keypoints_b[point_mask_mean_roi==False]
                
                # np.save("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/keypoints/points_before_" + str(epoch_index) + "_" + str(iteration_index) + ".npy", fg_points_selected.detach().cpu().numpy())
                # np.save("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/rois/rois_before_" + str(epoch_index) + "_" + str(iteration_index) + ".npy", rois_b.detach().cpu().numpy())  
                # np.save("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/rois/gt_of_rois_before_" + str(epoch_index) + "_" + str(iteration_index) + ".npy", gt_of_rois_b.numpy())
                # np.save("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/keypoints/PointCloud_" + str(epoch_index) + "_" + str(iteration_index) + ".npy", batch_dict['points'][:, 1:4].detach().cpu().numpy())
                
                fg_points_selected_new, occluded_points_list = [], [] # points
                weights_fg_points_list = []
                # print(rois_scores_new_selected_torch.shape, roi_local_label[-1] + 1, rois_scores_new_selected_torch)
                # print(roi_local_label[-1] + 1, len(rois_labels_new_selected))
                for i in range(roi_local_label[-1] + 1): # 以roi_local_label最后一个数 -- 去除false postive的roi的个数, 循环每一个roi
                    false_positive_start_index = roi_local_label.index(i)
                    roi_label = rois_labels_new_selected[i]
                    false_positive_point_mask_list = []
                    for j in range(static_roi_local_label[i]): 
                        false_positive_point_mask = [index1 for (index1, value1) in enumerate(fg_points_selected_index_basedAllRoi) if value1 == j + false_positive_start_index]
                        false_positive_point_mask_list.append(false_positive_point_mask)
                    fg_of_rois_b_selected_index = [index2 for (index2, value2) in enumerate(fg_points_selected_index_basedMeanRoi) if value2 == i]
                    fg_of_rois_b_selected = fg_points_selected[fg_of_rois_b_selected_index]

                    # voting
                    [roi_x_c, roi_y_c, roi_z_c, roi_dx, roi_dy, roi_dz, roi_angle] = rois_b_selected[i]
                    if roi_label == 2 or roi_label == 3: # if roi_dz > np.max([roi_dx, roi_dy]): # 竖直圆柱
                        cylinder_r = np.sqrt(roi_dx ** 2 + roi_dy ** 2) / 2
                    else: # 水平圆柱
                        if roi_dx < roi_dy:
                            cylinder_r = np.sqrt(roi_dx ** 2 + roi_dz ** 2) / 2
                        else:
                            cylinder_r = np.sqrt(roi_dy ** 2 + roi_dz ** 2) / 2
                    occluded_points_one_box, proposal2points_score_one_box = [], []
                    # proposal2points_score_one_box_fg = rois_scores_new_selected[i] * torch.ones((1, fg_of_rois_b_selected.shape[0]))
                    # distance2points_score_one_box_fg_d = torch.sqrt((roi_x_c - fg_of_rois_b_selected[:, 1]) ** 2 + (roi_y_c - fg_of_rois_b_selected[:, 2]) ** 2)
                    symmetry_type = 'Central'
                    for p in range(len(false_positive_point_mask_list)): # for all false positives
                        for q in range(len(false_positive_point_mask_list[p])): # for all points in one false positive
                            random_mirror = random.random()
                            # if_in = in_convex_polyhedron(compute_3d_box_cam2(roi_dz, roi_dy, roi_dx, roi_x_c, roi_y_c, roi_z_c - roi_dz / 2, roi_angle), np.expand_dims(fg_points_selected_falsepos[q][1:4].numpy(), 0))
                            if (false_positive_point_mask_list[p][q] in fg_of_rois_b_selected_index and random_mirror <= 1) or (false_positive_point_mask_list[p][q] not in fg_of_rois_b_selected_index and random_mirror <= 1):
                                [point_n, point_x, point_y, point_z, point_r] = fg_points_selected_falsepos[false_positive_point_mask_list[p][q]]
                                # if point_r > 0.2:
                                if symmetry_type == 'Central':
                                    point_x_new = 2 * roi_x_c - point_x
                                    point_y_new = 2 * roi_y_c - point_y
                                    M = 4 * (point_x ** 2 + point_y ** 2) * point_r
                                    point_r_new = M / (4 * (point_x_new ** 2 + point_y_new ** 2))
                                elif symmetry_type == 'Mirror':
                                    if roi_angle > np.pi / 2:
                                        kl = -1 / np.tan(roi_angle)
                                    else:
                                        kl = 1 / np.tan(roi_angle)
                                    bl = roi_x_c + kl * roi_y_c
                                    point_y_new = -(-point_y + point_x * kl - kl * bl) / (kl * kl + 1)
                                    point_x_new = (-kl * point_y - kl * kl * point_x + kl * bl + bl) / (kl * kl + 1)
                                    point_r_new = point_r
                                    
                                occluded_points_one_box.append(torch.from_numpy(np.array([[point_n, point_x_new, point_y_new, point_z, point_r_new]])))
                                proposal2points_score_one_box.append(rois_scores_new_selected[i])
                                # distance2points_score_one_box1.append(min_dis_basedAllRoi[false_positive_point_mask_list[p][q]])

                    if len(occluded_points_one_box) != 0 and len(occluded_points_one_box) != 1:
                        # 形状先验可以加在这里！单个物体
                        # 完整物体keypoints # target_code_name: target_car
                        # if roi_label == 1:
                        distance2points_score_one_box, if_in_cylinder_mask = [], []
                        whole_object_kp = torch.cat((torch.cat(occluded_points_one_box), fg_of_rois_b_selected))[:, 1:]
                        bs_whole_object_kp = torch.cat((torch.cat(occluded_points_one_box), fg_of_rois_b_selected))[:, 0]
                        f_whole_object_kp = torch.cat((torch.cat(occluded_points_one_box), fg_of_rois_b_selected))[:, 4]
                        whole_object_kp[:, 0] = whole_object_kp[:, 0] - roi_x_c
                        whole_object_kp[:, 1] = whole_object_kp[:, 1] - roi_y_c
                        whole_object_kp[:, 2] = whole_object_kp[:, 2] - roi_z_c
                        whole_object_kp = single_rotate_points_along_z(whole_object_kp.numpy(), -roi_angle)
                        # whole_object_kp_nor = pc_normalize(whole_object_kp)
                        points = torch.from_numpy(whole_object_kp.reshape(-1, 3)[:, 0:3]).unsqueeze(0).cuda()
                        # fg_of_rois_b_selected_torch = fg_of_rois_b_selected[:, 1:4].unsqueeze(0).cuda()
                        points_completion, deltas, raw_after, completion_after = self.completion_model(points, len(occluded_points_one_box)) ###################################################### completion model, input: [bs, points_number, 3]
                        points_completion_list.append(points_completion)
                        deltas_list.append(deltas)
                        dir_list.append(roi_angle)
                        roi_label_list.append(roi_label)
                        # points_completion_denor = pc_denormalize(points_completion[-1].squeeze(0).detach().cpu().numpy(), whole_object_kp)
                        whole_object_kp1 = single_rotate_points_along_z(points_completion[-1].squeeze(0).detach().cpu().numpy(), roi_angle)
                        whole_object_kp1[:, 0] = whole_object_kp1[:, 0] + roi_x_c
                        whole_object_kp1[:, 1] = whole_object_kp1[:, 1] + roi_y_c
                        whole_object_kp1[:, 2] = whole_object_kp1[:, 2] + roi_z_c
                        whole_object_kp1 = torch.from_numpy(whole_object_kp1)
                        occ = whole_object_kp1[:len(occluded_points_one_box), :]

                        for o0 in range(occ.shape[0]):
                            [point_x_new, point_y_new, point_z] = occ[o0]
                            if roi_label == 2 or roi_label == 3:  # if roi_dz > np.max([roi_dx, roi_dy]):
                                # 到proposal竖直中心线的距离
                                l1 = torch.sqrt((roi_x_c - point_x_new) ** 2 + (roi_y_c - point_y_new) ** 2)
                                distance2points_score_one_box.append(l1)
                                if l1 < torch.tensor(cylinder_r):
                                    if_in_cylinder_mask.append(True)
                                else:
                                    if_in_cylinder_mask.append(False)
                            else:
                                # 到proposal水平中心线距离
                                l2 = torch.sqrt((roi_x_c - point_x_new) ** 2 + (roi_y_c - point_y_new) ** 2) * torch.abs(torch.sin(torch.tensor(roi_angle)))
                                distance2points_score_one_box.append(l2)
                                if l2 < torch.tensor(cylinder_r):
                                    if_in_cylinder_mask.append(True)
                                else:
                                    if_in_cylinder_mask.append(False)
                        whole_object_kp_one_object = torch.cat((torch.cat((bs_whole_object_kp.unsqueeze(1), whole_object_kp1), 1), f_whole_object_kp.unsqueeze(1)), 1)
                        occ = whole_object_kp_one_object[:len(occluded_points_one_box), :]
                        fg = whole_object_kp_one_object[len(occluded_points_one_box):, :]
                        
                        fg_points_selected_new.append(fg)
                        weights_fg_points_list.append(torch.ones(1, fg.shape[0]).cuda())
                        if len([n for n in if_in_cylinder_mask if n == True]) != 0:
                            kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(occ.numpy()[if_in_cylinder_mask])
                            log_density = kde.score_samples(occ.numpy()[if_in_cylinder_mask])
                            density_true_recips = 10 ** (log_density)
                            density_true = normalization_x(density_true_recips, type='Numpy')
                            density = np.mean(density_true) * np.ones_like([np.array(if_in_cylinder_mask)])
                            density[0][if_in_cylinder_mask] = density_true

                            fg_points_selected_new.append(occ) # f 直接copy过来
                            occluded_points_list.append(occ)

                            # 计算权重
                            proposal2points_score_one_box_torch, distance2points_score_one_box_torch = torch.from_numpy(np.array(proposal2points_score_one_box)), torch.from_numpy(np.array(distance2points_score_one_box))

                            distance2points_score_one_box_torch_recip1 = 1.0 / (distance2points_score_one_box_torch + 1e-8)
                            distance2points_score_one_box_torch = normalization_x(distance2points_score_one_box_torch_recip1, type='Torch')
                            weight_matrix = (distance2points_score_one_box_torch.reshape(-1, 1) * 0.4 + torch.from_numpy(density).float().reshape(-1, 1) * 0.6).cuda()  # torchsize(n, 1)
                            weights_fg_points_list.append(weight_matrix.squeeze(1).unsqueeze(0))
                            
#                             # visualization
#                             pcd = o3d.geometry.PointCloud()
#                             pcd.points = o3d.utility.Vector3dVector(torch.cat(occluded_points_one_box)[:, 1:4].detach().cpu().numpy())
#                             o3d.io.write_point_cloud("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/points/occ" + str(i) + ".ply", pcd)
#                             pcd = o3d.geometry.PointCloud()
#                             pcd.points = o3d.utility.Vector3dVector(fg_of_rois_b_selected[:, 1:4].detach().cpu().numpy())
#                             o3d.io.write_point_cloud("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/points/raw" + str(i) + ".ply", pcd)

#                             pcd = o3d.geometry.PointCloud()
#                             pcd.points = o3d.utility.Vector3dVector(raw_after[0].squeeze(0).detach().cpu().numpy())
#                             o3d.io.write_point_cloud("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/points/raw_after1_" + str(i) + ".ply", pcd)
#                             pcd = o3d.geometry.PointCloud()
#                             pcd.points = o3d.utility.Vector3dVector(raw_after[1].squeeze(0).detach().cpu().numpy())
#                             o3d.io.write_point_cloud("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/points/raw_after2_" + str(i) + ".ply", pcd)
#                             pcd = o3d.geometry.PointCloud()
#                             pcd.points = o3d.utility.Vector3dVector(completion_after[0].squeeze(0).detach().cpu().numpy())
#                             o3d.io.write_point_cloud("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/points/completion_after1_" + str(i) + ".ply", pcd)
#                             pcd = o3d.geometry.PointCloud()
#                             pcd.points = o3d.utility.Vector3dVector(completion_after[-1].squeeze(0).detach().cpu().numpy())
#                             o3d.io.write_point_cloud("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/points/completion_after2_" + str(i) + ".ply", pcd)

                            # # visualization
                            # pcd = o3d.geometry.PointCloud()
                            # pcd.points = o3d.utility.Vector3dVector(points.squeeze(0).detach().cpu().numpy())
                            # o3d.io.write_point_cloud("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/points/whole_before" + str(i) + ".ply", pcd)
                            # pcd = o3d.geometry.PointCloud()
                            # pcd.points = o3d.utility.Vector3dVector(points_completion[-1].squeeze(0).detach().cpu().numpy())
                            # o3d.io.write_point_cloud("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/points/whole_after" + str(i) + ".ply", pcd)
                        # else:
                        #     distance2points_score_one_box, if_in_cylinder_mask = [], []
                        #     occluded_points_one_box_torch = torch.cat(occluded_points_one_box)
                        #     for o1 in range(occluded_points_one_box_torch.shape[0]):
                        #         [point_n, point_x_new, point_y_new, point_z, point_r_new] = occluded_points_one_box_torch[o1]
                        #         if roi_label == 2 or roi_label == 3:  # if roi_dz > np.max([roi_dx, roi_dy]):
                        #             # 到proposal竖直中心线的距离
                        #             l1 = torch.sqrt((roi_x_c - point_x_new) ** 2 + (roi_y_c - point_y_new) ** 2)
                        #             distance2points_score_one_box.append(l1)
                        #             if l1 < torch.tensor(cylinder_r):
                        #                 if_in_cylinder_mask.append(True)
                        #             else:
                        #                 if_in_cylinder_mask.append(False)
                        #         else:
                        #             # 到proposal水平中心线距离
                        #             # fi = np.arctan(np.abs((point_x - roi_x_c) / (point_y - roi_y_c)))
                        #             # l2 = torch.sqrt((roi_x_c - point_x_new) ** 2 + (roi_y_c - point_y_new) ** 2) * torch.abs(torch.sin(torch.tensor(roi_angle) - fi))
                        #             l2 = torch.sqrt((roi_x_c - point_x_new) ** 2 + (roi_y_c - point_y_new) ** 2) * torch.abs(torch.sin(torch.tensor(roi_angle)))
                        #             distance2points_score_one_box.append(l2)
                        #             if l2 < torch.tensor(cylinder_r):
                        #                 if_in_cylinder_mask.append(True)
                        #             else:
                        #                 if_in_cylinder_mask.append(False)
                        #     # calculate density
                        #     fg_points_selected_new.append(fg_of_rois_b_selected)
                        #     weights_fg_points_list.append(torch.ones(1, fg_of_rois_b_selected.shape[0]).cuda())
                        #     if len([n for n in if_in_cylinder_mask if n == True]) != 0:
                        #         kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(occluded_points_one_box_torch[:, 1:][if_in_cylinder_mask].cpu())
                        #         log_density = kde.score_samples(occluded_points_one_box_torch[:, 1:][if_in_cylinder_mask].cpu())
                        #         density_true_recips = 10 ** (log_density)
                        #         density_true = normalization_x(density_true_recips, type='Numpy')
                        #         density = np.mean(density_true) * np.ones_like([np.array(if_in_cylinder_mask)])
                        #         density[0][if_in_cylinder_mask] = density_true
                                
                        #         occluded_points_one_box = torch.cat(occluded_points_one_box)
                        #         fg_points_selected_new.append(occluded_points_one_box)
                        #         occluded_points_list.append(occluded_points_one_box)
                        #         # 计算权重
                        #         proposal2points_score_one_box_torch, distance2points_score_one_box_torch = torch.from_numpy(np.array(proposal2points_score_one_box)), torch.from_numpy(np.array(distance2points_score_one_box))

                        #         distance2points_score_one_box_torch_recip1 = 1.0 / (distance2points_score_one_box_torch + 1e-8)
                        #         distance2points_score_one_box_torch = normalization_x(distance2points_score_one_box_torch_recip1, type='Torch')
                        #         weight_matrix = (distance2points_score_one_box_torch.reshape(-1, 1) * 0.4 + torch.from_numpy(density).float().reshape(-1, 1) * 0.6).cuda()  # torchsize(n, 1)
                        #         weights_fg_points_list.append(weight_matrix.squeeze(1).unsqueeze(0))


                        # # Hough transform
                        # if roi_dz < np.max([roi_dx, roi_dy]):
                        #     whole_object_kp = torch.cat((torch.cat(occluded_points_one_box), fg_of_rois_b_selected))[:, 1:]
                        #     whole_object_kp[:, 0] = whole_object_kp[:, 0] - roi_x_c # 放缩到坐标原点
                        #     whole_object_kp[:, 1] = whole_object_kp[:, 1] - roi_y_c
                        #     whole_object_kp[:, 2] = whole_object_kp[:, 2] - roi_z_c
                        #     whole_object_kp = whole_object_kp.numpy()[:, 0:2].reshape(-1, 1, 2).astype(np.float32) # 不考虑z轴
                        #     pc_lines = cv2.HoughLinesPointSet(whole_object_kp, lines_max=1, threshold=1, min_rho=0.5, max_rho=4, rho_step=0.1, min_theta=0, max_theta=np.pi, theta_step=np.pi/100)
                        #     votes, rho, theta = pc_lines[:, 0][:, 0], pc_lines[:, 0][:, 1], pc_lines[:, 0][:, 2]
                        #     print(pc_lines)
                        #     # Convert to cartesian
                        #     theta[theta == 0.] = 1e-5  # to avoid division by 0 in next line
                        #     az = -1 / np.tan(theta)  # the implied lines are perpendicular to theta
                        #     xz = rho * np.cos(theta)
                        #     yz = rho * np.sin(theta)
                        #     bz = yz - az * xz
                        #     # Plot
                        #     xx = np.linspace(-5, 5)
                        #     print(az, bz)
                        #     for (ia, ib) in zip(az, bz):
                        #         yy = xx * ia + ib
                        #         plt.plot(xx, yy)
                        #         axes = plt.gca()
                        #         axes.set_xlim([-5, 5])
                        #         axes.set_ylim([-5, 5])
                        #         plt.savefig("/opt/data/private/code/MyOpenPCDet/OpenPCDet/output/visual/Hough/" + str(i) + ".jpg")
                        #     plt.clf()  
                
                if fg_points_selected_new != []:
                    fg_points_selected_new_torch = torch.cat(fg_points_selected_new)
                    weights_fg_points_torch = torch.cat((weights_fg_points_list), 1)
                    keypoints_b_new = torch.cat((fg_points_selected_new_torch[:, :4], bg_points_selected[:, :4]))
                    weights_all_points_torch = torch.cat((weights_fg_points_torch, torch.ones(1, bg_points_selected.shape[0]).cuda()), 1)
                    # print(weights_all_points_torch.shape[1], keypoints_b_new.shape[0])
                    assert weights_all_points_torch.shape[1] == keypoints_b_new.shape[0]
                    keypoints_list_new.append(keypoints_b_new)
                    weights_all_points_all_batch.append(weights_all_points_torch)

                    # # 可视化原始点云、关键点，补全后关键点
                    # points = keypoints_b_new.numpy().reshape(-1, 4)[:, 1:4]
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(points)
                    # o3d.io.write_point_cloud("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/points/keypoints_new" + str(i) + ".ply", pcd)
                    # points = keypoints_b.numpy()[:, 1:].reshape(-1, 4)[:, 0:3]
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(points)
                    # o3d.io.write_point_cloud("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/points/keypoints_old" + str(i) + ".ply", pcd)
                    # points = batch_dict['points'][:, 1:].detach().cpu().numpy().reshape(-1, 4)[:, 0:3]
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(points)
                    # o3d.io.write_point_cloud("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/points/pointcloud" + str(i) + ".ply", pcd)
                
                if len(occluded_points_list) != 0:
                    potential_occluded_points_list.append(torch.cat(occluded_points_list))
                else:
                    potential_occluded_points_list.append([0])
                # np.save("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/keypoints/points_after_" + str(epoch_index) + "_" + str(iteration_index) + ".npy", fg_points_selected_new_torch.detach().cpu().numpy())
                # np.save("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/rois/rois_after_" + str(epoch_index) + "_" + str(iteration_index) + ".npy", np.array(rois_b_selected))
                # np.save("/root/autodl-tmp/MyOpenPCDet/OpenPCDet/output/visal/rois/weights_" + str(epoch_index) + "_" + str(iteration_index) + ".npy", weights_fg_points_torch.squeeze(0).numpy())
            if keypoints_list_new != []:
                keypoints = torch.cat(keypoints_list_new).cuda()
                batch_dict['weights_all_points_all_batch'] = weights_all_points_all_batch
                self.completion_dict['points_completion_list'] = points_completion_list
                self.completion_dict['deltas_list'] = deltas_list
                self.completion_dict['dir_list'] = dir_list
                self.completion_dict['roi_label_list'] = roi_label_list

        ###############################################
        # from ....datasets.processor.data_processor import VoxelGeneratorWrapper
        # from ....models.backbones_3d.vfe.mean_vfe import MeanVFE
        # voxel_generator = VoxelGeneratorWrapper(
        #         vsize_xyz=[0.05, 0.05, 0.1],
        #         coors_range_xyz=[  0. , -40. ,  -3. ,  70.4 , 40.  ,  1. ],
        #         num_point_features=4,
        #         max_num_points_per_voxel=5,
        #         max_num_voxels=16000,
        #     )
        # voxels_keypoints, voxels_keypoints_coordinates, voxels_keypoints_num_points = voxel_generator.generate(keypoints.detach().cpu().numpy())
        # keypoints_batch_dict = {}
        # keypoints_batch_dict['voxels'], keypoints_batch_dict['voxel_num_points'] = torch.tensor(voxels_keypoints), torch.tensor(voxels_keypoints_num_points)
        # print(voxels_keypoints.shape, voxels_keypoints_coordinates.shape, voxels_keypoints_num_points.shape)
        # meanvfe = MeanVFE(model_cfg=None, num_point_features=None)
        # voxels_keypoints_features = meanvfe.forward(keypoints_batch_dict)['voxel_features']
        ###############################################

        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE: 
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride'], weights=None
            )
            point_features_list.append(point_bev_features)

        batch_size = batch_dict['batch_size']

        new_xyz = keypoints[:, 1:4].contiguous()
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (keypoints[:, 0] == k).sum()
        
        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            pooled_features = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size, aggregate_func=self.SA_rawpoints,
                xyz=raw_points[:, 1:4],
                xyz_features=raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None,
                xyz_bs_idxs=raw_points[:, 0],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, potential_occluded_points_list=potential_occluded_points_list, weights=None,
                filter_neighbors_with_roi=self.model_cfg.SA_LAYER['raw_points'].get('FILTER_NEIGHBOR_WITH_ROI', False),
                radius_of_neighbor=self.model_cfg.SA_LAYER['raw_points'].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                rois=batch_dict.get('rois', None)
            )
            point_features_list.append(pooled_features)
        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            cur_features = batch_dict['multi_scale_3d_features'][src_name].features.contiguous()

            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4], downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )

            pooled_features = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size, aggregate_func=self.SA_layers[k],
                xyz=xyz.contiguous(), xyz_features=cur_features, xyz_bs_idxs=cur_coords[:, 0],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, potential_occluded_points_list=None, weights=weights_all_points_all_batch,
                filter_neighbors_with_roi=self.model_cfg.SA_LAYER[src_name].get('FILTER_NEIGHBOR_WITH_ROI', False),
                radius_of_neighbor=self.model_cfg.SA_LAYER[src_name].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                rois=batch_dict.get('rois', None)
            )

            point_features_list.append(pooled_features)

        point_features = torch.cat(point_features_list, dim=-1)

        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))
        
        # if batch_dict['weights_all_points_all_batch'] != None:
        #     weights = batch_dict['weights_all_points_all_batch']
        #     features = point_features.permute(1, 0).unsqueeze(0)
        #     features = self.ReshapeWeight1(features)
        #     weights_torch = torch.cat((weights), 1).unsqueeze(0).expand(1, features.shape[1], features.shape[2]).cuda()
        #     features = features * weights_torch
        #     features = self.ReshapeWeight2(features)
        #     points_features = features.squeeze(dim=0).permute(1, 0)

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = keypoints  # (BxN, 4)
        # return batch_dict, points_completion, deltas
        return batch_dict
    
    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        points_completion_list, deltas_list, dir_list, roi_label_list = self.completion_dict['points_completion_list'], self.completion_dict['deltas_list'], self.completion_dict['dir_list'], self.completion_dict['roi_label_list']
        loss_cd_list, loss_pmd_list, loss_align_list = [], [], []
        for i in range(len(deltas_list)):
            pcds, deltas, dir, roi_label = points_completion_list[i], deltas_list[i], dir_list[i], roi_label_list[i]
            device = pcds[0].device
            # target_align_dir =  single_rotate_points_along_z(target_car.to_numpy(), -dir)
            if roi_label == 1:
                gt = target_car
            elif roi_label == 2:
                gt = target_pede
            elif roi_label == 3:
                gt = target_cycl
            # gt = pc_normalize(gt.to_numpy())
            gt = torch.as_tensor(torch.from_numpy(gt.to_numpy()).unsqueeze(0), dtype=torch.float, device=device)
            # print(pcds[0].shape, gt.shape)
            cd1 = chamfer(pcds[0], gt)
            cd2 = chamfer(pcds[1], gt)

            # visualization
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pcds[1].squeeze(0).detach().cpu().numpy())
            # o3d.io.write_point_cloud("/opt/data/private/code/MyOpenPCDet/OpenPCDet/output/visual/completion_keypoints/whole_before" + str(i) + ".ply", pcd)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(target_align_dir)
            # o3d.io.write_point_cloud("/opt/data/private/code/MyOpenPCDet/OpenPCDet/output/visual/completion_keypoints/gt" + str(i) + ".ply", pcd)
            # cd3 = chamfer(pcds[2], gt)

            # loss_cd = cd1 + cd2 + cd3
            loss_cd = cd1 + cd2
            # loss_cd = cd1
            loss_cd_list.append(loss_cd)

            delta_losses = []
            for delta in deltas:
                delta_losses.append(torch.sum(delta ** 2))
            loss_pmd = torch.sum(torch.stack(delta_losses)) / 2
            loss_pmd_list.append(loss_pmd)
            # loss_align_list.append(torch.square(points_feature_align[1][1] - points_feature_align[0][1]).mean() + torch.square(points_feature_align[1][0] - points_feature_align[0][0]).mean())
            # loss_align_list.append(torch.square(points_feature_align[1][0] - points_feature_align[0][0].cuda()).mean())


        loss_cd_sum = sum(loss_cd_list)
        loss_pmd_sum = sum(loss_pmd_list)
        # align_loss = 0.01 * sum(loss_align_list)
        # completion_loss = 0.005 * (loss_cd_sum + loss_pmd_sum * 0.01)
        completion_loss = 0.2 * (loss_cd_sum + loss_pmd_sum * 0.25)
        # completion_loss = 0.05 * (loss_cd_sum + loss_pmd_sum)
        # print(loss_cd_sum, loss_pmd_sum)
        tb_dict['completion_loss'] = completion_loss
        # tb_dict['align_loss'] = align_loss

        # return aux_compl_loss, tb_dict
        return completion_loss, tb_dict
