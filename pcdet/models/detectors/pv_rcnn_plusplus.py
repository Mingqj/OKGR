from .detector3d_template import Detector3DTemplate


i, e = 0, 0
class PVRCNNPlusPlus(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.start_training_epoch = 30

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        batch_dict = self.roi_head.proposal_layer(
            batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'])
        if self.training:
            global i, e
            i += 1
            if i == 928:
            # if i == 1871:
            # if i == 464:
            # if i == 936:
                i = 0
                e += 1
            targets_dict = self.roi_head.assign_targets(batch_dict)
            batch_dict['epoch_index'], batch_dict['iteration_index'], batch_dict['start_training_epoch'] = e, i, self.start_training_epoch
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_targets_dict'] = targets_dict
            num_rois_per_scene = targets_dict['rois'].shape[1]
            if 'roi_valid_num' in batch_dict:
                batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]
        else:
            batch_dict['epoch_index'], batch_dict['iteration_index'], batch_dict['start_training_epoch'] = -1, -1, -1

        batch_dict = self.pfe(batch_dict)
        #########################################################################
        # batch_dict = self.pfe(batch_dict, 'Step1')
        # if batch_dict['epoch_index'] > batch_dict['start_training_epoch'] or batch_dict['epoch_index'] == -1:
        #     batch_dict = self.vfe(batch_dict)
        #     batch_dict = self.backbone_3d(batch_dict)
        #     batch_dict = self.map_to_bev_module(batch_dict)
        #     batch_dict = self.backbone_2d(batch_dict)
        #     batch_dict = self.dense_head(batch_dict)
        #     batch_dict = self.roi_head.proposal_layer(
        #     batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'])
        # # if e > batch_dict['start_training_epoch'] or e == -1:
        # #     batch_dict = self.voxel_feature_alignment_module(batch_dict, i)
        # batch_dict = self.pfe(batch_dict, 'Step2')
        #########################################################################
        batch_dict = self.point_head(batch_dict)
        batch_dict = self.roi_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        global i, e
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        # if e > self.start_training_epoch:
        #     loss_bev, tb_dict = self.voxel_feature_alignment_module.get_loss(tb_dict)
        if e > self.start_training_epoch:
            loss_completion, tb_dict = self.pfe.get_loss(tb_dict)
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        if e > self.start_training_epoch:
            loss = loss_rpn + loss_point + loss_rcnn + loss_completion
            # print(loss_rpn, loss_point, loss_rcnn, loss_completion)
            # print(loss_completion)
        else:
            loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
