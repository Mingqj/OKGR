# OKGR: Occluded Keypoint Generation and Refinement for 3D Object Detection (PRCV 2023)
This is the official of [OKGR](https://github.com/Mingqj/OKGR/). (OKGR: Occluded Keypoint Generation and Refinement for 3D Object Detection). This code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), some codes are from [PMPNet](https://github.com/diviswen/PMP-Net).

Supplemental material and code for 'OKGR: Occluded Keypoint Generation and Refinement for 3D Object Detection'.
The supplemental material can be downloaded from [baiduwp](https://pan.baidu.com/s/18wt2LT4dgXg8pa0zYded-w) [mk22].

## Install
NOTE: Please re-install pcdet v0.5 by running `python setup.py develop` even if you have already installed previous version.

a. Install the dependent libraries as follows: 

Install the SparseConv library, we use the implementation from [Spconv](https://github.com/traveller59/spconv).

Install the dependent python libraries: 

```shell
pip install -r requirements.txt
```

b. Install this pcdet library and its dependent libraries by running the following command:
```shell
python setup.py develop
```

c. Install the PMPNet library and build pyTorch extensions:
```shell
cd ./pcdet/models/backbones_3d/Chamfer3D/
python setup.py install
```

## Data Preparation
Please follow the instructions in [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Main Results on KITTI
|                                             | Car@R40 | Pedestrian@R40 | Cyclist@R40  | download | 
|---------------------------------------------|:-------:|:-------:|:-------:|:---------:|
| PV-RCNN++ | 82.36 | 71.42 | 67.71 | - | 
| PV-RCNN++ with OKGR| 82.36 | 71.42 | 67.71 | - | 

## Main Results on Waymo Open Dataset
|                                             | Difficulty | Vehicle | Pedestrian | Cyclist | download | 
|---------------------------------------------|:-------:|:-------:|:-------:|:---------:|
| PV-RCNN++ | LEVEL-1 | 77.82 | 77.99 | 71.80 | - | 
| PV-RCNN++ with OKGR | LEVEL-1 | 78.14 | 79.66 | 74.13 | - | 
| PV-RCNN++ | LEVEL-1 | 69.07 | 69.92 | 69.31 | - | 
| PV-RCNN++ with OKGR | LEVEL-1 | 69.46 | 71.70 | 71.57 | - | 

## Acknowledgement
We thank the authors for the multiple great open-sourced repos, including [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [PMPNet](https://github.com/diviswen/PMP-Net).
