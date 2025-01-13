# PolarBEVDet

This is the official PyTorch implementation for our paper:
> [**PolarBEVDet: Exploring Polar Representation for Multi-View 3D Object Detection in Bird's-Eye-View**](https://arxiv.org/abs/2408.16200)<br>

![arch](figs/framework.png)

## Model Zoo

| Setting  | Pretrain | NDS  | MAP  | Weights |
|----------|:--------:|:----:|:----:|:-------:|
| [r50_704x256_24e](projects/configs/polarbevdet/r50_704x256_24e.py) | [ImageNet]([ImageNet](https://download.pytorch.org/models/resnet50-0676ba61.pth))  | 53.0 | 43.2 | [gdrive](https://drive.google.com/file/d/1ft34-pxLpHGo2Aw-jowEtCxyXcqszHNn/view) |
| [r50_704x256_60e](projects/configs/polarbevdet/r50_704x256_60e.py) | [ImageNet]([ImageNet](https://download.pytorch.org/models/resnet50-0676ba61.pth))  | 55.3 | 45.0 | [gdrive](https://drive.google.com/file/d/1C_Vn3iiSnSW1Dw1r0DkjJMwvHC5Y3zTN/view) |
| [r50_704x256_nuImg_60e](projects/configs/polarbevdet/r50_704x256_nuImg_60e.py) | [nuImg](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth)  | 56.7 | 46.9 | [gdrive](https://drive.google.com/file/d/1dKu5cR1fuo-O0ynyBh-RCPtHrgut29mN/view) |


## Environment
```
conda create -n polarbevdet python=3.9
conda activate polarbevdet
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install other dependencies:
```
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
mim install mmdet3d==1.0.0rc6
pip install setuptools==59.5.0
pip install numpy==1.23.5
pip install pillow==8.4.0
```

Compile CUDA extensions:
```
cd projects
python setup.py develop
```

## Prepare Dataset
1. Download nuScenes from [https://www.nuscenes.org/nuscenes](https://www.nuscenes.org/nuscenes) and put it in `data/nuscenes`.
2. Generate info files by:
```
python tools/create_data_bevdet.py
```
3. Folder structure:
```
data/nuscenes
├── maps
├── nuscenes_infos_test.pkl
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
├── samples
├── sweeps
├── v1.0-test
└── v1.0-trainval
```

## Training
Train PolarBEVDet with 4 GPUs:
```
bash tools/dist_train.sh projects/configs/polarbevdet/r50_704x256_24e.py 4 --work-dir work_dirs/polarbevdet/r50_704x256_24e
```

## Evaluation
```
python tools/swap_ema_and_non_ema.py work_dirs/polarbevdet/r50_704x256_24e/iter_21096.pth

bash tools/dist_test.sh projects/configs/polarbevdet/r50_704x256_24e.py work_dirs/polarbevdet/r50_704x256_24e/iter_21096_ema.pth 4 --eval map
```

## Acknowledgements

Many thanks to these excellent open-source projects:

* 3D Detection:[BEVDet](https://github.com/HuangJunJie2017/BEVDet), [SOLOFusion](https://github.com/Divadi/SOLOFusion), [StreamPETR](https://github.com/exiawsh/StreamPETR)
* Codebase: [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)

## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{yu2024polarbevdet,
  title={PolarBEVDet: Exploring Polar Representation for Multi-View 3D Object Detection in Bird's-Eye-View},
  author={Yu, Zichen and Liu, Quanli and Wang, Wei and Zhang, Liyong and Zhao, Xiaoguang},
  journal={arXiv preprint arXiv:2408.16200},
  year={2024}
}
```