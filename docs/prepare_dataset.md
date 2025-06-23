# Prepare Dataset

## Nuscenes
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
4. For Occupancy Prediction task, download (only) the 'gts' from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and arrange the folder as:
```
data/nuscenes
├── maps
├── nuscenes_infos_test.pkl
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
├── samples
├── sweeps
├── gts (new)
├── v1.0-test
└── v1.0-trainval
```

## Waymo
1. Download Waymo open dataset V1.4.1 [HERE](https://waymo.com/open/download/) and its data split [HERE](https://drive.google.com/drive/folders/18BVuF_RYJF0NjZpt8SnfzANiakoRMf0o?usp=sharing). Then put `.tfrecord` files into corresponding folders in `data/waymo/waymo_format/` and put the data split `.txt` files 
into `data/waymo/kitti_format/ImageSets`. Download ground truth `laser_gt_objects.bin` file for validation set [HERE](https://storage.googleapis.com/waymo_open_dataset_v_1_3_1/validation/laser_gt_objects.bin) and put it into `data/waymo/kitti_format/`. A tip is that you can use `gsutil` to download the large-scale dataset with commands. 
You can take this [tool](https://github.com/RalphMao/Waymo-Dataset-Tool) as an example for more details. 
2. Subsequently, prepare waymo data by running:
```
python tools/create_data_bevdet_waymo.py
```
3. Folder structure:
```
data/waymo
├── waymo_format
    ├──training
    ├──validation
    ├──testing
├── kitti_format
    ├──ImageSets
    ├──training
    ├──testing
    ├──laser_gt_objects.bin
    ├──bevdetv2-waymo_infos_train.pkl
    ├──bevdetv2-waymo_infos_val.pkl
```