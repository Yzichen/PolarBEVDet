# Copyright (c) OpenMMLab. All rights reserved.
import pickle
from os import path as osp
import sys
import numpy as np


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=10,
                    only_gt_database=False,
                    save_senor_data=True,
                    skip_cam_instances_infos=False):
    """Prepare waymo dataset. There are 3 steps as follows:

    Step 1. Extract camera images and lidar point clouds from waymo raw
        data in '*.tfreord' and save as kitti format.
    Step 2. Generate waymo train/val/test infos and save as pickle file.
    Step 3. Generate waymo ground truth database (point clouds within
        each 3D bounding box) for data augmentation in training.
    Steps 1 and 2 will be done in Waymo2KITTI, and step 3 will be done in
    GTDatabaseCreater.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default to 10. Here we store ego2global information of these
            frames for later use.
        only_gt_database (bool, optional): Whether to only generate ground
            truth database. Default to False.
        save_senor_data (bool, optional): Whether to skip saving
            image and lidar. Default to False.
        skip_cam_instances_infos (bool, optional): Whether to skip
            gathering cam_instances infos in Step 2. Default to False.
    """
    from tools.data_converter import waymo_converter as waymo

    if version == 'v1.4':
        splits = [
            'training', 'validation', 'testing',
            'testing_3d_camera_only_detection'
        ]
    elif version == 'v1.4-mini':
        splits = ['training', 'validation']
    else:
        raise NotImplementedError(f'Unsupported Waymo version {version}!')
    out_dir = osp.join(out_dir, 'kitti_format')

    if not only_gt_database:
        for i, split in enumerate(splits):
            load_dir = osp.join(root_path, 'waymo_format', split)
            if split == 'validation':
                save_dir = osp.join(out_dir, 'training')
            else:
                save_dir = osp.join(out_dir, split)
            converter = waymo.Waymo2KITTI(
                load_dir,
                save_dir,
                prefix=str(i),
                workers=workers,
                test_mode=(split
                           in ['testing', 'testing_3d_camera_only_detection']),
                info_prefix=info_prefix,
                max_sweeps=max_sweeps,
                split=split,
                save_senor_data=save_senor_data,
                save_cam_instances=not skip_cam_instances_infos)
            converter.convert()
            if split == 'validation':
                converter.merge_trainval_infos()

        from tools.data_converter.waymo_converter import \
            create_ImageSets_img_ids
        create_ImageSets_img_ids(out_dir, splits)

    print('Successfully preparing Waymo Open Dataset')


def get_3d_gt_info(info):
    cam_sync_instances = info['cam_sync_instances']
    gt_boxes = []
    gt_labels = []
    for instance_id in range(len(cam_sync_instances)):
        cur_bbox_3d = cam_sync_instances[instance_id]['bbox_3d']
        label_3d = cam_sync_instances[instance_id]['bbox_label_3d']
        gt_boxes.append(cur_bbox_3d)
        gt_labels.append(label_3d)
        print(cur_bbox_3d)

    return gt_boxes, gt_labels


def get_2d_gt_info(info):
    cam_instances = info['cam_instances']

    gt_3dbboxes_cams = []
    gt_2dbboxes_cams = []
    centers2d_cams = []
    gt_2dlabels_cams = []
    depths_cams = []

    for cam_type, cam_info in cam_instances.items():
        gt_3dbboxes = []
        gt_2dbboxes = []
        centers2d = []
        gt_2dlabels = []
        depths = []
        for instance_id in range(len(cam_info)):
            gt_2dbboxes.append(cam_info[instance_id]['bbox'])
            gt_3dbboxes.append(cam_info[instance_id]['bbox_3d'])
            gt_2dlabels.append(cam_info[instance_id]['bbox_label'])
            centers2d.append(cam_info[instance_id]['center_2d'])
            depths.append(cam_info[instance_id]['depth'])

        gt_3dbboxes_cam = np.array(gt_3dbboxes, dtype=np.float32)  # (N_gt, 7)
        gt_2dbboxes = np.array(gt_2dbboxes, dtype=np.float32)  # (N_gt, 4)  4: (x1, y1, x2, y2)
        gt_2dlabels = np.array(gt_2dlabels, dtype=np.int64)  # (N_gt, )
        centers2d = np.array(centers2d, dtype=np.float32)  # (N_gt, 2)
        depths = np.array(depths, dtype=np.float32)  # (N_gt, )

        gt_3dbboxes_cams.append(gt_3dbboxes_cam)
        gt_2dbboxes_cams.append(gt_2dbboxes)
        gt_2dlabels_cams.append(gt_2dlabels)
        centers2d_cams.append(centers2d)
        depths_cams.append(depths)

    ann_infos = dict(
        bboxes2d=gt_2dbboxes_cams,
        bboxes3d_cams=gt_3dbboxes_cams,
        labels2d=gt_2dlabels_cams,
        centers2d=centers2d_cams,
        depths=depths_cams,
    )

    return ann_infos


def add_ann_adj_info(extra_tag):
    dataroot = 'data/waymo/kitti_format'
    for set in ['train', 'val']:
        dataset = pickle.load(
            open('%s/%s_infos_%s.pkl' % (dataroot, 'waymo', set), 'rb'))
        for id in range(len(dataset['data_list'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['data_list'])))
            info = dataset['data_list'][id]

            info['ann_infos'] = get_3d_gt_info(info)
            info['2D_ann_infos'] = get_2d_gt_info(info)

        with open('%s/%s_infos_%s.pkl' % (dataroot, extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == '__main__':
    version = 'v1.4'
    root_path = 'data/waymo'
    out_dir = 'data/waymo'
    extra_tag = 'waymo'

    waymo_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=version,
        out_dir=out_dir,
        workers=64,
        max_sweeps=0
    )

    extra_tag = 'bevdetv2-waymo'
    add_ann_adj_info(extra_tag)