# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
import copy
import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet.datasets import DATASETS
from mmdet3d.datasets import KittiDataset
import math
from mmcv import print_log

try:
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
    from waymo_open_dataset.protos.metrics_pb2 import Objects
except ImportError:
    Objects = None
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')
import torch
from mmdet.core.bbox.iou_calculators import bbox_overlaps

import tensorflow as tf
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.protos import breakdown_pb2, metrics_pb2
from waymo_open_dataset.metrics.python import config_util_py as config_util


@DATASETS.register_module()
class WaymoDatasetBEVDet(KittiDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list(float), optional): The range of point cloud used
            to filter invalid predicted boxes.
            Default: [-85, -85, -5, 85, 85, 5].
    """

    CLASSES = ('Car', 'Cyclist', 'Pedestrian')
    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 data_prefix=dict(
                     pts='velodyne',
                     CAM_FRONT='image_0',
                     CAM_FRONT_LEFT='image_1',
                     CAM_FRONT_RIGHT='image_2',
                     CAM_SIDE_LEFT='image_3',
                     CAM_SIDE_RIGHT='image_4'),
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 img_info_prototype='mmcv',
                 multi_adj_frame_id_cfg=None,
                 seq_mode=False,
                 sequences_split_num=1,
                 waymo_bin_file='data/waymo/kitti_format/laser_gt_objects.bin',
                 refine_with_2dbbox=False,
                 **kwargs):
        self.load_interval = load_interval
        self.data_prefix = data_prefix
        self.classes = classes
        self.waymo_bin_file = waymo_bin_file
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range,
            **kwargs)

        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.seq_mode = seq_mode
        if seq_mode:
            self.seq_split_num = sequences_split_num
            self._set_sequence_group_flag()  # Must be called after load_annotations b/c load_annotations does sorting.

        # to load a subset, just set the load_interval in the dataset config
        self.data_infos = self.data_infos[::load_interval]
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]
        self.refine_with_2dbbox = refine_with_2dbbox

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        for idx in range(len(self.data_infos)):
            sample_idx = int(self.data_infos[idx]['sample_idx'])
            curr_sequence = sample_idx // 1000
            res.append(curr_sequence)

        # 找到每个sample_data所属的scene/sequence Id.
        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                # 将每个seq进一步划分为seq_split_num份，这样共得到N_seq * seq_split_num 个seq.
                # 因此，在stream training时， 每个seq的长度queue_length= num_samples_per_scenes / seq_split_num.
                bin_counts = np.bincount(self.flag)     # (N_seq, ): Ls0, Ls1, ...  记录每个seq中包含的samples数量.
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, bin_counts[curr_flag], math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])     # [0, Ls/2, Ls]

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format
        return mmcv.load(ann_file, file_format='pkl')['data_list']

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = copy.deepcopy(self.data_infos[index])
        sample_idx = info['sample_idx']
        lidar_path = osp.join(self.root_split, self.data_prefix['pts'], info['lidar_points']['lidar_path'])
        for cam_name, cam_info in info['images'].items():
            cam_info['img_path'] = osp.join(self.root_split, self.data_prefix[cam_name], cam_info['img_path'])

        input_dict = dict(
            sample_idx=sample_idx,
            scene_token=sample_idx//1000,
            pts_filename=lidar_path,
            timestamp=info['timestamp'] / 1e6,
        )
        if not self.test_mode and self.seq_mode:  # for seq_mode
            prev_exists = not (index == 0 or self.flag[index - 1] != self.flag[index])
        else:
            prev_exists = None
        input_dict['prev_exists'] = prev_exists
        input_dict['sequence_group_idx'] = self.flag[index]
        input_dict['start_of_sequence'] = index == 0 or self.flag[index - 1] != self.flag[index]

        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']

            input_dict['ann_infos']
        if '2D_ann_infos' in info:
            input_dict['2D_ann_infos'] = info['2D_ann_infos']

        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    lidar2img = np.array(cam_info['lidar2img'], dtype=np.float32)
                    lidar2img_rts.append(lidar2img)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos
            else:
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:     # 需要再读取历史帧的信息
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))

        return input_dict

    def get_adj_info(self, info, index):
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))     # bevdet4d: [1, ]  只利用前一帧.
        if self.stereo:
            assert self.multi_adj_frame_id_cfg[0] == 1
            assert self.multi_adj_frame_id_cfg[2] == 1
            # 如果使用stereo4d, 不仅当前帧需要利用前一帧图像计算stereo depth, 前一帧也需要利用它的前一帧计算stereo depth.
            # 因此, 我们需要额外读取一帧(也就是前一帧的前一帧).
            adj_id_list.append(self.multi_adj_frame_id_cfg[1])
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            if not self.data_infos[select_id]['sample_idx'] // 1000 == info[
                    'sample_idx'] // 1000:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def convert_one(self, res_idx: int):
        """Convert action for single file. It read the metainfo from the
        preprocessed file offline and will be faster.

        Args:
            res_idx (int): The indices of the results.
        """
        sample_idx = self.results[res_idx]['sample_idx']
        if len(self.results[res_idx]['labels_3d']) > 0:
            objects = self.parse_objects_from_origin(
                self.results[res_idx], self.results[res_idx]['context_name'],
                self.results[res_idx]['timestamp'])
        else:
            print(sample_idx, 'not found.')
            objects = metrics_pb2.Objects()

        return objects


    def format_results(self, results, result_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        if result_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results_dir')
        else:
            tmp_dir = None

        file_idx_list = []
        results_maps = {}

        for idx in range(len(results)):
            output = results[idx]['pts_bbox']
            box_preds, scores_3d, labels_3d = output['boxes_3d'], output['scores_3d'], output['labels_3d']

            data_info = self.get_data_info(idx)
            try:
                gt_bboxes, gt_labels = data_info['ann_infos']
            except:
                gt_bboxes = None

            box_preds.limit_yaw(offset=0.5, period=np.pi * 2)
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_inds = ((box_preds.center > limit_range[:3]) & (box_preds.center < limit_range[3:])).all(-1)
            sample_idx = data_info['sample_idx']
            timestamp = data_info['timestamp'] * 1e6

            cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_SIDE_LEFT', 'CAM_SIDE_RIGHT']
            img_shapes = [(1280, 1920, 3), (1280, 1920, 3), (1280, 1920, 3), (886, 1920, 3), (886, 1920, 3)]

            cam_valid_list = []
            n_pred = box_preds.tensor.shape[0]

            for cam_id, cam_name in enumerate(cam_names):
                if n_pred == 0:
                    break
                corners_3d = box_preds.corners
                num_bbox = corners_3d.shape[0]
                pts_4d = torch.cat([corners_3d.view(-1, 3), torch.ones((num_bbox * 8, 1))], dim=-1)

                cam_data = data_info['curr']['images'][cam_name]
                lidar2img = torch.Tensor(cam_data['lidar2img'])

                pts_2d = pts_4d @ lidar2img.T
                # every corner of obj should be in front of the camera
                valid_cam_inds = (pts_2d[:, 2].view(num_bbox, 8) > 0).all(-1)

                pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5, max=1e5)
                pts_2d[:, 0] /= pts_2d[:, 2]
                pts_2d[:, 1] /= pts_2d[:, 2]
                imgfov_pts_2d = pts_2d[..., :2].view(num_bbox, 8, 2)

                image_shape = box_preds.tensor.new_tensor(img_shapes[cam_id])

                minxy = imgfov_pts_2d.min(dim=1)[0]
                maxxy = imgfov_pts_2d.max(dim=1)[0]
                pred_2d_bboxes_camera = torch.cat([minxy, maxxy], dim=1)

                valid_cam_inds &= ((pred_2d_bboxes_camera[:, 0] < image_shape[1]) &
                                   (pred_2d_bboxes_camera[:, 1] < image_shape[0]) & (pred_2d_bboxes_camera[:, 2] > 0) &
                                   (pred_2d_bboxes_camera[:, 3] > 0))

                # calculate gt_2d_bboxes_camera
                if self.refine_with_2dbbox and gt_bboxes is not None:
                    bbox_in_cam = pred_2d_bboxes_camera[valid_cam_inds]
                    bbox_in_cam[:, [0, 2]] = torch.clamp(bbox_in_cam[:, [0, 2]], min=0, max=image_shape[1])
                    bbox_in_cam[:, [1, 3]] = torch.clamp(bbox_in_cam[:, [1, 3]], min=0, max=image_shape[0])

                    gt_bboxes_2d = data_info['2D_ann_infos']['bboxes2d'][cam_id]
                    gt_labels_2d = data_info['2D_ann_infos']['labels2d'][cam_id]

                    gt_bboxes_2d = torch.from_numpy(gt_bboxes_2d).to(torch.float32)
                    gt_labels_2d = torch.from_numpy(gt_labels_2d).to(torch.int64)

                    cls2id = {
                        'Car': 0,
                        'Pedestrian': 1,
                        'Cyclist': 2,
                    }

                    gt_labels_ = torch.tensor([cls2id[each] for each in gt_labels_2d])
                    len_gt = len(gt_labels_)

                    pairwise_iou = bbox_overlaps(bbox_in_cam, gt_bboxes_2d, device=bbox_in_cam.device)

                    _pred_labels = torch.as_tensor(labels_3d)[valid_cam_inds].unsqueeze(1).repeat(1, len_gt)
                    valid_cam_inds[valid_cam_inds.clone()] = ((pairwise_iou > 0.2) & (_pred_labels == gt_labels_)).any(
                        -1)

                    #valid_cam_inds[valid_cam_inds.clone()] = (pairwise_iou > 0.2).any(
                    #    -1)  # only bbox iou here, add labels_3d.numpy()[valid_inds] and gt_labels_3d should be better

                cam_valid_list.append(valid_cam_inds)
            if n_pred > 0:
                valid_cam_inds = cam_valid_list[0] | cam_valid_list[1] | cam_valid_list[2] | cam_valid_list[3] | \
                             cam_valid_list[4]
                valid_inds = valid_cam_inds & valid_inds
            if valid_inds.sum() > 0:
                result = dict(
                    bbox=None,
                    box3d_camera=None,
                    box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                    scores=scores_3d[valid_inds].numpy(),
                    label_preds=labels_3d[valid_inds].numpy(),
                    sample_idx=sample_idx,
                    timestamp=timestamp
                )
            else:
                result = dict(
                    bbox=np.zeros([0, 4]),
                    box3d_camera=np.zeros([0, 7]),
                    box3d_lidar=np.zeros([0, 7]),
                    scores=np.zeros([0]),
                    label_preds=np.zeros([0, 4]),
                    sample_idx=sample_idx,
                    timestamp=timestamp
                )

            if self.split == 'training' or self.split == 'testing_camera':
                # results_list.append(result)
                idx = (sample_idx // 1000) % 1000
                frame_idx = sample_idx % 1000
                file_idx_list.append(idx)
                if idx not in results_maps.keys():
                    results_maps[idx] = {frame_idx: result}
                else:
                    results_maps[idx][frame_idx] = result

        file_idx_list = list(set(file_idx_list))
        pd_bbox, pd_type, pd_frame_id, pd_score = [], [], [], []

        k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }
        k2w_cls_map = np.array([k2w_cls_map[each] for each in self.CLASSES])

        for file_idx in file_idx_list:
            result = results_maps[file_idx]
            for frame_num in result.keys():
                frame_result = result[frame_num]
                sample_idx = frame_result['sample_idx']
                timestamp = frame_result['timestamp']

                n_pred = len(frame_result['label_preds'])
                if n_pred == 0:
                    continue
                length = frame_result['box3d_lidar'][:, 3]
                width = frame_result['box3d_lidar'][:, 4]
                height = frame_result['box3d_lidar'][:, 5]
                x = frame_result['box3d_lidar'][:, 0]
                y = frame_result['box3d_lidar'][:, 1]
                z = frame_result['box3d_lidar'][:, 2]
                z += height / 2

                heading = frame_result['box3d_lidar'][:, 6]
                box = np.stack([x, y, z, length, width, height, heading], axis=-1)
                box = np.round(box, 4)

                cls = k2w_cls_map[frame_result['label_preds']]
                frame_id = np.full(n_pred, timestamp)
                score = np.round(frame_result['scores'], 4)

                pd_bbox.append(box)
                pd_type.append(cls)
                pd_frame_id.append(frame_id)
                pd_score.append(score)

        pd_bbox = tf.concat(pd_bbox, axis=0)
        pd_type = tf.concat(pd_type, axis=0)
        pd_frame_id = tf.concat(pd_frame_id, axis=0)
        pd_score = tf.concat(pd_score, axis=0)
        return (pd_bbox, pd_type, pd_frame_id, pd_score)

    def get_boxes_from_gtbin(self, remove_gt=True, valid_pd_frame_id=None, score_th=None, logger=None):

        if osp.exists(self.waymo_bin_file + '.pkl'):
            print_log(f'Directly load from {self.waymo_bin_file}.pkl', logger=logger)
            return mmcv.load(self.waymo_bin_file + '.pkl')

        pd_bbox, pd_type, pd_frame_id, difficulty = [], [], [], []

        stuff1 = metrics_pb2.Objects()
        with open(self.waymo_bin_file, 'rb') as rf:
            stuff1.ParseFromString(rf.read())

        print_log(f'Loading {len(stuff1.objects)} objects from gt_bin_file...', logger=logger)
        for i in range(len(stuff1.objects)):
            obj = stuff1.objects[i].object
            if obj.type == 3:  # label_pb2.Label.TYPE_SIGN
                continue
            if score_th is not None:
                if stuff1.objects[i].score <= score_th:
                    continue
            if remove_gt and obj.most_visible_camera_name == '':
                continue

            # Ignore objects that are fully-occluded to cameras.
            if obj.num_lidar_points_in_box == 0:
                continue

            if valid_pd_frame_id is not None and stuff1.objects[i].frame_timestamp_micros not in valid_pd_frame_id:
                continue
            pd_frame_id.append(stuff1.objects[i].frame_timestamp_micros)
            box = tf.constant([obj.camera_synced_box.center_x, obj.camera_synced_box.center_y, obj.camera_synced_box.center_z,
                   obj.camera_synced_box.length, obj.camera_synced_box.width, obj.camera_synced_box.height,
                   obj.camera_synced_box.heading], dtype=tf.float32)
            pd_bbox.append(box)
            pd_type.append(obj.type)

            if obj.num_lidar_points_in_box <= 5 or obj.detection_difficulty_level == 2:
                obj.detection_difficulty_level = label_pb2.Label.LEVEL_2
            else:
                obj.detection_difficulty_level = label_pb2.Label.LEVEL_1

            # Fill in unknown difficulties.
            # if obj.detection_difficulty_level == label_pb2.Label.UNKNOWN:
            #     obj.detection_difficulty_level = label_pb2.Label.LEVEL_2

            difficulty.append(np.uint8(obj.detection_difficulty_level))

        pd_bbox = tf.stack(pd_bbox)
        pd_type = tf.constant(pd_type, dtype=tf.uint8)
        pd_frame_id = tf.constant(pd_frame_id, dtype=tf.int64)
        difficulty = tf.constant(difficulty, dtype=tf.uint8)
        return pd_bbox, pd_type, pd_frame_id, difficulty

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.
        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        Returns:
            dict[str: float]: results of each evaluation metric
        """
        pd_bbox, pd_type, pd_frame_id, pd_score = self.format_results(results, jsonfile_prefix)
        self.waymo_results_final_path = osp.join(jsonfile_prefix, 'pred.pkl')
        print_log(f'pkl save to {self.waymo_results_final_path}', logger=logger)
        outputs = [pd_bbox.numpy(), pd_type.numpy(), pd_frame_id.numpy(), pd_score.numpy()]
        mmcv.dump(outputs, self.waymo_results_final_path)

        if self.split == 'testing_camera': return {}

        print_log(f'Starting get_boxes_from_gtbin...{self.waymo_bin_file}', logger=logger)
        gt_bbox, gt_type, gt_frame_id, difficulty = self.get_boxes_from_gtbin(logger=logger, valid_pd_frame_id=set(
            pd_frame_id.numpy()))

        decoded_outputs = [[gt_bbox, gt_type, gt_frame_id, difficulty], [pd_bbox, pd_type, pd_frame_id, pd_score]]
        print_log('Starting compute_ap...', logger=logger)
        metrics = self.compute_ap(decoded_outputs)
        print_log('End.', logger=logger)
        results = {}
        for key in metrics:
            if 'TYPE_SIGN' in key:
                continue
            metric_status = ('%s: %s') % (key, metrics[key].numpy())
            results[key] = metrics[key].numpy()
            print(metric_status)
        LET_AP = 0
        LET_APH = 0
        LET_APL = 0
        for cls in ['CYCLIST', 'PEDESTRIAN', 'VEHICLE']:
            LET_AP += metrics[f'3d_ap_OBJECT_TYPE_TYPE_{cls}_LEVEL_2'].numpy()
            LET_APH += metrics[f'3d_ap_ha_weighted_OBJECT_TYPE_TYPE_{cls}_LEVEL_2'].numpy()
            LET_APL += metrics[f'3d_ap_la_weighted_OBJECT_TYPE_TYPE_{cls}_LEVEL_2'].numpy()
        print('LET-AP', LET_AP / 3)
        print('LET-APH', LET_APH / 3)
        print('LET-APL', LET_APL / 3)

        return results

    def compute_ap(self, decoded_outputs):
        """Compute average precision."""
        [[gt_bbox, gt_type, gt_frame_id, difficulty], [pd_bbox, pd_type, pd_frame_id, pd_score]] = decoded_outputs

        scalar_metrics_3d, _ = self.build_waymo_metric(
            pd_bbox, pd_type, pd_score, pd_frame_id,
            gt_bbox, gt_type, gt_frame_id, difficulty)

        return scalar_metrics_3d

    def build_waymo_metric(self, pred_bbox, pred_class_id, pred_class_score,
                           pred_frame_id, gt_bbox, gt_class_id, gt_frame_id, difficulty,
                           gt_speed=None, box_type='3d', breakdowns=None):
        """Build waymo evaluation metric."""
        # metadata = waymo_metadata.WaymoMetadata()
        metadata = None
        if breakdowns is None:
            # breakdowns = ['RANGE', 'SIZE', 'OBJECT_TYPE']
            breakdowns = ['RANGE', 'OBJECT_TYPE']
        waymo_metric_config = self._build_waymo_metric_config(
            metadata, box_type, breakdowns)

        def detection_metrics(prediction_bbox,
                              prediction_type,
                              prediction_score,
                              prediction_frame_id,
                              prediction_overlap_nlz,
                              ground_truth_bbox,
                              ground_truth_type,
                              ground_truth_frame_id,
                              ground_truth_difficulty,
                              config,
                              ground_truth_speed=None):
            if ground_truth_speed is None:
                num_gt_boxes = tf.shape(ground_truth_bbox)[0]
                ground_truth_speed = tf.zeros((num_gt_boxes, 2), dtype=tf.float32)

            # metrics_module = tf.load_op_library(
            #     tf.compat.v1.resource_loader.get_path_to_datafile('metrics_ops.so'))

            from waymo_open_dataset.metrics.ops import py_metrics_ops
            return py_metrics_ops.detection_metrics(
                prediction_bbox=prediction_bbox,
                prediction_type=prediction_type,
                prediction_score=prediction_score,
                prediction_frame_id=prediction_frame_id,
                prediction_overlap_nlz=prediction_overlap_nlz,
                ground_truth_bbox=ground_truth_bbox,
                ground_truth_type=ground_truth_type,
                ground_truth_frame_id=ground_truth_frame_id,
                ground_truth_difficulty=ground_truth_difficulty,
                ground_truth_speed=ground_truth_speed,
                config=config)

        ap, ap_ha, ap_la, pr, pr_ha, pr_la, tmp = detection_metrics(
            prediction_bbox=tf.cast(pred_bbox, tf.float32),
            prediction_type=tf.cast(pred_class_id, tf.uint8),
            prediction_score=tf.cast(pred_class_score, tf.float32),
            prediction_frame_id=tf.cast(pred_frame_id, tf.int64),
            prediction_overlap_nlz=tf.zeros_like(pred_frame_id, dtype=tf.bool),
            ground_truth_bbox=tf.cast(gt_bbox, tf.float32),
            ground_truth_type=tf.cast(gt_class_id, tf.uint8),
            ground_truth_frame_id=tf.cast(gt_frame_id, tf.int64),
            ground_truth_difficulty=tf.cast(difficulty, dtype=tf.uint8),
            ground_truth_speed=None,
            config=waymo_metric_config.SerializeToString())

        # All tensors returned by Waymo's metric op have a leading dimension
        # B=number of breakdowns. At this moment we always use B=1 to make
        # it compatible to the python code.

        scalar_metrics = {'%s_ap' % box_type: ap[0],
                          '%s_ap_ha_weighted' % box_type: ap_ha[0],
                          '%s_ap_la_weighted' % box_type: ap_la[0]}
        curve_metrics = {'%s_pr' % box_type: pr[0],
                         '%s_pr_ha_weighted' % box_type: pr_ha[0], }
        # '%s_pr_la_weighted' % box_type: pr_la[0]}

        breakdown_names = config_util.get_breakdown_names_from_config(
            waymo_metric_config)
        for i, metric in enumerate(breakdown_names):
            # There is a scalar / curve for every breakdown.
            scalar_metrics['%s_ap_%s' % (box_type, metric)] = ap[i]
            scalar_metrics['%s_ap_ha_weighted_%s' % (box_type, metric)] = ap_ha[i]
            scalar_metrics['%s_ap_la_weighted_%s' % (box_type, metric)] = ap_la[i]
            curve_metrics['%s_pr_%s' % (box_type, metric)] = pr[i]
            curve_metrics['%s_pr_ha_weighted_%s' % (box_type, metric)] = pr_ha[i]
            # curve_metrics['%s_pr_la_weighted_%s' % (box_type, metric)] = pr_la[i]
        return scalar_metrics, curve_metrics

    def _build_waymo_metric_config(self, metadata, box_type, waymo_breakdown_metrics):
        """Build the Config proto for Waymo's metric op."""
        config = metrics_pb2.Config()
        # num_pr_points = metadata.NumberOfPrecisionRecallPoints()
        num_pr_points = 101
        config.score_cutoffs.extend(
            [i * 1.0 / (num_pr_points - 1) for i in range(num_pr_points)])
        config.matcher_type = metrics_pb2.MatcherProto.Type.TYPE_HUNGARIAN
        if box_type == '2d':
            config.box_type = label_pb2.Label.Box.Type.TYPE_2D
        else:
            config.box_type = label_pb2.Label.Box.Type.TYPE_3D
        # Default values
        config.iou_thresholds[:] = [0.0, 0.5, 0.3, 0.3, 0.3]
        diff = metrics_pb2.Difficulty()
        # diff.levels.append(1)
        diff.levels.append(2)

        config.let_metric_config.enabled = True
        config.let_metric_config.longitudinal_tolerance_percentage = 0.1
        config.let_metric_config.min_longitudinal_tolerance_meter = 0.5
        config.let_metric_config.sensor_location.x = 1.43
        config.let_metric_config.sensor_location.y = 0
        config.let_metric_config.sensor_location.z = 2.18

        config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.ONE_SHARD)
        config.difficulties.append(diff)
        # Add extra breakdown metrics.
        for breakdown_value in waymo_breakdown_metrics:
            breakdown_id = breakdown_pb2.Breakdown.GeneratorId.Value(breakdown_value)
            config.breakdown_generator_ids.append(breakdown_id)
            config.difficulties.append(diff)
        return config
