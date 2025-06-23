# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.builder import PIPELINES
from torchvision.transforms.functional import rotate


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    # vov99
    # mean = np.array([103.530, 116.280, 123.675], dtype=np.float32)
    # std = np.array([57.375, 57.120, 58.395], dtype=np.float32)
    # to_rgb = False
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential

    def choose_cams(self):
        """
        Returns:
            cam_names: List[CAM_Name0, CAM_Name1, ...]
        """
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        """
        Args:
            H:
            W:
            flip:
            scale:
        Returns:
            resize: resize比例float.
            resize_dims: (resize_W, resize_H)
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: 0 / 1
            rotate: 随机旋转角度float
        """
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])    # resize的比例, 位于[fW/W − 0.06, fW/W + 0.11]之间.
            resize_dims = (int(W * resize), int(H * resize))            # resize后的size
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH     # s * H - H_in
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))       # max(0, s * W - fW)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        """
        Args:
            img: PIL.Image
            post_rot: torch.eye(2)
            post_tran: torch.eye(2)
            resize: float, resize的比例.
            resize_dims: Tuple(W, H), resize后的图像尺寸
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: bool
            rotate: float 旋转角度
        Returns:
            img: PIL.Image
            post_rot: Tensor (2, 2)
            post_tran: Tensor (2, )
        """
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        # 将上述变换以矩阵表示.
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def get_sensor_transforms(self, info, cam_name):
        """
        Args:
            info:
            cam_name: 当前要读取的CAM.
        Returns:
            sensor2ego: (4, 4)
            ego2global: (4, 4)
        """
        w, x, y, z = info['cams'][cam_name]['sensor2ego_rotation']      # 四元数格式
        # sensor to ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        sensor2ego_tran = torch.Tensor(
            info['cams'][cam_name]['sensor2ego_translation'])   # (3, )
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran

        # ego to global
        w, x, y, z = info['cams'][cam_name]['ego2global_rotation']      # 四元数格式
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        ego2global_tran = torch.Tensor(
            info['cams'][cam_name]['ego2global_translation'])   # (3, )
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        """
        Args:
            results:
            flip:
            scale:

        Returns:
            imgs:  (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
            sensor2egos: (N_views, 4, 4)
            ego2globals: (N_views, 4, 4)
            intrins:     (N_views, 3, 3)
            post_rots:   (N_views, 3, 3)
            post_trans:  (N_views, 3)
        """
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []

        for cam_id, cam_name in enumerate(cam_names):
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            # 初始化图像增广的旋转和平移矩阵
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # 当前相机内参
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            # 获取当前相机的sensor2ego(4x4), ego2global(4x4)矩阵.
            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)

            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs

            # img: PIL.Image;  post_rot: Tensor (2, 2);  post_tran: Tensor (2, )
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            # 以3x3矩阵表示图像的增广
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))    # 保存未归一化的图像，应该是为了做可视化.
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    # 对选择的邻近帧图像也进行增广, 增广参数与当前帧图像相同.
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))

            intrins.append(intrin)      # 相机内参 (3, 3)
            sensor2egos.append(sensor2ego)      # camera2ego变换 (4, 4)
            ego2globals.append(ego2global)      # ego2global变换 (4, 4)
            post_rots.append(post_rot)          # 图像增广旋转 (3, 3)
            post_trans.append(post_tran)        # 图像增广平移 (3, ）

        if self.sequential:
            for adj_info in results['adjacent']:
                # adjacent与current使用相同的图像增广, 相机内参也相同.
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                for cam_name in cam_names:
                    # 获得adjacent帧对应的camera2ego变换 (4, 4)和ego2global变换 (4, 4).
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)    # (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)

        sensor2egos = torch.stack(sensor2egos)      # (N_views, 4, 4)
        ego2globals = torch.stack(ego2globals)      # (N_views, 4, 4)
        intrins = torch.stack(intrins)              # (N_views, 3, 3)
        post_rots = torch.stack(post_rots)          # (N_views, 3, 3)
        post_trans = torch.stack(post_trans)        # (N_views, 3)
        results['canvas'] = canvas      # List[(H, W, 3), (H, W, 3), ...]     len = 6

        N, _, H, W = imgs.shape
        results['pad_shape'] = [(H, W, 3)] * N
        results['img_shape'] = [(H, W, 3)] * N

        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):
    def __init__(self, bda_aug_conf, classes, is_train=True, align_camera_center=False):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes
        self.align_camera_center = align_camera_center

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        """
        Args:
            gt_boxes: (N, 9)
            rotate_angle:
            scale_ratio:
            flip_dx: bool
            flip_dy: bool

        Returns:
            gt_boxes: (N, 9)
            rot_mat: (3, 3）
        """
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:     # 沿着y轴翻转
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:     # 沿着x轴翻转
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)    # 变换矩阵(3, 3)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)     # 变换后的3D框中心坐标
            gt_boxes[:, 3:6] *= scale_ratio    # 变换后的3D框尺寸
            gt_boxes[:, 6] += rotate_angle     # 旋转后的3D框的方位角
            # 翻转也会进一步改变方位角
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']      # (N_gt, 9),  (N_gt, )
        gt_boxes, gt_labels = torch.from_numpy(np.array(gt_boxes, dtype=np.float32)), \
                              torch.from_numpy(np.array(gt_labels, dtype=np.int))
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()

        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        if self.align_camera_center:
            camera_center = sensor2egos[:6, ...].mean(dim=0)[:2, 3]
            center_mat = torch.eye(4)
            center_mat[:2, 3] = -camera_center
            if gt_boxes.shape[0] > 0:
                gt_boxes[:, :2] -= camera_center

            sensor2egos = center_mat.unsqueeze(dim=0) @ sensor2egos
            ego2globals = ego2globals @ torch.inverse(center_mat).unsqueeze(dim=0)

            results['camera_center'] = camera_center

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        # gt_boxes: (N, 9)  BEV增广变换后的3D框
        # bda_rot: (3, 3)   BEV增广矩阵, 包括旋转、缩放和翻转.
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot

        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels

        results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins, post_rots,
                                 post_trans, bda_rot)

        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda

        return results


@PIPELINES.register_module()
class PointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        """
        Args:
            points: (N_points, 3):  3: (u, v, d)
            height: int
            width: int

        Returns:
            depth_map：(H, W)
        """
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)     # (N_points, 2)  2: (u, v)
        depth = points[:, 2]    # (N_points, )哦
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        # 获取有效投影点.
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]    # CAM_TYPE
            # 猜测liadr和cam不是严格同步的，因此lidar_ego和cam_ego可能会不一致.
            # 因此lidar-->cam的路径不采用:   lidar --> ego --> cam
            # 而是： lidar --> lidar_ego --> global --> cam_ego --> cam
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            # lidar --> lidar_ego --> global --> cam_ego --> cam
            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)     # (N_points, 3)  3: (ud, vd, d)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)      # (N_points, 3):  3: (u, v, d)

            # 再考虑图像增广
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]      # (N_points, 3):  3: (u, v, d)
            depth_map = self.points2depthmap(points_img,
                                             imgs.shape[2],     # H
                                             imgs.shape[3]      # W
                                             )
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results


@PIPELINES.register_module()
class LoadAnnotations2D(object):
    def __init__(self, min_size=2.0, filter_invisible=True):
        self.min_size = min_size
        self.filter_invisible = filter_invisible

    def __call__(self, results):
        N = len(results['cam_names'])
        ann_infos_2d = results['2D_ann_infos']
        # (N_views, 3, 3), (N_views, 3)
        post_rots, post_trans = results['img_inputs'][4:]
        img_size = results['img_inputs'][0].shape[-2:]  # (img_h, img_w)

        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []

        for i in range(N):
            gt_bboxes = ann_infos_2d['bboxes2d'][i]
            gt_labels = ann_infos_2d['labels2d'][i]
            centers2d = ann_infos_2d['centers2d'][i]
            depths = ann_infos_2d['depths'][i]

            post_rot = post_rots[i].numpy()     # (3, 3)
            post_tran = post_trans[i].numpy()    # (3, )

            if len(gt_bboxes) != 0:
                gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                    gt_bboxes,  # (N_gt, 4)  4: (x1, y1, x2, y2)
                    centers2d,  # (N_gt, 2)
                    gt_labels,  # (N_gt, )
                    depths,     # (N_gt, )
                    post_rot=post_rot,  # (3, 3)
                    post_tran=post_tran,  # (3, )
                    img_size=img_size
                )

            if len(gt_bboxes) != 0 and self.filter_invisible:
                # bboxes: (N_valid, 4) 4: (x1, y1, x2, y2)
                # centers2d: (N_valid, 2)
                # gt_labels: (N_valid,)
                # depths: (N_valid,)
                gt_bboxes, centers2d, gt_labels, depths = self._filter_invisible(gt_bboxes, centers2d, gt_labels,
                                                                                 depths, img_size=img_size)

            gt_bboxes = torch.from_numpy(gt_bboxes).to(torch.float32)
            centers2d = torch.from_numpy(centers2d).to(torch.float32)
            gt_labels = torch.from_numpy(gt_labels).to(torch.int64)
            depths = torch.from_numpy(depths).to(torch.float32)

            new_gt_bboxes.append(gt_bboxes)
            new_centers2d.append(centers2d)
            new_gt_labels.append(gt_labels)
            new_depths.append(depths)

        results['gt_bboxes'] = new_gt_bboxes    # List[(N_gt0, 4), (N_gt1, 4), ...]   4: (x1, y1, x2, y2)
        results['centers2d'] = new_centers2d    # List[(N_gt0, 2), (N_gt1, 2), ...]   4: (x1, y1, x2, y2)
        results['gt_labels'] = new_gt_labels
        results['depths'] = new_depths

        return results

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths, post_rot, post_tran, img_size):
        """
        Args:
            bboxes: (N_gt, 4)  4: (x1, y1, x2, y2)
            centers2d: (N_gt, 2)
            gt_labels: (N_gt, )
            depths: (N_gt, )
            post_rot: (3, 3)
            post_tran: (3, )
            img_size: (img_h, img_w)
        Returns:
            bboxes: (N_gt, 4)  4: (x1, y1, x2, y2)
            centers2d: (N_gt, 2)
            gt_labels: (N_gt, )
            depths: (N_gt, )
        """
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = img_size
        N = bboxes.shape[0]
        corners = np.stack([
            bboxes[:, [0, 1]],  # 左上角 (x1, y1)
            bboxes[:, [2, 1]],  # 右上角 (x2, y1)
            bboxes[:, [2, 3]],  # 右下角 (x2, y2)
            bboxes[:, [0, 3]]   # 左下角 (x1, y2)
        ], axis=1)   # (N, 4, 2)
        corners = np.concatenate([corners, np.ones((N, 4, 1))], axis=-1)    # (N, 4, 3)
        corners = np.matmul(corners, post_rot.T) + post_tran    # (N, 4, 3)
        x1 = np.min(corners[..., 0], axis=1)    # (N, )
        y1 = np.min(corners[..., 1], axis=1)    # (N, )
        x2 = np.max(corners[..., 0], axis=1)    # (N, )
        y2 = np.max(corners[..., 1], axis=1)   # (N, )
        bboxes = np.stack([x1, y1, x2, y2], axis=-1)    # (N, 4)
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & (
                    (bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)  # (N_gt, )

        centers2d = np.concatenate([centers2d, np.ones((N, 1))], axis=-1)  # (N, 3)
        centers2d = np.matmul(centers2d, post_rot.T) + post_tran    # (N, 3)
        centers2d = centers2d[:, :2]

        bboxes = bboxes[keep]
        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths

    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths, img_size):
        """
        Args:
            bboxes: (N_gt, 4)  4: (x1, y1, x2, y2)
            centers2d: (N_gt, 2)
            gt_labels: (N_gt, )
            depths: (N_gt, )
            img_size: (img_h, img_w)
        Returns:
            bboxes: (N_valid, 4)  4: (x1, y1, x2, y2)
            centers2d: (N_valid, 2)
            gt_labels: (N_valid, )
            depths: (N_valid, )
        """
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = img_size
        indices_maps = np.zeros((fH, fW))       # (fH, fW)

        tmp_bboxes = np.zeros_like(bboxes)      # (N_gt, 4)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)        # (N_gt, 4)  4: (x1, y1, x2, y2)

        # 将gt_boxes2D按照深度进行排序, (N_gt, 4).
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)      # 只选择未被遮挡的gt_bboxes.
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths


@PIPELINES.register_module()
class PointToMultiView(object):
    def __init__(self, grid_config, downsample=1, with_seg=False):
        self.downsample = downsample
        self.grid_config = grid_config
        self.with_seg = with_seg

    def points2map(self, points, point_label, height, width):
        """
        Args:
            points: (N_points, 3):  3: (u, v, d)
            point_label: (N_points, )
            height: int
            width: int

        Returns:
            depth_map：(H, W)
        """
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        downsample_coor = torch.round(points[:, :2] / self.downsample)     # (N_points, 2)  2: (u, v)
        depth = points[:, 2]    # (N_points, )哦
        kept1 = (downsample_coor[:, 0] >= 0) & (downsample_coor[:, 0] < width) & (
            downsample_coor[:, 1] >= 0) & (downsample_coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])

        # 获取有效投影点.
        coor = downsample_coor.clone()
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth

        if self.with_seg:
            point_label = point_label.to(torch.int)
            assert torch.max(point_label) < 10
            seg_map = torch.zeros((height, width), dtype=torch.int)
            coor = downsample_coor.clone()
            coor, point_label = coor[kept1], point_label[kept1]  # (N, 2), (N, )
            ranks = coor[:, 0] + coor[:, 1] * width
            sort = (ranks + (point_label+1) / 11.).argsort(descending=True)
            coor, point_label, ranks = coor[sort], point_label[sort], ranks[sort]
            kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
            kept2[1:] = (ranks[1:] != ranks[:-1])
            coor, point_label = coor[kept2], point_label[kept2]
            coor = coor.to(torch.long)
            seg_map[coor[:, 1], coor[:, 0]] = point_label
        else:
            seg_map = None

        return depth_map, seg_map

    def __call__(self, results):
        # (N, 3, 3), (N, 3)
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        seg_map_list = []

        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]    # CAM_TYPE
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            point_filename = filename.replace('samples/', 'samples_point_label/').replace('.jpg', '.npy')
            point_label = np.load(point_filename).astype(np.float64)[:4].T  # (N_points, 4):  3: (u, v, d, seg_label)
            point_label = torch.from_numpy(point_label).to(torch.float32)

            points_img = point_label[:, :3]
            point_label = point_label[:, 3]
            # 再考虑图像增广
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]      # (N_points, 3):  3: (u, v, d)
            depth_map, seg_map = self.points2map(points_img, point_label, imgs.shape[2], imgs.shape[3])
            depth_map_list.append(depth_map)
            seg_map_list.append(seg_map)

        depth_map = torch.stack(depth_map_list)
        if self.with_seg:
            seg_map = torch.stack(seg_map_list)
        results['gt_depth'] = depth_map
        results['gt_semantic'] = seg_map
        return results


@PIPELINES.register_module()
class AlignCameraCenter(object):
    def __init__(self):
        pass

    def __call__(self, results):
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]

        camera_center = sensor2egos[:6, ...].mean(dim=0)[:2, 3]     # (tx, ty)
        center_mat = torch.eye(4)
        center_mat[:2, 3] = -camera_center

        sensor2egos = center_mat.unsqueeze(dim=0) @ sensor2egos
        ego2globals = ego2globals @ torch.inverse(center_mat).unsqueeze(dim=0)

        if results.get('gt_bboxes_3d', None) is not None:
            gt_bboxes_3d = results['gt_bboxes_3d']
            gt_bboxes_3d.tensor[:, :2] -= camera_center

        bda_rot = torch.eye(3)

        results['camera_center'] = camera_center
        results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins, post_rots,
                                 post_trans, bda_rot)
        return results

@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        semantics = torch.from_numpy(semantics)
        mask_lidar = torch.from_numpy(mask_lidar)
        mask_camera = torch.from_numpy(mask_camera)

        if results.get('flip_dx', False):
            semantics = torch.flip(semantics, [0])
            mask_lidar = torch.flip(mask_lidar, [0])
            mask_camera = torch.flip(mask_camera, [0])

        if results.get('flip_dy', False):
            semantics = torch.flip(semantics, [1])
            mask_lidar = torch.flip(mask_lidar, [1])
            mask_camera = torch.flip(mask_camera, [1])

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results