# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math

import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch import nn
import torch.nn.functional as F

from mmdet3d.core import circle_nms
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from ...core.post_processing import nms_bev
from mmdet3d.models import builder
from mmdet3d.models.utils import clip_sigmoid
from mmdet.core import build_bbox_coder, multi_apply, reduce_mean
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes


@HEADS.register_module(force=True)
class SeparateHead(BaseModule):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int, optional): Output channels.
            Default: 64.
        final_kernel (int, optional): Kernel size for the last conv layer.
            Default: 1.
        init_bias (float, optional): Initial bias. Default: -2.19.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(SeparateHead, self).__init__(init_cfg=init_cfg)
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg)
                )
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True)
            )

            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            if init_cfg is None:
                self.init_cfg = dict(type='Kaiming', layer='Conv2d')

        # self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@HEADS.register_module()
class Polar_CenterHead(BaseModule):
    """CenterHead for CenterPoint.

    Args:
        in_channels (list[int] | int, optional): Channels of the input
            feature map. Default: [128].
        tasks (list[dict], optional): Task information including class number
            and class names. Default: None.
        train_cfg (dict, optional): Train-time configs. Default: None.
        test_cfg (dict, optional): Test-time configs. Default: None.
        bbox_coder (dict, optional): Bbox coder configs. Default: None.
        common_heads (dict, optional): Conv information for common heads.
            Default: dict().
        loss_cls (dict, optional): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict, optional): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict, optional): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int, optional): Output channels for share_conv
            layer. Default: 64.
        num_heatmap_convs (int, optional): Number of conv layers for heatmap
            conv layer. Default: 2.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 init_cfg=None,
                 task_specific=True,
                 align_camera_center=False,
                 ):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(Polar_CenterHead, self).__init__(init_cfg=init_cfg)

        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_anchor_per_locs = [n for n in num_classes]
        self.fp16_enabled = False

        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        self.task_heads = nn.ModuleList()
        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
            self.task_heads.append(builder.build_head(separate_head))

        self.with_velocity = 'vel' in common_heads.keys()
        self.task_specific = task_specific
        self.align_camera_center = align_camera_center

        self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def forward_single(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        x = self.shared_conv(x)     # (B, C'=share_conv_channel, H, W)

        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            results: Tuple(
                List[ret_dict_task0_level0, ...],   len = num_levels = 1
                List[ret_dict_task1_level0, ...],
                ...
            ),   len = SeparateHead的数量, 负责预测指定类别的目标.
            ret_dict: {
              reg: (B, 2, H, W)
              height: (B, 1, H, W)
              dim: (B, 3, H, W)
              rot: (B, 2, H, W)
              vel: (B, 2, H, W)
              heatmap: (B, n_cls, H, W)
            }
        """
        return multi_apply(self.forward_single, feats)

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor, optional): Mask of the feature map with the
                shape of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets.

        How each output is transformed:

            Each nested list is transposed so that all same-index elements in
            each sub-list (1, ..., N) become the new sub-lists.
                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.     # List[(N_gt0, 7/9), (N_gt1, 7/9), ...]
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.     # List[(N_gt0, ), (N_gt1, ), ...]

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: (
                    heatmaps: List[(B, N_cls0, H, W), (B, N_cls1, H, W), ...]  len = num of SeparateHead
                    anno_boxes:
                    inds:
                    masks:
                )
        """
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d)

        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks

    def cart2polar(self, gt_bboxes_3d):
        """
        Args:
            gt_bboxes_3d: (N_gt, 9)
        Returns:
            polar_gt_bboxes_3d: (N_gt, 9)
        """
        x, y, z, dx, dy, dz, rot, vx, vy = torch.split(gt_bboxes_3d, split_size_or_sections=1, dim=-1)
        azimuth = torch.atan2(y, x)
        dis = torch.sqrt(x ** 2 + y ** 2)
        azim_rot = rot - azimuth

        v_abs = torch.sqrt(vx ** 2 + vy ** 2)
        v_angle = torch.atan2(vy, vx)
        v_azimuth = v_abs * torch.sin(v_angle - azimuth)
        v_radius = v_abs * torch.cos(v_angle - azimuth)
        polar_gt_bboxes_3d = torch.cat([azimuth, dis, z, dx, dy, dz, azim_rot, v_radius, v_azimuth], dim=-1)

        return polar_gt_bboxes_3d

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.  # (N_gt, 7/9)
            gt_labels_3d (torch.Tensor): Labels of boxes.   # (N_gt, )

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.
                - heatmaps: list[torch.Tensor]: Heatmap scores.   # List[(N_cls0, H, W), (N_cls1, H, W), ...]
                            len = num of tasks
                - anno_boxes: list[torch.Tensor]: Ground truth boxes.   # List[(max_objs, 10), (max_objs, 10), ...]
                - inds: list[torch.Tensor]: Indexes indicating the position
                        of the valid boxes.     # List[(max_objs, ), (max_objs, ), ...]
                - masks: list[torch.Tensor]: Masks indicating which boxes
                        are valid.              # List[(max_objs, ), (max_objs, ), ...]
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)       # (N_gt, 7/9)

        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])       # (Dx, Dy, Dz)
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']   # (W, H)

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)

        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))      # (N_cls, H, W)

            if self.with_velocity:
                anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                                  dtype=torch.float32)      # (max_objs, 10)
            else:
                anno_box = gt_bboxes_3d.new_zeros((max_objs, 8),
                                                  dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs, ), dtype=torch.int64)     # (max_objs, )
            mask = gt_bboxes_3d.new_zeros((max_objs, ), dtype=torch.uint8)    # (max_objs, )

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cur_gt_box = task_boxes[idx][k:k+1]     # (1, 9)
                width = task_boxes[idx][k][3]       # dx
                length = task_boxes[idx][k][4]      # dy
                cls_id = task_classes[idx][k] - 1

                # (azimuth, dis, z, dx, dy, dz, azim_rot, azim_vx, azim_vy)
                gt_bboxes_3d_polar = self.cart2polar(cur_gt_box).squeeze()      # (9, )

                corners = LiDARInstance3DBoxes(cur_gt_box, box_dim=9).corners  # (1, 8, 3)
                bottom_corners_bev = corners[:, [0, 3, 7, 4], :2]  # (1, 4, 2)   2:(x, y)
                rhos = torch.norm(bottom_corners_bev, dim=-1)  # (1, 4)
                azs = torch.atan2(bottom_corners_bev[..., 1], bottom_corners_bev[..., 0])  # (1, 4)
                min_az = torch.min(azs, dim=-1)[0]
                max_az = torch.max(azs, dim=-1)[0]
                min_rho = torch.min(rhos, dim=-1)[0]
                max_rho = torch.max(rhos, dim=-1)[0]
                da = max_az - min_az
                da[da > math.pi] = 2 * math.pi - da[da > math.pi]
                da = da / voxel_size[0] / self.train_cfg['out_size_factor']  # (1, )
                dr = (max_rho - min_rho) / voxel_size[1] / self.train_cfg['out_size_factor']  # (1, )

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (da, dr),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    azimuth, dis, z = gt_bboxes_3d_polar[0:3]

                    coor_x = (
                        azimuth - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        dis - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1

                    azim_rot = gt_bboxes_3d_polar[6]
                    box_dim = gt_bboxes_3d_polar[3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    if self.with_velocity:
                        azim_vx, azim_vy = gt_bboxes_3d_polar[7:9]
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device=device),   # tx, ty
                            z.unsqueeze(0), box_dim,        # z, log(dx), log(dy), log(dz)
                            torch.sin(azim_rot).unsqueeze(0),    # sin(rot)
                            torch.cos(azim_rot).unsqueeze(0),    # cos(rot)
                            azim_vx.unsqueeze(0),    # vx
                            azim_vy.unsqueeze(0)     # vy
                        ])      # [tx, ty, z, log(dx), log(dy), log(dz), sin(azim_rot), cos(azim_rot), azim_vx, azim_vy]
                    else:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device=device),
                            z.unsqueeze(0), box_dim,
                            torch.sin(azim_rot).unsqueeze(0),
                            torch.cos(azim_rot).unsqueeze(0)
                        ])

            heatmaps.append(heatmap)    # append (N_cls, H, W)
            anno_boxes.append(anno_box)     # append (max_objs, 10)
            masks.append(mask)      # append (max_objs, )
            inds.append(ind)        # append (max_objs, )
        return heatmaps, anno_boxes, inds, masks

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.     # List[(N_gt0, 7/9), (N_gt1, 7/9), ...]
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.     # List[(N_gt0, ), (N_gt1, ), ...]
            preds_dicts (dict): Tuple(
                    List[ret_dict_task0_level0, ...],   len = num_levels = 1
                    List[ret_dict_task1_level0, ...],
                    ...
                )
                ret_dict: {
                  reg: (B, 2, H, W)
                  height: (B, 1, H, W)
                  dim: (B, 3, H, W)
                  rot: (B, 2, H, W)
                  vel: (B, 2, H, W)
                  heatmap: (B, n_cls, H, W)
                }
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)

        loss_dict = dict()
        if not self.task_specific:
            loss_dict['loss'] = 0
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            cls_avg_factor = torch.clamp(
                reduce_mean(heatmaps[task_id].new_tensor(num_pos)),
                min=1).item()

            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],   # (B, cur_N_cls, H, W)
                heatmaps[task_id],          # (B, cur_N_cls, H, W)
                avg_factor=cls_avg_factor
            )

            # (B, max_objs, 10)
            # 10: (tx, ty, z, log(dx), log(dy), log(dz), sin(azim_rot), cos(azim_rot), azim_vx, azim_vy)
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (
                    preds_dict[0]['reg'],
                    preds_dict[0]['height'],
                    preds_dict[0]['dim'],
                    preds_dict[0]['rot'],
                    preds_dict[0]['vel'],
                ),
                dim=1,
            )   # (B, 10, H, W)    10: (tx, ty, z, log(dx), log(dy), log(dz), sin(azim_rot), cos(azim_rot), azim_vx, azim_vy)

            # Regression loss for dimension, offset, height, rotation
            num = masks[task_id].float().sum()
            ind = inds[task_id]     # (B, max_objs)
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()   # (B, H, W, 10)
            pred = pred.view(pred.size(0), -1, pred.size(3))    # (B, H*W, 10)
            pred = self._gather_feat(pred, ind)     # (B, max_objs, 10)
            # (B, max_objs) -->  (B, max_objs, 1) --> (B, max_objs, 10)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            num = torch.clamp(
                reduce_mean(target_box.new_tensor(num)), min=1e-4).item()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            bbox_weights = mask * mask.new_tensor(self.code_weights)
            if self.task_specific:
                name_list = ['xy', 'z', 'whl', 'yaw', 'vel']
                clip_index = [0, 2, 3, 6, 8, 10]
                for reg_task_id in range(len(name_list)):
                    pred_tmp = pred[..., clip_index[reg_task_id]:clip_index[reg_task_id + 1]]    # (B, max_objs, K)
                    target_box_tmp = target_box[..., clip_index[reg_task_id]:clip_index[reg_task_id + 1]]    # (B, max_objs, K)
                    bbox_weights_tmp = bbox_weights[..., clip_index[reg_task_id]:clip_index[reg_task_id + 1]]    # (B, max_objs, K)
                    loss_bbox_tmp = self.loss_bbox(
                        pred_tmp,
                        target_box_tmp,
                        bbox_weights_tmp,
                        avg_factor=(num + 1e-4))
                    loss_dict[f'task{task_id}.loss_%s' %
                              (name_list[reg_task_id])] = loss_bbox_tmp
                loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            else:
                loss_bbox = self.loss_bbox(
                    pred, target_box, bbox_weights, avg_factor=num)
                loss_dict['loss'] += loss_bbox
                loss_dict['loss'] += loss_heatmap

        return loss_dict

    @force_fp32(apply_to=('preds_dicts', ))
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
                Tuple(
                    List[ret_dict_task0_level0, ...],   len = num_levels = 1
                    List[ret_dict_task1_level0, ...],
                    ...
                )

                ret_dict: {
                  reg: (B, 2, H, W)
                  height: (B, 1, H, W)
                  dim: (B, 3, H, W)
                  rot: (B, 2, H, W)
                  vel: (B, 2, H, W)
                  heatmap: (B, n_cls, H, W)
                }
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
            ret_list: List[p_list0, p_list1, ...]
                p_list: List[(N, 9), (N, ), (N, )]
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()     # (B, n_cls, H, W)

            batch_reg = preds_dict[0]['reg']        # (B, 2, H, W)
            batch_hei = preds_dict[0]['height']     # (B, 1, H, W)

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])     # (B, 3, H, W)
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)    # (B, 1, H, W)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)    # (B, 1, H, W)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']    # (B, 2, H, W)
            else:
                batch_vel = None

            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)

            batch_reg_preds = [box['bboxes'] for box in temp]   # List[(K0, 9), (K1, 9), ...]   len = bs
            batch_cls_preds = [box['scores'] for box in temp]   # List[(K0, ), (K1, ), ...]   len = bs
            batch_cls_labels = [box['labels'] for box in temp]  # List[(K0, ), (K1, ), ...]   len = bs
            nms_type = self.test_cfg.get('nms_type')
            if isinstance(nms_type, list):
                nms_type = nms_type[task_id]
            if nms_type == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    # map the bbox from camera center to ege center
                    if self.align_camera_center:
                        boxes3d[:, :2] += img_metas[i]['camera_center'].to(boxes3d)

                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas,
                                             task_id))

        # Merge branches results
        num_samples = len(rets[0])      # bs

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])

        return ret_list

    def get_task_detections(self, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas,
                            task_id):
        """Rotate nms for each task.

        Args:
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].       # List[(K0, ), (K1, ), ...]   len = bs
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].    # List[(K0, 9), (K1, 9), ...]   len = bs
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].       # List[(K0, ), (K1, ), ...]   len = bs
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:
                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].

            List[p_dict0, p_dict1, ...]     len = batch_size
                p_dict: dict{
                    bboxes: (K', 9)
                    scores: (K', )
                    labels: (K', )
                }
        """
        predictions_dicts = []

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):
            # box_preds: (K, 9)
            # cls_preds: (K, )
            # cls_labels: (K, )
            default_val = [1.0 for _ in range(len(self.task_heads))]
            factor = self.test_cfg.get('nms_rescale_factor',
                                       default_val)[task_id]
            if isinstance(factor, list):
                for cid in range(len(factor)):
                    box_preds[cls_labels == cid, 3:6] = \
                        box_preds[cls_labels == cid, 3:6] * factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] * factor

            # Apply NMS in birdeye view
            top_labels = cls_labels.long()      # (K, )
            top_scores = cls_preds.squeeze(-1) if cls_preds.shape[0] > 1 \
                else cls_preds                  # (K, )

            if top_scores.shape[0] != 0:
                boxes_for_nms = img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev     # (K, 5) (x, y, dx, dy, yaw)
                # the nms in 3d detection just remove overlap boxes.
                if isinstance(self.test_cfg['nms_thr'], list):
                    nms_thresh = self.test_cfg['nms_thr'][task_id]
                else:
                    nms_thresh = self.test_cfg['nms_thr']

                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=nms_thresh,
                    pre_max_size=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'],
                    xyxyr2xywhr=False,
                    )
            else:
                selected = []

            if isinstance(factor, list):
                for cid in range(len(factor)):
                    box_preds[top_labels == cid, 3:6] = \
                        box_preds[top_labels == cid, 3:6] / factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] / factor

            # if selected is not None:
            selected_boxes = box_preds[selected]    # (K', 9)
            selected_labels = top_labels[selected]  # (K', )
            selected_scores = top_scores[selected]  # (K', )

            # map the bbox from camera center to ege center
            if self.align_camera_center:
                selected_boxes[:, :2] += img_metas[i]['camera_center'].to(box_preds)

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                predictions_dict = dict(
                    bboxes=selected_boxes,
                    scores=selected_scores,
                    labels=selected_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
