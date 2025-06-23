# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.misc import memory_refresh

from mmdet3d.models import DETECTORS
from .polarbevdet import PolarBEVDet
from mmdet3d.models.builder import build_head


@DETECTORS.register_module()
class PolarBEVDetOCC(PolarBEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 **kwargs):
        super(PolarBEVDetOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      centers2d=None,
                      depths=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        scene_tokens = [img_meta['scene_token'] for img_meta in img_metas]
        B = len(scene_tokens)
        if self.history_scene_tokens is None:
            for i in range(B):
                img_metas[i]['prev_exists'] = False
        else:
            for i in range(B):
                if scene_tokens[i] != self.history_scene_tokens[i]:
                    img_metas[i]['prev_exists'] = False
        self.history_scene_tokens = scene_tokens

        # img_feats: (B, N_views, C, fH, fW)
        # bev_feats: (B, C=64 * 4=256, bev_H, bev_W)
        # depth_digit: (B * N_views, D=112, H/16, W/16)
        img_feats, bev_feats, pts_feats, depth, pred_seg = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas)

        # If we're training depth...
        losses = dict()
        gt_depth = kwargs['gt_depth']  # (B, N_views, img_H, img_W)
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        if isinstance(loss_depth, dict):
            losses.update(loss_depth)
        else:
            losses['loss_depth'] = loss_depth

        # if self.seg_head is not None:
        #     losses_seg = self.seg_head.loss(pred_seg, kwargs['semantic_map'])
        #     losses['loss_seg'] = losses_seg

        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        loss_occ = self.forward_occ_train(bev_feats[0], voxel_semantics, mask_camera)
        losses.update(loss_occ)

        if self.aux_img_head is not None:
            if self.stride == 32:
                B, N, C, H, W = img_feats.shape
                img_feats = img_feats.flatten(0, 1)  # (B*N_views, C, fH, fW)
                img_feats = F.interpolate(img_feats, scale_factor=1/2, mode='bilinear', align_corners=True)
                img_feats = img_feats.reshape(B, N, C, H//2, W//2)

            location = self.prepare_location(img_metas, img_feats)  # (B*N_view, fh, fw, 2)
            outs_aux = self.aux_img_head(location, img_feats)
            loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_aux, img_metas]
            losses2d = self.aux_img_head.loss(*loss2d_inputs)
            losses.update(losses2d)

        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ


    def simple_test(self,
                    points,
                    img_metas,
                    img_inputs=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton.
        Returns:
            bbox_list: List[dict0, dict1, ...]   len = bs
            dict: {
                'pts_bbox':  dict: {
                              'boxes_3d': (N, 9)
                              'scores_3d': (N, )
                              'labels_3d': (N, )
                             }
            }
        """
        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            img_metas[0]['prev_exists'] = img_inputs[0].new_zeros(1).int()
            self.reset_memory()
            self.reset_stereo_memory()
        else:
            img_metas[0]['prev_exists'] = img_inputs[0].new_ones(1).int()

        img_feats, bev_feats, pts_feats, depth, pred_seg = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        occ_list = self.simple_test_occ(bev_feats[0], img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]

        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        # occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs

