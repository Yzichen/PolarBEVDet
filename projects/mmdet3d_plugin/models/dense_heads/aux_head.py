# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from ..utils.misc import draw_heatmap_gaussian, apply_center_offset, apply_ltrb
from mmdet.core import bbox_overlaps
from mmdet3d.models.utils import clip_sigmoid


@HEADS.register_module()
class AuxHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 embed_dims=256,
                 stride=16,
                 use_hybrid_tokens=False,
                 train_ratio=1.0,
                 infer_ratio=1.0,
                 sync_cls_avg_factor=False,
                 loss_cls2d=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_centerness=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
                 loss_centers2d=dict(type='L1Loss', loss_weight=5.0),
                 train_cfg=dict(
                     assigner2d=dict(
                         type='HungarianAssigner2D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                         centers2d_cost=dict(type='BBox3DL1Cost', weight=1.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        if train_cfg:
            assert 'assigner2d' in train_cfg, 'assigner2d should be provided '\
                'when train_cfg is set.'
            assigner2d = train_cfg['assigner2d']

            self.assigner2d = build_assigner(assigner2d)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.stride=stride
        self.use_hybrid_tokens=use_hybrid_tokens
        self.train_ratio=train_ratio
        self.infer_ratio=infer_ratio

        super(AuxHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls2d = build_loss(loss_cls2d)
        self.loss_bbox2d = build_loss(loss_bbox2d)
        self.loss_iou2d = build_loss(loss_iou2d)
        self.loss_centers2d = build_loss(loss_centers2d)
        self.loss_centerness = build_loss(loss_centerness)

        self._init_layers()

    def _init_layers(self):
        self.cls = nn.Conv2d(self.embed_dims, self.num_classes, kernel_size=1)

        self.shared_reg = nn.Sequential(
                                 nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.embed_dims),
                                 nn.ReLU(),)

        self.shared_cls = nn.Sequential(
                                 nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.embed_dims),
                                 nn.ReLU(),)

        self.centerness = nn.Conv2d(self.embed_dims, 1, kernel_size=1)
        self.ltrb = nn.Conv2d(self.embed_dims, 4, kernel_size=1)
        self.center2d = nn.Conv2d(self.embed_dims, 2, kernel_size=1)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls.bias, bias_init)
        nn.init.constant_(self.centerness.bias, bias_init)

    def forward(self, location, img_feats):
        """
        Args:
            location: (B*N_view, fH, fW, 2)
            img_feats: (B, N_views, C'=256, fH, fW)
        Returns:

        """
        src = img_feats    # (B, N_views, C'=256, fH, fW)
        bs, n, c, h, w = src.shape

        x = src.flatten(0, 1)   # (B*N_views, C=256, fH, fW)
        cls_feat = self.shared_cls(x)   # (B*N_views, C=256, fH, fW)
        cls = self.cls(cls_feat)    # (B*N_views, n_cls, fH, fW)
        centerness = self.centerness(cls_feat)  # (B*N_views, 1, fH, fW)

        # (B*N_views, n_cls, fH, fW) --> (B*N_views, fH, fW, n_cls)
        # --> (B*N_views, fH*fW, n_cls)
        cls_logits = cls.permute(0, 2, 3, 1).reshape(bs*n, -1, self.num_classes)
        # (B*N_views, 1, fH, fW) --> (B*N_views, fH, fW, 1)
        # --> (B*N_views, fH*fW, 1)
        centerness = centerness.permute(0, 2, 3, 1).reshape(bs*n, -1, 1)

        reg_feat = self.shared_reg(x)   # (B*N_views, C=256, fH, fW)
        # (B*N_views, C=256, fH, fW) --> (B*N_views, 4, fH, fW)
        # --> (B*N_views, fH, fW, 4)
        ltrb = self.ltrb(reg_feat).permute(0, 2, 3, 1).contiguous()
        ltrb = ltrb.sigmoid()   # (B*N_views, fH, fW, 4)
        # (B*N_views, C=256, fH, fW) --> (B*N_views, 2, fH, fW)
        # --> (B*N_views, fH, fW, 2)
        centers2d_offset = self.center2d(reg_feat).permute(0, 2, 3, 1).contiguous()

        centers2d = apply_center_offset(location, centers2d_offset)     # (B*N_views, fH, fW, 2)
        bboxes = apply_ltrb(location, ltrb)     # (B*N_views, fH, fW, 4)
            
        pred_bboxes = bboxes.view(bs*n, -1, 4)      # (B*N_views, fH, fW, 4) --> (B*N_views, fH*fW, 4)
        pred_centers2d = centers2d.view(bs*n, -1, 2)    # (B*N_views, fH, fW, 2) --> (B*N_views, fH*fW, 2)

        outs = {
                'enc_cls_scores': cls_logits,   # (B*N_views, fH*fW, n_cls)
                'enc_bbox_preds': pred_bboxes,  # (B*N_views, fH*fW, 4)
                'pred_centers2d': pred_centers2d,   # (B*N_views, fH*fW, 2)
                'centerness': centerness,   # (B*N_views, fH*fW, 1)
            }

        return outs
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes2d_list,
             gt_labels2d_list,
             centers2d,
             depths,
             preds_dicts,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list: Tuple(List[(N_gt0_0, 4), (N_gt0_1, 4), ...], List[(N_gt1_0, 4), (N_gt1_1, 4), ...], ...),   外部Tuple表示batch内不同样本， 内部List表示不同View.
            gt_labels_list: Tuple(List[(N_gt0_0, ), (N_gt0_1, ), ...], List[(N_gt1_0, ), (N_gt1_1, ), ...], ...),   外部Tuple表示batch内不同样本， 内部List表示不同View.
            centers2d: Tuple(List[(N_gt0_0, 2), (N_gt0_1, 2), ...], List[(N_gt1_0, 2), (N_gt1_1, 2), ...], ...),
            depths: Tuple(List[(N_gt0_0, ), (N_gt0_1, ), ...], List[(N_gt1_0, ), (N_gt1_1, ), ...], ...),
            preds_dicts: dict{
                'enc_cls_scores': (B*N_views, fH*fW, n_cls)
                'enc_bbox_preds': (B*N_views, fH*fW, 4)
                'pred_centers2d': (B*N_views, fH*fW, 2)
                'centerness': (B*N_views, fH*fW, 1)
                'topk_indexes': (B, num_sample_tokens, 1)
            }
            img_metas: Tuple(dict0, dict1, ...)
            gt_bboxes_ignore: None
            }

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        enc_cls_scores = preds_dicts['enc_cls_scores']      # (B*N_views, fH*fW, n_cls)
        enc_bbox_preds = preds_dicts['enc_bbox_preds']      # (B*N_views, fH*fW, 4)
        pred_centers2d = preds_dicts['pred_centers2d']      # (B*N_views, fH*fW, 2)
        centerness = preds_dicts['centerness']              # (B*N_views, fH*fW, 1)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        all_gt_bboxes2d_list = [bboxes2d for i in gt_bboxes2d_list for bboxes2d in i]  # List[(N_gt0, 4), (N_gt1, 4), ...]  len=B*N_views
        all_gt_labels2d_list = [labels2d for i in gt_labels2d_list for labels2d in i]  # List[(N_gt0, ), (N_gt1, ), ...]
        all_centers2d_list = [center2d for i in centers2d for center2d in i]    # List[(N_gt0, 2), (N_gt1, 2), ...]
        all_depths_list = [depth for i in depths for depth in i]    # List[(N_gt0, ), (N_gt1, ), ...]

        enc_loss_cls, enc_losses_bbox, enc_losses_iou, centers2d_losses, centerness_losses = \
            self.loss_single(enc_cls_scores, enc_bbox_preds, pred_centers2d, centerness,
                             all_gt_bboxes2d_list, all_gt_labels2d_list, all_centers2d_list,
                             all_depths_list, img_metas, gt_bboxes_ignore)
        loss_dict['enc_loss_cls'] = enc_loss_cls
        loss_dict['enc_loss_bbox'] = enc_losses_bbox
        loss_dict['enc_loss_iou'] = enc_losses_iou
        loss_dict['centers2d_losses'] = centers2d_losses
        loss_dict['centerness_losses'] = centerness_losses
    
        return loss_dict

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pred_centers2d,
                    centerness,
                    gt_bboxes_list,
                    gt_labels_list,
                    all_centers2d_list,
                    all_depths_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores: (B*N_views, fH*fW, n_cls)
            bbox_preds: (B*N_views, fH*fW, 4)   4: cx, cy, w, h
            pred_centers2d: (B*N_views, fH*fW, 2)   2: cx, cy
            centerness: (B*N_views, fH*fW, 1)
            gt_bboxes_list(list[Tensor]):  List[(N_gt0, 4), (N_gt1, 4), ...]  len=B*N_views     4: x1, y1, x2, y2
            gt_labels_list(list[Tensor]):  List[(N_gt0, ), (N_gt1, ), ...]
            all_centers2d_list(list[Tensor]):  List[(N_gt0, 2), (N_gt1, 2), ...]
            all_depths_list(list[Tensor]): List[(N_gt0, ), (N_gt1, ), ...]
            img_metas (list[dict]): Tuple(dict0, dict1, ...)
            gt_bboxes_ignore_list: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)   # num_imgs = B*N_view
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]  # List[(fH*fW, n_cls), (fH*fW, n_cls), ...]   len=B*N_views
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]  # List[(fH*fW, 4), (fH*fW, 4), ...]   len=B*N_views
        centers2d_preds_list = [pred_centers2d[i] for i in range(num_imgs)]     # List[(fH*fW, 2), (fH*fW, 2), ...]   len=B*N_views

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, centers2d_preds_list,
                                           gt_bboxes_list, gt_labels_list, all_centers2d_list,
                                           all_depths_list, img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, centers2d_targets_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)      # (B*N_view*fH*fW, )
        label_weights = torch.cat(label_weights_list, 0)    # (B*N_view*fH*fW, )
        bbox_targets = torch.cat(bbox_targets_list, 0)      # (B*N_view*fH*fW, 4)   4: normalized (cx, cy, w, h)
        bbox_weights = torch.cat(bbox_weights_list, 0)      # (B*N_view*fH*fW, 4)
        centers2d_targets = torch.cat(centers2d_targets_list, 0)    # (B*N_view*fH*fW, 2)   2: normalized (cx, cy)

        img_h, img_w, _ = img_metas[0]['pad_shape'][0]

        factors = []

        for bbox_pred in bbox_preds:    # bbox_pred: (fH*fW, 4)
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)    # (1, 4) --> (fH*fW, 4)
            factors.append(factor)
        factors = torch.cat(factors, 0)     # (B*N_view*fH*fW, 4)
        bbox_preds = bbox_preds.reshape(-1, 4)      # (B*N_view*fH*fW, 4)   4: cx, cy, w, h
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss
        loss_iou = self.loss_iou2d(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        iou_score = bbox_overlaps(bboxes_gt, bboxes, is_aligned=True).reshape(-1)   # (B*N_view*fH*fW, )

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)      # (B*N_view*fH*fW, n_cls)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls2d(
             cls_scores, (labels, iou_score.detach()), label_weights, avg_factor=cls_avg_factor)
        # loss_cls = self.loss_cls2d(
        #     cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # Centerness BCE loss
        img_shape = [img_metas[0]['pad_shape'][0]] * num_imgs
        (heatmaps, ) = multi_apply(self._get_heatmap_single, all_centers2d_list, gt_bboxes_list, img_shape)

        heatmaps = torch.stack(heatmaps, dim=0)
        centerness = clip_sigmoid(centerness)
        loss_centerness = self.loss_centerness(
            centerness,
            heatmaps.view(num_imgs, -1, 1),
            avg_factor=max(num_total_pos, 1))

        # regression L1 loss
        loss_bbox = self.loss_bbox2d(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        pred_centers2d = pred_centers2d.view(-1, 2)
        # centers2d L1 loss
        loss_centers2d = self.loss_centers2d(
            pred_centers2d, centers2d_targets, bbox_weights[:, 0:2], avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou, loss_centers2d, loss_centerness

    def _get_heatmap_single(self, obj_centers2d, obj_bboxes, img_shape):
        img_h, img_w, _ = img_shape
        heatmap = torch.zeros(img_h // self.stride, img_w // self.stride, device=obj_centers2d.device)
        if len(obj_centers2d) != 0:
            l = obj_centers2d[..., 0:1] - obj_bboxes[..., 0:1]
            t = obj_centers2d[..., 1:2] - obj_bboxes[..., 1:2]
            r = obj_bboxes[..., 2:3] - obj_centers2d[..., 0:1]
            b = obj_bboxes[..., 3:4] - obj_centers2d[..., 1:2]
            bound = torch.cat([l, t, r, b], dim=-1)
            radius = torch.ceil(torch.min(bound, dim=-1)[0] / self.stride)
            radius = torch.clamp(radius, 1.0).cpu().numpy().tolist()
            for center, r in zip(obj_centers2d, radius):
                heatmap = draw_heatmap_gaussian(heatmap, center / self.stride, radius=int(r), k=1)
        return (heatmap, )

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    centers2d_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    all_centers2d_list,
                    all_depths_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): List[(fH*fW, n_cls), (fH*fW, n_cls), ...]   len=B*N_views
            bbox_preds_list (list[Tensor]): List[(fH*fW, 4), (fH*fW, 4), ...]    4: cx, cy, w, h
            centers2d_preds_list (list[Tensor]):  List[(fH*fW, 2), (fH*fW, 2), ...]
            gt_bboxes_list (list[Tensor]): List[(N_gt0, 4), (N_gt1, 4), ...]   len=B*N_views
            gt_labels_list (list[Tensor]): List[(N_gt0, ), (N_gt1, ), ...]
            all_centers2d_list (list[Tensor]): List[(N_gt0, 2), (N_gt1, 2), ...]
            all_depths_list (list[Tensor]): List[(N_gt0, ), (N_gt1, ), ...]
            img_metas: Tuple(dict0, dict1, ...)
            gt_bboxes_ignore_list: None

        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): List[(N_pred=fH*fW, ), (N_pred=fH*fW, ), ...],    len=B*N_views
                - label_weights_list (list[Tensor]): List[(N_pred=fH*fW, ), (N_pred=fH*fW, ), ...]
                - bbox_targets_list (list[Tensor]): List[(N_pred=fH*fW, 4), (N_pred=fH*fW, 4), ...]    4: normalized (cx, cy, w, h)
                - bbox_weights_list (list[Tensor]): List[(N_pred=fH*fW, 4), (N_pred=fH*fW, 4), ...]
                - all_centers2d_list (list[Tensor]): List[(N_pred=fH*fW, 2), (N_pred=fH*fW, 2), ...]   2: normalized(cx, cy)
                - num_total_pos (int): N_pos
                - num_total_neg (int): N_neg
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)     # B*N_views
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        img_meta = {'pad_shape': img_metas[0]['pad_shape'][0]}      # img_meta={'pad_shape': (H, W, 3)}
        img_meta_list = [img_meta for _ in range(num_imgs)]     # List[{'pad_shape': (H, W, 3)}, {'pad_shape': (H, W, 3)}, ...]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         centers2d_targets_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list, centers2d_preds_list,
            gt_bboxes_list, gt_labels_list, all_centers2d_list,
            all_depths_list, img_meta_list, gt_bboxes_ignore_list)

        # labels_list: List[(N_pred=fH*fW, ), (N_pred=fH*fW, ), ...],    len=B*N_views
        # label_weights_list: List[(N_pred=fH*fW, ), (N_pred=fH*fW, ), ...]
        # bbox_targets: List[(N_pred=fH*fW, 4), (N_pred=fH*fW, 4), ...]    4: normalized (cx, cy, w, h)
        # bbox_weights: List[(N_pred=fH*fW, 4), (N_pred=fH*fW, 4), ...]
        # centers2d_targets: List[(N_pred=fH*fW, 2), (N_pred=fH*fW, 2), ...]   2: normalized(cx, cy)
        # pos_inds: List[(N_pos0, ), (N_pos1, ), ...]
        # neg_inds: List[(N_pos0, ), (N_pos1, ), ...]

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                centers2d_targets_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pred_centers2d,
                           gt_bboxes,
                           gt_labels,
                           centers2d,
                           depths,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): (N_pred=fH*fW, n_cls)
            bbox_pred (Tensor): (N_pred=fH*fW, 4)  4: (cx, cy, w, h)
            pred_centers2d (Tensor): (N_pred=fH*fW, 2)  2: (cx, cy)
            gt_bboxes (Tensor): (N_gt, 4)   4: (x1, y1, x2, y2)
            gt_labels (Tensor): (N_gt, )
            centers2d (Tensor): (N_gt, 2)   2: (cx, cy)
            img_meta (dict): {'pad_shape': (256, 704, 3)}
            gt_bboxes_ignore: None

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): (N_pred=fH*fW, )
                - label_weights (Tensor]): (N_pred=fH*fW, )
                - bbox_targets (Tensor): (N_pred=fH*fW, 4)  4: normalized (cx, cy, w, h)
                - bbox_weights (Tensor): (N_pred=fH*fW, 4)
                - centers2d_targets (Tensor): (N_pred=fH*fW, 2)  2: (cx, cy)
                - pos_inds (Tensor): (N_pos, )
                - neg_inds (Tensor): (N_neg, )
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner2d.assign(bbox_pred, cls_score, pred_centers2d, gt_bboxes,
                                               gt_labels, centers2d, img_meta, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds     # (N_pos, )
        neg_inds = sampling_result.neg_inds     # (N_neg, )

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['pad_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor   # normalized (N_pos, 4)  4: x1, y1, x2, y2
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)   # normalized (N_pos, 4)  4: cx, cy, w, h
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # centers2d target
        centers2d_targets = bbox_pred.new_full((num_bboxes, 2), 0.0, dtype=torch.float32)
        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert sampling_result.pos_assigned_gt_inds.numel() == 0
            centers2d_labels = torch.empty_like(gt_bboxes).view(-1, 2)
        else:
            centers2d_labels = centers2d[sampling_result.pos_assigned_gt_inds.long(), :]
        centers2d_labels_normalized = centers2d_labels / factor[:, 0:2]     # normalized (N_pos, 2)  2: cx, cy
        centers2d_targets[pos_inds] = centers2d_labels_normalized
        return (labels, label_weights, bbox_targets, bbox_weights, centers2d_targets,
                pos_inds, neg_inds)