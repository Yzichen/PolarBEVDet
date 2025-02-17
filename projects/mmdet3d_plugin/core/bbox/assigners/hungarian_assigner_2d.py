# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core import bbox_cxcywh_to_xyxy

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner2D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 centers2d_cost=dict(type='BBox3DL1Cost', weight=1.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.centers2d_cost = build_match_cost(centers2d_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               pred_centers2d,
               gt_bboxes,
               gt_labels,
               centers2d,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): (N_pred=fH*fW, 4)  4: (cx, cy, w, h)   normalized
            cls_pred (Tensor): (N_pred=fH*fW, n_cls)
            pred_centers2d (Tensor): (N_pred=fH*fW, 2)           normalized
            gt_bboxes (Tensor): (N_gt, 4)   4: (x1, y1, x2, y2)  real
            gt_labels (Tensor): (N_gt, )
            centers2d (Tensor): (N_gt, 2)  2: (cx, cy)  real
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)     # (N_pred, )
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)      # (N_pred, )
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta['pad_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)     # (1, 4)    4: (img_w, img_h, img_w, img_h)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)     # (N_pred, N_gt)
        # regression L1 cost
        normalize_gt_bboxes = gt_bboxes / factor    # real --> normalized
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)    # (N_pred, N_gt)
        # regression iou cost, defaultly giou is used in official DETR.
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)     # (N_pred, N_gt)

        # center2d L1 cost
        normalize_centers2d = centers2d / factor[:, 0:2]    # (N_gt, 2)  2: (cx, cy)  real --> normalized
        centers2d_cost = self.centers2d_cost(pred_centers2d, normalize_centers2d)   # (N_pred, N_gt)

        # weighted sum of above four costs
        cost = cls_cost + reg_cost + iou_cost + centers2d_cost      # (N_pred, N_gt)
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)       # (N_gt, )
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)       # (N_gt, )

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
