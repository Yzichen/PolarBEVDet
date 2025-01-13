# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.builder import LOSSES
from mmdet.models.losses import weighted_loss


@weighted_loss
def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2):
    """
    Args:
        pred: (B, H, W)
        target: (B, H, W)
        valid_mask: (B, H, W)
        smooth:
        exponent:

    Returns:

    """
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)  # (B, H*W)
    target = target.reshape(target.shape[0], -1)    # (B, H*W)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)    # (B, H*W)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((pred.pow(exponent) + target.pow(exponent)) * valid_mask, dim=1) + smooth

    return 1 - num / den


@LOSSES.register_module(force=True)
class DiceLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, smooth=1.0, exponent=2.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.smooth = smooth
        self.exponent = exponent

    def forward(self, pred, target, valid_mask):
        """
        :param pred: (B, C, H, W)
        :param target: (B, C, H, W)
        :param valid_mask: (B, H, W)
        :return:
        """
        assert pred.shape[0] == target.shape[0]
        total_loss = 0
        num_classes = pred.shape[1]
        for i in range(num_classes):
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[:, i],
                valid_mask=valid_mask,
                smooth=self.smooth,
                exponent=self.exponent
            )
            total_loss += dice_loss

        loss = total_loss / num_classes
        loss = self.loss_weight * loss
        return loss

