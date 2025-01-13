import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS, build_loss


def clip_sigmoid(x, eps=1e-4):
    """Sigmoid function for input feature.

    Args:
        x (torch.Tensor): Input feature map with the shape of [B, N, H, W].
        eps (float, optional): Lower bound of the range to be clamped to.
            Defaults to 1e-4.

    Returns:
        torch.Tensor: Feature map after sigmoid.
    """
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


# SAE in paper
# Initially, I wanted to apply bev segmentation to supervise the learning of attention weight,
# but the effect was not obvious.
@HEADS.register_module()
class SegHead(BaseModule):
    def __init__(self,
                 in_dim=256,
                 mid_dim=64,
                 num_classes=1,
                 ):
        super(SegHead, self).__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim

        self.seg_conv = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = x[0]
        pred_seg = self.seg_conv(x)      # (B, C=1, Dy, Dx)
        pred_seg = clip_sigmoid(pred_seg)   # (B, C=1, Dy, Dx)
        return pred_seg
