# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn
from torch.utils.checkpoint import checkpoint
from mmdet3d.models import NECKS


@NECKS.register_module(force=True)
class SECONDFPN(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 final_conv_feature_dim=None,
                 use_conv_for_no_stride=False,
                 with_cp=False,
                 init_cfg=None):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_cp = with_cp
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)
        
        if final_conv_feature_dim is not None:
            self.final_feature_dim = final_conv_feature_dim
            self.final_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=sum(out_channels), out_channels=sum(out_channels) // 2, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, sum(out_channels) // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=sum(out_channels) // 2, out_channels=final_conv_feature_dim, kernel_size=1, stride=1))
        else:
            self.final_feature_dim = sum(out_channels)
            self.final_conv = None

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): List[
                (B, C=256, H/4, W/4), (B, C=512, H/8, W/8), (B, C=1024, H/16, W/16), (B, C=2048, H/32, W/32)
            ]
        Returns:
            list[torch.Tensor]: List[(B, 512, H/16, W/16), ]
        """
        assert len(x) == len(self.in_channels)
        # List[(B, 128, H/16, W/16), (B, 128, H/16, W/16), ...]
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)     # (B, 512, H/16, W/16)
        else:
            out = ups[0]

        if self.final_conv is not None:
            if self.with_cp:
                out = checkpoint(self.final_conv, out)
            else:
                # 3个残差blocks +ASPP/DCN + Conv(c_mid-->D)
                out = self.final_conv(out)
            # out = self.final_conv(out)

        return [out]
