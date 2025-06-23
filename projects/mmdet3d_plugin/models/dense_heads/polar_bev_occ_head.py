# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
from ..loss.semkitti_loss import sem_scal_loss, geo_scal_loss
from ..loss.lovasz_softmax import lovasz_softmax


nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])


def calculate_bev_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension

@HEADS.register_module()
class Polar_BEVOCCHead2D(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 polar_grid_config=None,
                 cart_grid_config=None,
                 loss_occ=None,
                 ):
        super(Polar_BEVOCCHead2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz

        # voxel-level prediction
        self.occ_convs = nn.ModuleList()
        self.final_conv = ConvModule(
            in_dim,
            self.out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes

        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
        self.loss_occ = build_loss(loss_occ)

        self.polar_grid_config = polar_grid_config
        self.cart_grid_config = cart_grid_config

        polar_bev_resolution, polar_bev_start, polar_bev_dimension = calculate_bev_parameters(
            self.polar_grid_config['azimuth'], self.polar_grid_config['radius'], self.polar_grid_config['z'],
        )

        cart_bev_resolution, cart_bev_start, cart_bev_dimension = calculate_bev_parameters(
            self.cart_grid_config['x'], self.cart_grid_config['y'], self.cart_grid_config['z'],
        )

        self.grid = self.gen_grid(polar_bev_resolution, polar_bev_start, polar_bev_dimension,
                                  cart_bev_resolution, cart_bev_start, cart_bev_dimension)

        self.fp16_enabled = False

    def gen_grid(self, polar_bev_resolution, polar_bev_start, polar_bev_dimension,
                 cart_bev_resolution, cart_bev_start, cart_bev_dimension):

        cart_W, cart_H = cart_bev_dimension[:2]
        x = torch.linspace(0, cart_W - 1, cart_W).view(1, cart_W).expand(cart_H, cart_W)  # (Dy, Dx)
        y = torch.linspace(0, cart_H - 1, cart_H).view(cart_H, 1).expand(cart_H, cart_W)  # (Dy, Dx)
        x = cart_bev_start[0] + x * cart_bev_resolution[0]
        y = cart_bev_start[1] + y * cart_bev_resolution[1]

        azimuth = torch.atan2(y, x)
        dis = torch.sqrt(x ** 2 + y ** 2)
        azimuth = (azimuth - polar_bev_start[0]) / polar_bev_resolution[0]
        dis = (dis - polar_bev_start[1]) / polar_bev_resolution[1]
        grid = torch.stack([azimuth, dis], dim=-1)      # (Dy, Dx, 2)

        polar_W, polar_H = polar_bev_dimension[:2]
        normalize_factor = torch.tensor([polar_W - 1.0, polar_H - 1.0])  # (2, )
        grid = grid / normalize_factor.view(1, 1, 2) * 2.0 - 1.0

        return grid

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, D_r, D_a)
            img_feats: [(B, C, 100, 100), (B, C, 50, 50), (B, C, 25, 25)]   if ms
        Returns:

        """
        B = img_feats.shape[0]
        grid = self.grid.to(img_feats)
        grid = grid.unsqueeze(dim=0).repeat(B, 1, 1, 1)  # (B, Dy, Dx, 2)
        img_feats = F.grid_sample(img_feats.float(), grid, mode='nearest', align_corners=True)

        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    @force_fp32(apply_to=('occ_pred'))
    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()    # (B, Dx, Dy, Dz)
        preds = occ_pred.permute(0, 4, 1, 2, 3).contiguous()    # (B, n_cls, Dx, Dy, Dz)
        loss_occ = self.loss_occ(
            preds,
            voxel_semantics,
            weight=self.cls_weights.to(preds),
        ) * 100.0
        loss['loss_occ'] = loss_occ
        loss['loss_voxel_sem_scal'] = sem_scal_loss(preds, voxel_semantics)
        loss['loss_voxel_geo_scal'] = geo_scal_loss(preds, voxel_semantics, non_empty_idx=17)
        loss['loss_voxel_lovasz'] = lovasz_softmax(torch.softmax(preds, dim=1), voxel_semantics)

        return loss

    @force_fp32(apply_to=('occ_pred'))
    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)

    @force_fp32(apply_to=('occ_pred'))
    def get_occ_gpu(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1).int()      # (B, Dx, Dy, Dz)
        return list(occ_res)