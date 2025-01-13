# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.misc import memory_refresh

from mmdet3d.models import DETECTORS
from .bevdet import BEVDet
from mmdet3d.models import builder
from ..utils.misc import locations


@DETECTORS.register_module()
class PolarBEVDet(BEVDet):
    def __init__(self,
                 stereo_neck=None,
                 pre_process=None,
                 do_history=False,
                 do_history_stereo_fusion=False,
                 history_cat_num=1,
                 history_cat_conv_out_channels=None,    # Number of history key frames to cat
                 aux_img_head=None,
                 seg_head=None,
                 stride=16,
                 with_cp=False,
                 **kwargs):
        super(PolarBEVDet, self).__init__(**kwargs)
        # Prior to history fusion, do some per-sample pre-processing.
        self.single_bev_num_channels = self.img_view_transformer.out_channels
        self.grid = None
        self.prev_scene_token = None
        self.with_cp = with_cp

        # Lightweight MLP
        self.embed = nn.Sequential(
            nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True))

        # Preprocessing like BEVDet4D
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)

        # Deal with history
        self.do_history = do_history
        self.history_cat_num = history_cat_num
        self.history_cam_sweep_freq = 0.5   # seconds between each frame
        history_cat_conv_out_channels = history_cat_conv_out_channels if history_cat_conv_out_channels is not None \
            else self.single_bev_num_channels

        # Embed each sample with its relative temporal offset with current timestep
        self.history_keyframe_time_conv = nn.Sequential(
            nn.Conv2d(self.single_bev_num_channels + 1,
                    self.single_bev_num_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True))

        # Then concatenate and send them through an MLP.
        self.history_keyframe_cat_conv = nn.Sequential(
            nn.Conv2d(self.single_bev_num_channels * (self.history_cat_num + 1),
                    history_cat_conv_out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1),
            nn.BatchNorm2d(history_cat_conv_out_channels),
            nn.ReLU(inplace=True))

        self.reset_memory()

        self.do_history_stereo_fusion = do_history_stereo_fusion
        if self.do_history_stereo_fusion:
            self.reset_stereo_memory()
            self.stereo_neck = builder.build_neck(stereo_neck)

        if aux_img_head is not None:
            self.aux_img_head = builder.build_head(aux_img_head)
        else:
            self.aux_img_head = None
        self.stride = stride

        if seg_head is not None:
            self.seg_head = builder.build_head(seg_head)
        else:
            self.seg_head = None

    def reset_memory(self):
        self.history_scene_tokens = None
        self.history_bev = None
        self.history_sweep_time = None
        self.history_seq_ids = None
        self.history_egopose = None     # ego --> global

    def pre_update_memory(self, curr_bev, seq_ids, ego2globals, prev_exists):
        """
        :param curr_bev: (B, C=80, Dy, Dx)
        :param seq_ids:  (B, )
        :param ego2globals: (B, 4, 4)
        :param prev_exists: (B, )
        :return:
        """
        B, C, bev_h, bev_w = curr_bev.shape
        if self.history_bev is None:
            self.history_bev = curr_bev.new_zeros(B, self.history_cat_num, C, bev_h, bev_w)
            self.history_sweep_time = curr_bev.new_zeros(B, self.history_cat_num)
            self.history_seq_ids = curr_bev.new_zeros(B, self.history_cat_num)
            self.history_egopose = curr_bev.new_zeros((B, 4, 4), dtype=torch.float32)      # global --> ego
        else:
            self.history_sweep_time += 1
            # curr_ego --> global --> prev_ego
            history_egopose = self.history_egopose.double() @ ego2globals.double()
            self.history_egopose = history_egopose.float()

            self.history_bev = memory_refresh(self.history_bev[:, :self.history_cat_num], prev_exists)
            self.history_sweep_time = memory_refresh(self.history_sweep_time[:, :self.history_cat_num], prev_exists)
            self.history_seq_ids = memory_refresh(self.history_seq_ids[:, :self.history_cat_num], prev_exists)
            self.history_egopose = memory_refresh(self.history_egopose, prev_exists)

        prev_exists = prev_exists.float()
        self.history_bev = self.history_bev + (1-prev_exists).view(B, 1, 1, 1, 1) * curr_bev.unsqueeze(dim=1).\
            repeat(1, self.history_cat_num, 1, 1, 1)    # (B, N_his, C, bev_h, bev_w)
        self.history_seq_ids = self.history_seq_ids + (1-prev_exists).view(B, 1) * seq_ids.unsqueeze(dim=-1).\
            repeat(1, self.history_cat_num)     # (B, N_his)
        self.history_egopose = self.history_egopose + (1-prev_exists).view(B, 1, 1) * torch.eye(4, device=curr_bev.device)

    def post_update_memory(self, feats_cat, seq_ids, ego2globals):
        """
        :param feats_cat: (B, (1+N_his)*C, Dy, Dx)
        :param seq_ids:  (B, )
        :param ego2globals: (B, 4, 4)
        :return:
        """
        self.history_bev = feats_cat.view(feats_cat.shape[0], 1+self.history_cat_num, -1,
                                          feats_cat.shape[2], feats_cat.shape[3])   # (B, (1+N_his), C=80, Dy, Dx)
        self.history_seq_ids = torch.cat([seq_ids.unsqueeze(dim=1), self.history_seq_ids], dim=1)   # (B, (1+N_his))
        self.history_egopose = torch.inverse(ego2globals)  # global --> ego

    def reset_stereo_memory(self):
        self.history_stereo_feat = None
        self.history_global2img = None
        self.history_img_aug = None

    def pre_update_stereo_memory(self, curr_stereo_feat, curr_global2img, curr_img_aug, prev_exists):
        """
        :param curr_stereo_feat: (B, N, C_stereo, fH_stereo, fW_stereo)
        :param curr_global2img: (B, N, 4, 4)
        :param curr_img_aug: (B, N, 4, 4)
        :param prev_exists: (B, )
        :return:
        """
        B, N, C, stereo_H, stereo_W = curr_stereo_feat.shape

        if self.history_stereo_feat is None:
            self.history_stereo_feat = curr_stereo_feat.new_zeros(B, N, C, stereo_H, stereo_W)
            self.history_global2img = curr_global2img.new_zeros((B, N, 4, 4), dtype=torch.float32)    # global --> img
            self.history_img_aug = curr_img_aug.new_zeros((B, N, 4, 4), dtype=torch.float32)
        else:
            self.history_stereo_feat = memory_refresh(self.history_stereo_feat, prev_exists)
            self.history_global2img = memory_refresh(self.history_global2img, prev_exists)
            self.history_img_aug = memory_refresh(self.history_img_aug, prev_exists)

        prev_exists = prev_exists.float()
        self.history_stereo_feat = self.history_stereo_feat + (1-prev_exists).view(B, 1, 1, 1, 1) * curr_stereo_feat
        self.history_stereo_feat = self.history_stereo_feat.to(curr_stereo_feat)

        # (B, N, 4, 4)
        self.history_global2img = self.history_global2img + (1-prev_exists).view(B, 1, 1, 1) * curr_global2img
        self.history_img_aug = self.history_img_aug + (1-prev_exists).view(B, 1, 1, 1) * curr_img_aug

    def post_update_stereo_memory(self, stereo_feat, global2img, img_aug):
        """
        :param stereo_feat: (B, N, C, stereo_H, stereo_W)
        :param global2img: (B, N, 4, 4)
        :return:
        """
        self.history_stereo_feat = stereo_feat.detach().clone()
        self.history_global2img = global2img
        self.history_img_aug = img_aug

    @auto_fp16()
    def image_encoder(self, imgs, stereo=False):
        """
        Args:
            img: (B, N, 3, H, W)
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B, N, C_stereo, fH_stereo, fW_stereo) / None
        """
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        backbone_feats = self.img_backbone(imgs)

        if isinstance(backbone_feats, dict):
            backbone_feats = list(backbone_feats.values())

        if self.with_img_neck:
            neck_feats = self.img_neck(backbone_feats)
            if type(neck_feats) in [list, tuple]:
                neck_feats = neck_feats[0]      # (B*N_views, C=512, fH=H/16, fW=W/16)
        _, output_dim, ouput_H, output_W = neck_feats.shape
        neck_feats = neck_feats.view(B, N, output_dim, ouput_H, output_W)

        stereo_feats = None
        if stereo:
            backbone_feats_detached = [tmp.detach() for tmp in backbone_feats]
            stereo_feats = self.stereo_neck(backbone_feats_detached)
            if type(stereo_feats) in [list, tuple]:
                stereo_feats = stereo_feats[0]      # (B*N_views, C=256, fH=H/4, fW=W/4)
            stereo_feats = F.normalize(stereo_feats, dim=1, eps=1e-5)
            _, output_dim, ouput_H, output_W = stereo_feats.shape
            stereo_feats = stereo_feats.view(B, N, output_dim, ouput_H, output_W)

        return neck_feats, stereo_feats

    def gen_grid(self, input, keyego2adjego):
        """
        Args:
            input: (B, C, Dy, Dx)  bev_feat
            keyego2adjego: (B, 4, 4)
        Returns:
            grid: (B, Dy, Dx, 2),  介于(-1, 1)
        """
        def polar2cart(polar_points):
            """
            :param polar_points: (B, Dy, Dx, 3, 1)  3: (r, d, z)
            :return:
                cart_points: (B, Dy, Dx, 3, 1)  3: (x, y, z)
            """
            azimuth, dis, z = torch.split(polar_points, split_size_or_sections=1, dim=3)
            x = dis * torch.cos(azimuth)
            y = dis * torch.sin(azimuth)
            cart_points = torch.cat([x, y, z], dim=3)
            return cart_points

        def cart2polar(cart_points):
            """
            :param cart_points: (B, Dy, Dx, 3, 1)  3: (r, d, z)
            :return:
                polar_points: (B, Dy, Dx, 3, 1)  3: (x, y, z)
            """
            x, y, z = torch.split(cart_points, split_size_or_sections=1, dim=3)
            azimuth = torch.atan2(y, x)
            dis = torch.sqrt(x ** 2 + y ** 2)
            polar_points = torch.cat([azimuth, dis, z], dim=3)
            return polar_points

        B, C, H, W = input.shape
        if self.grid is None:
            # generate grid
            xs = torch.linspace(
                0, W - 1, W, dtype=input.dtype,
                device=input.device).view(1, W).expand(H, W)    # (Dy, Dx)
            ys = torch.linspace(
                0, H - 1, H, dtype=input.dtype,
                device=input.device).view(H, 1).expand(H, W)    # (Dy, Dx)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)   # (Dy, Dx, 3)   3: (x, y, 1)
            self.grid = grid
        else:
            grid = self.grid
        # (Dy, Dx, 3)  --> (1, Dy, Dx, 3) --> (B, Dy, Dx, 3) --> (B, Dy, Dx, 3, 1))     3: (grid_x, grid_y, 1)
        grid = grid.view(1, H, W, 3).expand(B, H, W, 3).view(B, H, W, 3, 1)
        # (B, 4, 4) --> (B, 1, 1, 4, 4)
        keyego2adjego = keyego2adjego.view(B, 1, 1, 4, 4)

        # (B, 1, 1, 3, 3)
        keyego2adjego = keyego2adjego[..., [True, True, False, True], :][..., [True, True, False, True]]

        # x = grid_x * vx + x_min;  y = grid_y * vy + y_min;
        # feat2bev:
        # [[vx, 0, x_min],
        #  [0, vy, y_min],
        #  [0,  0,   1  ]]
        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid.device)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)       # (1, 3, 3)

        # (B, 1, 1, 3, 3) @ (B, Dy, Dx, 3, 1) --> (B, Dy, Dx, 3, 1)
        polar_points = feat2bev.matmul(grid)
        cart_points = polar2cart(polar_points)
        cart_points_adj = keyego2adjego.matmul(cart_points)
        polar_points_adj = cart2polar(cart_points_adj)
        grid = torch.inverse(feat2bev).matmul(polar_points_adj)
        normalize_factor = torch.tensor([W - 1.0, H - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)  # (2, )
        # (B, Dy, Dx, 2),  介于(-1, 1)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0

        return grid

    @force_fp32()
    def shift_feature(self, input, keyego2adjego):
        """
        :param input: (B, N_his, C=80, bev_h, bev_w)
        :param keyego2adjego: (B, 4, 4)
        :return:
            output:  (B, N_his*C, bev_h, bev_w)
        """
        B, N_his, C, bev_H, bev_W = input.shape
        # (B, N_his, C=80, bev_h, bev_w) --> (B, N_his*C, bev_h, bev_w)
        input = input.view(B, -1, bev_H, bev_W).contiguous()
        grid = self.gen_grid(input, keyego2adjego)   # grid: (B, bev_h, bev_w, 2),  介于(-1, 1)
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)     # (B, N_his*C, bev_h, bev_w)
        return output

    @force_fp32()
    def fuse_history(self, curr_bev, bda, ego2globals, img_metas):
        """
        :param curr_bev: (B, C=80, Dy, Dx)
        :param bda: (B, 3, 3)
        :param ego2globals: (B, N, 4, 4)
        :return:
        """
        B = curr_bev.shape[0]
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx']
            for single_img_metas in img_metas]).to(curr_bev.device)     # (B, )
        prev_exists = torch.BoolTensor([
            single_img_metas['prev_exists']
            for single_img_metas in img_metas]).to(curr_bev.device)     # (B, )

        ego2globals = ego2globals[:, 0, ...]    # (B, 4, 4)

        bda_mat = torch.zeros(B, 4, 4).to(curr_bev)
        bda_mat[:, 3, 3] = 1
        bda_mat[:, :3, :3] = bda
        bda_mat_inv = torch.inverse(bda_mat)    # (B, 4, 4)
        # (B, 4, 4) @ (B, 4, 4) --> (B, 4, 4)
        ego2globals = ego2globals @ bda_mat_inv

        self.pre_update_memory(curr_bev, seq_ids, ego2globals, prev_exists)

        # First, sanity check. For every non-start of sequence, history id and seq id should be same.
        assert (self.history_seq_ids != seq_ids.unsqueeze(dim=-1)).sum() == 0, \
            "{}, {}".format(self.history_seq_ids, seq_ids)

        self.history_bev = self.history_bev.detach()    # (B, N_his, C, bev_h, bev_w)
        keyego2adjego = self.history_egopose    # (B, 4, 4)

        sampled_history_bev = self.shift_feature(self.history_bev, keyego2adjego)   # (B, N_his*C, bev_h, bev_w)

        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1)      # (B, 1+N_history)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1)   # (B, (1+N_his)*C, bev_H, bev_W)

        # Reshape and concatenate features and timestep
        # (B, (1+N_his)*C, bev_H, bev_W) --> (B, 1+N_his, C=80, bev_H, bev_W)
        feats_to_return = feats_cat.reshape(
            feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, feats_cat.shape[2], feats_cat.shape[3])
        # cat[(B, 1+N_his, C=80, bev_H, bev_W), (B, 1+N_his, 1, bev_H, bev_W)]
        # --> (B, 1+N_his, C=80+1, bev_H, bev_W)   80: feature; 1: time_interval.
        feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2)

        # Time conv
        # (B, 1+N_his, C=80+1, bev_H, bev_W) --> (B*(1+N_his), C=80+1, bev_H, bev_W)
        # --> (B*(1+N_his), C=80, bev_H, bev_W) --> (B, (1+N_his), C=80, bev_H, bev_W)
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:])

        # Cat keyframes & conv
        # (B, (1+N_his), C=80, bev_H, bev_W) --> (B, (1+N_his)*80, bev_H, bev_W)
        # --> (B, C=160, bev_H, bev_W)
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, feats_to_return.shape[3], feats_to_return.shape[4]))

        self.post_update_memory(feats_cat.detach(), seq_ids, ego2globals)

        return feats_to_return.clone()

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)
        global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    @force_fp32(apply_to=('sensor2egos', 'ego2globals', 'intrin', 'post_rot', 'post_tran', 'bda'))
    def prepare_stereo_metas(self, stereo_feat, sensor2egos, ego2globals, intrin,
                             post_rot, post_tran, bda, img_metas):
        """
        :param stereo_feat: (B, N, C_stereo, fH_stereo, fW_stereo)
        :param sensor2egos: (B, N_views, 4, 4)
        :param ego2globals: (B, N_views, 4, 4)
        :param intrin: (B, N_views, 3, 3)
        :param post_rot: (B, N_views, 3, 3)
        :param post_tran: (B, N_views, 3)
        :param bda: (B, 3, 3)
        :param img_metas:
        :return:
        """
        prev_exists = torch.BoolTensor([
            single_img_metas['prev_exists']
            for single_img_metas in img_metas]).to(stereo_feat.device)     # (B, )

        B, N, C, stereo_H, stereo_W = stereo_feat.shape

        # 计算curr的 global --> img
        # global --> key_ego --> camera     # (B, N_views, 4, 4)
        ego2globals = ego2globals[:, 0, ...].view(B, 1, 4, 4).repeat(1, N, 1, 1)
        global2sensor = torch.inverse(sensor2egos.double()) @ torch.inverse(ego2globals.double())
        global2sensor = global2sensor.float()

        intrins4x4 = sensor2egos.new_zeros((B, N, 4, 4))    # (B, N_views, 4, 4)
        intrins4x4[..., 3, 3] = 1.0
        intrins4x4[..., :3, :3] = intrin
        global2img = intrins4x4 @ global2sensor   # (B, N_views, 4, 4)

        img_aug = sensor2egos.new_zeros((B, N, 4, 4))      # (B, N_views, 4, 4)
        img_aug[..., 3, 3] = 1.0
        img_aug[..., :3, :3] = post_rot
        img_aug[..., :3, 3] = post_tran

        # (B*N, C_stereo, fH_stereo, fW_stereo) --> (B, N, C_stereo, fH_stereo, fW_stereo)
        curr_stereo_feat = stereo_feat
        curr_global2img = global2img
        curr_img_aug = img_aug
        self.pre_update_stereo_memory(curr_stereo_feat, curr_global2img, curr_img_aug, prev_exists)
        prev_stereo_feat = self.history_stereo_feat.detach().clone()
        prev_global2img = self.history_global2img.clone()
        prev_img_aug = self.history_img_aug.clone()

        metas = dict(
            cv_downsample=4,  # stereo feature 对应的downsample
            downsample=self.img_view_transformer.downsample,  # feature map for LSS 对应的downsample
            grid_config=self.img_view_transformer.grid_config,
            global2img=[prev_global2img, curr_global2img],  # (B, N, 4, 4)
            img_aug=[prev_img_aug, curr_img_aug],   # (B, N, 4, 4)
            cv_feat_list=[prev_stereo_feat, curr_stereo_feat]        # 上一帧和当前帧对应的stereo feature.
        )

        self.post_update_stereo_memory(curr_stereo_feat, curr_global2img, curr_img_aug)

        return metas

    def prepare_bev_feat(self, img, sensor2egos, ego2globals, intrin, post_rot, post_tran,
                         bda, mlp_input, img_metas):
        """
        Args:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
            mlp_input:
            img_metas:
        Returns:
            x: (B, N_views, C, fH, fW)
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        # x: (B, N_views, C, fH, fW)
        # stereo_feat: (B, N_views, C_stereo, fH_stereo, fW_stereo)
        x, stereo_feat = self.image_encoder(img, stereo=self.do_history_stereo_fusion)

        if self.do_history_stereo_fusion:
            stereo_metas = self.prepare_stereo_metas(
                stereo_feat, sensor2egos, ego2globals, intrin,
                post_rot, post_tran, bda, img_metas
            )
        else:
            stereo_metas = None

        # bev_feat: (B, C * Dz(=1), Dy, Dx)
        # depth: (B * N, D, fH, fW)
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2egos, ego2globals, intrin, post_rot, post_tran, bda, mlp_input], stereo_metas)

        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]    # (B, C, Dy, Dx)

        bev_feat = self.embed(bev_feat)

        return x, bev_feat, depth

    def extract_img_feat(self, img_inputs, img_metas, gt_bboxes_3d=None):
        """
        Args:
            img_inputs: List[
                img: (B, N, 3, H, W),
                rots: (B, N, 3, 3)
                trans: (B, N, 3)
                intrins: (B, N, 3, 3)
                post_rots: (B, N, 3, 3)
                post_trans: (B, N, 3)
                depth_map: (B, N, fH, fW)
            ]
            img_metas:
            gt_bboxes_3d:

        Returns:
            x: (B, C=64*4=256, bev_H, bev_W)
            depth: (B*N_views, D=112, H/16, W/16)
        """
        imgs, sensor2keyegos, ego2globals, intrins, \
        post_rots, post_trans, bda = self.prepare_inputs(img_inputs)

        # 为了支持BEVDepth.
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)  # (B, N_views, 27)

        inputs_curr = [imgs, sensor2keyegos, ego2globals, intrins, post_rots,
                       post_trans, bda, mlp_input]

        # bev_feat: (B, C, Dy, Dx)
        # depth: (B*N_views, D, fH, fW)
        img_feat, bev_feat, depth = self.prepare_bev_feat(*inputs_curr, img_metas)

        # Fuse History
        B = bev_feat.shape[0]
        if not self.do_history:
            for i in range(B):
                img_metas[i]['prev_exists'] = False

        bev_feat = self.fuse_history(bev_feat, bda, ego2globals, img_metas)  # (B, C=160 bev_H=128, bev_W=128)

        if self.seg_head is not None:
            pred_seg = self.seg_head([bev_feat])
            bev_feat = bev_feat + bev_feat * pred_seg
        else:
            pred_seg = None

        x = self.bev_encoder(bev_feat)  # (B, C=64*4=256, bev_H, bev_W)

        return img_feat, [x], depth, pred_seg

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        """
        points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
        img_inputs:
                imgs:  (B, N_views, 3, H, W)        
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        """
        img_feats, bev_feats, depth, pred_seg = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = None
        return img_feats, bev_feats, pts_feats, depth, pred_seg

    def prepare_location(self, img_metas, img_feats):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = img_feats.shape[:2]     # B, N_view
        x = img_feats.flatten(0, 1)     # (B, N_views, C'=256, fH, fW) --> (B*N_views, C'=256, fH, fW)
        # (fh, fw, 2) --> (B*N_view, fh, fw, 2)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

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

        # Get box losses
        losses_pts = self.forward_pts_train(bev_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

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

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(bev_feats, img_metas, rescale=rescale)
        # bbox_pts: List[dict0, dict1, ...],  len = batch_size
        # dict: {
        #   'boxes_3d': (N, 9)
        #   'scores_3d': (N, )
        #   'labels_3d': (N, )
        # }
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


