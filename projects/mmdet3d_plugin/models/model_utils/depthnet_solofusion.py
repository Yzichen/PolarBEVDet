import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn import build_conv_layer, ConvModule
from torch.utils.checkpoint import checkpoint
from scipy.ndimage import gaussian_filter1d
from torch.nn.utils.rnn import pad_sequence
from mmcv.runner import auto_fp16, force_fp32
from mmdet3d.models import builder


def interp_zeroends(x, xp, fp):
    """
    For convenience, assumes the sampling dimension is 0.
    This also fills in the ends with 0

    Args:
        x: (D=112, 1, 1, 1)
        xp: (K, B*N, fH, fW),  sample_depth_idxs
        fp: (K, B*N, fH, fW),  stereo_depth_digit

    Returns:
        the interpolated values, same size as `x`.
    """
    assert len(x.shape) == len(xp.shape)
    assert xp.shape == fp.shape

    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    m = torch.cat([m.new_zeros((1, *m.shape[1:])), m, m.new_zeros((1, *m.shape[1:]))], dim=0)
    b = torch.cat([b.new_zeros((1, *b.shape[1:])), b, b.new_zeros((1, *b.shape[1:]))], dim=0)

    indicies = torch.sum(torch.ge(x.unsqueeze(1), xp.unsqueeze(0)), dim=1).long()

    res = torch.gather(m, dim=0, index=indicies) * x + torch.gather(b, dim=0, index=indicies)
    res.scatter_(dim=0, index=xp[[-1]].long(), src=fp[[-1]])

    return res


class SELikeModule(nn.Module):
    def __init__(self, in_channel=512, feat_channel=256, intrinsic_channel=33):
        super(SELikeModule, self).__init__()
        self.input_conv = nn.Conv2d(in_channel, feat_channel, kernel_size=1, padding=0)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(intrinsic_channel),
            nn.Linear(intrinsic_channel, feat_channel),
            nn.Sigmoid()
        )

    def forward(self, x, cam_params):
        """
        :param x: (B*N, C=512, fH, fW)
        :param cam_params: (B, N_views, 15)
        :return:
        """
        x = self.input_conv(x)      # (B*N, C=256, fH, fW)
        B, C, _, _ = x.shape
        y = self.fc(cam_params).view(B, C, 1, 1)
        return x * y.expand_as(x)


class DepthNetSOLOFusion(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 with_cp=False,
                 extra_depth_net=dict(
                     type='ResNetForBEVDet',
                     numC_input=256,
                     num_layer=[3, ],
                     num_channels=[256, ],
                     stride=[1, ]),
                 stereo=False,
                 depth_bound=None,
                 stereo_sampling_num=7,
                 stereo_group_num=8,
                 stereo_gauss_bin_stdev=2,
                 stereo_eps=1e-5,
                 cam_channel=33,
                 **kwargs
                 ):
        super(DepthNetSOLOFusion, self).__init__()
        self.extra_depthnet = builder.build_backbone(extra_depth_net)
        self.featnet = nn.Conv2d(in_channels, context_channels, kernel_size=1, padding=0)
        self.depthnet = nn.Conv2d(mid_channels, depth_channels, kernel_size=1, padding=0)
        self.dcn = nn.Sequential(*[build_conv_layer(dict(type='DCNv2', deform_groups=1),
                                                    in_channels=mid_channels,
                                                    out_channels=mid_channels,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    dilation=1,
                                                    bias=False),
                                   nn.BatchNorm2d(mid_channels)
                                   ])
        self.se = SELikeModule(in_channels, feat_channel=mid_channels,
                               intrinsic_channel=cam_channel)

        self.with_cp = with_cp
        self.D = depth_channels

        self.fp16_enabled = False
        if stereo:
            self.depth_bound = depth_bound
            self.stereo_sampling_num = stereo_sampling_num
            self.stereo_group_num = stereo_group_num
            self.stereo_gauss_bin_stdev = stereo_gauss_bin_stdev
            self.stereo_eps = stereo_eps

            self.similarity_net = nn.Sequential(
                ConvModule(in_channels=self.stereo_group_num,
                           out_channels=16,
                           kernel_size=1,
                           stride=(1, 1, 1),
                           padding=0,
                           conv_cfg=dict(type='Conv3d'),
                           norm_cfg=dict(type='BN3d'),
                           act_cfg=dict(type='ReLU', inplace=True)),
                ConvModule(in_channels=16,
                           out_channels=8,
                           kernel_size=1,
                           stride=(1, 1, 1),
                           padding=0,
                           conv_cfg=dict(type='Conv3d'),
                           norm_cfg=dict(type='BN3d'),
                           act_cfg=dict(type='ReLU', inplace=True)),
                nn.Conv3d(in_channels=8,
                          out_channels=1,
                          kernel_size=1,
                          stride=1,
                          padding=0)
            )
            # Setup gaussian sampling
            gaussians = torch.from_numpy(gaussian_filter1d(F.one_hot(torch.arange(self.D)).float().numpy(),
                                                           stereo_gauss_bin_stdev, mode='constant', cval=0))
            gaussians = gaussians / gaussians.max()
            inv_gaussians = 1 - gaussians
            log_inv_gaussians = torch.log(inv_gaussians + self.stereo_eps)
            log_inv_gaussians[torch.arange(len(log_inv_gaussians)), torch.arange(len(log_inv_gaussians))] = -1000
            self.log_inv_gaussians = nn.Parameter(log_inv_gaussians, requires_grad=False)   # (D=112, D=112)
            self.bin_centers = nn.Parameter(self.get_bin_centers(), requires_grad=False)    # (D=112, )

    def get_bin_centers(self):
        depth_bins = torch.arange(self.depth_bound[0],
                                  self.depth_bound[1],
                                  self.depth_bound[2])    # (D, )
        depth_bins = depth_bins + self.depth_bound[2] / 2
        assert len(depth_bins) == self.D
        return depth_bins

    @force_fp32(apply_to=('sample_depths', 'prev_global2img', 'curr_global2img', 'prev_img_aug', 'curr_img_aug'))
    def gen_grid(self, sample_depths, prev_global2img, curr_global2img, prev_img_aug, curr_img_aug,
                 B, N, stereo_H, stereo_W, imgH, imgW):
        """
        Args:
            sample_depths: (B, N, K=7, stereo_H, stereo_W)
            prev_global2img: (B, N_prev, 4, 4)
            curr_global2img: (B, N_curr, 4, 4)
            prev_img_aug: (B, N_prev, 4, 4)
            curr_img_aug: (B, N_curr, 4, 4)
            B: batchsize
            N: N_views
            stereo_H: fH_stereo
            stereo_W: fW_stereo
            imgH: H_img
            imgW: W_img
        Returns:
            grid: (B, N_prev, N_curr, K, stereo_H, stereo_W, 2)
            valid_mask: (B, N_prev, N_curr, K, stereo_H, stereo_W)
        """
        K = self.stereo_sampling_num
        stereo_downsample = imgH // stereo_H
        grid_y, grid_x = torch.meshgrid(torch.arange(stereo_H), torch.arange(stereo_W))
        meshgrid = torch.stack([grid_x, grid_y], dim=-1)  # (stereo_H, stereo_W, 2)
        meshgrid = (meshgrid * stereo_downsample + stereo_downsample / 2.0).to(curr_global2img)

        # (B, N, K, stereo_H, stereo_W, 4)  4: (u, v, d, 1)
        meshgrid_uvd1 = torch.cat([
            meshgrid[None, None, None, :, :, :].repeat(B, N, K, 1, 1, 1),
            sample_depths.unsqueeze(dim=-1),
            sample_depths.new_ones((B, N, K, stereo_H, stereo_W, 1))
        ], dim=5)
        curr_points = meshgrid_uvd1.unsqueeze(dim=-1)  # (B, N, K, stereo_H, stereo_W, 4, 1)  4: (u, v, d, 1)
        # (B, N, 1, 1, 1, 4, 4) @ (B, N, K, stereo_H, stereo_W, 4, 1)
        # --> (B, N, K, stereo_H, stereo_W, 4, 1)       4: (u, v, d, 1)
        curr_points = torch.inverse(curr_img_aug)[:, :, None, None, None, :, :] @ curr_points

        # (u, v, d) --> (du, dv, d)
        curr_points[..., :2, 0] *= curr_points[..., [2], 0]
        # (B, N, 1, 1, 1, 4, 4) @ (B, N, K, stereo_H, stereo_W, 4, 1)
        # --> (B, N, K, stereo_H, stereo_W, 4, 1)
        curr_global_points = torch.inverse(curr_global2img)[:, :, None, None, None, :, :] @ curr_points

        # (B, N_prev, 1, 1, 1, 1, 4, 4) @ (B, 1, N_curr, K, stereo_H, stereo_W, 4, 1)
        # --> (B, N_prev, N_curr, K, stereo_H, stereo_W, 4, 1)      4: (du, dv, d, 1)
        prev_points = prev_global2img[:, :, None, None, None, None, :, :] @ curr_global_points.unsqueeze(dim=1)
        # (B, N_prev, N_curr, K, stereo_H, stereo_W, 4, 1)      4: (u, v, d, 1)
        prev_points[..., :2, 0] /= torch.maximum(prev_points[..., [2], 0],
                                                 torch.ones_like(prev_points[..., [2], 0]) * 1e-5)

        # (B, N, 1, 1, 1, 1, 4, 4) @ (B, N_prev, N_curr, K, stereo_H, stereo_W, 4, 1)
        # --> (B, N_prev, N_curr, K, stereo_H, stereo_W, 4, 1)       4: (u, v, d, 1)
        prev_points = prev_img_aug[:, :, None, None, None, None, :, :] @ prev_points

        points = prev_points.squeeze(dim=-1)  # (B, N_prev, N_curr, K, stereo_H, stereo_W, 4)
        valid_mask = points[..., 2] > 1e-5
        px = points[..., 0] / imgW * 2.0 - 1.0
        py = points[..., 1] / imgH * 2.0 - 1.0
        grid = torch.stack([px, py], dim=-1)    # (B, N_prev, N_curr, K, stereo_H, stereo_W, 2)

        # (B, N_prev, N_curr, K, stereo_H, stereo_W)
        valid_mask = valid_mask & (px > -1.0) & (px < 1.0) & (py > -1.0) & (py < 1.0)
        return grid, valid_mask

    @auto_fp16(apply_to=('prev_stereo_feats', 'curr_stereo_feats'))
    def calculate_cost_volumn(self, prev_stereo_feats, curr_stereo_feats, grids, valid_mask):
        """
        Args:
            prev_stereo_feats: (B, N_views, C_stereo, stereo_H, stereo_W)
            curr_stereo_feats: (B, N_views, C_stereo, stereo_H, stereo_W)
            grids: (B, N_prev, N_curr, K, stereo_H, stereo_W, 2)
            valid_mask: (B, N_prev, N_curr, K, stereo_H, stereo_W)
        Returns:
            cost_volumn: (B, N, N_group, K, stereo_H, stereo_W)
        """
        B, N, C, stereo_H, stereo_W = prev_stereo_feats.shape
        group_size = (C // self.stereo_group_num)
        cost_volume = curr_stereo_feats.new_zeros(B, N, self.stereo_group_num, self.stereo_sampling_num, stereo_H,
                                                  stereo_W)
        grids = grids.to(prev_stereo_feats)

        for i in range(N):
            cur_prev_feat = prev_stereo_feats[:, i, ...]  # (B, C, stereo_H, stereo_W)
            cur_grid = grids[:, i, ...]  # (B, N_curr, K, stereo_H, stereo_W, 2)  2: (u, v)
            cur_valid_mask = valid_mask[:, i, ...]  # (B, N_curr, K, stereo_H, stereo_W)

            # Then, want to only get features from curr stereo for valid locations, so need to prepare for padding
            # and unpadding that.
            # Tuple((N_valid, ), (N_valid, ), ...)   (batch_id, N_id, K_id, H_id, W_id)
            cur_valid_mask_where = torch.where(cur_valid_mask)

            # List[Tuple((N_valid, ), (N_valid, ), ...), Tuple(...), ...]    len=bs, (N_id, K_id, H_id, W_id)
            cur_valid_mask_where_list = [
                [dim_where[cur_valid_mask_where[0] == batch_idx] for dim_where in cur_valid_mask_where[1:]] for
                batch_idx in range(B)]
            cur_valid_mask_num_list = [
                len(tmp[0]) for tmp in cur_valid_mask_where_list]  # List[N_valid0, N_valid1, ...]
            cur_valid_mask_padded_valid_mask = torch.stack(
                [torch.arange(max(cur_valid_mask_num_list), device=cur_prev_feat.device) < tmp_len for
                 tmp_len in cur_valid_mask_num_list], dim=0)  # (B, max_len), Fasle表示pad.

            # Now get the sampled features in padded form
            # List[(N_valid0, 2), (N_valid1, 2), ...], len = batch_size
            cur_valid_grid_list = [
                tmp[tmp_mask, :] for tmp, tmp_mask in zip(cur_grid, cur_valid_mask)]
            # (B, max_len, 2)
            cur_valid_grid_padded = pad_sequence(cur_valid_grid_list, batch_first=True)
            # (B, C, max_len, 1)
            cur_prev_sampled_feats_padded = F.grid_sample(cur_prev_feat, cur_valid_grid_padded.unsqueeze(dim=2))
            # (B, C, max_len, 1) --> (B, C, max_len) --> (B, max_len, C)
            cur_prev_sampled_feats_padded = cur_prev_sampled_feats_padded.squeeze(dim=-1).permute(0, 2, 1)

            # Get the corresponding curr features. Doing this to avoid the max-size tensor B x N x C x 118 x stereo_H x stereo_W.
            # Biggest tensor we have is B x max_length x C, which should be around B x C x 118 x stereo_H x stereo_W, so without the N factor.
            with torch.set_grad_enabled(curr_stereo_feats.requires_grad):
                # List[(N_valid0, C), (N_valid1, C), ...], len = batch_size
                cur_curr_stereo_feats_valid_list = [
                    tmp[tmp_where[0], :, tmp_where[2], tmp_where[3]] for tmp, tmp_where in
                    zip(curr_stereo_feats, cur_valid_mask_where_list)]
                # (B, max_len, C)
                cur_curr_stereo_feats_valid_padded = pad_sequence(cur_curr_stereo_feats_valid_list, batch_first=True)

                # Compute the group correlation
                # (B, max_len, C) * (B, max_len, C) --> (B, max_len, C)
                curr_cost_volume = cur_prev_sampled_feats_padded * cur_curr_stereo_feats_valid_padded
                # (B, max_len, C) --> (B, max_len, N_group, C')
                curr_cost_volume = curr_cost_volume.view(B, curr_cost_volume.shape[1], self.stereo_group_num,
                                                         group_size)
                # (B, max_len, N_group, C') --> (B, max_len, N_group)
                curr_cost_volume = curr_cost_volume.sum(dim=3)

                # Now fill in cost_volume. Add it incrementally for now, will average later. Dot product is commutative
                # with average.
                cost_volume[cur_valid_mask_where[0], cur_valid_mask_where[1], :, cur_valid_mask_where[2],
                            cur_valid_mask_where[3], cur_valid_mask_where[4]] += \
                    curr_cost_volume[cur_valid_mask_padded_valid_mask]

                del curr_cost_volume, cur_prev_sampled_feats_padded, cur_curr_stereo_feats_valid_padded

        with torch.set_grad_enabled(curr_stereo_feats.requires_grad):
            # Some points are projected to multiple prev cameras; average over those.
            num_valid_per_point = valid_mask.float().sum(dim=1)  # (B, N_curr, K, stereo_H, stereo_W)
            num_valid_per_point = num_valid_per_point.unsqueeze(2)  # (B, N_curr, 1, K, stereo_H, stereo_W)
            # (B, N_curr, N_group, K, stereo_H, stereo_W)
            cost_volume = cost_volume / torch.maximum(num_valid_per_point, torch.ones_like(num_valid_per_point))

        assert curr_stereo_feats.requires_grad == cost_volume.requires_grad
        return cost_volume

    @auto_fp16(apply_to=('mono_depth_digit', 'prev_stereo_feat', 'curr_stereo_feat'))
    def stereo_depth(self, mono_depth_digit, prev_stereo_feat, curr_stereo_feat, stereo_metas):
        """
        :param mono_depth_digit: (B*N_views, D, fH, fW)
        :param prev_stereo_feat: (B, N, C_stereo, fH_stereo, fW_stereo)
        :param curr_stereo_feat: (B, N, C_stereo, fH_stereo, fW_stereo)
        :param stereo_metas:  None or dict{
                    cv_downsample: 4,  # stereo feature 对应的downsample
                    downsample: 16 # feature map for LSS 对应的downsample
                    grid_config: dict,
                    global2img: [prev_global2img, curr_global2img],  # (B, N, 4, 4)
                    img_aug=[prev_img_aug, curr_img_aug],    # (B, N, 4, 4)
                    cv_feat_list=[prev_stereo_feat, stereo_feat]        # 上一帧和当前帧对应的stereo feature.
                }
        :return:
            depth_digit: (B*N, D, fH, fW)
        """
        cv_downsample = stereo_metas['cv_downsample']
        downsample = stereo_metas['downsample']
        prev_global2img, curr_global2img = stereo_metas['global2img']
        prev_img_aug, curr_img_aug = stereo_metas['img_aug']

        fH, fW = mono_depth_digit.shape[-2:]
        B, N, C, stereo_H, stereo_W = curr_stereo_feat.shape
        imgH, imgW = stereo_H * cv_downsample, stereo_W * cv_downsample    # H_img, W_img

        with torch.no_grad():
            # Stereo Sampling
            gauss_sample_distr_log = mono_depth_digit.log_softmax(dim=1)  # (B*N_views, D=112, fH, fW)
            gauss_sample_depth_idxs = []
            for _ in range(self.stereo_sampling_num):
                curr_gauss_sample_depth_idxs = gauss_sample_distr_log.argmax(dim=1)  # (B*N_views, fH, fW)
                # (B*N_views, fH, fW, D=112) --> (B*N_views, D=112, fH, fW)
                uncertainty_reduction = self.log_inv_gaussians[curr_gauss_sample_depth_idxs].permute(0, 3, 1, 2)
                gauss_sample_distr_log = gauss_sample_distr_log + uncertainty_reduction  # 惩罚当前深度采样附近的score.
                gauss_sample_depth_idxs.append(curr_gauss_sample_depth_idxs)

            gauss_sample_depth_idxs = torch.stack(gauss_sample_depth_idxs, dim=1)  # (B*N_views, K=7, fH, fW)
            gauss_sample_depth_idxs = gauss_sample_depth_idxs.sort(dim=1).values   # (B*N_views, K=7, fH, fW)
            gauss_sample_depths = self.bin_centers[gauss_sample_depth_idxs]  # (B*N_views, K=7, fH, fW)

            # Now we have depth idxs and their depths. upsample it (via repeat) up to stereo_H & stereo_W.
            # (B*N_views, K=7, fH, fW)
            sample_depth_idxs = gauss_sample_depth_idxs.view(B * N, self.stereo_sampling_num, fH, fW)

            # (B*N_views, K=7, fH, fW) --> (B*N_views, K=7, stereo_H, stereo_W)
            # --> (B, N_views, K=7, stereo_H, stereo_W)
            sample_depths = F.interpolate(
                gauss_sample_depths, scale_factor=(downsample // cv_downsample), mode='nearest').\
                view(B, N, self.stereo_sampling_num, stereo_H, stereo_W)

            # get sampling points
            # grid: (B, N_prev, N_curr, K, stereo_H, stereo_W, 2)
            # valid_mask: (B, N_prev, N_curr, K, stereo_H, stereo_W)
            grid, valid_mask = self.gen_grid(sample_depths, prev_global2img, curr_global2img, prev_img_aug,
                                             curr_img_aug, B, N, stereo_H, stereo_W, imgH, imgW)

            cost_volume = self.calculate_cost_volumn(prev_stereo_feat, curr_stereo_feat, grid, valid_mask)

        # Get the cost volume logits
        # (B, N_curr, N_group, K, stereo_H, stereo_W) --> (B*N, N_group=8, K, stereo_H, stereo_W)
        # --> (B*N, 16, K, stereo_H, stereo_W) --> (B*N, 8, K, stereo_H, stereo_W)
        # --> (B*N, 1, K, stereo_H, stereo_W)
        cost_volume = self.similarity_net(
            cost_volume.view(B * N, self.stereo_group_num, self.stereo_sampling_num, stereo_H, stereo_W))
        stereo_depth_digit = cost_volume.squeeze(1)  # (B*N, K, stereo_H, stereo_W)

        # (B*N, K, stereo_H, stereo_W) --> (B*N, K, f_H, f_W)
        stereo_depth_digit = F.avg_pool2d(
            stereo_depth_digit.view(B * N, self.stereo_sampling_num, stereo_H, stereo_W),
            downsample // cv_downsample, downsample // cv_downsample)

        # (D, B*N, fH, fW) --> (B*N, D, fH, fW)
        stereo_depth_digit_interp = interp_zeroends(
            torch.arange(self.D).to(sample_depth_idxs.device)[:, None, None, None],  # (D=112, 1, 1, 1)
            sample_depth_idxs.permute(1, 0, 2, 3),  # (K, B*N, fH, fW)
            stereo_depth_digit.permute(1, 0, 2, 3)  # (K, B*N, fH, fW)
        ).permute(1, 0, 2, 3)

        depth_digit = mono_depth_digit + stereo_depth_digit_interp  # (B*N, D, fH, fW)
        return depth_digit

    @auto_fp16(apply_to=('x', ))
    def forward(self, x, mlp_input, stereo_metas=None):
        """
        Args:
            x: (B*N_views, C, fH, fW)
            mlp_input: (B, N_views, 27)
            stereo_metas:  None or dict{
                k2s_sensor: (B, N_views, 4, 4)    表示当前帧到上一帧相机坐标系的变换.
                intrins: (B, N_views, 3, 3)    相机内参
                post_rots: (B, N_views, 3, 3)    图像增广旋转矩阵
                post_trans: (B, N_views, 3)       图像增广平移矩阵
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,    # stereo feature 对应的downsample
                downsample: self.img_view_transformer.downsample=16,   # feature map for LSS 对应的downsample
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]   # 上一帧和当前帧对应的stereo feature.
            }
        Returns:
            x: (B*N_views, D+C_context, fH, fW)
        """
        mlp_input = mlp_input.reshape(-1, mlp_input.shape[-1])   # (B*N_views, 15)
        context = self.featnet(x)

        depth_feat = x
        depth_feat = self.se(depth_feat, mlp_input)
        depth_feat = self.extra_depthnet(depth_feat)[0]
        depth_feat = self.dcn(depth_feat)
        depth = self.depthnet(depth_feat)

        if stereo_metas is not None:
            prev_stereo_feat, curr_stereo_feat = stereo_metas['cv_feat_list']
            depth = self.stereo_depth(depth, prev_stereo_feat, curr_stereo_feat,
                                      stereo_metas)  # (B*N_views, D, fH_stereo, fW_stereo)
        return torch.cat([depth, context], dim=1)