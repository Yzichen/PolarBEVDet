# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module(force=True)
class PolarCenterPointBBoxCoder(BaseBBoxCoder):
    """Bbox coder for CenterPoint.

    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float], optional): Limit of the center.
            Default: None.
        max_num (int, optional): Max number to be kept. Default: 100.
        score_threshold (float, optional): Threshold to filter boxes
            based on score. Default: None.
        code_size (int, optional): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 code_size=9):

        self.pc_range = pc_range    # [azimuth_min, dis_min, ...]
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range  # [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.code_size = code_size

    def _gather_feat(self, feats, inds, feat_masks=None):
        """Given feats and indexes, returns the gathered feats.

        Args:
            feats (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            inds (torch.Tensor): Indexes with the shape of [B, N].
            feat_masks (torch.Tensor, optional): Mask of the feats.
                Default: None.

        Returns:
            torch.Tensor: Gathered feats.
        """
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores, K=80):
        """Get indexes based on scores.

        Args:
            scores (torch.Tensor): scores with the shape of (B, N_cls, H, W).
            K (int, optional): Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]
                torch.Tensor: Selected scores with the shape of [B, K].
                torch.Tensor: Selected indexes with the shape of [B, K].
                torch.Tensor: Selected classes with the shape of [B, K].
                torch.Tensor: Selected y coord with the shape of [B, K].
                torch.Tensor: Selected x coord with the shape of [B, K].
        """
        batch, cat, height, width = scores.size()

        # 先是针对每个类别的预测都取topK.
        # (B, N_cls, K), (B, N_cls, K)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)    # (B, N_cls, K), topK对应的像素索引(0, H*W-1).
        topk_ys = (topk_inds.float() /
                   torch.tensor(width, dtype=torch.float)).int().float()    # (B, N_cls, K), y坐标.
        topk_xs = (topk_inds % width).int().float()     # (B, N_cls, K), x坐标.

        # 然后对将所有类别得到的topK数据再次进行topK.
        # (B, K), (B, K)
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()      # (B, K)  对应的类别.
        # (B, N_cls*K, 1) --gather--> (B, K, 1) --> (B, K)  topK对应的像素坐标索引(0, H*W-1).
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1),
                                      topk_ind).view(batch, K)
        # (B, N_cls*K, 1) --gather--> (B, K, 1) --> (B, K)  topK对应的y坐标.
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1),
                                    topk_ind).view(batch, K)
        # (B, N_cls*K, 1) --gather--> (B, K, 1) --> (B, K)  topK对应的x坐标.
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1),
                                    topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat, ind):
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat (torch.Tensor): Features to be transposed and gathered
                with the shape of (B, N_c, H, W).
            ind (torch.Tensor): Indexes with the shape of [B, K].

        Returns:
            torch.Tensor: Transposed and gathered feats.
        """
        # (B, N_c, H, W) --> (B, H, W, N_c) --> (B, H*W, N_c)
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)     # (B, K, N_c)
        return feat

    def encode(self):
        pass

    def limit_period(self, val, offset=0.5, period=torch.pi):
        limited_val = val - torch.floor(val / period + offset) * period
        return limited_val

    def polar2cart(self, polar_boxes):
        """
        Args:
            polar_boxes: (N, 9)  9: (azimuth, dis, z, dx, dy, dz, azimu_rot, azimu_vx, azimu_vy)

        Returns:

        """
        azimuth, dis, z, dx, dy, dz, azimu_rot = torch.split(polar_boxes[..., :7], split_size_or_sections=1, dim=-1)
        x = dis * torch.cos(azimuth)
        y = dis * torch.sin(azimuth)
        rot = azimu_rot + azimuth
        rot = self.limit_period(rot, 0.5, 2*torch.pi)
        cart_boxes = torch.cat([x, y, z, dx, dy, dz, rot], dim=-1)

        if polar_boxes.shape[-1] > 7:
            v_radius, v_azimuth = torch.split(polar_boxes[..., 7:], split_size_or_sections=1, dim=-1)
            v_abs = torch.sqrt(v_azimuth ** 2 + v_radius ** 2)
            v_angle = azimuth + torch.atan2(v_azimuth, v_radius)
            vx = v_abs * torch.cos(v_angle)
            vy = v_abs * torch.sin(v_angle)
            cart_boxes = torch.cat([cart_boxes, vx, vy], dim=-1)

        return cart_boxes

    def decode(self,
               heat,
               rot_sine,
               rot_cosine,
               hei,
               dim,
               vel,
               reg=None,
               task_id=-1):
        """Decode bboxes.

        Args:
            heat (torch.Tensor): Heatmap with the shape of (B, N_cls, H, W).
            rot_sine (torch.Tensor): Sine of rotation with the shape of (B, 1, H, W).
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of (B, 1, H, W).
            hei (torch.Tensor): Height of the boxes with the shape of (B, 1, H, W).
            dim (torch.Tensor): Dim of the boxes with the shape of (B, 3, H, W).
            vel (torch.Tensor): Velocity with the shape of (B, 1, H, W).
            reg (torch.Tensor, optional): Regression value of the boxes in
                2D with the shape of (B, 2, H, W). Default: None.
            task_id (int, optional): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.  List[p_dict0, p_dict1, ...]
                p_dict = {
                    'bboxes': boxes3d,      # (K', 9)
                    'scores': scores,       # (K', )
                    'labels': labels        # (K', )
                }
        """
        batch, cat, _, _ = heat.size()

        # (B, K)
        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)    # (B, K, 2)
            reg = reg.view(batch, self.max_num, 2)
            xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]    # (B, K, 1) + (B, K, 1) --> (B, K, 1)
            ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]    # (B, K, 1) + (B, K, 1) --> (B, K, 1)
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)   # (B, K, 1)
        rot_sine = rot_sine.view(batch, self.max_num, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)  # (B, K, 1)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        azimu_rot = torch.atan2(rot_sine, rot_cosine)     # (B, K, 1)

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)      # (B, K, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)      # (B, K, 3)

        # class label
        clses = clses.view(batch, self.max_num).float()     # (B, K)
        scores = scores.view(batch, self.max_num)   # (B, K)

        # 计算真实的polar坐标.
        azimuth = xs.view(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        dis = ys.view(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]

        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([azimuth, dis, hei, dim, azimu_rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self._transpose_and_gather_feat(vel, inds)    # (B, K, 2)
            azimu_vel = vel.view(batch, self.max_num, 2)
            final_box_preds = torch.cat([azimuth, dis, hei, dim, azimu_rot, azimu_vel], dim=2)    # (B, K, 9)

        final_box_preds = self.polar2cart(final_box_preds)

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold   # (B, K)

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heat.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)      # (B, K, 3) --> (B, K)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)     # (B, K, 3) --> (B, K)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]      # (K, )
                if self.score_threshold:
                    cmask &= thresh_mask[i]     # (K, )

                boxes3d = final_box_preds[i, cmask]     # (K', 9)
                scores = final_scores[i, cmask]         # (K', )
                labels = final_preds[i, cmask]          # (K', )
                predictions_dict = {
                    'bboxes': boxes3d,      # (K', 9)
                    'scores': scores,       # (K', )
                    'labels': labels        # (K', )
                }

                # List[p_dict0, p_dict1, ...]   len = batch_size
                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts
