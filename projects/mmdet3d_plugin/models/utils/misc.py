import torch
import torch.nn as nn
import numpy as np
from mmdet.core import bbox_xyxy_to_cxcywh
from mmdet.models.utils.transformer import inverse_sigmoid


def memory_refresh(memory, prev_exist):
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:])
    return memory * prev_exist


def topk_gather(feat, topk_indexes):
    """
    Args:
        feat: (B, num_tokens, C)
        topk_indexes: (B, n_topk, 1)

    Returns:

    """
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape

        view_shape = [1 for _ in range(len(feat_shape))]
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)

        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat


def apply_ltrb(locations, pred_ltrb):
    """
    Args:
        locations: (B*N_views, fH, fW, 2)   2: (x, y)
        pred_ltrb: (B*N_views, fH, fW, 4)
    Returns:
        pred_boxes: (B*N_views, fH, fW, 4)
    """
    pred_boxes = torch.zeros_like(pred_ltrb)  # (B*N_views, fH, fW, 4)
    pred_boxes[..., 0] = (locations[..., 0] - pred_ltrb[..., 0])
    pred_boxes[..., 1] = (locations[..., 1] - pred_ltrb[..., 1])
    pred_boxes[..., 2] = (locations[..., 0] + pred_ltrb[..., 2])
    pred_boxes[..., 3] = (locations[..., 1] + pred_ltrb[..., 3])
    min_xy = pred_boxes[..., 0].new_tensor(0)
    max_xy = pred_boxes[..., 0].new_tensor(1)
    pred_boxes = torch.where(pred_boxes < min_xy, min_xy, pred_boxes)
    pred_boxes = torch.where(pred_boxes > max_xy, max_xy, pred_boxes)
    pred_boxes = bbox_xyxy_to_cxcywh(pred_boxes)

    return pred_boxes


def apply_center_offset(locations, center_offset):
    """
    :param locations:  (B*N_views, fH, fW, 2)
    :param pred_ltrb:  (B*N_views, fH, fW, 2)
    """
    centers_2d = torch.zeros_like(center_offset)
    locations = inverse_sigmoid(locations)
    centers_2d[..., 0] = locations[..., 0] + center_offset[..., 0]  # x1
    centers_2d[..., 1] = locations[..., 1] + center_offset[..., 1]  # y1
    centers_2d = centers_2d.sigmoid()

    return centers_2d


@torch.no_grad()
def locations(features, stride, pad_h, pad_w):
    """
    Arguments:
        features:  (B*N_view, C, fH, fW)
        stride: int
        pad_h: H
        pad_w: W
    Return:
        locations:  (fH, fW, 2),  feature map中每个grid在原图中的归一化坐标.
    """

    h, w = features.size()[-2:]  # fh, fw
    device = features.device

    shifts_x = (torch.arange(
        0, stride * w, step=stride,
        dtype=torch.float32, device=device
    ) + stride // 2) / pad_w  # (fw, )
    shifts_y = (torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    ) + stride // 2) / pad_h  # (fh, )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)  # (fh, fw)
    shift_x = shift_x.reshape(-1)  # (fh*fw, )
    shift_y = shift_y.reshape(-1)  # (fh*fw, )
    locations = torch.stack((shift_x, shift_y), dim=1)  # (fh*fw, 2)

    locations = locations.reshape(h, w, 2)  # (fh*fw, 2) --> (fh, fw, 2)

    return locations


def gaussian_2d(shape, sigma=1.0):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float, optional): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
        radius - left:radius + right]).to(heatmap.device,
                                          torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class SELayer_Linear(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        """
        Args:
            x: (B, num_sample_tokens, C=embed_dims)
            x_se: (B, num_sample_tokens, C)
        Returns:

        """
        x_se = self.conv_reduce(x_se)  # (B, num_sample_tokens, C=embed_dims)
        x_se = self.act1(x_se)  # (B, num_sample_tokens, C=embed_dims)
        x_se = self.conv_expand(x_se)  # (B, num_sample_tokens, C=embed_dims)
        return x * self.gate(x_se)  # (B, num_sample_tokens, C=embed_dims)

