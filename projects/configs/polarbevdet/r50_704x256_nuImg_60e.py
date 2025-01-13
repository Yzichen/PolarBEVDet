_base_ = ['./r50_704x256_60e.py']

model = dict(
    img_backbone=dict(
        frozen_stages=0,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        pretrained=None
    )
)

load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]
