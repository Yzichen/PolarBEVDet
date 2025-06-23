_base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
          '../../../mmdetection3d/configs/_base_/default_runtime.py']
import math

work_dir = None
load_from = None
resume_from = None

resume_optimizer = False
find_unused_parameters = False

# Because we use a custom sampler to load data in sequentially during training,
# we can only use IterBasedRunner instead of EpochBasedRunner. To train for a
# fixed # of epochs, we need to know how many iterations are in each epoch. The
# # of iters in each epoch depends on the overall batch size, which is # of
# GPUs (num_gpus) and batch size per GPU (batch_size). "28130" is # of training
# samples in nuScenes.
num_gpus = 4
batch_size = 4
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 24
num_epochs_single_frame = 4
checkpoint_epoch_interval = 24

# By default, 3D detection datasets randomly choose another sample if there is
# no GT object in the current sample. This does not make sense when doing
# sequential sampling of frames, so we disable it.
filter_empty_gt = False

# Intermediate Checkpointing to save GPU memory.
with_cp = True

###############################################################################
# High-level Model & Training Details
base_bev_channels = 80

# Long-Term Fusion Parameters
do_history = True
do_history_stereo_fusion = True
stereo_out_feats = 64
history_cat_num = 16
history_cat_conv_out_channels = 160

# BEV Head Parameters
bev_encoder_in_channels = (
    base_bev_channels if not do_history else history_cat_conv_out_channels)

# Loss Weights
loss_depth_weight = 3.0
velocity_code_weight = 1.0

###############################################################################
# General Dataset & Augmentation Details.
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-math.pi, 0, -5.0, math.pi, 51.2, 3.0]
point_cloud_range_cart = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
data_config={
    'cams': ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
polar_grid_config = {
    'azimuth': [-math.pi, math.pi, math.pi/192],
    'radius': [0, 56.6, 56.6/104],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 57.0, 0.5],
}

cart_grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
}


voxel_size = [math.pi/192, 56.6/104, 1]
align_camera_center = False
###############################################################################
# Set-up the model.

model = dict(
    type='PolarBEVDetOCC',
    # Long-Term Fusion
    do_history=do_history,
    do_history_stereo_fusion=do_history_stereo_fusion,
    history_cat_num=history_cat_num,
    history_cat_conv_out_channels=history_cat_conv_out_channels,
    # Standard R50 + FPN for Image Encoder
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=with_cp,
        style='pytorch'),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]
    ),
    stereo_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[1, 2, 4, 8],
        out_channels=[stereo_out_feats, stereo_out_feats, stereo_out_feats,
                      stereo_out_feats],
        final_conv_feature_dim=stereo_out_feats
    ),
    img_view_transformer=dict(type='PolarLSSViewTransformerSOLOFusion',
                              grid_config=polar_grid_config,
                              input_size=data_config['input_size'],
                              in_channels=512,
                              mid_channels=256,
                              out_channels=base_bev_channels,
                              loss_depth_weight=loss_depth_weight,
                              depthnet_cfg=dict(stereo=do_history_stereo_fusion),
                              downsample=16,
                              stereo_sampling_num=7,
                              stereo_group_num=8,
                              stereo_gauss_bin_stdev=2,
                              stereo_eps=1e-5,
                              ),
    # Pre-processing of BEV features before using Long-Term Fusion
    pre_process=dict(
        type='CustomResNet',
        numC_input=base_bev_channels,
        num_layer=[2, ],
        num_channels=[base_bev_channels, ],
        stride=[1, ],
        backbone_output_ids=[0, ]
    ),
    seg_head=dict(
        type='SegHead',
        in_dim=bev_encoder_in_channels,
        mid_dim=128,
        num_classes=1,
    ),
    # After using long-term fusion, process BEV for detection head.
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=bev_encoder_in_channels,
        num_layer=[2, 2, 2],
        num_channels=[base_bev_channels * 2,
                      base_bev_channels * 4,
                      base_bev_channels * 8],
        stride=[2, 2, 2],
        backbone_output_ids=[0, 1, 2],
    ),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=base_bev_channels * 8 + base_bev_channels * 2,
        out_channels=256
    ),
    occ_head=dict(
        type='Polar_BEVOCCHead2D',
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=False,
        num_classes=18,
        use_predicter=True,
        class_balance=True,
        polar_grid_config=polar_grid_config,
        cart_grid_config=cart_grid_config,
        loss_occ=dict(
            type='CustomFocalLoss',
            use_sigmoid=True,
            loss_weight=1.0
        ),
    ),
    # aux_img_head=dict(
    #     type='AuxHead',
    #     num_classes=10,
    #     in_channels=512,
    #     loss_cls2d=dict(
    #         type='QualityFocalLoss',
    #         use_sigmoid=True,
    #         beta=2.0,
    #         loss_weight=2.0),
    #     loss_centerness=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    #     loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
    #     loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
    #     loss_centers2d=dict(type='L1Loss', loss_weight=10.0),
    #     train_cfg=dict(
    #         assigner2d=dict(
    #             type='HungarianAssigner2D',
    #             cls_cost=dict(type='FocalLossCost', weight=2.),
    #             reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
    #             iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
    #             centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0)))
    # )

)

###############################################################################
# Set-up the dataset
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadAnnotations2D',
        min_size=2.0,
        filter_invisible=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=polar_grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera',
                                'gt_bboxes', 'centers2d', 'gt_labels', 'depths'])

]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False,
        align_camera_center=align_camera_center
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        modality=input_modality,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet',
        seq_mode=True,
        sequences_split_num=1,
        filter_empty_gt=filter_empty_gt
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        modality=input_modality,
        img_info_prototype='bevdet',
        seq_mode=True,
        sequences_split_num=1
    ),
    test=dict(
         type=dataset_type,
         data_root=data_root,
         pipeline=test_pipeline,
         classes=class_names,
         ann_file=data_root + 'nuscenes_infos_val.pkl',
         modality=input_modality,
         img_info_prototype='bevdet',
         seq_mode=True,
         sequences_split_num=1
    ),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=2,
        num_iters_to_seq=-1,
        random_drop=0.0
    ),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

###############################################################################
# Optimizer & Training

# Default is 2e-4 learning rate for batch size 64. When I used a smaller
# batch size, I linearly scale down the learning rate. To do this
# "automatically" over both per-gpu batch size and # of gpus, I set-up the
# lr as-if I'm training with batch_size per gpu for 8 GPUs below, then also
# use the autoscale-lr flag when doing training, which scales the learning
# rate based on actual # of gpus used, assuming the given learning rate is
# w.r.t 8 gpus.
lr = (2e-4 / 64) * (num_gpus * batch_size)
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)

# Mixed-precision training scales the loss up by a factor before doing
# back-propagation. I found that in early iterations, the loss, once scaled by
# 512, goes beyond the Fp16 maximum 65536 and causes nan issues. So, the
# initial scaling here is 1.0 for "num_iters_per_epoch // 4" iters (1/4 of
# first epoch), then goes to 512.0 afterwards.
# Note that the below does not actually affect the effective loss being
# backpropagated, it's just a trick to get FP16 to not overflow.
optimizer_config = dict(
    type='WarmupFp16OptimizerHook',
    grad_clip=dict(max_norm=5, norm_type=2),
    warmup_loss_scale_value=1.0,
    warmup_loss_scale_iters=num_iters_per_epoch // 4,
    loss_scale=512.0
)

lr_config = None
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
checkpoint_config = dict(
    interval=checkpoint_epoch_interval * num_iters_per_epoch)
evaluation = dict(
    interval=num_epochs * num_iters_per_epoch, pipeline=test_pipeline)

custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.001,
        save_interval=checkpoint_epoch_interval * num_iters_per_epoch,
        priority=49
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_iter=num_epochs_single_frame * num_iters_per_epoch, with_velo=False
    ),
]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# fp16 = dict(loss_scale='dynamic')

load_from = "ckpts/r50_704x256_60e.pth"
revise_keys = None


# ===> per class IoU of 6019 samples:
# ===> others - IoU = 12.31
# ===> barrier - IoU = 47.27
# ===> bicycle - IoU = 30.1
# ===> bus - IoU = 43.08
# ===> car - IoU = 46.51
# ===> construction_vehicle - IoU = 25.33
# ===> motorcycle - IoU = 29.07
# ===> pedestrian - IoU = 28.12
# ===> traffic_cone - IoU = 33.8
# ===> trailer - IoU = 26.56
# ===> truck - IoU = 35.3
# ===> driveable_surface - IoU = 59.26
# ===> other_flat - IoU = 33.44
# ===> sidewalk - IoU = 36.34
# ===> terrain - IoU = 32.31
# ===> manmade - IoU = 27.66
# ===> vegetation - IoU = 26.18
# ===> mIoU of 6019 samples: 33.68
# {'mIoU': array([0.123, 0.473, 0.301, 0.431, 0.465, 0.253, 0.291, 0.281, 0.338,
#        0.266, 0.353, 0.593, 0.334, 0.363, 0.323, 0.277, 0.262, 0.844])}


# +----------------------+----------+----------+----------+
# |     Class Names      | RayIoU@1 | RayIoU@2 | RayIoU@4 |
# +----------------------+----------+----------+----------+
# |        others        |  0.115   |  0.125   |  0.127   |
# |       barrier        |  0.440   |  0.477   |  0.492   |
# |       bicycle        |  0.287   |  0.319   |  0.325   |
# |         bus          |  0.538   |  0.641   |  0.696   |
# |         car          |  0.531   |  0.599   |  0.623   |
# | construction_vehicle |  0.218   |  0.301   |  0.337   |
# |      motorcycle      |  0.254   |  0.295   |  0.317   |
# |      pedestrian      |  0.361   |  0.409   |  0.424   |
# |     traffic_cone     |  0.374   |  0.390   |  0.398   |
# |       trailer        |  0.242   |  0.314   |  0.386   |
# |        truck         |  0.453   |  0.551   |  0.594   |
# |  driveable_surface   |  0.546   |  0.634   |  0.727   |
# |      other_flat      |  0.280   |  0.318   |  0.354   |
# |       sidewalk       |  0.253   |  0.305   |  0.358   |
# |       terrain        |  0.238   |  0.310   |  0.376   |
# |       manmade        |  0.376   |  0.451   |  0.501   |
# |      vegetation      |  0.266   |  0.383   |  0.471   |
# +----------------------+----------+----------+----------+
# |         MEAN         |  0.340   |  0.401   |  0.442   |
# +----------------------+----------+----------+----------+
# {'RayIoU': 0.3941487268772801, 'RayIoU@1': 0.3395460897479281, 'RayIoU@2': 0.40138092428521027, 'RayIoU@4': 0.44151916659870194}