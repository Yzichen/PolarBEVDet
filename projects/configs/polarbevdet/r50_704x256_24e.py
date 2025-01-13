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
batch_size = 8
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 24
num_epochs_single_frame = 4
checkpoint_epoch_interval = 1

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
grid_config = {
    'azimuth': [-math.pi, math.pi, math.pi/128],
    'radius': [0, 51.2, 51.2/64],
    'z': [-10.0, 10.0, 20.0],
    'depth': [1.0, 52.0, 0.5],
}

voxel_size = [math.pi/128, 51.2/64, 1]
align_camera_center = True
###############################################################################
# Set-up the model.

model = dict(
    type='PolarBEVDet',
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
        out_channels=[128, 128, 128, 128],
        with_cp=with_cp
    ),
    stereo_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[1, 2, 4, 8],
        out_channels=[stereo_out_feats, stereo_out_feats, stereo_out_feats,
                      stereo_out_feats],
        final_conv_feature_dim=stereo_out_feats
    ),
    # 2D -> BEV Image View Transformer.
    img_view_transformer=dict(type='PolarLSSViewTransformerSOLOFusion',
                              grid_config=grid_config,
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
        backbone_output_ids=[0, ],
        with_cp=with_cp
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
    # Same detection head used in BEVDet, BEVDepth, etc
    pts_bbox_head=dict(
        type='Polar_CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=10, class_names=['car', 'truck',
                                            'construction_vehicle',
                                            'bus', 'trailer',
                                            'barrier',
                                            'motorcycle', 'bicycle',
                                            'pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='PolarCenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=6.),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=1.5),
        norm_bbox=True,
        align_camera_center=align_camera_center
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[256, 64, 1],
            voxel_size=voxel_size,
            out_size_factor=1,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
        )
    ),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    ),
    aux_img_head=dict(
        type='AuxHead',
        num_classes=10,
        in_channels=512,
        loss_cls2d=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='L1Loss', loss_weight=10.0),
        train_cfg=dict(
            assigner2d=dict(
                type='HungarianAssigner2D',
                cls_cost=dict(type='FocalLossCost', weight=2.),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0)))
    ),
)

###############################################################################
# Set-up the dataset
dataset_type = 'NuScenesDatasetBEVDet'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

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
        align_camera_center=align_camera_center
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_cart),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth',
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
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-7)

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
    interval=checkpoint_epoch_interval * num_iters_per_epoch, max_keep_ckpts=5)
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
        temporal_start_iter=num_epochs_single_frame * num_iters_per_epoch,
    ),
]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# fp16 = dict(loss_scale='dynamic')