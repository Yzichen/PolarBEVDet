_base_ = ['../../../mmdetection3d/configs/_base_/default_runtime.py']
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
load_interval = 5
num_gpus = 4
batch_size = 4
num_iters_per_epoch = 158081 // load_interval // (num_gpus * batch_size)
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
do_history_stereo_fusion = False
stereo_out_feats = 64
history_cat_num = 8
history_cat_conv_out_channels = 160

# BEV Head Parameters
bev_encoder_in_channels = (
    base_bev_channels if not do_history else history_cat_conv_out_channels)

# Loss Weights
loss_depth_weight = 3.0

###############################################################################
# General Dataset & Augmentation Details.
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-0.64*math.pi, 0, -3.0, 0.64*math.pi, 75, 4.0]
point_cloud_range_cart = [-35, -75, -3.0, 75, 75, 4.0]

class_names = ['Car', 'Pedestrian', 'Cyclist']

data_config={
    'cams': ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
             'CAM_SIDE_LEFT', 'CAM_SIDE_RIGHT'],
    'Ncams': 5,
    'input_size': (832, 1920),
    'src_size': (1280, 1920),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'azimuth': [-0.64*math.pi, 0.64*math.pi, math.pi/200],
    'radius': [0, 75, 75/96],
    'z': [-3.0, 4.0, 7.0],
    'depth': [1.0, 75.0, 0.5],
}

voxel_size = [math.pi/200, 75/96, 1]
align_camera_center = True
###############################################################################
# Set-up the model.

model = dict(
    type='PolarBEVDetWaymo',
    # Long-Term Fusion
    do_history=do_history,
    do_history_stereo_fusion=do_history_stereo_fusion,
    history_cat_num=history_cat_num,
    history_cat_conv_out_channels=history_cat_conv_out_channels,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
        with_cp=with_cp
    ),
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
    # img_view_transformer=dict(type='PolarLSSViewTransformerBEVDepth',
    #                           grid_config=grid_config,
    #                           input_size=data_config['input_size'],
    #                           in_channels=512,
    #                           mid_channels=256,
    #                           out_channels=base_bev_channels,
    #                           loss_depth_weight=loss_depth_weight,
    #                           depthnet_cfg=dict(stereo=do_history_stereo_fusion),
    #                           ),
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
            dict(num_class=3, class_names=['Car', 'Pedestrian', 'Cyclist']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='PolarCenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-35.0, -75.0, -2, 75.0, 75.0, 4],
            max_num=2000,
            score_threshold=0.01,
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=6.),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=1.5),
        norm_bbox=True,
        align_camera_center=align_camera_center
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[256, 96, 1],
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
            post_center_limit_range=[-35.0, -75.0, -2, 75.0, 75.0, 4],
            max_pool_nms=False,

            # Scale-NMS
            score_threshold=0.01,
            max_per_img=500,
            pre_max_size=2000,
            post_max_size=500,
            nms_type=['rotate'],
            nms_thr=[0.01],
            nms_rescale_factor=[1.0, 1.0, 1.0]
        )
    ),
    aux_img_head = dict(
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
    )
)

###############################################################################
# Set-up the dataset
dataset_type = 'WaymoDatasetBEVDet'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='WaymoPrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadAnnotations2D',
        min_size=2.0,
        filter_invisible=True),
    dict(
        type='WaymoLoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        align_camera_center=align_camera_center
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='WaymoPointToMultiView', downsample=1, grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_cart),
    dict(type='CircleObjectRangeFilter', class_dist_thred=[75] * len(class_names)),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth',
                                 'gt_bboxes', 'centers2d', 'gt_labels', 'depths'])
]

test_pipeline = [
    dict(type='WaymoPrepareImageInputs', data_config=data_config, sequential=False),
    dict(
        type='WaymoLoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False,
        align_camera_center=align_camera_center
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
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
        split='training',
        ann_file=data_root + 'bevdetv2-waymo_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        load_interval=load_interval,
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
        split='training',
        ann_file=data_root + 'bevdetv2-waymo_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        load_interval=1,
        img_info_prototype='bevdet',
        seq_mode=True,
        sequences_split_num=1,
        filter_empty_gt=filter_empty_gt
    ),
    test=dict(
         type=dataset_type,
         data_root=data_root,
         split='training',
         ann_file=data_root + 'bevdetv2-waymo_infos_val.pkl',
         pipeline=test_pipeline,
         classes=class_names,
         modality=input_modality,
         load_interval=1,
         img_info_prototype='bevdet',
         seq_mode=True,
         sequences_split_num=1,
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
# checkpoint_config = dict(
#     interval=checkpoint_epoch_interval * num_iters_per_epoch)
checkpoint_config = dict(interval=num_iters_per_epoch)
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
        with_velo=False
    ),
    # dict(
    #     # we use syncbn to prevent loss divergency
    #     type='SyncbnControlHook',
    #     syncbn_start_iter=2 * num_iters_per_epoch,
    # ),
]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# fp16 = dict(loss_scale='dynamic')
load_from = 'pretrain/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth'
revise_keys = [('backbone', 'img_backbone')]


# 3d_ap: 0.6777992
# 3d_ap_ha_weighted: 0.6444646
# 3d_ap_la_weighted: 0.52016234
# 3d_ap_ONE_SHARD_LEVEL_2: 0.6777992
# 3d_ap_ha_weighted_ONE_SHARD_LEVEL_2: 0.6444646
# 3d_ap_la_weighted_ONE_SHARD_LEVEL_2: 0.52016234
# 3d_ap_RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2: 0.8220792
# 3d_ap_ha_weighted_RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2: 0.8079692
# 3d_ap_la_weighted_RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2: 0.65374565
# 3d_ap_RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2: 0.67430204
# 3d_ap_ha_weighted_RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2: 0.6590105
# 3d_ap_la_weighted_RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2: 0.53093547
# 3d_ap_RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2: 0.46135467
# 3d_ap_ha_weighted_RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2: 0.44473106
# 3d_ap_la_weighted_RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2: 0.34544703
# 3d_ap_RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2: 0.77733546
# 3d_ap_ha_weighted_RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2: 0.69064474
# 3d_ap_la_weighted_RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2: 0.57053435
# 3d_ap_RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2: 0.57803243
# 3d_ap_ha_weighted_RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2: 0.45533633
# 3d_ap_la_weighted_RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2: 0.419548
# 3d_ap_RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2: 0.21432495
# 3d_ap_ha_weighted_RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2: 0.14388873
# 3d_ap_la_weighted_RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2: 0.15328078
# 3d_ap_RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2: 0.62185794
# 3d_ap_ha_weighted_RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2: 0.58466285
# 3d_ap_la_weighted_RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2: 0.43883464
# 3d_ap_RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2: 0.3184568
# 3d_ap_ha_weighted_RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2: 0.2892626
# 3d_ap_la_weighted_RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2: 0.19688952
# 3d_ap_RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2: 0.24049684
# 3d_ap_ha_weighted_RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2: 0.21612066
# 3d_ap_la_weighted_RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2: 0.15488364
# 3d_ap_OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2: 0.6987833
# 3d_ap_ha_weighted_OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2: 0.68285763
# 3d_ap_la_weighted_OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2: 0.5446032
# 3d_ap_OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2: 0.62354296
# 3d_ap_ha_weighted_OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2: 0.53221124
# 3d_ap_la_weighted_OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2: 0.45167542
# 3d_ap_OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2: 0.46455625
# 3d_ap_ha_weighted_OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2: 0.43224525
# 3d_ap_la_weighted_OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2: 0.31933624
# LET-AP 0.5956274966398875
# LET-APH 0.5491047104199728
# LET-APL 0.4385382930437724