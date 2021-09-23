_base_ = ['../_base_/datasets/ID_card.py']
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512-8ae85183_20200713.pth'
resume_from = None #'/mnt/ssd/marley/ID_Card/mmpose/work_dirs/overfit_higherhrnet_w32_IDCard_512x512_evaluate_batch>1/latest.pth'
dist_params = dict(backend='nccl')
workflow = [('train', 1), ('val', 1)]
checkpoint_config = dict(interval=100)
evaluation = dict(interval=1, metric='mAP', save_best='AP')
work_dir = './work_dirs/iter_run'
optimizer = dict(
    type='Adam',
    lr=0.0015,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[200, 260])

log_config = dict(
    interval=1, 
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
total_epochs = 5
runner = dict(type='EpochBasedRunner', max_epochs=200)
# runner = dict(type='IterBasedRunner', max_iters=3000)

channel_cfg = dict(
    dataset_joints=4,
    dataset_channel=[
        [0, 1, 2, 3],
    ],
    inference_channel=[
        0, 1, 2, 3
    ])

data_cfg = dict(
    image_size=512,
    base_size=256,
    base_sigma=2,
    heatmap_size=[128, 256],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=2,
    scale_aware_sigma=False,
)

# model settings
model = dict(
    type='AssociativeEmbedding',
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/hrnet_w32-36af842e.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
    ),
    keypoint_head=dict(
        type='AEHigherResolutionHead',
        in_channels=32,
        num_joints=4,
        tag_per_joint=True,
        extra=dict(final_conv_kernel=1, ),
        num_deconv_layers=1,
        num_deconv_filters=[32],
        num_deconv_kernels=[4],
        num_basic_blocks=4,
        cat_output=[True],
        with_ae_loss=[True, False],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=4,
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0])),
    train_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        img_size=data_cfg['image_size'],
        topk=3,
        base_size=data_cfg['base_size']),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=5,
        scale_factor=[1],
        with_heatmaps=[True, True],
        with_ae=[True, False],
        project2image=True,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        flip_test=True))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=0,
        scale_factor=[1.0, 1.0],
        scale_type='short',
        trans_factor=0),
    # dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='BottomUpGenerateTarget',
        sigma=2,
        max_num_people=5,
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=['image_file', 'center', 'scale', 'test_scale_factor', 'base_size']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=0,
        scale_factor=[1.0, 1.0],
        scale_type='short',
        trans_factor=0),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='BottomUpGenerateTestTarget',
        sigma=2,
        max_num_people=5,
    ),
    dict(
        type='BottomUpGenerateTarget',
        sigma=2,
        max_num_people=5,
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index', 'aug_mask', 'aug_joints', 
            'aug_targets', 'num_scales', 'num_joints', 'max_num_people'
        ]),
]

test_pipeline = val_pipeline

data_root = '/mnt/ssd/marley/ID_Card/ID_card_data'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    val_dataloader=dict(samples_per_gpu=16),
    test_dataloader=dict(samples_per_gpu=16),
    train=dict(
        type='BottomUpIDCardDataset',
        ann_file=f'{data_root}/annotations/mini_annotations.json',
        img_prefix=f'{data_root}/mini/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='BottomUpIDCardDataset',
        ann_file=f'{data_root}/annotations/mini_annotations.json',
        img_prefix=f'{data_root}/mini/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='BottomUpIDCardDataset',
        ann_file=f'{data_root}/annotations/val_annotations.json',
        img_prefix=f'{data_root}/val/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
