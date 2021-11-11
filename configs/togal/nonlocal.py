wandb_project ='mmsegmentation'
wandb_experiment_name = 'ResNet50, PointRend finetuning, 2 classes'

######################################################################
# optimizer
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.00005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='exp', gamma=0.999994, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=300000)
checkpoint_config = dict(by_epoch=False, interval=20000)
evaluation = dict(interval=10000, metric='mIoU', pre_eval=False, tb_log_dir="./work_dirs/tf_logs")

######################################################################
# runtime settings
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project=wandb_project,
                name=wandb_experiment_name))
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None# 'work_dirs/standard_unet/latest.pth'
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True

######################################################################
# dataset settings

dataset_type = 'CustomDataset'

data_root = 'data/'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size = (448, 448)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='RGB2Gray')
    dict(type='RandomRotate', prob=0.2, degree=180),
    dict(type='Corrupt', prob = 0.5, corruption=['gaussian_blur', 'gaussian_noise', 'shot_noise', 'brightness', 'contrast', 'jpeg_compression', 'pixelate']),
    dict(type='RandomFlip', prob=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale = (448, 448),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=test_pipeline))



# model settings
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='NLHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dropout_ratio=0.1,
        reduction=2,
        use_scale=True,
        mode='embedded_gaussian',
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
