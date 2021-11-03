wandb_project ='mmdetection'
wandb_experiment_name = 'test1delete'

######################################################################
# optimizer
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.00005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=3e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=400000)
checkpoint_config = dict(by_epoch=False, interval=20000)
evaluation = dict(interval=30000, metric='mIoU', pre_eval=False)

######################################################################
# runtime settings
log_config = dict(
    interval=10,
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
workflow = [('train', 3), ('val', 1)]
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
    dict(type='RandomRotate', prob=0.3, degree=180),
    #dict(type='PhotoMetricDistortion',
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
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UnetPlusPlus',
        encoder_name="tu-tf_efficientnetv2_b3"
),
    decode_head=dict(
        type='FCNHead',
        in_channels=16,
        in_index=-1,
        channels=16,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))
