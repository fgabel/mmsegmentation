_base_ = [
'../_base_/datasets/togal.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='Unet',
        encoder_name="tu-tf_efficientnet_b3_ap",
        decoder_channels=(256,128,64,32,16),
        encoder_depth=5
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
    test_cfg=dict(mode='whole'))