# optimizer
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.00005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=3e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=400000)
checkpoint_config = dict(by_epoch=False, interval=20000)
evaluation = dict(interval=30000, metric='mIoU', pre_eval=False)
