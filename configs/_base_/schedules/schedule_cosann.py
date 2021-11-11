# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0003)
optimizer_config = dict()
# learning policy
lr_config= dict(policy='CosineRestart', periods=[30000, 60000, 90000, 120000, 150000, 180000, 220000, 260000, 300000, 340000, 400000], restart_weights = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.09, 0.08, 0.07, 0.05], min_lr=1.5e-5, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=400000)
checkpoint_config = dict(by_epoch=False, interval=20000)
evaluation = dict(interval=10000, metric='mIoU')
