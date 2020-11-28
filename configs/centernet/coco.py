use_coco = True
model = dict(
    type='CenterNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        heads=dict(hm=21 if use_coco else 21,
            wh=2,
            reg=2),
        style='pytorch'
        )
    )

train_cfg = dict(a = 10)
test_cfg = dict(a = 5)

dataset_type = 'Ctdet'
if use_coco:
    data_root = '/data/lktime-seg-tp/dataset/PASCALVOC/VOCdevkit/'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
else:
    data_root = '/data/lktime-seg-tp/dataset/PASCALVOC/VOCdevkit/'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        use_coco=use_coco,
        ann_file=data_root + 'PASCAL_VOC/' +
            ('pascal_train2012.json' if use_coco else 'pascal_train2012.json'),
        img_prefix=data_root + ('VOC2012/JPEGImages/' if use_coco else 'images/'),
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'PASCAL_VOC/pascal_val2012.json',
        img_prefix=data_root + ('VOC2012/JPEGImages/' if use_coco else 'images/'),
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'PASCAL_VOC/pascal_val2012.json',
        img_prefix=data_root + ('VOC2012/JPEGImages/' if use_coco else 'images/'),
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
        with_mask=False,
        with_label=False,
        test_mode=True))

# optimizer
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/centernet_hg'
load_from = None
resume_from = None
workflow = [('train', 1)]