class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
checkpoint = './pretrained_models/IS-Fusion_epoch_10.pth'
debug = True
max_epochs = 1000
camou_path = './workdir/20251111_215729/2camou.npy'
freq = 0
num_samples = 1
allowed_words = './nuscenes_words.txt'
target_class = 'car'
dynamic_ratio = False
area_ratio = 0.002
lr = 0.01
color_map = [[0, 0, 0], [255, 255, 255], [0, 18, 79], [5, 80, 214],
             [71, 178, 243], [178, 159, 211], [77, 58, 0], [211, 191, 167],
             [247, 110, 26], [110, 76, 16]]
lambda_reduce = 1
lambda_smooth = 0
lambda_nps = 0
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5, 54, 54, 3]
img_scale = (384, 1056)
total_epochs = 10
res_factor = 1
out_size_factor = 8
voxel_shape = 1440
bev_size = 180
grid_size = [[180, 180, 1], [90, 90, 1]]
region_shape = [(6, 6, 1), (6, 6, 1)]
region_drop_info = [
    dict({0: dict(max_tokens=36, drop_range=(0, 100000))}),
    dict({0: dict(max_tokens=36, drop_range=(0, 100000))})
]
model = dict(
    type='ISFusionDetector',
    detach=True,
    pc_range=[-54, -54, -5, 54, 54, 3],
    voxel_size=[0.075, 0.075, 0.2],
    out_size_factor=8,
    img_backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=True,
        convert_weights=False),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3),
    pts_voxel_layer=dict(
        point_cloud_range=[-54, -54, -5, 54, 54, 3],
        max_num_points=-1,
        voxel_size=[0.075, 0.075, 0.2],
        max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=5,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=[0.075, 0.075, 0.2],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-54, -54, -5, 54, 54, 3],
        norm_cfg=dict(type='naiveSyncBN1d', eps=0.001, momentum=0.01)),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 1440, 1440],
        base_channels=32,
        output_channels=256,
        order=('conv', 'norm', 'act'),
        encoder_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256,
                                                                         256)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    fusion_encoder=dict(
        type='ISFusionEncoder',
        num_points_in_pillar=12,
        embed_dims=256,
        bev_size=180,
        num_views=6,
        region_shape=[(6, 6, 1), (6, 6, 1)],
        grid_size=[[180, 180, 1], [90, 90, 1]],
        region_drop_info=[
            dict({0: dict(max_tokens=36, drop_range=(0, 100000))}),
            dict({0: dict(max_tokens=36, drop_range=(0, 100000))})
        ],
        instance_num=200),
    pts_backbone=dict(
        type='SECONDV2',
        in_channels=128,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='TransFusionHeadV2',
        num_proposals=200,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=10,
        num_decoder_layers=1,
        num_heads=8,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(
            center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54, -54],
            voxel_size=[0.075, 0.075],
            out_size_factor=8,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10),
        loss_cls=dict(
            type='ClassAttackLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0,
            debug=False,
            use_original_focal=False,
            lambda_original=1.0,
            use_reverse_focal=False,
            lambda_reverse=1.0,
            use_complement_loss=False,
            lambda_complement=1.0,
            use_uniform_confusion=False,
            lambda_uniform=1.0,
            use_margin_confusion=False,
            lambda_margin=1.0,
            use_hard_wrong_class=False,
            lambda_hard_wrong=1.0,
            margin=0.0),
        loss_bbox=dict(
            type='BBoxAttackLoss',
            reduction='mean',
            loss_weight=0.25,
            debug=False,
            use_original_l1=False,
            lambda_original=1.0,
            use_reverse_l1=False,
            lambda_reverse=1.0,
            use_translation_attack=False,
            lambda_translation=1.0,
            use_orbit_attack=False,
            lambda_orbit=1.0,
            use_scale_attack=False,
            lambda_scale=1.0,
            use_orientation_attack=False,
            lambda_orientation=1.0,
            orbit_radius=2.0),
        loss_heatmap=dict(
            type='GIADLoss',
            reduction='mean',
            loss_weight=1.0,
            debug=False,
            use_original_gaussian_loss=False,
            lambda_original=0.0,
            use_reverse_gaussian_loss=False,
            lambda_reverse=1.0,
            use_ring_loss=False,
            lambda_ring=1.0,
            use_attention_diffusion=True,
            lambda_diffusion=1000.0,
            use_entropy_loss=False,
            lambda_entropy=1.0,
            use_contrast_loss=False,
            lambda_contrast=1.0,
            use_veiling_luminance=False,
            lambda_veiling=1.0,
            ring_radius=3.0,
            ring_sigma=1.0,
            diffusion_iterations=3,
            diffusion_kernel_size=5,
            contrast_window=11,
            veiling_power=2.0,
            veiling_epsilon=1e-06)),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],
            voxel_size=[0.075, 0.075, 0.2],
            out_size_factor=8,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=[-54, -54, -5, 54, 54, 3])),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=8,
            pc_range=[-54, -54],
            voxel_size=[0.075, 0.075],
            nms_type=None,
            use_rotate_nms=True,
            nms_thr=0.2,
            max_num=200)))
dataset_type = 'NuScenesDataset'
data_root2 = './data/nuscenes/'
data_root = './data/nuscenes/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
db_sampler = dict(
    type='MMDataBaseSamplerV2',
    data_root='./data/nuscenes/',
    info_path='./data/nuscenes/nuscenes_dbinfos_train.pkl',
    rate=1.0,
    img_num=6,
    blending_type=None,
    depth_consistent=True,
    check_2D_collision=True,
    collision_thr=[0, 0.3, 0.5, 0.7],
    mixup=0.7,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4]))
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFilesV2_Camou',
        to_float32=True,
        mask_path='/standard/eng_vivastorage/nuscenes/nuscenes/nuscenes_masks',
        index_file='./nuscenes_masks_index.txt'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        painting=False),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        painting=False),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='ObjectSampleV2',
        stop_epoch=8,
        db_sampler=dict(
            type='MMDataBaseSamplerV2',
            data_root='./data/nuscenes/',
            info_path='./data/nuscenes/nuscenes_dbinfos_train.pkl',
            rate=1.0,
            img_num=6,
            blending_type=None,
            depth_consistent=True,
            check_2D_collision=True,
            collision_thr=[0, 0.3, 0.5, 0.7],
            mixup=0.7,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5)),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4])),
        sample_2d=True),
    dict(type='ModalMask3D', mode='train', stop_epoch=8),
    dict(
        type='ImageAug3D_Camou',
        final_dim=(384, 1056),
        resize_lim=[0.57, 0.825],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='GlobalRotScaleTransV2',
        resize_lim=[0.9, 1.1],
        rot_lim=[-0.78539816, 0.78539816],
        trans_lim=0.5,
        is_train=True),
    dict(type='RandomFlip3DV2'),
    dict(
        type='PointsRangeFilter', point_cloud_range=[-54, -54, -5, 54, 54, 3]),
    dict(
        type='ObjectRangeFilter', point_cloud_range=[-54, -54, -5, 54, 54, 3]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='ImageNormalize_Camou',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='PointShuffle'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3DV2',
        keys=['points', 'img', 'masks', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera',
            'camera2lidar', 'lidar2img', 'img_aug_matrix', 'lidar_aug_matrix'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        painting=False),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        painting=False),
    dict(
        type='LoadMultiViewImageFromFilesV2_Camou',
        to_float32=True,
        mask_path='/standard/eng_vivastorage/nuscenes/nuscenes/nuscenes_masks',
        index_file='./nuscenes_masks_index.txt'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(384, 1056),
        pts_scale_ratio=1.0,
        flip=False,
        pcd_horizontal_flip=False,
        pcd_vertical_flip=False,
        transforms=[
            dict(
                type='ImageAug3D_Camou',
                final_dim=(384, 1056),
                resize_lim=[0.72, 0.72],
                bot_pct_lim=[0.0, 0.0],
                rot_lim=[0.0, 0.0],
                rand_flip=False,
                is_train=False),
            dict(
                type='ImageNormalize_Camou',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='GlobalRotScaleTransV2',
                resize_lim=[1.0, 1.0],
                rot_lim=[0.0, 0.0],
                trans_lim=0.0,
                is_train=False),
            dict(type='RandomFlip3DV2'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-54, -54, -5, 54, 54, 3]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3DV2',
                keys=['points', 'img', 'masks'],
                meta_keys=[
                    'camera_intrinsics', 'camera2ego', 'lidar2ego',
                    'lidar2camera', 'camera2lidar', 'lidar2img',
                    'img_aug_matrix', 'lidar_aug_matrix'
                ])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type='NuScenesDataset',
            data_root='./data/nuscenes/',
            ann_file='./data/nuscenes/nuscenes_infos_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFilesV2_Camou',
                    to_float32=True,
                    mask_path=
                    '/standard/eng_vivastorage/nuscenes/nuscenes/nuscenes_masks',
                    index_file='./nuscenes_masks_index.txt'),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    painting=False),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4],
                    painting=False),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_bbox=True,
                    with_label=True),
                dict(
                    type='ObjectSampleV2',
                    stop_epoch=8,
                    db_sampler=dict(
                        type='MMDataBaseSamplerV2',
                        data_root='./data/nuscenes/',
                        info_path='./data/nuscenes/nuscenes_dbinfos_train.pkl',
                        rate=1.0,
                        img_num=6,
                        blending_type=None,
                        depth_consistent=True,
                        check_2D_collision=True,
                        collision_thr=[0, 0.3, 0.5, 0.7],
                        mixup=0.7,
                        prepare=dict(
                            filter_by_difficulty=[-1],
                            filter_by_min_points=dict(
                                car=5,
                                truck=5,
                                bus=5,
                                trailer=5,
                                construction_vehicle=5,
                                traffic_cone=5,
                                barrier=5,
                                motorcycle=5,
                                bicycle=5,
                                pedestrian=5)),
                        classes=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        sample_groups=dict(
                            car=2,
                            truck=3,
                            construction_vehicle=7,
                            bus=4,
                            trailer=6,
                            barrier=2,
                            motorcycle=6,
                            bicycle=6,
                            pedestrian=2,
                            traffic_cone=2),
                        points_loader=dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=5,
                            use_dim=[0, 1, 2, 3, 4])),
                    sample_2d=True),
                dict(type='ModalMask3D', mode='train', stop_epoch=8),
                dict(
                    type='ImageAug3D_Camou',
                    final_dim=(384, 1056),
                    resize_lim=[0.57, 0.825],
                    bot_pct_lim=[0.0, 0.0],
                    rot_lim=[-5.4, 5.4],
                    rand_flip=True,
                    is_train=True),
                dict(
                    type='GlobalRotScaleTransV2',
                    resize_lim=[0.9, 1.1],
                    rot_lim=[-0.78539816, 0.78539816],
                    trans_lim=0.5,
                    is_train=True),
                dict(type='RandomFlip3DV2'),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[-54, -54, -5, 54, 54, 3]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-54, -54, -5, 54, 54, 3]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='ImageNormalize_Camou',
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                dict(type='PointShuffle'),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3DV2',
                    keys=[
                        'points', 'img', 'masks', 'gt_bboxes_3d',
                        'gt_labels_3d'
                    ],
                    meta_keys=[
                        'camera_intrinsics', 'camera2ego', 'lidar2ego',
                        'lidar2camera', 'camera2lidar', 'lidar2img',
                        'img_aug_matrix', 'lidar_aug_matrix'
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            modality=dict(
                use_lidar=True,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            test_mode=False,
            use_valid_flag=False,
            box_type_3d='LiDAR',
            img_num=6,
            split_file='./nuscenes_infos_train_05pct.txt',
            load_interval=1)),
    val=dict(
        type='NuScenesDataset',
        data_root='./data/nuscenes/',
        ann_file='./data/nuscenes/nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                painting=False),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                use_dim=[0, 1, 2, 3, 4],
                painting=False),
            dict(
                type='LoadMultiViewImageFromFilesV2_Camou',
                to_float32=True,
                mask_path=
                '/standard/eng_vivastorage/nuscenes/nuscenes/nuscenes_masks',
                index_file='./nuscenes_masks_index.txt'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(384, 1056),
                pts_scale_ratio=1.0,
                flip=False,
                pcd_horizontal_flip=False,
                pcd_vertical_flip=False,
                transforms=[
                    dict(
                        type='ImageAug3D_Camou',
                        final_dim=(384, 1056),
                        resize_lim=[0.72, 0.72],
                        bot_pct_lim=[0.0, 0.0],
                        rot_lim=[0.0, 0.0],
                        rand_flip=False,
                        is_train=False),
                    dict(
                        type='ImageNormalize_Camou',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    dict(
                        type='GlobalRotScaleTransV2',
                        resize_lim=[1.0, 1.0],
                        rot_lim=[0.0, 0.0],
                        trans_lim=0.0,
                        is_train=False),
                    dict(type='RandomFlip3DV2'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[-54, -54, -5, 54, 54, 3]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3DV2',
                        keys=['points', 'img', 'masks'],
                        meta_keys=[
                            'camera_intrinsics', 'camera2ego', 'lidar2ego',
                            'lidar2camera', 'camera2lidar', 'lidar2img',
                            'img_aug_matrix', 'lidar_aug_matrix'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        img_num=6,
        split_file='./nuscenes_infos_val_05pct.txt',
        box_type_3d='LiDAR'),
    test=dict(
        type='NuScenesDataset',
        data_root='./data/nuscenes/',
        ann_file='./data/nuscenes/nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                painting=False),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                use_dim=[0, 1, 2, 3, 4],
                painting=False),
            dict(
                type='LoadMultiViewImageFromFilesV2_Camou',
                to_float32=True,
                mask_path=
                '/standard/eng_vivastorage/nuscenes/nuscenes/nuscenes_masks',
                index_file='./nuscenes_masks_index.txt'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(384, 1056),
                pts_scale_ratio=1.0,
                flip=False,
                pcd_horizontal_flip=False,
                pcd_vertical_flip=False,
                transforms=[
                    dict(
                        type='ImageAug3D_Camou',
                        final_dim=(384, 1056),
                        resize_lim=[0.72, 0.72],
                        bot_pct_lim=[0.0, 0.0],
                        rot_lim=[0.0, 0.0],
                        rand_flip=False,
                        is_train=False),
                    dict(
                        type='ImageNormalize_Camou',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    dict(
                        type='GlobalRotScaleTransV2',
                        resize_lim=[1.0, 1.0],
                        rot_lim=[0.0, 0.0],
                        trans_lim=0.0,
                        is_train=False),
                    dict(type='RandomFlip3DV2'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[-54, -54, -5, 54, 54, 3]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3DV2',
                        keys=['points', 'img', 'masks'],
                        meta_keys=[
                            'camera_intrinsics', 'camera2ego', 'lidar2ego',
                            'lidar2camera', 'camera2lidar', 'lidar2img',
                            'img_aug_matrix', 'lidar_aug_matrix'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        img_num=6,
        box_type_3d='LiDAR',
        split_file='./nuscenes_infos_val_05pct.txt'))
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True, priority='HIGH')]
runner = dict(type='CustomEpochBasedRunner', max_epochs=10)
evaluation = dict(interval=5)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/only_vehicles'
load_from = './pretrain_models/swint-nuimages-pretrained-e2e.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)
find_unused_parameters = True
