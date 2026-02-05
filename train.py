
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division

import copy
import warnings
from mmcv.runner import get_dist_info, init_dist
from os import path as osp
import cv2

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.utils import collect_env, get_root_logger
from mmseg import __version__ as mmseg_version

from mmcv import Config, DictAction
from mmdet.apis import set_random_seed
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
import torchvision.transforms as transforms
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
import mmcv
import torch
from mmdet3d.apis import collect_results_gpu, collect_results_cpu
from yolo_test import run_yolo8, plot_masks
import matplotlib.patches as patches
from mmcv.parallel import DataContainer as DC
import os
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
import argparse
import time
from copy import deepcopy
from torch import optim
import numpy as np
from datetime import datetime
from test import mask_imgs, test_attack, load_camou
from augmentation import get_augmentation

from torch.utils.data import Subset
# export PYTHONPATH=$PYTHONPATH:$(pwd)/IS-Fusion
    
def loss_smooth(img):
    b, c, w, h = img.shape
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    return torch.square(torch.sum(s1 + s2)) / (b*c*w*h)
    

def loss_nps(img, color_set):
    # img: [batch_size, h, w, 3]
    # color_set: [color_num, 3]
    _, h, w, c = img.shape
    color_num, c = color_set.shape
    img1 = img.unsqueeze(1)
    color_set1 = color_set.unsqueeze(1).unsqueeze(1).unsqueeze(0)
    gap = torch.min(torch.sum(torch.abs(img1 - color_set1)/255, -1), 1).values
    return torch.sum(gap)/h/w

def list_stats(val_dataset, outputs, cfg):
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='bbox'))
    print(val_dataset.evaluate(outputs, **eval_kwargs))

def train_attack(
    model, 
    yolo_model, 
    train_loaders, 
    val_loader,
    rank,
    world_size,
    device='cpu',
    timestamp=None,
    work_dir=None,
    num_samples=1,
    bias=10,
    max_epochs=10,
    allowed_words= ['car', 'bicycle', 'person'], 
    cfg=None, 
    img_size=(384, 1056), 
    H=1056, 
    W=1056, 
    resolution=8):
    

    h, w = int(H/resolution), int(W/resolution)

    color_set = torch.tensor(cfg.color_map).to(device).float() / 255
    ##################################### SETUP Transpose #############################################
    expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(device)
    expand_kernel.weight.data.fill_(0)
    expand_kernel.bias.data.fill_(0)
    for i in range(3):
        expand_kernel.weight[i, i, :, :].data.fill_(1)
    ###################################################################################################

    dataset = train_loaders.dataset
    val_dataset = val_loader.dataset

    #####################################################################################
    # continuous color
    if cfg.camou_path is not None:
        camou_para, camou_para1 = load_camou(cfg.camou_path, expand_kernel, device=device)
    else:
        camou_para = torch.rand([1, h, w, 3]).float().to(device)
        camou_para.requires_grad_(True)
        begin_para = deepcopy(camou_para)
        camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    
    optimizer = optim.Adam([camou_para], lr=cfg.lr)
    
    # if work_dir is None:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     directory_name = os.path.join(os.getcwd(), "paras", f"paras_{timestamp}")
    #     os.makedirs(directory_name, exist_ok=True)
    # else:
    os.makedirs(work_dir, exist_ok=True)

    if cfg.freq != 0:
        tmpdir=os.path.join('./tmp', f'tmp_{timestamp}', 'pkl')
        os.makedirs(tmpdir, exist_ok=True)
        tmpdir=os.path.join(tmpdir, 'out.pkl')
        # run validation
        outputs = test_attack(
            model=model,
            no_attack=True,
            yolo_model=yolo_model, 
            data_loader=val_loader,
            tex_trans=tex_trans,
            camou_para1=None,
            allowed_words=allowed_words, 
            tmpdir=tmpdir,
            gpu_collect=False, 
            cfg=cfg
        )

        if rank == 0:
            list_stats(val_dataset, outputs, cfg)

    if rank == 0:
        try:
            camou_png = cv2.cvtColor((camou_para1[0].detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(work_dir+'/begin_para.png', camou_png)
            np.save(work_dir+'/begin_para.npy', begin_para.detach().cpu().numpy())
        except:
            print(f"failed to print or save {work_dir}")
    #####################################################################################
    
    tex_trans = get_augmentation(img_size)

    # model.eval()
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    if device == 'cpu':
        model.detach = False
    else:
        model.module.detach = False
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        time.sleep(2)  # This line can prevent deadlock problem in some cases.

        for data in train_loaders: 
            optimizer.zero_grad(set_to_none=True)
            
            imgs = data['img'].data[0]
            batch_size = imgs.shape[0]
            # imgs = imgs.to(device=device)
            camou_trans = tex_trans(camou_para1.permute(0, 3, 1, 2))
            learned_camou = mask_imgs(yolo_model, imgs, camou_trans, allowed_words, device=imgs.device, dynamic_check=cfg.dynamic_ratio, ratio_check=cfg.area_ratio, num_samples=num_samples, debug=cfg.debug)[0]
            # assert learned_camou.data[0].requires_grad, "Learned_camou does not require gradient"
            # learned_camou.data[0] = learned_camou.data[0].cpu()
            
            data['img'] = learned_camou
            
            losses = model(return_loss=True, **data)
            # out = model.train_step(data, optimizer)
            heatmap_loss = cfg.gamma_heatmap*losses['loss_heatmap']
            cls_loss = cfg.gamma_cls*losses['layer_-1_loss_cls']
            bbox_loss = cfg.gamma_bbox*losses['layer_-1_loss_bbox']

            loss_tensor = heatmap_loss+cls_loss+bbox_loss
            # Want to increase the error
            total_loss = cfg.lambda_reduce*(2 - (loss_tensor))
            # Smoothing of the camouflage
            total_loss = total_loss + cfg.lambda_smooth*(loss_smooth(camou_para))
            total_loss = total_loss + cfg.lambda_nps*(loss_nps(camou_para, color_set))
            total_loss.backward()
            running_loss += total_loss.item()
            optimizer.step()
            camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            camou_para1 = torch.clamp(camou_para1, 0, 1)
            if rank == 0:
                batch_size = imgs.shape[0]
                for _ in range(batch_size * world_size):
                    prog_bar.update()
        camou_png = cv2.cvtColor((camou_para1[0].detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if rank == 0:
            print(f'\nCurrent loss: {running_loss/len(train_loaders)}')
            try:
                cv2.imwrite(work_dir+'/'+str(epoch)+'camou.png', camou_png)
                np.save(work_dir+'/'+str(epoch)+'camou.npy', camou_para.detach().cpu().numpy())
            except:
                print(f"failed to print or save ./ {epoch}")
                raise

        if cfg.freq != 0:
            if epoch % cfg.freq == 0:
                # run validation
                outputs = test_attack(
                    model=model,
                    no_attack=False,
                    yolo_model=yolo_model, 
                    tex_trans=tex_trans,
                    data_loader=val_loader,
                    camou_para1=camou_para1,
                    allowed_words=allowed_words, 
                    tmpdir=tmpdir,
                    gpu_collect=False, 
                    cfg=cfg
                )

                if rank == 0:
                    list_stats(val_dataset, outputs, cfg)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--timestamp', help='')
    parser.add_argument('--extra_tag', type=str, default=None, help='extra tag for this experiment')
    parser.add_argument('--log-dir', type=str, help='')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    # parser.add_argument(
    #     '--allowed-words', type=str, default='./allowed_words.txt', help='')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def load_gpu(cfg, logger):
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                    'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

def setup(args, cfg):
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    if args.extra_tag is not None:
        cfg.work_dir = osp.join(cfg.work_dir, args.extra_tag)
    
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)
    
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.distributed = False
    else:
        cfg.distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    else:
        import random
        args.seed = int(random.random()*10e7)
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    return logger

def load_words(file_name):
    allowed_words = []
    with open(file_name) as f:
        for line in f:
            line = line.replace('\n', '')
            allowed_words.append(line)
    return allowed_words

def load_model(cfg, args, dataset):
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, cfg.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    return model

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    logger = setup(args, cfg)

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {cfg.distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)



    def load_words(file_name):
        allowed_words = []
        with open(file_name) as f:
            for line in f:
                line = line.replace('\n', '')
                allowed_words.append(line)
        return allowed_words

    allowed_words = load_words(cfg.allowed_words)

    datasets = build_dataset(cfg.data.train)
    load_gpu(cfg, logger)
    datasets.is_vis_on_test = False
    train_loaders = build_dataloader( 
        datasets, 
        cfg.data.samples_per_gpu, 
        cfg.data.workers_per_gpu, 
        len(cfg.gpu_ids), 
        dist=cfg.distributed, 
        seed=cfg.seed
    )
    val_dataset = build_dataset(cfg.data.test)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=cfg.distributed,
        shuffle=False
    )
    
    model = load_model(cfg, args, datasets)
    logger.info(f'Model:\n{model}')
    logger = get_root_logger(cfg.log_level)
    


    

    # put model on gpus
    if not cfg.distributed:
        # model = MMDataParallel(
        #     model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        model = model.cpu()
    else:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

    for param in model.parameters():
        param.requires_grad = False


    rank, world_size = get_dist_info()
    yolo_model=YOLO(f'yolo/yolo{rank}/yolov8n-seg.pt')

    train_attack(
        model=model,
        yolo_model=yolo_model, 
        work_dir=args.log_dir,
        train_loaders=train_loaders,
        val_loader=val_loader,
        rank=rank,
        device='cpu',
        world_size=world_size,
        num_samples=cfg.num_samples,
        max_epochs=cfg.max_epochs,
        allowed_words=allowed_words, 
        cfg=cfg,
        timestamp=args.timestamp
    )

if __name__ == '__main__':
    main()