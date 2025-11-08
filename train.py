
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
# export PYTHONPATH=$PYTHONPATH:$(pwd)/IS-Fusion

def plot_bbox(image, bboxes, labels, print_labels=False, name='test.png'):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image.permute(1, 2, 0))  
      
    # Plot each bounding box  
    for bbox, label in zip(bboxes, labels):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        if print_labels:
            plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.savefig(name)  

def print_images(img, idx, norm=True, name='test.png'):
    img = img[idx].permute(0, 2, 3, 1)
    plt.figure()
    fig, ax = plt.subplots(2, 3)

    if norm:
        img = (img - img.min()) / (img.max() - img.min())
    

    img1 = img[0, ...]
    img2 = img[1, ...]
    img3 = img[2, ...]
    img4 = img[3, ...]
    img5 = img[4, ...]
    img6 = img[5, ...]

    ax[0, 0].imshow(img1)
    ax[0, 0].set_title(f"CAMERA_FRONT")
    ax[0, 0].axis('off')
    ax[0, 1].imshow(img2)
    ax[0, 1].set_title(f"CAMERA_FRONT_RIGHT")
    ax[0, 1].axis('off')
    ax[0, 2].imshow(img3)
    ax[0, 2].set_title(f"CAMERA_FRONT_LEFT")
    ax[0, 2].axis('off')
    ax[1, 0].imshow(img4)
    ax[1, 0].set_title(f"CAMERA_BACK")
    ax[1, 0].axis('off')
    ax[1, 1].imshow(img5)
    ax[1, 1].set_title(f"CAMERA_BACK_LEFT")
    ax[1, 1].axis('off')
    ax[1, 2].imshow(img6)
    ax[1, 2].set_title(f"CAMERA_BACK_RIGHT")
    ax[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(name)

    return img1, img2, img3, img4, img5, img6

def overlay_image(image, mask, texture, debug=False):
    contour = torch.where((mask == 1), torch.zeros(1, device=mask.device), torch.ones(1, device=mask.device)).to(device=mask.device)
    overlayed = torch.where((contour == 1.), image.to(device=contour.device), texture.to(device=contour.device)).to(device=mask.device)

    return overlayed

def bbox_xyxy_from_mask_torch(mask: torch.Tensor):
    """
    mask: (B, H, W) tensor of 0/1 or bool
    returns: (B, 4) tensor of bounding boxes (x1, y1, x2, y2)
             if a mask has no positive pixels, it will return (0, 0, 0, 0) for that batch
    """
    assert mask.ndim == 3, "mask should be (B, H, W)"
    B, H, W = mask.shape
    mask = mask.bool()

    # Create coordinate grids
    y_coords = torch.arange(H, device=mask.device).view(1, H, 1)
    x_coords = torch.arange(W, device=mask.device).view(1, 1, W)

    # Masked positions (set non-object to inf/-inf for reduction)
    x_min = torch.where(mask, x_coords, torch.full_like(x_coords, W)).amin(dim=(1,2))
    x_max = torch.where(mask, x_coords, torch.full_like(x_coords, -1)).amax(dim=(1,2))
    y_min = torch.where(mask, y_coords, torch.full_like(y_coords, H)).amin(dim=(1,2))
    y_max = torch.where(mask, y_coords, torch.full_like(y_coords, -1)).amax(dim=(1,2))

    # Handle empty masks (where no 1s)
    empty = (x_max < 0) | (y_max < 0)
    x_min[empty] = 0
    y_min[empty] = 0
    x_max[empty] = 0
    y_max[empty] = 0

    # Stack into (B, 4)
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
    return boxes
    
def tex_trans(camou, size=4096):
    """
    Flip, rotate, and crop the camouflage texture
    """
    camou_column = []
    for i in range(6):
        camou_row_list = []
        for j in range(6):
            camou1 = transforms.RandomHorizontalFlip(p=0.5)(camou.permute(0, 3, 1, 2)[0])
            camou2 = transforms.RandomVerticalFlip(p=0.5)(camou1)
            if np.random.rand(1)>0.5:
                camou3 = transforms.functional.rotate(camou2, 90)
            else:
                camou3 = camou2
            camou_row_list.append(camou3)
        camou_row = torch.cat(tuple(camou_row_list), 1)
        camou_column.append(camou_row)
    camou_full = torch.cat(tuple(camou_column), 2).unsqueeze(0)
    camou_crop = transforms.RandomCrop(size)(camou_full).permute(0, 2, 3, 1)
    return camou_crop
            
def mask_imgs(yolo_model, imgs, camou_para, allowed_words, num_samples = 1, ratio_check=2e-3, debug=False):
    
    if debug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory_name = f"output_imgs_{timestamp}"
        os.makedirs(directory_name, exists_ok=True)
        print(f"Directory '{directory_name}' created.")
        print_images(imgs, 0, norm=False, name=f'{directory_name}/dark.png')
        print_images(imgs, 0, norm=True, name=f'{directory_name}/norm.png')
    
    range_num = imgs.max() - imgs.min()
    min_num = imgs.min()
    # B x 6 x 3 x H x W
    imgs_norm = (imgs - min_num) / (range_num)
    # imgs = torch.nn.functional(img, size=())
    mask_entry, labels, batches, angles = run_yolo8(
            yolo_model, 
            imgs_norm, 
            imgs_norm.shape[-2], 
            imgs_norm.shape[-1],
            search_labels=allowed_words
    )
    
    imgs_processed = imgs
    if len(mask_entry) > 0:
        masks = torch.stack(mask_entry)
        
        bboxes = bbox_xyxy_from_mask_torch(masks)
        areas = (bboxes[:, 2] - bboxes[:, 0])*(bboxes[:, 3] - bboxes[:, 1])
        total_area = imgs.shape[-2]*imgs.shape[-1]
        ratio = areas/total_area
        ratio_indices = (ratio > ratio_check).nonzero(as_tuple=True)[0]
        if ratio_indices.numel() != 0:
            if debug:
                plot_masks(masks, labels, batches, angles, imgs, name=directory_name+'/test{}.png')
            
            
            H, W = imgs_norm.shape[-2], imgs_norm.shape[-1]  # example shape
            # green = torch.zeros(3, H, W)
            # green[1, :, :] = 1.0  # set G channel to 1
            # lily_img = Image.open('./Lily.jpg').resize((imgs_norm[0].shape[-1], imgs_norm[0].shape[-2]))
            # lily_img = transforms.ToTensor()(lily_img)
            imgs_overlayed = overlay_image(imgs_norm[batches, angles, :, :], masks.unsqueeze(1).repeat(1, 3, 1, 1), camou_para.permute(0, 3, 1, 2))

            # unnormalize
            imgs_processed = (imgs_overlayed * range_num) + min_num


            # Have to switch to cpu because it does not handle small tensors well 
            for i in range(imgs.shape[0]):
                t = torch.nonzero(torch.tensor(batches) == i).flatten()

                tmp_angles = torch.tensor(angles)[t]
                
                ratio_indices = (ratio[t] > ratio_check).nonzero(as_tuple=True)[0]
                prob = (ratio[ratio_indices] / ratio[ratio_indices].sum())
                # FIXME when prob is empty the sum of the probability is 0 and this throws an error
                if len(prob) == 0:
                    continue
                
                r_idx = torch.multinomial(prob.cpu(), num_samples, replacement=True).to(prob.device)

                imgs_tmp = imgs_processed[t][ratio_indices].to(device=imgs.device)
                choice = ratio_indices[r_idx]
                imgs[i, tmp_angles[choice], :, :, :] = imgs_tmp[r_idx, :, :, :]
        if debug:
            print_images(imgs.detach(), 0, norm=True, name=f'{directory_name}/masked.png')
            labels = [yolo_model.names[label] for i, label in enumerate(labels) if i in ratio_indices]
            batches = [label for i, label in enumerate(batches) if i in ratio_indices]
            angles = [label for i, label in enumerate(angles) if i in ratio_indices]
            bboxes = bboxes[ratio_indices]
            plot_bbox(imgs_norm[batches, angles, :, :][0], [bboxes[0]], [labels[0]])
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(imgs_processed[0].permute(1,2, 0).cpu().detach().numpy())
            plt.title('car')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(masks[0].cpu().detach().numpy())
            plt.title(f'mask')
            plt.axis('off')
            plt.savefig(f'{directory_name}/overlay.png')
    return [DC([imgs], stack=False, cpu_only=False)]

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

def train_attack(
    model, 
    yolo_model, 
    data_loaders, 
    rank,
    world_size,
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

    color_set = torch.tensor([[0,0,0],[255,255,255],[0,18,79],[5,80,214],[71,178,243],[178,159,211],[77,58,0],[211,191,167],[247,110,26],[110,76,16]]).to(model.device).float() / 255
    ##################################### SETUP Transpose #############################################
    expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(model.device)
    expand_kernel.weight.data.fill_(0)
    expand_kernel.bias.data.fill_(0)
    for i in range(3):
        expand_kernel.weight[i, i, :, :].data.fill_(1)
    ###################################################################################################

    dataset = data_loaders.dataset
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    #####################################################################################
    # continuous color
    camou_para = torch.rand([1, h, w, 3]).float().to(model.device)
    camou_para.requires_grad_(True)
    begin_para = deepcopy(camou_para)
    camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    optimizer = optim.Adam([camou_para], lr=0.01)
    
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory_name = f"paras/paras_{timestamp}"
        os.makedirs(directory_name, exist_ok=True)
        try:
            camou_png = cv2.cvtColor((camou_para1[0].detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(directory_name+'/begin_para.png', camou_png)
            np.save(directory_name+'/begin_para.npy', begin_para.detach().cpu().numpy())
        except:
            print(f"failed to print or save ./ {epoch}")
    #####################################################################################
    
    # model.eval()
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    debug=False

    model.module.detach = False
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0
        for data in tqdm(data_loaders, total=len(data_loaders)): 
            optimizer.zero_grad(set_to_none=True)
            
            imgs = data['img'].data[0]
            camou_trans = tex_trans(camou_para1, size=img_size)
            learned_camou = mask_imgs(yolo_model, imgs, camou_trans, allowed_words, num_samples=num_samples, debug=debug)[0]
            assert learned_camou.data[0].requires_grad, "Learned_camou does not require gradient"
            
            data['img'] = learned_camou
            
            # losses = model(return_loss=True, **data)
            out = model.train_step(data, optimizer)
            loss_tensor = out['loss']
            lambda_reduce = 2
            lambda_smooth = .05
            lambda_nps = .6
            # Want to increase the error
            total_loss = lambda_reduce*(bias - (loss_tensor))
            # Smoothing of the camouflage
            total_loss = total_loss + lambda_smooth*(loss_smooth(camou_para))
            total_loss = total_loss + lambda_nps*(loss_nps(camou_para, color_set))
            total_loss.backward()
            running_loss += total_loss.item()
            optimizer.step()
            camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            camou_para1 = torch.clamp(camou_para1, 0, 1)
        camou_png = cv2.cvtColor((camou_para1[0].detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if rank == 0:
            print(f'Current loss: {running_loss/len(data_loaders)}')
            try:
                cv2.imwrite(directory_name+'/'+str(epoch)+'camou.png', camou_png)
                np.save(directory_name+'/'+str(epoch)+'camou.npy', camou_para.detach().cpu().numpy())
            except:
                print(f"failed to print or save ./ {epoch}")
                raise

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--extra_tag', type=str, default=None, help='extra tag for this experiment')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
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
    parser.add_argument('--seed', type=int, default=0, help='random seed')
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


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
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

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

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
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
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

    def load_words(file_name):
        allowed_words = []
        with open(file_name) as f:
            for line in f:
                line = line.replace('\n', '')
                allowed_words.append(line)
        return allowed_words

    allowed_words = load_words('./allowed_words.txt')

    logger.info(f'Model:\n{model}')
    logger = get_root_logger(cfg.log_level)
    
    
    datasets = build_dataset(cfg.data.train)
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

    data_loaders = build_dataloader(
        datasets,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # cfg.gpus will be ignored if distributed
        len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed
    )

    # put model on gpus
    if not distributed:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
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

    max_epochs=100
    num_samples=1
    rank, world_size = get_dist_info()
    yolo_model=YOLO(f'yolo/yolo{rank}/yolov8n-seg.pt')
    outputs = train_attack(
        model=model,
        yolo_model=yolo_model, 
        data_loaders=data_loaders,
        rank=rank,
        world_size=world_size,
        num_samples=num_samples,
        max_epochs=max_epochs,
        allowed_words=allowed_words, 
        cfg=cfg
    )

if __name__ == '__main__':
    main()