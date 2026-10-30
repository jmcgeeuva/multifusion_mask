
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
import torch.distributed as dist
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
    result = val_dataset.evaluate(outputs, **eval_kwargs)
    print(result)
    return result


def save_if_best(eval_result, best_mAP, camou_para1, camou_para, work_dir):
    mAP = next((v for k, v in eval_result.items() if 'mAP' in k), None)
    if mAP is None:
        return best_mAP
    if mAP > best_mAP:
        camou_png = cv2.cvtColor(
            (camou_para1[0].detach().cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(os.path.join(work_dir, 'best_camou.png'), camou_png)
        np.save(os.path.join(work_dir, 'best_camou.npy'), camou_para.detach().cpu().numpy())
        print(f'[Best] New best mAP: {mAP:.4f} (prev {best_mAP:.4f}), saved best_camou')
        return mAP
    return best_mAP

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
    if cfg.get('camou_path', None) is not None:
        camou_para, camou_para1 = load_camou(cfg.camou_path, expand_kernel, device=device)
        begin_para = camou_para.detach().clone()
    else:
        camou_para = torch.rand([1, h, w, 3]).float().to(device)
        camou_para.requires_grad_(True)
        begin_para = deepcopy(camou_para)
        camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    # Ensure all ranks start from the same texture (rank 0 is authoritative).
    if cfg.distributed:
        dist.broadcast(camou_para.data, src=0)

    optimizer = optim.Adam([camou_para], lr=cfg.lr)

    os.makedirs(work_dir, exist_ok=True)

    tex_trans = get_augmentation(img_size)

    if cfg.freq != 0:
        tmpdir = os.path.join('./tmp', f'tmp_{timestamp}', 'pkl')
        os.makedirs(tmpdir, exist_ok=True)
        tmpdir = os.path.join(tmpdir, 'out.pkl')
        outputs, _ = test_attack(
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

    # model.eval()
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    underlying = model.module if hasattr(model, 'module') else model
    underlying.detach = False
    best_mAP = float('-inf')

    # _debug_file = None
    _grad_file = None
    _grad_step = 0
    _grad_needs_header = False
    if cfg.get('debug', False) and work_dir is not None:
        # _debug_path = os.path.join(work_dir, f'train_debug_rank{rank}.txt')
        # _debug_file = open(_debug_path, 'a', buffering=1)

        _grad_path = os.path.join(work_dir, f'texture_grad_rank{rank}.tsv')
        _grad_needs_header = not os.path.exists(_grad_path)
        _grad_file = open(_grad_path, 'a', buffering=1)

    # def dprint(msg):
    #     # print(msg)
    #     if _debug_file is not None:
    #         _debug_file.write(msg + '\n')

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0
        _viz_done_this_epoch = False
        if hasattr(train_loaders.sampler, 'set_epoch'):
            train_loaders.sampler.set_epoch(epoch)
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(train_loaders))
        time.sleep(2)  # This line can prevent deadlock problem in some cases.

        for data in train_loaders:
            optimizer.zero_grad(set_to_none=True)

            imgs = data['img'].data[0].to(device)
            mask_img = data['masks'].data[0].to(device)
            _img_metas = data['img_metas'].data[0]
            batch_size = imgs.shape[0]
            camou_trans = tex_trans(camou_para1.permute(0, 3, 1, 2))
            learned_camou, _batch_attack_meta = mask_imgs(
                imgs,
                mask_img,
                camou_trans,
                allowed_words,
                device=device,
                dynamic_check=cfg.dynamic_ratio,
                ratio_check=cfg.area_ratio,
                num_samples=num_samples,
                debug=cfg.debug,
                target_class=getattr(cfg, 'target_class', None),
                img_metas=_img_metas,
            )
            _batch_obj_count = sum(1 for m in _batch_attack_meta if m is not None)
            _batch_size = len(_batch_attack_meta)

            if cfg.get('debug', False) and rank == 0 and not _viz_done_this_epoch:
                _viz_dir = os.path.join(work_dir, 'debug_views')
                os.makedirs(_viz_dir, exist_ok=True)
                _t = learned_camou[0].data[0].detach().cpu().float()
                _s = _t[0]  # [N_cams, 3, H, W]
                n_cams = _s.shape[0]
                cols = min(n_cams, 3)
                rows = (n_cams + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), squeeze=False)
                for ci in range(n_cams):
                    v = _s[ci]  # [3, H, W]
                    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
                    axes[ci // cols][ci % cols].imshow(v.permute(1, 2, 0).numpy())
                    axes[ci // cols][ci % cols].set_title(f'Cam {ci}')
                    axes[ci // cols][ci % cols].axis('off')
                for ci in range(n_cams, rows * cols):
                    axes[ci // cols][ci % cols].set_visible(False)
                plt.suptitle(f'Epoch {epoch}', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(_viz_dir, f'epoch{epoch:04d}.png'), dpi=80, bbox_inches='tight')
                plt.close(fig)
                _viz_done_this_epoch = True

            _debug_step = cfg.get('debug_step', None)
            if _debug_step is not None and _grad_step in _debug_step:
                _viz_dir = os.path.join(work_dir, 'debug_views')
                os.makedirs(_viz_dir, exist_ok=True)
                _t = learned_camou[0].data[0].detach().cpu().float()
                _s = _t[0]  # [N_cams, 3, H, W]
                n_cams = _s.shape[0]
                cols = min(n_cams, 3)
                rows = (n_cams + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), squeeze=False)
                for ci in range(n_cams):
                    v = _s[ci]  # [3, H, W]
                    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
                    axes[ci // cols][ci % cols].imshow(v.permute(1, 2, 0).numpy())
                    axes[ci // cols][ci % cols].set_title(f'Cam {ci}')
                    axes[ci // cols][ci % cols].axis('off')
                for ci in range(n_cams, rows * cols):
                    axes[ci // cols][ci % cols].set_visible(False)
                plt.suptitle(f'Step {_grad_step} mask debug  obj_count={_batch_obj_count}', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(_viz_dir, f'step{_grad_step:05d}_mask_debug_{rank}.png'), dpi=80, bbox_inches='tight')
                plt.close(fig)
                print(f'[debug_step] saved mask debug for step {_grad_step} → {_viz_dir}')

            data['img'] = learned_camou
            # Move all remaining DC-wrapped tensors to the model's device.
            # MMDataParallel.scatter normally does this, but the attack loop
            # calls the model directly so we handle it explicitly here.
            for key, val in data.items():
                if key == 'img':
                    continue
                if hasattr(val, '_data') and isinstance(val._data, list) and val._data:
                    inner = val._data[0]
                    if isinstance(inner, torch.Tensor):
                        val._data[0] = inner.to(device)
                    elif isinstance(inner, list):
                        val._data[0] = [
                            x.to(device) if isinstance(x, torch.Tensor) else x
                            for x in inner
                        ]

            losses = model(return_loss=True, **data)
            giad_keys = [k for k in losses if k.startswith('loss_giad_')]
            giad_losses = [losses[k] for k in losses if k.startswith('loss_giad_')]
            giad_sum = torch.stack(giad_losses).sum()

            catk_keys = [k for k in losses if k.startswith('layer_-1_loss_catk_')]
            catk_losses = [losses[k] for k in losses if k.startswith('layer_-1_loss_catk_')]
            catk_sum = torch.stack(catk_losses).sum()

            batk_keys = [k for k in losses if k.startswith('layer_-1_loss_batk_')]
            batk_losses = [losses[k] for k in losses if k.startswith('layer_-1_loss_batk_')]
            batk_sum = torch.stack(batk_losses).sum()

            loss_tensor =  giad_sum + catk_sum + batk_sum
            loss_tensor = loss_tensor + cfg.lambda_smooth*(loss_smooth(camou_para))
            loss_tensor = loss_tensor + cfg.lambda_nps*(loss_nps(camou_para, color_set))
            loss_tensor.backward()
            # print(camou_para.requires_grad)
            # print(camou_para.is_leaf)
            # print(camou_para.grad is None)

            # Optional debug logging for individual GIAD sub-components.
            # When loss_heatmap type is GIADLoss, the head adds
            # 'loss_giad_*' keys to the losses dict automatically.
            if cfg.get('debug', False):
                # if _debug_file is not None:
                #     _debug_file.write(f'--- epoch={epoch} ---\n')
                print(f"loss_heatmap: {losses.get('loss_heatmap', None)}")
                print(f"loss_heatmap_ins: {losses.get('loss_heatmap_ins', None)}")
                print(f"matched_ious: {losses.get('matched_ious', None)}")
                if giad_keys:
                    parts = '  '.join(
                        f'{k}={losses[k].item():.4f}' for k in sorted(giad_keys)
                    )
                    print(f'[GIAD] {parts}, sum: {giad_sum}')
                if catk_keys:
                    parts = '  '.join(
                        f'{k}={losses[k].item():.4f}' for k in sorted(catk_keys)
                    )
                    print(f'[ClassAttack] {parts}, sum: {catk_sum}')
                if batk_keys:
                    parts = '  '.join(
                        f'{k}={losses[k].item():.4f}' for k in sorted(batk_keys)
                    )
                    print(f'[BBoxAttack] {parts}, sum: {batk_sum}')

            # Want to increase the error
            # FIXME NON-ADVERSARIAL CAN BE TESTED HERE BY REMOVING "2 -" and making it a loss minimization problem
            # non_adv = True
            # if not non_adv:
            #     total_loss = cfg.lambda_reduce * (2 - loss_tensor)
            # else:
            #     total_loss = cfg.lambda_reduce * loss_tensor
            # Smoothing of the camouflage
            # Average camou_para gradients across all ranks so every rank
            # applies the same optimizer update and textures stay in sync.
            if cfg.distributed:
                # Use a zero tensor as a stand-in when this rank had no mask
                # coverage so that dist.all_reduce (a collective) is always
                # called by every rank and the job never hangs.
                if camou_para.grad is None:
                    camou_para.grad = torch.zeros_like(camou_para)
                    
                # rank = dist.get_rank()

                # if camou_para.grad is not None:
                #     print(
                #         rank,
                #         camou_para.grad.norm().item(),
                #         camou_para.grad.abs().mean().item(),
                #     )
                dist.all_reduce(camou_para.grad.data, op=dist.ReduceOp.SUM)
                # print(
                #     rank,
                #     "after",
                #     camou_para.grad.norm().item(),
                #     camou_para.grad.abs().mean().item(),
                # )
                # camou_para.grad.data /= world_size
            if cfg.get('debug', False) and _grad_file is not None and camou_para.grad is not None:
                g = camou_para.grad  # shape [1, h, w, 3]
                B_cur = len(_batch_attack_meta)
                if _grad_needs_header:
                    obj_cols = '\t'.join(f'obj_size_{i}' for i in range(B_cur))
                    _grad_file.write(
                        f'step\tepoch\tgrad_norm\tgrad_mean_abs\tgrad_max_abs\t'
                        f'grad_norm_r\tgrad_norm_g\tgrad_norm_b\t{obj_cols}\tobj_count\n'
                    )
                    _grad_needs_header = False
                def _obj_size(meta):
                    if meta is None:
                        return 'n/a'
                    x1, y1, x2, y2 = meta['bbox_2d']
                    return str((x2 - x1) * (y2 - y1))
                obj_sizes = '\t'.join(_obj_size(m) for m in _batch_attack_meta)
                _grad_file.write(
                    f'{_grad_step}\t{epoch}\t'
                    f'{g.norm().item():.6f}\t'
                    f'{g.abs().mean().item():.6f}\t'
                    f'{g.abs().max().item():.6f}\t'
                    f'{g[0, :, :, 0].norm().item():.6f}\t'
                    f'{g[0, :, :, 1].norm().item():.6f}\t'
                    f'{g[0, :, :, 2].norm().item():.6f}\t'
                    f'{obj_sizes}\t'
                    f'{_batch_obj_count}\n'
                )
                _grad_step += 1
            running_loss += loss_tensor.item()
            optimizer.step()
            with torch.no_grad():
                camou_para.clamp_(0, 1)
            camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            # print(camou_para1.min())
            # print(camou_para1.max())
            if rank == 0:
                prog_bar.update()
        camou_png = cv2.cvtColor((camou_para1[0].detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if cfg.distributed:
            loss_t = torch.tensor(running_loss, device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            running_loss = loss_t.item() / world_size
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
                outputs, attack_log = test_attack(
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
                    eval_result = list_stats(val_dataset, outputs, cfg)
                    best_mAP = save_if_best(eval_result, best_mAP, camou_para1, camou_para, work_dir)


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
    
    # set random seeds — priority: CLI --seed > cfg.seed > random
    seed = args.seed if args.seed is not None else cfg.get('seed', None)
    if seed is not None:
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(seed, deterministic=args.deterministic)
    else:
        import random
        seed = int(random.random()*10e7)
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(seed, deterministic=args.deterministic)
    args.seed = seed
    cfg.seed = seed
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

    if args.log_dir:
        os.environ['ATTACK_LOG_DIR'] = args.log_dir

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
    # Build a wider loader for the no-attack baseline evaluation so MMDataParallel
    # can spread inference across all available GPUs.  batch_size = num_gpus × 1,
    # with the DC collated into num_gpus groups so scatter sends 1 sample per GPU.
    _n_eval_gpus = torch.cuda.device_count() if not cfg.distributed else 1
    if _n_eval_gpus > 1:
        val_loader_noatk = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=_n_eval_gpus,
            dist=cfg.distributed,
            shuffle=False
        )
    else:
        val_loader_noatk = val_loader
    
    model = load_model(cfg, args, datasets)
    underlying = model.module if hasattr(model, 'module') else model
    for m in underlying.modules():
        if hasattr(m, 'with_cp'):
            m.with_cp = False
    logger.info(f'Model:\n{model}')
    logger = get_root_logger(cfg.log_level)
    


    

    # put model on gpus
    if not cfg.distributed:
        model = model.cuda(cfg.gpu_ids[0])
        print(f'[GPU] CUDA available: {torch.cuda.is_available()}')
        print(f'[GPU] Using device: cuda:{cfg.gpu_ids[0]} ({torch.cuda.get_device_name(cfg.gpu_ids[0])})')
        print(f'[GPU] Model device: {next(model.parameters()).device}')
    else:
        # All model params are frozen (requires_grad=False), so no params
        # participate in backward. find_unused_parameters=True tells DDP not
        # to error when its reducer finds empty gradient buckets.
        find_unused_parameters = cfg.get('find_unused_parameters', True)
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
        device=f'cuda:{cfg.gpu_ids[0]}' if not cfg.distributed else f'cuda:{torch.cuda.current_device()}',
        world_size=world_size,
        num_samples=cfg.num_samples,
        max_epochs=cfg.max_epochs,
        allowed_words=allowed_words, 
        cfg=cfg,
        timestamp=args.timestamp
    )

if __name__ == '__main__':
    main()