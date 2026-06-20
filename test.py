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
from datetime import datetime
import numpy as np
from augmentation import get_augmentation
import random
# export PYTHONPATH=$PYTHONPATH:$(pwd)/IS-Fusion

# Maps YOLO/COCO class names to nuScenes detection class names.
YOLO_TO_NUSCENES = {
    'car':          'car',
    'truck':        'truck',
    'bus':          'bus',
    'bicycle':      'bicycle',
    'motorcycle':   'motorcycle',
    'person':       'pedestrian',
    'trailer':      'trailer',
}

# class RandomRotate():
#     def __init__(self, p, angle):
#         self.angle = angle
#         self.p = p

#     def __call__(self, camou2):
#         if np.random.rand(1)>self.p:
#             camou3 = transforms.functional.rotate(camou2, self.angle)
#         else:
#             camou3 = camou2
#         return camou3

# def tex_trans(camou, num_rows=6, num_cols=6, size=4096):
#     """
#     Flip, rotate, and crop the camouflage texture
#     """
#     horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
#     vertical_flip = transforms.RandomVerticalFlip(p=0.5)
#     random_rotate = RandomRotate(p=0.5, angle=90)
#     random_crop = transforms.RandomCrop(size)

#     camou_column = []
#     for i in range(num_cols):
#         camou_row_list = []
#         for j in range(num_rows):
#             camou1 = horizontal_flip(camou.permute(0, 3, 1, 2)[0])
#             camou2 = vertical_flip(camou1)
#             camou3 = random_rotate(camou2)
#             camou_row_list.append(camou3)
#         camou_row = torch.cat(tuple(camou_row_list), 1)
#         camou_column.append(camou_row)
#     camou_full = torch.cat(tuple(camou_column), 2).unsqueeze(0)
#     camou_crop = random_crop(camou_full).permute(0, 2, 3, 1)
#     return camou_crop

def overlay_image(image, mask, texture):
    contour = (mask < 0.5).float().to(device=mask.device)
    overlayed = torch.where(contour == 1., image.to(device=contour.device), texture.to(device=contour.device)).to(device=mask.device)
    return overlayed
            
def mask_imgs(yolo_model, imgs, mask_img, camou_para, allowed_words, device, num_samples = 1, dynamic_check=False, ratio_check=2e-3, debug=False, target_class=None, img_metas=None):
    # # B x 6 x 3 x H x W
    # imgs = data['img'][0].data[0]
    # mask_img = data['masks'][0].data[0]

    B = imgs.shape[0]
    attack_meta = [None] * B

    # normalize image
    range_num = imgs.max() - imgs.min()
    min_num = imgs.min()
    imgs_norm = (imgs - min_num) / (range_num)

    imgs_overlayed = overlay_image(imgs_norm, mask_img, camou_para.permute(0, 3, 1, 2))
    imgs_processed = (imgs_overlayed * range_num) + min_num

    # print(f'[mask_imgs] mask_img shape={mask_img.shape} max={mask_img.max():.4f} min={mask_img.min():.4f}')

    # Derive per-sample attack metadata from the precomputed mask.
    # Priority: nuscenes_class from img_metas (set by loading.py from the mask
    # directory name) → explicit target_class → first allowed_word fallback.
    _fallback_cls = (target_class or
                     (YOLO_TO_NUSCENES.get(allowed_words[0], allowed_words[0])
                      if allowed_words else 'car'))
    try:
        m = mask_img.detach()
        # Collapse any channel dim so m is [B, 6, H, W] or [1, 6, H, W] or [6, H, W].
        while m.dim() > 4:
            m = m[..., 0, :, :]
        if m.dim() == 3:
            m = m.unsqueeze(0).expand(B, -1, -1, -1)
        elif m.shape[0] == 1 and B > 1:
            m = m.expand(B, -1, -1, -1)
        for b in range(B):
            # Resolve the nuScenes class for this sample.
            raw_cls = ''
            if img_metas is not None and b < len(img_metas):
                raw_cls = img_metas[b].get('nuscenes_class', '')
            # print(f'The RAW CLASS is "{raw_cls}"')
            nuscenes_cls = YOLO_TO_NUSCENES.get(raw_cls, raw_cls) if raw_cls else _fallback_cls

            per_cam = m[b]  # [6, H, W]
            areas = (per_cam > 0).float().sum(dim=(1, 2))  # [6]
            if areas.max() > 0:
                cam_idx = int(areas.argmax())
                cam_mask = per_cam[cam_idx] > 0  # [H, W] bool
                rows = cam_mask.any(dim=1)
                cols = cam_mask.any(dim=0)
                y1 = int(rows.nonzero(as_tuple=False)[0])
                y2 = int(rows.nonzero(as_tuple=False)[-1])
                x1 = int(cols.nonzero(as_tuple=False)[0])
                x2 = int(cols.nonzero(as_tuple=False)[-1])
                attack_meta[b] = {
                    'nuscenes_class': nuscenes_cls,
                    'camera_idx': cam_idx,
                    'bbox_2d': [x1, y1, x2, y2],
                }
    except Exception as e:
        import traceback
        print(f'[mask_imgs] exception: {e}')
        traceback.print_exc()

    return [DC([imgs_processed], stack=False, cpu_only=False)], attack_meta
    # H, W = imgs_norm.shape[-2], imgs_norm.shape[-1]  # example shape
    # masks, labels, batches, angles = run_yolo8(
    #         yolo_model, 
    #         imgs_norm, 
    #         H, W, 
    #         device, 
    #         search_labels=allowed_words
    # )
    
    # mask_img = torch.zeros_like(imgs)
    # this_mask = transforms.ToTensor()(Image.open(camera_mask_dir)).to(dtype=int)
    # print(f'JM: {this_mask.shape} {mask_img.shape}')
    # mask_img[:, camera_view, :] = this_mask[0]
    # if len(masks) > 0:
    # if True:
        # masks = torch.stack(masks).to(device=device)
        
        # bboxes = bbox_xyxy_from_mask_torch(masks)
        # areas = (bboxes[:, 2] - bboxes[:, 0])*(bboxes[:, 3] - bboxes[:, 1])
        # total_area = imgs.shape[-2]*imgs.shape[-1]
        # ratio = areas/total_area
        # # Want the largest objects
        # if dynamic_check:
        #     ratio_check = (min(ratio) + max(ratio))/2
        # ratio_indices = (ratio > ratio_check).nonzero(as_tuple=True)[0]


    # directory_name = f"output_imgs"
    # os.makedirs(directory_name, exist_ok=True)
    # print(f"Directory '{directory_name}' created.")
    # print_images(imgs_processed, 0, separate=True, norm=True, name=f'{directory_name}/norm')


        # batches = torch.tensor(batches, device=device)
        # angles = torch.tensor(angles, device=device)
        # if ratio_indices.numel() != 0:
        #     # if debug:
        #     #     plot_masks(masks, labels, batches, angles, imgs, separate=True, name=directory_name+'/test')

        #     # Have to switch to cpu because it does not handle small tensors well 
        #     for i in range(imgs.shape[0]):
        #         t = torch.nonzero(batches == i).flatten()
                
        #         # Filter out the small targets in the batch
        #         ratio_indices = (ratio[t] > ratio_check).nonzero(as_tuple=True)[0]
        #         t_filtered = t[ratio_indices].flatten()
        #         filtered_ratio = ratio[t_filtered]
        #         prob = (filtered_ratio / filtered_ratio.sum())
        #         # FIXME when prob is empty the sum of the probability is 0 and this throws an error
        #         if len(prob) == 0:
        #             continue
                
        #         # Sample from the distribution of target sizes num_samples amount of examples
        #         r_idx = torch.multinomial(prob.cpu(), 1, replacement=True).to(device)

        #         choice = t_filtered[r_idx].to(device=device)
                
        #         assert masks.device == imgs.device, f"devices of mask and model do not match {masks.device} != {imgs.device}"
        #         assert angles.device == imgs.device, f"devices of angles and model do not match {angles.device} != {imgs.device}"
        #         masks_chosen = masks[choice]
        #         angles_chosen = angles[choice]
            
        #         # sum the masks from the same classes so that there are multiple images in one mask
        #         # if num_samples > 1:
        #         #     unique_angles, inverse_indices = torch.unique(angles_chosen, return_inverse=True)
        #         #     num_unique = unique_angles.numel()
        #         #     summed_masks = torch.zeros(num_unique, *masks_chosen.shape[1:], device=masks_chosen.device, dtype=masks_chosen.dtype)
        #         #     summed_masks.scatter_add_(0, inverse_indices.view(-1, 1, 1).expand_as(masks_chosen), masks_chosen)
                    
        #         #     imgs_overlayed = overlay_image(imgs_norm[i, unique_angles, :, :], summed_masks.unsqueeze(1).repeat(1, 3, 1, 1), camou_para.permute(0, 3, 1, 2))
        #         #     imgs_chosen = (imgs_overlayed * range_num) + min_num
        #         #     imgs_processed[i, unique_angles, :, :, :] = imgs_chosen.to(device=device)
                
        #         imgs_overlayed = overlay_image(imgs_norm[i, angles_chosen, :, :], masks_chosen.unsqueeze(1).repeat(1, 3, 1, 1), camou_para.permute(0, 3, 1, 2))
        #         imgs_chosen = (imgs_overlayed * range_num) + min_num
        #         imgs_processed[i, angles_chosen, :, :, :] = imgs_chosen.to(device=device)
        # if debug:
        #     # just print first batch
        #     print_images(imgs_processed.detach(), 0, norm=True, separate=True, name=f'{directory_name}/masked')
            
            
        #     labels = [yolo_model.names[label] for i, label in enumerate(labels) if i in ratio_indices]
        #     batches = [label for i, label in enumerate(batches) if i in ratio_indices]
        #     angles = [label for i, label in enumerate(angles) if i in ratio_indices]
        #     bboxes = bboxes[ratio_indices]
        #     plot_bbox(imgs_norm[batches, angles, :, :][0], [bboxes[0]], [labels[0]], name=f'{directory_name}/bbox')
            
            
        #     # plt.figure()
        #     # plt.subplot(1, 2, 1)
        #     # plt.imshow(imgs_processed[batches, angles, :, :][0].permute(1,2, 0).cpu().detach().numpy())
        #     # plt.title('car')
        #     # plt.axis('off')
        #     # plt.subplot(1, 2, 2)
        #     # plt.imshow(masks[0].cpu().detach().numpy())
        #     # plt.title(f'mask')
        #     # plt.axis('off')
        #     # plt.savefig(f'{directory_name}/overlay.png')
    

def test_attack(model, yolo_model, data_loader, camou_para1, tex_trans, no_attack=False, allowed_words= ['car', 'bicycle', 'person'], cfg=None, img_size=(384, 1056), H=1056, W=1056, resolution=8, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    attack_log_local = {}
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    debug=False

    # Determine the device the model is on so we can move data there.
    _underlying = model.module if hasattr(model, 'module') else model
    _device = next(_underlying.parameters()).device

    def _dc_to_device(val, device):
        """Move a DataContainer's inner tensors to device, in-place."""
        if hasattr(val, '_data') and isinstance(val._data, list) and val._data:
            inner = val._data[0]
            if isinstance(inner, torch.Tensor):
                val._data[0] = inner.to(device)
            elif isinstance(inner, list):
                val._data[0] = [x.to(device) if isinstance(x, torch.Tensor) else x
                                 for x in inner]
        return val

    # When running the no-attack baseline, use MMDataParallel so inference is
    # spread across all GPUs.  The attack path stays on one GPU because the
    # camouflage tensor lives on _device and must propagate gradients there.
    _num_gpus = torch.cuda.device_count()
    _use_dp = (
        no_attack
        and _num_gpus > 1
        and world_size == 1
        and not isinstance(model, (MMDataParallel, MMDistributedDataParallel))
    )
    if _use_dp:
        eval_model = MMDataParallel(_underlying, device_ids=list(range(_num_gpus)))
        eval_model.eval()
    else:
        eval_model = model

    # DistributedSampler with shuffle=False is deterministic; iterate once to
    # get the exact dataset-index for each position in the loader.
    sampler_indices = list(iter(data_loader.sampler))
    batch_size_dl = data_loader.batch_size or 1

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if not no_attack:
                camou_trans = tex_trans(camou_para1.permute(0, 3, 1, 2))

                imgs = data['img'][0].data[0].to(_device)
                mask_img = data['masks'][0].data[0].to(_device)
                _img_metas = data['img_metas'][0].data[0]
                learned_img, attack_meta = mask_imgs(
                    yolo_model, 
                    imgs, 
                    mask_img, 
                    camou_trans, 
                    allowed_words, 
                    device=_device, 
                    dynamic_check=cfg.dynamic_ratio, 
                    ratio_check=cfg.area_ratio, 
                    num_samples=cfg.num_samples, 
                    debug=cfg.debug, 
                    target_class=getattr(cfg, 'target_class', None), 
                    img_metas=_img_metas)
                data['img'] = learned_img

                # Map batch positions to nuScenes sample tokens.
                batch_ds_indices = sampler_indices[i * batch_size_dl : (i + 1) * batch_size_dl]
                for b_idx, info in enumerate(attack_meta):
                    if info is not None:
                        ds_idx = batch_ds_indices[b_idx]
                        token = dataset.data_infos[ds_idx]['token']
                        attack_log_local[token] = info
                        # print(f'attack {type(token)}')

            if _use_dp:
                # MMDataParallel.scatter handles device placement automatically;
                # data stays on CPU and is scattered to each GPU by the framework.
                pass
            else:
                # Single-GPU path: manually move DC tensors to _device before
                # the forward call.  'img' in the no_attack path arrives as
                # [DC(cpu_tensor)] and needs moving; in the attack path it was
                # already replaced with a GPU tensor by mask_imgs above.
                for key, val in data.items():
                    if key == 'img' and no_attack and isinstance(val, list):
                        for dc in val:
                            _dc_to_device(dc, _device)
                    elif key != 'img':
                        _dc_to_device(val, _device)

            result = eval_model(
                return_loss=False,  # FIXME turn this to true and the whole thing explodes
                rescale=True,
                **data
            )
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]

        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)

    # Merge per-GPU attack logs into a single dict on rank 0.
    if world_size > 1:
        # Barrier ensures all ranks have exited the inference loop and completed
        # collect_results_gpu before entering the second collective. Without this,
        # a slow rank can still be inside the loop while a fast rank reaches
        # all_gather_object, causing a collective mismatch / hang.
        torch.distributed.barrier()
        all_logs = [None] * world_size
        torch.distributed.all_gather_object(all_logs, attack_log_local)
        attack_log = {k: v for log in all_logs for k, v in log.items()}
    else:
        attack_log = attack_log_local

    return results, attack_log

def load_camou(camou_path, expand_kernel, device):
    arr = np.load(camou_path)   # e.g., shape (H, W, 3) or any dimensions
    # Convert to torch tensor
    camou_para = torch.from_numpy(arr).to(device)
    camou_para.requires_grad_(True)
    camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    camou_para1 = torch.clamp(camou_para1, 0, 1)
    return camou_para, camou_para1

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--camou', help='')
    parser.add_argument(
        '--test-random',
        type=str,
        default=None)
    parser.add_argument(
        '--no-attack',
        action='store_true',
        help='')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--result_dir', help='directory where results are saved')
    parser.add_argument(
        '--bs',
        type=int,
        default=1,
        help='batch size')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show_bev', action='store_true', help='show bev results')
    parser.add_argument(
        '--show_dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--attack-filter',
        choices=['none', 'sample_class', 'instance'],
        default='none',
        help='Scope evaluation to only the attacked objects. '
             '"none": standard full-split eval (default). '
             '"sample_class": restrict gt/pred to samples where an attack occurred '
             'and only the attacked class. '
             '"instance": additionally projects 3D GT boxes to camera space and '
             'matches against the precomputed mask bbox to isolate the attacked instance.')
    parser.add_argument(
        '--reference-log',
        default=None,
        help='Path to _attack_log.json saved by a prior attack run. '
             'Use with --no-attack + --attack-filter so the no-attack baseline '
             'is evaluated on the exact same samples/instances as the attack run.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
            'results / save the results) with the argument "--out", "--eval"'
            ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

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

    cfg.model.pretrained = None
    cfg.data.test.update(dict(samples_per_gpu=args.bs))

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        cfg.data.workers_per_gpu = 1
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    else:
        import random
        args.seed = int(random.random()*10e7)
        # logger.info(f'Set random seed to {args.seed}, '
        #             f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # export PYTHONPATH=$PYTHONPATH:$(pwd)/IS-Fusion
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    cfg.model.train_cfg = None
    print(f'[DEBUG] Building model...', flush=True)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    print(f'[DEBUG] Model built successfully.', flush=True)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        print(f'[DEBUG] Wrapping fp16 model...', flush=True)
        wrap_fp16_model(model)
        print(f'[DEBUG] fp16 wrap done.', flush=True)
    print(f'[DEBUG] Loading checkpoint from: {cfg.checkpoint}', flush=True)
    checkpoint = load_checkpoint(model, cfg.checkpoint, map_location='cpu')
    print(f'[DEBUG] Checkpoint loaded successfully.', flush=True)
    if args.fuse_conv_bn:
        print(f'[DEBUG] Fusing conv+bn...', flush=True)
        model = fuse_conv_bn(model)
        print(f'[DEBUG] Fuse done.', flush=True)
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
    print(f'[DEBUG] Model classes/palette set. Moving model to GPU...', flush=True)

    def load_words(file_name):
        allowed_words = []
        with open(file_name) as f:
            for line in f:
                line = line.replace('\n', '')
                allowed_words.append(line)
        return allowed_words

    allowed_words = load_words(cfg.allowed_words)


    attack_log = {}
    if not distributed:
        # model = MMDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
        # train_attack_single_gpu(model, data_loader)
        raise ValueError('Error: Single GPU not implemented')
    else:
        print(f'[DEBUG] Moving model to GPU...', flush=True)
        model_on_gpu = model.cuda()
        print(f'[DEBUG] Model on GPU. Wrapping in MMDistributedDataParallel...', flush=True)
        model = MMDistributedDataParallel(
            model_on_gpu,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        print(f'[DEBUG] MMDistributedDataParallel wrap done.', flush=True)
        print(f'[DEBUG] Loading YOLO model...', flush=True)
        yolo_model=YOLO('yolov8n-seg.pt')
        print(f'[DEBUG] YOLO model loaded.', flush=True)
            
        H=1056
        W=1056
        resolution=8
        h, w = int(H/resolution), int(W/resolution)

        ##################################### SETUP Transpose #############################################
        expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(model.device)
        expand_kernel.weight.data.fill_(0)
        expand_kernel.bias.data.fill_(0)
        for i in range(3):
            expand_kernel.weight[i, i, :, :].data.fill_(1)
        ###################################################################################################

        #####################################################################################
        # continuous color
        if cfg.camou_path is None:
            raise ValueError('camou path is None')

        if args.test_random == 'rand':
            camou_para = torch.rand([1, h, w, 3]).float().to(model.device)
            camou_para.requires_grad_(True)
            begin_para = deepcopy(camou_para)
            camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        elif args.test_random == 'img':
            lily_img = transforms.ToTensor()(Image.open('./Lily.jpg').resize((w, h))).float().to(model.device)
            camou_para = lily_img.unsqueeze(dim=0)
            camou_para.requires_grad_(True)
            begin_para = deepcopy(camou_para)
            camou_para1 = expand_kernel(camou_para).permute(0, 2, 3, 1)
        else:
            camou_para, camou_para1 = load_camou(cfg.camou_path, expand_kernel, device=model.device)
        img_size=(384, 1056)
        tex_trans = get_augmentation(img_size)
        #####################################################################################
        outputs, attack_log = test_attack(
            model=model,
            no_attack=args.no_attack,
            yolo_model=yolo_model,
            data_loader=data_loader,
            camou_para1=camou_para1,
            tex_trans=tex_trans,
            allowed_words=allowed_words,
            tmpdir=args.tmpdir,
            gpu_collect=args.gpu_collect,
            cfg=cfg
        )

    rank, _ = get_dist_info()
    if rank == 0:
        # For no-attack baseline runs: substitute the attack log from a prior
        # attack run so the filter scopes to the exact same samples/instances.
        if args.reference_log:
            import json
            with open(args.reference_log) as _f:
                attack_log = json.load(_f)
            print(f'Using reference attack log ({len(attack_log)} samples) '
                  f'from {args.reference_log}')

        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
            if args.attack_filter != 'none' and not args.no_attack:
                log_path = args.out.replace('.pkl', '_attack_log.json')
                if len(attack_log) == 0:
                    raise ValueError('attack_log is blank')
                mmcv.dump(attack_log, log_path)
                print(f'attack log ({len(attack_log)} samples) written to {log_path}')
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            if args.attack_filter != 'none':
                eval_kwargs.update(
                    attack_log=attack_log,
                    attack_filter=args.attack_filter,
                )
            print(dataset.evaluate(outputs, out_dir='./', show=False, **eval_kwargs))


if __name__ == '__main__':
    main()