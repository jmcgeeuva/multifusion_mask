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

def train_attack_multi_gpu(model, yolo_model, data_loader, allowed_words= ['car', 'bicycle', 'person'], cfg=None, img_size=(384, 1056), H=1056, W=1056, resolution=8, tmpdir=None, gpu_collect=False):
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
    camou_para = torch.rand([1, h, w, 3]).float().to(model.device)
    camou_para.requires_grad_(True)
    begin_para = deepcopy(camou_para)
    optimizer = optim.Adam([camou_para], lr=0.01)
    camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    #####################################################################################



    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    debug=True

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            imgs = data['img'][0].data[0]
            camou_trans = tex_trans(camou_para1, size=img_size)
            result = model(
                return_loss=False,  # FIXME turn this to true and the whole thing explodes
                rescale=True, 
                points=data['points'],
                img=mask_imgs(yolo_model, imgs, camou_trans, allowed_words, debug=debug),
                camera_intrinsics=data['camera_intrinsics'],
                camera2ego=data['camera2ego'],
                lidar2ego=data['lidar2ego'],
                lidar2camera=data['lidar2camera'],
                camera2lidar=data['camera2lidar'],
                lidar2img=data['lidar2img'],
                img_aug_matrix=data['img_aug_matrix'],
                lidar_aug_matrix=data['lidar_aug_matrix'],
                img_metas=data['img_metas']
            )
            # result = model(return_loss=False, rescale=True, **data)
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
    return results

            
def mask_imgs(yolo_model, imgs, camou_para, allowed_words, ratio_check=2e-3, debug=False):
    
    if debug:
        print_images(imgs, 0, norm=False, name='dark.png')
        print_images(imgs, 0, norm=True, name='norm.png')
    
    range_num = imgs.max() - imgs.min()
    min_num = imgs.min()
    imgs_norm = (imgs - min_num) / (range_num)
    # imgs = torch.nn.functional(img, size=())
    mask_entry, labels, vids = run_yolo8(
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
                plot_masks(masks, labels, vids, imgs[0])
            
            
            H, W = imgs_norm[0].shape[-2], imgs_norm[0].shape[-1]  # example shape
            # green = torch.zeros(3, H, W)
            # green[1, :, :] = 1.0  # set G channel to 1
            # lily_img = Image.open('./Lily.jpg').resize((imgs_norm[0].shape[-1], imgs_norm[0].shape[-2]))
            # lily_img = transforms.ToTensor()(lily_img)
            imgs_overlayed = overlay_image(imgs_norm[0][vids], masks.unsqueeze(1).repeat(1, 3, 1, 1), camou_para.permute(0, 3, 1, 2))

            # unnormalize
            imgs_processed = (imgs_overlayed * range_num) + min_num
            imgs_processed = imgs_processed[ratio_indices].to(device=imgs.device)
            bboxes = bboxes[ratio_indices]

            prob = ratio[ratio_indices] / ratio[ratio_indices].sum()
            # Have to switch to cpu because it does not handle small tensors well
            r_idx = torch.multinomial(prob.cpu(), 1, replacement=True).to(prob.device)
            choice = ratio_indices[r_idx]
            imgs[:, vids[choice], :, :, :] = imgs_processed[r_idx]

        if debug:
            print_images(imgs, 0, norm=True, name='masked.png')
            labels = [yolo_model.names[label] for i, label in enumerate(labels) if i in ratio_indices]
            vids = [label for i, label in enumerate(vids) if i in ratio_indices]
            plot_bbox(imgs_norm[0][vids][0], [bboxes[0]], [labels[0]])
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(imgs_processed[0].permute(1,2, 0).cpu().detach().numpy())
            plt.title('car')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(masks[0].cpu().detach().numpy())
            plt.title(f'mask')
            plt.axis('off')
            plt.savefig('overlay.png')
    return [DC([imgs], stack=False, cpu_only=False)]

def train_attack_single_gpu(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    ratio_check=2e-3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    yolo_model=YOLO('yolov8n-seg.pt')
    yolo_model.to(device=model.device)

    # yolo_model.eval()
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    debug = False
    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = model(
                return_loss=False, 
                rescale=True, 
                points=data['points'],
                img=mask_imgs(yolo_model, data, camou_para, debug=debug),
                camera_intrinsics=data['camera_intrinsics'],
                camera2ego=data['camera2ego'],
                lidar2ego=data['lidar2ego'],
                lidar2camera=data['lidar2camera'],
                camera2lidar=data['camera2lidar'],
                lidar2img=data['lidar2img'],
                img_aug_matrix=data['img_aug_matrix'],
                lidar_aug_matrix=data['lidar_aug_matrix'],
                img_metas=data['img_metas']
            )

        if show:
            model.module.show_results(data, result, out_dir)

        if len(result) > 1:
            results.append(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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
    parser.add_argument('--seed', type=int, default=0, help='random seed')
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
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
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


    if not distributed:
        model = MMDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
        train_attack_single_gpu(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        yolo_model=YOLO('yolov8n-seg.pt')
        outputs = train_attack_multi_gpu(
            model=model,
            yolo_model=yolo_model, 
            data_loader=data_loader,
            allowed_words=allowed_words, 
            tmpdir=args.tmpdir,
            gpu_collect=args.gpu_collect, 
            cfg=cfg
        )

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
            # outputs = mmcv.load(args.out)
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
            if args.result_dir is not None:
                eval_kwargs.update(pklfile_prefix=os.path.dirname(args.result_dir))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()