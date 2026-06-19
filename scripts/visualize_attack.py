#!/usr/bin/env python3
"""
Unit-test visualizer for the adversarial texture pipeline.

Iterates through val samples, runs each one through the exact same transforms
used in training/eval (LoadMultiViewImageFromFilesV2_Camou → ImageAug3D_Camou
→ ImageNormalize_Camou → mask_imgs / tex_trans), then saves a 3-row × 6-column
PNG for each sample:
  Row 0 — original camera images (denormalized)
  Row 1 — images with adversarial texture applied (+ red mask overlay)
  Row 2 — raw binary mask per camera (hot colormap)

The attacked camera column is highlighted in red.  Press Enter to advance to
the next sample, or type 'q' + Enter to quit.

Usage (run from the multifusion_mask/ directory):
    python scripts/visualize_attack.py
    python scripts/visualize_attack.py --camou ./workdir/20251111_215729/2camou.npy
    python scripts/visualize_attack.py --out-dir ./viz_out --start 50
"""

import sys
import os

# Make sure parent dir and IS-Fusion are importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'IS-Fusion'))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')            # non-interactive — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mmcv import Config
from mmdet3d.datasets import build_dataloader, build_dataset
from mmcv.parallel import DataContainer as DC

from test import mask_imgs, load_camou
from augmentation import tex_trans

# ── constants ────────────────────────────────────────────────────────────────

CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_BACK',
    'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK_RIGHT',
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── helpers ───────────────────────────────────────────────────────────────────

def denorm(tensor):
    """(3,H,W) float tensor in ImageNet-normalized space → (H,W,3) uint8."""
    img = tensor.detach().cpu().float().numpy()   # (3,H,W)
    img = img.transpose(1, 2, 0)                  # (H,W,3)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def mask_to_rgba(mask_tensor):
    """
    mask_tensor: (3,H,W) float in [0,1] — all three channels are identical
                 (compose_camou does repeat(3,1,1)).
    Returns (H,W,4) RGBA for overlay: red where mask>0.5, transparent elsewhere.
    """
    m = mask_tensor[0].detach().cpu().float().numpy()   # (H,W)
    rgba = np.zeros((*m.shape, 4), dtype=np.float32)
    rgba[..., 0] = 1.0      # R
    rgba[..., 3] = (m > 0.5).astype(np.float32) * 0.45   # alpha
    return rgba


def make_expand_kernel(resolution, device):
    k = torch.nn.ConvTranspose2d(3, 3, resolution,
                                  stride=resolution, padding=0).to(device)
    k.weight.data.fill_(0)
    k.bias.data.fill_(0)
    for i in range(3):
        k.weight[i, i, :, :].data.fill_(1)
    return k


def build_camou(camou_path, resolution, img_scale, device):
    """Return camou_para1 tensor (1, H, W, 3) ready for tex_trans."""
    expand_kernel = make_expand_kernel(resolution, device)
    H, W = img_scale

    if camou_path and os.path.isfile(camou_path):
        print(f'Loading camou: {camou_path}')
        _, camou_para1 = load_camou(camou_path, expand_kernel, device)
    else:
        h, w = H // resolution, W // resolution
        print(f'No --camou given — using random texture ({h}×{w} base, '
              f'expanded to {H}×{W})')
        camou_para = torch.rand([1, h, w, 3], device=device)
        with torch.no_grad():
            camou_para1 = expand_kernel(
                camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            camou_para1 = torch.clamp(camou_para1, 0, 1)

    return camou_para1


# ── display ───────────────────────────────────────────────────────────────────

def save_sample(orig_imgs, atk_imgs, raw_masks,
                camera_idx, sample_idx, out_path, token='', obj_class='',
                size_ratio=None):
    """
    Render a 3-row × 6-col grid and save it to out_path (PNG).
      Row 0 — original camera images
      Row 1 — attacked images + semi-transparent red mask overlay
      Row 2 — raw binary mask per camera (hot colormap)
    The attacked camera column is outlined in red.
    """
    fig = plt.figure(figsize=(26, 10))

    title = f'Sample {sample_idx}'
    if token:
        title += f'   token: {str(token)[:16]}…'
    if camera_idx >= 0:
        title += f'   attacked: {CAMERA_NAMES[camera_idx]}'
        if obj_class:
            title += f'   label: {obj_class}'
        if size_ratio is not None:
            title += f'   size: {size_ratio * 100:.2f}% of frame'
    else:
        title += '   [no mask found — nothing attacked]'
    fig.suptitle(title, fontsize=11, y=0.99)

    gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.06, wspace=0.02,
                           top=0.95, bottom=0.03, left=0.04)

    for col in range(6):
        attacked = (col == camera_idx)

        # row 0 — original
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(orig_imgs[col])
        ax0.axis('off')
        ax0.set_title(CAMERA_NAMES[col], fontsize=7, pad=2,
                      color='red' if attacked else 'black',
                      fontweight='bold' if attacked else 'normal')

        # row 1 — attacked + optional mask overlay
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(atk_imgs[col])
        if attacked and camera_idx >= 0:
            ax1.imshow(mask_to_rgba(raw_masks[col]))
        ax1.axis('off')

        # row 2 — raw mask (hot colormap)
        ax2 = fig.add_subplot(gs[2, col])
        ax2.imshow(raw_masks[col][0].detach().cpu().numpy(),
                   cmap='hot', vmin=0, vmax=1)
        ax2.axis('off')

        # red border on attacked column
        if attacked:
            for ax in (ax0, ax1, ax2):
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2.5)

    for row, label in enumerate(['Original', 'Attacked', 'Mask']):
        fig.text(0.01, 0.79 - row * 0.315, label,
                 va='center', rotation='vertical', fontsize=9, color='gray')

    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', default='config/only_vehicles.py',
                        help='mmdet3d config (default: %(default)s)')
    parser.add_argument('--camou', default=None,
                        help='Path to saved camou .npy.  Random texture if omitted.')
    parser.add_argument('--resolution', type=int, default=8,
                        help='Expand-kernel stride used during training (default: 8)')
    parser.add_argument('--device', default='cpu',
                        help='Torch device for texture/overlay (default: cpu; '
                             'no model is loaded so GPU is optional)')
    parser.add_argument('--start', type=int, default=0,
                        help='Skip to this sample index (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for random texture (default: 42)')
    parser.add_argument('--out-dir', default='./viz_out',
                        help='Directory to save PNG files (default: %(default)s)')
    parser.add_argument("--print-separate", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    cfg = Config.fromfile(args.config)
    img_scale = tuple(cfg.img_scale)   # (H, W) e.g. (384, 1056)

    # ── dataset & loader ─────────────────────────────────────────────────────
    print('Building val dataset …')
    val_dataset = build_dataset(cfg.data.val)
    val_loader  = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,   # no sub-processes — easier to ctrl-c and step
        num_gpus=1,
        dist=False,
        shuffle=False,
    )
    print(f'Val dataset: {len(val_dataset)} samples')

    # ── output directory ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Saving PNGs to {os.path.abspath(args.out_dir)}/')

    # ── camouflage texture ────────────────────────────────────────────────────
    camou_para1 = build_camou(args.camou, args.resolution, img_scale, device)

    # ── iterate ───────────────────────────────────────────────────────────────
    for sample_idx, data in enumerate(val_loader):
        if sample_idx < args.start:
            continue

        # unwrap DataContainers — test pipeline wraps in [DC] for TTA
        imgs_dc  = data['img'][0]     # DC after MultiScaleFlipAug3D
        masks_dc = data['masks'][0]   # DC

        imgs  = imgs_dc.data[0].to(device)    # (1, 6, 3, H, W)
        masks = masks_dc.data[0].to(device)   # (1, 6, 3, H, W)

        # ── apply texture ─────────────────────────────────────────────────────
        size = (img_scale[0], img_scale[1])
        with torch.no_grad():
            camou_trans = tex_trans(camou_para1, size=size)   # (1, H, W, 3)
            learned_dc_list, attack_meta = mask_imgs(
                yolo_model=None,      # yolo path is fully commented-out
                imgs=imgs,
                mask_img=masks,
                camou_para=camou_trans,
                allowed_words=['car'],
                device=device,
                num_samples=1,
                dynamic_check=False,
                ratio_check=0.0,
                debug=False,
                target_class=getattr(cfg, 'target_class', 'car'),
            )

        # mask_imgs returns [DC([imgs_processed])]; unwrap to (1,6,3,H,W)
        atk_imgs_tensor = learned_dc_list[0].data[0]   # (1, 6, 3, H, W)

        # ── identify attacked camera ──────────────────────────────────────────
        meta = attack_meta[0]
        camera_idx = meta['camera_idx'] if meta is not None else -1

        # ── build display arrays ──────────────────────────────────────────────
        orig_imgs = [denorm(imgs[0, c])        for c in range(6)]
        atk_list  = [denorm(atk_imgs_tensor[0, c]) for c in range(6)]
        raw_masks = [masks[0, c]               for c in range(6)]   # (3,H,W) tensors

        # ── token for title ───────────────────────────────────────────────────
        try:
            token = data['img_metas'][0].data[0][0].get('sample_idx', '')
        except Exception:
            token = ''

        # ── save figure ───────────────────────────────────────────────────────
        if args.print_separate:
            out_path = os.path.join(args.out_dir, f'sample_{sample_idx:05d}.png')
        else:
            out_path = os.path.join(args.out_dir, f'sample.png')
        obj_class = meta.get('nuscenes_class', '') if meta else ''
        if camera_idx >= 0:
            m = raw_masks[camera_idx][0]   # (H,W) float in [0,1]
            size_ratio = (m > 0.5).float().sum().item() / m.numel()
        else:
            size_ratio = None
        save_sample(orig_imgs, atk_list, raw_masks,
                    camera_idx, sample_idx, out_path,
                    token=token, obj_class=obj_class, size_ratio=size_ratio)

        # ── terminal summary + pause ──────────────────────────────────────────
        bbox = meta['bbox_2d'] if meta else None
        meta_str = (f"  camera={CAMERA_NAMES[camera_idx]}  "
                    f"class={meta.get('nuscenes_class', '?')}  "
                    f"bbox={bbox}") if meta else '  (no attack)'
        print(f'\n[{sample_idx}/{len(val_dataset)}]{meta_str}')
        print(f'  saved → {out_path}')

        try:
            inp = input('  Enter=next   q=quit > ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if inp == 'q':
            break

    print('Done.')


if __name__ == '__main__':
    main()
