#!/usr/bin/env python3
"""
Visualize detection results using IS-Fusion's own pipeline and visualization code.

The data loader re-runs the EXACT same preprocessing that produced the results
pkl, so the lidar→image projection matrices and the rendered images are always
pixel-perfectly aligned.  No manual matrix reconstruction needed.

Usage
-----
# Single results file (baseline or attack):
python visualize_results.py config/best.py results.pkl --show-dir vis_out

# Side-by-side comparison (baseline vs attack):
python visualize_results.py config/best.py results.pkl \
    --compare baseline_attack.pkl \
    --show-dir vis_comparison

Options:
  --score-thr    Detection confidence threshold  (default 0.2)
  --max-samples  Max number of samples to render (default 50)
  --token        Render only one specific sample token
  --classes      Space-separated class names to highlight in a different colour
"""

import argparse
import os
import pickle
import sys

import cv2
import mmcv
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IS-Fusion'))

from mmcv import Config
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.utils.visualize import visualize_camera
from mmdet3d.datasets import build_dataloader, build_dataset

# ── constants ──────────────────────────────────────────────────────────────────
CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# ImageNormalize parameters from config (mean/std in 0-1 range)
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Camera order in IS-Fusion (matches the order images are stacked in data['img'])
CAM_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK',  'CAM_BACK_LEFT',   'CAM_BACK_RIGHT',
]


# ── helpers ────────────────────────────────────────────────────────────────────

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def denorm_img(chw_tensor):
    """Normalized CHW float tensor → HWC uint8 RGB."""
    img = chw_tensor.cpu().float().numpy().transpose(1, 2, 0)   # HWC
    img = (img * _IMG_STD + _IMG_MEAN) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def unpack_dc(val):
    """Pull the inner tensor/object out of a mmcv DataContainer (or list of one).

    Handles all common nesting patterns produced by the DataLoader collate:
      - DC directly            → .data[0]
      - [DC]  (list of one)   → val[0].data[0]
      - [[DC]]                → val[0][0].data[0]  (unlikely but safe)
    """
    while isinstance(val, list):
        val = val[0]
    if hasattr(val, 'data'):
        inner = val.data
        if isinstance(inner, list):
            return inner[0]
        return inner
    return val


def get_result_boxes(result, score_thr, keep_labels=None):
    """Return (LiDARInstance3DBoxes, labels_np) after score + optional label filter."""
    pb     = result.get('pts_bbox', result)
    boxes  = pb['boxes_3d']
    scores = pb['scores_3d'].cpu().numpy()
    labels = pb['labels_3d'].cpu().numpy().astype(int)

    mask = scores >= score_thr
    if keep_labels is not None:
        mask &= np.isin(labels, keep_labels)

    return boxes[mask], labels[mask]


# ── per-sample render ──────────────────────────────────────────────────────────

def render_sample(
        data, result_a, result_b,
        sample_dir, score_thr,
        highlight_labels=None,
        label_a='RESULT', label_b='COMPARE'):
    """
    Render all six cameras for one sample.

    result_a  – primary results (always rendered)
    result_b  – optional comparison results (None → single-result mode)
    """
    os.makedirs(sample_dir, exist_ok=True)

    # ── unpack data ────────────────────────────────────────────────────────
    # Access patterns confirmed from nusc_vis_pred.py and test.py:
    #   data['img'][0].data[0]           → tensor (batch, n_cam, 3, H, W)
    #   data['img_metas'][0].data[0]     → list of meta dicts (one per sample)
    #   data['lidar2img']                → DC or [DC], unpacked via unpack_dc
    imgs_tensor   = unpack_dc(data['img'])          # (1, N_cam, 3, H, W)
    l2i_tensor    = unpack_dc(data['lidar2img'])    # (1, N_cam, 4, 4)  or (N_cam, 4, 4)
    img_metas_raw = unpack_dc(data['img_metas'])    # list[dict] or dict
    # img_metas_raw is a list of dicts (one per sample in batch)
    meta = img_metas_raw[0] if isinstance(img_metas_raw, list) else img_metas_raw

    # Squeeze batch dim if present
    if imgs_tensor.dim() == 5:
        imgs_tensor = imgs_tensor[0]          # (N_cam, 3, H, W)
    if l2i_tensor.dim() == 3:
        pass                                  # already (N_cam, 4, 4)
    elif l2i_tensor.dim() == 4:
        l2i_tensor = l2i_tensor[0]           # (N_cam, 4, 4)

    n_cams = imgs_tensor.shape[0]

    boxes_a, labels_a = get_result_boxes(result_a, score_thr, highlight_labels)
    if result_b is not None:
        boxes_b, labels_b = get_result_boxes(result_b, score_thr, highlight_labels)

    cam_imgs_a = []
    cam_imgs_b = []

    for cam_i in range(n_cams):
        img_rgb = denorm_img(imgs_tensor[cam_i])          # HWC uint8 RGB
        l2i     = l2i_tensor[cam_i]                       # (4,4) tensor

        cam_name = CAM_NAMES[cam_i] if cam_i < len(CAM_NAMES) else f'cam{cam_i}'
        fname_a  = os.path.join(sample_dir, f'{cam_name}_{label_a}.jpg')

        visualize_camera(
            fname_a, img_rgb,
            bboxes=boxes_a if len(boxes_a) > 0 else None,
            labels=labels_a if len(boxes_a) > 0 else None,
            transform=l2i,
            classes=CLASSES,
        )
        panel_a = cv2.imread(fname_a)
        _add_label(panel_a, f'{label_a} | {cam_name}', color=(0, 200, 0))
        cv2.imwrite(fname_a, panel_a)
        cam_imgs_a.append(panel_a)

        if result_b is not None:
            fname_b = os.path.join(sample_dir, f'{cam_name}_{label_b}.jpg')
            visualize_camera(
                fname_b, img_rgb,
                bboxes=boxes_b if len(boxes_b) > 0 else None,
                labels=labels_b if len(boxes_b) > 0 else None,
                transform=l2i,
                classes=CLASSES,
            )
            panel_b = cv2.imread(fname_b)
            _add_label(panel_b, f'{label_b} | {cam_name}', color=(0, 0, 220))
            cv2.imwrite(fname_b, panel_b)
            cam_imgs_b.append(panel_b)

    # ── composite surrounding-view image ──────────────────────────────────
    _write_surround(cam_imgs_a, os.path.join(sample_dir, f'surround_{label_a}.jpg'))
    if result_b is not None:
        _write_surround(cam_imgs_b, os.path.join(sample_dir, f'surround_{label_b}.jpg'))
        # side-by-side top row: front cameras; bottom row: back cameras
        _write_side_by_side(cam_imgs_a, cam_imgs_b, label_a, label_b,
                            os.path.join(sample_dir, 'comparison.jpg'))

    return len(boxes_a)


def _add_label(img, text, color=(255, 255, 255)):
    """Burn a small text label into the top-left of img in-place."""
    if img is None:
        return
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)


def _write_surround(cam_imgs, path):
    """Write a 2-row × 3-column surround-view composite."""
    if len(cam_imgs) < 6 or any(img is None for img in cam_imgs[:6]):
        return
    top    = np.hstack(cam_imgs[:3])
    bottom = np.hstack(cam_imgs[3:])
    cv2.imwrite(path, np.vstack([top, bottom]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def _write_side_by_side(imgs_a, imgs_b, label_a, label_b, path):
    """Write a side-by-side comparison: left=imgs_a, right=imgs_b."""
    if len(imgs_a) < 6 or len(imgs_b) < 6:
        return
    if any(img is None for img in imgs_a[:6] + imgs_b[:6]):
        return

    def surround(imgs):
        top    = np.hstack(imgs[:3])
        bottom = np.hstack(imgs[3:])
        return np.vstack([top, bottom])

    left  = surround(imgs_a)
    right = surround(imgs_b)

    # Annotate panel headers
    for panel, txt, col in [(left, label_a, (0, 200, 0)), (right, label_b, (0, 0, 220))]:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(panel, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, col, 2, cv2.LINE_AA)

    cv2.imwrite(path, np.hstack([left, right]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


# ── arg parsing ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Visualize IS-Fusion detection results')
    p.add_argument('config',  help='Config file (e.g. config/best.py)')
    p.add_argument('results', help='Primary results pkl (e.g. results.pkl)')
    p.add_argument('--compare', default=None,
                   help='Optional second results pkl for side-by-side comparison '
                        '(e.g. baseline_attack.pkl)')
    p.add_argument('--show-dir',    default='vis_out',
                   help='Output directory  (default: vis_out)')
    p.add_argument('--score-thr',   type=float, default=0.2,
                   help='Detection score threshold  (default: 0.2)')
    p.add_argument('--max-samples', type=int,   default=50,
                   help='Max samples to render  (default: 50)')
    p.add_argument('--token',       default=None,
                   help='Render only this sample token')
    p.add_argument('--classes', nargs='+', default=None,
                   help='Only show these class names (space-separated). '
                        'Default: all classes.')
    p.add_argument('--result-label',  default='RESULT',
                   help='Label for the primary results  (default: RESULT)')
    p.add_argument('--compare-label', default='COMPARE',
                   help='Label for the comparison results  (default: COMPARE)')
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.show_dir, exist_ok=True)

    # ── load results ──────────────────────────────────────────────────────
    print(f'Loading {args.results} ...')
    results_a = load_pkl(args.results)
    print(f'  {len(results_a)} entries')

    results_b = None
    if args.compare:
        print(f'Loading {args.compare} ...')
        results_b = load_pkl(args.compare)
        print(f'  {len(results_b)} entries')

    # ── build dataset ─────────────────────────────────────────────────────
    print('Building dataset from config ...')
    cfg = Config.fromfile(args.config)
    # Ensure test mode and single-sample batching
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=min(cfg.data.get('workers_per_gpu', 2), 4),
        dist=False,
        shuffle=False,
    )
    print(f'  {len(dataset)} samples in dataset')

    # Sanity check: warn if sizes disagree (indicates misaligned pkl)
    for tag, res in [(args.result_label, results_a),
                     (args.compare_label, results_b)]:
        if res is not None and len(res) != len(dataset):
            print(f'  WARNING: {tag} pkl has {len(res)} entries but dataset '
                  f'has {len(dataset)} samples — indices will be misaligned!')

    # ── optional class filter ─────────────────────────────────────────────
    keep_labels = None
    if args.classes:
        keep_labels = [CLASSES.index(c) for c in args.classes if c in CLASSES]
        unknown = [c for c in args.classes if c not in CLASSES]
        if unknown:
            print(f'  WARNING: unknown classes ignored: {unknown}')
        print(f'  Filtering to classes: {args.classes}  → label ids {keep_labels}')

    # ── optional token filter ─────────────────────────────────────────────
    target_token = args.token

    # ── iterate ───────────────────────────────────────────────────────────
    rendered = 0
    for i, data in enumerate(data_loader):
        if rendered >= args.max_samples:
            break

        # Retrieve the token for this sample so we can name the output folder
        # and support --token filtering.
        # Use dataset.data_infos for the ground-truth token (guaranteed correct),
        # and fall back to pts_filename from img_metas.
        if hasattr(dataset, 'data_infos') and i < len(dataset.data_infos):
            token = dataset.data_infos[i].get('token', str(i))
        else:
            img_metas = unpack_dc(data['img_metas'])
            meta_list = img_metas if isinstance(img_metas, list) else [img_metas]
            meta      = meta_list[0]
            pts_fname = meta.get('pts_filename', '') or meta.get('lidar_path', '')
            token = os.path.basename(pts_fname).split('.')[0] if pts_fname else str(i)

        if target_token and token != target_token:
            continue

        result_a = results_a[i]
        result_b = results_b[i] if results_b is not None else None

        sample_dir = os.path.join(args.show_dir, f'{i:05d}_{str(token)[:32]}')

        n_boxes = render_sample(
            data, result_a, result_b,
            sample_dir=sample_dir,
            score_thr=args.score_thr,
            highlight_labels=keep_labels,
            label_a=args.result_label,
            label_b=args.compare_label,
        )

        print(f'  [{rendered + 1:>4}] sample {i:>5}  token={str(token)[:32]}'
              f'  boxes≥{args.score_thr}={n_boxes}  → {sample_dir}/')
        rendered += 1

    print(f'\nDone. Rendered {rendered} sample(s) to {args.show_dir}/')


if __name__ == '__main__':
    main()
