#!/usr/bin/env python3
"""
Side-by-side visualization of baseline vs. attack detections.

For each sample in the attack log, renders two panels:
  LEFT  – baseline predictions on the attacked camera image
  RIGHT – attack predictions on the same image

Predicted boxes of the attacked class are highlighted.
The specific GT box that was attacked is drawn in magenta on both panels.
Also writes a BEV (top-down) comparison panel per sample.

Usage
-----
python visualize_attack.py \
    --baseline  baseline_results.pkl \
    --attack    attack_results.pkl \
    --attack-log results_attack_log.json \
    --info-file /data/nuscenes/nuscenes_infos_val.pkl \
    --data-root /data/nuscenes \
    --out-dir   vis_comparison \
    [--score-thr 0.2] \
    [--max-samples 50] \
    [--token <specific_sample_token>]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IS-Fusion'))

# nuScenes detection class names in IS-Fusion label-index order
CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# Camera channels in the order IS-Fusion iterates them in the data pkl
CAM_CHANNELS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

# Colour scheme
BASELINE_COLOR    = (0, 200, 0)    # BGR – baseline detections
ATTACK_COLOR      = (0, 0, 220)    # BGR – attack detections
HIGHLIGHT_COLOR   = (0, 165, 255)  # BGR – attacked class boxes
ATTACKED_GT_COLOR = (255, 0, 255)  # BGR – magenta for the specific attacked GT box

# IS-Fusion preprocessing constants (resize then center-crop to 1056×384)
_RESIZE = 0.72
_CROP_X = 48    # left crop offset after resize
_CROP_Y = 264   # top crop offset after resize


# ─── geometry helpers ────────────────────────────────────────────────────────

def lidar_box_corners(box_np):
    """Return (N, 8, 3) corners using IS-Fusion's LiDARInstance3DBoxes.

    box_np: (N, 7+) – prediction boxes in bottom-center convention
    (origin (0.5, 0.5, 0) per LiDARInstance3DBoxes default).
    """
    import torch
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    arr = box_np[:, :7].astype(np.float32)
    boxes = LiDARInstance3DBoxes(torch.from_numpy(arr))
    return boxes.corners.numpy()  # (N, 8, 3)


def gt_box_corners(box_np):
    """Return (N, 8, 3) corners for GT boxes loaded from data_infos.

    data_infos gt_boxes are stored with gravity-center z, so origin=(0.5,0.5,0.5).
    """
    import torch
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    arr = box_np[:, :7].astype(np.float32)
    boxes = LiDARInstance3DBoxes(torch.from_numpy(arr), origin=(0.5, 0.5, 0.5))
    return boxes.corners.numpy()  # (N, 8, 3)


def project_corners_to_img(corners_3d, lidar2img):
    """Project (N,8,3) LiDAR corners to image pixel coords.

    Returns (N,8,2) and a boolean mask (N,) indicating boxes where at
    least one corner projects in front of the camera.
    """
    N = corners_3d.shape[0]
    pts = corners_3d.reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=-1)
    proj = (lidar2img @ pts_h.T).T            # (N*8, 4)
    depth = proj[:, 2].reshape(N, 8)
    proj[:, 2] = np.clip(proj[:, 2], 1e-5, 1e9)
    px = (proj[:, 0] / proj[:, 2]).reshape(N, 8)
    py = (proj[:, 1] / proj[:, 2]).reshape(N, 8)
    pts_2d = np.stack([px, py], axis=-1)      # (N,8,2)
    valid = (depth > 0).any(axis=1)           # at least one corner in front
    return pts_2d, valid


def draw_box_on_img(img, corners_2d, color, thickness=2):
    """Draw one 3-D box (given 8 projected 2-D corners) onto img in-place."""
    EDGES = [(0,1),(1,2),(2,3),(3,0),   # bottom face
             (4,5),(5,6),(6,7),(7,4),   # top face
             (0,4),(1,5),(2,6),(3,7)]   # verticals
    H, W = img.shape[:2]
    for a, b in EDGES:
        pa = tuple(np.round(corners_2d[a]).astype(int))
        pb = tuple(np.round(corners_2d[b]).astype(int))
        # skip edges where both endpoints are far off-screen
        if (max(pa[0], pb[0]) < -W or min(pa[0], pb[0]) > 2*W or
                max(pa[1], pb[1]) < -H or min(pa[1], pb[1]) > 2*H):
            continue
        cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


def render_detections_on_img(raw_img, boxes_tensor, scores, labels,
                             lidar2img, score_thr,
                             default_color, highlight_label=None,
                             highlight_color=None,
                             attacked_gt_corners=None):
    """Return a copy of raw_img with all detection boxes drawn.

    attacked_gt_corners: (8, 3) LiDAR corners of the specific GT box that was
    attacked. Drawn in magenta with extra thickness. None to skip.
    """
    img = raw_img.copy()
    H, W = img.shape[:2]

    # Draw the attacked GT box first (so it renders beneath prediction boxes)
    if attacked_gt_corners is not None:
        gt_arr = attacked_gt_corners[np.newaxis]  # (1, 8, 3)
        gt_2d, gt_valid = project_corners_to_img(gt_arr, lidar2img)
        if gt_valid[0]:
            draw_box_on_img(img, gt_2d[0], ATTACKED_GT_COLOR, thickness=4)
            mid_raw = gt_2d[0][4:6].mean(axis=0)
            if -W < mid_raw[0] < 2*W and -H < mid_raw[1] < 2*H:
                mid_x = int(np.clip(mid_raw[0], 2, W - 2))
                mid_y = int(np.clip(mid_raw[1], 12, H - 2))
                cv2.putText(img, 'ATTACKED GT',
                            (mid_x, mid_y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            ATTACKED_GT_COLOR, 2, cv2.LINE_AA)

    if len(boxes_tensor) == 0:
        return img

    mask = scores >= score_thr
    if mask.sum() == 0:
        return img

    bt = boxes_tensor[mask]
    sc = scores[mask]
    lb = labels[mask]

    corners = lidar_box_corners(bt)
    pts_2d, valid = project_corners_to_img(corners, lidar2img)

    for i in range(len(bt)):
        if not valid[i]:
            continue
        is_highlight = (highlight_label is not None and lb[i] == highlight_label)
        color = highlight_color if is_highlight else default_color
        thick  = 3 if is_highlight else 2
        draw_box_on_img(img, pts_2d[i], color, thickness=thick)

        # Score label near the top-front edge, clipped to image bounds
        mid_raw = pts_2d[i][4:6].mean(axis=0)
        if -W < mid_raw[0] < 2*W and -H < mid_raw[1] < 2*H:
            mid_x = int(np.clip(mid_raw[0], 2, W - 2))
            mid_y = int(np.clip(mid_raw[1], 12, H - 2))
            txt = f'{CLASSES[lb[i]]} {sc[i]:.2f}'
            font_scale = 0.5 if is_highlight else 0.4
            cv2.putText(img, txt,
                        (mid_x, mid_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1,
                        cv2.LINE_AA)
    return img


# ─── BEV helper ──────────────────────────────────────────────────────────────

def bev_box_polygon(box_row):
    """Return (5,2) x-y polygon (closed) for a LiDAR box in BEV."""
    cx, cy, dx, dy, yaw = box_row[0], box_row[1], box_row[3], box_row[4], box_row[6]
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners_local = np.array([
        [ dx/2,  dy/2],
        [ dx/2, -dy/2],
        [-dx/2, -dy/2],
        [-dx/2,  dy/2],
    ])
    corners = (R @ corners_local.T).T + np.array([cx, cy])
    return np.vstack([corners, corners[0]])   # close the polygon


def render_bev(baseline_boxes, baseline_scores, baseline_labels,
               attack_boxes,   attack_scores,   attack_labels,
               attacked_label, score_thr, attacked_gt_box=None,
               pc_range=(-54, -54, 54, 54)):
    """Return a matplotlib Figure comparing BEV detections."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    titles = ['Baseline', 'Attack']
    all_boxes  = [baseline_boxes,  attack_boxes]
    all_scores = [baseline_scores, attack_scores]
    all_labels = [baseline_labels, attack_labels]

    for ax, title, boxes, scores, labels in zip(
            axes, titles, all_boxes, all_scores, all_labels):
        ax.set_xlim(pc_range[0], pc_range[2])
        ax.set_ylim(pc_range[1], pc_range[3])
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')
        ax.set_title(title, fontsize=14, color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        # Draw attacked GT box first so predictions render on top
        if attacked_gt_box is not None:
            poly = bev_box_polygon(attacked_gt_box)
            ax.plot(poly[:, 0], poly[:, 1], color='#ff00ff',
                    linewidth=3, linestyle='--')

        if len(boxes) == 0:
            continue
        mask = scores >= score_thr
        if mask.sum() == 0:
            continue

        for box_row, lbl in zip(boxes[mask], labels[mask]):
            is_attacked = (lbl == attacked_label)
            color = '#ff4444' if is_attacked else '#44ff44'
            poly = bev_box_polygon(box_row)
            ax.plot(poly[:, 0], poly[:, 1], color=color,
                    linewidth=2 if is_attacked else 1)

    legend_elements = [
        Line2D([0], [0], color='#ff4444', linewidth=2, label='Attacked class'),
        Line2D([0], [0], color='#44ff44', linewidth=1, label='Other classes'),
        Line2D([0], [0], color='#ff00ff', linewidth=3, linestyle='--', label='Attacked GT'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               facecolor='#1a1a2e', labelcolor='white', fontsize=10)
    fig.patch.set_facecolor('#1a1a2e')
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ─── data loading ────────────────────────────────────────────────────────────

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_token_index(data_infos):
    """Return {sample_token: list_index} for fast lookup."""
    return {info['token']: i for i, info in enumerate(data_infos)}


def get_lidar2img(cam_info):
    """Compute raw lidar→image matrix from cam_info (original resolution).

    Matches IS-Fusion's NuScenesDataset.get_data_info() exactly.
    """
    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3]  = -lidar2cam_t
    intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float32)
    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    return (viewpad @ lidar2cam_rt.T).astype(np.float32)


def extract_boxes(result):
    """Return (tensor_np, scores_np, labels_np) from one result dict."""
    pb = result.get('pts_bbox', result)
    boxes  = pb['boxes_3d'].tensor.cpu().numpy()
    scores = pb['scores_3d'].cpu().numpy()
    labels = pb['labels_3d'].cpu().numpy().astype(int)
    return boxes, scores, labels


def find_attacked_gt_box(info, atk_cls_name, bbox_2d, lidar2img, img_shape):
    """Find the GT box from data_infos closest to the YOLO-detected attack target.

    bbox_2d is in preprocessed image space (after _RESIZE scale and _CROP offset).
    We undo that to get the approximate original-image pixel location, then find
    the GT box whose projected 2D centre is closest.

    Returns a (1, 7) numpy array of the matched GT box, or None.
    """
    gt_boxes = info.get('gt_boxes')
    gt_names = info.get('gt_names')
    if gt_boxes is None or gt_names is None or len(gt_boxes) == 0:
        return None

    cls_mask = gt_names == atk_cls_name
    if not cls_mask.any():
        return None

    cls_boxes = gt_boxes[cls_mask]  # (M, 7), gravity-center z

    # Convert bbox_2d centre from preprocessed → original image coords
    pp_cx = (bbox_2d[0] + bbox_2d[2]) / 2.0
    pp_cy = (bbox_2d[1] + bbox_2d[3]) / 2.0
    orig_cx = (pp_cx + _CROP_X) / _RESIZE
    orig_cy = (pp_cy + _CROP_Y) / _RESIZE

    corners = gt_box_corners(cls_boxes)          # (M, 8, 3)
    pts_2d, valid = project_corners_to_img(corners, lidar2img)  # (M, 8, 2)

    best_dist = float('inf')
    best_idx  = None
    for i in range(len(cls_boxes)):
        if not valid[i]:
            continue
        box_cx = pts_2d[i, :, 0].mean()
        box_cy = pts_2d[i, :, 1].mean()
        dist = (box_cx - orig_cx) ** 2 + (box_cy - orig_cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx  = i

    if best_idx is None:
        return None
    return cls_boxes[best_idx:best_idx + 1]  # (1, 7)


# ─── all-views grid ──────────────────────────────────────────────────────────

def render_all_views_grid(
        info, data_root,
        b_boxes, b_scores, b_labels,
        a_boxes, a_scores, a_labels,
        atk_cam_idx, atk_cls_name, atk_label,
        attacked_gt_corners_3d,
        score_thr):
    """Return a stacked image with one row per camera (baseline | attack).

    The attacked camera row is highlighted with an orange border and label.
    """
    rows = []
    for cam_i, cam_name in enumerate(CAM_CHANNELS):
        cam_info = info['cams'][cam_name]

        rel_path = cam_info['data_path']
        for _prefix in ('./data/nuscenes/', 'data/nuscenes/'):
            if rel_path.startswith(_prefix):
                rel_path = rel_path[len(_prefix):]
                break
        img_path = os.path.join(data_root, rel_path)
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            raw_img = np.zeros((384, 1056, 3), dtype=np.uint8)
            cv2.putText(raw_img, f'[MISSING] {cam_name}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (100, 100, 100), 2)

        lidar2img = get_lidar2img(cam_info)
        is_attacked = (cam_i == atk_cam_idx)

        # Only draw the attacked GT box on the camera that was actually attacked
        gt_corners = attacked_gt_corners_3d if is_attacked else None

        b_panel = render_detections_on_img(
            raw_img, b_boxes, b_scores, b_labels,
            lidar2img, score_thr,
            default_color=BASELINE_COLOR,
            highlight_label=atk_label,
            highlight_color=HIGHLIGHT_COLOR,
            attacked_gt_corners=gt_corners,
        )
        a_panel = render_detections_on_img(
            raw_img, a_boxes, a_scores, a_labels,
            lidar2img, score_thr,
            default_color=ATTACK_COLOR,
            highlight_label=atk_label,
            highlight_color=HIGHLIGHT_COLOR,
            attacked_gt_corners=gt_corners,
        )

        h, w = raw_img.shape[:2]
        strip_h = 30
        atk_flag = '  ◄ ATTACKED' if is_attacked else ''
        for panel, side_label, text_color in [
                (b_panel, 'BASELINE', BASELINE_COLOR),
                (a_panel, 'ATTACK',   ATTACK_COLOR)]:
            strip = np.zeros((strip_h, w, 3), dtype=np.uint8)
            if is_attacked:
                strip[:] = (0, 40, 60)   # dark teal tint for attacked camera rows
            cv2.putText(strip,
                        f'{side_label} | {cam_name}{atk_flag}',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, text_color, 1, cv2.LINE_AA)
            panel[:] = np.vstack([strip, panel[strip_h:]])

        # Orange border around attacked camera panels
        if is_attacked:
            ORANGE = (0, 165, 255)
            cv2.rectangle(b_panel, (0, 0), (w - 1, h - 1), ORANGE, 4)
            cv2.rectangle(a_panel, (0, 0), (w - 1, h - 1), ORANGE, 4)

        rows.append(np.hstack([b_panel, a_panel]))

    return np.vstack(rows)


# ─── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Visualize baseline vs attack')
    p.add_argument('--baseline',   required=True, help='baseline_results.pkl')
    p.add_argument('--attack',     required=True, help='attack_results.pkl')
    p.add_argument('--attack-log', required=True, help='results_attack_log.json')
    p.add_argument('--info-file',  required=True,
                   help='nuscenes_infos_val.pkl (data_infos list)')
    p.add_argument('--data-root',  required=True,
                   help='NuScenes root directory (contains samples/)')
    p.add_argument('--out-dir',    default='vis_comparison')
    p.add_argument('--score-thr',  type=float, default=0.2,
                   help='Detection score threshold (default 0.2)')
    p.add_argument('--max-samples', type=int, default=50,
                   help='Maximum number of samples to render (default 50)')
    p.add_argument('--token', default=None,
                   help='Render only this specific sample token')
    p.add_argument('--bev', action='store_true',
                   help='Also write a BEV comparison image per sample')
    p.add_argument('--all-views', action='store_true',
                   help='Also write a grid image showing all 6 camera views '
                        '(baseline | attack) for each sample, not just the '
                        'attacked camera')
    p.add_argument('--baseline-token-order', default=None,
                   help='Optional path to a text file listing sample tokens '
                        'one-per-line in the order they appear in the baseline '
                        'pkl.  Use this when the baseline was produced from a '
                        'different evaluation run so its positional index does '
                        'not match data_infos order.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading results...')
    baseline_results = load_pkl(args.baseline)
    attack_results   = load_pkl(args.attack)

    print('Loading attack log...')
    with open(args.attack_log) as f:
        attack_log = json.load(f)

    print('Loading nuScenes data infos...')
    data_infos_raw = load_pkl(args.info_file)
    if isinstance(data_infos_raw, dict):
        data_infos = data_infos_raw['infos']
    else:
        data_infos = data_infos_raw

    token_to_idx = build_token_index(data_infos)
    print(f'  {len(data_infos)} samples indexed, '
          f'{len(attack_log)} tokens in attack log')

    # ── baseline index remapping ───────────────────────────────────────────
    # baseline_results[i] is valid only when its i-th entry corresponds to
    # data_infos[i].  If the baseline pkl was generated from a differently-
    # ordered or differently-sized dataset, the positional index is wrong.
    #
    # Detect the mismatch via length check and optionally remap with an
    # explicit token-order file (--baseline-token-order).
    baseline_token_to_idx = None  # None → use the same data_infos index
    if args.baseline_token_order:
        with open(args.baseline_token_order) as f:
            b_tokens = [line.strip() for line in f if line.strip()]
        baseline_token_to_idx = {tok: i for i, tok in enumerate(b_tokens)}
        print(f'  Loaded baseline token order: {len(b_tokens)} entries '
              f'from {args.baseline_token_order}')
        if len(b_tokens) != len(baseline_results):
            print(f'  WARNING: baseline token order file has {len(b_tokens)} '
                  f'tokens but baseline pkl has {len(baseline_results)} results. '
                  f'Counts should match.')
    else:
        n_base = len(baseline_results)
        n_info = len(data_infos)
        if n_base != n_info:
            print(
                f'\n  *** INDEX MISMATCH WARNING ***\n'
                f'  baseline_results has {n_base} entries but data_infos has '
                f'{n_info} entries.\n'
                f'  baseline_results[idx] will map to the WRONG sample for most '
                f'tokens.\n'
                f'  This is the likely cause of misaligned baseline projections.\n'
                f'  Fix: re-run the baseline evaluation using the same val dataset '
                f'as the attack, or supply --baseline-token-order with a file '
                f'listing the tokens in the order they appear in the baseline pkl.\n'
            )
        else:
            print(f'  baseline ({n_base}) and data_infos ({n_info}) lengths match ✓')

    tokens = [args.token] if args.token else list(attack_log.keys())
    tokens = [t for t in tokens if t in token_to_idx]
    if not tokens:
        print('ERROR: No matching tokens found in data_infos.')
        return
    tokens = tokens[:args.max_samples]
    print(f'Rendering {len(tokens)} samples → {args.out_dir}/')

    for sample_num, token in enumerate(tokens):
        idx = token_to_idx[token]
        info = data_infos[idx]
        atk_info = attack_log[token]

        cam_idx      = atk_info['camera_idx']
        atk_cls_name = atk_info['nuscenes_class']
        atk_label    = CLASSES.index(atk_cls_name) if atk_cls_name in CLASSES else -1
        bbox_2d      = atk_info.get('bbox_2d')

        cam_name  = CAM_CHANNELS[cam_idx]
        cam_info  = info['cams'][cam_name]

        rel_path = cam_info['data_path']
        for _prefix in ('./data/nuscenes/', 'data/nuscenes/'):
            if rel_path.startswith(_prefix):
                rel_path = rel_path[len(_prefix):]
                break
        img_path  = os.path.join(args.data_root, rel_path)
        lidar2img = get_lidar2img(cam_info)

        raw_img = cv2.imread(img_path)
        if raw_img is None:
            print(f'  [skip] cannot read image: {img_path}')
            continue

        # Find the specific attacked GT box
        attacked_gt_box = None
        attacked_gt_corners_3d = None
        if bbox_2d is not None:
            attacked_gt_box = find_attacked_gt_box(
                info, atk_cls_name, bbox_2d, lidar2img, raw_img.shape)
        if attacked_gt_box is not None:
            attacked_gt_corners_3d = gt_box_corners(attacked_gt_box)[0]  # (8, 3)

        # Use the remapped baseline index when a token-order file was supplied;
        # otherwise fall back to the same positional index used for attack.
        if baseline_token_to_idx is not None:
            b_idx = baseline_token_to_idx.get(token)
            if b_idx is None:
                print(f'  [skip baseline] token {token[:16]} not in '
                      f'--baseline-token-order file')
                b_idx = idx   # fall back so the rest of the loop still runs
        else:
            b_idx = idx

        b_boxes, b_scores, b_labels = extract_boxes(baseline_results[b_idx])
        a_boxes, a_scores, a_labels = extract_boxes(attack_results[idx])

        baseline_img = render_detections_on_img(
            raw_img, b_boxes, b_scores, b_labels,
            lidar2img, args.score_thr,
            default_color=BASELINE_COLOR,
            highlight_label=atk_label,
            highlight_color=HIGHLIGHT_COLOR,
            attacked_gt_corners=attacked_gt_corners_3d,
        )
        attack_img = render_detections_on_img(
            raw_img, a_boxes, a_scores, a_labels,
            lidar2img, args.score_thr,
            default_color=ATTACK_COLOR,
            highlight_label=atk_label,
            highlight_color=HIGHLIGHT_COLOR,
            attacked_gt_corners=attacked_gt_corners_3d,
        )

        strip_h = 40
        h, w = raw_img.shape[:2]
        for panel_img, label, color in [
                (baseline_img, 'BASELINE', BASELINE_COLOR),
                (attack_img,   'ATTACK',   ATTACK_COLOR)
            ]:
            strip = np.zeros((strip_h, w, 3), dtype=np.uint8)
            cv2.putText(strip,
                        f'{label}  |  {cam_name}  |  attacked: {atk_cls_name}  |  thr={args.score_thr}',
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
            panel_img[:] = np.vstack([strip, panel_img[strip_h:]])

        def _count(boxes, scores, labels, label_idx, thr):
            m = (scores >= thr) & (labels == label_idx)
            return int(m.sum())

        b_count  = _count(b_boxes, b_scores, b_labels, atk_label, args.score_thr)
        a_count  = _count(a_boxes, a_scores, a_labels, atk_label, args.score_thr)
        gt_found = 'yes' if attacked_gt_corners_3d is not None else 'no'
        footer_txt = (
            f'Baseline detects {b_count} {atk_cls_name}(s)  |  '
            f'Attack detects {a_count} {atk_cls_name}(s)  |  '
            f'attacked GT found: {gt_found}  |  '
            f'token: {token[:16]}...'
        )
        footer = np.zeros((36, w * 2, 3), dtype=np.uint8)
        cv2.putText(footer, footer_txt, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        side_by_side = np.hstack([
            baseline_img, 
            attack_img
        ])
        final = np.vstack([
            side_by_side, 
            footer
            ])

        out_path = os.path.join(args.out_dir, f'{sample_num:04d}_{token[:16]}_{cam_name}.jpg')
        cv2.imwrite(out_path, final, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if args.all_views:
            all_views_img = render_all_views_grid(
                info, args.data_root,
                b_boxes, b_scores, b_labels,
                a_boxes, a_scores, a_labels,
                atk_cam_idx=cam_idx,
                atk_cls_name=atk_cls_name,
                atk_label=atk_label,
                attacked_gt_corners_3d=attacked_gt_corners_3d,
                score_thr=args.score_thr,
            )
            all_views_path = os.path.join(
                args.out_dir,
                f'{sample_num:04d}_{token[:16]}_all_views.jpg')
            cv2.imwrite(all_views_path, all_views_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

        if args.bev:
            bev_gt = attacked_gt_box[0] if attacked_gt_box is not None else None
            fig = render_bev(
                b_boxes, b_scores, b_labels,
                a_boxes, a_scores, a_labels,
                attacked_label=atk_label,
                score_thr=args.score_thr,
                attacked_gt_box=bev_gt,
            )
            bev_path = os.path.join(args.out_dir,
                                    f'{sample_num:04d}_{token[:16]}_bev.png')
            fig.savefig(bev_path, dpi=120, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            plt.close(fig)

        print(f'  [{sample_num+1}/{len(tokens)}] {cam_name} '
              f'| {atk_cls_name}: baseline={b_count} attack={a_count} '
              f'| GT marked: {gt_found} '
              f'| {os.path.basename(out_path)}')

    print(f'\nDone. Wrote {len(tokens)} image(s) to {args.out_dir}/')


if __name__ == '__main__':
    main()
