#!/usr/bin/env python3
"""
Side-by-side visualization of baseline vs. attack detections.

For each sample in the attack log, renders two panels:
  LEFT  – baseline predictions on the attacked camera image
  RIGHT – attack predictions on the same image

Predicted boxes of the attacked class are highlighted.
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

# Colour scheme: baseline=green, attack=red, gt=blue
BASELINE_COLOR = (0, 200, 0)    # BGR
ATTACK_COLOR   = (0, 0, 220)    # BGR
HIGHLIGHT_COLOR = (0, 165, 255) # BGR – attacked class boxes in attack panel


# ─── geometry helpers ────────────────────────────────────────────────────────

def lidar_box_corners(box_tensor):
    """Return (N, 8, 3) corners in LiDAR coords for a batch of LiDAR boxes.

    box_tensor: (N, 9) – [cx, cy, cz, dx, dy, dz, yaw, vx, vy]
    """
    cx, cy, cz = box_tensor[:, 0], box_tensor[:, 1], box_tensor[:, 2]
    dx, dy, dz = box_tensor[:, 3], box_tensor[:, 4], box_tensor[:, 5]
    yaw = box_tensor[:, 6]
    N = len(cx)

    # unit box corners in local frame (bottom-centre origin)
    unit = np.array([
        [ 1,  1, 0], [ 1, -1, 0], [-1, -1, 0], [-1,  1, 0],   # bottom
        [ 1,  1, 1], [ 1, -1, 1], [-1, -1, 1], [-1,  1, 1],   # top
    ], dtype=np.float32) * 0.5  # (8,3)

    corners = np.zeros((N, 8, 3), dtype=np.float32)
    for i in range(N):
        c, s = np.cos(yaw[i]), np.sin(yaw[i])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        pts = unit * np.array([dx[i], dy[i], dz[i]])  # scale
        pts = (R @ pts.T).T                             # rotate
        pts += np.array([cx[i], cy[i], cz[i] + dz[i] / 2])  # translate (gravity centre)
        corners[i] = pts
    return corners


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
                             highlight_color=None):
    """Return a copy of raw_img with all detection boxes drawn."""
    img = raw_img.copy()
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

        # score label above the front-top edge midpoint
        mid = pts_2d[i][4:6].mean(axis=0)
        txt = f'{CLASSES[lb[i]]} {sc[i]:.2f}'
        font_scale = 0.5 if is_highlight else 0.4
        cv2.putText(img, txt,
                    (int(mid[0]), int(mid[1]) - 4),
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
               attacked_label, score_thr, pc_range=(-54, -54, 54, 54)):
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
        Line2D([0], [0], color='#ff4444', linewidth=2, label=f'Attacked class'),
        Line2D([0], [0], color='#44ff44', linewidth=1, label='Other classes'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
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
    """Compute raw lidar→image matrix from cam_info (original resolution)."""
    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3]  = -lidar2cam_t
    intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float32)
    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:3, :3] = intrinsic
    return (viewpad @ lidar2cam_rt.T).astype(np.float32)


def extract_boxes(result):
    """Return (tensor_np, scores_np, labels_np) from one result dict."""
    pb = result.get('pts_bbox', result)
    boxes  = pb['boxes_3d'].tensor.cpu().numpy()
    scores = pb['scores_3d'].cpu().numpy()
    labels = pb['labels_3d'].cpu().numpy().astype(int)
    return boxes, scores, labels


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
    # nuscenes_infos_val.pkl is a dict with key 'infos' in some versions
    if isinstance(data_infos_raw, dict):
        data_infos = data_infos_raw['infos']
    else:
        data_infos = data_infos_raw

    token_to_idx = build_token_index(data_infos)
    print(f'  {len(data_infos)} samples indexed, '
          f'{len(attack_log)} tokens in attack log')

    # Select tokens to render
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

        # Camera metadata
        cam_name  = CAM_CHANNELS[cam_idx]
        cam_info  = info['cams'][cam_name]

        # data_path in the pkl starts with './data/nuscenes/' — strip it
        rel_path = cam_info['data_path']
        for _prefix in ('./data/nuscenes/', 'data/nuscenes/'):
            if rel_path.startswith(_prefix):
                rel_path = rel_path[len(_prefix):]
                break
        img_path  = os.path.join(args.data_root, rel_path)
        lidar2img = get_lidar2img(cam_info)

        # Load original camera image
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            print(f'  [skip] cannot read image: {img_path}')
            continue

        # Extract detections
        b_boxes, b_scores, b_labels = extract_boxes(baseline_results[idx])
        a_boxes, a_scores, a_labels = extract_boxes(attack_results[idx])

        # Render camera views
        baseline_img = render_detections_on_img(
            raw_img, b_boxes, b_scores, b_labels,
            lidar2img, args.score_thr,
            default_color=BASELINE_COLOR,
            highlight_label=atk_label,
            highlight_color=HIGHLIGHT_COLOR,
        )
        attack_img = render_detections_on_img(
            raw_img, a_boxes, a_scores, a_labels,
            lidar2img, args.score_thr,
            default_color=ATTACK_COLOR,
            highlight_label=atk_label,
            highlight_color=HIGHLIGHT_COLOR,
        )

        # Annotate header strips
        strip_h = 40
        h, w = raw_img.shape[:2]
        for panel_img, label, color in [
                (baseline_img, 'BASELINE', BASELINE_COLOR),
                (attack_img,   'ATTACK',   ATTACK_COLOR)]:
            strip = np.zeros((strip_h, w, 3), dtype=np.uint8)
            cv2.putText(strip,
                        f'{label}  |  {cam_name}  |  attacked: {atk_cls_name}  |  thr={args.score_thr}',
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
            panel_img[:] = np.vstack([strip, panel_img[strip_h:]])

        # Count detected boxes of the attacked class above threshold
        def _count(boxes, scores, labels, label_idx, thr):
            m = (scores >= thr) & (labels == label_idx)
            return int(m.sum())

        b_count = _count(b_boxes, b_scores, b_labels, atk_label, args.score_thr)
        a_count = _count(a_boxes, a_scores, a_labels, atk_label, args.score_thr)
        footer_txt = (
            f'Baseline detects {b_count} {atk_cls_name}(s)  |  '
            f'Attack detects {a_count} {atk_cls_name}(s)  |  '
            f'token: {token[:16]}...'
        )
        footer = np.zeros((36, w * 2, 3), dtype=np.uint8)
        cv2.putText(footer, footer_txt, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        side_by_side = np.hstack([baseline_img, attack_img])
        final = np.vstack([side_by_side, footer])

        out_path = os.path.join(args.out_dir, f'{sample_num:04d}_{token[:16]}_{cam_name}.jpg')
        cv2.imwrite(out_path, final, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Optional BEV
        if args.bev:
            fig = render_bev(
                b_boxes, b_scores, b_labels,
                a_boxes, a_scores, a_labels,
                attacked_label=atk_label,
                score_thr=args.score_thr,
            )
            bev_path = os.path.join(args.out_dir,
                                    f'{sample_num:04d}_{token[:16]}_bev.png')
            fig.savefig(bev_path, dpi=120, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            plt.close(fig)

        print(f'  [{sample_num+1}/{len(tokens)}] {cam_name} '
              f'| {atk_cls_name}: baseline={b_count} attack={a_count} '
              f'| {os.path.basename(out_path)}')

    print(f'\nDone. Wrote {len(tokens)} image(s) to {args.out_dir}/')


if __name__ == '__main__':
    main()
