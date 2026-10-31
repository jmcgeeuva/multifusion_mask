#!/usr/bin/env python3
"""Shared CPU helpers for the paper figure scripts (fig1/fig2/fig3).

Consumes the figure bundles written by attack_multi.py --dump: a pickle list of
per-scene dicts with display images (BGR uint8, [Ncam,H,W,3]), the raw calibration
(lidar2img / img_aug / lidar_aug), LiDAR points, GT boxes, and clean/attacked
decoded detections.

Boxes and points are in the AUGMENTED LiDAR frame and the display image is the
AUGMENTED (resize+crop) image, so projection mirrors grad_check.project_centres
exactly.  Pure CPU: needs numpy, matplotlib, cv2, and the mmdet3d package (only for
box-corner geometry; no GPU).  Run on any CPU node in the ssiai_adv env.
"""
import os
import sys
import pickle

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from visualize_attack import lidar_box_corners, bev_box_polygon, CLASSES   # noqa: E402

VEHICLE_LABELS = {CLASSES.index(c) for c in ('car', 'truck', 'bus') if c in CLASSES}

BGR = dict(clean=(0, 200, 0), attack=(0, 0, 220), ghost=(0, 0, 230),
           hidden=(0, 150, 255), gt=(255, 0, 255), kept=(0, 170, 0))
RGB = dict(clean=(0.0, 0.72, 0.0), attack=(0.85, 0.0, 0.0), ghost=(0.90, 0.05, 0.05),
           hidden=(1.0, 0.55, 0.0), gt=(0.85, 0.0, 0.85), kept=(0.0, 0.62, 0.0),
           pts=(0.62, 0.62, 0.66))

# ─── semantic detection-outcome palette (used in every figure, both panels) ───
#   correct = detection on a real object (TP) · ghost = detection with no object (FP)
#   missed  = real object with no detection (FN)
SEM_BGR = dict(correct=(0, 180, 0), ghost=(200, 0, 200), missed=(0, 0, 230))
SEM_RGB = dict(correct=(0.0, 0.70, 0.0), ghost=(0.80, 0.0, 0.80), missed=(0.88, 0.0, 0.0),
               pts=(0.62, 0.62, 0.66))

_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]


# ─── bundle IO ───────────────────────────────────────────────────────────────

def load_bundle(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def to_rgb(img_bgr):
    return np.ascontiguousarray(img_bgr[..., ::-1])


def vehicle_mask(labels):
    return np.array([l in VEHICLE_LABELS for l in labels], dtype=bool)


# ─── projection (augmented LiDAR -> augmented image), mirrors grad_check ──────

def project_pts(pts, l2i, ia, la):
    """pts [N,3] augmented-LiDAR -> (uv [N,2], valid [N]) in the augmented image."""
    pts = np.asarray(pts, np.float64)
    la3, lat = la[:3, :3], la[:3, 3]
    p = np.linalg.inv(la3) @ (pts - lat).T          # [3,N] undo lidar aug
    p = l2i[:3, :3] @ p + l2i[:3, 3:4]              # [3,N] lidar -> image ray
    z = p[2].copy()
    p2 = p[:2] / np.clip(p[2:3], 1e-5, None)
    p3 = np.vstack([p2, np.ones((1, p2.shape[1]))])
    q = ia[:3, :3] @ p3 + ia[:3, 3:4]              # apply image aug (resize+crop)
    return q[:2].T, z > 1e-3


def cam_calib(entry, cam):
    return entry['lidar2img'][cam], entry['img_aug'][cam], entry['lidar_aug']


def pick_camera_showing(entry, boxes_xyz):
    """Camera that frames the most of the given box centres in-view (ties -> most central).
    boxes_xyz [N,3+]. Falls back to camera 0 if none project in-frame."""
    if not len(boxes_xyz):
        return 0
    H, W = entry['img_clean'].shape[1:3]
    cen = np.asarray(boxes_xyz)[:, :3]
    best, best_key = 0, (-1, 1e18)
    for ci in range(len(entry['lidar2img'])):
        uv, ok = project_pts(cen, *cam_calib(entry, ci))
        inframe = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        n = int(inframe.sum())
        if n == 0:
            continue
        cost = float(np.mean((uv[inframe, 0] - W / 2) ** 2 + (uv[inframe, 1] - H / 2) ** 2))
        key = (n, -cost)
        if key > (best_key[0], -best_key[1]):
            best, best_key = ci, (n, cost)
    return best


def pick_camera(entry, center_xyz):
    """Camera that frames a LiDAR point best (in-frame, nearest image centre)."""
    H, W = entry['img_clean'].shape[1:3]
    best, best_cost = 0, 1e18
    for ci in range(len(entry['lidar2img'])):
        uv, ok = project_pts(np.asarray(center_xyz)[None], *cam_calib(entry, ci))
        if not ok[0]:
            continue
        u, v = uv[0]
        if not (0 <= u < W and 0 <= v < H):
            continue
        cost = (u - W / 2) ** 2 + (v - H / 2) ** 2
        if cost < best_cost:
            best_cost, best = cost, ci
    return best


# ─── drawing ─────────────────────────────────────────────────────────────────

def _seg(img, pa, pb, color, thick, dashed):
    p0 = np.round(pa).astype(int); p1 = np.round(pb).astype(int)
    if not dashed:
        cv2.line(img, tuple(p0), tuple(p1), color, thick, cv2.LINE_AA)
        return
    a = np.asarray(pa, float); b = np.asarray(pb, float)
    d = b - a; L = float(np.hypot(*d))
    if L < 1:
        return
    step = 9.0
    n = int(L // step)
    for i in range(0, n + 1, 2):                       # draw every other segment -> dashes
        s = a + d * (i * step / L)
        e = a + d * (min((i + 1) * step, L) / L)
        cv2.line(img, tuple(np.round(s).astype(int)), tuple(np.round(e).astype(int)),
                 color, thick, cv2.LINE_AA)


def _draw_box(img, uv, valid, color, thick, dashed=False):
    H, W = img.shape[:2]
    for a, b in _EDGES:
        if not (valid[a] or valid[b]):
            continue
        pa, pb = uv[a], uv[b]
        if max(abs(pa[0]), abs(pb[0])) > 4 * W or max(abs(pa[1]), abs(pb[1])) > 4 * H:
            continue
        _seg(img, pa, pb, color, thick, dashed)


def draw_cam(entry, cam, which, boxes, scores, labels, color, score_thr=0.0,
             highlight=None, hl_color=None, hl_box=None, label_txt=True):
    """Return an RGB image: detections drawn on one camera of the bundle.

    highlight: iterable of box indices drawn in hl_color (e.g. ghosts).
    hl_box:    a single extra box row [7+] drawn in magenta (e.g. the hidden target).
    """
    img = entry['img_%s' % which][cam].copy()          # BGR
    l2i, ia, la = cam_calib(entry, cam)
    hl = set(int(i) for i in (highlight if highlight is not None else []))
    if hl_box is not None:
        c = lidar_box_corners(np.asarray(hl_box, np.float32).reshape(1, -1))[0]
        uv, ok = project_pts(c, l2i, ia, la)
        _draw_box(img, uv, ok, BGR['gt'], 3)
    keep = np.where(np.asarray(scores) >= score_thr)[0] if len(scores) else []
    if len(keep):
        corners = lidar_box_corners(boxes[keep])
        for k, bi in enumerate(keep):
            uv, ok = project_pts(corners[k], l2i, ia, la)
            if not ok.any():
                continue
            col = hl_color if (int(bi) in hl and hl_color) else color
            thick = 3 if (int(bi) in hl) else 2
            _draw_box(img, uv, ok, col, thick)
            if label_txt:
                j = int(np.argmax(ok))
                px, py = uv[j]
                if -img.shape[1] < px < 2 * img.shape[1] and -img.shape[0] < py < 2 * img.shape[0]:
                    cv2.putText(img, f'{CLASSES[labels[bi]]} {scores[bi]:.2f}',
                                (int(np.clip(px, 2, img.shape[1] - 2)), int(np.clip(py, 12, img.shape[0] - 2)) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return to_rgb(img)


# ─── semantic classification: correct / ghost / missed ───────────────────────

def classify(preds, gt, score_thr, ref=None, tol=2.0):
    """Split a prediction set against GT by 2 m centre matching.
    correct/ghost : prediction indices.  missed : GT indices (no detection).
    If `ref` (the clean predictions) is given, missed GTs are split by whether the
    reference also missed them:
      missed_attack   : clean DETECTED it but this panel lost it -> attack-caused (solid red)
      missed_baseline : clean ALSO missed it -> would be missed anyway (dotted red)
    Without `ref`, every miss is baseline (dotted)."""
    boxes = preds.get('boxes', np.zeros((0, 7)))
    scores = np.asarray(preds.get('scores', np.zeros(0)))
    keep = np.where(scores >= score_thr)[0] if len(scores) else np.zeros(0, int)
    G = len(gt)
    if len(keep) and G:
        d = np.linalg.norm(boxes[keep, None, :2] - gt[None, :, :2], axis=2)   # (K,G)
        correct = keep[d.min(axis=1) < tol]
        ghost = keep[d.min(axis=1) >= tol]
        missed = np.where(d.min(axis=0) >= tol)[0]
    else:
        correct = np.zeros(0, int)
        ghost = keep
        missed = np.arange(G)
    # split missed into attack-caused vs pre-existing, using the clean reference
    ref_hit = np.zeros(len(missed), bool)
    if ref is not None and len(missed):
        rb = ref.get('boxes', np.zeros((0, 7)))
        rs = np.asarray(ref.get('scores', np.zeros(0)))
        rk = rb[rs >= score_thr] if len(rs) else np.zeros((0, 7))
        if len(rk):
            dr = np.linalg.norm(gt[missed, None, :2] - rk[None, :, :2], axis=2).min(axis=1)
            ref_hit = dr < tol
    return dict(correct=correct, ghost=ghost, missed=missed,
                missed_attack=missed[ref_hit], missed_baseline=missed[~ref_hit])


def _draw_boxes_on(img, entry, cam, boxes, color, thick, scores=None, labels=None, dashed=False):
    """Draw a set of LiDAR boxes on one camera in a single colour (mutates img)."""
    if not len(boxes):
        return
    l2i, ia, la = cam_calib(entry, cam)
    corners = lidar_box_corners(np.asarray(boxes, np.float32))
    for k in range(len(boxes)):
        uv, ok = project_pts(corners[k], l2i, ia, la)
        if not ok.any():
            continue
        _draw_box(img, uv, ok, color, thick, dashed=dashed)
        if scores is not None and labels is not None:
            j = int(np.argmax(ok)); px, py = uv[j]
            if -img.shape[1] < px < 2 * img.shape[1] and -img.shape[0] < py < 2 * img.shape[0]:
                cv2.putText(img, f'{CLASSES[labels[k]]} {scores[k]:.2f}',
                            (int(np.clip(px, 2, img.shape[1] - 2)), int(np.clip(py, 12, img.shape[0] - 2)) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_cam_semantic(entry, cam, which, preds, gt, score_thr, ref=None, show_missed=True, tol=2.0):
    """Camera image with detections coloured by outcome.
      green = correct · magenta = ghost · solid red = missed-by-attack · dotted red = missed anyway
    Returns (rgb_image, classification dict)."""
    img = entry['img_%s' % which][cam].copy()
    c = classify(preds, gt, score_thr, ref=ref, tol=tol)
    b, s, l = preds['boxes'], preds['scores'], preds['labels']
    if show_missed:
        if len(c['missed_baseline']):          # clean also missed -> dotted
            _draw_boxes_on(img, entry, cam, gt[c['missed_baseline']], SEM_BGR['missed'], 2, dashed=True)
        if len(c['missed_attack']):            # attack hid a clean detection -> solid, thicker
            _draw_boxes_on(img, entry, cam, gt[c['missed_attack']], SEM_BGR['missed'], 3, dashed=False)
    _draw_boxes_on(img, entry, cam, b[c['correct']], SEM_BGR['correct'], 2, s[c['correct']], l[c['correct']])
    _draw_boxes_on(img, entry, cam, b[c['ghost']], SEM_BGR['ghost'], 3, s[c['ghost']], l[c['ghost']])
    return to_rgb(img), c


def bev_semantic_sets(preds, gt, c):
    """Build draw_bev box_sets for a classified prediction set."""
    b = preds['boxes']
    g = lambda idx: gt[idx] if len(idx) else np.zeros((0, 7))
    p = lambda idx: b[idx] if len(idx) else np.zeros((0, 7))
    return [(g(c['missed_baseline']), SEM_RGB['missed'], 2.0, 'missed (clean also missed)', ':'),
            (g(c['missed_attack']), SEM_RGB['missed'], 2.4, 'missed (hidden by attack)', '-'),
            (p(c['correct']), SEM_RGB['correct'], 1.6, 'correct detection'),
            (p(c['ghost']), SEM_RGB['ghost'], 2.4, 'ghost (no object)')]


# ─── clean vs attacked detection matching ────────────────────────────────────

def match_dets(clean, attack, tol=2.0):
    cb, ab = clean['boxes'], attack['boxes']
    nc, na = len(cb), len(ab)
    hidden, ghost, kept_c, kept_a = list(range(nc)), list(range(na)), [], []
    if nc and na:
        d = np.linalg.norm(cb[:, None, :2] - ab[None, :, :2], axis=2)
        pairs = sorted(((d[i, j], i, j) for i in range(nc) for j in range(na)), key=lambda t: t[0])
        uc, ua = set(), set()
        for dist, i, j in pairs:
            if dist > tol or i in uc or j in ua:
                continue
            uc.add(i); ua.add(j); kept_c.append(i); kept_a.append(j)
        hidden = [i for i in range(nc) if i not in uc]
        ghost = [j for j in range(na) if j not in ua]
    return dict(hidden=np.array(hidden, int), ghost=np.array(ghost, int),
                kept_c=np.array(kept_c, int), kept_a=np.array(kept_a, int))


def ghost_indices(entry, score_thr, tol=2.0):
    """Attack-FABRICATED ghosts: attacked dets (score>=thr) that are >tol from EVERY GT box
    AND >tol from every clean detection -- i.e. the clean model did NOT produce them, so they
    are created by the attack (not pre-existing detector false positives).

    Returns (ghost_idx[int], in_hull_mask[bool]) where in_hull_mask flags the ghosts
    whose BEV centre lies inside the painted-vehicle convex hull (needs >=3 painted)."""
    pa = entry['pred_attack']
    boxes, scores = pa['boxes'], pa['scores']
    if not len(boxes):
        return np.zeros(0, int), np.zeros(0, bool)
    keep = np.where(np.asarray(scores) >= score_thr)[0]
    gt = entry['gt_boxes']
    if len(gt) and len(keep):
        d = np.linalg.norm(boxes[keep, None, :2] - gt[None, :, :2], axis=2).min(axis=1)
        ghost = keep[d > tol]
    else:
        ghost = keep
    # drop anything the clean model already detected there (pre-existing FP, not attack-made)
    pc = entry.get('pred_clean', {})
    cb = pc.get('boxes', np.zeros((0, 7)))
    cs = pc.get('scores', np.zeros(0))
    if len(ghost) and len(cb):
        ckeep = cb[np.asarray(cs) >= score_thr]
        if len(ckeep):
            dc = np.linalg.norm(boxes[ghost, None, :2] - ckeep[None, :, :2], axis=2).min(axis=1)
            ghost = ghost[dc > tol]
    in_hull = np.zeros(len(ghost), bool)
    ag = entry.get('attacked_gt', [])
    if len(ag) >= 3 and len(ghost):
        hull = convex_hull_xy(entry['gt_boxes'][ag, :2])
        if hull is not None:
            try:
                from matplotlib.path import Path
                in_hull = Path(hull[:, :2]).contains_points(boxes[ghost, :2])
            except Exception:
                pass
    return ghost, in_hull


def convex_hull_xy(centers):
    p = np.asarray(centers, float)
    if len(p) < 3:
        return None
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h[:-1]

    hull = np.array(half(p) + half(p[::-1]))
    return np.vstack([hull, hull[0]]) if len(hull) >= 3 else None


# ─── BEV ─────────────────────────────────────────────────────────────────────

def draw_bev(ax, points, box_sets, rng=54.0, title='', hull=None, axis_labels=True):
    """box_sets: list of (boxes[N,7+], rgb_color, linewidth, label[, linestyle])."""
    if points is not None and len(points):
        m = (np.abs(points[:, 0]) < rng) & (np.abs(points[:, 1]) < rng)
        ax.scatter(points[m, 1], points[m, 0], s=0.15, c=[RGB['pts']], alpha=0.5, linewidths=0)
    if hull is not None:
        ax.plot(hull[:, 1], hull[:, 0], color='0.45', lw=1.0, ls='--', alpha=0.8)
    seen = set()
    for item in box_sets:
        boxes, color, lw, label = item[0], item[1], item[2], item[3]
        ls = item[4] if len(item) > 4 else '-'
        for b in boxes:
            poly = bev_box_polygon(b)
            ax.plot(poly[:, 1], poly[:, 0], color=color, lw=lw, ls=ls,
                    label=(label if label not in seen else None))
            ctr, front = poly[:4].mean(0), poly[:2].mean(0)
            ax.plot([ctr[1], front[1]], [ctr[0], front[0]], color=color, lw=lw * 0.7, ls=ls)
            seen.add(label)
    ax.plot(0, 0, marker='^', color='k', ms=7)
    ax.set_xlim(rng, -rng); ax.set_ylim(-rng, rng); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)
    if axis_labels:
        ax.set_xlabel('left (m)  ←', fontsize=7); ax.set_ylabel('→ forward (m)', fontsize=7)


def semantic_handles():
    """Proxy legend handles: correct / ghost / missed-by-attack (solid) / missed-anyway (dotted)."""
    import matplotlib.lines as mlines
    return [mlines.Line2D([], [], color=SEM_RGB['correct'], lw=2.2, label='correct detection'),
            mlines.Line2D([], [], color=SEM_RGB['ghost'], lw=2.2, label='ghost (no real object)'),
            mlines.Line2D([], [], color=SEM_RGB['missed'], lw=2.4, label='missed — hidden by attack'),
            mlines.Line2D([], [], color=SEM_RGB['missed'], lw=2.2, ls=':', label='missed — clean missed too')]


def paired_grid(cam_aspect, side=4.5, headers=('Clean', 'Attacked')):
    """A tight 2x2 figure: camera row (top) abutting a square BEV row (bottom), 2 columns.
    Returns (fig, [[cam0, cam1], [bev0, bev1]]).  No vertical gap between view and BEV."""
    r = float(cam_aspect)
    fig = plt.figure(figsize=(2 * side, side + side / r))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0 / r, 1.0], hspace=0.0, wspace=0.015,
                          left=0.045, right=0.995, top=0.955, bottom=0.015)
    ax = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
          [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]]
    for j, h in enumerate(headers):
        ax[0][j].set_title(h, fontsize=12, pad=4)
    return fig, ax


def savefig(fig, path, dpi=200):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)
