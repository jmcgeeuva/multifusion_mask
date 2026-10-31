#!/usr/bin/env python3
"""Figure 3 - baseline attacks compared on one scene.

Columns: the clean prediction, then each attack (full-vehicle texture, patch
20%/40%, tile, screen, random, ...) side by side on the SAME scene and SAME
camera.  Rows: 3-D-box camera view (top) and BEV (bottom).  Each attack column
is annotated with the targeted-vehicle confidence so the reader can rank them.

Each attack is a separate figure bundle dumped for the SAME --dump-scenes index,
e.g.:
  attack_multi.py <cfg> --mode universal --place bbox --load-bank <full>/bank_last.npy \
      --steps 0 --inference-norm --dump full.pkl --dump-scenes "7"
  attack_multi.py <cfg> --attack patch --patch-frac 0.2 --mode perscene ... --dump patch20.pkl --dump-scenes "7"
  ...

CPU only.  Pass bundles as NAME:PATH (order = column order):
  python tools/fig3_baselines.py --scene 7 --out paper/figs/baselines.png \
      --bundle "Full texture:full.pkl" --bundle "Patch 20%:patch20.pkl" \
      --bundle "Patch 40%:patch40.pkl" --bundle "Tile:tile.pkl" \
      --bundle "Screen:screen.pkl" --bundle "Random:random.pkl"
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import fig_lib as F


def get_scene(bundle, scene):
    if scene is None:
        return bundle[0]
    return next((x for x in bundle if x['scene'] == scene), None)


def targeted_conf(entry, thr):
    """Mean clean->attacked confidence of the painted (targeted) vehicles, if recorded."""
    if not len(entry.get('attacked_gt', [])):
        return None
    gt = entry['gt_boxes'][entry['attacked_gt'], :2]
    pa = entry['pred_attack']
    if not len(pa['boxes']):
        return 0.0
    d = np.linalg.norm(gt[:, None, :] - pa['boxes'][None, :, :2], axis=2)  # (T, N)
    best = []
    for t in range(len(gt)):
        near = pa['scores'][d[t] < 2.0]
        best.append(float(near.max()) if len(near) else 0.0)
    return float(np.mean(best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', action='append', required=True,
                    help='NAME:PATH, repeatable; column order follows the flags')
    ap.add_argument('--scene', type=int, default=None, help='shared scene index across bundles')
    ap.add_argument('--out', default='figs/fig3_baselines.png')
    ap.add_argument('--score-thr', type=float, default=0.2)
    args = ap.parse_args()
    thr = args.score_thr

    cols = []
    for spec in args.bundle:
        name, path = spec.split(':', 1)
        e = get_scene(F.load_bundle(path), args.scene)
        if e is None:
            raise SystemExit(f'scene {args.scene} not in {path}')
        cols.append((name, e))

    ref = cols[0][1]                               # clean reference
    gt = ref['gt_boxes'] if len(ref['gt_boxes']) else np.zeros((0, 7))
    cpred = ref['pred_clean']
    # frame the painted (textured) vehicles so they are prominent (not a distant cluster)
    ag = ref.get('attacked_gt', [])
    cam = F.pick_camera_showing(ref, ref['gt_boxes'][ag]) if len(ag) else F.pick_camera(ref, np.zeros(3))

    # one cell = camera over BEV, per method; 7 methods wrapped 4-on-top / 3-on-bottom + legend
    panels = [('Clean', ref, 'clean', cpred, None)]
    for name, e in cols:
        tconf = targeted_conf(e, thr)
        tag = f'{name}' + (f'  (tgt {tconf:.2f})' if tconf is not None else '')
        panels.append((tag, e, 'attack', e['pred_attack'], cpred))

    H, W = ref['img_clean'].shape[1:3]
    r = W / H
    NCOL = 4
    side = 2.7
    rh = [1.0 / r, 1.0, 0.18, 1.0 / r, 1.0]        # cam, bev, spacer, cam, bev
    fig = plt.figure(figsize=(NCOL * side, side * sum(rh)))
    gs = fig.add_gridspec(5, NCOL, height_ratios=rh, hspace=0.0, wspace=0.015,
                          left=0.004, right=0.996, top=0.955, bottom=0.01)

    def cell(super_row, col, title, e, which, preds, ref_pred):
        rc = 0 if super_row == 0 else 3
        acam = fig.add_subplot(gs[rc, col]); abev = fig.add_subplot(gs[rc + 1, col])
        img, c = F.draw_cam_semantic(e, cam, which, preds, gt, thr, ref=ref_pred)
        acam.imshow(img, aspect='auto'); acam.set_xticks([]); acam.set_yticks([])
        weight = 'bold' if title.startswith('Ours') else 'normal'
        acam.set_title(title, fontsize=10, pad=3, fontweight=weight)
        F.draw_bev(abev, e['points'], F.bev_semantic_sets(preds, gt, c), rng=46.0, axis_labels=False)

    for i, (title, e, which, preds, ref_pred) in enumerate(panels):
        sr = 0 if i < NCOL else 1
        col = i if i < NCOL else i - NCOL
        cell(sr, col, title, e, which, preds, ref_pred)

    # legend fills the empty bottom-right cell
    lax = fig.add_subplot(gs[3:5, NCOL - 1]); lax.axis('off')
    lax.legend(handles=F.semantic_handles(), loc='center', fontsize=10, frameon=False)
    F.savefig(fig, args.out)


if __name__ == '__main__':
    main()
