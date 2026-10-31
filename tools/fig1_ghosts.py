#!/usr/bin/env python3
"""Figure 1 - the attack fabricates ghost detections.

Side by side: the model's predictions on the clean scene (left) vs. on the
attacked scene (right), as a 3-D-box camera view AND a BEV.  A ghost is an
attacked detection that sits >2 m from EVERY ground-truth object (the paper's
false-positive metric); ghosts inside the painted-vehicle convex hull are the
headline.  Ghosts are drawn thick red and labelled; ground-truth boxes are
overlaid faint magenta in BEV so the reader sees the ghosts have no real object
under them.  The painted-vehicle hull is dashed red in BEV.

Input: a figure bundle from
  attack_multi.py <cfg> --mode perscene --mask all --region-mode hull_saturate \
      --place bbox --inference-norm --fp-eval --perscene-steps 100 \
      --dump ghosts.pkl --dump-scenes "0,1,...,N"

CPU only.  Example:
  python tools/fig1_ghosts.py --bundle ghosts.pkl --out paper/figs/ghosts.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import fig_lib as F


def pick_scene(bundle, thr):
    """Scene with the most in-hull ghosts (fall back to total ghosts)."""
    best, best_key = bundle[0], (-1, -1)
    for e in bundle:
        g, inh = F.ghost_indices(e, thr)
        key = (int(inh.sum()), len(g))
        if key > best_key:
            best, best_key = e, key
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', default='figs/fig1_ghosts.png')
    ap.add_argument('--scene', type=int, default=None, help='scene index; default auto-pick')
    ap.add_argument('--score-thr', type=float, default=0.3)
    args = ap.parse_args()

    bundle = F.load_bundle(args.bundle)
    thr = args.score_thr
    e = next((x for x in bundle if x['scene'] == args.scene), None) if args.scene is not None else pick_scene(bundle, thr)
    if e is None:
        raise SystemExit('scene not found in bundle')

    ghost, in_hull = F.ghost_indices(e, thr)       # net-fabricated, for scene framing
    pc, pa = e['pred_clean'], e['pred_attack']
    gt = e['gt_boxes'] if len(e['gt_boxes']) else np.zeros((0, 7))

    # hull of the painted vehicles (for BEV); camera chosen to actually SHOW the ghosts
    ag = e.get('attacked_gt', [])
    hull = F.convex_hull_xy(e['gt_boxes'][ag, :2]) if len(ag) >= 3 else None
    if len(ghost):
        cam = F.pick_camera_showing(e, pa['boxes'][ghost])          # frame the fabricated boxes
    elif len(ag):
        cam = F.pick_camera(e, e['gt_boxes'][ag, :3].mean(axis=0))
    else:
        cam = 0

    # semantic colouring (green = correct, magenta = ghost, red dotted = missed) in BOTH panels
    img_clean, cc = F.draw_cam_semantic(e, cam, 'clean', pc, gt, thr)
    img_atk, ca = F.draw_cam_semantic(e, cam, 'attack', pa, gt, thr, ref=pc)

    H, W = e['img_clean'].shape[1:3]
    fig, ax = F.paired_grid(W / H, side=4.5, headers=('Clean', 'Attacked'))
    for a, im in ((ax[0][0], img_clean), (ax[0][1], img_atk)):
        a.imshow(im, aspect='auto'); a.set_xticks([]); a.set_yticks([])
    RNG = 46.0
    F.draw_bev(ax[1][0], e['points'], F.bev_semantic_sets(pc, gt, cc), rng=RNG, axis_labels=False)
    F.draw_bev(ax[1][1], e['points'], F.bev_semantic_sets(pa, gt, ca), rng=RNG, hull=hull,
               axis_labels=False)
    leg = ax[1][0].legend(handles=F.semantic_handles(), loc='lower left', ncol=1, fontsize=8.5,
                          framealpha=0.92, borderpad=0.6, handlelength=1.6)
    leg.get_frame().set_edgecolor('0.7')
    F.savefig(fig, args.out)


if __name__ == '__main__':
    main()
