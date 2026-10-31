#!/usr/bin/env python3
"""Figure 2 - the attack makes a detection disappear.

Same layout as Figure 1 (clean vs. attacked, camera + BEV).  A confident clean
vehicle detection that the attack drives below threshold is highlighted in
orange on the clean panels and marked (dashed) where it vanished on the attacked
panels.

Input: a figure bundle from a hiding attack, e.g. universal bbox
  attack_multi.py <cfg> --mode universal --mask all --place bbox --inference-norm \
      --load-bank <run>/bank_last.npy --steps 0 --dump hide.pkl --dump-scenes "i,j"
or per-scene suppress.  CPU only.

  python tools/fig2_hiding.py --bundle hide.pkl --out paper/figs/hiding.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import fig_lib as F


def pick_scene(bundle, thr):
    """Scene with the most confident vehicle that the attack hides."""
    best, best_s = bundle[0], -1.0
    for e in bundle:
        m = F.match_dets(e['pred_clean'], e['pred_attack'])
        hid = m['hidden']
        if not len(hid):
            continue
        veh = hid[F.vehicle_mask(e['pred_clean']['labels'][hid])]
        veh = veh[e['pred_clean']['scores'][veh] >= thr]
        if not len(veh):
            continue
        s = e['pred_clean']['scores'][veh].max()
        if s > best_s:
            best, best_s = e, s
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', default='figs/fig2_hiding.png')
    ap.add_argument('--scene', type=int, default=None)
    ap.add_argument('--score-thr', type=float, default=0.2)
    args = ap.parse_args()

    bundle = F.load_bundle(args.bundle)
    thr = args.score_thr
    e = next((x for x in bundle if x['scene'] == args.scene), None) if args.scene is not None else pick_scene(bundle, thr)
    if e is None:
        raise SystemExit('scene not found in bundle')

    pc, pa = e['pred_clean'], e['pred_attack']
    gt = e['gt_boxes'] if len(e['gt_boxes']) else np.zeros((0, 7))
    m = F.match_dets(pc, pa)
    hid = m['hidden']
    hid = hid[F.vehicle_mask(pc['labels'][hid])]
    hid = hid[pc['scores'][hid] >= thr]
    if not len(hid):
        raise SystemExit('no hidden vehicle detection in this scene; try another')
    # the single most confident hidden vehicle is the headline
    target = hid[np.argmax(pc['scores'][hid])]
    focus = pc['boxes'][target, :3]
    cam = F.pick_camera(e, focus)

    # semantic colouring (green = correct, magenta = ghost, red dotted = missed) in BOTH panels
    img_clean, cc = F.draw_cam_semantic(e, cam, 'clean', pc, gt, thr)
    img_atk, ca = F.draw_cam_semantic(e, cam, 'attack', pa, gt, thr, ref=pc)

    H, W = e['img_clean'].shape[1:3]
    fig, ax = F.paired_grid(W / H, side=4.5, headers=('Clean', 'Attacked'))
    for a, im in ((ax[0][0], img_clean), (ax[0][1], img_atk)):
        a.imshow(im, aspect='auto'); a.set_xticks([]); a.set_yticks([])

    # circle the headline (hidden) target in both BEV panels for the eye
    tgt = pc['boxes'][target:target + 1]
    RNG = 46.0
    F.draw_bev(ax[1][0], e['points'], F.bev_semantic_sets(pc, gt, cc), rng=RNG, axis_labels=False)
    F.draw_bev(ax[1][1], e['points'], F.bev_semantic_sets(pa, gt, ca), rng=RNG, axis_labels=False)
    for a in (ax[1][0], ax[1][1]):
        a.plot(tgt[0, 1], tgt[0, 0], marker='o', ms=15, mfc='none', mec='k', mew=1.4, zorder=5)
    leg = ax[1][0].legend(handles=F.semantic_handles(), loc='lower left', ncol=1, fontsize=8.5,
                          framealpha=0.92, borderpad=0.6, handlelength=1.6)
    leg.get_frame().set_edgecolor('0.7')
    F.savefig(fig, args.out)


if __name__ == '__main__':
    main()
