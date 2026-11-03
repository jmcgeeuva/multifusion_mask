#!/usr/bin/env python3
"""Cross-family transfer figure: a texture optimized ONLY on IS-Fusion hides a real vehicle
on a never-optimized victim (BEVFusion / CMT).

Clean vs attacked, camera + LiDAR BEV, but DECLUTTERED for the transfer story: only
  green  = correct detection (matches GT)
  red    = the GT vehicle the attack hides (detected clean, missed under attack)
are drawn. Pre-existing detector ghosts and far-range GT both models miss are omitted, since
they are not what the transfer demonstrates. Reads a bundle from build_transfer_bundle.py.

  python tools/fig_transfer_hiding.py --bundle b.pkl --out fig.png [--scene i] [--score-thr .4]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import fig_lib as F


def hidden_vehicle(entry, thr, tol=2.0):
    """GT vehicle indices detected clean (>=thr) but missed under attack; + the headline one."""
    pc, pa = entry['pred_clean'], entry['pred_attack']
    gt, gtl = entry['gt_boxes'], np.asarray(entry['gt_labels'])
    ca = F.classify(pa, gt, thr, ref=pc, tol=tol)
    ma = np.asarray(ca['missed_attack'], int)
    if not len(ma):
        return ma, None
    veh = ma[F.vehicle_mask(gtl[ma])]
    if not len(veh):
        return veh, None
    # headline = hidden GT vehicle nearest the ego (most salient)
    head = veh[np.argmin(np.linalg.norm(gt[veh, :2], axis=1))]
    return veh, head


def pick_scene(bundle, thr):
    best, best_key = None, (-1, 1e18)
    for e in bundle:
        veh, head = hidden_vehicle(e, thr)
        if head is None:
            continue
        ncorr = len(F.classify(e['pred_clean'], e['gt_boxes'], thr)['correct'])
        key = (len(veh), ncorr)          # most hidden vehicles, then fewest boxes (tie-break)
        if (key[0], -key[1]) > (best_key[0], -best_key[1]):
            best, best_key = e, key
    return best


def draw(entry, cam, which, preds, thr, hidden_gt=None):
    """Camera image: green correct dets (+ solid-red hidden GT on the attacked panel)."""
    img = entry['img_%s' % which][cam].copy()
    c = F.classify(preds, entry['gt_boxes'], thr)
    b, s, l = preds['boxes'], preds['scores'], preds['labels']
    if hidden_gt is not None and len(hidden_gt):
        F._draw_boxes_on(img, entry, cam, entry['gt_boxes'][hidden_gt], F.SEM_BGR['missed'], 4)
    F._draw_boxes_on(img, entry, cam, b[c['correct']], F.SEM_BGR['correct'], 2,
                     s[c['correct']], l[c['correct']])
    return F.to_rgb(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', default='figs/fig_transfer_hiding.png')
    ap.add_argument('--scene', type=int, default=None)
    ap.add_argument('--score-thr', type=float, default=0.4)
    ap.add_argument('--victim', default='BEVFusion', help='name for the panel subtitle')
    args = ap.parse_args()
    thr = args.score_thr

    bundle = F.load_bundle(args.bundle)
    e = next((x for x in bundle if x['scene'] == args.scene), None) if args.scene is not None \
        else pick_scene(bundle, thr)
    if e is None:
        raise SystemExit('no scene with a GT-matched hidden vehicle; try another bundle/thr')

    veh, head = hidden_vehicle(e, thr)
    if head is None:
        raise SystemExit(f'scene {e["scene"]}: attack hides no GT-matched vehicle here; try another')
    gt = e['gt_boxes']
    cam = F.pick_camera(e, gt[head, :3])
    pc, pa = e['pred_clean'], e['pred_attack']

    img_clean = draw(e, cam, 'clean', pc, thr)                      # green only
    img_atk = draw(e, cam, 'attack', pa, thr, hidden_gt=veh)        # green + red hidden

    H, W = e['img_clean'].shape[1:3]
    fig, ax = F.paired_grid(W / H, side=4.5,
                            headers=(f'Clean ({args.victim})', f'Attacked ({args.victim})'))
    for a, im in ((ax[0][0], img_clean), (ax[0][1], img_atk)):
        a.imshow(im, aspect='auto'); a.set_xticks([]); a.set_yticks([])

    cc = F.classify(pc, gt, thr)
    ca = F.classify(pa, gt, thr, ref=pc)
    RNG = 46.0
    clean_sets = [(pc['boxes'][cc['correct']] if len(cc['correct']) else np.zeros((0, 7)),
                   F.SEM_RGB['correct'], 1.8, 'correct detection')]
    atk_sets = [(pa['boxes'][ca['correct']] if len(ca['correct']) else np.zeros((0, 7)),
                 F.SEM_RGB['correct'], 1.8, 'correct detection'),
                (gt[veh], F.SEM_RGB['missed'], 2.6, 'hidden by attack', '-')]
    F.draw_bev(ax[1][0], e['points'], clean_sets, rng=RNG, axis_labels=False)
    F.draw_bev(ax[1][1], e['points'], atk_sets, rng=RNG, axis_labels=False)
    for a in (ax[1][0], ax[1][1]):                                  # circle the headline target
        a.plot(gt[head, 1], gt[head, 0], marker='o', ms=16, mfc='none', mec='k', mew=1.6, zorder=5)
    h = [plt.Line2D([0], [0], color=F.SEM_RGB['correct'], lw=3, label='correct detection'),
         plt.Line2D([0], [0], color=F.SEM_RGB['missed'], lw=3, label='hidden by transferred attack')]
    leg = ax[1][0].legend(handles=h, loc='lower left', fontsize=9, framealpha=0.92, borderpad=0.6)
    leg.get_frame().set_edgecolor('0.7')
    F.savefig(fig, args.out)
    print('scene', e['scene'], '| hidden vehicles', len(veh), '-> wrote', args.out)


if __name__ == '__main__':
    main()
