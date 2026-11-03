#!/usr/bin/env python3
"""Does the transferred attack hide LiDAR-SPARSE vehicles preferentially?

For every GT vehicle (car/truck/bus): count LiDAR points inside its 2D footprint,
decide if it is DETECTED clean (a confident det within tol) and HIDDEN under attack
(detected clean, no confident attacked det within tol). Then:
  (1) attack-success rate binned by on-target LiDAR point count   [sparsity hypothesis]
  (2) attack-success rate binned by #vehicles in the scene        [multi-vehicle angle]

Run in the victim's env so the det pkls unpickle (bevfusion_env / cmt_env):
  python tools/analyze_lidar_density.py --clean c.pkl --attack a.pkl \
      --infos infos.pkl --out logs/transfer/density_bevf.txt [--thr .3 --tol 2.0]
"""
import argparse, os.path as osp, pickle
import numpy as np

CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
VEH = {'car', 'truck', 'bus'}


def dets_to_np(o):
    d = o['pts_bbox'] if 'pts_bbox' in o else o
    b = d['boxes_3d']
    boxes = b.tensor.detach().cpu().numpy() if hasattr(b, 'tensor') else np.asarray(b)
    s, l = d['scores_3d'], d['labels_3d']
    s = s.detach().cpu().numpy() if hasattr(s, 'detach') else np.asarray(s)
    l = l.detach().cpu().numpy() if hasattr(l, 'detach') else np.asarray(l)
    return boxes.astype(np.float32), s.astype(np.float32), l.astype(np.int64)


def raw_path(dp, root):
    return osp.join(root, '/'.join(dp.split('/')[-4:]))


def pts_in_footprint(pts_xy, cx, cy, dx, dy, yaw, margin=0.1):
    """Count points inside the rotated 2D box footprint (z ignored: robust to origin)."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    px = pts_xy[:, 0] - cx; py = pts_xy[:, 1] - cy
    xr = c * px - s * py; yr = s * px + c * py
    return int(((np.abs(xr) <= dx / 2 + margin) & (np.abs(yr) <= dy / 2 + margin)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', required=True)
    ap.add_argument('--attack', required=True)
    ap.add_argument('--infos', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--raw-root', default='/standard/eng_vivastorage/nuscenes/nuscenes')
    ap.add_argument('--thr', type=float, default=0.3)
    ap.add_argument('--tol', type=float, default=2.0)
    ap.add_argument('--name', default='victim')
    args = ap.parse_args()
    import mmcv
    clean, attack = mmcv.load(args.clean), mmcv.load(args.attack)
    infos = pickle.load(open(args.infos, 'rb'))
    infos = infos['infos'] if isinstance(infos, dict) else infos
    assert len(clean) == len(attack) == len(infos), (len(clean), len(attack), len(infos))

    rows = []            # (npts, hit_clean, hidden, nveh_scene)
    for i in range(len(infos)):
        info = infos[i]
        gt = np.asarray(info['gt_boxes'], np.float32)
        names = np.asarray(info['gt_names'])
        if not len(gt):
            continue
        vmask = np.array([n in VEH for n in names])
        if not vmask.any():
            continue
        cb, cs, _ = dets_to_np(clean[i]); ab, as_, _ = dets_to_np(attack[i])
        cc = cb[cs >= args.thr, :2] if (cs >= args.thr).any() else np.zeros((0, 2))
        aa = ab[as_ >= args.thr, :2] if (as_ >= args.thr).any() else np.zeros((0, 2))
        lp = raw_path(info['lidar_path'], args.raw_root)
        pts = np.fromfile(lp, dtype=np.float32).reshape(-1, 5)[:, :2]
        nveh = int(vmask.sum())
        for j in np.where(vmask)[0]:
            cx, cy = gt[j, 0], gt[j, 1]
            npts = pts_in_footprint(pts, cx, cy, gt[j, 3], gt[j, 4], gt[j, 6])
            hit = len(cc) and (np.linalg.norm(cc - [cx, cy], axis=1) < args.tol).any()
            if not hit:
                continue                      # only vehicles the clean model actually found
            hidden = not (len(aa) and (np.linalg.norm(aa - [cx, cy], axis=1) < args.tol).any())
            rows.append((npts, 1, int(hidden), nveh))
    R = np.array(rows)
    npts, hidden, nveh = R[:, 0], R[:, 2], R[:, 3]
    N = len(R); H = int(hidden.sum())

    def rate(mask):
        m = mask & (np.ones(N, bool))
        return (hidden[m].mean() if m.any() else float('nan')), int(m.sum())

    lines = [f'=== {args.name}: LiDAR-sparsity vs transferred-attack hiding ===',
             f'clean-detected vehicles: {N}   hidden under attack: {H} ({100*H/N:.1f}%)', '',
             '(1) hide-rate by on-target LiDAR point count']
    edges = [0, 5, 20, 50, 150, 500, 10**9]
    labs = ['0-4', '5-19', '20-49', '50-149', '150-499', '500+']
    for lo, hi, lab in zip(edges[:-1], edges[1:], labs):
        r, n = rate((npts >= lo) & (npts < hi))
        lines.append(f'  {lab:>8} pts : hide {100*r:5.1f}%   (n={n})')
    # Spearman-ish: correlation of hidden vs log-points
    lp_ = np.log1p(npts)
    if np.std(lp_) > 0:
        corr = np.corrcoef(lp_, hidden)[0, 1]
        lines.append(f'  corr(hidden, log points) = {corr:+.3f}  (negative = sparse easier)')
    lines += ['', '(2) hide-rate by #vehicles in scene (multi-vehicle angle)']
    for lo, hi, lab in [(1, 2, '1'), (2, 4, '2-3'), (4, 7, '4-6'), (7, 11, '7-10'), (11, 999, '11+')]:
        r, n = rate((nveh >= lo) & (nveh < hi))
        lines.append(f'  {lab:>6} veh : hide {100*r:5.1f}%   (n={n})')
    if np.std(nveh.astype(float)) > 0:
        lines.append(f'  corr(hidden, #veh) = {np.corrcoef(nveh, hidden)[0,1]:+.3f}')
    out = '\n'.join(lines)
    print(out)
    open(args.out, 'w').write(out + '\n')


if __name__ == '__main__':
    main()
