#!/usr/bin/env python3
"""Build a fig_lib figure-bundle for a cross-model TRANSFER victim (BEVFusion / CMT),
so tools/fig2_hiding.py can render the clean-vs-attacked vanishing-detection figure on a
detector the texture was NEVER optimized against.

Mirrors attack_multi.make_dump_entry's schema but with pure-numpy detections (no mmdet3d
objects), so the figure can be drawn in a different env. The perturbed images are the raw
composited JPGs, so projection is at RAW resolution: img_aug = lidar_aug = identity and
lidar2img is built from each camera's intrinsic + sensor2lidar extrinsic in the infos.

Run in the victim's env (to unpickle its output boxes), e.g. bevfusion_env from the
bevfusion repo, or cmt_env from the cmt repo:

  python tools/build_transfer_bundle.py --clean <clean.pkl> --attack <none.pkl> \
      --infos <val_infos.pkl> --img-root <composited_dir> --out bundle_bevf.pkl \
      [--raw-root /standard/eng_vivastorage/nuscenes/nuscenes] [--topk 8]
"""
import argparse, os, os.path as osp, pickle
import numpy as np
import cv2

# nuScenes detection class order (car/truck/bus are the painted vehicle classes)
CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
VEH_IDS = {CLASSES.index(c) for c in ('car', 'truck', 'bus')}


def dets_to_np(o):
    """One single_gpu_test output -> dict(boxes[N,7+], scores[N], labels[N]) as numpy."""
    d = o['pts_bbox'] if 'pts_bbox' in o else o
    b = d['boxes_3d']
    boxes = b.tensor.detach().cpu().numpy() if hasattr(b, 'tensor') else np.asarray(b)
    s = d['scores_3d']; l = d['labels_3d']
    s = s.detach().cpu().numpy() if hasattr(s, 'detach') else np.asarray(s)
    l = l.detach().cpu().numpy() if hasattr(l, 'detach') else np.asarray(l)
    return dict(boxes=boxes.astype(np.float32), scores=s.astype(np.float32),
                labels=l.astype(np.int64))


def lidar2img_raw(cam):
    """lidar -> raw image pixels: intrinsic @ lidar2cam, from sensor2lidar extrinsic."""
    R = np.asarray(cam['sensor2lidar_rotation'], np.float64)       # cam->lidar rot
    t = np.asarray(cam['sensor2lidar_translation'], np.float64)
    l2c = np.eye(4)
    l2c[:3, :3] = R.T
    l2c[:3, 3] = -R.T @ t
    K = np.eye(4); K[:3, :3] = np.asarray(cam['cam_intrinsic'], np.float64)
    return (K @ l2c).astype(np.float32)


def raw_path(dp, raw_root):
    return osp.join(raw_root, '/'.join(dp.split('/')[-4:]))


def n_hidden_vehicles(pc, pa, gt, gtl, thr=0.3, tol=2.0):
    """Confident clean vehicle dets that (a) match a real GT vehicle AND (b) vanish under
    attack (no attacked det within tol). Requiring the GT match keeps the headline target a
    REAL hidden car, not a roadside ghost. Returns (count, max confidence)."""
    veh = (pc['scores'] >= thr) & np.isin(pc['labels'], list(VEH_IDS))
    if not veh.any():
        return 0, 0.0
    gtv = gt[np.isin(gtl, list(VEH_IDS)), :2] if len(gt) else np.zeros((0, 2))
    # "vanished" = no CONFIDENT attacked det nearby; the raw pkl keeps ~500 sub-threshold boxes,
    # so threshold the attacked side too or a crushed 0.8->0.1 box still counts as present.
    akeep = pa['scores'] >= thr
    ab = pa['boxes'][akeep, :2] if akeep.any() else np.zeros((0, 2))
    n, smax = 0, 0.0
    for idx in np.where(veh)[0]:
        c = pc['boxes'][idx, :2]
        matched = len(gtv) and (np.linalg.norm(gtv - c, axis=1) < tol).any()
        vanished = not (len(ab) and (np.linalg.norm(ab - c, axis=1) < tol).any())
        if matched and vanished:
            n += 1
            smax = max(smax, float(pc['scores'][idx]))
    return n, smax


def _emit(pick, clean, attack, infos, args):
    """Build full bundle entries for the chosen scene indices and write the pkl."""
    bundle = []
    for i in pick:
        info = infos[i]
        cams = list(info['cams'].items())                       # fixed order
        l2i, imgs_c, imgs_a = [], [], []
        for _name, cam in cams:
            l2i.append(lidar2img_raw(cam))
            rp = raw_path(cam['data_path'], args.raw_root)
            cp = osp.join(args.img_root, osp.basename(cam['data_path']))
            ic = cv2.imread(rp)                                  # BGR
            ia = cv2.imread(cp) if osp.exists(cp) else ic
            imgs_c.append(ic); imgs_a.append(ia)
        lp = raw_path(info['lidar_path'], args.raw_root)
        pts = np.fromfile(lp, dtype=np.float32).reshape(-1, 5)[:, :3]
        gt = np.asarray(info['gt_boxes'], np.float32)
        gtl = np.asarray([CLASSES.index(n) if n in CLASSES else -1 for n in info['gt_names']], np.int64)
        bundle.append(dict(
            scene=int(i), file=osp.basename(info['lidar_path']), place='bbox', channels='bgr',
            img_clean=np.stack(imgs_c), img_attack=np.stack(imgs_a),
            lidar2img=np.stack(l2i).astype(np.float32),
            img_aug=np.stack([np.eye(4, dtype=np.float32)] * len(cams)),
            lidar_aug=np.eye(4, dtype=np.float32),
            cam_files=[osp.basename(c['data_path']) for _n, c in cams],
            points=pts.astype(np.float32),
            gt_boxes=gt, gt_labels=gtl, attacked_gt=[],
            pred_clean=dets_to_np(clean[i]), pred_attack=dets_to_np(attack[i])))
        print(f'  scene {i}: built', flush=True)
    pickle.dump(bundle, open(args.out, 'wb'))
    print(f'wrote {len(bundle)} scenes -> {args.out}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', required=True)
    ap.add_argument('--attack', required=True)
    ap.add_argument('--infos', required=True)
    ap.add_argument('--img-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--raw-root', default='/standard/eng_vivastorage/nuscenes/nuscenes')
    ap.add_argument('--topk', type=int, default=8, help='build bundle for the K best hiding scenes')
    ap.add_argument('--thr', type=float, default=0.3)
    ap.add_argument('--max-veh', type=int, default=5, help='max confident vehicle dets (sparse scenes)')
    ap.add_argument('--min-conf', type=float, default=0.5, help='min confidence of the hidden vehicle')
    ap.add_argument('--scenes', default='', help='comma-separated scene indices to force (skip auto-selection)')
    ap.add_argument('--token', default='', help='build the scene whose info token matches this (skip selection)')
    args = ap.parse_args()

    import mmcv
    clean = mmcv.load(args.clean)
    attack = mmcv.load(args.attack)
    infos = pickle.load(open(args.infos, 'rb'))
    infos = infos['infos'] if isinstance(infos, dict) else infos
    assert len(clean) == len(attack) == len(infos), (len(clean), len(attack), len(infos))

    # explicit scene / token override (to show the SAME physical scene across victims)
    if args.scenes or args.token:
        if args.token:
            pick = [i for i in range(len(infos)) if infos[i].get('token') == args.token]
        else:
            pick = [int(x) for x in args.scenes.split(',') if x.strip()]
        print(f'forced scenes {pick}', flush=True)
        _emit(pick, clean, attack, infos, args)
        return

    # Prefer a CLEAN story: a sparse (road, not parking-lot) scene with a confident, isolated
    # vehicle the attack hides. Rank by the hidden vehicle's confidence, keep only scenes with
    # few total confident vehicle dets (so the figure is legible, not a wall of boxes).
    scored = []
    for i in range(len(clean)):
        pc, pa = dets_to_np(clean[i]), dets_to_np(attack[i])
        info = infos[i]
        gt = np.asarray(info['gt_boxes'], np.float32)
        gtl = np.asarray([CLASSES.index(n) if n in CLASSES else -1 for n in info['gt_names']], np.int64)
        ntot = int(((pc['scores'] >= args.thr) & np.isin(pc['labels'], list(VEH_IDS))).sum())
        n, smax = n_hidden_vehicles(pc, pa, gt, gtl, args.thr)
        if n >= 1 and ntot <= args.max_veh and smax >= args.min_conf:
            scored.append((smax, -ntot, i))       # most-confident hidden real vehicle, fewest boxes
    scored.sort(reverse=True)
    pick = [i for _, _, i in scored[:args.topk]]
    print(f'{len(scored)} sparse scenes hide a confident vehicle; building {pick}', flush=True)
    _emit(pick, clean, attack, infos, args)


if __name__ == '__main__':
    main()
