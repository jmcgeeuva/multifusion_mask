#!/usr/bin/env python3
"""Composite an IS-Fusion universal camouflage texture onto the RAW nuScenes val
camera images, for the cross-model (BEVFusion) transfer eval.

The attack optimised a 132x132 texture bank (bank_last.npy, [k,132,132,3] in [0,1],
RGB) that `attack_multi.render` pastes in `--place bbox` mode: at eval time the full
132x132 param is bilinearly resized to each vehicle instance mask's 2-D bounding box
and written where the mask is True. We reproduce exactly that compositing, but on the
full-resolution 1600x900 raw JPG (no pipeline aug-warp), so the perturbation is a
property of the IMAGE and any detector that loads the JPG inherits it.

Both IS-Fusion and MIT-BEVFusion load images with PIL `Image.open` (RGB), so the
RGB texture pastes with no channel swap. Only camera images that carry >=1 vehicle
mask are written; the eval falls back to the original image for every other basename.

  python tools/composite_transfer.py --bank <bank_last.npy> --out <dir> \
      --infos <nuscenes_infos_val.pkl> [--vehicles car,truck,bus] [--workers 16] [--limit N]
"""
import argparse
import os
import os.path as osp
import pickle
import numpy as np
from PIL import Image
from multiprocessing import Pool

MASK_ROOT = '/standard/eng_vivastorage/nuscenes/nuscenes/nuscenes_masks'
RAW_ROOT = '/standard/eng_vivastorage/nuscenes/nuscenes'   # + last-4 path comps -> raw .jpg

# globals populated per worker by _init
TEX = IDX = VEH = OUT = RANDOM = None


def load_index(path):
    """path_key -> [(mask_file, cls)]; mirrors grad_check.load_mask_index fields."""
    idx = {}
    with open(path) as fh:
        for line in fh:
            p = line.rstrip('\n').split('\t')
            if len(p) < 3:
                continue
            cls = p[3] if len(p) >= 4 else ''
            idx.setdefault(p[0], []).append((osp.join(p[1], p[2]), cls))
    return idx


def masks_for(dp):
    key = osp.normpath(osp.join(MASK_ROOT, '/'.join(dp.split('/')[-4:]).replace('.jpg', '')))
    return IDX.get(key, [])


def raw_path(dp):
    return osp.join(RAW_ROOT, '/'.join(dp.split('/')[-4:]))


def _init(bank_path, idx_path, vehicles, out_root, random_tex):
    global TEX, IDX, VEH, OUT, RANDOM
    b = np.load(bank_path)                       # (k,132,132,3) float [0,1], RGB
    TEX = (b[0].clip(0, 1) * 255).astype(np.uint8)
    IDX = load_index(idx_path)
    VEH = set(vehicles)
    OUT = out_root
    RANDOM = random_tex
    if random_tex:                               # control: a fixed random texture, same seed everywhere
        rng = np.random.RandomState(0)
        TEX = (rng.rand(*TEX.shape) * 255).astype(np.uint8)


def process(dp):
    """Paint every vehicle instance mask on one camera image; return #masks painted."""
    masks = [(f, c) for f, c in masks_for(dp) if c in VEH]
    if not masks:
        return 0
    img = np.asarray(Image.open(raw_path(dp)).convert('RGB')).copy()   # (H,W,3) RGB
    H, W = img.shape[:2]
    tex_im = Image.fromarray(TEX)
    painted = 0
    for f, _c in masks:
        try:
            m = Image.open(f).convert('L').resize((W, H), Image.NEAREST)
        except (FileNotFoundError, OSError):
            continue
        m = np.asarray(m) > 127
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        t = np.asarray(tex_im.resize((x1 - x0, y1 - y0), Image.BILINEAR))
        sub, sm = img[y0:y1, x0:x1], m[y0:y1, x0:x1]
        sub[sm] = t[sm]
        painted += 1
    if painted:
        Image.fromarray(img).save(osp.join(OUT, osp.basename(dp)), quality=95)
    return painted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--infos', required=True, help='nuScenes val infos pkl (defines the sample set)')
    ap.add_argument('--index', default='./nuscenes_masks_index.txt')
    ap.add_argument('--vehicles', default='car,truck,bus')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='>0: only first N samples (smoke)')
    ap.add_argument('--random', action='store_true', help='random-texture control instead of the bank')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    vehicles = [v.strip() for v in args.vehicles.split(',') if v.strip()]

    d = pickle.load(open(args.infos, 'rb'))
    infos = d['infos'] if isinstance(d, dict) else d
    if args.limit:
        infos = infos[:args.limit]
    dps = [cam['data_path'] for info in infos for cam in info['cams'].values()]
    print(f'{len(infos)} samples -> {len(dps)} camera images; vehicles={vehicles} '
          f'random={args.random}', flush=True)

    with Pool(args.workers, initializer=_init,
              initargs=(args.bank, args.index, vehicles, args.out, args.random)) as pool:
        counts = pool.map(process, dps, chunksize=64)

    imgs_written = sum(1 for c in counts if c > 0)
    print(f'DONE: wrote {imgs_written} perturbed images ({sum(counts)} vehicle masks painted) '
          f'-> {args.out}', flush=True)


if __name__ == '__main__':
    main()
