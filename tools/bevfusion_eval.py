#!/usr/bin/env python3
"""Single-GPU clean/transfer eval for MIT-BEVFusion, no MPI / torchpack.distributed.

Mirrors bevfusion/tools/test.py config+model construction but forces the
non-distributed path (MMDataParallel + single_gpu_test), so it needs no mpi4py.

Two modes:
  --limit N   : run N samples, print a detection summary, skip official eval
                (fast smoke that the CUDA ops + data format + checkpoint work).
  (no limit)  : full val, official nuScenes evaluate() -> mAP / NDS.

Optional --img-root <dir> rewrites every camera data_path to <dir>/<basename-of
-original-relative-path> so a directory of pre-composited adversarial images can
be swapped in for the transfer eval (clean path untouched when omitted).

  python tools/bevfusion_eval.py <config> <checkpoint> --eval bbox
  python tools/bevfusion_eval.py <config> <checkpoint> --limit 8
"""
import argparse
import os
import numpy as np
import torch
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack.utils.config import configs
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--eval', nargs='+', default=['bbox'])
    ap.add_argument('--limit', type=int, default=0,
                    help='>0: run only this many samples, print summary, no official eval')
    ap.add_argument('--img-root', default=None,
                    help='swap camera image root (for transfer eval on composited images)')
    ap.add_argument('--out', default=None,
                    help='.pkl of raw outputs (full mode auto-saves one if omitted)')
    ap.add_argument('--from-pkl', default=None,
                    help='skip inference; load this outputs .pkl and just run evaluate() (CPU)')
    ap.add_argument('--no-eval', action='store_true',
                    help='run inference + save outputs .pkl but skip the slow evaluate() step')
    ap.add_argument('--fp16', action='store_true',
                    help='wrap model in fp16 (off by default; triggers spconv N=0 bug)')
    ap.add_argument('--data-root', default=None,
                    help='override dataset_root (e.g. local-staged nuScenes to avoid slow '
                         'ceph table loads in the nuScenes metric step)')
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.from_pkl and torch.cuda.is_available():   # --from-pkl scores on CPU, no GPU
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    if args.data_root:                       # point dataset + nuScenes-eval at staged root
        cfg.data.test.dataset_root = args.data_root
        if 'ann_file' in cfg.data.test:
            cfg.data.test.ann_file = os.path.join(
                args.data_root, os.path.basename(cfg.data.test.ann_file))

    # Strip pipeline steps the detection model + our data don't need:
    #  - LoadRadarPointsMultiSweeps: our infos carry no radar (get_data_info pops
    #    'radar') -> KeyError; the camera+lidar det model never consumes radar.
    #  - LoadBEVSegmentation: needs the nuScenes map-expansion JSONs we don't have,
    #    and the detection config has no segmentation head.
    _DROP = ('Radar', 'BEVSegmentation')
    _DROP_KEYS = ('radar', 'gt_masks_bev')
    pipe = cfg.data.test.get('pipeline', [])
    pipe = [t for t in pipe if not any(d in t.get('type', '') for d in _DROP)]
    for t in pipe:                      # drop unused keys from Collect3D/format
        if 'keys' in t and isinstance(t['keys'], (list, tuple)):
            t['keys'] = [k for k in t['keys'] if k not in _DROP_KEYS]
    cfg.data.test.pipeline = pipe

    dataset = build_dataset(cfg.data.test)

    if args.limit and args.limit > 0:
        dataset.data_infos = dataset.data_infos[:args.limit]
        if hasattr(dataset, 'flag'):
            dataset.flag = dataset.flag[:args.limit]

    # optional image-root swap for transfer eval (keep basename, change dir).
    # The compositor writes ONLY perturbed images (those with a vehicle mask); for
    # every other camera image fall back to the original path so the clean regions
    # are byte-identical to the clean baseline.
    if args.img_root:
        n_swap = 0
        for info in dataset.data_infos:
            for cam in info['cams'].values():
                cand = os.path.join(args.img_root, os.path.basename(cam['data_path']))
                if os.path.exists(cand):
                    cam['data_path'] = cand
                    n_swap += 1
        print(f'--img-root: {n_swap} camera images swapped to {args.img_root} '
              f'(rest fall back to original)', flush=True)

    # --from-pkl: skip model + inference entirely, just score saved outputs (CPU).
    if args.from_pkl:
        outputs = mmcv.load(args.from_pkl)
        print(f'loaded {len(outputs)} outputs from {args.from_pkl}; running evaluate()...', flush=True)
        import time as _t; t0 = _t.time()
        metrics = dataset.evaluate(outputs, metric=args.eval)
        print(f'evaluate() took {_t.time()-t0:.0f}s', flush=True)
        print('ALL METRICS:', metrics)
        return

    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # NOTE: wrap_fp16_model makes the bundled spconv indice generation hit an
    # N=0 kernel-launch error on this torch/mmcv/spconv stack; fp32 eval is
    # numerically fine for mAP/NDS, so skip fp16 wrapping unless --fp16 given.
    if args.fp16 and cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    ckpt = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = ckpt.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model = MMDataParallel(model, device_ids=[0])

    outputs = single_gpu_test(model, data_loader)

    # Always persist inference so the slow nuScenes metric step can't lose it.
    out_path = args.out
    if not out_path and not (args.limit and args.limit > 0):
        out_path = os.path.join('logs/transfer',
                                f'outputs_{os.environ.get("SLURM_JOB_ID","x")}.pkl')
    if out_path:
        mmcv.dump(outputs, out_path)
        print(f'saved outputs -> {out_path}', flush=True)

    if args.limit and args.limit > 0:
        print(f'\n=== SMOKE: {len(outputs)} samples ===')
        for i, o in enumerate(outputs):
            pts = o['pts_bbox'] if 'pts_bbox' in o else o
            scores = pts['scores_3d'].numpy() if torch.is_tensor(pts['scores_3d']) else np.asarray(pts['scores_3d'])
            n_hi = int((scores > 0.3).sum())
            print(f'  sample {i}: {len(scores):4d} boxes, {n_hi:3d} with score>0.3, '
                  f'max={scores.max():.3f}' if len(scores) else f'  sample {i}: 0 boxes')
        print('SMOKE OK')
        return

    if args.no_eval:
        print('--no-eval: inference saved, skipping evaluate()', flush=True)
        return

    import time as _t; t0 = _t.time()
    print('running nuScenes evaluate() (slow: loads GT tables)...', flush=True)
    metrics = dataset.evaluate(outputs, metric=args.eval)
    print(f'\n=== BEVFusion clean eval (evaluate {_t.time()-t0:.0f}s) ===')
    print('ALL METRICS:', metrics)


if __name__ == '__main__':
    main()
