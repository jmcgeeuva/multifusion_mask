#!/usr/bin/env python
"""
Evaluate a saved results pkl against the nuScenes val split.

Decouples evaluation from inference so you can re-evaluate with different
--attack-filter settings without re-running the model.

Usage:
    python eval_results.py <config> <results.pkl> [options]

Examples:
    # Standard full-split eval
    python eval_results.py IS-Fusion/configs/isfusion/isfusion_0075voxel.py \
        results/attack_results.pkl

    # Sample+class filter (requires the attack_log saved alongside the pkl)
    python eval_results.py IS-Fusion/configs/isfusion/isfusion_0075voxel.py \
        results/attack_results.pkl --attack-filter sample_class

    # Instance-level filter
    python eval_results.py IS-Fusion/configs/isfusion/isfusion_0075voxel.py \
        results/attack_results.pkl --attack-filter instance

    # Explicit attack-log path (defaults to <pkl>_attack_log.json)
    python eval_results.py IS-Fusion/configs/isfusion/isfusion_0075voxel.py \
        results/attack_results.pkl --attack-filter sample_class \
        --attack-log results/attack_results_attack_log.json
"""

import argparse
import os
import sys

os.environ['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), 'IS-Fusion') \
    + ':' + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IS-Fusion'))

import mmcv
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a saved results pkl against the nuScenes val split.')
    parser.add_argument('config', help='IS-Fusion config file')
    parser.add_argument('pkl', help='results pkl produced by test.py --out')
    parser.add_argument(
        '--attack-filter',
        choices=['none', 'sample_class', 'instance'],
        default='none',
        help='"none": full-split eval. '
             '"sample_class": restrict to attacked samples and their class. '
             '"instance": additionally match the specific attacked GT box via 2D projection.')
    parser.add_argument(
        '--attack-log',
        default=None,
        help='Path to the attack_log JSON written by test.py. '
             'Defaults to <pkl>_attack_log.json when --attack-filter != none.')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Directory for nuScenes JSON output. Defaults to a temp dir.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options, e.g. data.test.ann_file=path/to/infos.pkl')
    return parser.parse_args()


def load_attack_log(args):
    if args.attack_filter == 'none':
        return None
    log_path = args.attack_log or args.pkl.replace('.pkl', '_attack_log.json')
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f'Attack log not found at {log_path}. '
            f'Re-run test.py with --attack-filter {args.attack_filter} --out {args.pkl} '
            f'to generate it, or pass --attack-log <path> explicitly.'
        )
    attack_log = mmcv.load(log_path)
    if len(attack_log) == 0:
        raise ValueError(
            f'Attack log at {log_path} is empty (0 samples). '
            f'The attack did not fire during inference — check that mask_img '
            f'has non-zero pixels and that the camouflage path is correct.'
        )
    print(f'Loaded attack log: {len(attack_log)} attacked samples')
    return attack_log


def main():
    args = parse_args()

    # ── Load config and build dataset ────────────────────────────────────────
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    # ── Load predictions ─────────────────────────────────────────────────────
    if not os.path.exists(args.pkl):
        raise FileNotFoundError(f'Results pkl not found: {args.pkl}')
    results = mmcv.load(args.pkl)
    print(f'Loaded {len(results)} predictions from {args.pkl}')

    # ── Load attack log if filtering ─────────────────────────────────────────
    attack_log = load_attack_log(args)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    eval_kwargs = dict(
        metric='bbox',
        attack_log=attack_log,
        attack_filter=args.attack_filter,
    )
    if args.out_dir:
        eval_kwargs['jsonfile_prefix'] = args.out_dir

    print(f'\n{"="*48}')
    print(f'  Attack filter : {args.attack_filter}')
    if attack_log:
        print(f'  Attacked samples evaluated: {len(attack_log)}')
    print(f'{"="*48}\n')

    metrics = dataset.evaluate(results, **eval_kwargs)

    # ── Print summary ─────────────────────────────────────────────────────────
    prefix = next(iter(metrics)).rsplit('/', 1)[0]  # e.g. 'pts_bbox_NuScenes'
    mAP = metrics.get(f'{prefix}/mAP', metrics.get('mAP'))
    NDS = metrics.get(f'{prefix}/NDS', metrics.get('NDS'))

    print(f'\n{"="*48}')
    print(f'  mAP : {mAP:.4f}')
    print(f'  NDS : {NDS:.4f}')
    print(f'{"="*48}')

    # Per-class APs
    classes = dataset.CLASSES
    print('\nPer-class AP (averaged over distance thresholds):')
    for cls in classes:
        aps = [metrics.get(f'{prefix}/{cls}_AP_dist_{d}') for d in [0.5, 1.0, 2.0, 4.0]]
        aps = [a for a in aps if a is not None]
        if aps:
            print(f'  {cls:25s}: {sum(aps)/len(aps):.4f}')


if __name__ == '__main__':
    main()
