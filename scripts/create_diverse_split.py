#!/usr/bin/env python3
"""
Create a diverse temporal split of a NuScenes PKL annotation file for quick testing.

Diversity strategy:
  1. Group all samples by scene (extracted from lidar_path basename)
  2. Allocate the target count across scenes proportionally to scene size,
     so every scene contributes roughly split_percent of its frames
  3. Within each scene, divide into equal temporal segments and pick the
     highest-scoring sample from each segment (score = GT object count +
     log-scaled mask pixel coverage from nuscenes_masks_index.txt)

This avoids over-representing any single scene or run of consecutive frames
while biasing selection toward object-rich moments.

Output: a plain-text file with one sample token per line, readable by the
patched NuScenesDataset (split_file= parameter).

Usage:
    python scripts/create_diverse_split.py --split-percent 10
    python scripts/create_diverse_split.py --split-percent 5 --pkl ./data/nuscenes/nuscenes_infos_train.pkl
    python scripts/create_diverse_split.py --split-percent 10 --num-workers 8
"""

import argparse
import math
import os
import pickle
import re
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


# ---------------------------------------------------------------------------
# Multiprocessing worker — must be top-level so it is picklable
# ---------------------------------------------------------------------------

def _parse_lines_chunk(lines: list) -> dict:
    """Parse a chunk of nuscenes_masks_index.txt lines.

    Returns dict: cam_key -> total pixel area (int)
    """
    counts: dict = {}
    for line in lines:
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 5:
            continue
        try:
            cam_key = os.path.basename(parts[0].rstrip('/'))
            pixel_count = int(parts[4])
            counts[cam_key] = counts.get(cam_key, 0) + pixel_count
        except (ValueError, IndexError):
            continue
    return counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_scene_id(lidar_path: str) -> str:
    """Return the scene prefix from a lidar_path.

    Example:
      './data/nuscenes/train/samples/LIDAR_TOP/n015-2018-08-02-17-16-37+0800__LIDAR_TOP__1533201470448696.pcd.bin'
      → 'n015-2018-08-02-17-16-37+0800'
    """
    basename = os.path.basename(lidar_path)
    m = re.match(r'(.+?)__LIDAR_TOP__\d+\.pcd\.bin', basename)
    return m.group(1) if m else basename


def load_mask_pixel_counts(masks_index_path: str, num_workers: int = 1) -> dict:
    """Parse nuscenes_masks_index.txt in parallel and accumulate pixel area per camera-frame key.

    File format (tab-separated):
      <cam_dir>  <class_dir>  <mask_file>  <class_name>  <pixel_count>

    Returns dict: cam_key -> total pixel area (int)
    """
    if not os.path.exists(masks_index_path):
        return {}

    with open(masks_index_path) as f:
        all_lines = f.readlines()

    if not all_lines:
        return {}

    n_workers = max(1, min(num_workers, len(all_lines)))
    chunk_size = math.ceil(len(all_lines) / n_workers)
    chunks = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]

    merged: dict = {}
    if n_workers == 1:
        merged = _parse_lines_chunk(chunks[0])
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_parse_lines_chunk, chunk) for chunk in chunks]
            for fut in as_completed(futures):
                for k, v in fut.result().items():
                    merged[k] = merged.get(k, 0) + v

    return merged


NUSC_DETECTION_CLASSES = frozenset([
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
])


def filter_objectless_samples(data_infos: list, mask_counts: dict) -> tuple:
    """Remove samples that have no detectable objects across all views.

    A sample is considered empty if:
      - Its gt_names contains no NuScenes detection-class objects, AND
      - It has no mask pixel coverage across any camera (when mask_counts available)

    Both conditions must hold to drop a sample; if mask_counts is absent only
    the gt_names check is used.

    Returns:
        kept   (list[dict]): samples that have at least one object of interest
        dropped (list[str]): tokens of the dropped samples
    """
    kept = []
    dropped = []

    for info in data_infos:
        gt_names = info.get('gt_names', [])
        has_gt = any(n in NUSC_DETECTION_CLASSES for n in gt_names)

        has_mask = False
        if mask_counts:
            for cam_info in info.get('cams', {}).values():
                data_path = cam_info.get('data_path', '')
                cam_key = os.path.splitext(os.path.basename(data_path))[0]
                if mask_counts.get(cam_key, 0) > 0:
                    has_mask = True
                    break

        if has_gt or (mask_counts and has_mask):
            kept.append(info)
        else:
            dropped.append(info['token'])

    return kept, dropped


def compute_sample_scores(data_infos: list, mask_counts: dict) -> list:
    """Compute a richness score for every sample.

    Score = GT object count (primary)
            + log-scaled total mask pixel area across all 6 cameras (secondary).

    The log scaling keeps pixel area from swamping the object count signal.
    If mask_counts is empty (e.g., for val data), only GT count is used.
    """
    scores = []
    for info in data_infos:
        n_objects = len(info.get('gt_names', []))

        pixel_coverage = 0
        if mask_counts:
            for cam_info in info.get('cams', {}).values():
                data_path = cam_info.get('data_path', '')
                cam_key = os.path.splitext(os.path.basename(data_path))[0]
                pixel_coverage += mask_counts.get(cam_key, 0)

        pixel_score = math.log1p(pixel_coverage) / 10.0 if pixel_coverage > 0 else 0.0
        scores.append(n_objects + pixel_score)
    return scores


def diverse_sample(data_infos: list, n_target: int, scores: list, seed: int = 42) -> list:
    """Return indices into data_infos for a diverse subset of size n_target.

    For each scene:
      - Divide its sorted-by-timestamp frames into n_alloc equal temporal segments
      - Pick the highest-scoring frame from each segment
    Allocation per scene is proportional to scene size (largest-remainder rounding).
    """
    random.seed(seed)
    np.random.seed(seed)

    n_total = len(data_infos)
    n_target = min(n_target, n_total)

    scene_map: dict = defaultdict(list)
    for i, info in enumerate(data_infos):
        scene_map[extract_scene_id(info['lidar_path'])].append(i)

    scenes = []
    for scene_id, indices in scene_map.items():
        sorted_idx = sorted(indices, key=lambda i: data_infos[i]['timestamp'])
        scenes.append({
            'id': scene_id,
            'indices': sorted_idx,
            'scores': [scores[i] for i in sorted_idx],
            'size': len(sorted_idx),
        })

    n_scenes = len(scenes)

    # Proportional allocation with largest-remainder rounding to hit exactly n_target
    raw = [n_target * s['size'] / n_total for s in scenes]
    floor_allocs = [max(0, int(r)) for r in raw]
    remainders = sorted(range(n_scenes), key=lambda i: -(raw[i] - floor_allocs[i]))
    shortfall = n_target - sum(floor_allocs)
    for i in remainders[:shortfall]:
        floor_allocs[i] += 1

    selected = []
    for scene, n_alloc in zip(scenes, floor_allocs):
        indices = scene['indices']
        scene_scores = scene['scores']
        n = min(n_alloc, len(indices))
        if n <= 0:
            continue
        if n >= len(indices):
            selected.extend(indices)
            continue

        seg_size = len(indices) / n
        for k in range(n):
            start = int(k * seg_size)
            end = min(int((k + 1) * seg_size), len(indices))
            if start >= end:
                end = start + 1
            best_local = int(np.argmax(scene_scores[start:end]))
            selected.append(indices[start + best_local])

    return sorted(set(selected))


def print_stats(data_infos: list, selected_indices: list) -> None:
    n_scenes_total = len(set(extract_scene_id(i['lidar_path']) for i in data_infos))
    n_scenes_sel = len(set(extract_scene_id(data_infos[i]['lidar_path']) for i in selected_indices))

    all_objs = [len(info.get('gt_names', [])) for info in data_infos]
    sel_objs = [len(data_infos[i].get('gt_names', [])) for i in selected_indices]

    print(f'  Scenes covered : {n_scenes_sel}/{n_scenes_total} ({100*n_scenes_sel/n_scenes_total:.0f}%)')
    print(f'  Mean objects   : {np.mean(sel_objs):.1f} (full set: {np.mean(all_objs):.1f})')
    print(f'  Min/max objects: {min(sel_objs)}/{max(sel_objs)} (full set: {min(all_objs)}/{max(all_objs)})')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create a diverse NuScenes split for quick testing'
    )
    parser.add_argument(
        '--split-percent', type=float, required=True,
        help='Percentage of total samples to keep (0 < value <= 100)',
    )
    parser.add_argument(
        '--pkl', type=str,
        default='./data/nuscenes/nuscenes_infos_val.pkl',
        help='Source NuScenes PKL annotation file',
    )
    parser.add_argument(
        '--masks-index', type=str,
        default='./nuscenes_masks_index.txt',
        help='Path to nuscenes_masks_index.txt (used for pixel-coverage scoring)',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output split-token file (default: auto-named next to the PKL)',
    )
    parser.add_argument(
        '--num-workers', type=int, default=os.cpu_count() or 4,
        help='Worker processes for masks index parsing (default: all CPU cores)',
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not (0 < args.split_percent <= 100):
        parser.error('--split-percent must be in (0, 100]')

    t0 = time.time()

    print(f'Loading {args.pkl} ...')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    data_infos = sorted(data['infos'], key=lambda e: e['timestamp'])
    print(f'Total samples : {len(data_infos)}')

    print(f'\nLoading masks index ({args.num_workers} workers): {args.masks_index} ...')
    mask_counts = load_mask_pixel_counts(args.masks_index, num_workers=args.num_workers)
    if mask_counts:
        print(f'  {len(mask_counts)} camera-frame entries loaded')
    else:
        print('  Not found or empty — using GT object counts only')

    print('\nFiltering object-less samples ...')
    data_infos, dropped_tokens = filter_objectless_samples(data_infos, mask_counts)
    if dropped_tokens:
        print(f'  WARNING: {len(dropped_tokens)} sample(s) dropped — no detectable objects in any view:')
        for tok in dropped_tokens:
            print(f'    WARNING: {tok}')
    else:
        print(f'  All samples contain at least one object of interest')

    n_total = len(data_infos)
    n_target = max(1, round(args.split_percent / 100.0 * n_total))
    print(f'Eligible samples: {n_total}')
    print(f'Target samples  : {n_target} ({args.split_percent:.1f}%)')

    print('\nScoring samples ...')
    scores = compute_sample_scores(data_infos, mask_counts)

    print('Running diverse selection ...')
    selected = diverse_sample(data_infos, n_target, scores, seed=args.seed)
    tokens = [data_infos[i]['token'] for i in selected]
    print(f'Selected {len(tokens)} samples\n')
    print_stats(data_infos, selected)

    if args.output is None:
        stem = os.path.splitext(args.pkl)[0]
        pct_str = f'{args.split_percent:.0f}'.replace('.', 'p')
        args.output = f'{stem}_diverse_split_{pct_str}pct.txt'

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        for token in tokens:
            f.write(token + '\n')

    print(f'\nSplit file : {args.output}')
    print(f'Elapsed    : {time.time() - t0:.1f}s')
    print('\nTo use in your config, add to the dataset dict:')
    print(f"    split_file='{args.output}',")


if __name__ == '__main__':
    main()
