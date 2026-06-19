#!/usr/bin/env python3
"""
Build a mask index file for LoadMultiViewImageFromFilesV2_Camou.

Expected directory layout:
    <mask_path>/A/B/C/<sample>/
        <object_class>/
            mask1.jpg  mask2.jpg ...

One line is written per jpg:
    path_key <TAB> root <TAB> filename <TAB> object_class <TAB> pixel_area

  path_key     = mask_path + 4 path components (A/B/C/<sample>) — the key
                 used by LoadMultiViewImageFromFilesV2_Camou for dict lookup
  root         = full directory path that contains the jpg
  filename     = jpg basename
  object_class = basename of root (the immediate parent directory = class name)
  pixel_area   = number of foreground pixels (mask > 127) — used as sampling
                 weight so larger objects are selected proportionally more often

Single-process usage:
    python scripts/preprocess_masks.py \\
        --mask-path ./data/nuscenes/nuscenes_masks \\
        --output    ./nuscenes_masks_index.txt \\
        --threads   16

SLURM array usage (one job per camera, threads within each job):
    sbatch scripts/preprocess_masks.sh
    # then merge:
    cat nuscenes_masks_index_parts/index_*.txt > nuscenes_masks_index.txt

Use --scan-dir to restrict a single run to one camera subtree while keeping
--mask-path for correct path_key computation.
"""

import os
import os.path as osp
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm
import numpy as np
from PIL import Image


def _pixel_area(path):
    """Count foreground pixels in a binary mask jpg. Returns 1 on error."""
    try:
        arr = np.array(Image.open(path).convert('L'), dtype=np.uint8)
        return int((arr > 127).sum())
    except Exception:
        return 1


def scan_dir(directory):
    """Scan one directory level (no recursion).

    Opens each jpg to compute its pixel area so the index carries weights for
    area-proportional sampling in LoadMultiViewImageFromFilesV2_Camou.

    Returns (directory, subdirs, [(jpg_filename, pixel_area)]).
    """
    subdirs, jpgs = [], []
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    subdirs.append(entry.path)
                elif entry.is_file() and entry.name.lower().endswith('.jpg'):
                    area = _pixel_area(entry.path)
                    jpgs.append((entry.name, area))
    except (PermissionError, OSError):
        pass
    return directory, subdirs, jpgs


def parallel_walk(root, num_threads):
    """BFS directory walk using a thread pool.

    Each completed scan immediately submits its subdirectories as new tasks so
    threads stay busy across the full tree.  Returns a list of
    (dirpath, [(jpg_filename, pixel_area)]) for every directory with jpgs.
    """
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        pending = {executor.submit(scan_dir, root)}
        with tqdm(desc='Dirs scanned', unit='dir') as bar:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    dirpath, subdirs, jpgs = future.result()
                    if jpgs:
                        results.append((dirpath, jpgs))
                    for sd in subdirs:
                        pending.add(executor.submit(scan_dir, sd))
                    bar.update(1)
    return results


def build_index(mask_path, output_file, num_threads, scan_dir=None):
    mask_path = osp.normpath(mask_path)
    mask_depth = len(mask_path.split(os.sep))
    # path_key sits exactly 4 components below mask_path — matches the lookup
    # key computed by LoadMultiViewImageFromFilesV2_Camou.__call__:
    #   osp.join(mask_path, '/'.join(img_name.split('/')[-4:]).replace('.jpg',''))
    key_depth = mask_depth + 4

    root = osp.normpath(scan_dir) if scan_dir else mask_path
    print(f'Scanning {root} with {num_threads} threads (key_depth={key_depth}) ...')
    dir_entries = parallel_walk(root, num_threads)
    print(f'Found {len(dir_entries)} directories containing jpg files.')

    count = 0
    with open(output_file, 'w') as f:
        for dirpath, jpgs in tqdm(dir_entries, desc='Writing index', unit='dir'):
            dirpath_norm = osp.normpath(dirpath)
            parts = dirpath_norm.split(os.sep)
            if len(parts) < key_depth:
                # Directory is too shallow to have a valid path_key; skip.
                continue
            path_key = os.sep.join(parts[:key_depth])
            object_class = osp.basename(dirpath_norm)
            for filename, area in sorted(jpgs, key=lambda x: x[0]):
                f.write(f'{path_key}\t{dirpath_norm}\t{filename}\t{object_class}\t{area}\n')
                count += 1

    print(f'Done. Wrote {count} entries to {output_file}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--mask-path',
        default='./data/nuscenes/nuscenes_masks',
        help='Root of the precomputed mask tree (default: %(default)s)',
    )
    parser.add_argument(
        '--output',
        default='./nuscenes_masks_index.txt',
        help='Output index file (default: %(default)s)',
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=os.cpu_count(),
        help='Worker threads for directory scanning (default: cpu count = %(default)s)',
    )
    parser.add_argument(
        '--scan-dir',
        default=None,
        help='Subtree to actually walk (default: mask_path). path_key is '
             'still computed relative to mask_path, so partial outputs from '
             'different --scan-dir runs can be safely cat-merged.',
    )
    args = parser.parse_args()

    if not osp.isdir(args.mask_path):
        print(f'Error: mask_path not found: {args.mask_path}', file=sys.stderr)
        sys.exit(1)
    scan = args.scan_dir or args.mask_path
    if not osp.isdir(scan):
        print(f'Error: scan_dir not found: {scan}', file=sys.stderr)
        sys.exit(1)

    build_index(args.mask_path, args.output, args.threads, scan_dir=args.scan_dir)


if __name__ == '__main__':
    main()
