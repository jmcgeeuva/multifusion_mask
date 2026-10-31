#!/usr/bin/env python3
"""Aggregate the naturalness-reg (Task 1) and targeting (Task 2) perscene runs.

For each run dir with an eval.json it reports the targeted-vehicle confidence drop
(clean -> learned), recall at 0.3, and in-hull ghosts/scene (learned vs clean).

Usage:
  python tools/agg_reg.py logs/attack/natreg_<JOB>     # Task 1 (sorts by n, reg)
  python tools/agg_reg.py logs/attack/targets_<JOB>    # Task 2
"""
import argparse
import glob
import json
import os
import re


def read(path):
    d = json.load(open(path))
    s = d.get('summary', {}).get('all', {})
    def g(who, field):
        return s.get(who, {}).get('all', {}).get(field)
    fp = d.get('extra', {}).get('false_positives', {})
    def hull(who):
        return fp.get(who, {}).get('hull', {}).get('per_scene')
    return dict(clean_conf=g('clean', 'conf'), learned_conf=g('learned', 'conf'),
                clean_rec=g('clean', 'recall_0p3'), learned_rec=g('learned', 'recall_0p3'),
                n=g('learned', 'n'),
                ghost_hull_clean=hull('clean'), ghost_hull_learned=hull('learned'),
                args=d.get('args', {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    args = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(args.root, '*', 'eval.json'))):
        name = os.path.basename(os.path.dirname(p))
        try:
            rows.append((name, read(p)))
        except Exception as e:
            print(f'  [skip] {name}: {e}')

    if not rows:
        raise SystemExit('no eval.json yet under ' + args.root)

    # natreg dirs are n<N>_tv<..>_st<..>; sort by (N, w_style)
    def key(nm):
        m = re.match(r'n(\d+)_tv([\d.]+)_st([\d.]+)', nm)
        return (int(m.group(1)), float(m.group(3))) if m else (0, nm)
    rows.sort(key=lambda r: key(r[0]))

    print(f'{"run":28s} {"n":>3} {"clean":>6} {"learn":>6} {"drop":>6} {"rec_c":>6} {"rec_l":>6} '
          f'{"ghost_c":>7} {"ghost_l":>7}')
    print('-' * 86)
    for name, r in rows:
        cc, lc = r['clean_conf'] or 0, r['learned_conf'] or 0
        def f(x, w=6, p=3):
            return f'{x:{w}.{p}f}' if isinstance(x, (int, float)) else f'{"-":>{w}}'
        print(f'{name:28s} {str(r["n"]):>3} {f(cc)} {f(lc)} {f(cc - lc)} '
              f'{f(r["clean_rec"])} {f(r["learned_rec"])} '
              f'{f(r["ghost_hull_clean"],7)} {f(r["ghost_hull_learned"],7)}')

    # Task-1 hypothesis read-out: attack strength (conf drop) vs reg, for N=1 vs N=5
    byn = {}
    for name, r in rows:
        m = re.match(r'n(\d+)_tv[\d.]+_st([\d.]+)', name)
        if m and r['clean_conf'] is not None and r['learned_conf'] is not None:
            byn.setdefault(int(m.group(1)), []).append((float(m.group(2)), r['clean_conf'] - r['learned_conf']))
    if byn:
        print('\nattack strength (conf drop) vs style weight:')
        for n in sorted(byn):
            pts = sorted(byn[n])
            print(f'  N={n}: ' + '  '.join(f'st{int(w)}:{d:.3f}' for w, d in pts))
        if 1 in byn and 5 in byn:
            print('  hypothesis: reg should hurt N=1 more than N=5 (drop falls faster for N=1).')


if __name__ == '__main__':
    main()
