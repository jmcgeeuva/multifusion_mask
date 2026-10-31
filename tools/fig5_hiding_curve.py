#!/usr/bin/env python3
"""Figure 5 - hiding over optimization time, with naturalness regularization.

Parses the universal-attack slurm logs (natreguni_<JOB>_<task>.out), which log held-out
own-class confidence on the painted vehicles at each eval step for three regularization
levels (none / strong / very strong).  Plots confidence vs. optimization step: the attack
deepens hiding over training, and the regularized curves sit at or below the un-regularized
one -- naturalness does not cost hiding (it slightly helps).  Lower = stronger hiding.

CPU only.
  python tools/fig5_hiding_curve.py --job 22214923 --out paper/figs/fig5_hiding_curve.png
"""
import argparse
import glob
import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EVAL_RE = re.compile(r'\[eval step (\d+)\]\s*(\{.*?\})\s*\|\s*collateral')
ARGS_RE = re.compile(r'"w_tv":\s*([0-9.]+),\s*"w_style":\s*([0-9.]+)')


def parse_log(path):
    """Return (w_tv, w_style, steps[], conf[], recall[]) from one slurm .out."""
    txt = open(path, errors='ignore').read()
    m = ARGS_RE.search(txt)
    w_tv, w_style = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    steps, conf, rec = [], [], []
    for sm in EVAL_RE.finditer(txt):
        step = int(sm.group(1))
        try:
            d = json.loads(sm.group(2))
            learned = d['all'].get('learned')
        except (json.JSONDecodeError, KeyError):
            continue
        if learned is None:            # step 0 has no learned texture yet
            continue
        steps.append(step); conf.append(learned['conf']); rec.append(learned['recall_0p3'])
    # clean / random references (constant across steps) from the first eval line
    clean = rand = None
    m0 = EVAL_RE.search(txt)
    if m0:
        d0 = json.loads(m0.group(2))['all']
        clean = d0['clean']['conf']; rand = d0['random']['conf']
    order = np.argsort(steps)
    return (w_tv, w_style, np.array(steps)[order], np.array(conf)[order],
            np.array(rec)[order], clean, rand)


def label_for(w_tv, w_style):
    if not w_tv and not w_style:
        return 'no regularization'
    return f'naturalization (TV {w_tv:g}, style {w_style:g})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', default='22214923')
    ap.add_argument('--base', default='/home/tkg5kq/workdir/av_project/multifusion_mask/logs/attack')
    ap.add_argument('--metric', choices=['conf', 'recall'], default='conf')
    ap.add_argument('--out', default='paper/figs/fig5_hiding_curve.png')
    args = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(args.base, f'natreguni_{args.job}_*.out')))
    runs = [parse_log(p) for p in logs]
    runs = [r for r in runs if len(r[2])]                   # keep ones with eval points
    runs.sort(key=lambda r: (r[0] or 0) + (r[1] or 0))      # none -> strong -> vstrong

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    colors = ['#b00000', '#1f77b4', '#2ca02c']
    styles = ['-', '--', ':']
    clean = rand = None
    for i, (w_tv, w_style, steps, conf, rec, c, rr) in enumerate(runs):
        y = conf if args.metric == 'conf' else rec
        ax.plot(steps, y, marker='o', ms=6, lw=2.2, color=colors[i % 3],
                ls=styles[i % 3], label=label_for(w_tv, w_style))
        clean = c if clean is None else clean
        rand = rr if rand is None else rand

    if args.metric == 'conf':
        ylo, yhi = 0.25, 0.45
        if clean is not None and clean <= yhi:       # only draw refs that fall in view
            ax.axhline(clean, color='0.35', lw=1.3, ls='-')
            ax.text(ax.get_xlim()[1], clean, '  clean', va='center', ha='left',
                    fontsize=8.5, color='0.35')
        if rand is not None and rand <= yhi:
            ax.axhline(rand, color='0.6', lw=1.1, ls=':')
            ax.text(ax.get_xlim()[1], rand, '  random texture', va='center', ha='left',
                    fontsize=8.5, color='0.6')
        ax.set_ylabel('held-out own-class confidence\non painted vehicles  (lower = stronger)',
                      fontsize=11)
        ax.set_ylim(ylo, yhi)
    else:
        ax.set_ylabel('held-out recall @ 0.3  (lower = stronger)', fontsize=11)

    ax.set_xlabel('optimization step', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', args.out)
    for w_tv, w_style, steps, conf, rec, c, rr in runs:
        print(f'  {label_for(w_tv, w_style):34s} conf {np.round(conf,3).tolist()}  (clean {c})')


if __name__ == '__main__':
    main()
