#!/usr/bin/env python3
"""Figure 6 - TEST-TIME hiding vs number of masked vehicles.

A fixed universal texture (trained once) is scored on held-out scenes while varying how many
vehicles are painted at eval time (--paint-n).  Plots own-class confidence on the painted
vehicles vs. that count, for the no-reg and naturalized textures.  Isolates test-time masking
from training.  Lower = stronger hiding.

CPU only.  Reads logs/attack/testmask_<JOB>/<bank>/pn<K>/eval.json.
  python tools/fig6_testmask.py logs/attack/testmask_22220242 --out paper/figs/fig6_testmask.png
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

LABELS = {'none': 'no regularization', 'strong': 'naturalization (TV 6, style 35)'}
COLORS = {'none': '#b00000', 'strong': '#1f77b4'}
STYLES = {'none': '-', 'strong': '--'}


def conf_of(path):
    d = json.load(open(path))
    s = d.get('summary', {}).get('all', {}).get('learned', {}).get('all', {})
    return s.get('conf'), s.get('recall_0p3'), s.get('n')


def load(root):
    out = {}
    for p in glob.glob(os.path.join(root, '*', 'pn*', 'eval.json')):
        m = re.search(r'/([^/]+)/pn(\d+)/eval\.json$', p)
        if not m:
            continue
        bank, k = m.group(1), int(m.group(2))
        try:
            c, r, n = conf_of(p)
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        if c is not None:
            out.setdefault(bank, {})[k] = (c, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out', default='paper/figs/fig6_testmask.png')
    ap.add_argument('--clean', type=float, default=0.763)
    args = ap.parse_args()

    data = load(args.root)
    if not data:
        raise SystemExit('no testmask eval.json under ' + args.root)

    # x positions: paint-n 1..5 then 'all' (stored as key 0) at the far right
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for bank in ('none', 'strong'):
        if bank not in data:
            continue
        ks = sorted(k for k in data[bank] if k != 0)
        xs = list(ks)
        ys = [data[bank][k][0] for k in ks]
        if 0 in data[bank]:                       # 'all' point past the last integer
            xs = xs + [(ks[-1] + 1) if ks else 6]
            ys = ys + [data[bank][0][0]]
        ax.plot(xs, ys, marker='o', ms=7, lw=2.2, color=COLORS[bank],
                ls=STYLES[bank], label=LABELS.get(bank, bank))

    ax.axhline(args.clean, color='0.35', lw=1.3)
    ax.text(ax.get_xlim()[1], args.clean, '  clean', va='center', ha='left',
            fontsize=8.5, color='0.35')
    # relabel the 'all' tick
    ks_any = sorted(k for b in data.values() for k in b if k != 0)
    if ks_any:
        last = ks_any[-1]
        ax.set_xticks(list(range(1, last + 1)) + [last + 1])
        ax.set_xticklabels([str(i) for i in range(1, last + 1)] + ['all'])
    ax.set_xlabel('number of vehicles painted at test time', fontsize=11)
    ax.set_ylabel('own-class confidence on painted vehicles\n(lower = stronger hiding)', fontsize=11)
    ax.set_title('Hiding vs. test-time mask count (fixed universal texture)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', args.out)
    for bank in data:
        print(f'  {bank}: ' + '  '.join(f'pn{k}:{data[bank][k][0]:.3f}' for k in sorted(data[bank])))


if __name__ == '__main__':
    main()
