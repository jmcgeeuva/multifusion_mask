#!/usr/bin/env python3
"""Figure 4 - ghost dose-response: fabricated detections vs number of painted vehicles.

The headline multi- vs single-vehicle result.  Painting N randomly-chosen vehicles and
sweeping N shows ghost fabrication is a genuine multi-object interaction: N=1 (the single-
vehicle control) produces ~0 ghosts, and the count rises (super-linearly) with N.  Net =
attack minus clean, averaged over 3 seeds; error bars are +/-1 std.

CPU only.  Reads logs/attack/nsweep_<JOB>/n<N>_s<S>/eval.json via agg_nsweep.load_sweep.
  python tools/fig4_nsweep.py logs/attack/nsweep_22196457 --out paper/figs/fig4_nsweep.png
"""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import agg_nsweep as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default=None)
    ap.add_argument('--base', default='/home/tkg5kq/workdir/av_project/multifusion_mask/logs/attack')
    ap.add_argument('--out', default='paper/figs/fig4_nsweep.png')
    ap.add_argument('--ext', default=None,
                    help='nsweepext_<JOB> dir with natreg/ and nonveh/ subdirs (extra condition lines)')
    ap.add_argument('--allref', type=float, default=None,
                    help='optional mask=all hull reference line (e.g. 10.3)')
    args = ap.parse_args()

    root = args.root
    if root is None:
        root = sorted(glob.glob(os.path.join(args.base, 'nsweep_*')), key=os.path.getmtime)[-1]

    def hr_series(by_n):
        """hull+ring mean/std over seeds, aligned to the sorted N present in this sweep."""
        ns = sorted(by_n)
        mu, sd = [], []
        for n in ns:
            vals = [by_n[n][s]['hull+ring'] for s in by_n[n]]
            mu.append(np.mean(vals)); sd.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        return ns, np.array(mu), np.array(sd)

    by_n = A.load_sweep(root)
    ns, hr_mu, hr_sd = hr_series(by_n)

    # optional extra condition lines; auto-pick newest nsweepext_* if --ext omitted
    ext = args.ext
    if ext is None:
        cands = sorted(glob.glob(os.path.join(args.base, 'nsweepext_*')), key=os.path.getmtime)
        ext = cands[-1] if cands else None
    ext_lines = []  # (label, color, linestyle, marker, ns, mu, sd)
    if ext:
        for sub, lab, col, ls, mk in (
                ('natreg', '+ naturalization', '#1f77b4', '--', '^'),
                ('nonveh', '+ non-vehicles',   '#2ca02c', ':',  'D')):
            d = os.path.join(ext, sub)
            if os.path.isdir(d):
                bn = A.load_sweep(d)
                if bn:
                    e_ns, e_mu, e_sd = hr_series(bn)
                    ext_lines.append((lab, col, ls, mk, e_ns, e_mu, e_sd))

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.errorbar(ns, hr_mu, yerr=hr_sd, marker='o', ms=7, lw=2.4, capsize=4,
                color='#b00000', label='vehicles', zorder=5)
    for lab, col, ls, mk, e_ns, e_mu, e_sd in ext_lines:
        ax.errorbar(e_ns, e_mu, yerr=e_sd, marker=mk, ms=6, lw=1.9, capsize=3,
                    color=col, ls=ls, label=lab)
    # tight y-scale 0 .. max+1 so the effect reads strongly
    allmax = [float(np.max(hr_mu + hr_sd))]
    for *_, e_mu, e_sd in ext_lines:
        allmax.append(float(np.max(e_mu + e_sd)))
    ymax = max(allmax) + 1.0
    if args.allref:                                    # context note only, no line
        ax.text(0.985, 0.955, f'paint-all = {args.allref:.1f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='0.45')
    # annotate the single-vehicle control
    ax.annotate('single-vehicle\ncontrol ~ 0', xy=(ns[0], hr_mu[0]),
                xytext=(ns[0] + 0.3, hr_mu[0] + ymax * 0.1), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='0.3'))
    ax.set_xticks(ns)
    ax.set_xlabel('number of painted vehicles  $N$', fontsize=11)
    ax.set_ylabel('fabricated ghost detections / scene\n(attack $-$ clean)', fontsize=11)
    ax.set_title('Ghost fabrication is a multi-vehicle effect', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', args.out)
    print('N:', ns)
    print('hull+ring mean:', hr_mu.round(2).tolist(), '+/-', hr_sd.round(2).tolist())


if __name__ == '__main__':
    main()
