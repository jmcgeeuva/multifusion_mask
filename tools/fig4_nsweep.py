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
    ap.add_argument('--allref', type=float, default=None,
                    help='optional mask=all hull reference line (e.g. 10.3)')
    args = ap.parse_args()

    root = args.root
    if root is None:
        root = sorted(glob.glob(os.path.join(args.base, 'nsweep_*')), key=os.path.getmtime)[-1]
    by_n = A.load_sweep(root)

    ns = sorted(by_n)
    def series(zone):
        mu, sd = [], []
        for n in ns:
            vals = [by_n[n][s][zone] for s in by_n[n]]
            mu.append(np.mean(vals)); sd.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        return np.array(mu), np.array(sd)

    hull_mu, hull_sd = series('hull')
    hr_mu, hr_sd = series('hull+ring')

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.errorbar(ns, hr_mu, yerr=hr_sd, marker='o', ms=7, lw=2.2, capsize=4,
                color='#b00000', label='in-hull + ring')
    ax.errorbar(ns, hull_mu, yerr=hull_sd, marker='s', ms=6, lw=1.8, capsize=4,
                color='#e08000', ls='--', label='in-hull only')
    # tight y-scale 0 .. max+1 so the effect reads strongly
    ymax = float(np.max(hr_mu + hr_sd)) + 1.0
    ax.set_ylim(0, ymax)
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
