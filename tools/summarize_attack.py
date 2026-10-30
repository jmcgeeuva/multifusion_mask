"""Summarise attack_multi.py runs: one row per arm, plus distance / LiDAR-density splits.

  python tools/summarize_attack.py logs/attack/<array job id> [--baseline logs/multi_run/<job>]
"""
import argparse
import glob
import json
import os.path as osp


def fmt(s):
    return '   -   ' if not s else f"{s['conf']:.3f} / {s['recall_0p3']:.2f} (n={s['n']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--baseline', default=None)
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(osp.join(args.root, '*', 'eval.json'))):
        r = json.load(open(f))
        a = r['args']
        mk = a['mask']
        s = r['summary'].get(mk, {})
        rows.append((osp.basename(osp.dirname(f)), r['step'], a, s, r['summary']))
    print('conf / recall@0.3 on the arm\'s own textured vehicles (held-out val, 40 scenes)\n')
    print(f"{'arm':24s} {'step':>6s} {'clean':>22s} {'random tex':>22s} {'learned tex':>22s}")
    for name, step, a, s, _ in rows:
        print(f"{name:24s} {str(step):>6s} {fmt((s.get('clean') or {}).get('all')):>22s} "
              f"{fmt((s.get('random') or {}).get('all')):>22s} {fmt((s.get('learned') or {}).get('all')):>22s}")
    for name, step, a, s, full in rows:
        if a['mode'] == 'universal':
            o = full.get('single' if a['mask'] == 'all' else 'all', {})
            if o.get('learned'):
                print(f"  {name}: learned texture scored on the {'single' if a['mask']=='all' else 'all'}-vehicle "
                      f"mask: {fmt(o['learned']['all'])} (clean {fmt(o['clean']['all'])})")
    print('\nsplits (learned texture; clean in brackets)')
    for name, step, a, s, _ in rows:
        if not s.get('learned'):
            continue
        L, C = s['learned'], s['clean']
        parts = [f"{k}: {fmt(L[k])} [{fmt(C[k])}]" for k in L if k != 'all' and L[k]]
        print(f"{name}\n    " + '\n    '.join(parts))
    if args.baseline:
        h = json.load(open(osp.join(args.baseline, 'history.json')))['history']
        print('\nbaseline (train_multi.py, scene-wide margin loss, all-vehicle shared c8):')
        print('   step 0 ' + json.dumps({k: v for k, v in h[0]['eval'].items()}))
        print(f"   step {h[-1]['step']} " + json.dumps(h[-1]['eval']))


if __name__ == '__main__':
    main()
