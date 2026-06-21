"""
Compare two camouflage PNG files to check how much they differ.

Usage:
    python compare_camou.py <img_a> <img_b> [--save <out.png>]

Example:
    python compare_camou.py workdir/run1/0camou.png workdir/run1/1camou.png
"""
import argparse
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load(path):
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: could not load '{path}'")
        sys.exit(1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_a', help='first camou PNG')
    parser.add_argument('img_b', help='second camou PNG')
    parser.add_argument('--save', default=None, help='save comparison figure to this path')
    args = parser.parse_args()

    a = load(args.img_a)
    b = load(args.img_b)

    if a.shape != b.shape:
        print(f"Shape mismatch: {a.shape} vs {b.shape}")
        sys.exit(1)

    diff = b - a                          # signed float32
    abs_diff = np.abs(diff)

    # ── stats ──────────────────────────────────────────────────────────────────
    max_diff   = abs_diff.max()
    mean_diff  = abs_diff.mean()
    l2         = np.linalg.norm(diff)
    pct_pixels_changed = (abs_diff.max(axis=-1) > 0).mean() * 100

    print(f"  A:                  {args.img_a}")
    print(f"  B:                  {args.img_b}")
    print(f"  Shape:              {a.shape}")
    print(f"  Max |diff|:         {max_diff:.3f}  (pixel units 0-255)")
    print(f"  Mean |diff|:        {mean_diff:.4f}")
    print(f"  L2 norm of diff:    {l2:.2f}")
    print(f"  Pixels changed:     {pct_pixels_changed:.2f}%")
    print(f"  Per-channel mean |diff|:  R={abs_diff[...,0].mean():.4f}  G={abs_diff[...,1].mean():.4f}  B={abs_diff[...,2].mean():.4f}")

    if max_diff == 0:
        print("\n  >> Images are IDENTICAL — no change.")
    else:
        print(f"\n  >> Images differ (max change {max_diff:.1f}/255 = {max_diff/255*100:.2f}%).")

    # ── figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    ax_a    = fig.add_subplot(gs[0])
    ax_b    = fig.add_subplot(gs[1])
    ax_diff = fig.add_subplot(gs[2])
    ax_hist = fig.add_subplot(gs[3])

    ax_a.imshow(a.astype(np.uint8))
    ax_a.set_title(f'A: {args.img_a.split("/")[-1]}', fontsize=9)
    ax_a.axis('off')

    ax_b.imshow(b.astype(np.uint8))
    ax_b.set_title(f'B: {args.img_b.split("/")[-1]}', fontsize=9)
    ax_b.axis('off')

    # Amplify diff for visibility (scale to 0-255 if non-zero, else show black)
    if max_diff > 0:
        diff_vis = np.clip((abs_diff / max_diff) * 255, 0, 255).astype(np.uint8)
    else:
        diff_vis = np.zeros_like(a, dtype=np.uint8)
    im = ax_diff.imshow(diff_vis)
    ax_diff.set_title(f'|A - B| (amplified, max={max_diff:.1f})', fontsize=9)
    ax_diff.axis('off')
    plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)

    # Histogram of absolute differences
    ax_hist.hist(abs_diff.ravel(), bins=64, color='steelblue', edgecolor='none')
    ax_hist.set_xlabel('|diff| (pixel units)')
    ax_hist.set_ylabel('count')
    ax_hist.set_title('Histogram of |A - B|', fontsize=9)
    ax_hist.set_yscale('log')

    fig.suptitle(
        f'mean|diff|={mean_diff:.4f}   L2={l2:.1f}   {pct_pixels_changed:.1f}% pixels changed',
        fontsize=10
    )

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"\n  Figure saved to: {args.save}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
