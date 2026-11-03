#!/usr/bin/env python3
"""Deterministically recompute every seed-aggregated paper number from the eval.json files,
and write them as CSV + Markdown so the results can be verified independently of the LaTeX.

Reuses the SAME manifests as tools/paper_tables.py (seed directories, arms, extractors), so this
script and the generated tables can never disagree. Deterministic: same eval.json in -> identical
CSV/MD out (sorted rows, fixed rounding, no timestamps, no glob order).

  python tools/aggregate_results.py --root . --out results_final
Writes <out>/results.csv and <out>/RESULTS.md.
"""
import argparse, csv, os, os.path as osp, sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import paper_tables as PT          # noqa: E402  (manifests + extractors, no side effects on import)


def stats(vals):
    """(mean, std, n) over non-None values; std is population std."""
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
    return m, sd, len(v)


def rows_ghosts(root):
    out = []
    base = PT.GHOST_ARMS[0][1]
    for tag in ("clean", "random"):
        for z in PT.GHOST_ZONES:
            vals = PT._ghost_zone_vals(root, base, tag, z)
            out.append(("ghosts", tag, z, vals))
    for label, arm in PT.GHOST_ARMS:
        for z in PT.GHOST_ZONES:
            vals = PT._ghost_zone_vals(root, arm, "learned", z)
            out.append(("ghosts", label, z, vals))
    return out


def rows_attribution(root):
    out = []
    for atk, key, eff_raw in (("suppress", "suppress", PT._raw_collat),
                              ("hull", "hull", PT._raw_ghost)):
        for st in PT.ABL_STAGES:
            seeds = [PT.load(root, f"{d}/{PT.ABL_ARM[key] % st}/eval.json") for d in PT.ABL_SEED_DIRS[st]]
            out.append(("attribution", f"{atk}:{st}", "clean", [PT._raw_clean(r) for r in seeds]))
            out.append(("attribution", f"{atk}:{st}", "learned", [PT._raw_learned(r) for r in seeds]))
            out.append(("attribution", f"{atk}:{st}", "effect", [eff_raw(r) for r in seeds]))
    return out


def rows_targets(root):
    out = []
    def metric(r, which):
        if not r:
            return None
        a = r.get("summary", {}).get("all", {})
        return {"clean": a.get("clean", {}).get("all", {}).get("conf"),
                "attack": a.get("learned", {}).get("all", {}).get("conf"),
                "recall": a.get("learned", {}).get("all", {}).get("recall_0p3"),
                "ghosts": r.get("extra", {}).get("false_positives", {}).get("learned", {}).get("hull", {}).get("per_scene"),
                }[which]
    for label, arm in PT.TARGET_ARMS:
        seeds = [PT.load(root, f"{d}/{arm}/eval.json") for d in PT.TARGET_SEED_DIRS]
        for which in ("clean", "attack", "recall", "ghosts"):
            out.append(("targeting", label, which, [metric(r, which) for r in seeds]))
    return out


def rows_hiding(root):
    out = []
    for label, arm, which, dirs in PT.HIDING_ROWS:
        for field in ("clean", "random", "learned"):
            out.append(("hiding", label, field, PT._hid_conf(root, dirs, arm, which, field)))
        out.append(("hiding", label, "recall", PT._hid_recall(root, dirs, arm, which)))
    return out


def rows_patch(root):
    out = []
    for label, arm, which, _area, dirs in PT.PATCH_ROWS:
        out.append(("patch", label, "clean", PT._hid_conf(root, dirs, arm, which, "clean")))
        out.append(("patch", label, "learned", PT._hid_conf(root, dirs, arm, which, "learned")))
        out.append(("patch", label, "recall", PT._hid_recall(root, dirs, arm, which)))
    return out


def rows_lidar_transfer(root):
    out = []
    for victim, rel in PT.DENSITY_SRC:
        d = PT.parse_density(root, rel)
        for key, _lab in PT.DENSITY_BINS:
            v = d[key][0] if d and key in d else None
            out.append(("lidar_transfer", victim, f"{key}_pts_hide%", [v]))
        out.append(("lidar_transfer", victim, "corr_hide_logpts", [d.get("corr") if d else None]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=osp.dirname(osp.dirname(osp.abspath(__file__))))
    ap.add_argument("--out", default="results_final")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    families = (rows_hiding(args.root) + rows_ghosts(args.root) + rows_attribution(args.root)
                + rows_targets(args.root) + rows_patch(args.root) + rows_lidar_transfer(args.root))
    # one record per (table,row,column): per-seed values + mean/std/n
    recs = []
    for table, row, col, vals in families:
        m, sd, n = stats(vals)
        recs.append(dict(table=table, row=row, column=col,
                         seeds=";".join("" if v is None else f"{v:.4f}" for v in vals),
                         n=n, mean="" if m is None else f"{m:.4f}", std="" if sd is None else f"{sd:.4f}"))
    recs.sort(key=lambda r: (r["table"], r["row"], r["column"]))

    csv_path = osp.join(args.out, "results.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["table", "row", "column", "seeds", "n", "mean", "std"])
        w.writeheader()
        w.writerows(recs)

    md = ["# Final seed-aggregated results (deterministic)", "",
          "Recomputed from eval.json by `tools/aggregate_results.py` (same manifests as "
          "`tools/paper_tables.py`). `seeds` lists the per-seed values; `mean`/`std` are over them.", ""]
    cur = None
    for r in recs:
        if r["table"] != cur:
            cur = r["table"]
            md += ["", f"## {cur}", "", "| row | column | seeds | mean | std | n |",
                   "|---|---|---|---|---|---|"]
        md.append(f"| {r['row']} | {r['column']} | {r['seeds']} | {r['mean']} | {r['std']} | {r['n']} |")
    open(osp.join(args.out, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print(f"wrote {csv_path} and {osp.join(args.out, 'RESULTS.md')} ({len(recs)} metrics)")


if __name__ == "__main__":
    main()
