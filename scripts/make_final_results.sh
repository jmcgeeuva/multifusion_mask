#!/bin/bash
# Assemble the final results folder for the AV camouflage-transfer paper, deterministically.
# Regenerates every LaTeX table from the eval.json files, recomputes the seed-aggregated numbers
# as a verifiable CSV + Markdown, and collects the final artifacts into results_final/.
# Re-running with unchanged eval.json files produces byte-identical output.
#   bash scripts/make_final_results.sh
set -eo pipefail
cd "$(dirname "$0")/.."
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
DEST=results_final

# 1. regenerate all generated .tex tables (deterministic)
bash scripts/sync_paper_tables.sh >/dev/null

# 2. recompute the seed-aggregated numbers as CSV + Markdown (verifiable, standalone)
mkdir -p "$DEST"
"$PY" tools/aggregate_results.py --root . --out "$DEST"

# 3. collect final artifacts (copies, so the paper's generated/ and the eval.json stay in place)
mkdir -p "$DEST/tables" "$DEST/density"
cp -f paper_overleaf/experiments/generated/*.tex "$DEST/tables/"
cp -f logs/transfer/density_*_multi.txt "$DEST/density/" 2>/dev/null || true
cp -f logs/transfer/RESULTS_*.md "$DEST/" 2>/dev/null || true

echo "final results assembled in $DEST/"
ls -R "$DEST"
