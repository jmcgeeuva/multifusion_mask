#!/bin/bash
# Generate the Experiments tables and move them into the paper's experiments/generated/ folder.
# Deterministic: re-running with unchanged eval.json files produces identical .tex.
#   bash scripts/sync_paper_tables.sh
set -eo pipefail
cd "$(dirname "$0")/.."
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
STAGE=tools/paper_tables_out
DEST=paper_overleaf/experiments/generated          # paper_overleaf is the symlink into this repo
"$PY" tools/paper_tables.py --results-root . --out "$STAGE"
mkdir -p "$DEST"
mv -f "$STAGE"/*.tex "$DEST"/
rmdir "$STAGE" 2>/dev/null || true
# official nuScenes AP/NDS table (separate: reads the full-val official_*.json dumps)
"$PY" tools/official_table.py --tex "$DEST/official.tex" >/dev/null || echo "WARN: official_table skipped (no official_*.json yet)"
# cross-model transfer table (reads logs/transfer/bevf_metric_*.out; BEVFusion now, TransFusion later)
"$PY" tools/transfer_table.py --tex "$DEST/transfer.tex" >/dev/null || echo "WARN: transfer_table skipped (no metric logs yet)"
echo "moved generated tables -> $DEST"
ls -1 "$DEST"
