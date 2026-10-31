#!/bin/bash
# Regenerate ALL paper figures from saved data (CPU only, no GPU needed).
# Data sources (persistent, on tardis GPFS home -- NOT purged scratch):
#   figs_data/*.pkl                     -> fig1 (ghosts), fig2 (hiding), fig3 (baselines)
#   logs/attack/nsweep_<JOB>/           -> fig4 (N-vehicle ghost dose-response)
# Colour convention (all figures, both panels):
#   green = correct detection · magenta = ghost (no object) · red = missed detection
# Usage:  bash tools/make_figs.sh   [NSWEEP_DIR]
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/IS-Fusion:$PWD:$PYTHONPATH
PY=/home/tkg5kq/.conda/envs/ssiai_adv/bin/python
D=figs_data
OUT=paper/figs
NSWEEP=${1:-$(ls -dt logs/attack/nsweep_* 2>/dev/null | head -1)}
mkdir -p $OUT
cd tools

echo "fig1 ghosts   .."; $PY fig1_ghosts.py  --bundle ../$D/ghosts_strong.pkl --out ../$OUT/fig1_ghosts.png
echo "fig2 hiding   .."; $PY fig2_hiding.py   --bundle ../$D/hide.pkl          --out ../$OUT/fig2_hiding.png
echo "fig3 baselines.."; $PY fig3_baselines.py --scene 1 --out ../$OUT/fig3_baselines.png \
    --bundle "Random:../$D/random.pkl"      --bundle "Tile:../$D/tile.pkl" \
    --bundle "Screen:../$D/screen.pkl"      --bundle "Patch 20%:../$D/patch20.pkl" \
    --bundle "Patch 40%:../$D/patch40.pkl"  --bundle "Ours:../$D/full.pkl"
if [ -n "$NSWEEP" ]; then
  echo "fig4 dose-resp.."; $PY fig4_nsweep.py "../$NSWEEP" --out ../$OUT/fig4_nsweep.png --allref 10.3
fi
echo "done -> $OUT/"
