#!/usr/bin/env bash
# Rebuild all IS-Fusion CUDA extensions for the current Python and GPU architecture.
# Run from the multifusion_mask root inside the target conda environment.
set -e

export TORCH_CUDA_ARCH_LIST="7.0 8.0 8.6 8.9 9.0+PTX"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ISFUSION_DIR="$SCRIPT_DIR/../IS-Fusion"

echo "=== Removing old Python 3.7/3.8 .so files ==="
find "$ISFUSION_DIR" -name "*.cpython-37*.so" -delete
find "$ISFUSION_DIR" -name "*.cpython-38*.so" -delete
rm -rf "$ISFUSION_DIR/build" 2>/dev/null || true

echo "=== Rebuilding IS-Fusion main CUDA ops ==="
cd "$ISFUSION_DIR"
pip install -v . --no-build-isolation

echo "=== Rebuilding TorchEx ==="
cd "$ISFUSION_DIR/mmdet3d/ops/TorchEx"
pip install -v . --no-build-isolation
cd "$ISFUSION_DIR"

echo "=== Rebuilding bevfusion-ops spconv (sparse_conv_ext) ==="
cd "$ISFUSION_DIR/mmdet3d/ops/bevfusion-ops/spconv"
python setup.py build_ext --inplace
cd "$ISFUSION_DIR"

echo "=== Rebuilding MultiScaleDeformableAttention ops ==="
cd "$ISFUSION_DIR/ops"
python setup.py build_ext --inplace
cd "$ISFUSION_DIR"

echo ""
echo "All extensions rebuilt successfully."
echo "Verify with: python -c \"import mmdet3d.ops.voxel; print('OK')\""
