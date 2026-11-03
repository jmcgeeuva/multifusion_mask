#!/usr/bin/env python3
"""Isolate: does the compiled bundled spconv run at all on a tiny synthetic tensor?"""
import torch
from mmdet3d.ops import spconv
from mmdet3d.ops.spconv import SparseConv3d, SubMConv3d, SparseConvTensor, SparseSequential

print("spconv module:", spconv.__file__)
dev = 'cuda:0'
# 10 random in-bounds voxels in a [41,1440,1440]-ish small grid for speed: use [41,100,100]
shape = [1440, 1440, 41]         # (x,y,z) — real BEVFusion 0p075 scale
N = 17000
torch.manual_seed(0)
xs = torch.randint(0, shape[0], (N,))
ys = torch.randint(0, shape[1], (N,))
zs = torch.randint(0, shape[2], (N,))
b  = torch.zeros(N, dtype=torch.int32)
coords = torch.stack([b, xs.int(), ys.int(), zs.int()], dim=1).to(dev)  # (batch,x,y,z)
feats = torch.randn(N, 4, device=dev)
print("synthetic coords", coords.shape, "feats", feats.shape)

sp = SparseConvTensor(feats, coords, shape, 1)
print("SparseConvTensor built; indices", sp.indices.shape)

try:
    net = SparseSequential(
        SubMConv3d(4, 8, 3, padding=1, bias=False, indice_key="subm1"),
        SparseConv3d(8, 16, 3, stride=2, padding=1, bias=False),   # downsample
    ).to(dev)
    out = net(sp)
    print("SPCONV FWD OK -> out features", out.features.shape, "spatial", out.spatial_shape)
    print("SYNTHETIC SPCONV OK")
except Exception as e:
    import traceback; traceback.print_exc()
    print("SYNTHETIC SPCONV FAILED")
