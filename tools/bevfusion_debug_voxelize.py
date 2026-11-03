#!/usr/bin/env python3
"""Debug: build model, pull one sample, inspect points -> voxelize output."""
import sys
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, scatter
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

CFG = sys.argv[1]; CKPT = sys.argv[2]
configs.load(CFG, recursive=True)
cfg = Config(recursive_eval(configs), filename=CFG)
cfg.model.pretrained = None; cfg.model.train_cfg = None
cfg.data.test.test_mode = True
pipe = [t for t in cfg.data.test.get('pipeline', []) if 'Radar' not in t.get('type','')]
for t in pipe:
    if 'keys' in t and isinstance(t['keys'], (list, tuple)):
        t['keys'] = [k for k in t['keys'] if k != 'radar']
cfg.data.test.pipeline = pipe
ds = build_dataset(cfg.data.test)
ds.data_infos = ds.data_infos[:1]
dl = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, CKPT, map_location='cpu')
model = MMDataParallel(model.cuda(), device_ids=[0]).eval()

data = next(iter(dl))
# scatter to cuda like MMDataParallel does
inp = scatter(data, [0])[0]
pts = inp['points']
print("points type:", type(pts))
if isinstance(pts, list):
    print("  list len:", len(pts), "elem type:", type(pts[0]))
    p0 = pts[0]
    if isinstance(p0, list): p0 = p0[0]
    print("  point tensor:", p0.shape, p0.dtype, "device", p0.device)
    print("  xyz min:", p0[:, :3].min(0).values.tolist())
    print("  xyz max:", p0[:, :3].max(0).values.tolist())
    # run voxelize directly
    m = model.module
    feats, coords, sizes = m.voxelize([p0] if not isinstance(pts[0], list) else pts[0], 'lidar')
    print("VOXELIZE -> feats", feats.shape, "coords", coords.shape, "sizes", sizes.shape)
    print("  coords min/max:", coords.min(0).values.tolist(), coords.max(0).values.tolist())
    print("  coords dtype", coords.dtype, "contig", coords.is_contiguous(),
          "unique rows", torch.unique(coords, dim=0).shape[0])

    # instrument the real sparse encoder: print active sites entering each spconv module
    backbone = m.encoders['lidar']['backbone']
    from mmdet3d.ops.spconv.conv import SparseConvolution
    def mkhook(name):
        def hook(mod, inp):
            t = inp[0]
            nact = t.features.shape[0] if hasattr(t, 'features') else '?'
            print(f"   [pre] {name:28s} active={nact} spatial={getattr(t,'spatial_shape',None)}")
        return hook
    for nm, sub in backbone.named_modules():
        if isinstance(sub, SparseConvolution):
            sub.register_forward_pre_hook(mkhook(nm))
    bs = int(coords[-1, 0].item()) + 1
    try:
        out = backbone(feats, coords, bs, sizes=sizes)
        print("BACKBONE OK ->", type(out))
    except Exception as e:
        import traceback; traceback.print_exc()
        print("BACKBONE FAILED")
print("DEBUG DONE")
