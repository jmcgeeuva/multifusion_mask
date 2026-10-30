"""Targeted camouflage attack on IS-Fusion: all-vehicle vs single-vehicle, shared texture vs a
bank of per-vehicle textures, universal vs per-scene. One GPU (world-size-1 DDP, the forward
grad_check.py verified).

Loss (computed here from the head's outputs, only on the TEXTURED vehicles):
  vanish  = max sigmoid of the vehicle's own class on the dense BEV heatmap within +-2 cells of
            its centre (IS-Fusion seeds its proposals from this heatmap's peaks)
          + max own-class probability among proposals within 2 m of its centre
  margin  = relu(z_own - z_best_other + 1) over those proposals (pushes towards a wrong class)
  loss    = w_vanish * vanish + w_margin * margin, averaged over textured vehicles.
A vehicle is textured when its projected 3-D centre lands inside the texture mask.

Textures: a bank of K textures with cells of --cell pixels (K=1 is one shared texture). Each
vehicle mask gets one bank entry: random per step in training, fixed by (scene, mask) at eval.

Modes:
  universal  one bank trained over the training split, evaluated on a fixed held-out set
             (val 5% split, scene-disjoint, deterministic augmentation) every --eval-every steps.
  perscene   upper bound: a fresh bank optimised on each held-out scene alone, then scored there.

Per textured vehicle the eval records: class, distance from ego, LiDAR points inside its box,
and the best own-class probability within 2 m (clean / random texture / learned texture), so
results can be split by distance and LiDAR density.

  python attack_multi.py config/classification/classification4.py --out logs/attack/x --mask all --bank 4 --cell 4
"""
import argparse
import copy
import json
import os
import os.path as osp
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import DataContainer as DC
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from torch import optim

import grad_check as G
from augmentation import get_augmentation

MASK_ROOT = '/standard/eng_vivastorage/nuscenes/nuscenes/nuscenes_masks'
from collections import defaultdict as _dd
FP_ACC = _dd(list)
log = G.log
DIST_BINS = [(0, 20), (20, 35), (35, 1e9)]
PTS_BINS = [(0, 20), (20, 100), (100, 1e9)]


# ------------------------------------------------------------------ data
def drop_types(pipeline, types):
    return [p for p in pipeline if p.get('type') not in types]


def make_deterministic(pipeline):
    out = []
    for p in drop_types(pipeline, {'ModalMask3D', 'ObjectSampleV2', 'RandomFlip3DV2'}):
        p = copy.deepcopy(p)
        if p.get('type') in ('ImageAug3D_Camou', 'GlobalRotScaleTransV2'):
            p['is_train'] = False
        out.append(p)
    return out


def instance_masks(meta, aug_mats, hw, index, vehicles, mode, pipeline_mask):
    """List of (camera, [H,W] bool) vehicle masks for one sample."""
    if mode == 'single':
        if meta.get('nuscenes_class', '') in vehicles and int(meta.get('camera_view', -1)) >= 0:
            cam = int(meta['camera_view'])
            m = pipeline_mask[cam, 0] >= 0.5
            return [(cam, m)] if m.any() else []
        return []
    out = []
    for cam, f in enumerate(meta['filename']):
        for path, _, c in G.masks_for_camera(index, MASK_ROOT, f):
            if c in vehicles:
                wm = G.warp_mask(path, aug_mats[cam], hw)
                if wm is not None and wm.any():
                    out.append((cam, torch.from_numpy(wm)))
    return out


def patch_rect(mask, frac, on_vehicle):
    """Shrink a [H,W] bool instance mask to a centered square patch of `frac` of its 2D-box area."""
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        return mask
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    s = max(frac, 1e-6) ** 0.5
    ph, pw = max(1, int(round((y1 - y0) * s))), max(1, int(round((x1 - x0) * s)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    r = torch.zeros_like(mask)
    r[max(cy - ph // 2, 0):cy - ph // 2 + ph, max(cx - pw // 2, 0):cx - pw // 2 + pw] = True
    if on_vehicle:
        r &= mask
    return r


def to_patch(insts, frac, on_vehicle):
    """FiNE-style object patch: replace each full vehicle mask with a small centered rectangle."""
    return [(cam, patch_rect(m, frac, on_vehicle)) for cam, m in insts]


def union(insts, ncam, hw, device):
    u = torch.zeros(ncam, *hw, dtype=torch.bool, device=device)
    for cam, m in insts:
        u[cam] |= m.to(device)
    return u


def lidar_points_in_boxes(data, b):
    boxes = data['gt_bboxes_3d'].data[0][b]
    pts = data['points'].data[0][b]
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long)
    from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
    # the op takes 7-dof boxes; nuScenes boxes carry 2 extra velocity values
    p = pts[:, :3].contiguous().float().cuda()
    bx = boxes.tensor[:, :7].contiguous().float().to(p.device)
    idx = points_in_boxes_gpu(p[None], bx[None])[0]
    idx = idx[idx >= 0].long()
    return torch.bincount(idx, minlength=len(boxes)).cpu()


# ------------------------------------------------------------------ texture
class Bank:
    def __init__(self, k, cell, size, device, tex_trans, seed):
        g = torch.Generator(device='cpu').manual_seed(seed)
        n = size // cell
        self.param = torch.rand([k, n, n, 3], generator=g).to(device).requires_grad_(True)
        self.init = self.param.detach().clone()
        self.k, self.cell, self.tex_trans = k, cell, tex_trans
        self.expand = torch.nn.ConvTranspose2d(3, 3, cell, stride=cell, padding=0).to(device)
        self.expand.weight.data.fill_(0)
        self.expand.bias.data.fill_(0)
        for i in range(3):
            self.expand.weight[i, i].data.fill_(1)
        for p in self.expand.parameters():
            p.requires_grad = False

    def tiles(self, param=None):
        """One randomly tiled/flipped/cropped image per bank entry: list of [3,H,W]."""
        param = self.param if param is None else param
        full = self.expand(param.permute(0, 3, 1, 2))            # [K,3,size,size]
        return [self.tex_trans(full[k:k + 1])[0].permute(2, 0, 1) for k in range(self.k)]

    def png(self, path):
        full = self.expand(self.param.detach().permute(0, 3, 1, 2)).permute(0, 2, 3, 1).cpu().numpy()
        img = np.concatenate(list(full), axis=1) * 255
        cv2.imwrite(path, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))


def paste(imgs, insts_per_sample, tiles, assign):
    """imgs [B,6,3,H,W] (normalised); insts_per_sample[b] = [(cam, mask)];
    assign[b][j] = bank index for mask j. Same value mapping as test.mask_imgs."""
    lo, rg = imgs.min(), imgs.max() - imgs.min()
    x = (imgs - lo) / rg
    for b, insts in enumerate(insts_per_sample):
        for j, (cam, m) in enumerate(insts):
            m = m.to(imgs.device)
            x[b, cam] = torch.where(m[None], tiles[assign[b][j]], x[b, cam])
    return [DC([x * rg + lo], stack=False, cpu_only=False)]


PLACE = dict(mode='tile', jitter=0.2, shift=128)


def render(bank, imgs, insts_per_sample, assign, param=None, train=False):
    """Put bank textures on the vehicle masks.
    tile   : the original screen-space texture, re-tiled/flipped/rotated/cropped at random
             every call (callers seed it for reproducible evals).
    screen : one fixed screen-space tiling; training adds only a random shift (<= shift px).
    bbox   : the texture is stretched over each mask's 2-D box (paint follows the vehicle);
             training adds a random sub-crop of the texture (up to `jitter` of its side)."""
    param = bank.param if param is None else param
    mode = PLACE['mode']
    if mode == 'tile':
        return paste(imgs, insts_per_sample, bank.tiles(param), assign)
    if mode == 'screen':
        full = bank.expand(param.permute(0, 3, 1, 2)).repeat(1, 1, 2, 2)     # [K,3,2S,2S]
        H, W = imgs.shape[-2:]
        dy, dx = (np.random.randint(0, PLACE['shift'] + 1, size=2) if train else (0, 0))
        tiles = [full[k, :, dy:dy + H, dx:dx + W] for k in range(bank.k)]
        return paste(imgs, insts_per_sample, tiles, assign)
    lo, rg = imgs.min(), imgs.max() - imgs.min()
    x = (imgs - lo) / rg
    src = param.permute(0, 3, 1, 2)                                          # [K,3,n,n]
    n = src.shape[-1]
    for b, insts in enumerate(insts_per_sample):
        for j, (cam, m) in enumerate(insts):
            m = m.to(imgs.device)
            ys, xs = torch.nonzero(m, as_tuple=True)
            if ys.numel() == 0:            # empty paste mask (e.g. a tiny-box patch that rounded to zero px): nothing to paint
                continue
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            t = src[assign[b][j]:assign[b][j] + 1]
            if train and PLACE['jitter'] > 0:
                c = max(2, int(round(n * (1 - np.random.uniform(0, PLACE['jitter'])))))
                oy, ox = np.random.randint(0, n - c + 1, size=2)
                t = t[..., oy:oy + c, ox:ox + c]
            t = F.interpolate(t, size=(y1 - y0, x1 - x0), mode='bilinear', align_corners=False)[0]
            x[b, cam, :, y0:y1, x0:x1] = torch.where(m[None, y0:y1, x0:x1], t, x[b, cam, :, y0:y1, x0:x1])
    return [DC([x * rg + lo], stack=False, cpu_only=False)]


# ------------------------------------------------------------------ loss / scoring
def head_outputs(head_out, P):
    res = head_out[0][0] if isinstance(head_out[0], (list, tuple)) else head_out[0]
    return res['dense_heatmap'], res['heatmap'][..., -P:], res['center'][..., -P:]


def _agg(z, form, k):
    """Vanishing penalty on a set of own-class logits z.
    form prob: sigmoid (saturates, slope p(1-p)); bce: softplus(z) = -log(1-p) (slope p).
    k = 1 is a hard max; k > 1 averages the top-k so the whole local peak is pushed."""
    z = z.flatten()
    top = z.topk(min(k, z.numel())).values
    return (top.sigmoid() if form == 'prob' else F.softplus(top)).mean()


def bev_cell_coords(n, cfg, swap, device):
    """Metre (x, y) of each dense-heatmap cell, in the array's own axis order.
    Per-vehicle code indexes dense[.., iy, ix] when swap else dense[.., ix, iy]."""
    cell = cfg.voxel_size[0] * cfg.out_size_factor
    idx = torch.arange(n, device=device)
    i0, i1 = torch.meshgrid(idx, idx)            # torch 1.10: 'ij' indexing
    iy, ix = (i0, i1) if swap else (i1, i0)
    x = cfg.point_cloud_range[0] + (ix.float() + 0.5) * cell
    y = cfg.point_cloud_range[1] + (iy.float() + 0.5) * cell
    return x, y


def in_hull(anchors, pts):
    """Bool mask: which pts (N,2) lie in the convex hull of anchors (K,2).
    K<3 -> anchor bounding box. Pure numpy in/out."""
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 2)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(anchors) < 3:
        lo, hi = anchors.min(0), anchors.max(0)
        return np.all((pts >= lo) & (pts <= hi), axis=-1)
    try:
        from scipy.spatial import Delaunay
        return Delaunay(anchors).find_simplex(pts) >= 0
    except Exception:
        lo, hi = anchors.min(0), anchors.max(0)
        return np.all((pts >= lo) & (pts <= hi), axis=-1)


def region_term(dense_b, anchors, gt_all, cfg, swap, mode, radius):
    """Signed penalty on sigmoid(dense) over a set of BEV cells.
    suppress: push the region DOWN (glare/fog).  saturate: pull the disc UP (flatten).
    hull_saturate: pull UP the hull between >=2 anchors, excluding cells near any real GT
    (so we score MANUFACTURED response, not re-lit real objects). Returns None if no cells."""
    n = dense_b.shape[-1]
    xg, yg = bev_cell_coords(n, cfg, swap, dense_b.device)
    if mode in ('suppress', 'saturate'):
        M = torch.zeros(n, n, dtype=torch.bool, device=dense_b.device)
        for a in anchors:
            M |= ((xg - a[0]) ** 2 + (yg - a[1]) ** 2) <= radius ** 2
    else:  # hull_saturate
        A = torch.stack(list(anchors)).detach().cpu().numpy()
        pts = torch.stack([xg.reshape(-1), yg.reshape(-1)], -1).cpu().numpy()
        M = torch.from_numpy(in_hull(A, pts).reshape(n, n)).to(dense_b.device)
        for g in gt_all:
            M &= ((xg - g[0]) ** 2 + (yg - g[1]) ** 2) > radius ** 2
    if not bool(M.any()):
        return None
    s = dense_b.sigmoid().amax(0)[M]
    return s.mean() if mode == 'suppress' else (1 - s).mean()


def false_positives(hm, xy, data, b, anchors, cfg, thresh, radius):
    """Ghost proposals (score>thresh, >=2 m from every GT) bucketed by BEV zone:
    inside the anchors' hull / within `radius` of an anchor / elsewhere."""
    scores, _ = hm[b].max(0)                       # [P]
    gt = data['gt_bboxes_3d'].data[0][b].gravity_center[:, :2].to(xy.device)
    A = torch.stack(list(anchors)).detach().cpu().numpy() if anchors else np.zeros((0, 2))
    zones = {'hull': [0, 0.0], 'ring': [0, 0.0], 'elsewhere': [0, 0.0]}
    for p in (scores > thresh).nonzero(as_tuple=False).flatten().tolist():
        c = xy[b, p]
        if len(gt) and float((gt - c).norm(dim=-1).min()) < 2.0:
            continue                               # matches a real object -> not a ghost
        cn = c.detach().cpu().numpy()
        if len(A) >= 3 and bool(in_hull(A, cn[None])[0]):
            z = 'hull'
        elif len(A) and float(np.sqrt(((A - cn) ** 2).sum(1)).min()) < radius:
            z = 'ring'
        else:
            z = 'elsewhere'
        zones[z][0] += 1
        zones[z][1] += float(scores[p])
    return {k: {'count': v[0], 'score_sum': v[1]} for k, v in zones.items()}


def targeted_loss(head, data, targets, cfg, P, swap, w_vanish, w_margin, margin=1.0, lcfg=None, rcfg=None):
    lc = dict(form='prob', k_dense=1, k_prop=1, margin_cap=0.0)
    lc.update(lcfg or {})
    rc = dict(mode='off', radius=6.0, w=0.0)
    rc.update(rcfg or {})
    dense, logits, ctr = head_outputs(head, P)
    vs, pc, osf = cfg.voxel_size, cfg.point_cloud_range, cfg.out_size_factor
    cell = vs[0] * osf
    x = ctr[:, 0] * cell + pc[0]
    y = ctr[:, 1] * cell + pc[1]
    terms_v, terms_m, terms_r = [], [], []
    for b, tg in enumerate(targets):
        if not tg:
            continue
        boxes = data['gt_bboxes_3d'].data[0][b]
        labels = data['gt_labels_3d'].data[0][b]
        c = boxes.gravity_center[:, :2].to(dense.device)
        for j in tg:
            cls = int(labels[j])
            ix = int((c[j, 0] - pc[0]) / cell)
            iy = int((c[j, 1] - pc[1]) / cell)
            r, n = 2, dense.shape[-1]
            a0, a1, b0, b1 = max(ix - r, 0), min(ix + r + 1, n), max(iy - r, 0), min(iy + r + 1, n)
            win = dense[b, cls, b0:b1, a0:a1] if swap else dense[b, cls, a0:a1, b0:b1]
            v = _agg(win, lc['form'], lc['k_dense']) if win.numel() else dense.new_zeros(())
            d = torch.sqrt((x[b] - c[j, 0]) ** 2 + (y[b] - c[j, 1]) ** 2)
            near = d < 2.0
            if near.any():
                z = logits[b][:, near]                              # [C, n]
                v = v + _agg(z[cls], lc['form'], lc['k_prop'])
                other = torch.cat([z[:cls], z[cls + 1:]], 0).max(0).values
                mt = F.relu(z[cls] - other + margin)
                if lc['margin_cap'] > 0:
                    mt = mt.clamp(max=lc['margin_cap'])
                terms_m.append(mt.mean())
            terms_v.append(v)
        if rc['mode'] != 'off':
            tr = region_term(dense[b], [c[j] for j in tg], c, cfg, swap, rc['mode'], rc['radius'])
            if tr is not None:
                terms_r.append(tr)
    if not terms_v:
        return None, {}
    lv = torch.stack(terms_v).mean()
    lm = torch.stack(terms_m).mean() if terms_m else lv.new_zeros(())
    lr = torch.stack(terms_r).mean() if terms_r else lv.new_zeros(())
    total = w_vanish * lv + w_margin * lm + rc['w'] * lr
    return total, dict(vanish=float(lv), margin=float(lm), region=float(lr), n=len(terms_v))


def summarize(records, key):
    def stats(rs):
        v = np.array([r[key] for r in rs if r.get(key) is not None])
        return None if not len(v) else dict(conf=round(float(v.mean()), 4),
                                             recall_0p3=round(float((v > 0.3).mean()), 4), n=int(len(v)))
    out = dict(all=stats(records))
    for lo, hi in DIST_BINS:
        out[f'dist_{lo}-{int(min(hi, 999))}m'] = stats([r for r in records if lo <= r['dist'] < hi])
    for lo, hi in PTS_BINS:
        out[f'lidar_pts_{lo}-{int(min(hi, 9999))}'] = stats([r for r in records if lo <= r['npts'] < hi])
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['universal', 'perscene', 'perseq'], default='universal')
    ap.add_argument('--mask', choices=['all', 'single'], default='all')
    ap.add_argument('--bank', type=int, default=1, help='number of textures (1 = one shared texture)')
    ap.add_argument('--cell', type=int, default=8, help='texture cell size in pixels')
    ap.add_argument('--vehicles', default='car,truck,bus')
    ap.add_argument('--w-vanish', type=float, default=1.0)
    ap.add_argument('--w-margin', type=float, default=1.0)
    # loss shape; defaults reproduce the original loss exactly
    ap.add_argument('--vanish-form', choices=['prob', 'bce'], default='prob')
    ap.add_argument('--k-dense', type=int, default=1, help='top-k heatmap cells averaged (1 = max)')
    ap.add_argument('--k-prop', type=int, default=1, help='top-k nearby proposals averaged (1 = max)')
    ap.add_argument('--margin-cap', type=float, default=0.0, help='clamp margin term per proposal (0 = off)')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--heldout', type=int, default=40)
    ap.add_argument('--perscene-steps', type=int, default=100)
    ap.add_argument('--perscene-lr', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--inference-norm', action='store_true',
                    help='BatchNorm/Dropout in eval mode (stored statistics), as at deployment')
    ap.add_argument('--load-bank', default=None, help='bank .npy to load; with --steps 0 just score it')
    ap.add_argument('--place', choices=['tile', 'screen', 'bbox'], default='tile')
    ap.add_argument('--jitter', type=float, default=0.2, help='bbox: max fraction of the texture cropped away')
    ap.add_argument('--shift', type=int, default=128, help='screen: max random shift in px')
    ap.add_argument('--grad-accum', type=int, default=1, help='forward passes per optimizer step')
    ap.add_argument('--lr-sched', choices=['const', 'cosine'], default='const')
    ap.add_argument('--seq-logs', type=int, default=6, help='perseq: number of drives')
    ap.add_argument('--seq-min-frames', type=int, default=16)
    ap.add_argument('--seq-steps', type=int, default=300, help='perseq: forward passes per drive')
    # region-effect losses + ghost metric (all neutral by default)
    ap.add_argument('--region-mode', choices=['off', 'suppress', 'saturate', 'hull_saturate'], default='off')
    ap.add_argument('--w-region', type=float, default=0.0)
    ap.add_argument('--region-radius', type=float, default=6.0, help='metres')
    ap.add_argument('--fp-eval', action='store_true', help='count ghost (false-positive) detections')
    ap.add_argument('--fp-thresh', type=float, default=0.3)
    # FiNE-style patch baseline + fusion-stage attribution ablation (default off)
    ap.add_argument('--attack', choices=['texture', 'patch'], default='texture')
    ap.add_argument('--patch-frac', type=float, default=0.3, help='patch area as fraction of each 2D box')
    ap.add_argument('--patch-on-vehicle', dest='patch_on_vehicle', action='store_true', default=True)
    ap.add_argument('--no-patch-on-vehicle', dest='patch_on_vehicle', action='store_false')
    ap.add_argument('--ablate', choices=['none', 'instance_fusion'], default='none',
                    help='instance_fusion: run IS-Fusion instance->scene attention but discard its effect\n                          on the scene features (keep its instance heatmap so losses stay valid)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    vehicles = [v.strip() for v in args.vehicles.split(',') if v.strip()]
    vehicle_ids = {G.CLASS_NAMES.index(v) for v in vehicles if v in G.CLASS_NAMES}
    LC = dict(form=args.vanish_form, k_dense=args.k_dense, k_prop=args.k_prop, margin_cap=args.margin_cap)
    RC = dict(mode=args.region_mode, radius=args.region_radius, w=args.w_region)
    PLACE.update(mode=args.place, jitter=args.jitter, shift=args.shift)

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', str(29800 + int(os.environ.get('SLURM_JOB_ID', '0')) % 1000))
    dist.init_process_group('nccl', rank=0, world_size=1)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')

    cfg = Config.fromfile(args.config)
    cfg.distributed = True
    cfg.gpu_ids = [0]
    G.patch_pipeline(cfg)
    held_cfg = copy.deepcopy(cfg.data.train.dataset)
    cfg.data.train.dataset.pipeline = drop_types(cfg.data.train.dataset.pipeline,
                                                 {'ModalMask3D', 'ObjectSampleV2'})
    held_cfg.ann_file = cfg.data.val.ann_file
    held_cfg.split_file = cfg.data.val.split_file
    held_cfg.pipeline = make_deterministic(held_cfg.pipeline)
    log(f'GPU {torch.cuda.get_device_name(0)} | ' + json.dumps(vars(args)))

    index = G.load_mask_index(osp.join(G.HERE, 'nuscenes_masks_index.txt'))
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, cfg.checkpoint, map_location='cpu')
    for m in model.modules():
        if hasattr(m, 'with_cp'):
            m.with_cp = False
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.0
    det = model
    model = MMDistributedDataParallel(model.cuda(), device_ids=[0], broadcast_buffers=False,
                                      find_unused_parameters=True)
    for p in model.parameters():
        p.requires_grad = False
    det.detach = False
    if args.ablate == 'instance_fusion':
        _orig_if = det.fusion_encoder.instance_fusion   # bound method
        def _if_ablate(bev, scene, bs, **k):
            _, ins_hm = _orig_if(bev, scene, bs, **k)    # keep the instance heatmap (losses need it)
            return scene, ins_hm                         # but discard the fusion's effect on scene feats
        det.fusion_encoder.instance_fusion = _if_ablate
        log('ablation: instance->scene fusion neutralized (scene features pass through)')
    model.train()
    if args.inference_norm:
        n_eval = 0
        for m in det.modules():
            if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout)):
                m.eval()
                n_eval += 1
        log(f'inference-norm: {n_eval} BatchNorm/Dropout layers use stored statistics')
    P = det.pts_bbox_head.num_proposals
    head = {}
    det.pts_bbox_head.register_forward_hook(lambda m_, i, o: head.__setitem__('o', o))

    def forward(data, imgs, rng=None):
        d = G.shallow(data)
        d['img'] = imgs
        if rng is not None:
            G.seed_all(rng)
        model(return_loss=True, **d)
        return head['o']

    tex_trans = get_augmentation(tuple(cfg.img_scale))

    # ------------------------------------------------ held-out set, cached on GPU
    held_set = build_dataset(held_cfg)
    held_loader = build_dataloader(held_set, 1, args.workers, 1, dist=False, shuffle=False)
    def prep(data):
        """Masks, targets and per-vehicle facts for one collated sample (moved to GPU)."""
        meta = data['img_metas'].data[0][0]
        hw = tuple(data['masks'].data[0].shape[-2:])
        aug = data['img_aug_matrix'].data[0][0].numpy()
        ins_all = instance_masks(meta, aug, hw, index, vehicles, 'all', None)
        if not ins_all:
            return None
        ins_one = instance_masks(meta, aug, hw, index, vehicles, 'single', data['masks'].data[0][0])
        G.to_device(data, device)
        ins_all = [(c, m.to(device)) for c, m in ins_all]
        ins_one = [(c, m.to(device)) for c, m in ins_one]
        ncam = len(meta['filename'])
        boxes = data['gt_bboxes_3d'].data[0][0]
        h = dict(data=data, imgs=data['img'].data[0].to(device), hw=hw,
                 ins=dict(all=ins_all, single=ins_one),
                 tg=dict(all=G.targeted_gts(data, 0, union(ins_all, ncam, hw, device), hw, vehicle_ids),
                         single=G.targeted_gts(data, 0, union(ins_one, ncam, hw, device), hw, vehicle_ids)),
                 dist=boxes.gravity_center[:, :2].norm(dim=1).tolist(),
                 npts=lidar_points_in_boxes(data, 0).tolist(),
                 labels=data['gt_labels_3d'].data[0][0].tolist(),
                 file=meta['filename'][0])
        h['ut'] = {mk: [j for j, l in enumerate(h['labels']) if l in vehicle_ids and j not in h['tg'][mk]]
                   for mk in ('all', 'single')}
        if args.attack == 'patch':   # paste on a patch; targets/ut stay full-mask
            h['ins'] = {mk: to_patch(h['ins'][mk], args.patch_frac, args.patch_on_vehicle)
                        for mk in ('all', 'single')}
        return h

    held = []
    G.seed_all(1)
    for data in held_loader:
        if args.mode == 'perseq':
            break
        h = prep(data)
        if h is None:
            continue
        held.append(h)
        if len(held) >= args.heldout:
            break
    if args.mode == 'perseq':          # the axis check below needs a few frames
        from mmcv.parallel import collate
        for idx in range(0, len(held_set), 30):
            h = prep(collate([held_set[idx]], samples_per_gpu=1))
            if h is not None:
                held.append(h)
            if len(held) >= 10:
                break
    log(f'held-out: {len(held)} val samples, {sum(len(h["tg"]["all"]) for h in held)} textured '
        f'vehicles (all), {sum(len(h["tg"]["single"]) for h in held)} (single) ({time.time()-t0:.0f}s)')

    # dense-heatmap axis convention, checked on clean data
    with torch.no_grad():
        a = b_ = 0.0
        for i, h in enumerate(held[:10]):
            out = forward(h['data'], [DC([h['imgs']], stack=False, cpu_only=False)], rng=2000 + i)
            dense = head_outputs(out, P)[0].sigmoid()
            boxes = h['data']['gt_bboxes_3d'].data[0][0]
            c = boxes.gravity_center[:, :2]
            cell = cfg.voxel_size[0] * cfg.out_size_factor
            for j in h['tg']['all']:
                ix = int((c[j, 0] - cfg.point_cloud_range[0]) / cell)
                iy = int((c[j, 1] - cfg.point_cloud_range[1]) / cell)
                cls = h['labels'][j]
                a += float(dense[0, cls, max(ix-1, 0):ix+2, max(iy-1, 0):iy+2].max())
                b_ += float(dense[0, cls, max(iy-1, 0):iy+2, max(ix-1, 0):ix+2].max())
        swap = b_ > a
        log(f'dense heatmap at GT centres: [x,y] order {a:.2f} vs [y,x] order {b_:.2f} -> swap={swap}')

    def assign_eval(i, n, k):
        return [(i * 31 + j) % k for j in range(n)]

    @torch.no_grad()
    def score(i, h, imgs, mkey, tag=None):
        out = forward(h['data'], imgs, rng=2000 + i)
        hm, xy = G.head_scores(out, cfg, P)
        if args.fp_eval and tag is not None:
            gc = h['data']['gt_bboxes_3d'].data[0][0].gravity_center[:, :2].to(xy.device)
            anchors = [gc[j] for j in h['tg'][mkey]]
            FP_ACC[tag].append(false_positives(hm, xy, h['data'], 0, anchors, cfg,
                                               args.fp_thresh, args.region_radius))
        return G.gt_conf(hm, xy, h['data'], 0, h['tg'][mkey] + h['ut'][mkey])

    def records_for(i, h, mkey):
        def rec(j, t):
            return dict(scene=i, gt=j, cls=G.CLASS_NAMES[h['labels'][j]], dist=round(h['dist'][j], 2),
                        npts=int(h['npts'][j]) if j < len(h['npts']) else 0, targeted=t)
        return [rec(j, True) for j in h['tg'][mkey]] + [rec(j, False) for j in h['ut'][mkey]]

    @torch.no_grad()
    def eval_universal(bank, tag, recs):
        FP_ACC[tag] = []
        for mkey in ('all', 'single'):
            for i, h in enumerate(held):
                if not h['tg'][mkey]:
                    continue
                G.seed_all(1000 + i)
                imgs = render(bank, h['imgs'], [h['ins'][mkey]], [assign_eval(i, len(h['ins'][mkey]), bank.k)],
                              param=bank.param if tag != 'random' else bank.init)
                for r, v in zip(recs[mkey][i], score(i, h, imgs, mkey, tag)):
                    r[f'{tag}'] = round(v, 4)

    recs = {mk: [records_for(i, h, mk) for i, h in enumerate(held)] for mk in ('all', 'single')}
    FP_ACC['clean'] = []
    with torch.no_grad():
        for mkey in ('all', 'single'):
            for i, h in enumerate(held):
                if h['tg'][mkey]:
                    for r, v in zip(recs[mkey][i],
                                    score(i, h, [DC([h['imgs']], stack=False, cpu_only=False)], mkey, 'clean')):
                        r['clean'] = round(v, 4)

    def flat(mkey, targeted=True):
        return [r for rs in recs[mkey] for r in rs if r.get('targeted', True) == targeted]

    def report(step, extra=None):
        summ = {mk: {k: summarize(flat(mk), k) for k in ('clean', 'random', 'learned')
                     if any(k in r for r in flat(mk))} for mk in ('all', 'single')}
        collat = {mk: {k: summarize(flat(mk, False), k)['all'] for k in ('clean', 'random', 'learned')
                       if any(k in r for r in flat(mk, False))} for mk in ('all', 'single')}
        extra = dict(extra or {}, collateral_untextured_vehicles=collat)
        if args.fp_eval:
            extra['false_positives'] = {tag: {z: dict(
                per_scene=round(sum(d[z]['count'] for d in lst) / max(len(lst), 1), 3),
                total=sum(d[z]['count'] for d in lst),
                mean_score=round(sum(d[z]['score_sum'] for d in lst) / max(sum(d[z]['count'] for d in lst), 1), 4))
                for z in ('hull', 'ring', 'elsewhere')} for tag, lst in FP_ACC.items() if lst}
        json.dump(dict(args=vars(args), step=step, summary=summ, records=recs, extra=extra or {}),
                  open(osp.join(args.out, 'eval.json'), 'w'), indent=1)
        line = {mk: {k: (v['all'] or {}) for k, v in summ[mk].items()} for mk in summ}
        log(f'[eval step {step}] ' + json.dumps(line) + ' | collateral ' + json.dumps(collat) + (' | ghosts ' + json.dumps(extra.get('false_positives', {}).get('learned', {})) if args.fp_eval else ''))

    # ------------------------------------------------ per-scene upper bound
    if args.mode == 'perscene':
        mkey = args.mask
        FP_ACC['random'] = []
        FP_ACC['learned'] = []
        for i, h in enumerate(held):
            if not h['tg'][mkey]:
                continue
            bank = Bank(args.bank, args.cell, 1056, device, tex_trans, seed=args.seed + i)
            opt = optim.Adam([bank.param], lr=args.perscene_lr)
            with torch.no_grad():
                G.seed_all(1000 + i)
                imgs = render(bank, h['imgs'], [h['ins'][mkey]], [assign_eval(i, len(h['ins'][mkey]), bank.k)],
                              param=bank.init)
                for r, v in zip(recs[mkey][i], score(i, h, imgs, mkey, 'random')):
                    r['random'] = round(v, 4)
            first = last = None
            for s in range(args.perscene_steps):
                opt.zero_grad(set_to_none=True)
                G.seed_all(1000 + i)                 # fixed placement: the upper bound
                imgs = render(bank, h['imgs'], [h['ins'][mkey]], [assign_eval(i, len(h['ins'][mkey]), bank.k)])
                out = forward(h['data'], imgs, rng=2000 + i)
                loss, parts = targeted_loss(out, h['data'], [h['tg'][mkey]], cfg, P, swap,
                                            args.w_vanish, args.w_margin, lcfg=LC, rcfg=RC)
                if loss is None:
                    break
                loss.backward()
                opt.step()
                with torch.no_grad():
                    bank.param.clamp_(0, 1)
                first = float(loss) if first is None else first
                last = float(loss)
            with torch.no_grad():
                G.seed_all(1000 + i)
                imgs = render(bank, h['imgs'], [h['ins'][mkey]], [assign_eval(i, len(h['ins'][mkey]), bank.k)])
                for r, v in zip(recs[mkey][i], score(i, h, imgs, mkey, 'learned')):
                    r['learned'] = round(v, 4)
            cl = np.mean([r['clean'] for r in recs[mkey][i] if r['targeted']])
            le = np.mean([r['learned'] for r in recs[mkey][i] if r['targeted']])
            log(f'[scene {i}] {len(recs[mkey][i])} vehicles | loss {first:.3f} -> {last:.3f} | '
                f'conf clean {cl:.3f} -> learned {le:.3f} | {time.time()-t0:.0f}s')
            if i % 5 == 4:
                report(i)
        report('final')
        return

    # ------------------------------------------------ per-sequence: first half of a drive -> second half
    if args.mode == 'perseq':
        from collections import defaultdict
        from mmcv.parallel import collate
        mkey = args.mask
        logs = defaultdict(list)
        for idx, inf in enumerate(held_set.data_infos):
            name = osp.basename(inf['lidar_path']).split('__')[0]
            logs[name].append((inf['timestamp'], idx))
        chosen = sorted([k for k in logs if len(logs[k]) >= args.seq_min_frames],
                        key=lambda k: -len(logs[k]))[:args.seq_logs]
        log(f'perseq: {len(chosen)} drives: ' + ', '.join(f'{k} ({len(logs[k])})' for k in chosen))
        seq_recs, summ = [], {}
        for li, name in enumerate(chosen):
            frames = []
            for _, idx in sorted(logs[name]):
                h = prep(collate([held_set[idx]], samples_per_gpu=1))
                if h is not None and name in h['file'] and h['tg'][mkey]:
                    frames.append(h)
            half = len(frames) // 2
            train_f, test_f = frames[:half], frames[half:]
            if not train_f or not test_f:
                continue
            bank = Bank(args.bank, args.cell, 1056, device, tex_trans, seed=args.seed + li)
            opt = optim.Adam([bank.param], lr=args.perscene_lr)
            rs = np.random.RandomState(li)
            n_up = max(1, args.seq_steps // args.grad_accum)
            losses = []
            for s_ in range(args.seq_steps):
                h = train_f[rs.randint(len(train_f))]
                assign = [rs.randint(0, bank.k, size=len(h['ins'][mkey])).tolist()]
                out = forward(h['data'], render(bank, h['imgs'], [h['ins'][mkey]], assign, train=True))
                loss, _ = targeted_loss(out, h['data'], [h['tg'][mkey]], cfg, P, swap,
                                        args.w_vanish, args.w_margin, lcfg=LC, rcfg=RC)
                if loss is None:
                    continue
                (loss / args.grad_accum).backward()
                losses.append(float(loss))
                if (s_ + 1) % args.grad_accum == 0:
                    if args.lr_sched == 'cosine':
                        for g_ in opt.param_groups:
                            g_['lr'] = args.perscene_lr * 0.5 * (1 + np.cos(np.pi * (s_ // args.grad_accum) / n_up))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        bank.param.clamp_(0, 1)
            with torch.no_grad():
                for split, fs in (('train', train_f), ('test', test_f)):
                    for fi, h in enumerate(fs):
                        a_ = [assign_eval(fi, len(h['ins'][mkey]), bank.k)]
                        r_ = records_for(fi, h, mkey)
                        conds = (('clean', [DC([h['imgs']], stack=False, cpu_only=False)]),
                                 ('random', render(bank, h['imgs'], [h['ins'][mkey]], a_, param=bank.init)),
                                 ('learned', render(bank, h['imgs'], [h['ins'][mkey]], a_)))
                        for tag, imgs in conds:
                            for r, v in zip(r_, score(fi, h, imgs, mkey)):
                                r[tag] = round(v, 4)
                        for r in r_:
                            r.update(drive=name, split=split, frame=fi)
                        seq_recs.extend(r_)
            def m_(split, tag, t=True):
                v = [r[tag] for r in seq_recs if r['drive'] == name and r['split'] == split and r['targeted'] == t]
                return float(np.mean(v)) if v else float('nan')
            log(f'[drive {li}] {name}: {len(train_f)} train / {len(test_f)} test frames | loss '
                f'{np.mean(losses[:20]):.3f} -> {np.mean(losses[-20:]):.3f} | test conf clean {m_("test", "clean"):.3f} '
                f'random {m_("test", "random"):.3f} learned {m_("test", "learned"):.3f} | train-frame learned '
                f'{m_("train", "learned"):.3f} | untextured test {m_("test", "clean", False):.3f} -> '
                f'{m_("test", "learned", False):.3f} | {time.time()-t0:.0f}s')
            del frames, train_f, test_f
            torch.cuda.empty_cache()
            summ = {f'{sp}_{"targeted" if t else "collateral"}': {k: summarize([r for r in seq_recs if r['split'] == sp and r['targeted'] == t], k)
                                                                  for k in ('clean', 'random', 'learned')}
                    for sp in ('train', 'test') for t in (True, False)}
            json.dump(dict(args=vars(args), summary=summ, records=seq_recs),
                      open(osp.join(args.out, 'eval.json'), 'w'), indent=1)
        log('[perseq final] ' + json.dumps({k: (v.get('learned') or {}).get('all') for k, v in summ.items()}))
        return

    # ------------------------------------------------ universal training
    G.seed_all(args.seed)
    train_set = build_dataset(cfg.data.train)
    train_loader = build_dataloader(train_set, args.bs, args.workers, 1, dist=True, seed=args.seed, shuffle=True)
    bank = Bank(args.bank, args.cell, 1056, device, tex_trans, seed=args.seed)
    if args.load_bank:
        loaded = torch.from_numpy(np.load(args.load_bank)).to(device)
        assert loaded.shape == bank.param.shape, (loaded.shape, bank.param.shape)
        bank.param.data.copy_(loaded)
        log(f'loaded bank {args.load_bank} {tuple(loaded.shape)}')
    opt = optim.Adam([bank.param], lr=args.lr)
    eval_universal(bank, 'random', recs)
    if args.steps == 0:
        eval_universal(bank, 'learned', recs)
        report('rescored')
        return
    report(0)
    rng = np.random.RandomState(args.seed)
    tl = open(osp.join(args.out, 'train_log.tsv'), 'w', buffering=1)
    tl.write('step\tloss\tvanish\tmargin\tn_targets\tgrad_norm\tdelta_from_init\n')
    step, skipped, epoch = 0, 0, 0
    win = []
    while step < args.steps:
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        for data in train_loader:
            metas = data['img_metas'].data[0]
            hw = tuple(data['masks'].data[0].shape[-2:])
            aug = data['img_aug_matrix'].data[0].numpy()
            insts = [instance_masks(metas[b], aug[b], hw, index, vehicles, args.mask, data['masks'].data[0][b])
                     for b in range(len(metas))]
            if not any(insts):
                skipped += 1
                continue
            G.to_device(data, device)
            ncam = len(metas[0]['filename'])
            targets = [G.targeted_gts(data, b, union(insts[b], ncam, hw, device), hw, vehicle_ids)
                       for b in range(len(metas))]
            if not any(targets):
                skipped += 1
                continue
            if args.attack == 'patch':   # paste on patches; targets already from full masks
                insts = [to_patch(x, args.patch_frac, args.patch_on_vehicle) for x in insts]
            imgs = data['img'].data[0].to(device)
            assign = [rng.randint(0, bank.k, size=len(x)).tolist() for x in insts]
            out = forward(data, render(bank, imgs, insts, assign, train=True))
            loss, parts = targeted_loss(out, data, targets, cfg, P, swap, args.w_vanish, args.w_margin, lcfg=LC, rcfg=RC)
            if loss is None:
                skipped += 1
                continue
            (loss / args.grad_accum).backward()
            g = 0.0 if bank.param.grad is None else float(bank.param.grad.norm())
            step += 1
            if step % args.grad_accum == 0:
                if args.lr_sched == 'cosine':
                    for g_ in opt.param_groups:
                        g_['lr'] = args.lr * 0.5 * (1 + np.cos(np.pi * step / args.steps))
                opt.step()
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    bank.param.clamp_(0, 1)
            delta = float((bank.param.detach() - bank.init).abs().mean())
            tl.write(f'{step}\t{float(loss):.5f}\t{parts["vanish"]:.5f}\t{parts["margin"]:.5f}\t'
                     f'{parts["n"]}\t{g:.4e}\t{delta:.4f}\n')
            win.append((float(loss), parts['vanish'], parts['margin'], g))
            if step % 50 == 0:
                w = np.array(win)
                log(f'step {step}: loss {w[:, 0].mean():.4f} (vanish {w[:, 1].mean():.4f}, margin '
                    f'{w[:, 2].mean():.4f}) | grad {w[:, 3].mean():.3e} | texture moved {delta:.3f} | '
                    f'skipped {skipped} | {time.time()-t0:.0f}s')
                win = []
            if step % args.eval_every == 0 or step == args.steps:
                eval_universal(bank, 'learned', recs)
                report(step, dict(delta_from_init=delta))
                bank.png(osp.join(args.out, f'texture_{step:05d}.png'))
                np.save(osp.join(args.out, 'bank_last.npy'), bank.param.detach().cpu().numpy())
            if step >= args.steps:
                break
        epoch += 1
    log(f'done: {step} steps, {skipped} batches skipped, {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
