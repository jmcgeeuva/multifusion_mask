"""Find where the camouflage gradient dies, and measure how much IS-Fusion uses the camera.

One GPU, one process. The model is wrapped in a world-size-1 MMDistributedDataParallel,
so the forward pass is the same one the 2-GPU suites ran.

  A. Data scan (no forward): how often the texture is visible at all, how often
     ModalMask3D blacks out the textured camera, how often the chosen mask is not a
     vehicle, and how many vehicles each sample could texture (single vs multi).
  B. Gradient trace: one attack step exactly as train.py does it, with a tap on every
     stage from camou_para to the loss. After backward, each tap reports whether it is
     in the graph and the norm of the gradient that reached it. The break is the first
     stage whose gradient is None or 0.
  C. Camera vs LiDAR: forward passes on the same batch with identical RNG state
       clean (twice; the gap is the noise floor), single-vehicle texture, multi-vehicle
       texture, camera off (all six images at the mean colour), LiDAR thinned to 10%,
     scored on the targeted vehicles' detection confidence. Plus the gradient of the
     attack loss with respect to the clean image, inside vs outside the vehicle masks,
     and with respect to the texture for single vs multi masks.

  python grad_check.py config/classification/classification4.py --out logs/grad_check/x
"""
import argparse
import inspect
import json
import os
import os.path as osp
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, 'IS-Fusion'))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDistributedDataParallel  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet3d.datasets import build_dataloader, build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402

from augmentation import get_augmentation  # noqa: E402
from test import mask_imgs, overlay_image  # noqa: E402

VEHICLES_DEFAULT = 'car,truck,bus'
CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
               'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']


def log(*a):
    print(*a, flush=True)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- pipeline patches
def patch_pipeline(cfg):
    """Record which mask file the loader picked, and carry it plus camera_view into
    img_metas. Nothing else in the pipeline changes."""
    import mmdet3d.datasets.pipelines.loading as loading
    from mmdet3d.datasets.pipelines.formating import Collect3DV2

    cls = loading.LoadMultiViewImageFromFilesV2_Camou
    if not getattr(cls, '_gc_patched', False):
        orig = cls.__call__

        def call(self, results):
            captured = {}
            real = loading.random.choices

            def spy(*a, **k):
                r = real(*a, **k)
                captured['c'] = r
                return r
            loading.random.choices = spy
            try:
                out = orig(self, results)
            finally:
                loading.random.choices = real
            out['mask_file'] = captured['c'][0][1] if 'c' in captured else ''
            return out
        cls.__call__ = call
        cls._gc_patched = True

    default = inspect.signature(Collect3DV2.__init__).parameters['meta_lis_keys'].default
    extra = tuple(default) + ('camera_view', 'mask_file')

    def fix(node):
        if isinstance(node, dict):
            if node.get('type') == 'Collect3DV2':
                node['meta_lis_keys'] = extra
            for v in node.values():
                fix(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                fix(v)
    fix(cfg.data.train)


def load_mask_index(path):
    idx = {}
    with open(path) as fh:
        for line in fh:
            p = line.rstrip('\n').split('\t')
            if len(p) < 3:
                continue
            cls = p[3] if len(p) >= 4 else ''
            area = int(p[4]) if len(p) >= 5 else 1
            idx.setdefault(p[0], []).append((osp.join(p[1], p[2]), area, cls))
    return idx


def masks_for_camera(index, mask_root, img_path):
    key = osp.normpath(osp.join(mask_root, '/'.join(img_path.split('/')[-4:]).replace('.jpg', '')))
    return index.get(key, [])


def warp_mask(path, aug, out_hw):
    """Apply a camera's img_aug_matrix (orig pixel -> augmented pixel) to a mask file."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    # Mask files are saved at 1056x384 (the full camera frame squashed); the pipeline
    # resizes them to the same absolute size as the 1600x900 image, so bring them to
    # camera resolution before applying the camera-pixel augmentation matrix.
    m = cv2.resize(m, (1600, 900), interpolation=cv2.INTER_NEAREST)
    A = np.asarray(aug, dtype=np.float64)
    M = np.array([[A[0, 0], A[0, 1], A[0, 3]], [A[1, 0], A[1, 1], A[1, 3]]])
    H, W = out_hw
    return cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0) > 127


# ---------------------------------------------------------------- taps
class Taps:
    """Records tensors flowing through chosen modules / methods, keeps their grads."""

    def __init__(self):
        self.on = False
        self.rec = []

    @staticmethod
    def tensors(o, out=None, depth=0):
        out = [] if out is None else out
        if depth > 4 or len(out) > 64:
            return out
        if torch.is_tensor(o):
            if o.is_floating_point():
                out.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                Taps.tensors(v, out, depth + 1)
        elif isinstance(o, (list, tuple)):
            for v in o:
                Taps.tensors(v, out, depth + 1)
        elif hasattr(o, 'data') and not torch.is_tensor(o):   # DataContainer
            Taps.tensors(o.data, out, depth + 1)
        return out

    def add(self, name, obj):
        if not self.on:
            return
        ts = self.tensors(obj)
        live = [t for t in ts if t.requires_grad]
        fn = next((type(t.grad_fn).__name__ for t in live if t.grad_fn is not None), '-')
        self.rec.append(dict(name=name, n=len(ts), n_live=len(live), grad_fn=fn, live=live,
                             numel=sum(t.numel() for t in ts)))

    CAMERA_PATH = ('camou_para', 'mask_imgs', 'img_backbone [input', 'extract_img_feat',
                   'img_point_sampling', 'img_fv_to_bev', 'conv_fusion', 'instance_fusion',
                   'pts_neck', 'pts_bbox_head.heatmap_head')

    def stage_grads(self, loss):
        """|d loss / d stage| for the camera-path stages, via autograd.grad (no hooks:
        retain_grad hooks broke this model's backward)."""
        self.grads = {}
        picks = [(i, r['live'][0]) for i, r in enumerate(self.rec)
                 if r['live'] and any(k in r['name'] for k in self.CAMERA_PATH)]
        for i, t in picks:
            try:
                g, = torch.autograd.grad(loss, [t], retain_graph=True, allow_unused=True)
                self.grads[i] = 'None' if g is None else float(g.double().norm())
            except RuntimeError as e:
                self.grads[i] = 'error: ' + str(e).split('\n')[0][:60]

    def report(self):
        rows = []
        for i, r in enumerate(self.rec):
            if not r['live']:
                g = 'not in graph'
            else:
                g = getattr(self, 'grads', {}).get(i, '-')
            rows.append(dict(name=r['name'], tensors=r['n'], requires_grad=r['n_live'] > 0,
                             grad_fn=r['grad_fn'], grad_norm=g))
        return rows


def install_taps(det, taps):
    hooks = []
    deep = ('fusion_encoder', 'pts_bbox_head', 'img_neck')
    for name, m in det.named_modules():
        if not name:
            continue
        d = name.count('.')
        if d == 0 or (d == 1 and name.split('.')[0] in deep):
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=name: taps.add(n, o)))
    if hasattr(det, 'img_backbone'):
        hooks.append(det.img_backbone.register_forward_pre_hook(
            lambda mod, i: taps.add('img_backbone [input image]', i)))

    def wrap(obj, meth, label, pick=None):
        orig = getattr(obj, meth)

        def w(*a, **k):
            out = orig(*a, **k)
            taps.add(label, pick(out) if pick else out)
            return out
        setattr(obj, meth, w)
    wrap(det, 'extract_img_feat', 'extract_img_feat [image features]')
    fe = det.fusion_encoder
    wrap(fe, 'img_point_sampling', 'fusion_encoder.img_point_sampling [sampled feats]', pick=lambda o: o[2])
    wrap(fe, 'img_fv_to_bev', 'fusion_encoder.img_fv_to_bev [camera BEV]')
    wrap(fe, 'instance_fusion', 'fusion_encoder.instance_fusion')
    return hooks


# ---------------------------------------------------------------- helpers
def attack_loss(losses):
    keys = [k for k in losses if k.startswith('loss_giad_') or k.startswith('layer_-1_loss_catk_')
            or k.startswith('layer_-1_loss_batk_')]
    total = sum(losses[k] for k in keys) if keys else None
    active = {k: float(losses[k].detach()) for k in keys if float(losses[k].detach().abs()) > 0}
    return total, active


def to_device(data, device):
    for key, val in data.items():
        if key == 'img':
            continue
        if hasattr(val, '_data') and isinstance(val._data, list) and val._data:
            inner = val._data[0]
            if torch.is_tensor(inner):
                val._data[0] = inner.to(device)
            elif isinstance(inner, list):
                val._data[0] = [x.to(device) if torch.is_tensor(x) else x for x in inner]


def shallow(data):
    """Copy of the batch dict whose DataContainers can be swapped without touching the
    original (mask_imgs-style img lists, thinned points)."""
    from mmcv.parallel import DataContainer as DC
    out = {}
    for k, v in data.items():
        if isinstance(v, DC):
            d = v.data
            d = [list(x) if isinstance(x, list) else x for x in d] if isinstance(d, list) else d
            out[k] = DC(d, stack=v.stack, padding_value=v.padding_value, cpu_only=v.cpu_only,
                        pad_dims=v.pad_dims)
        else:
            out[k] = v
    return out


def project_centres(centres, b, cam, data, img_hw):
    """Augmented-LiDAR points [N,3] -> augmented image pixels of camera `cam`, mirroring
    fusion_encoder.img_point_sampling. Returns (u, v, valid)."""
    la = data['lidar_aug_matrix'].data[0][b].float()
    l2i = data['lidar2img'].data[0][b][cam].float()
    ia = data['img_aug_matrix'].data[0][b][cam].float()
    p = centres.float() - la[:3, 3]
    p = torch.inverse(la[:3, :3]) @ p.T
    p = l2i[:3, :3] @ p + l2i[:3, 3:4]
    z = p[2].clone()
    p2 = p[:2] / p[2:3].clamp(min=1e-5)
    p3 = torch.cat([p2, torch.ones_like(p2[:1])], 0)
    q = ia[:3, :3] @ p3 + ia[:3, 3:4]
    u, v = q[0], q[1]
    H, W = img_hw
    valid = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, valid


def targeted_gts(data, b, mask6, img_hw, vehicle_ids):
    """GT vehicles whose projected 3-D centre lands inside the textured mask."""
    boxes = data['gt_bboxes_3d'].data[0][b]
    labels = data['gt_labels_3d'].data[0][b]
    if len(labels) == 0:
        return []
    c = boxes.gravity_center.to(labels.device)
    hits = []
    for cam in range(mask6.shape[0]):
        if not mask6[cam].any():
            continue
        u, v, ok = project_centres(c, b, cam, data, img_hw)
        for j in torch.nonzero(ok).flatten().tolist():
            if int(labels[j]) in vehicle_ids and mask6[cam, int(v[j]), int(u[j])]:
                hits.append(j)
    return sorted(set(hits))


def head_scores(head_out, cfg, P):
    """Last-decoder-layer proposals: BEV centres in metres and per-class probabilities."""
    res = head_out[0][0] if isinstance(head_out[0], (list, tuple)) else head_out[0]
    hm = res['heatmap'][..., -P:].sigmoid()            # [B, C, P]
    ctr = res['center'][..., -P:]                       # [B, 2, P] feature-grid units
    vs, pc, osf = cfg.voxel_size, cfg.point_cloud_range, cfg.out_size_factor
    x = ctr[:, 0] * osf * vs[0] + pc[0]
    y = ctr[:, 1] * osf * vs[1] + pc[1]
    return hm.detach(), torch.stack([x, y], -1).detach()   # [B,C,P], [B,P,2]


def gt_conf(hm, xy, data, b, gt_ids, radius=2.0):
    """For each GT: best probability of its own class among proposals within `radius` m."""
    if not gt_ids:
        return []
    boxes = data['gt_bboxes_3d'].data[0][b]
    labels = data['gt_labels_3d'].data[0][b]
    c = boxes.gravity_center[:, :2].to(xy.device)
    out = []
    for j in gt_ids:
        d = (xy[b] - c[j]).norm(dim=-1)
        near = d < radius
        out.append(float(hm[b, int(labels[j])][near].max()) if near.any() else 0.0)
    return out


def safe_backward(t, where, errors):
    """backward() that records a failure instead of killing the run."""
    if t is None or not t.requires_grad:
        errors.append(dict(where=where, error='loss not in graph (requires_grad=False)'))
        return False
    try:
        t.backward()
        return True
    except RuntimeError as e:
        import traceback
        errors.append(dict(where=where, error=str(e).split('\n')[0]))
        log(f'[backward failed: {where}] {e}')
        traceback.print_exc()
        return False


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--out', required=True)
    ap.add_argument('--scan', type=int, default=60, help='batches for the data scan')
    ap.add_argument('--batches', type=int, default=6, help='batches for parts B and C')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--vehicles', default=VEHICLES_DEFAULT)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--bs', type=int, default=1, help='samples per batch (train.py used 2)')
    ap.add_argument('--anomaly', action='store_true', help='torch anomaly mode: names the op whose backward fails')
    ap.add_argument('--no-taps', action='store_true', help='no stage taps (exact train.py graph)')
    ap.add_argument('--skip-c', action='store_true', help='stop after part B')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', str(29600 + int(os.environ.get('SLURM_JOB_ID', '0')) % 1000))
    dist.init_process_group('nccl', rank=0, world_size=1)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    log(f'GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__} | config {args.config}')

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    cfg.distributed = True
    cfg.gpu_ids = [0]
    cfg.data.samples_per_gpu = args.bs
    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)
    patch_pipeline(cfg)
    vehicles = [v.strip() for v in args.vehicles.split(',') if v.strip()]
    vehicle_ids = {CLASS_NAMES.index(v) for v in vehicles if v in CLASS_NAMES}

    seed_all(args.seed)
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(dataset, args.bs, args.workers, 1, dist=True, seed=args.seed, shuffle=True)
    mask_index = load_mask_index(osp.join(HERE, 'nuscenes_masks_index.txt'))
    mask_root = '/standard/eng_vivastorage/nuscenes/nuscenes/nuscenes_masks'

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, cfg.checkpoint, map_location='cpu')
    for m in model.modules():
        if hasattr(m, 'with_cp'):
            m.with_cp = False
    n_bn = 0
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.0      # train-mode forward, but running stats stay fixed
            n_bn += 1
    det = model
    model = MMDistributedDataParallel(model.cuda(), device_ids=[0], broadcast_buffers=False,
                                      find_unused_parameters=True)
    for p in model.parameters():
        p.requires_grad = False
    det.detach = False
    model.train()
    P = det.pts_bbox_head.num_proposals
    log(f'model loaded ({time.time()-t0:.0f}s); {n_bn} BatchNorm layers (momentum set to 0); '
        f'num_proposals {P}')

    # texture exactly as train_attack builds it
    res = 8
    Hc = Wc = 1056
    h, w = Hc // res, Wc // res
    expand = torch.nn.ConvTranspose2d(3, 3, res, stride=res, padding=0).to(device)
    expand.weight.data.fill_(0)
    expand.bias.data.fill_(0)
    for i in range(3):
        expand.weight[i, i].data.fill_(1)
    img_size = tuple(cfg.img_scale)
    tex_trans = get_augmentation(img_size)

    report = dict(config=args.config, gpu=torch.cuda.get_device_name(0), scan={}, trace=None,
                  per_batch=[], sensitivity=[], backward_errors=[])

    # ------------------------------------------------------------ A. data scan
    it = iter(loader)
    scan = dict(batches=0, with_mask=0, mask_px=[], cam_dropped=0, chosen_cls={},
                vehicles_in_sample=[], vehicles_in_cam=[], warp_iou=[])
    keep = []
    for _ in range(args.scan):
        data = next(it)
        meta = data['img_metas'].data[0][0]
        m = data['masks'].data[0][0]          # [6,3,H,W]
        scan['batches'] += 1
        cv_ = int(meta.get('camera_view', -1))
        cls = meta.get('nuscenes_class', '')
        scan['chosen_cls'][cls or '(none)'] = scan['chosen_cls'].get(cls or '(none)', 0) + 1
        nveh_all, nveh_cam = 0, 0
        for cam, f in enumerate(meta['filename']):
            ms = [x for x in masks_for_camera(mask_index, mask_root, f) if x[2] in vehicles]
            nveh_all += len(ms)
            if cam == cv_:
                nveh_cam = len(ms)
        scan['vehicles_in_sample'].append(nveh_all)
        if cv_ < 0:
            continue
        px = int((m[cv_, 0] >= 0.5).sum())
        scan['mask_px'].append(px)
        if px > 0:
            scan['with_mask'] += 1
        scan['vehicles_in_cam'].append(nveh_cam)
        dropped = cv_ in [int(x) for x in meta.get('img_mask_idx', [-1])]
        scan['cam_dropped'] += int(dropped)
        aug = data['img_aug_matrix'].data[0][0][cv_].numpy()
        wm = warp_mask(meta['mask_file'], aug, m.shape[-2:]) if meta.get('mask_file') else None
        if wm is not None and px > 0:
            pm = (m[cv_, 0] >= 0.5).numpy()
            scan['warp_iou'].append(float((wm & pm).sum() / max((wm | pm).sum(), 1)))
        if px > 0 and len(keep) < args.batches:
            keep.append(data)
    s = scan
    HW = None
    if keep:
        HW = tuple(keep[0]['masks'].data[0][0].shape[-2:])
    tot_px = (HW[0] * HW[1]) if HW else 1
    va = np.array(s['vehicles_in_sample'])
    report['scan'] = dict(
        batches=s['batches'],
        texture_visible=s['with_mask'],
        chosen_class=s['chosen_cls'],
        textured_camera_blacked_out=s['cam_dropped'],
        mask_px_median=float(np.median(s['mask_px'])) if s['mask_px'] else 0,
        mask_frac_of_one_camera_median=float(np.median(s['mask_px']) / tot_px) if s['mask_px'] else 0,
        vehicle_masks_per_sample=dict(mean=float(va.mean()), frac_ge2=float((va >= 2).mean()),
                                      frac_ge5=float((va >= 5).mean())),
        warp_check_iou_median=float(np.median(s['warp_iou'])) if s['warp_iou'] else None,
    )
    log('\n=== A. data scan ===')
    log(json.dumps(report['scan'], indent=1))
    if not keep:
        log('no batch with a visible texture in the scan; stopping')
        json.dump(report, open(osp.join(args.out, 'report.json'), 'w'), indent=1)
        return

    taps = Taps()
    hooks = [] if args.no_taps else install_taps(det, taps)
    head_out = {}
    hooks.append(det.pts_bbox_head.register_forward_hook(lambda m_, i, o: head_out.__setitem__('o', o)))

    def run(data, imgs, pts=None, clear_drop=False, rng=1234):
        d = shallow(data)
        d['img'] = imgs
        if pts is not None:
            d['points'].data[0] = pts
        if clear_drop:
            for mt in d['img_metas'].data[0]:
                mt['img_mask_idx'] = [-1, -1, -1]
        seed_all(rng)
        losses = model(return_loss=True, **d)
        return losses

    for bi, data in enumerate(keep):
        to_device(data, device)
        meta = data['img_metas'].data[0][0]
        imgs = data['img'].data[0].to(device)
        mask = data['masks'].data[0].to(device)
        cv_ = int(meta['camera_view'])
        dropped = cv_ in [int(x) for x in meta.get('img_mask_idx', [-1])]
        orig_drop = list(meta.get('img_mask_idx', [-1, -1, -1]))

        # ---------------------------------------------------- B. gradient trace
        seed_all(args.seed + bi)
        camou_para = torch.rand([1, h, w, 3], device=device).requires_grad_(True)
        camou_para1 = expand(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        camou_trans = tex_trans(camou_para1.permute(0, 3, 1, 2))
        for t in (camou_para1, camou_trans):
            t.retain_grad()
        learned, _ = mask_imgs(imgs, mask, camou_trans, [], device=device,
                               target_class=getattr(cfg, 'target_class', None),
                               img_metas=data['img_metas'].data[0])
        pasted = learned[0].data[0]
        pasted.retain_grad()
        taps.rec = []
        taps.on = bi == 0
        if taps.on:
            taps.add('camou_para [texture, leaf]', camou_para)
            taps.add('camou_para1 [expanded]', camou_para1)
            taps.add('tex_trans [tiled/cropped]', camou_trans)
            taps.add('mask_imgs [textured images]', pasted)
        losses = run(data, learned)
        total, active = attack_loss(losses)
        if taps.on:
            taps.add('attack loss (sum fed to backward)', total)
            taps.stage_grads(total)
        ok_b = safe_backward(total, f'batch{bi} attack step', report['backward_errors'])
        g_tex = camou_para.grad
        g_img = pasted.grad
        m6 = (mask[0, :, 0] >= 0.5)
        row = dict(batch=bi, token=meta.get('token'), chosen_class=meta.get('nuscenes_class'),
                   camera=cv_, mask_px=int(m6[cv_].sum()), camera_blacked_out=bool(dropped),
                   img_mask_idx=orig_drop, active_loss=active,
                   loss_requires_grad=bool(total.requires_grad),
                   grad_texture=None if g_tex is None else float(g_tex.norm()),
                   grad_textured_image_in_mask=None if g_img is None else
                   float(g_img[0][m6[:, None].expand(-1, 3, -1, -1)].norm()),
                   grad_textured_image_total=None if g_img is None else float(g_img.norm()))
        if bi == 0:
            report['trace'] = taps.report()
            log('\n=== B. gradient trace (batch 0, exactly as train.py) ===')
            log(f'{"stage":58s} {"in graph":>8s} {"grad_fn":>28s} {"|grad|":>12s}')
            for r in report['trace']:
                g = r['grad_norm']
                g = f'{g:.3e}' if isinstance(g, float) else g
                log(f'{r["name"][:58]:58s} {str(r["requires_grad"]):>8s} {r["grad_fn"][:28]:>28s} {g:>12s}')
            taps.on = False
            taps.rec = []

        # same step with the camera dropout removed (if it hit this sample)
        if dropped:
            camou_para.grad = None
            camou_para1 = expand(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            learned2, _ = mask_imgs(imgs, mask, tex_trans(camou_para1.permute(0, 3, 1, 2)), [],
                                    device=device, img_metas=data['img_metas'].data[0])
            l2, _ = attack_loss(run(data, learned2, clear_drop=True))
            safe_backward(l2, f'batch{bi} no-dropout step', report['backward_errors'])
            row['grad_texture_without_camera_dropout'] = (None if camou_para.grad is None
                                                          else float(camou_para.grad.norm()))
            for mt in data['img_metas'].data[0]:
                mt['img_mask_idx'] = orig_drop
        row['backward_ok'] = ok_b
        report['per_batch'].append(row)
        log(f'\n[batch {bi}] ' + json.dumps(row))
        json.dump(report, open(osp.join(args.out, 'report.json'), 'w'), indent=1)
        if args.skip_c:
            continue

        # ---------------------------------------------------- C. camera vs LiDAR
        # all-vehicle mask: every vehicle mask in every camera, warped like the image
        allm = torch.zeros_like(m6)
        nveh = 0
        for cam, f in enumerate(meta['filename']):
            for path, _, c in masks_for_camera(mask_index, mask_root, f):
                if c not in vehicles:
                    continue
                wm = warp_mask(path, data['img_aug_matrix'].data[0][0][cam].cpu().numpy(), HW)
                if wm is not None and wm.any():
                    allm[cam] |= torch.from_numpy(wm).to(device)
                    nveh += 1
        mask_all = allm[None, :, None].expand(-1, -1, 3, -1, -1).float()

        with torch.no_grad():
            seed_all(777)
            tex = tex_trans(expand(torch.rand([1, h, w, 3], device=device).permute(0, 3, 1, 2))
                            .permute(0, 2, 3, 1).permute(0, 3, 1, 2))
            lo, rg = imgs.min(), imgs.max() - imgs.min()

            def paste(msk):
                return [type(learned[0])([overlay_image((imgs - lo) / rg, msk, tex.permute(0, 3, 1, 2)) * rg + lo],
                                         stack=False, cpu_only=False)]
            pts = data['points'].data[0]
            g = torch.Generator(device='cpu').manual_seed(0)
            thin = [p[torch.randperm(p.shape[0], generator=g)[: max(1, p.shape[0] // 10)].to(p.device)]
                    for p in pts]
            conds = {
                'clean': dict(imgs=[type(learned[0])([imgs], stack=False, cpu_only=False)]),
                'clean_repeat': dict(imgs=[type(learned[0])([imgs], stack=False, cpu_only=False)]),
                'single_vehicle_texture': dict(imgs=paste(mask)),
                'multi_vehicle_texture': dict(imgs=paste(mask_all)),
                'camera_off': dict(imgs=[type(learned[0])([torch.zeros_like(imgs)], stack=False, cpu_only=False)]),
                'lidar_10pct': dict(imgs=[type(learned[0])([imgs], stack=False, cpu_only=False)], pts=thin),
            }
            tg_single = targeted_gts(data, 0, m6, HW, vehicle_ids)
            tg_multi = targeted_gts(data, 0, allm, HW, vehicle_ids)
            out = dict(batch=bi, vehicles_textured_multi=nveh,
                       targeted_gt_single=len(tg_single), targeted_gt_multi=len(tg_multi),
                       multi_mask_frac=float(allm.float().mean()), single_mask_frac=float(m6.float().mean()))
            h0 = None
            for name, c in conds.items():
                losses = run(data, c['imgs'], pts=c.get('pts'), clear_drop=True)
                tot, _ = attack_loss(losses)
                hm, xy = head_scores(head_out['o'], cfg, P)
                car = hm[0, 0]
                if h0 is None:
                    h0 = hm
                cs = gt_conf(hm, xy, data, 0, tg_single)
                cm = gt_conf(hm, xy, data, 0, tg_multi)
                out[name] = dict(
                    attack_loss=float(tot) if tot is not None else None,
                    car_top20=float(car.topk(min(20, car.numel())).values.mean()),
                    car_props_gt_0p3=int((car > 0.3).sum()),
                    heatmap_rel_change=float((hm - h0).norm() / h0.norm().clamp(min=1e-12)),
                    targeted_conf_single=float(np.mean(cs)) if cs else None,
                    targeted_conf_multi=float(np.mean(cm)) if cm else None,
                )

        # gradient of the attack loss w.r.t. the clean image and w.r.t. the texture
        img_req = imgs.clone().requires_grad_(True)
        tot, _ = attack_loss(run(data, [type(learned[0])([img_req], stack=False, cpu_only=False)],
                                 clear_drop=True))
        if not safe_backward(tot, f'batch{bi} image grad', report['backward_errors']) or img_req.grad is None:
            gi = torch.zeros_like(imgs[0, :, 0])
        else:
            gi = img_req.grad[0].abs().mean(1)          # [6,H,W]
        out['img_grad_mean_in_single_mask'] = float(gi[m6].mean()) if m6.any() else None
        out['img_grad_mean_in_multi_mask'] = float(gi[allm].mean()) if allm.any() else None
        out['img_grad_mean_outside_masks'] = float(gi[~allm & ~m6].mean())
        out['img_grad_textured_camera_mean'] = float(gi[cv_].mean())
        for tag, msk in (('single', mask), ('multi', mask_all)):
            cp = torch.rand([1, h, w, 3], device=device).requires_grad_(True)
            cp1 = expand(cp.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            ct = tex_trans(cp1.permute(0, 3, 1, 2))
            lc, _ = mask_imgs(imgs, msk, ct, [], device=device, img_metas=data['img_metas'].data[0])
            tl, _ = attack_loss(run(data, lc, clear_drop=True))
            safe_backward(tl, f'batch{bi} texture grad {tag}', report['backward_errors'])
            out[f'texture_grad_{tag}'] = None if cp.grad is None else float(cp.grad.norm())
        report['sensitivity'].append(out)
        log(f'[batch {bi}] sensitivity ' + json.dumps(out))
        torch.cuda.empty_cache()

    json.dump(report, open(osp.join(args.out, 'report.json'), 'w'), indent=1)
    log(f'\nwrote {osp.join(args.out, "report.json")} ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
