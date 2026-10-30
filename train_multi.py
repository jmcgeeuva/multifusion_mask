"""Train a camouflage texture against IS-Fusion with an all-vehicle (or single-vehicle) mask.

One GPU, the same world-size-1 DDP forward that grad_check.py verified. Differences from
train.py, all deliberate:
  - masks: --mask all textures every car/truck/bus mask in all six cameras;
    --mask single textures the loader's one chosen mask, only if it is a vehicle.
  - ModalMask3D (camera dropout) and ObjectSampleV2 (pasted objects) are removed from
    the training pipeline: the first zeroes the texture gradient on ~25% of steps, the
    second pastes other objects over the texture.
  - BatchNorm running statistics are frozen (momentum 0), so the detector cannot drift.
  - Adam lr 0.01 (train.py used 0.07-0.7, which jumps the texture to 0/1 in one step).
  - progress is measured on a fixed held-out set (val 5% split, scene-disjoint from
    training, deterministic augmentation), not on the training loss.

Held-out metric, per textured vehicle: the detector's best own-class probability among
proposals within 2 m (BEV) of that vehicle's ground-truth centre. A vehicle is "textured"
when its projected 3-D centre lands inside the texture mask. Reported as mean confidence
and recall@0.3, for clean / random texture / learned texture, under both mask types.

  python train_multi.py config/classification/classification4.py --out logs/multi_run/x --mask all
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
from mmcv import Config
from mmcv.parallel import DataContainer as DC
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from torch import optim

import grad_check as G
from augmentation import get_augmentation
from test import mask_imgs

MASK_ROOT = '/standard/eng_vivastorage/nuscenes/nuscenes/nuscenes_masks'
log = G.log


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


def vehicle_mask(meta, aug_mats, hw, index, vehicles, mode, pipeline_mask):
    """[6,H,W] bool texture mask for one sample."""
    if mode == 'single':
        if meta.get('nuscenes_class', '') in vehicles:
            return pipeline_mask[:, 0] >= 0.5
        return torch.zeros(pipeline_mask.shape[0], *hw, dtype=torch.bool)
    m = torch.zeros(len(meta['filename']), *hw, dtype=torch.bool)
    for cam, f in enumerate(meta['filename']):
        for path, _, c in G.masks_for_camera(index, MASK_ROOT, f):
            if c in vehicles:
                wm = G.warp_mask(path, aug_mats[cam], hw)
                if wm is not None:
                    m[cam] |= torch.from_numpy(wm)
    return m


def as5(m6):
    """[B,6,H,W] bool -> [B,6,3,H,W] float, the layout mask_imgs expects."""
    return m6[:, :, None].expand(-1, -1, 3, -1, -1).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mask', choices=['all', 'single'], default='all')
    ap.add_argument('--vehicles', default='car,truck,bus')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--heldout', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    vehicles = [v.strip() for v in args.vehicles.split(',') if v.strip()]
    vehicle_ids = {G.CLASS_NAMES.index(v) for v in vehicles if v in G.CLASS_NAMES}

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', str(29700 + int(os.environ.get('SLURM_JOB_ID', '0')) % 1000))
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
    log(f'GPU {torch.cuda.get_device_name(0)} | {args.config} | mask={args.mask} '
        f'vehicles={vehicles} lr={args.lr} bs={args.bs} steps={args.steps}')
    log('train pipeline: ' + ' > '.join(p['type'] for p in cfg.data.train.dataset.pipeline))

    G.seed_all(args.seed)
    train_set = build_dataset(cfg.data.train)
    train_loader = build_dataloader(train_set, args.bs, args.workers, 1, dist=True,
                                    seed=args.seed, shuffle=True)
    held_set = build_dataset(held_cfg)
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
    model.train()
    P = det.pts_bbox_head.num_proposals
    head_out = {}
    det.pts_bbox_head.register_forward_hook(lambda m_, i, o: head_out.__setitem__('o', o))

    def forward(data, imgs, rng=None):
        d = G.shallow(data)
        d['img'] = imgs
        if rng is not None:
            G.seed_all(rng)
        return model(return_loss=True, **d)

    # texture, built exactly as train_attack does
    res, H = 8, 1056
    expand = torch.nn.ConvTranspose2d(3, 3, res, stride=res, padding=0).to(device)
    expand.weight.data.fill_(0)
    expand.bias.data.fill_(0)
    for i in range(3):
        expand.weight[i, i].data.fill_(1)
    for p in expand.parameters():
        p.requires_grad = False
    tex_trans = get_augmentation(tuple(cfg.img_scale))
    G.seed_all(args.seed)
    camou = torch.rand([1, H // res, H // res, 3], device=device).requires_grad_(True)
    init = camou.detach().clone()
    opt = optim.Adam([camou], lr=args.lr)

    def tiled(param):
        return tex_trans(expand(param.permute(0, 3, 1, 2)))     # [1,h,w,3]

    # ------------------------------------------------ held-out set (cached on GPU)
    held_loader = build_dataloader(held_set, 1, args.workers, 1, dist=False, shuffle=False)
    held = []
    G.seed_all(1)
    for data in held_loader:
        meta = data['img_metas'].data[0][0]
        hw = tuple(data['masks'].data[0].shape[-2:])
        aug = data['img_aug_matrix'].data[0][0].numpy()
        m_all = vehicle_mask(meta, aug, hw, index, vehicles, 'all', None)
        if not m_all.any():
            continue
        m_one = vehicle_mask(meta, aug, hw, index, vehicles, 'single', data['masks'].data[0][0])
        G.to_device(data, device)
        m_all, m_one = m_all.to(device), m_one.to(device)
        held.append(dict(data=data, imgs=data['img'].data[0].to(device), all=m_all, single=m_one,
                         tg_all=G.targeted_gts(data, 0, m_all, hw, vehicle_ids),
                         tg_single=G.targeted_gts(data, 0, m_one, hw, vehicle_ids)))
        if len(held) >= args.heldout:
            break
    n_all = sum(len(h['tg_all']) for h in held)
    n_one = sum(len(h['tg_single']) for h in held)
    log(f'held-out: {len(held)} val samples, {n_all} textured vehicles (all-mask), '
        f'{n_one} (single-mask) ({time.time()-t0:.0f}s)')

    @torch.no_grad()
    def evaluate(param, conds):
        out = {}
        for cname, mkey in conds:
            confs, losses, cars = [], [], []
            for i, h in enumerate(held):
                tg = h['tg_all'] if mkey == 'all' else h['tg_single']
                if param is None or mkey is None:
                    imgs = [DC([h['imgs']], stack=False, cpu_only=False)]
                else:
                    G.seed_all(1000 + i)           # same texture placement every eval
                    imgs, _ = mask_imgs(h['imgs'], as5(h[mkey][None]), tiled(param), [],
                                        device=device, img_metas=h['data']['img_metas'].data[0])
                lo = forward(h['data'], imgs, rng=2000 + i)
                tot, _ = G.attack_loss(lo)
                hm, xy = G.head_scores(head_out['o'], cfg, P)
                losses.append(float(tot))
                cars.append(int((hm[0, 0] > 0.3).sum()))
                if mkey is None:   # clean: score both target sets
                    out.setdefault('clean_conf_all', []).extend(G.gt_conf(hm, xy, h['data'], 0, h['tg_all']))
                    out.setdefault('clean_conf_single', []).extend(G.gt_conf(hm, xy, h['data'], 0, h['tg_single']))
                else:
                    confs.extend(G.gt_conf(hm, xy, h['data'], 0, tg))
            r = dict(attack_loss=float(np.mean(losses)), car_props_gt_0p3=float(np.mean(cars)))
            if mkey is not None:
                c = np.array(confs)
                r.update(conf=float(c.mean()) if len(c) else None,
                         recall_0p3=float((c > 0.3).mean()) if len(c) else None, n=len(c))
            out[cname] = r
        for k in ('clean_conf_all', 'clean_conf_single'):
            if k in out:
                c = np.array(out[k])
                out[k] = dict(conf=float(c.mean()) if len(c) else None,
                              recall_0p3=float((c > 0.3).mean()) if len(c) else None, n=len(c))
        return out

    history = []
    base = evaluate(init, [('clean', None), ('random_all', 'all'), ('random_single', 'single')])
    history.append(dict(step=0, eval=base))
    log('[eval step 0] ' + json.dumps(base))

    def save(step, ev):
        img = (expand(camou.detach().permute(0, 3, 1, 2))[0].permute(1, 2, 0).cpu().numpy() * 255)
        cv2.imwrite(osp.join(args.out, f'texture_{step:05d}.png'),
                    cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        np.save(osp.join(args.out, 'texture_last.npy'), camou.detach().cpu().numpy())
        json.dump(dict(args=vars(args), history=history), open(osp.join(args.out, 'history.json'), 'w'),
                  indent=1)

    # ------------------------------------------------ training
    step, skipped, zero_grad = 0, 0, 0
    win = dict(loss=[], grad=[], cover=[])
    tl = open(osp.join(args.out, 'train_log.tsv'), 'w', buffering=1)
    tl.write('step\tloss\tgrad_norm\tmask_frac\tdelta_from_init\n')
    epoch = 0
    while step < args.steps:
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        for data in train_loader:
            metas = data['img_metas'].data[0]
            hw = tuple(data['masks'].data[0].shape[-2:])
            aug = data['img_aug_matrix'].data[0].numpy()
            m = torch.stack([vehicle_mask(metas[b], aug[b], hw, index, vehicles, args.mask,
                                          data['masks'].data[0][b]) for b in range(len(metas))])
            if not m.any():
                skipped += 1
                continue
            G.to_device(data, device)
            imgs = data['img'].data[0].to(device)
            opt.zero_grad(set_to_none=True)
            learned, _ = mask_imgs(imgs, as5(m.to(device)), tiled(camou), [], device=device,
                                   img_metas=metas)
            loss, _ = G.attack_loss(forward(data, learned))
            loss.backward()
            g = 0.0 if camou.grad is None else float(camou.grad.norm())
            zero_grad += int(g == 0.0)
            if camou.grad is not None:
                opt.step()
            with torch.no_grad():
                camou.clamp_(0, 1)
            step += 1
            delta = float((camou.detach() - init).abs().mean())
            mf = float(m.float().mean())
            tl.write(f'{step}\t{float(loss):.5f}\t{g:.4e}\t{mf:.4f}\t{delta:.4f}\n')
            for k, v in (('loss', float(loss)), ('grad', g), ('cover', mf)):
                win[k].append(v)
            if step % 50 == 0:
                log(f'step {step}: loss {np.mean(win["loss"]):.4f} | grad {np.mean(win["grad"]):.3e} | '
                    f'mask {np.mean(win["cover"]):.3f} of pixels | texture moved {delta:.3f} | '
                    f'skipped {skipped} zero-grad {zero_grad} | {time.time()-t0:.0f}s')
                win = dict(loss=[], grad=[], cover=[])
            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate(camou.detach(), [('learned_all', 'all'), ('learned_single', 'single')])
                history.append(dict(step=step, eval=ev, delta_from_init=delta))
                log(f'[eval step {step}] ' + json.dumps(ev))
                save(step, ev)
            if step >= args.steps:
                break
        epoch += 1
    save(step, None)
    log(f'done: {step} steps, {skipped} batches skipped (no vehicle mask), {zero_grad} zero-grad steps, '
        f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
