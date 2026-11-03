# Cross-model transfer: IS-Fusion → MIT-BEVFusion (nuScenes val, 6019 samples)

Textures optimized ONLY on IS-Fusion (universal, all car/truck/bus, `natreguni_22214923`),
composited onto the RAW val camera images (full-res, on the vehicle segmentation silhouette),
evaluated on a never-optimized MIT-BEVFusion (convfuser swint_v0p075, bevfusion-det.pth).
Only vehicle-bearing camera images are rewritten; all other pixels are byte-identical to clean.

## Overall + per-class AP (mean over the 4 nuScenes dist thresholds)

| condition            | mAP    | NDS    | car AP | truck AP | bus AP | car+truck+bus mean |
|----------------------|--------|--------|--------|----------|--------|--------------------|
| clean                | 0.6866 | 0.7149 | 0.893  | 0.647    | 0.753  | 0.7643             |
| random-paint control | 0.6683 | 0.7037 | 0.854  | 0.587    | 0.736  | 0.7256 (−3.9)      |
| transfer (none)      | 0.6594 | 0.6988 | 0.824  | 0.573    | 0.719  | 0.7052 (−5.9)      |
| transfer (strong)    | 0.6573 | 0.6961 | 0.817  | 0.568    | 0.714  | 0.6997 (−6.5)      |

"none" = strongest adversarial (no naturalness reg); "strong" = naturalness-regularized
(TV + VGG style). Δ in parentheses = painted-class mean AP points vs clean (×100).

## Reading
- IS-Fusion-optimized camouflage transfers to BEVFusion: painted-class mean AP −5.9 to −6.5,
  car AP −6.9 to −7.6. Supports a shared camera-side leak across the BEV-fusion family,
  exploitable black-box via transfer.
- Adversarial beats the random-paint control by ~2 AP (−5.9/−6.5 vs −3.9): ~1/3 of the drop is
  adversarial-specific transfer, the rest is generic paint/occlusion. The control is load-bearing.
- The naturalness-regularized texture transfers slightly BETTER than the raw one (−6.5 vs −6.0):
  making the camouflage look natural did not cost transferability.

## Single-vehicle vs multi-vehicle optimization (cross-model thesis test)

Matched pair, IDENTICAL universal bbox recipe (inference-norm, bce vanish, w_margin=0, 1500 steps,
w_tv=w_style=0) differing ONLY in `--mask single` (vanish one target vehicle/scene) vs `--mask all`
(vanish all jointly). Both composited on ALL vehicles, same BEVFusion eval. The multi texture is the
same `natreguni/none` as above; the single texture is `natsingle_22240147` trained for this test.

| condition         | mAP    | NDS    | car   | truck | bus   | car+truck+bus mean | Δ vs clean |
|-------------------|--------|--------|-------|-------|-------|--------------------|------------|
| clean             | 0.6866 | 0.7149 | 0.893 | 0.647 | 0.753 | 0.7643             | —          |
| random paint      | 0.6683 | 0.7037 | 0.854 | 0.587 | 0.736 | 0.7256             | −3.88      |
| single-optimized  | 0.6641 | 0.7026 | 0.842 | 0.575 | 0.734 | 0.7169             | −4.75      |
| multi-optimized   | 0.6594 | 0.6988 | 0.824 | 0.573 | 0.719 | 0.7052             | −5.91      |

**Multi-vehicle optimization yields the more transferable universal camouflage**: −5.91 vs −4.75
veh-mean AP (1.16-pt gap), ordering consistent across mAP, NDS, and all three classes. Both beat
random paint (−3.88), so both carry real adversarial signal. Supports the paper's single-vs-multi
thesis cross-model and black-box.

CAVEAT (first attempt, superseded): the pre-existing older-recipe pair (job 22118064, tile/screen
placement, w_margin=1) composited in bbox mode degraded to ~random-paint level (single −3.30 /
multi −3.26, i.e. indistinguishable) — an insensitive test. The matched bbox-recipe pair above is
the valid comparison.

## Second victim: CMT (ICCV'23 Cross-Modal Transformer) — cross-family transfer

CMT = DETR-style, position-guided queries cross-attend to image + point tokens (no explicit
BEV-fusion stage): architecturally the furthest from BEVFusion, so the strongest "shared leak"
test. Same perturbed image roots, same 4 conditions. Clean baseline mAP 0.6689 / NDS 0.7017
reproduces the published R50 fusion model (0.679/0.708) within ~1 pt (fp32 vanilla-attention patch
replacing flash-attn). Repo /sfs/weka/scratch/tkg5kq/transfer/cmt, env cmt_env.

| condition         | mAP    | NDS    | car   | truck | bus   | car+truck+bus mean | Δ vs clean |
|-------------------|--------|--------|-------|-------|-------|--------------------|------------|
| clean             | 0.6689 | 0.7017 | 0.869 | 0.640 | 0.753 | 0.7537             | —          |
| random paint      | 0.6552 | 0.6931 | 0.844 | 0.592 | 0.726 | 0.7205             | −3.32      |
| single-optimized  | 0.6539 | 0.6928 | 0.842 | 0.590 | 0.722 | 0.7176             | −3.61      |
| multi-optimized   | 0.6460 | 0.6883 | 0.814 | 0.572 | 0.718 | 0.7012             | −5.25      |
| multi+nat         | 0.6474 | 0.6887 | 0.814 | 0.568 | 0.721 | 0.7008             | −5.29      |

**Same story cross-family**: transfer works (multi −5.25, car −5.5, beats random −3.32); multi >
single (−5.25 vs −3.61, a 1.64-pt gap — WIDER than BEVFusion's 1.16); naturalness free. On CMT the
single-regime texture is barely above random, which further isolates the multi-vehicle regime as
the carrier of transferable adversarial signal. (Note: TransFusion was the original 2nd-victim
plan but its fusion weights are unreleased — Huawei policy; CMT chosen as the distinct substitute.)

CMT job IDs: clean metric 22242179 (infer 22242178); transfer metrics random 22242981 / single
22242983 / multi 22242985 / multi+nat 22242987.

## Provenance (job IDs, 2026-10-31/11-01)
- single-mask training: 22240147 (natsingle); transfer chain composite 22240148, infer 22240641 (22240149 died on a sticky-bad node), metric 22240642
- old-recipe single/multi (superseded): banks logs/attack/22118064/uni_{single,multi}_shared_c8; metric 22239387 (single) / 22239389 (multi)
- clean baseline metric: 22231533 (infer outputs_22228850.pkl)
- compositing (CPU, full val, 20949 imgs / 64202 masks each): none 22237247, strong 22237255, random 22237258
- infer (GPU): none 22237672, strong 22237256, random 22237647  (earlier 22237251/22237645/22237259 died on the flaky spconv N=0 and were resubmitted)
- metric (CPU): none 22237673, strong 22237257, random 22237648
- perturbed image roots: /scratch/tkg5kq/transfer/imgroots/{none,strong,random}
- scripts: tools/composite_transfer.py, scripts/composite_transfer.sh, tools/bevfusion_eval.py, scripts/bevfusion_eval.sh, scripts/bevfusion_metric.sh
