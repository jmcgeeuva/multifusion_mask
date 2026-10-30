# Attack-loss design notes: region effects on the BEV head

Status: **discussion only, nothing implemented.** This records three loss ideas for the
IS-Fusion camouflage study so the design is on record before any code is written.

## Scope and framing

This is a digital robustness study on a **frozen** detector and a benchmark (nuScenes val).
Everything below is evaluated as a *measured, disclosed finding* — "does this detector have
this weakness, and which component is responsible" — not as a deployable road device. Two
rules carry through every idea here:

- **Evaluate under `--inference-norm`** (BatchNorm/Dropout on stored statistics). We already
  saw a train-mode BatchNorm collapse where one texture drove every vehicle to ~0.03,
  including untextured ones; it vanished under stored stats. Any "region-wide" effect must be
  shown under deployment norms or it is just that artifact.
- **Keep deployment claims out of the loss target and the paper's claims.** We report what the
  attack does on the benchmark and which fusion stage leaks; we do not optimize or claim a
  physical "paint job blinds the road" capability. For the region ideas this matters doubly,
  because false-positive (ghost) detections are a spoofing threat model with a more direct
  real-world-harm profile than hiding.

## Head mechanics the ideas depend on

TransFusion/IS-Fusion head (`transfusion_head_v2.py`):

1. `dense_heatmap = heatmap_head(lidar_feat)`; **`heatmap = dense_heatmap.detach().sigmoid()`**
   — the copy used for *proposal selection is detached*. Selection cannot be pushed directly;
   it is influenced only through the upstream features that produce `dense_heatmap`.
2. Selection = **local-max NMS** (`max_pool2d`, `nms_kernel_size`) → keep points that are local
   maxima → **top-k** (k = `num_proposals` = 200). So a detection must (a) be a local maximum
   and (b) rank in the top 200. Selection keys on **contrast**, not absolute height.
3. Camera→BEV (`fusion_encoder.img_point_sampling`, `grid_sample`) is close to **local**: a
   painted object's pixels inject features near that object's own BEV cell.
4. **Global mixing exists after fusion:** region/grid attention (`grid2region_att`) and
   instance-to-scene attention (`instance_fusion`). This is the only path by which one BEV
   location can affect another, and the only architectural opening for any "region" effect.
5. **LiDAR contradiction:** fused BEV = conv over `cat([img_bev, lidar_bev])`. An empty cell has
   near-zero LiDAR. Hiding only needs to *suppress* existing evidence; creating a ghost needs to
   *manufacture* evidence against a "no points here" LiDAR signal — strictly harder.

Current loss (baseline for all three): **per-vehicle suppression** — push down `dense_heatmap`
in a ±2-cell window at each painted vehicle's centre, plus its own-class probability among
proposals within 2 m. Vanishing-only log-space form (`--vanish-form bce --k-dense 5 --k-prop 3
--w-margin 0`) was best in the sweep.

---

## Idea A — region suppression ("glare" / fog)

**Target.** Instead of each painted car's own cell, suppress a *neighborhood* of the fused BEV
feature map, so other objects in that neighborhood also lose evidence. A term on BEV feature
energy or objectness over a disc around each painted car, not just its centre.

**Why there's an opening.** The global attention stages (4 above) give one location a path to
its neighbors.

**Why to bet against it.** Projection is local (3); the only cross-object route is attention,
which is trained to refine real objects, not broadcast corruption; and the natural "whole scene
drops at once" mechanism is the BatchNorm artifact, which did not survive inference-norm.

**Prediction.** Collateral ≈ 0 under inference-norm → a *reassuring* result (fusion localizes
the attack). If collateral is real → a disclosed vulnerability in a named attention stage.

## Idea B — peak flattening / "pull up"

**Idea (user).** Instead of pushing the heatmap down, pull it up so everything reads "hot" — a
glare that makes the signal uninformative.

**The real mechanism is contrast destruction, not "sees cars everywhere."** Because selection is
local-max + top-k (2), flooding a region toward saturation removes the peakiness real objects
need to be selected, and near-tied high cells eat the 200-proposal budget. So "everything hot"
acts as **suppression of real objects via the selection mechanism** — arguably a more robust
lever than per-logit push-down, since it attacks *how* proposals are chosen.

**The fork to watch: ghost detections.** If flooded cells survive local-max NMS, the outcome is
*phantom* objects, not "nothing detected." That is a different (spoofing) threat model and must
be measured separately, not left to drift.

**Gradient subtlety.** The selection heatmap is detached (1); act through upstream features.

**Prediction.** Under inference-norm, flattening suppresses real objects where it can raise the
local floor (mostly near painted cells); broad ghost creation is limited by the LiDAR
contradiction (5).

## Idea C — multi-object region saturation → false detections (user hypothesis)

**Hypothesis.** Painting *several* objects in the same camera view could manufacture **false
detections in the region between and around them**, not just at the painted cells.

**Why it is a sharper version of A/B.** With one painted object you drive one BEV location
directly; the inter-object region is unreachable except through attention. With several painted
objects you get **multiple directly-controllable anchor BEV cells that bracket the region
between them**, which helps in two ways:

- **Attention interpolation.** Region/grid and instance-to-scene attention (4) mix across cells.
  Corrupting several anchors at once gives the mixing more to propagate/interpolate into the
  cells between them than a single anchor does — a plausibly super-additive effect, which is the
  core of the hypothesis and worth testing directly against the single-object case.
- **Projection geometry.** An empty BEV cell between the objects, back-projected to the image
  (3, `grid_sample`), may land on or near painted pixels — so cells you do not "own" can still
  draw their image feature from paint you control. Multiple objects widen the set of inter-cells
  whose image ray passes through paint.

**Why to stay skeptical.** (i) A ghost in an empty cell has **no LiDAR** support (5); the image
branch must overpower "no points here" — hard. (ii) A ghost must become a local maximum and beat
real objects into the top-200 (2). (iii) Same BatchNorm-artifact trap: ghosts that appear in
train-mode may vanish under inference-norm.

**Prediction.** If any config produces real ghosts under inference-norm, multi-object is where it
shows up, and the inter-object region (vs. around a lone object) is the signature. If LiDAR
dominates as elsewhere, ghosts stay sparse and the finding is "fusion resists camera-only
spoofing," still worth reporting.

---

## Measurement protocol (shared)

All under `--inference-norm`, nuScenes val held-out, per-vehicle records already carry distance
and LiDAR-point count.

- **Suppression (A, B):** targeted conf and recall@0.3 on painted vehicles; **collateral** =
  same on untextured vehicles in the same scenes (already wired in). A real region effect = a
  drop in collateral beyond the noise floor (the clean-vs-clean repeat).
- **False positives (B, C):** count proposals above a score threshold that match **no** GT box
  (Hungarian/≥2 m to any GT). Report FP count and score, split by location:
  **(i)** inside the convex hull of the painted objects (the "between" region),
  **(ii)** a ring around each painted object, **(iii)** elsewhere. The C hypothesis predicts (i)
  rises with ≥2 painted objects and is super-additive vs. one object.
- **Ablation for C:** 1 vs 2 vs 3+ painted objects per camera, same total painted area, so the
  effect is "number of anchors," not "more paint."
- **Attribution:** if any region/FP effect survives, zero out each global stage (`grid2region_att`,
  `instance_fusion`) in turn to name the responsible component — this is the defensive deliverable.

## Go / no-go

Any of these graduates from "idea" to "result" only if the effect (collateral drop, or inter-region
FPs) **exceeds the clean-repeat noise floor under inference-norm** and is attributable to a named
stage. Otherwise the honest finding is that fusion localizes camera-only attacks, which is itself
a reportable robustness result and the expected outcome given the LiDAR contradiction.
