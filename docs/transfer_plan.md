# Cross-model transfer plan (IS-Fusion → BEVFusion → TransFusion)

## Why
The attack is currently white-box on IS-Fusion. If a texture optimized on IS-Fusion also degrades a
**different, never-optimized** fusion detector, the claim upgrades from "IS-Fusion is vulnerable" to
"the camera-side leak is shared across the BEV-fusion family, exploitable black-box via transfer."

## Key feasibility insight
We perturb the **image** (segmenter mask + GT-box placement + compositing are all nuScenes-level,
detector-agnostic). The composited adversarial images are reusable as-is; per detector we only need
its **inference wrapper** and its **score/metric readout**. No re-optimization, no per-detector
compositing.

## Victims
1. **BEVFusion** (first) — canonical BEV-fusion baseline; cleanest "shared leak" story.
2. **TransFusion** (second) — transformer/proposal-level fusion; breadth across fusion *types*.

## Environment (main risk)
BEVFusion / TransFusion need their own mmdet3d + CUDA ops, likely a **separate conda env** from
`ssiai_adv` (torch 1.10.1 / mmcv-full 1.4.0). Budget: stand up one second detector and reproduce its
**clean** nuScenes mAP/NDS before any attack.

## Protocol
- Take the **frozen** IS-Fusion universal textures (no-reg `natreguni/none/bank_last.npy` and
  naturalized `natreguni/strong/bank_last.npy`, already saved).
- Composite onto the **same held-out scenes**, run detector B, no optimization.
- Conditions per detector: **clean · random-texture (control) · transfer (IS-Fusion texture) ·
  white-box (optimized directly on B = upper bound / transfer-gap reference)**.
- Transfer matrix over {IS-Fusion, B}: diagonal = white-box, off-diagonal = transfer.

## Metrics
- Primary: official nuScenes **mAP / NDS** drop on B from the transferred texture vs clean B.
- Secondary: per-vehicle hiding (own-class conf if B exposes a comparable heatmap; else
  detection-level "GT vehicle detected > 0.3?").
- Report the **transfer gap** (white-box B − transfer) to be honest about model-specificity.

## Deliverable
Transfer table (detectors × {clean, random, transfer, white-box} with mAP/NDS) + optional
transfer-matrix figure. New experiments subsection "Cross-model transfer."

## Order of work
1. Stand up BEVFusion env + reproduce clean nuScenes mAP/NDS.
2. Thin `eval_transfer.py`: ingest our composited images, run B's official eval (reuse the
   `official_table.py` pipeline).
3. Run clean / random / transfer for both saved textures → first transfer numbers.
4. If strong: add TransFusion for family breadth; add white-box-on-B upper bound.

## Risks & mitigations
- **Env/checkpoint** — pick a detector with a known-good nuScenes checkpoint first; validate clean
  metrics before attacking.
- **Weak transfer** — still publishable (quantifies model-specificity); optionally boost with
  input-diversity/EoT during the original optimization or a 2-model ensemble objective (follow-up).
- **Readout mismatch** — fall back to detection-level hiding if B's internal heatmap isn't
  comparable; official mAP/NDS is always comparable.
