# AUTO-GENERATED ABLATION CONFIG
# Base: only_vehicles.py
# Suite: heatmap
# Experiment: reverse gaussian

_base_ = ['../only_vehicles.py']

model = dict(
    pts_bbox_head=dict(
        loss_heatmap=dict(
            # ── active component ───────────────────────────────────────────
            use_reverse_gaussian_loss=True,
            lambda_reverse=1.0,
            # ── all other GIAD components disabled ─────────────────────────
            use_original_gaussian_loss=False,
            lambda_original=1.0,
            use_ring_loss=False,
            lambda_ring=1.0,
            use_attention_diffusion=False,
            lambda_diffusion=1.0,
            use_entropy_loss=False,
            lambda_entropy=1.0,
            use_contrast_loss=False,
            lambda_contrast=1.0,
            use_veiling_luminance=False,
            lambda_veiling=1.0,
        ),
    )
)

work_dir = 'work_dirs/heatmap/heatmap1'
