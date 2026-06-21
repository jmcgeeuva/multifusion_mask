# AUTO-GENERATED ABLATION CONFIG
# Base: only_vehicles.py
# Suite: bbox
# Experiment: baseline (original L1 loss only)

_base_ = ['../only_vehicles.py']

model = dict(
    pts_bbox_head=dict(
        loss_bbox=dict(
            # ── active component ───────────────────────────────────────────
            use_original_l1=True,
            lambda_original=1.0,
            # ── all other BBoxAttack components disabled ────────────────────
            use_reverse_l1=False,
            lambda_reverse=1.0,
            use_translation_attack=False,
            lambda_translation=1.0,
            use_orbit_attack=False,
            lambda_orbit=1.0,
            use_scale_attack=False,
            lambda_scale=1.0,
            use_orientation_attack=False,
            lambda_orientation=1.0,
        ),
        # ── all GIAD components disabled ───────────────────────────────────
        loss_heatmap=dict(
            use_original_gaussian_loss=False,
            lambda_original=1.0,
            use_reverse_gaussian_loss=False,
            lambda_reverse=1.0,
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

work_dir = 'work_dirs/bbox/bbox0'
