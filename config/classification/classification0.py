# AUTO-GENERATED ABLATION CONFIG
# Base: only_vehicles.py
# Suite: classification
# Experiment: baseline (original focal loss only)

_base_ = ['../only_vehicles.py']

model = dict(
    pts_bbox_head=dict(
        loss_cls=dict(
            # ── active component ───────────────────────────────────────────
            use_original_focal=True,
            lambda_original=1.0,
            # ── all other ClassAttack components disabled ───────────────────
            use_reverse_focal=False,
            lambda_reverse=1.0,
            use_complement_loss=False,
            lambda_complement=1.0,
            use_uniform_confusion=False,
            lambda_uniform=1.0,
            use_margin_confusion=False,
            lambda_margin=1.0,
            use_hard_wrong_class=False,
            lambda_hard_wrong=1.0,
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

work_dir = 'work_dirs/classification/classification0'
