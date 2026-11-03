# Final seed-aggregated results (deterministic)

Recomputed from eval.json by `tools/aggregate_results.py` (same manifests as `tools/paper_tables.py`). `seeds` lists the per-seed values; `mean`/`std` are over them.


## attribution

| row | column | seeds | mean | std | n |
|---|---|---|---|---|---|
| hull:both | clean | 0.0600;0.0607;0.0602 | 0.0603 | 0.0003 | 3 |
| hull:both | effect | 2.6500;2.5500;2.4500 | 2.5500 | 0.0816 | 3 |
| hull:both | learned | 0.0392;0.0338;0.0389 | 0.0373 | 0.0025 | 3 |
| hull:grid2region | clean | 0.2008;0.2013;0.2008 | 0.2010 | 0.0002 | 3 |
| hull:grid2region | effect | 4.0500;3.3500;3.9750 | 3.7917 | 0.3138 | 3 |
| hull:grid2region | learned | 0.1257;0.1341;0.1117 | 0.1238 | 0.0092 | 3 |
| hull:instance_fusion | clean | 0.8202;0.8202;0.8201 | 0.8202 | 0.0000 | 3 |
| hull:instance_fusion | effect | 7.8250;8.2500;7.6000 | 7.8917 | 0.2695 | 3 |
| hull:instance_fusion | learned | 0.6896;0.6718;0.6968 | 0.6861 | 0.0105 | 3 |
| hull:none | clean | 0.7634;0.7634;0.7634 | 0.7634 | 0.0000 | 3 |
| hull:none | effect | 8.6000;9.7750;9.1750 | 9.1833 | 0.4797 | 3 |
| hull:none | learned | 0.6149;0.6213;0.6066 | 0.6143 | 0.0060 | 3 |
| suppress:both | clean | 0.0589;0.0592;0.0592 | 0.0591 | 0.0001 | 3 |
| suppress:both | effect | 0.0102;0.0105;0.0098 | 0.0102 | 0.0003 | 3 |
| suppress:both | learned | 0.0001;0.0001;0.0001 | 0.0001 | 0.0000 | 3 |
| suppress:grid2region | clean | 0.2008;0.2011;0.2011 | 0.2010 | 0.0001 | 3 |
| suppress:grid2region | effect | 0.1138;0.1073;0.1018 | 0.1076 | 0.0049 | 3 |
| suppress:grid2region | learned | 0.0002;0.0002;0.0003 | 0.0002 | 0.0000 | 3 |
| suppress:instance_fusion | clean | 0.8201;0.8201;0.8201 | 0.8201 | 0.0000 | 3 |
| suppress:instance_fusion | effect | 0.4908;0.4997;0.4993 | 0.4966 | 0.0041 | 3 |
| suppress:instance_fusion | learned | 0.0022;0.0021;0.0022 | 0.0022 | 0.0000 | 3 |
| suppress:none | clean | 0.7634;0.7634;0.7634 | 0.7634 | 0.0000 | 3 |
| suppress:none | effect | 0.4795;0.4843;0.4762 | 0.4800 | 0.0033 | 3 |
| suppress:none | learned | 0.0035;0.0041;0.0040 | 0.0039 | 0.0003 | 3 |

## ghosts

| row | column | seeds | mean | std | n |
|---|---|---|---|---|---|
| clean | elsewhere | 3.9670;3.9500;3.9670 | 3.9613 | 0.0080 | 3 |
| clean | hull | 0.3830;0.3830;0.3830 | 0.3830 | 0.0000 | 3 |
| clean | ring | 0.2170;0.2170;0.2170 | 0.2170 | 0.0000 | 3 |
| hull (multi) | elsewhere | 4.2750;4.7000;4.4000 | 4.4583 | 0.1783 | 3 |
| hull (multi) | hull | 8.7250;9.0000;8.8750 | 8.8667 | 0.1124 | 3 |
| hull (multi) | ring | 0.4250;0.4250;0.4250 | 0.4250 | 0.0000 | 3 |
| hull (single) | elsewhere | 6.4000;6.3000;6.4500 | 6.3833 | 0.0624 | 3 |
| hull (single) | hull | 0.0000;0.0000;0.0000 | 0.0000 | 0.0000 | 3 |
| hull (single) | ring | 0.4000;0.4500;0.4000 | 0.4167 | 0.0236 | 3 |
| pull-up | elsewhere | 3.9750;4.1000;4.1250 | 4.0667 | 0.0656 | 3 |
| pull-up | hull | 4.5000;4.5500;5.0250 | 4.6917 | 0.2366 | 3 |
| pull-up | ring | 5.0500;4.4500;5.2250 | 4.9083 | 0.3319 | 3 |
| random | elsewhere | 3.5250;3.4250;3.3500 | 3.4333 | 0.0717 | 3 |
| random | hull | 0.5250;0.5000;0.5500 | 0.5250 | 0.0204 | 3 |
| random | ring | 0.2500;0.2250;0.2750 | 0.2500 | 0.0204 | 3 |
| suppress | elsewhere | 3.4250;3.6250;3.5750 | 3.5417 | 0.0850 | 3 |
| suppress | hull | 0.5750;0.7250;0.6000 | 0.6333 | 0.0656 | 3 |
| suppress | ring | 0.3000;0.3250;0.3000 | 0.3083 | 0.0118 | 3 |

## hiding

| row | column | seeds | mean | std | n |
|---|---|---|---|---|---|
| Per-scene (upper bound) | clean | 0.7635;0.7634;0.7634 | 0.7634 | 0.0000 | 3 |
| Per-scene (upper bound) | learned | 0.0211;0.0229;0.0229 | 0.0223 | 0.0008 | 3 |
| Per-scene (upper bound) | random | 0.6630;0.6418;0.6203 | 0.6417 | 0.0174 | 3 |
| Per-scene (upper bound) | recall | 0.0045;0.0169;0.0169 | 0.0128 | 0.0058 | 3 |
| Per-sequence (bbox, held-out) | clean | 0.7145;0.7144;0.7143 | 0.7144 | 0.0001 | 3 |
| Per-sequence (bbox, held-out) | learned | 0.5244;0.5435;0.5310 | 0.5330 | 0.0079 | 3 |
| Per-sequence (bbox, held-out) | random | 0.6462;0.6467;0.6464 | 0.6464 | 0.0002 | 3 |
| Per-sequence (bbox, held-out) | recall | 0.7606;0.7880;0.7731 | 0.7739 | 0.0112 | 3 |
| Universal, screen | clean | 0.7635;0.7634;0.7634 | 0.7634 | 0.0000 | 3 |
| Universal, screen | learned | 0.6063;0.5996;0.6023 | 0.6027 | 0.0028 | 3 |
| Universal, screen | random | 0.6470;0.6416;0.6427 | 0.6438 | 0.0023 | 3 |
| Universal, screen | recall | 0.8348;0.8393;0.8482 | 0.8408 | 0.0056 | 3 |
| Universal, tile | clean | 0.7634;0.7634;0.7634 | 0.7634 | 0.0000 | 3 |
| Universal, tile | learned | 0.6068;0.6048;0.6062 | 0.6059 | 0.0008 | 3 |
| Universal, tile | random | 0.6700;0.6703;0.6707 | 0.6703 | 0.0003 | 3 |
| Universal, tile | recall | 0.8438;0.8393;0.8438 | 0.8423 | 0.0021 | 3 |
| \textbf{Universal, bbox} | clean | 0.7634;0.7634;0.7634 | 0.7634 | 0.0000 | 3 |
| \textbf{Universal, bbox} | learned | 0.2924;0.2886;0.2821 | 0.2877 | 0.0043 | 3 |
| \textbf{Universal, bbox} | random | 0.6633;0.6623;0.6642 | 0.6633 | 0.0008 | 3 |
| \textbf{Universal, bbox} | recall | 0.4241;0.4286;0.4152 | 0.4226 | 0.0056 | 3 |

## lidar_transfer

| row | column | seeds | mean | std | n |
|---|---|---|---|---|---|
| BEVFusion | 0-4_pts_hide% | 21.0000 | 21.0000 | 0.0000 | 1 |
| BEVFusion | 150-499_pts_hide% | 1.1000 | 1.1000 | 0.0000 | 1 |
| BEVFusion | 20-49_pts_hide% | 4.4000 | 4.4000 | 0.0000 | 1 |
| BEVFusion | 5-19_pts_hide% | 8.4000 | 8.4000 | 0.0000 | 1 |
| BEVFusion | 50-149_pts_hide% | 3.6000 | 3.6000 | 0.0000 | 1 |
| BEVFusion | 500+_pts_hide% | 0.5000 | 0.5000 | 0.0000 | 1 |
| BEVFusion | corr_hide_logpts | -0.2340 | -0.2340 | 0.0000 | 1 |
| CMT | 0-4_pts_hide% | 14.1000 | 14.1000 | 0.0000 | 1 |
| CMT | 150-499_pts_hide% | 4.5000 | 4.5000 | 0.0000 | 1 |
| CMT | 20-49_pts_hide% | 7.8000 | 7.8000 | 0.0000 | 1 |
| CMT | 5-19_pts_hide% | 10.1000 | 10.1000 | 0.0000 | 1 |
| CMT | 50-149_pts_hide% | 5.8000 | 5.8000 | 0.0000 | 1 |
| CMT | 500+_pts_hide% | 2.4000 | 2.4000 | 0.0000 | 1 |
| CMT | corr_hide_logpts | -0.1380 | -0.1380 | 0.0000 | 1 |

## patch

| row | column | seeds | mean | std | n |
|---|---|---|---|---|---|
| Full texture (per-scene) | clean | 0.7635;0.7634;0.7634 | 0.7634 | 0.0000 | 3 |
| Full texture (per-scene) | learned | 0.0211;0.0229;0.0229 | 0.0223 | 0.0008 | 3 |
| Full texture (per-scene) | recall | 0.0045;0.0169;0.0169 | 0.0128 | 0.0058 | 3 |
| Full texture (universal bbox) | clean | 0.7634;0.7634;0.7634 | 0.7634 | 0.0000 | 3 |
| Full texture (universal bbox) | learned | 0.2924;0.2886;0.2821 | 0.2877 | 0.0043 | 3 |
| Full texture (universal bbox) | recall | 0.4241;0.4286;0.4152 | 0.4226 | 0.0056 | 3 |
| Patch 20\% box (per-scene) | clean | 0.7634;0.7634;0.7634 | 0.7634 | 0.0000 | 3 |
| Patch 20\% box (per-scene) | learned | 0.0851;0.0770;0.0844 | 0.0822 | 0.0037 | 3 |
| Patch 20\% box (per-scene) | recall | 0.0670;0.0678;0.0678 | 0.0675 | 0.0004 | 3 |
| Patch 20\% box (universal) | clean | 0.7634;0.7634;0.7634 | 0.7634 | 0.0000 | 3 |
| Patch 20\% box (universal) | learned | 0.6967;0.6978;0.6923 | 0.6956 | 0.0024 | 3 |
| Patch 20\% box (universal) | recall | 0.9420;0.9420;0.9420 | 0.9420 | 0.0000 | 3 |
| Patch 40\% box (per-scene) | clean | 0.7634;0.7634;0.7634 | 0.7634 | 0.0000 | 3 |
| Patch 40\% box (per-scene) | learned | 0.0179;0.0174;0.0212 | 0.0188 | 0.0017 | 3 |
| Patch 40\% box (per-scene) | recall | 0.0134;0.0169;0.0169 | 0.0157 | 0.0016 | 3 |

## targeting

| row | column | seeds | mean | std | n |
|---|---|---|---|---|---|
| Both | attack | 0.0177;0.0198;0.0178 | 0.0184 | 0.0010 | 3 |
| Both | clean | 0.7641;0.7641;0.7639 | 0.7640 | 0.0001 | 3 |
| Both | ghosts | 0.9500;0.8000;0.8500 | 0.8667 | 0.0624 | 3 |
| Both | recall | 0.0079;0.0158;0.0119 | 0.0119 | 0.0032 | 3 |
| Non-vehicles (ped/bike/moto) | attack | 0.0394;0.0415;0.0384 | 0.0398 | 0.0013 | 3 |
| Non-vehicles (ped/bike/moto) | clean | 0.8159;0.8159;0.8160 | 0.8159 | 0.0000 | 3 |
| Non-vehicles (ped/bike/moto) | ghosts | 0.1000;0.1000;0.1000 | 0.1000 | 0.0000 | 3 |
| Non-vehicles (ped/bike/moto) | recall | 0.0492;0.0410;0.0574 | 0.0492 | 0.0067 | 3 |
| Vehicles (car/truck/bus) | attack | 0.0032;0.0056;0.0049 | 0.0046 | 0.0010 | 3 |
| Vehicles (car/truck/bus) | clean | 0.7520;0.7520;0.7520 | 0.7520 | 0.0000 | 3 |
| Vehicles (car/truck/bus) | ghosts | 0.7500;0.9500;0.9500 | 0.8833 | 0.0943 | 3 |
| Vehicles (car/truck/bus) | recall | 0.0000;0.0000;0.0000 | 0.0000 | 0.0000 | 3 |
