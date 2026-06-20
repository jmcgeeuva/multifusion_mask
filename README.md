# multifusion_mask

```
bash scripts/dist_test.sh ./config/only_vehicles.py 4 --attack-filter sample_class --out results.pkl --eval bbox
bash scripts/dist_train.sh ./config/only_vehicles.py 4 
```

To test the baseline on the run with the same sample_class filter run the following:
```
sh ./scripts/dist_test.sh ./config/only_vehicles.py 4 --no-attack --reference-log ./results_attack_log.json --attack-filter sample_class --eval bbox --out baseline_attack.json
```


For visualization:
```
conda activate ssiai_adv
cd /sfs/gpfs/tardis/home/tkg5kq/workdir/av_project/multifusion_mask

python visualize_attack.py \
    --baseline  baseline_results.pkl \
    --attack    attack_results.pkl \
    --attack-log results_attack_log.json \
    --info-file /standard/eng_vivastorage/nuscenes/nuscenes/nuscenes_infos_val.pkl \
    --data-root /standard/eng_vivastorage/nuscenes/nuscenes \
    --out-dir   vis_comparison \
    --score-thr 0.2 \
    --max-samples 50 \
    --bev
```
What each output file contains:

0000_<token>_CAM_FRONT.jpg — 1600×(900+40+36) side-by-side JPEG:
Left panel: baseline detections in green, attacked-class boxes highlighted in orange
Right panel: attack detections in blue, attacked-class boxes highlighted in orange
Header strip names the camera and class; footer counts how many instances of the attacked class were detected in each
0000_<token>_bev.png (with --bev) — BEV top-down view, attacked class in red vs other classes in green, both runs side by side
To look at a specific token (e.g. one you know was attacked):


python visualize_attack.py ... --token 30e55a3ec6184d8cb1944b39ba19d622 --max-samples 1
One note: your baseline_results.pkl already exists, but you'll need to generate attack_results.pkl with your test.py using --out attack_results.pkl (no --no-attack flag). The script assumes both pkls have the same number of results in the same sample order as nuscenes_infos_val.pkl.

Training IS-Fusion

## Data Preparation

### Bash Terminal

In order to process each CAM_TYPE (CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT) for both sweeps and samples run the python script:

```
python ./yolo_test.py <CAM_TYPE> <sweeps/samples>
```

### Slurm

To process the masks faster SLURM can be used. The following will start 12 machines on Rivanna in order to run all CAM types on both sweeps and samples at the same time

```
sbatch ./scripts/yolo_test.sh sweeps
sbatch ./scripts/yolo_test.sh samples
```