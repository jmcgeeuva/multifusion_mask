# multifusion_mask

```
bash scripts/dist_test.sh ./config/only_vehicles.py 4 --eval bbox
bash scripts/dist_train.sh ./config/only_vehicles.py 4 
```

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