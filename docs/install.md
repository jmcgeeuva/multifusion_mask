# Step-by-step installation instructions MultiSensorAttack


**1. Create a conda virtual environment and activate it.**
```shell
conda create -n OpenOccupancy python=3.7 -y
source activate OpenOccupancy
```

**2. Install PyTorch and torchvision (tested on torch==1.10.1 & cuda=11.3).**
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 cudatoolkit-dev -c pytorch -c conda-forge -y
```

**3. Install gcc>=5 in conda env.**
```shell
conda install -c omgarcia gcc-6 -y # gcc-6.2
```

**4. Install MMCV following the [official instructions](https://github.com/open-mmlab/mmcv).**
```shell
pip install mmcv-full==1.4.0 --upgrade --force-reinstall -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```

**5. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
conda install ninja
```

**6. Install other dependencies.**
```shell
pip install ultralytics==8.0.145
pip install iopath==0.1.9 
pip install timm==0.6.13
pip install typing-extensions==4.5.0
pip install pylint 
pip install ipython==8.12
pip install numpy==1.19.5 numba==0.48.0 
pip install matplotlib==3.5.2 
pip install pandas
pip install scikit-image==0.19.3 
pip install fvcore
pip install einops
pip install seaborn
pip install open3d-python
pip install pymcubes==0.1.4
pip install spconv-cu113
pip install gdown
pip install yapf==0.40.1
pip install setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
```

**7. Install IS-Fusion**

```
git clone --recursive git@github.com:jmcgeeuva/multifusion_mask.git
cd multifusion_mask/IS-Fusion
# Install the IS-Fusion custom mmdetection3d
python -m pip install -v -e .
python setup.py install
```