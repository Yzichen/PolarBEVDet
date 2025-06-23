## Environment
1. Create a conda virtual environment and activate
```
conda create -n polarbevdet python=3.9
conda activate polarbevdet
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

2. Install PyTorch and torchvision following the official instructions
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. Install other dependencies:
```
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
mim install mmdet3d==1.0.0rc6
pip install setuptools==59.5.0
pip install numpy==1.23.5
pip install pillow==8.4.0
pip install waymo-open-dataset-tf-2-6-0=1.4.9
```

4. Compile CUDA extensions:
```
cd projects
python setup.py develop
```