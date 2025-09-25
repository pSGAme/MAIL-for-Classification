# Installation

This codebase is tested on Ubuntu 22.04.4 LTS with python 3.10.4. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n mail python=3.10.4

# Activate the environment
conda activate mail

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone MAIL code repository and install requirements
```bash
# Clone MaPLe code base
git clone https://github.com/pSGAme/MAIL-for-Classification.git

cd MAIL-for-Classification/
# Install requirements

pip install -r requirements.txt

# Update setuptools package 
pip install setuptools==59.5.0
```
