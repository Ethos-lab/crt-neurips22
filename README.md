# Accelerating Certified Robustness Training via Knowledge Transfer (NeurIPS'22)

This repository contains code for our paper:

**Accelerating Certified Robustness Training via Knowledge Transfer**<br>
_Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati_<br>

[//]: # (Paper: <add link>)


### Citation
```
@inproceedings{vaishnavi2022crt,
 author = {Vaishnavi, Pratik and Eykholt, Kevin and Rahmati, Amir},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {SmoothMix: Training Confidence-calibrated Smoothed Classifiers for Certified Robustness},
 year = {2022}
}
```

## Getting Started

- **Step 1:** Set a environment variable `PROJ_HOME` to the location of your project home directory.

- **Step 2:** Get code and install dependencies:
```
cd $PROJ_HOME
git clone https://github.com/Ethos-lab/crt-neurips22.git

conda create -n crt python=3.7
conda activate crt
pip install -r requirements.txt
# Installs PyTorch version 1.9 with with CUDA 11.1; see https://pytorch.org/ for the correct command for your system
```

- **Step 3:** Download all required data and place it inside `$PROJ_HOME/data` directory.<br>
The CIFAR10 dataset will download automatically.<br>
For downloading the ImageNet dataset, see instructions [here](https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).

At the end of the setup process, the `$PROJ_HOME` will have the following structure:
````bash
$PROJ_HOME
├── crt-neurips22 (code)
├── data
│   ├── CIFAR10
│   │   └── cifar-10-batches-py
│   │       ├── data_batch_1
│   │       ├── data_batch_2
│   │       ├── data_batch_3
│   │       ├── data_batch_4
│   │       └── test_batch
│   └── imagenet
│       ├── train
│       └── val
└── checkpoints //will be created automatically to store model checkpoints
    └── crt
````

## Training
CRT is a general-purpose method for accelerating randomized smoothing based methods for training certifiably robust NNs. To train a NN (like ResNet110) using CRT on CIFAR-10, follow these steps:

- **Step 1:** Train a small teacher network (like ResNet20) to be certifiably robust using a state-of-the-art method (like [SmoothMix](https://github.com/jh-jeong/smoothmix)). 
- **Step 2:** Train the student network (ResNet110) using CRT:
```
python main.py --gpu 0 --config configs/cifar/xfer.yml
```
Please appropriately set the parameters in the config file. See [Config File](#config-file) for more details.

## Certifying

The certification script is based on the one from the [MACER repo](https://github.com/RuntianZ/macer). To certify the robustness of a smoothed classifier, set the `load_path` in the config file and run:

```python main.py --gpu 0 --config configs/cifar/xfer.yml --do-certify```

This will load the base classifier saved at path provided at `load_path`, smooth it using noise level `noise_sd` (also in the config file),
and certify with parameters `N0=100`, `N=100000`, and `alpha=0.001`.

For CIFAR-10, the entire test set will be used. For ImageNet, 500 test images will be used.

## Config File

We provide example config files for both CIFAR-10 and ImageNet. It is described below:
```
method: 'xfer'      => don't change
arch: 'resnet20'    => student architecture
noise_sd: 0.25      => noise level (sigma)
exp_id: ''          => (optional) unique identifier for experiment
METHOD:
  t_arch: 'resnet110'       => teacher architecture
  t_loadpath: '/path/to/teacher/checkpoint.pth.tar'     => load path for teacher checkpoint
  alpha: 1.0        => weight for transfer loss (we use 1). cross entropy loss gets weight of 1 - alpha.
TRAIN:
  epochs: 200
  lr: 0.1
  momentum: 0.9
  weight_decay: 0.0001
  lr_scheduler: 'multistep'
  resume: False     => to resume training, set this to true and provide load path below.
  resume_ckpt: ''
DATA:
  dataset: 'cifar'
  num_classes: 10
  batch_size: 128
  test_batch_size: 1000
  workers: 4
CERTIFY:
  load_path: ''     => to perform certification, provide load path here and pass --do-certify flag with main.py.
  start_img: 0
  skip: 1
  beta: 1.0
  certify_interval: 1
  certify_after: 200     
```