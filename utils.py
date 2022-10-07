import os
import sys
import yaml
import logging
from easydict import EasyDict
import numpy as np

import torch
import torch.nn as nn

import cifar_models
import imagenet_models
from datasets import get_normalize_layer


def get_config(args):
    assert os.path.exists(args.config)
    # read config file
    with open(args.config) as f:
        config = EasyDict(yaml.load(f))

    for k, v in vars(args).items():
        config[k] = v

    # parse outdir
    config.outdir = f'{os.environ["PROJ_HOME"]}/checkpoints/{os.environ["PROJ_NAME"]}/{config.method}/{config.arch}-{config.DATA.dataset}-{config.noise_sd}'
    # add method params
    config.outdir += f'-alpha={config.METHOD.alpha}'
    config.outdir += f'-teacher={config.METHOD.t_arch}'
    # add experiment identifier
    if config.exp_id != '':
        config.outdir += f'-{config.exp_id}'
    # create outdir
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir, exist_ok=True)

    return config


def init_logger(outdir, logfilename, resume):
    logfile = os.path.join(outdir, logfilename)
    if os.path.exists(logfile) and (not resume):
        os.remove(logfile)

    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(logfile))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger


def get_model(arch, dataset, device):
    '''
    Prepares model as follows:
    1. initialize architecture
    2. wrap in DataParallel if multi-gpu mode
    3. prepend normalization layer

    Parameters
    ----------
    arch: for initialization
    dataset: for setting proper normalization layer
    device: to move to gpu

    Returns
    -------
    model: ready to use model
    '''

    # Init model
    if dataset == 'cifar':
        model = cifar_models.__dict__[arch]()
    elif dataset == 'imagenet':
        model = imagenet_models.__dict__[arch]()
    else:
        raise ValueError

    if device == torch.device('cuda') and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Add normalization layer
    normalize_layer = get_normalize_layer(dataset, device)
    model = torch.nn.Sequential(normalize_layer, model)

    return model


def get_scheduler(optimizer, lr_scheduler, lr, lr_steps, milestones):
    if lr_scheduler == 'multistep':
        assert milestones != None
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif lr_scheduler == 'cosine':
        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (
                    1 + np.cos(step / total_steps * np.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                lr_steps,
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / lr))
    else:
        raise ValueError

    return scheduler

