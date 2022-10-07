import os
import sys
import time
import argparse

import torch
from torch.optim import SGD, Optimizer
'''
Set a environment variable 'PROJ_HOME' to the location of your project home directory.
The $PROJ_HOME directory should have the following structure:
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
|       └── val
└── checkpoints
    └── crt (where all model checkpoints will be saved)
'''

if not "PROJ_HOME" in os.environ:
    raise RuntimeError("environment variable for project home not set")

import helpers
from datasets import get_loader
from utils import get_config, init_logger, get_model, get_scheduler

torch.autograd.set_detect_anomaly(True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--do-certify', action='store_true')
    args = parser.parse_args()

    return args


def main():
    os.environ['PROJ_NAME'] = 'crt'

    # Getting Started
    args = get_args()
    config = get_config(args)
    del args

    logfilename = "certify.log" if config.do_certify else "output.log"
    logger = init_logger(config.outdir, logfilename, config.TRAIN.resume)
    logger.info(config)

    # GPU stuff
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    if torch.cuda.is_available():
        print('Running on GPU !!!')
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('Running on CPU !!!')
        device = torch.device('cpu')

    # Init dataloaders
    trainloader = get_loader(config.DATA.dataset, 'train', config.DATA.batch_size, config.DATA.workers)
    testloader = get_loader(config.DATA.dataset, 'test', config.DATA.test_batch_size, config.DATA.workers)

    # Prepare model
    model = get_model(config.arch, config.DATA.dataset, device)

    helper_args = {'config': config,
                   'trainloader': trainloader,
                   'testloader': testloader,
                   'device': device,
                   'logger': logger}
    helper = helpers.__dict__[config.method](**helper_args)

    # Certify and exit
    if config.do_certify:
        # load pre-trained model
        model, _, _, _ = helpers.load_ckpt(config.CERTIFY.load_path, model, resume=False, logger=logger)

        # standard testing base classifier
        test_report = helper.test_routine(model)
        logger.info(test_report)

        # certification
        start = time.time()
        helper.certify_routine(model, matfile=os.path.join(config.outdir, 'certificate.mat'))
        logger.info(f'Saving certificate.mat to: {config.outdir}')
        logger.info(f'Time taken: {time.time() - start:.4f}s')

        sys.exit()

    # Set-up for training
    optimizer = \
        SGD(model.parameters(), lr=config.TRAIN.lr, momentum=config.TRAIN.momentum, weight_decay=config.TRAIN.weight_decay)
    lr_steps = config.TRAIN.epochs * len(trainloader)
    milestones = [lr_steps / 2, lr_steps * 3 / 4]
    scheduler = get_scheduler(optimizer, config.TRAIN.lr_scheduler, config.TRAIN.lr, lr_steps, milestones)

    # Resume training or start from scratch
    if config.TRAIN.resume:
        # load pre-trained model and resume state of optimizer and scheduler
        model, optimizer, scheduler, curr_epoch =\
            helpers.load_ckpt(config.TRAIN.resume_ckpt, model, resume=True, optimizer=optimizer, scheduler=scheduler, logger=logger)
        helper.start_epoch = curr_epoch

    # training
    train_args = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}
    teacher = get_model(config.METHOD.t_arch, config.DATA.dataset, device)
    teacher, _, _, _ = helpers.load_ckpt(config.METHOD.t_loadpath, teacher, resume=False, logger=logger)
    teacher.eval()
    train_args['teacher'] = teacher

    helper.train_routine(**train_args)


if __name__ == '__main__':
    main()