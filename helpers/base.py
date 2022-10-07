import os
import time
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from .helper_utils import AverageMeter, accuracy, save_ckpt, CIFAR_CERT_GRID, IMAGENET_CERT_GRID
from .rs.certify import certify


class BaseHelper(object):
    def __init__(self, method, config, trainloader: DataLoader, testloader: DataLoader, device, logger):
        logger.info(f'Using {method} helper !!!')
        self.outdir = config.outdir
        self.logger = logger
        self.device = device

        # Dataloaders
        self.trainloader = trainloader
        self.testloader = testloader

        # Hyperparams
        self.arch = config.arch
        self.noise_sd = config.noise_sd
        self.epochs = config.TRAIN.epochs
        self.num_classes = config.DATA.num_classes
        self.start_epoch = 0

        # For testing/certification
        self.test_criterion = nn.CrossEntropyLoss()
        self.start_img = config.CERTIFY.start_img
        self.skip = config.CERTIFY.skip
        self.beta = config.CERTIFY.beta
        self.certify_interval = config.CERTIFY.certify_interval
        self.certify_after = config.CERTIFY.certify_after
        if config.DATA.dataset == 'cifar':
            self.cert_grid = CIFAR_CERT_GRID
        elif config.DATA.dataset == 'imagenet':
            self.cert_grid = IMAGENET_CERT_GRID

    def train_routine(self, model, optimizer, scheduler, **kwargs):
        '''
        General training wrapper with per-epoch testing and periodic certification.
        Parameters
        ----------
        model
        optimizer
        scheduler

        Returns
        -------
        N/A
        '''
        for epoch in range(self.start_epoch, self.epochs):
            before = time.time()
            train_report = self._train_epoch_routine(epoch, model, optimizer, scheduler, **kwargs)
            after = time.time()
            test_report = self.test_routine(model)

            curr_lr = scheduler.get_last_lr()[0]
            log_str = f'Epoch: {epoch:03d}/{self.epochs-1} (lr: {curr_lr:.4f}) | Time elapsed: {after - before:.4f}s | {train_report} | {test_report}'
            self.logger.info(log_str)

            save_path = os.path.join(self.outdir, 'checkpoint.pth.tar')
            save_ckpt(save_path, epoch, model, optimizer, scheduler)
            if (epoch + 1) % 50 == 0:
                save_path = os.path.join(self.outdir, f'epoch_{epoch}.pth.tar')
                print(f'Saving: {save_path}')
                save_ckpt(save_path, epoch, model, optimizer, scheduler)

            if (epoch + 1) % self.certify_interval == 0 and (epoch + 1) >= self.certify_after:
                self.certify_routine(model, matfile=os.path.join(self.outdir, '{}.mat'.format(epoch)))

    def _train_epoch_routine(self, epoch, model, optimizer, scheduler, **kwargs):
        # Method specific
        raise NotImplementedError

    def test_routine(self, model):
        avg_loss_clean, avg_loss_noise, acc_clean, acc_noise = self._standard_test(model)
        test_report = f'Clean Test: (loss) {avg_loss_clean:.4f}, (acc) {acc_clean:.2f}% | Noisy Test: (loss) {avg_loss_noise:.4f}, (acc) {acc_noise:.2f}%'
        return test_report

    def _standard_test(self, model):
        '''
        Evaluates on clean and noisy (Gaussian noise) samples.
        Parameters
        ----------
        model: target classifier
        criterion: loss function to compute test loss

        Returns
        -------
        losses.avg: average loss on clean test set.
        losses_noise.avg: average loss on noisy test set.
        top1.avg: accuracy on clean test set
        top1_noise.avg: accuracy on noisy test set
        '''
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_noise = AverageMeter()
        top1 = AverageMeter()
        top1_noise = AverageMeter()
        end = time.time()

        # switch to eval mode
        model.eval()

        pbar = tqdm.tqdm(total=len(self.testloader), leave=False)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.testloader):
                # measure data loading time
                data_time.update(time.time() - end)

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                loss = self.test_criterion(outputs, targets)

                # measure accuracy and record loss
                acc1, _ = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1.item(), inputs.size(0))

                # augment inputs with noise
                inputs = inputs + torch.randn_like(inputs, device='cuda') * self.noise_sd

                # compute output
                outputs = model(inputs)
                loss = self.test_criterion(outputs, targets)

                # measure accuracy and record loss
                acc1, _ = accuracy(outputs, targets, topk=(1, 5))
                losses_noise.update(loss.item(), inputs.size(0))
                top1_noise.update(acc1.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                pbar.set_description(f'Test: [{i}/{len(self.testloader)}] '
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                      f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                      f'Acc {top1.val:.3f} ({top1.avg:.3f}) '
                      f'Noise Loss {losses_noise.val:.4f} ({losses_noise.avg:.4f}) '
                      f'Noise Acc {top1_noise.val:.3f} ({top1_noise.avg:.3f})')
                pbar.update(1)
            pbar.close()

        return (losses.avg, losses_noise.avg, top1.avg, top1_noise.avg)

    def certify_routine(self, model, matfile=None, do_pbar=True):
        model.eval()
        return certify(model, self.device, self.testloader.dataset, self.num_classes,
                       mode='hard', start_img=self.start_img, sigma=self.noise_sd, beta=self.beta,
                       matfile=matfile, logger=self.logger, do_pbar=do_pbar, skip=self.skip,
                       grid=self.cert_grid)
