import time
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseHelper
from .helper_utils import AverageMeter, accuracy

__all__ = ['XFERHelper', 'xfer']


class XFERHelper(BaseHelper):
    def __init__(self, config, trainloader: DataLoader, testloader: DataLoader, device, logger):
        super(XFERHelper, self).__init__('xfer', config, trainloader, testloader, device, logger)
        # method specific params
        self.alpha = config.METHOD.alpha
        self.cls_criterion = nn.CrossEntropyLoss()

    def _train_epoch_routine(self, epoch: int, model: nn.Module, optimizer, scheduler, **kwargs):
        avg_train_loss, train_acc = self._train_epoch_logit(epoch, model, optimizer, scheduler, **kwargs)
        train_report = f'Train: (loss) {avg_train_loss:.4f}, (acc) {train_acc:.2f}%'

        return train_report

    def _train_epoch(self, epoch: int, model: nn.Module, optimizer, scheduler, teacher: nn.Module):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        logit_losses = AverageMeter()
        cls_losses = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        # switch to train mode
        model.train()

        pbar = tqdm.tqdm(total=len(self.trainloader), leave=False)
        for i, (inputs, targets) in enumerate(self.trainloader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            noisy_inputs = inputs + torch.randn_like(inputs, device=self.device) * self.noise_sd

            # compute student output
            outputs = model(noisy_inputs)

            # compute teacher output
            with torch.no_grad():
                t_outputs = teacher(noisy_inputs)
            t_outputs = t_outputs.detach()

            # compute l2 loss between logits
            logit_loss = torch.mean(torch.norm(outputs - t_outputs, dim=1))

            # cross-entropy loss
            cls_loss = self.cls_criterion(outputs, targets)
            loss = self.alpha * logit_loss + (1.0 - self.alpha) * cls_loss

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            logit_losses.update(logit_loss.item(), inputs.size(0))
            cls_losses.update(cls_loss.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_description(f'Epoch: {epoch:03d}/{self.epochs - 1} '
                                 f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                                 f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                                 f'Transfer Loss {logit_losses.val:.4f} ({logit_losses.avg:.4f}) '
                                 f'CLS Loss {cls_losses.val:.4f} ({cls_losses.avg:.4f}) '
                                 f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                                 f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                                 f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
            pbar.update(1)
        pbar.close()

        return (losses.avg, top1.avg)


xfer = XFERHelper