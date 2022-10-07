import os
import torch
import torch.nn as nn

CIFAR_CERT_GRID = (0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25)
IMAGENET_CERT_GRID = (0.50, 1.00, 1.50, 2.00, 2.50, 3.00)

__all__ = ['AverageMeter', 'accuracy', 'load_ckpt', 'save_ckpt', 'CIFAR_CERT_GRID', 'IMAGENET_CERT_GRID']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_ckpt(ckpt_path: str, model: nn.Module, resume=False, optimizer=None, scheduler=None, logger=None):
    assert os.path.exists(ckpt_path), ckpt_path
    sd = torch.load(ckpt_path)
    assert 'state_dict' in sd.keys()

    epoch = ''
    if 'epoch' in sd.keys():
        epoch = sd['epoch']

    if logger:
        print_fn = logger.info
    else:
        print_fn = print

    print_fn(f'Loading checkpoint from: {ckpt_path} (epoch {epoch})')
    model.load_state_dict(sd['state_dict'])
    if resume:
        assert (optimizer is not None) and (scheduler is not None)
        if 'optimizer' in sd.keys():
            print_fn('Resuming optimizer')
            optimizer.load_state_dict(sd['optimizer'])
        if 'scheduler' in sd.keys():
            print_fn('Resuming scheduler')
            scheduler.load_state_dict(sd['scheduler'])

    return model, optimizer, scheduler, sd['epoch'] + 1


def save_ckpt(save_path: str, epoch: int, model: nn.Module, optimizer, scheduler):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, save_path)
    return