import os
from typing import *
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


# list of all datasets
DATASETS = ["imagenet", "cifar"]

# For imagenet data, make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_PATH = f"{os.environ['PROJ_HOME']}/data/imagenet"
CIFAR_PATH = f"{os.environ['PROJ_HOME']}/data/CIFAR10"


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    assert dataset in DATASETS
    assert split in ['train', 'test']

    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar":
        return _cifar10(split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    assert dataset in DATASETS
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar":
        return 10


def get_normalize_layer(dataset: str, device) -> torch.nn.Module:
    assert dataset in DATASETS
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV, device)
    elif dataset == "cifar":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV, device)


def _cifar10(split: str) -> Dataset:
    if split == "train":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        return datasets.CIFAR10(CIFAR_PATH, train=True, download=True, transform=transform_train)
    elif split == "test":
        return datasets.CIFAR10(CIFAR_PATH, train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str) -> Dataset:
    if split == "train":
        subdir = os.path.join(IMAGENET_PATH, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(IMAGENET_PATH, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


def get_loader(dataset, split, batch_size, workers):
    assert split in ['train', 'test']
    pin_memory = (dataset == "imagenet")
    shuffle = (split == 'train')
    dataset = get_dataset(dataset, split)

    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=workers, pin_memory=pin_memory)
    loader = DataPrefetcher(loader)

    return loader


class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float], device):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


