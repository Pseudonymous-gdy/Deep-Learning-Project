"""CIFAR dataset constructors and standard transforms.

Provides `make_cifar100` and `make_cifar10` which return torchvision Datasets
configured with the project's standard normalization and lightweight
augmentations used for Week-1 experiments.
"""
from torchvision import datasets, transforms
from .stats import DATASET_STATS

def _build_transforms(name: str, is_train: bool):
    mean, std = DATASET_STATS[name]["mean"], DATASET_STATS[name]["std"]
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

def make_cifar100(root: str, split: str = "train", download: bool = True):
    """Return CIFAR-100 dataset for given split (train/test)."""
    is_train = (split == "train")
    tfm = _build_transforms("cifar100", is_train)
    return datasets.CIFAR100(root=root, train=is_train, transform=tfm, download=download)

def make_cifar10(root: str, split: str = "train", download: bool = True):
    """Return CIFAR-10 dataset for given split (train/test)."""
    is_train = (split == "train")
    tfm = _build_transforms("cifar10", is_train)
    return datasets.CIFAR10(root=root, train=is_train, transform=tfm, download=download)
