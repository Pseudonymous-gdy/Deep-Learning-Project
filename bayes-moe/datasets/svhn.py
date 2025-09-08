"""SVHN helper used as an OOD placeholder dataset.

Only a minimal transform (ToTensor + Normalize) is applied. SVHN supports
splits 'train', 'test' and 'extra'; for OOD checks prefer 'test' or 'extra'.
"""

from torchvision import datasets, transforms
from .stats import DATASET_STATS


def make_svhn(root: str, split: str = "test", download: bool = True):
    '''
    Generate a SVHN dataset from torchvision.

    Inputs:
    - root: str, path to store the dataset.
    - split: str, split of dataset.
    - download: bool, whether to download the dataset.
    '''
    mean, std = DATASET_STATS["svhn"]["mean"], DATASET_STATS["svhn"]["std"] # get mean and std of svhn for normalization
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) # apply normalization
    # split could be 'train' | 'test' | 'extra'；When completting OOD, focus on 'test' or 'extra' first.
    return datasets.SVHN(root=root, split=split, transform=tfm, download=download)