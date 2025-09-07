from torch.utils.data import DataLoader
from .cifar import make_cifar10, make_cifar100
from .svhn import make_svhn

def build_dataset(name: str, root: str, split: str, download: bool = True):
    """Return a torchvision Dataset for the supported dataset `name`.

    Supported names: 'cifar100', 'cifar10', 'svhn'.
    """
    name = name.lower()
    if name == "cifar100": return make_cifar100(root, split, download)
    if name == "cifar10":  return make_cifar10(root,  split, download)
    if name == "svhn":     return make_svhn(root,     split, download)  # OOD 占位
    raise ValueError(f"Unknown dataset: {name}")


def build_loader(ds, batch_size: int, shuffle: bool, num_workers: int = 4):
    """Create a DataLoader with common defaults used by training scripts."""
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )