# Dataset mean/std statistics used for normalization.
# Values taken from standard references for CIFAR-10/100 and SVHN.
DATASET_STATS = {
    "cifar10":  {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "cifar100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
    "svhn":     {"mean": (0.4377, 0.4438, 0.4728), "std": (0.1980, 0.2010, 0.1970)},
}