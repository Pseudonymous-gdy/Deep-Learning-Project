import torch
import os, argparse, yaml, random, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from metrics.ece import expected_calibration_error
from metrics.nll import negative_log_likelihood
from utils.reliability import plot_reliability


def set_seed(s):
    """Set random seeds for reproducibility (CPU and CUDA)."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_loaders(name, root, bs, nw):
    """Return (train_loader, test_loader) for supported datasets.

    Currently supports: cifar100 (standard torchvision loaders).
    """
    if name.lower() == "cifar100":
        # CIFAR-100 dataset normalization stats (mean,std)
        mean, std = (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)
        T_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean, std)])
        T_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train = datasets.CIFAR100(root=root, train=True, download=True, transform=T_train)
        test  = datasets.CIFAR100(root=root, train=False, download=True, transform=T_test)
        return DataLoader(train, bs, shuffle=True, num_workers=nw, pin_memory=True), \
               DataLoader(test,  bs, shuffle=False, num_workers=nw, pin_memory=True)
    raise ValueError(f"Unknown dataset {name}")


def evaluate(net, loader, device):
    """Run a full evaluation pass and compute accuracy, ECE and NLL.

    Returns: acc, ece, nll, bin_stats
    """
    net.eval(); total, correct = 0, 0
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = net(x)
            # collect on CPU to save GPU memory
            all_logits.append(logits.cpu()); all_targets.append(y.cpu())
            pred = logits.argmax(1); total += y.size(0); correct += (pred==y).sum().item()
    logits = torch.cat(all_logits); targets = torch.cat(all_targets)
    acc = correct/total
    # compute calibration and likelihood metrics
    ece, bin_stats = expected_calibration_error(logits, targets, n_bins=15)
    nll = negative_log_likelihood(logits, targets)
    return acc, ece, nll, bin_stats


def main(cfg):
    """Main training and evaluation entrypoint.

    This runs the training loop, evaluates once at the end, saves
    reliability plot, metrics and optionally a checkpoint.
    """
    set_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["log_dir"], exist_ok=True)

    train_loader, test_loader = get_loaders(cfg["dataset"], cfg["data_root"],
                                            cfg["batch_size"], cfg["num_workers"])
    # create a ResNet-18 with no pretrained weights and 100 classes
    net = models.resnet18(weights=None, num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr=cfg["optimizer"]["lr"],
                    momentum=cfg["optimizer"]["momentum"],
                    weight_decay=cfg["optimizer"]["weight_decay"])
    if cfg["scheduler"]["name"] == "cosine":
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    else:
        sch = None

    # standard training loop (no validation during training here)
    for epoch in range(1, cfg["epochs"]+1):
        net.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(net(x), y)
            loss.backward(); opt.step()
        if sch: sch.step()

    # one final evaluation and save artifacts
    acc, ece, nll, bin_stats = evaluate(net, test_loader, device)
    plot_reliability(bin_stats, os.path.join(cfg["log_dir"], "reliability.png"))
    with open(os.path.join(cfg["log_dir"], "metrics.txt"), "w") as f:
        f.write(f"acc={acc:.4f}, ece={ece:.4f}, nll={nll:.4f}\n")
    if cfg.get("save_ckpt", True):
        # save model state_dict (binary file)
        torch.save(net.state_dict(), os.path.join(cfg["log_dir"], "ckpt.pt"))
    print(f"[TEST] acc={acc:.4f} ece={ece:.4f} nll={nll:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="bayes-moe/configs/cifar100_resnet18.yaml")
    args = ap.parse_args()
    with open(args.cfg, "r") as f: cfg = yaml.safe_load(f)
    main(cfg)
