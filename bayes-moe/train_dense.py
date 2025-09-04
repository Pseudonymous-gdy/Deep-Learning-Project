import torch

@torch.no_grad()
def expected_calibration_error(logits: torch.Tensor,
                               targets: torch.Tensor,
                               n_bins: int = 15):
    """
    logits: [N, C], raw scores
    targets: [N], long
    return: ece (float), bin_stats dict
    """
    probs = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    targets = targets.to(preds.device)

    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1, device=confs.device)
    ece = torch.tensor(0., device=confs.device)
    bin_acc, bin_conf, bin_count = [], [], []

    N = confs.numel()
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        count = in_bin.sum().item()
        if count > 0:
            acc = (preds[in_bin] == targets[in_bin]).float().mean()
            conf = confs[in_bin].mean()
            gap = (conf - acc).abs()
            ece += gap * (count / N)
            bin_acc.append(acc.item()); bin_conf.append(conf.item()); bin_count.append(count)
        else:
            bin_acc.append(0.0); bin_conf.append(0.0); bin_count.append(0)
    return ece.item(), {"acc": bin_acc, "conf": bin_conf, "count": bin_count}

import os, argparse, yaml, random, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from metrics.ece import expected_calibration_error
from metrics.nll import negative_log_likelihood
from utils.reliability import plot_reliability

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_loaders(name, root, bs, nw):
    if name.lower() == "cifar100":
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
    net.eval(); total, correct = 0, 0
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = net(x)
            all_logits.append(logits.cpu()); all_targets.append(y.cpu())
            pred = logits.argmax(1); total += y.size(0); correct += (pred==y).sum().item()
    logits = torch.cat(all_logits); targets = torch.cat(all_targets)
    acc = correct/total
    ece, bin_stats = expected_calibration_error(logits, targets, n_bins=15)
    nll = negative_log_likelihood(logits, targets)
    return acc, ece, nll, bin_stats

def main(cfg):
    set_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["log_dir"], exist_ok=True)

    train_loader, test_loader = get_loaders(cfg["dataset"], cfg["data_root"],
                                            cfg["batch_size"], cfg["num_workers"])
    net = models.resnet18(weights=None, num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr=cfg["optimizer"]["lr"],
                    momentum=cfg["optimizer"]["momentum"],
                    weight_decay=cfg["optimizer"]["weight_decay"])
    if cfg["scheduler"]["name"] == "cosine":
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    else:
        sch = None

    for epoch in range(1, cfg["epochs"]+1):
        net.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(net(x), y)
            loss.backward(); opt.step()
        if sch: sch.step()

    acc, ece, nll, bin_stats = evaluate(net, test_loader, device)
    plot_reliability(bin_stats, os.path.join(cfg["log_dir"], "reliability.png"))
    with open(os.path.join(cfg["log_dir"], "metrics.txt"), "w") as f:
        f.write(f"acc={acc:.4f}, ece={ece:.4f}, nll={nll:.4f}\n")
    if cfg.get("save_ckpt", True):
        torch.save(net.state_dict(), os.path.join(cfg["log_dir"], "ckpt.pt"))
    print(f"[TEST] acc={acc:.4f} ece={ece:.4f} nll={nll:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="bayes-moe/configs/cifar100_resnet18.yaml")
    args = ap.parse_args()
    with open(args.cfg, "r") as f: cfg = yaml.safe_load(f)
    main(cfg)
