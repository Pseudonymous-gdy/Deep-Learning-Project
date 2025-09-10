
import os, argparse, yaml, time
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pandas as pd

from models.moe.block import MoEHead
from models.moe.balance_loss import balance_loss
from metrics import expected_calibration_error as ece_fn, negative_log_likelihood as nll_fn

def set_seed(s: int):
    import random, numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

def get_num_classes(name: str) -> int:
    return 100 if name.lower() == "cifar100" else 10

def get_loaders(name, root, bs, nw):
    name = name.lower()
    if name not in {"cifar10","cifar100"}:
        raise ValueError("dataset must be cifar10 or cifar100")
    if name == "cifar100":
        mean, std = [0.5071,0.4865,0.4409], [0.2673,0.2564,0.2762]
        Train = datasets.CIFAR100; Test = datasets.CIFAR100
    else:
        mean, std = [0.4914,0.4822,0.4465], [0.2470,0.2435,0.2616]
        Train = datasets.CIFAR10; Test = datasets.CIFAR10
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = Train(root=root, train=True, download=True, transform=tf_train)
    test_set  = Test(root=root,  train=False, download=True, transform=tf_test)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, test_loader

def build_optim_sched(params, cfg, steps_per_epoch):
    # Support nested optimizer dict (your YAML) or flat fields
    opt_cfg = cfg.get("optimizer") or {}
    name = (opt_cfg.get("name") or cfg.get("optimizer_name") or "sgd").lower()
    lr = opt_cfg.get("lr", cfg.get("lr", 0.1))
    momentum = opt_cfg.get("momentum", cfg.get("momentum", 0.9))
    weight_decay = opt_cfg.get("weight_decay", cfg.get("weight_decay", 5e-4))
    if name == "sgd":
        opt = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif name == "adamw":
        opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    # scheduler: support "scheduler.name" or default cosine with warmup 5%
    sch_name = (cfg.get("scheduler", {}) or {}).get("name", "cosine").lower()
    epochs = cfg["epochs"]
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.05 * total_steps)
    if sch_name == "cosine":
        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(t * 3.1415926))).item()
        sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        sch = None
    return opt, sch

def evaluate(backbone, head, loader, device, ece_bins=15):
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct, total = 0, 0
    nll_sum = 0.0
    logits_all, labels_all = [], []
    backbone.eval(); head.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = backbone(imgs)
            feats = torch.flatten(feats, 1)  # (B,512)
            logits, _ = head(feats)
            logits_all.append(logits)
            labels_all.append(labels)
            nll_sum += ce(logits, labels).item()
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)
    logits_all = torch.cat(logits_all, 0)
    labels_all = torch.cat(labels_all, 0)
    ece = ece_fn(logits_all, labels_all, n_bins=ece_bins)
    acc = correct / max(1, total)
    nll = nll_sum / max(1, total)
    return acc, nll, ece

def main(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    dataset = cfg.get("dataset", "cifar100").lower()
    data_root = cfg.get("data_root", "./data")
    batch_size = cfg.get("batch_size", 128)
    num_workers = cfg.get("num_workers", 2)
    epochs = cfg.get("epochs", 20)
    ece_bins = cfg.get("metrics", {}).get("ece_bins", 15)
    moe_cfg = cfg.get("moe") or {}
    d_model = int(moe_cfg.get("d_model", 512))
    num_classes = get_num_classes(dataset)

    # Backbone: torchvision resnet18 (no fc)
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)

    head = MoEHead(
        d_model=d_model, num_classes=num_classes,
        num_experts=int(moe_cfg.get("num_experts", 4)),
        top_k=int(moe_cfg.get("top_k", 1)),
        hidden_factor=int(moe_cfg.get("hidden_factor", 4)),
        temperature=float(moe_cfg.get("temperature", 1.0)),
    ).to(device)

    # Data
    train_loader, test_loader = get_loaders(dataset, data_root, batch_size, num_workers)
    opt, sch = build_optim_sched(list(backbone.parameters()) + list(head.parameters()), cfg, len(train_loader))
    lb_coef = float(moe_cfg.get("lb_coef", 0.01))

    ce = nn.CrossEntropyLoss()
    rows = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        backbone.train(); head.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            feats = backbone(imgs)
            feats = torch.flatten(feats, 1)  # (B, 512)
            logits, aux = head(feats)
            loss_ce = ce(logits, labels)
            lb = balance_loss(aux["probs_mean"], lb_coef)
            loss = loss_ce + lb

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sch is not None: sch.step()
            global_step += 1

            with torch.no_grad():
                acc_batch = (logits.argmax(-1) == labels).float().mean().item()
            pbar.set_postfix(acc=f"{acc_batch:.3f}", ce=f"{loss_ce.item():.3f}", lb=f"{lb.item():.4f}")

        # Eval
        acc, nll, ece = evaluate(backbone, head, test_loader, device, ece_bins)
        rows.append(dict(epoch=epoch, acc=acc, nll=nll, ece=ece))
        print(f"[Eval] epoch={epoch} acc={acc:.4f} nll={nll:.4f} ece={ece:.4f}")

    # Save artifacts
    exp = cfg.get("exp_name", f"{dataset}-resnet18-moe")
    torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(), "cfg": cfg},
               f"checkpoints/last.pt")
    pd.DataFrame(rows).to_csv(f"results/metrics_{exp}.csv", index=False)

    # Plots
    try:
        import matplotlib.pyplot as plt
        df = pd.DataFrame(rows)
        plt.figure()
        df[["acc","nll","ece"]].plot()
        plt.title("Metrics")
        plt.xlabel("epoch")
        plt.savefig(f"plots/curves_{exp}.png")
        plt.close()

        # Expert load (mean probs)
        with torch.no_grad():
            for imgs, _ in train_loader:
                feats = backbone(imgs.to(device)); feats = torch.flatten(feats, 1)
                _, aux = head(feats)
                break
        pm = aux["probs_mean"].detach().cpu().numpy()
        plt.figure()
        plt.bar(range(len(pm)), pm)
        plt.title("Expert Load (mean prob)")
        plt.xlabel("expert id"); plt.ylabel("mean prob")
        plt.savefig(f"plots/load_hist_{exp}.png")
        plt.close()
    except Exception as e:
        print("Plotting skipped:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
