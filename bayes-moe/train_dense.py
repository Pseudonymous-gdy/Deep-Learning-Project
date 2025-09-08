"""train_dense: training driver for CIFAR / placeholder OOD experiments.

Contains the CLI entrypoint, data-loader wiring (uses bayes-moe/datasets when
available), the training loop, evaluation (accuracy / ECE / NLL) and
artifact exporting (metrics CSV, metrics.txt, reliability plot, checkpoint).

Run example:
    python bayes-moe/train_dense.py --cfg bayes-moe/configs/cifar100_resnet18.yaml
"""

import os, argparse, yaml, random, json, csv, time, warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# build datasets designed in datasets/*
try:
    # expect: bayes-moe/datasets/__init__.py defines build_dataset/build_loader
    from datasets import build_dataset, build_loader
    HAS_DATASETS_MODULE = True
except Exception as _e:
    HAS_DATASETS_MODULE = False
    warnings.warn(
        "[train_dense] datasets module not found; "
        "falling back to built-in CIFAR-100 only loader. "
        "Create bayes-moe/datasets/* for CIFAR-10/SVHN/OOD support."
    )

# Provide names for static type checkers when the optional package is not
# importable at analysis time. This has no runtime effect.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # These are only used by type-checkers and linters to avoid false positives
    def build_dataset(name: str, root: str, split: str, download: bool = True): ...
    def build_loader(ds, batch_size: int, shuffle: bool, num_workers: int = 4): ...

# --- metrics & plotting block import ---
from metrics.ece import expected_calibration_error
from metrics.nll import negative_log_likelihood
from utils.reliability import plot_reliability


# -----------------------------
# Utilities
# -----------------------------
def set_seed(s: int):
    """Set random seeds for reproducibility (CPU and CUDA). This is for convenience."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    '''Ensure directory exists; create if not.'''
    os.makedirs(p, exist_ok=True)


def append_metrics_row(csv_path: str, fieldnames: list, row: dict):
    """Append one row to CSV; create with header if not exists."""
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


# -----------------------------
# Built-in fallback loaders
# -----------------------------
def _builtin_cifar_loaders(name, root, bs, nw):
    """
    This is a fallback loader for CIFAR-100 only. It is carried out when the datasets are unavailable.
    Fallback: only CIFAR-100. Use external datasets module for CIFAR-10/SVHN/OOD.
    Returns (train_loader, test_loader).
    """
    if name.lower() != "cifar100": # require only cifar100
        raise ValueError(
            f"[train_dense] dataset={name} requires bayes-moe/datasets/ module. "
            f"Fallback supports only cifar100."
        )
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761) # normalization for CIFAR-100
    # transformations to strengthen generalization of cifar100
    T_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # randomly crop a 32x32 patch with 4 pixels padding
        transforms.RandomHorizontalFlip(), # randomly flip the image horizontally
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    T_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train = datasets.CIFAR100(root=root, train=True, download=True, transform=T_train) # load train set
    test = datasets.CIFAR100(root=root, train=False, download=True, transform=T_test) # load test set
    train_loader = DataLoader(train, bs, shuffle=True, num_workers=nw, pin_memory=True,
                              persistent_workers=(nw > 0))
    test_loader = DataLoader(test, bs, shuffle=False, num_workers=nw, pin_memory=True,
                             persistent_workers=(nw > 0))
    return train_loader, test_loader


def get_id_loaders(name, root, bs, nw):
    """Return ID (train_loader, test_loader) loaders for cifar100/cifar10."""
    if HAS_DATASETS_MODULE:
        # When the optional `datasets` package is present these helpers are
        # provided by bayes-moe/datasets/__init__.py. Static analyzers may mark
        # `build_dataset`/`build_loader` as possibly-unbound, but at runtime the
        # import above guarantees they exist when HAS_DATASETS_MODULE is True.
        train_ds = build_dataset(name, root, split="train", download=True)
        test_ds = build_dataset(name, root, split="test", download=True)
        return (
            build_loader(train_ds, batch_size=bs, shuffle=True, num_workers=nw),
            build_loader(test_ds, batch_size=bs, shuffle=False, num_workers=nw),
        )
    # fallback: only CIFAR-100
    return _builtin_cifar_loaders(name, root, bs, nw)


def get_ood_loader(ood_name, root, bs, nw, split="test"):
    """Return OOD loader (placeholder). Requires datasets module.
    Adoptable if OOD dataset is supported and required.
    """
    if ood_name is None:
        return None
    if not HAS_DATASETS_MODULE:
        warnings.warn(
            "[train_dense] OOD requested but datasets module not found; "
            "skip OOD loader."
        )
        return None
    # see note above about optional import; safe because guarded by HAS_DATASETS_MODULE
    ood_ds = build_dataset(ood_name, root, split=split, download=True)
    return build_loader(ood_ds, batch_size=bs, shuffle=False, num_workers=nw)

from contextlib import nullcontext
from typing import Callable, ContextManager, Any

# Type alias: a factory that returns a context manager when called.
AutocastCtx = Callable[[], ContextManager[Any]]

def _get_autocast_ctx(device) -> AutocastCtx:
    """
    Return a callable that produces an autocast context manager appropriate
    for the runtime environment and PyTorch version.

    Behavior:
    - If device is CPU (or None), return a no-op context (nullcontext).
    - If device is CUDA:
        1) Try the unified new API (torch.amp.autocast(device_type="cuda"))
        2) Then try torch.autocast(device_type="cuda")
        3) Fall back to legacy torch.cuda.amp.autocast
        4) If none available, return nullcontext for safety
    This makes the code backward- and forward-compatible across PyTorch releases.
    """
    dev = str(device).lower() if device is not None else "cpu"
    # If not CUDA, use a no-op context (do not autocast on CPU)
    if not dev.startswith("cuda"):
        return lambda: nullcontext()

    # 1) Prefer the new unified AMP API (torch.amp.autocast)
    try:
        return lambda: torch.amp.autocast(device_type="cuda")
    except Exception:
        pass

    # 2) Some versions expose torch.autocast - try it
    try:
        return lambda: torch.autocast(device_type="cuda")
    except Exception:
        pass

    # 3) Fall back to legacy CUDA-specific autocast
    try:
        from torch.cuda.amp import autocast as _legacy_autocast
        return _legacy_autocast
    except Exception:
        # 4) Safe fallback: no-op
        return lambda: nullcontext()

# -----------------------------
# Train & Eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, device, criterion, use_amp=False, scaler=None):
    """
    Run one training epoch over `loader` and return the average loss per sample.

    Arguments:
    - model: nn.Module to train
    - loader: iterable/DataLoader yielding (inputs, targets)
    - optimizer: optimizer instance (e.g., SGD/Adam)
    - device: device string or torch.device (e.g., 'cuda' or 'cpu')
    - criterion: loss function returning a scalar loss
    - use_amp: whether to attempt mixed precision (AMP)
    - scaler: GradScaler instance when using AMP on CUDA, otherwise None

    Returns:
    - average loss (float) computed as sum(loss * batch_size) / total_samples
    """
    model.train() # train mode
    total_loss, total, step = 0.0, 0, 0 # initialization
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True) # move to device, non_blocking for faster transfer
        optimizer.zero_grad(set_to_none=True) # clear the gradients as None rather than 0 for memory efficiency

        # Use AMP only when requested, scaler provided, and running on CUDA.
        if use_amp and scaler is not None and str(device).lower().startswith("cuda"): # automatic mixed precision
            ac = _get_autocast_ctx(device)
            with ac():
                logits = model(x)
                loss = criterion(logits, y) # compute loss with logits and targets
            scaler.scale(loss).backward() # scale loss and backward
            scaler.step(optimizer) # step optimizer
            scaler.update() # update scaler for next iteration
        else: # non-amp block
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        bs = y.size(0) # batch size
        total_loss += float(loss.detach()) * bs # accumulate total loss
        total += bs # accumulate total samples
        step += 1 # increment step
    return total_loss / max(total, 1)


@torch.no_grad() # require no gradient computation for evaluation
def evaluate(model, loader, device, n_bins=15):
    """Compute accuracy, ECE, NLL and collect bin stats for reliability diagram."""
    model.eval()
    total, correct = 0, 0
    all_logits, all_targets = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
        pred = logits.argmax(1) # get predicted class
        total += y.size(0) # count total samples
        correct += (pred == y).sum().item() # count correct predictions
    logits = torch.cat(all_logits) # concatenate all logits
    targets = torch.cat(all_targets) # concatenate all targets
    acc = correct / max(total, 1) # compute accuracy, avoid division by zero
    ece, bin_stats = expected_calibration_error(logits, targets, n_bins=n_bins)
    nll = negative_log_likelihood(logits, targets)
    return acc, ece, nll, bin_stats


# -----------------------------
# Main
# -----------------------------
def main(cfg, cli_override):
    """
    Runs training on ID dataset (cifar100/cifar10) and evaluates ID metrics (Acc/ECE/NLL).
    Optionally builds an OOD loader (e.g., SVHN) as a **placeholder** for next week's OOD eval.
    Artifacts:
      - reliability.png
      - metrics.txt
      - metrics.csv (append)
      - ckpt.pt (optional)
    """
    # ---- seed & device ----
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- I/O & dirs ----
    log_dir = cfg.get("log_dir", "./logs/w1")
    ensure_dir(log_dir)

    # ---- dataset choices ----
    dataset = cli_override.dataset or cfg.get("dataset", "cifar100")
    data_root = cfg.get("data_root", "./data")
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 4))
    n_bins = int(cfg.get("ece_bins", 15))

    # OOD placeholder (does not affect training)
    ood_name = cli_override.ood_dataset
    ood_split = cli_override.ood_split

    # ---- model & opt ----
    num_classes = 100 if dataset.lower() == "cifar100" else 10
    model = models.resnet18(weights=None, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    opt_cfg = cfg.get("optimizer", {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4})
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 0.1)),
        momentum=float(opt_cfg.get("momentum", 0.9)),
        weight_decay=float(opt_cfg.get("weight_decay", 5e-4)),
        nesterov=bool(opt_cfg.get("nesterov", False)),
    )

    sch_cfg = cfg.get("scheduler", {"name": "cosine"})
    if sch_cfg.get("name", "cosine").lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.get("epochs", 120)))
    else:
        scheduler = None

    use_amp = bool(cfg.get("amp", False))
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device == "cuda") else None

    # ---- loaders ----
    train_loader, test_loader = get_id_loaders(dataset, data_root, batch_size, num_workers)
    ood_loader = get_ood_loader(ood_name, data_root, batch_size, num_workers, split=ood_split)

    if ood_loader is not None:
        # Sanity check only; full OOD metrics in Week-2/3.
        xb, yb = next(iter(ood_loader))
        print(f"[OOD] Built loader: {ood_name}:{ood_split} batch={tuple(xb.shape)}")

    # ---- training ----
    epochs = int(cfg.get("epochs", 120))
    print(f"[INFO] Start training: dataset={dataset} epochs={epochs} device={device} amp={use_amp}")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, use_amp, scaler)
        if scheduler is not None:
            scheduler.step()
        if epoch % max(epochs // 10, 1) == 0 or epoch in (1, epochs):
            print(f"[E{epoch:03d}] train_loss={tr_loss:.4f} lr={optimizer.param_groups[0]['lr']:.5f}")
    t_train = time.time() - t0

    # ---- final evaluation ----
    acc, ece, nll, bin_stats = evaluate(model, test_loader, device, n_bins=n_bins)
    plot_reliability(bin_stats, os.path.join(log_dir, "reliability.png"))

    # ---- write metrics ----
    with open(os.path.join(log_dir, "metrics.txt"), "w") as f:
        f.write(f"acc={acc:.4f}, ece={ece:.4f}, nll={nll:.4f}\n")
        f.write(f"train_time_sec={t_train:.2f}\n")

    # also append to a CSV with metadata for future ablations
    csv_path = os.path.join(log_dir, "results.csv")
    fieldnames = [
        "timestamp", "commit_sha",
        "dataset", "ood_dataset", "ood_split",
        "seed", "epochs", "batch_size", "lr", "momentum", "weight_decay",
        "acc", "ece", "nll", "train_time_sec", "ckpt_path", "fig_path"
    ]
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit_sha": os.environ.get("COMMIT_SHA", ""),
        "dataset": dataset,
        "ood_dataset": (ood_name or ""),
        "ood_split": (ood_split or ""),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": float(opt_cfg.get("lr", 0.1)),
        "momentum": float(opt_cfg.get("momentum", 0.9)),
        "weight_decay": float(opt_cfg.get("weight_decay", 5e-4)),
        "acc": round(float(acc), 6),
        "ece": round(float(ece), 6),
        "nll": round(float(nll), 6),
        "train_time_sec": round(float(t_train), 2),
        "ckpt_path": os.path.join(log_dir, "ckpt.pt") if bool(cfg.get("save_ckpt", True)) else "",
        "fig_path": os.path.join(log_dir, "reliability.png"),
    }
    append_metrics_row(csv_path, fieldnames, row)

    # ---- save checkpoint ----
    if bool(cfg.get("save_ckpt", True)):
        torch.save(model.state_dict(), os.path.join(log_dir, "ckpt.pt"))

    print(f"[TEST] acc={acc:.4f} ece={ece:.4f} nll={nll:.4f}")
    print(f"[INFO] artifacts at: {log_dir}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="bayes-moe/configs/cifar100_resnet18.yaml")
    # dataset overrides (keep backward compatible with YAML)
    ap.add_argument("--dataset", type=str, default=None, choices=[None, "cifar100", "cifar10"])
    # OOD placeholder (requires datasets module)
    ap.add_argument("--ood-dataset", type=str, default=None, choices=[None, "svhn"])
    ap.add_argument("--ood-split", type=str, default="test", choices=["train", "test", "extra"])
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Create log_dir if given in cfg; otherwise default inside main()
    if "log_dir" in cfg:
        os.makedirs(cfg["log_dir"], exist_ok=True)

    main(cfg, cli_override=args)
