import os
import argparse
import yaml
import time
# Allow continuing when multiple OpenMP runtimes are present (common on some
# Windows/Conda setups where MKL/OpenMP get linked by multiple libs). This
# is an unsafe workaround but prevents the process from aborting with
# "OMP: Error #15"; it is preferable to fix the environment, but set here
# to make the script robust for local runs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image


# Small helper: convert a PIL Image to a torch.FloatTensor without using NumPy.
# This avoids calling into NumPy's C-API from within torch/torchvision which
# can fail on some Windows/Conda DLL setups. It's slightly less efficient
# than the NumPy path but reliable for small datasets like CIFAR.
def pil_to_tensor_no_numpy(pic: Image.Image) -> torch.Tensor:
    '''
    Convert a PIL Image to a torch.FloatTensor without using NumPy.
    This avoids calling into NumPy's C-API from within torch/torchvision which can fail on some Windows/Conda DLL setups.
    It's slightly less efficient than the NumPy path but reliable for small datasets like CIFAR.

    Args:
        pic (PIL.Image): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    '''
    if isinstance(pic, torch.Tensor):
        return pic.float()
    if not isinstance(pic, Image.Image):
        raise TypeError(f"pil_to_tensor_no_numpy expects PIL Image, got {type(pic)}")
    # pic.getdata() returns a sequence of tuples (R,G,B). Convert to flat tensor
    data = list(pic.getdata())
    arr = torch.tensor(data, dtype=torch.uint8)
    arr = arr.view(pic.size[1], pic.size[0], len(pic.getbands()))
    arr = arr.permute(2, 0, 1).contiguous().float().div(255.0)
    return arr

# ====== Repository utilities: prefer using existing metrics and reliability plot implementations ======
try:
    from metrics.ece import expected_calibration_error as ece_fn
except Exception:
    import torch.nn.functional as F
    @torch.no_grad()
    def ece_fn(logits, targets, n_bins=15):
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        correct = preds.eq(targets).float()
        bins = torch.linspace(0, 1, steps=n_bins+1, device=logits.device)
        ece = torch.zeros([], device=logits.device)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i+1]
            mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
            if mask.any():
                ece += (mask.float().mean()) * (confs[mask].mean() - correct[mask].mean()).abs()
        return float(ece)

try:
    from metrics.nll import negative_log_likelihood as nll_fn
except Exception:
    import torch.nn.functional as F
    @torch.no_grad()
    def nll_fn(logits, targets):
        logp = F.log_softmax(logits, dim=1)
        return float(-logp[torch.arange(logits.size(0), device=logits.device), targets].mean())

try:
    from utils.reliability import plot_reliability  # use repository-provided reliability plotting if available
except Exception:
    # Fallback simplified reliability diagram implementation (used only if project util isn't available)
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    @torch.no_grad()
    def plot_reliability(logits, targets, n_bins, save_path):
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        correct = preds.eq(targets).float()
        bins = torch.linspace(0, 1, steps=n_bins+1, device=logits.device)
        xs, accs, confm = [], [], []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i+1]
            mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
            if mask.any():
                xs.append(((lo+hi)/2).item())
                accs.append(correct[mask].mean().item())
                confm.append(confs[mask].mean().item())
        plt.figure()
        plt.plot([0,1],[0,1],'--', linewidth=1)
        plt.plot(confm, accs, marker='o')
        plt.xlabel('Confidence'); plt.ylabel('Accuracy'); plt.title('Reliability Diagram')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160); plt.close()

else:
    # If the repository-provided plot_reliability exists, adapt it if its
    # signature differs. Some older variants accept (bin_stats, save_path)
    # instead of (logits, targets, n_bins, save_path). Wrap when needed.
    import inspect
    import torch.nn.functional as F
    # save original to avoid recursive calls from the wrapper
    _orig_plot_reliability = plot_reliability
    try:
        sig = inspect.signature(_orig_plot_reliability)
        params = len(sig.parameters)
    except Exception:
        params = None

    if params == 2:
        # assume signature (bin_stats, save_path) -> adapt
        @torch.no_grad()
        def _wrapped_plot_reliability(logits, targets, n_bins, save_path):
            '''
            Adapted wrapper for reliability plotting.
            Converts logits/targets to bin_stats_dict and calls the original function.
            '''
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            correct = preds.eq(targets).float()
            bins = torch.linspace(0, 1, steps=n_bins+1, device=logits.device)
            accs, confs_list, counts = [], [], []
            for i in range(n_bins):
                lo, hi = bins[i], bins[i+1]
                mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
                if mask.any():
                    confs_list.append(confs[mask].mean().item())
                    accs.append(correct[mask].mean().item())
                    counts.append(int(mask.sum().item()))
                else:
                    confs_list.append(0.0)
                    accs.append(0.0)
                    counts.append(0)
            # convert to the dict-of-lists format expected by the repo helper
            bin_stats_dict = {"acc": accs, "conf": confs_list, "count": counts}
            _orig_plot_reliability(bin_stats_dict, save_path)

        plot_reliability = _wrapped_plot_reliability
    else:
        # keep the imported function as-is (assumed to accept logits/targets/n_bins/save_path)
        pass

# ====== MoE head ======
from models.moe.block import MoEHead
from models.moe.balance_loss import balance_loss


def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_loaders(name: str, root: str, bs: int, nw: int):
    name = name.lower()
    if name not in {"cifar100", "cifar10"}:
        raise ValueError("dataset must be cifar100 or cifar10")
    if name == "cifar100":
        mean, std = [0.5071,0.4867,0.4408], [0.2675,0.2565,0.2761]
        Train = datasets.CIFAR100; Test = datasets.CIFAR100
        num_classes = 100
    else:
        mean, std = [0.4914,0.4822,0.4465], [0.2470,0.2435,0.2616]
        Train = datasets.CIFAR10; Test = datasets.CIFAR10
        num_classes = 10
    # Training data augmentation and normalization transforms
    tfm_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda pic: pil_to_tensor_no_numpy(pic)),
        transforms.Normalize(mean, std),
    ])
    # Test-time transforms (only normalization)
    tfm_te = transforms.Compose([
    transforms.Lambda(lambda pic: pil_to_tensor_no_numpy(pic)),
        transforms.Normalize(mean, std),
    ])
    # On Windows, DataLoader worker processes are started with spawn() and
    # can fail to import NumPy due to DLL loading issues in some environments
    # (common with certain Conda/BLAS setups). To avoid that runtime error,
    # force num_workers=0 when running on Windows unless the user explicitly
    # requests otherwise and is aware of the DLL configuration.
    if os.name == "nt" and nw > 0:
        print("Warning: Detected Windows platform — forcing num_workers=0 to avoid DataLoader worker NumPy DLL errors.")
        nw = 0

    train_set = Train(root=root, train=True, download=True, transform=tfm_tr)
    test_set  = Test(root=root,  train=False, download=True, transform=tfm_te)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, test_loader, num_classes


def build_optimizer_and_scheduler(params, cfg, steps_per_epoch):
    opt_cfg = cfg.get("optimizer", {})
    name = (opt_cfg.get("name") or "sgd").lower()
    lr = float(opt_cfg.get("lr", 0.1))
    wd = float(opt_cfg.get("weight_decay", 5e-4))
    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        opt = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    elif name == "adamw":
        opt = optim.AdamW(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    # Scheduler selection and warmup configuration
    sch_name = (cfg.get("scheduler", {}) or {}).get("name", "cosine").lower()
    epochs = int(cfg.get("epochs", 20))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(0.05 * total_steps))
    if sch_name == "cosine":
        # Cosine decay with linear warmup for the first `warmup_steps` steps.
        def lr_lambda(step):
            if step < warmup_steps:
                # linear warmup from 0 -> 1
                return (step + 1) / warmup_steps
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            # cosine annealing between 1.0 -> 0.0
            return 0.5 * (1.0 + torch.cos(torch.tensor(t * 3.1415926))).item()
        sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        sch = None
    return opt, sch


@torch.no_grad()
def evaluate(backbone, head, loader, device, ece_bins=15):
    '''
    Evaluation function. Returns accuracy, NLL, ECE, and logits/labels for plotting reliability plots.
    '''
    ce = nn.CrossEntropyLoss(reduction="sum")
    backbone.eval(); head.eval()
    total, ce_sum, correct = 0, 0.0, 0
    logits_all, labels_all = [], []
    # iterate through evaluation loader and accumulate logits/labels for metrics
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        feats = backbone(x)
        feats = torch.flatten(feats, 1)
        logits, _ = head(feats)
        logits_all.append(logits)
        labels_all.append(y)
        ce_sum += ce(logits, y).item()
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    logits_all = torch.cat(logits_all, 0)
    labels_all = torch.cat(labels_all, 0)
    # Some project-provided metric implementations return (value, aux)
    # or other non-scalar shapes. Normalize to a float here so callers
    # can safely format and write the metric values.
    def _to_float(v):
        if isinstance(v, (tuple, list)):
            return float(v[0])
        try:
            return float(v)
        except Exception:
            # Fallback: convert tensors to scalar
            if isinstance(v, torch.Tensor):
                return float(v.item())
            raise

    nll = _to_float(nll_fn(logits_all, labels_all))
    ece = _to_float(ece_fn(logits_all, labels_all, n_bins=ece_bins))
    acc = correct / max(1, total)
    return acc, nll, ece, logits_all, labels_all


def main(cfg_path: str):
    # ==== Load configuration ====
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==== Output directory (follow repository convention) ====
    exp_dir = cfg.get("log_dir", "bayes-moe/runs/cifar100-resnet18-moe")
    os.makedirs(exp_dir, exist_ok=True)

    # ==== Data loaders ====
    train_loader, test_loader, num_classes = get_loaders(
        cfg.get("dataset", "cifar100"),
        cfg.get("data_root", "bayes-moe/data"),
        int(cfg.get("batch_size", 128)),
        int(cfg.get("num_workers", 2)),
    )

    # ==== Model: ResNet18 backbone + MoE head ====
    backbone = models.resnet18(weights=None)
    assert cfg.get("model", "resnet18") == "resnet18", "This script only supports resnet18 as the backbone"
    d_model = int(cfg.get("moe", {}).get("d_model", 512))
    # replace final fc with identity -> backbone returns (B, 512) feature vectors
    backbone.fc = nn.Identity()  # type: ignore[assignment]
    backbone = backbone.to(device)

    moe_cfg = cfg.get("moe", {})
    use_moe = bool(moe_cfg.get("use_moe_head", True))
    if not use_moe:
        # Optionally disable MoE and use a simple linear classification head
        head = nn.Linear(d_model, num_classes).to(device)
        # create a small wrapper so the rest of the code can always expect (logits, aux)
        forward_head = lambda feats: (head(feats), {"probs_mean": torch.full((1,), 1.0/num_classes, device=feats.device)})
    else:
        head = MoEHead(
            d_model=d_model, num_classes=num_classes,
            num_experts=int(moe_cfg.get("num_experts", 4)),
            top_k=int(moe_cfg.get("top_k", 1)),
            hidden_factor=int(moe_cfg.get("hidden_factor", 4)),
            temperature=float(moe_cfg.get("temperature", 1.0)),
        ).to(device)
        # MoEHead is callable; use it directly as the forward wrapper
        forward_head = head

    params = list(backbone.parameters()) + list(head.parameters())
    opt, sch = build_optimizer_and_scheduler(params, cfg, len(train_loader))
    ce = nn.CrossEntropyLoss()

    ece_bins = int(cfg.get("metrics", {}).get("ece_bins", 15))
    lb_coef = float(moe_cfg.get("lb_coef", 0.01))
    save_ckpt = bool(cfg.get("save_ckpt", True))

    # ===== Training loop =====
    best_acc = -1.0
    results_csv = os.path.join(exp_dir, "results.csv")
    # write CSV header if missing
    if not os.path.exists(results_csv):
        with open(results_csv, "w", encoding="utf-8") as f:
            f.write("epoch,acc,nll,ece\n")

    global_step = 0
    for epoch in range(1, int(cfg.get("epochs", 20)) + 1):
        backbone.train(); head.train()
        for it, (x, y) in enumerate(train_loader, 1):
            # Move batch to device and compute features
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            feats = backbone(x); feats = torch.flatten(feats, 1) # resnet18 outputs (B, 512, 1, 1); flatten to (B, 512)
            logits, aux = forward_head(feats) # logits: (B, num_classes); aux: dict with gate statistics

            # primary cross-entropy loss
            loss_ce = ce(logits, y)
            # add balance regularizer only when using the MoE head and gate statistics are present
            if use_moe and "probs_mean" in aux:
                lb = balance_loss(aux["probs_mean"], lb_coef)
            else:
                lb = torch.zeros((), device=logits.device)
            loss = loss_ce + lb

            # standard optimizer step
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sch is not None:
                # scheduler.step() is called per-update (this fits LambdaLR usage here)
                sch.step()
            global_step += 1

            # periodic training progress logging
            if it % 50 == 0:
                with torch.no_grad():
                    acc_batch = (logits.argmax(-1) == y).float().mean().item()
                print(f"Epoch {epoch} | it {it}/{len(train_loader)} | loss {loss.item():.4f} | acc {acc_batch:.3f} | lb {lb.item():.4f}")

        # ===== Validation =====
        acc, nll, ece, logits_all, labels_all = evaluate(backbone, head, test_loader, device, ece_bins)
        print(f"[Eval] epoch={epoch} acc={acc:.4f} nll={nll:.4f} ece={ece:.4f}")
        best_acc = max(best_acc, acc)

        # write results to csv
        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{acc:.6f},{nll:.6f},{ece:.6f}\n")

        # save checkpoint
        if save_ckpt:
            torch.save({
                "epoch": epoch,
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "cfg": cfg,
                "best_acc": best_acc
            }, os.path.join(exp_dir, "ckpt.pt"))

    # ===== Post-training: write metrics and produce plots (reliability, expert load) =====
    final_acc, final_nll, final_ece, logits_all, labels_all = evaluate(backbone, head, test_loader, device, ece_bins)
    with open(os.path.join(exp_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"acc: {final_acc:.6f}\n")
        f.write(f"nll: {final_nll:.6f}\n")
        f.write(f"ece({ece_bins}bins): {final_ece:.6f}\n")
        if use_moe:
            f.write(f"num_experts: {moe_cfg.get('num_experts', 4)}  top_k: {moe_cfg.get('top_k', 1)}  lb_coef: {lb_coef}\n")

    # Reliability diagram (calibration plot)
    plot_reliability(logits_all, labels_all, ece_bins, os.path.join(exp_dir, "reliability.png"))

    # Expert load bar plot (uses probs_mean from the first training batch for visualization)
    if use_moe:
        # ensure aux is always defined for the subsequent access (silences some static analyzers)
        aux = {}
        with torch.no_grad():
            for x, _ in train_loader:
                feats = backbone(x.to(device)); feats = torch.flatten(feats, 1)
                _, aux = forward_head(feats)
                break
        probs_mean = aux.get("probs_mean", None)
        if probs_mean is not None:
            import matplotlib.pyplot as plt
            # use .tolist() to avoid calling NumPy (some environments cannot load NumPy in workers)
            pm = probs_mean.detach().cpu().tolist()
            plt.figure()
            plt.bar(range(len(pm)), pm)
            plt.xlabel("expert id"); plt.ylabel("mean prob")
            plt.title("Expert Load (mean soft prob)")
            plt.savefig(os.path.join(exp_dir, "load_hist.png"), dpi=160)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config (e.g., bayes-moe/configs/xxx.yaml)")
    args = parser.parse_args()
    main(args.cfg)
