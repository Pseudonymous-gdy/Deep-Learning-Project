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
