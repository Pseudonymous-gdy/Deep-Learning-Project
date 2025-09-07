"""Calibration utilities.

Primary export: `expected_calibration_error` which computes ECE and
per-bin statistics used for reliability diagrams.
"""

import torch


@torch.no_grad()
def expected_calibration_error(logits: torch.Tensor,
                               targets: torch.Tensor,
                               n_bins: int = 15):
    """Compute expected calibration error (ECE) and per-bin stats.

    Parameters
    - logits: [N, C] raw model outputs (before softmax)
    - targets: [N] integer labels
    - n_bins: number of confidence bins

    Returns
    - ece (float): expected calibration error
    - bin_stats (dict): {'acc': [...], 'conf': [...], 'count': [...]} per bin
    """
    # Convert logits -> probabilities and get confidence/predictions
    probs = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    targets = targets.to(preds.device)

    # Create bin boundaries [0,1] with n_bins equal-width bins
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1, device=confs.device)
    ece = torch.tensor(0., device=confs.device)
    bin_acc, bin_conf, bin_count = [], [], []

    N = confs.numel()
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        # include left edge on first bin so confidences==0 fall into bin 0
        in_bin = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        count = in_bin.sum().item()
        if count > 0:
            acc = (preds[in_bin] == targets[in_bin]).float().mean()
            conf = confs[in_bin].mean()
            gap = (conf - acc).abs()
            # weighted contribution to ECE
            ece += gap * (count / N)
            bin_acc.append(acc.item()); bin_conf.append(conf.item()); bin_count.append(count)
        else:
            # keep placeholders for empty bins to keep lengths consistent
            bin_acc.append(0.0); bin_conf.append(0.0); bin_count.append(0)
    return ece.item(), {"acc": bin_acc, "conf": bin_conf, "count": bin_count}
