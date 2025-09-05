import torch
import torch.nn.functional as F


@torch.no_grad()
def negative_log_likelihood(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean negative log-likelihood (cross-entropy).

    Returns a Python float for convenience.
    """
    return F.cross_entropy(logits, targets, reduction="mean").item()
