import torch
import torch.nn.functional as F

@torch.no_grad()
def negative_log_likelihood(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return F.cross_entropy(logits, targets, reduction="mean").item()
