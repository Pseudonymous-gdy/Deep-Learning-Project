
import torch

def balance_loss(probs_mean: torch.Tensor, lb_coef: float) -> torch.Tensor:
    """Encourage uniform expert usage via mean soft probabilities."""
    if lb_coef <= 0:
        return torch.zeros((), device=probs_mean.device)
    E = probs_mean.numel()
    target = torch.full_like(probs_mean, 1.0 / E)
    return lb_coef * ((probs_mean - target) ** 2).sum() * E
