import torch


def balance_loss(probs_mean: torch.Tensor, lb_coef: float) -> torch.Tensor:
    """
    Regularization to encourage balanced expert usage.

    This function computes a simple mean-squared-error between the per-expert
    mean soft-probabilities (computed across the batch) and the uniform
    distribution. Multiplying by E stabilizes the scale so the loss magnitude
    does not vanish when increasing the number of experts.

    Args:
        - probs_mean: Tensor of shape (E,) containing the mean softmax probability
                                    assigned to each expert across the current batch.
        - lb_coef: scalar coefficient controlling the strength of the balancing
                                regularizer. If <= 0, the function returns a zero tensor.

    Returns:
        - scalar tensor representing the balance regularization term.
    """
    if lb_coef <= 0:
            return torch.zeros((), device=probs_mean.device)
    E = probs_mean.numel()
    target = torch.full_like(probs_mean, 1.0 / E)
    # MSE to the uniform distribution, scaled by E for numerical stability.
    return lb_coef * ((probs_mean - target) ** 2).sum() * E
