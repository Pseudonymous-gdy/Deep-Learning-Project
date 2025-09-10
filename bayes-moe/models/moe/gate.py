
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKGate(nn.Module):
    """
    Given (B, D) features, project to num_experts, pick top-k experts.
    Returns:
      - topk_idx: (B, k) expert indices
      - combine_w: (B, k) normalized weights across the k
      - aux: dict with "probs_mean" and "entropy" for balancing/monitoring
    """
    def __init__(self, d_model: int, num_experts: int, k: int = 1, temperature: float = 1.0):
        super().__init__()
        assert 1 <= k <= num_experts
        self.w_g = nn.Linear(d_model, num_experts, bias=False)
        self.num_experts = num_experts
        self.k = k
        self.temperature = temperature

    def forward(self, x: torch.Tensor):
        logits = self.w_g(x)  # (B, E)
        probs = F.softmax(logits / max(self.temperature, 1e-6), dim=-1)  # (B, E)
        topk_val, topk_idx = torch.topk(logits, k=self.k, dim=-1)  # (B, k)
        combine_w = F.softmax(topk_val, dim=-1)  # (B, k)
        aux = {
            "probs_mean": probs.mean(dim=0),  # (E,)
            "entropy": -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1).mean(),
        }
        return topk_idx, combine_w, aux
