
import torch
import torch.nn as nn
from .gate import TopKGate

class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, num_classes: int, hidden_factor: int = 4):
        super().__init__()
        hidden = d_model * hidden_factor
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        return self.net(x)

class MoEHead(nn.Module):
    """Top-k MoE classification head for pooled features (B, D)."""
    def __init__(self, d_model: int, num_classes: int, num_experts: int = 4,
                 top_k: int = 1, hidden_factor: int = 4, temperature: float = 1.0):
        super().__init__()
        self.gate = TopKGate(d_model, num_experts, top_k, temperature)
        self.experts = nn.ModuleList([ExpertMLP(d_model, num_classes, hidden_factor) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        # x: (B, D)
        topk_idx, combine_w, aux = self.gate(x)  # (B,k),(B,k)
        B, k = topk_idx.shape
        device = x.device
        num_classes = self.experts[0].net[-1].out_features
        y = torch.zeros(B, num_classes, device=device)

        # route & aggregate
        for e in range(self.num_experts):
            mask = (topk_idx == e).any(dim=-1)  # (B,)
            if not mask.any():
                continue
            x_e = x[mask]
            y_e = self.experts[e](x_e)  # (Be, C)
            w_sel = combine_w[mask]     # (Be, k)
            idx_sel = topk_idx[mask]    # (Be, k)
            w_e = (w_sel * (idx_sel == e).float()).sum(dim=-1)  # (Be,)
            y[mask] += y_e * w_e.unsqueeze(-1)
        return y, aux
