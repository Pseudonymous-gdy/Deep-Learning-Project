import torch
import torch.nn as nn
from .gate import TopKGate



class ExpertMLP(nn.Module):
    """
    Small MLP used as an expert. Maps a d_model-dim feature vector to class logits.
    The architecture is a simple 2-layer MLP with a GELU activation.
    """
    def __init__(self, d_model: int, num_classes: int, hidden_factor: int = 4):
        super().__init__()
        hidden = d_model * hidden_factor
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class MoEHead(nn.Module):
    """
    Mixture-of-Experts classification head.

    Designed to be appended to a backbone (e.g. ResNet18) that outputs a
    d_model-dimensional feature per example. The head routes each example to
    up to `top_k` experts using a `TopKGate`, evaluates the selected experts,
    and forms the final logits as a weighted combination of expert outputs.

    Returns:
      - logits: Tensor (B, C)
      - aux: dict containing gate statistics (e.g. probs_mean and entropy)
    """
    def __init__(self, d_model: int, num_classes: int, num_experts: int = 4,
                 top_k: int = 1, hidden_factor: int = 4, temperature: float = 1.0):
        super().__init__()
        # gating network that selects top-k experts per example
        self.gate = TopKGate(d_model, num_experts, top_k, temperature)
        # collection of expert MLPs
        self.experts = nn.ModuleList([ExpertMLP(d_model, num_classes, hidden_factor)
                                      for _ in range(num_experts)])
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        """
        x: input features with shape (B, D)

        Process:
        1. Use the gate to obtain top-k expert indices and combine weights.
        2. For each expert, evaluate it on the subset of examples routed to it.
        3. Accumulate weighted expert logits into the final output tensor.

        Returns:
          - logits: (B, C)
          - aux: dict from the gate with monitoring statistics
        """
        topk_idx, combine_w, aux = self.gate(x)   # topk_idx: (B, k), combine_w: (B, k)
        B, k = topk_idx.shape
        device = x.device
        num_classes = self.experts[0].net[-1].out_features
        y = torch.zeros(B, num_classes, device=device)

        # Aggregate outputs from each expert. For each expert e we:
        # - find which examples selected e in their top-k
        # - run the expert on that subset and weight its output by the
        #   corresponding combine weights
        for e in range(self.num_experts):
            # mask of shape (B,) indicating examples that selected expert e
            mask = (topk_idx == e).any(dim=-1)
            if not mask.any():
                continue
            x_e = x[mask]                         # (Be, D)
            y_e = self.experts[e](x_e)           # (Be, C)
            w_sel = combine_w[mask]               # (Be, k)
            idx_sel = topk_idx[mask]              # (Be, k)
            # pick the combine weight corresponding to expert e for each selected example
            w_e = (w_sel * (idx_sel == e).float()).sum(dim=-1)  # (Be,)
            # accumulate weighted expert logits into the output
            y[mask] += y_e * w_e.unsqueeze(-1)

        return y, aux
