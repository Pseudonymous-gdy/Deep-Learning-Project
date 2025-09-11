import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKGate(nn.Module):
    """
    Top-k gating module used to route input feature vectors to a subset of experts.

    Given input features of shape (B, D), this layer projects them to logits for
    E experts and selects the top-k experts per example. It returns three items:
        - topk_idx: LongTensor of shape (B, k) containing selected expert indices
        - combine_w: FloatTensor of shape (B, k) containing normalized weights for
                                    the selected experts (softmax over the selected logits)
        - aux: dictionary with monitoring statistics:
                    * "probs_mean": mean softmax probability across the batch for each expert (E,)
                    * "entropy": average entropy of the expert selection distribution (scalar)

    Notes:
    - The module keeps a linear projection (no bias) from feature space to
        expert logits. We compute softmaxed probabilities for monitoring only
        (not used directly for routing decisions). Routing uses the raw logits
        to select the top-k experts and a local softmax over the selected
        logits to form combine weights.
    - Temperature can be applied when computing the monitoring softmax to
        control the sharpness of the probability statistics.
    """

    def __init__(self, d_model: int, num_experts: int, k: int = 1, temperature: float = 1.0):
        super().__init__()
        assert 1 <= k <= num_experts
        self.proj = nn.Linear(d_model, num_experts, bias=False)
        self.num_experts = num_experts
        self.k = k
        self.temperature = temperature

    def forward(self, x: torch.Tensor):
        # Project input to expert logits: shape (B, E)
        logits = self.proj(x)

        # Compute softmax probabilities across experts for monitoring purposes.
        # These probabilities are not directly used to perform routing; routing
        # uses the raw logits (top-k selection). Temperature can be applied
        # to the monitoring softmax to control sharpness.
        probs = F.softmax(logits / max(self.temperature, 1e-6), dim=-1)  # (B, E)

        # Select top-k experts per example using the raw logits. topk_val are
        # the logits corresponding to the chosen experts and topk_idx are their
        # integer indices. Shapes: both (B, k).
        topk_val, topk_idx = torch.topk(logits, k=self.k, dim=-1)       # (B, k)

        # Convert the selected logits into normalized combine weights. This
        # is a small softmax computed only over the k selected logits for each
        # example. The resulting combine_w is used to weight expert outputs.
        combine_w = F.softmax(topk_val, dim=-1)                          # (B, k)

        # Prepare auxiliary monitoring values to help with load-balancing and
        # diagnostics: per-expert mean probability across the batch, and the
        # average entropy of the per-example expert distribution.
        aux = {
                "probs_mean": probs.mean(dim=0),  # (E,) mean soft probability per expert
                "entropy": -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1).mean()
        }

        return topk_idx, combine_w, aux
