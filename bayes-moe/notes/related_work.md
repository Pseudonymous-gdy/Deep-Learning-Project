# Related Work

**Table of Contents:**
- [Work Related to Mixture of Experts](https://github.com/Pseudonymous-gdy/Deep-Learning-Project/blob/main/bayes-moe/notes/related_work.md#Mixture-of-Experts)
  - [Shazeer17: Sparsely-Gated MoE](https://github.com/Pseudonymous-gdy/Deep-Learning-Project/blob/main/bayes-moe/notes/related_work.md#shazeer17sparsely-gated-moe)
    - [Gating Computation](https://github.com/Pseudonymous-gdy/Deep-Learning-Project/blob/main/bayes-moe/notes/related_work.md#1-gating-computation)
    - [Conditional Computation](https://github.com/Pseudonymous-gdy/Deep-Learning-Project/blob/main/bayes-moe/notes/related_work.md#2-conditional-computation)
    - [Output & Auxiliary Loss](https://github.com/Pseudonymous-gdy/Deep-Learning-Project/blob/main/bayes-moe/notes/related_work.md#3-output-&-auxiliary-loss)
    - [Flow Chart](https://github.com/Pseudonymous-gdy/Deep-Learning-Project/blob/main/bayes-moe/notes/related_work.md#4-flow-chart)
- [Work Related to Bayesian DL](https://github.com/Pseudonymous-gdy/Deep-Learning-Project/blob/main/bayes-moe/notes/related_work.md#Bayesian-Deep-Learning)

---
## Mixture of Experts
This part records previous works related to MoE architecture.
### Shazeer17：Sparsely-Gated MoE

#### 1. Gating Computation
  - add controllable noise:
    $$H(x) \gets (X \cdot W_g) + \text{StandardNormal}() \cdot \text{Softplus}(x \cdot W_{\text{noise}})$$
  - select only top-k gates
  - softmax application
#### 2. Conditional Computation
  - conduct only on non-zero weighted experts (**to realize sparsity**)
#### 3. Output & Auxiliary Loss
  - take the weighted sum
  - compute Loss of Importance $L_\text{importance}$ and Loss of Load $L_\text{load}$
  - combine them with main loss as objective function for optimization.
#### 4. Flow Chart
```mermaid
flowchart TD
A["Input x"] --> B[Gate Network G]
A --> C[Expert Network E₁]
A --> D[Expert Network E₂]
A --> E[...]
A --> F[Expert Network Eₙ]

B --> G["H(x) = x·W_g + Noise·Softplus(x·W_noise)"]
G --> H{"KeepTopK(H(x), k)"}
H --> I["G(x) = Softmax(KeepTopK(...))"]
I --> J["Sparse Gate Weight G(x)"]

J -- Weight g₁ --> C
J -- Weight g₂ --> D
J -- Weight gₙ --> F

subgraph ConditionComputation[Conditional Computation: Operate on Selected Experts only]
    C -- Output y₁ --> K{"Weighted Sum<br>y = ∑ G(x)ᵢ • Eᵢ(x)"}
    D -- Output y₂ --> K
    F -- Output yₙ --> K
end

K -- Final Output y --> L[Next Layer]

J --> M[Calculate Auxiliary Loss]
M --> N["Total Loss: L_total<br>= L_main + L_importance + L_load"]
```

## Bayesian Deep Learning









