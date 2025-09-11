# README

## Current Training/Evaluation

### Quick Start
Directly run the script `train_dense.py`, e.g.,
```bash
> & _/python.exe _/Deep-Learning-Project/bayes-moe/train_dense.py
```

### Model Training/Evaluation Information

---
| Tag | Description | Model | Dataset | Training | Evaluation |
| --- | --- | --- | --- | --- | --- |
| Week 1 | Running a Demo | Resnet18 from torchvision | CIFAR-10 | CrossEntropyLoss + SGD | Accuracy + ECE + NLL|
---

### Architecture

```plaintext
bayes-moe/
|- configs/
|   |- cifar100-resnet18.yaml // config file for training and evaluation
|- data/
|   |- cifar-100-python/
|   |   |- train
|   |   |- test
|   |   |- meta
|   |   |- file.txt~ //empty
|- utils/
|   |- reliability.py // evaluation visualization
|- runs/
|   |- cifar100-resnet18/
|   |   |- ckpt.pt // model state dictionary
|- metrics/ // evaluation metrics
|   |- ece.py
|   |- nll.py
|- train_dense.py // load dataset + resnet training & evaluation demo
|- README.md
|- notes/related_work.md
```


### MOE Architecture

Below is a concise description and diagram of the Mixture-of-Experts (MoE)
classification head used in this repo. It explains the components, the
forward/inference flow, the training losses, and how the pieces map to the
implementation files in the codebase.

```mermaid
flowchart LR
	A[Backbone
	(ResNet18)\nfeatures: (B, D)] --> B[Top-K Gate\n(proj -> logits -> top-k)]
	B --> C1[Selected expert indices\n(B, k) + combine weights (B, k)]
	C1 --> D{Experts}
	subgraph EXPERTS [E experts]
		direction TB
		E1[Expert 0\nMLP -> logits (Be0, C)]
		E2[Expert 1\nMLP -> logits (Be1, C)]
		E3[Expert 2\nMLP -> logits (Be2, C)]
		E4[Expert ...]
	end
	D --> F[Weighted aggregation\n(sum over experts -> (B, C))]
	F --> G[Final logits (B, C) -> CrossEntropyLoss]
	B --> H[Aux stats: probs_mean (E,), entropy]
	H --> I[Balance regularizer\nbalance_loss(probs_mean)]
	I --> G

	style EXPERTS stroke:#333,stroke-width:1px
```

#### Architecture (detailed)

- Backbone: a standard image backbone (ResNet-18 in the repo) produces a
	per-sample feature vector of dimension D. See the training scripts for
	backbone wiring.
- Gate: a small linear projection from D -> E expert logits (no bias). The
	gate computes two related quantities:
	- a monitoring softmax over all E logits (probs) used for diagnostics
		and the balancing regularizer, and
	- a raw top-k selection using the logits themselves to pick the k highest
		experts per sample. From the selected k logits we compute a local softmax
		to produce the combine weights (one vector of k weights per sample).
	Implementation: `models/moe/gate.py`.
- Experts: a collection of E small MLPs (ExpertMLP) that map D -> C (class
	logits). Each expert is evaluated only on the subset of examples that
	selected it in their top-k. Implementation: `models/moe/block.py` (the
	`ExpertMLP` and `MoEHead`).

#### Forward pass (per minibatch)

1. Backbone computes features X with shape (B, D).
2. Gate projects X to expert logits (B, E) and: (a) computes monitoring
	 probabilities `probs = softmax(logits / temperature)`, (b) picks top-k
	 indices `topk_idx` (B, k) and corresponding logits `topk_val` (B, k).
3. Combine weights are `combine_w = softmax(topk_val, dim=-1)` (B, k).
4. For each expert e:
	 - find mask of samples that contain e in their top-k;
	 - run expert e on the masked subset to obtain logits (Be, C);
	 - extract the per-sample scalar weight for expert e from `combine_w`
		 (the implementation selects the weight whose corresponding index in
		 `topk_idx` equals e) and multiply the expert logits by that scalar;
	 - accumulate into the final logits tensor Y (B, C).

#### Training objective and monitoring

- Primary loss: CrossEntropyLoss between final logits Y and class labels.
- Auxiliary balance regularizer: `balance_loss(probs_mean, lb_coef)` which
	penalizes deviation of the per-expert mean soft-probabilities from the
	uniform distribution. Implementation: `models/moe/balance_loss.py`.
- Final training loss = CrossEntropy + lb_coef * balance_loss.
- Metrics: accuracy, NLL, and ECE are computed during evaluation and
	reliability plots are generated using utilities in `utils/reliability.py`.

#### Inference

- Use the same forward path: gate -> top-k -> experts -> weighted sum.
- Optionally use top-1 routing for faster inference: pick the highest-prob
	expert per sample and use only that expert's logits (or still use combine
	weights if you want a soft combination).
- Switch model to `eval()` to disable dropout/batch-norm updates.

#### Implementation notes and tradeoffs

- The repository implements a per-expert evaluation loop: only experts that
	receive at least one sample are evaluated. This saves compute when k is
	small and the load is sparse, at the cost of some indexing overhead.
- An alternative vectorized strategy is to scatter-add selected combine
	weights into a (Be, E) accumulation and call all experts in a batched
	fashion; this reduces Python-loop overhead but requires more temporary
	memory (Be * E) and may be slower if E is large.
- On Windows, DataLoader worker processes and library/DLL ordering can cause
	import issues for numerical libraries; that is an environment concern and
	not an algorithmic requirement of the MoE code.

#### Where to look in the code

- Gate: `bayes-moe/models/moe/gate.py`
- MoE head & experts: `bayes-moe/models/moe/block.py`
- Balance regularizer: `bayes-moe/models/moe/balance_loss.py`
- Training loop & config: `bayes-moe/train_moe.py` and `bayes-moe/configs/*.yaml`

This section should help you understand how the pieces fit together and
where to modify routing, expert architectures, or the training objective.