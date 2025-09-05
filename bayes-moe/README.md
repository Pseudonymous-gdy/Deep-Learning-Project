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