# hCNN — Multiscale Hierarchical Convolutional Networks

PyTorch reimplementation of **Jacobsen et al., ICML 2017**  
📄 [Paper](https://arxiv.org/abs/1703.04140) · 🤗 Pretrained weights : https://huggingface.co/overfitter007/hcnn-pytorch
---

## What is hCNN?

hCNN is a structured deep convolutional network where each layer is indexed by progressively higher-dimensional attribute vectors learned from training data. Unlike standard CNNs, hCNN computes invariants by **averaging along hierarchical attribute dimensions**, making it mathematically interpretable.

**Key idea (paper eq. 8):**
```
w_vj(u, v) = Σ_q  h_j,q(u) × g_vj,q(v)
```
Each 4D filter is factorized into `Q` separable spatial (`h`) and attribute (`g`) filters.

---

## Results

| Dataset   | Model      | Q  | Params | Accuracy | Paper  |
|-----------|------------|-----|--------|----------|--------|
| CIFAR-10  | hCNN       | 9   | 0.10M  | 89.34%   | 91.43% |
| CIFAR-10  | hCNN       | 32  | 0.36M  | 91.63%   | —      |
| CIFAR-10  | hCNN+      | 32  | 0.36M  | 91.77%   | 92.50% |
| CIFAR-100 | hCNN       | 32  | 0.56M  | 63.53%   | 62.01% |

---

## Installation

```bash
git clone https://github.com/VOTRE_USERNAME/hcnn-pytorch
cd hcnn-pytorch
pip install -r requirements.txt
```

---

## Quick Start

### Load pretrained model

```python
from hcnn.pretrained import hcnn_cifar10, hcnn_plus_cifar10, hcnn_cifar100

# hCNN Q=9 — CIFAR-10
model = hcnn_cifar10(Q=9, pretrained=True)
model.eval()

# hCNN+ Q=32 — CIFAR-10
model = hcnn_plus_cifar10(pretrained=True)

# hCNN — CIFAR-100
model = hcnn_cifar100(pretrained=True)
```

### Build from config

```python
from hcnn import HCNN, HCNNPlus, get_default_config
from hcnn.config import HCNNConfig

# Paper hCNN (Q=9)
model = HCNN(get_default_config())

# Custom Q
model = HCNN(HCNNConfig(Q=16, num_classes=10))

# hCNN+ (Q=32)
model = HCNNPlus(HCNNConfig(Q=32))
```

### Inference

```python
import torch

x = torch.randn(1, 3, 32, 32)   # CIFAR image
with torch.no_grad():
    logits = model(x)            # (1, 10)
    pred   = logits.argmax(1)    # predicted class
```

---

## Training

```bash
# CIFAR-10
python examples/train_cifar10.py

# CIFAR-100 (hCNN + hCNN+ sequentially)
python examples/train_cifar100.py
```

---

## Architecture

```
Input (N, 3, 32, 32)
    │
    ▼
FirstBlock
    ├── HConv2D  (3→16, 3×3)
    ├── HConv3D  (attr_k=11)
    ├── HConv4D  (Q filters, reshape trick)
    └── reduce_dim_mean
    │
    ▼
Block × 4  (subsample=True/False alternating)
    ├── HConv4D → reduce_dim_mean
    └── HConv4D → reduce_dim_mean
    │
    ▼
3× mean (spatial + attribute dims)
    │
    ▼
Linear → (N, num_classes)
```

**HConv4D** implements the separable factorization of paper eq.(8):

```python
# Q spatial filters h_j,q(u)
conv_sp = Conv2d(1, Q, 3×3)

# Q→K attribute filters g_vj,q(v)  
conv_ch = Conv2d(Q, K, 7×11)
```

---

## File Structure

```
hcnn-pytorch/
├── hcnn/
│   ├── __init__.py      # public API
│   ├── config.py        # HCNNConfig, get_default_config()
│   ├── layers.py        # HConv2D, HConv3D, HConv4D(Q)
│   ├── blocks.py        # FirstBlock, Block, reduce_dim_mean
│   ├── model.py         # HCNN, HCNNPlus
│   ├── pretrained.py    # hcnn_cifar10(pretrained=True)
│   └── utils.py         # count_parameters, count_params_paper
├── examples/
│   ├── train_cifar10.py
│   └── train_cifar100.py
├── tests/
│   └── test_all.py
└── requirements.txt
```

---

## Citation

```bibtex
@inproceedings{jacobsen2017multiscale,
  title     = {Multiscale Hierarchical Convolutional Networks},
  author    = {Jacobsen, J{\"o}rn-Henrik and Oyallon, Edouard and
               Mallat, St{\'e}phane and Smeulders, Arnold W.M.},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  year      = {2017}
}
```

---

## License

MIT
