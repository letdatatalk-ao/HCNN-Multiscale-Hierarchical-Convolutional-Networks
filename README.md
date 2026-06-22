# hCNN — Multiscale Hierarchical Convolutional Networks

PyTorch reimplementation of **Jacobsen et al., ICML 2017**

📄 [Paper](https://arxiv.org/abs/1703.04140) · 🤗 [Pretrained Weights](https://huggingface.co/overfitter007/hcnn-pytorch) · 💻 [GitHub](https://github.com/letdatatalk-ao/HCNN-Multiscale-Hierarchical-Convolutional-Networks)

---

## Table of Contents

1. [What is hCNN?](#what-is-hcnn)
2. [Results](#results)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Training on Existing Datasets](#training-on-existing-datasets)
6. [Training on a Custom Dataset](#training-on-a-custom-dataset)
7. [Adding Pretrained Weights to HuggingFace](#adding-pretrained-weights-to-huggingface)
8. [Architecture](#architecture)
9. [File Structure](#file-structure)
10. [Citation](#citation)

---

## What is hCNN?

hCNN is a structured deep convolutional network where each layer is indexed by progressively higher-dimensional **attribute vectors** learned from training data. Unlike standard CNNs, hCNN computes invariants by averaging along hierarchical attribute dimensions, making it mathematically interpretable.

**Key idea — paper eq. (8):**

```
w_vj(u, v) = sum_q  h_j,q(u) x g_vj,q(v)    q = 1..Q
```

Each 4D filter is factorized into `Q` separable filters:
- `h_j,q(u)` — spatial filter, kernel 3x3
- `g_vj,q(v)` — attribute filter, kernel 7x11

| Q  | Role       | Params |
|----|------------|--------|
| 9  | hCNN base  | ~0.10M |
| 32 | hCNN/hCNN+ | ~0.36M |

---

## Results

| Dataset   | Model | Q  | Params | Ours   | Paper  |
|-----------|-------|----|--------|--------|--------|
| CIFAR-10  | hCNN  | 9  | 0.10M  | 89.34% | 91.43% |
| CIFAR-10  | hCNN  | 32 | 0.36M  | 91.63% | —      |
| CIFAR-10  | hCNN+ | 32 | 0.36M  | 91.77% | 92.50% |
| CIFAR-100 | hCNN  | 32 | 0.56M  | 63.53% | 62.01% |

---

## Installation

```bash
git clone https://github.com/letdatatalk-ao/HCNN-Multiscale-Hierarchical-Convolutional-Networks
cd HCNN-Multiscale-Hierarchical-Convolutional-Networks
pip install -r requirements.txt
```

---

## Quick Start

### Load a pretrained model

```python
from hcnn.pretrained import hcnn_cifar10, hcnn_plus_cifar10, hcnn_cifar100

model = hcnn_cifar10(Q=9,  pretrained=True)   # 89.34%
model = hcnn_cifar10(Q=32, pretrained=True)   # 91.63%
model = hcnn_plus_cifar10(pretrained=True)    # 91.77%
model = hcnn_cifar100(pretrained=True)        # 63.53%
```

### Build from config (no pretrained weights)

```python
from hcnn import HCNN, HCNNPlus
from hcnn.config import HCNNConfig

model = HCNN(HCNNConfig(Q=9,  num_classes=10))   # hCNN  Q=9
model = HCNN(HCNNConfig(Q=32, num_classes=10))   # hCNN  Q=32
model = HCNNPlus(HCNNConfig(Q=32))               # hCNN+
model = HCNN(HCNNConfig(Q=9,  num_classes=200))  # any number of classes
```

### Inference

```python
import torch

model.eval()
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    logits = model(x)          # (1, num_classes)
    pred   = logits.argmax(1)  # predicted class index
```

---

## Training on Existing Datasets

### CIFAR-10

```bash
python examples/train_cifar10.py
```

### CIFAR-100

```bash
python examples/train_cifar100.py           # hCNN + hCNN+ sequentially
python examples/train_cifar100.py --model hcnn   # hCNN only
python examples/train_cifar100.py --model plus   # hCNN+ only
```

Hyperparameters used (faithful to paper):
- SGD, lr=0.25, momentum=0.9, weight_decay=2e-4
- LR schedule: x0.1 at epochs 40, 80, 120, 160, 200
- Augmentation: random crop (pad=4) + horizontal flip

---

## Training on a Custom Dataset

Step-by-step guide to train hCNN on any image classification dataset.

### Step 1 — Organize your data

**Option A — ImageFolder** (recommended, one folder per class):

```
my_dataset/
├── train/
│   ├── class_0/   img1.jpg  img2.jpg ...
│   ├── class_1/   ...
│   └── class_N/   ...
└── val/
    ├── class_0/
    └── ...
```

**Option B — Custom Dataset class** (for non-standard formats):

```python
# my_dataset.py
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples   = [...]   # list of (image_path, label_int)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
```

### Step 2 — Compute normalization values

Run this once before training to get the correct mean and std:

```python
# compute_stats.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

loader = DataLoader(
    datasets.ImageFolder("data/my_dataset/train",
                         transform=transforms.ToTensor()),
    batch_size=512, num_workers=4,
)
mean = torch.zeros(3)
std  = torch.zeros(3)
for images, _ in loader:
    mean += images.mean(dim=(0, 2, 3))
    std  += images.std(dim=(0, 2, 3))
mean /= len(loader)
std  /= len(loader)
print(f"Mean: {mean.tolist()}")
print(f"Std : {std.tolist()}")
```

### Step 3 — Create your training script

Create `examples/train_custom.py` with these imports and settings:

```python
# examples/train_custom.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hcnn import HCNN, HCNNPlus
from hcnn.config import HCNNConfig
from hcnn.utils import count_params_paper

# ── Settings — change these for your dataset ────────────────────
NUM_CLASSES  = 10                      # your number of classes
IMAGE_SIZE   = 32                      # resize all images to this
Q            = 9                       # 9 = hCNN base, 32 = hCNN+
EPOCHS       = 240
BATCH_SIZE   = 50
LR           = 0.25
WEIGHT_DECAY = 2e-4
DATA_DIR     = "./data/my_dataset"
MY_MEAN      = (0.5,  0.5,  0.5)      # replace with your computed values
MY_STD       = (0.25, 0.25, 0.25)     # replace with your computed values

# ── Transforms ──────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(IMAGE_SIZE, padding=4, padding_mode="constant"),
    transforms.ToTensor(),
    transforms.Normalize(MY_MEAN, MY_STD),
])
transform_val = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MY_MEAN, MY_STD),
])

# ── DataLoaders ─────────────────────────────────────────────────
train_loader = DataLoader(
    datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform_train),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
)
val_loader = DataLoader(
    datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform_val),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
)
# If using Option B custom dataset, replace ImageFolder with MyDataset above

# ── Model ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = HCNNConfig(Q=Q, num_classes=NUM_CLASSES)
model  = HCNN(config).to(device)    # replace HCNN with HCNNPlus for hCNN+

print(f"Device  : {device}")
print(f"Model   : HCNN Q={Q}, {NUM_CLASSES} classes")
print(f"Params  : {count_params_paper(model)/1e6:.4f}M")

# ── Optimizer ───────────────────────────────────────────────────
optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[40, 80, 120, 160, 200], gamma=0.1)
criterion = nn.CrossEntropyLoss()

# ── Training loop ───────────────────────────────────────────────
best_acc  = 0.0
save_name = f"hcnn_custom_Q{Q}_best.pth"

for epoch in range(1, EPOCHS + 1):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
        acc = 100. * correct / len(val_loader.dataset)
        print(f"Epoch {epoch:3d}/{EPOCHS} | Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_name)

print(f"Best accuracy : {best_acc:.2f}%")
print(f"Weights saved : {save_name}")
```

### Step 4 — Run

```bash
python examples/train_custom.py
```

---

## Adding Pretrained Weights to HuggingFace

After training, share your weights so collaborators can load them with one line.

### Step 1 — Install and login

```bash
pip install huggingface_hub
huggingface-cli login     # paste your HuggingFace token
```

### Step 2 — Upload the .pth file

```bash
huggingface-cli upload overfitter007/hcnn-pytorch \
    hcnn_custom_Q9_best.pth \
    weights/hcnn_custom_Q9_best.pth
```

### Step 3 — Register in `hcnn/pretrained.py`

Open `hcnn/pretrained.py` and add two things:

```python
# 1. Add the URL to MODEL_URLS
MODEL_URLS["hcnn_custom_Q9"] = (
    "https://huggingface.co/overfitter007/hcnn-pytorch"
    "/resolve/main/weights/hcnn_custom_Q9_best.pth"
)

# 2. Add a factory function
def hcnn_custom(Q=9, num_classes=10, pretrained=True):
    config = HCNNConfig(Q=Q, num_classes=num_classes)
    model  = HCNN(config)
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))   # warm up LazyLinear
    if pretrained:
        _load_weights(model, MODEL_URLS["hcnn_custom_Q9"])
    return model
```

### Step 4 — Push to GitHub

```bash
git add hcnn/pretrained.py README.md
git commit -m "Add custom dataset pretrained weights"
git push
```

Now anyone can load your model with:

```python
from hcnn.pretrained import hcnn_custom
model = hcnn_custom(pretrained=True)
model.eval()
```

---

## Architecture

```
Input  (N, 3, H, W)
       |
       v
 FirstBlock
   HConv2D  (3 -> v_j, 3x3)
   HConv3D  (adds attribute dimension, kernel 11x3x3)
   HConv4D  (Q separable filters, reshape trick)
   reduce_dim_mean (mean on one attribute axis)
       |
       v
 Block x4  (subsample=True/False alternating)
   HConv4D -> reduce_dim_mean
   HConv4D -> reduce_dim_mean
       |
       v
 3x mean (H, W, attr)
       |
       v
 Linear (v_j -> num_classes)
```

**HConv4D — attribute grid is always preserved (6D output):**

```python
# Spatial conv   h_j,q(u) : Q filters 3x3
conv_sp = Conv2d(1, Q, kernel=3x3)

# Attribute conv g_vj,q(v): Q -> K filters 7x11
conv_ch = Conv2d(Q, K, kernel=7x11)

# Output: (N, H', W', n3', n4', K)  -- 6D, grid intact
```

---

## File Structure

```
HCNN-Multiscale-Hierarchical-Convolutional-Networks/
├── hcnn/
│   ├── __init__.py        public API
│   ├── config.py          HCNNConfig(Q, v_j, v_1, v_2, ...)
│   ├── layers.py          HConv2D, HConv3D, HConv4D(Q), HConv4DPlus
│   ├── blocks.py          FirstBlock, Block, reduce_dim_mean
│   ├── model.py           HCNN, HCNNPlus
│   ├── pretrained.py      hcnn_cifar10(pretrained=True)
│   └── utils.py           count_parameters, count_params_paper
├── examples/
│   ├── train_cifar10.py
│   ├── train_cifar100.py
│   └── train_custom.py    template for any dataset
├── tests/
│   └── test_all.py
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@inproceedings{jacobsen2017multiscale,
  title     = {Multiscale Hierarchical Convolutional Networks},
  author    = {Jacobsen, J{\"o}rn-Henrik and Oyallon, Edouard and
               Mallat, St{\'e}phane and Smeulders, Arnold W.M.},
  booktitle = {Proceedings of the 34th International Conference
               on Machine Learning},
  year      = {2017},
  url       = {https://arxiv.org/abs/1703.04140}
}
```

---

## License

MIT
