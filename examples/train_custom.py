# examples/train_custom.py
"""
Template for training hCNN on any custom dataset.
See README.md > "Training on a Custom Dataset" for full documentation.
"""
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
NUM_CLASSES  = 10
IMAGE_SIZE   = 32
Q            = 9        # 9 = hCNN base, 32 = hCNN+
EPOCHS       = 240
BATCH_SIZE   = 50
LR           = 0.25
WEIGHT_DECAY = 2e-4
DATA_DIR     = "./data/my_dataset"
MY_MEAN      = (0.5,  0.5,  0.5)   # replace with compute_stats.py output
MY_STD       = (0.25, 0.25, 0.25)  # replace with compute_stats.py output

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

# ── Model ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = HCNNConfig(Q=Q, num_classes=NUM_CLASSES)
model  = HCNN(config).to(device)

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

print(f"\nBest accuracy : {best_acc:.2f}%")
print(f"Weights saved : {save_name}")
