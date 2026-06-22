# compute_stats.py
"""
Compute mean and std of a custom dataset before training.
Place this file at the root of your project and run:
    python compute_stats.py

Then copy the output values into train_custom.py:
    MY_MEAN = (...)
    MY_STD  = (...)
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = "./data/my_dataset/train"  # change to your train folder

loader = DataLoader(
    datasets.ImageFolder(DATA_DIR, transform=transforms.ToTensor()),
    batch_size=512, shuffle=False, num_workers=4,
)

mean = torch.zeros(3)
std  = torch.zeros(3)

print("Computing dataset statistics...")
for images, _ in loader:
    mean += images.mean(dim=(0, 2, 3))
    std  += images.std(dim=(0, 2, 3))

mean /= len(loader)
std  /= len(loader)

print(f"\nDataset : {DATA_DIR}")
print(f"Batches : {len(loader)}")
print(f"\nCopy these values into train_custom.py:")
print(f"  MY_MEAN = ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")
print(f"  MY_STD  = ({std[0]:.4f},  {std[1]:.4f},  {std[2]:.4f})")
