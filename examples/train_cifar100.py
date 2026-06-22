# examples/train_cifar100.py
"""
Entraînement hCNN / hCNN+ sur CIFAR-100 — fidèle au paper Table 2.

Différences vs CIFAR-10 (paper Section 4.2) :
  v_1 = 11  →  kernel attribut 11×11 (au lieu de 7×11)
  num_classes = 100
  Normalisation CIFAR-100 spécifique

Résultats paper (Table 2) :
  hCNN   (Q=9)  : 62.01%  — 0.25M params
  hCNN+  (Q=32) : 63.19%  — 0.89M params

Usage :
  python examples/train_cifar100.py           # hCNN base
  python examples/train_cifar100.py --plus    # hCNN+
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hcnn import HCNN, HCNNPlus, get_cifar100_config
from hcnn.utils import count_parameters, count_params_paper

parser = argparse.ArgumentParser()
parser.add_argument('--plus', action='store_true',
                    help='Utiliser hCNN+ (Q=32) au lieu de hCNN (Q=9)')
args = parser.parse_args()


def main():
    config = get_cifar100_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    variant = 'HCNNPlus' if args.plus else 'HCNN'

    # ── Modèle ──────────────────────────────────────────────────
    ModelClass = HCNNPlus if args.plus else HCNN
    model = ModelClass(config).to(device)

    n_full  = count_parameters(model)
    n_paper = count_params_paper(model)
    print(f"Modèle {variant} (CIFAR-100) initialisé sur {device}.")
    print(f"  v_j={config.v_j}, v_1={config.v_1}, v_2={config.v_2} "
          f"→ kernel g = {config.v_1}×{config.v_2}")
    print(f"  count_parameters()   = {n_full/1e6:.4f}M  (conv+BN+bias)")
    print(f"  count_params_paper() = {n_paper/1e6:.4f}M  (conv only — méthode paper)")
    print(f"  Paper target         = {'0.25M' if not args.plus else '0.89M'}")

    # ── Normalisation CIFAR-100 ──────────────────────────────────
    # Valeurs calculées sur le jeu d'entraînement CIFAR-100
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    # ── Data loaders ────────────────────────────────────────────
    train_loader = DataLoader(
        datasets.CIFAR100('./data', train=True,  download=True,
                          transform=transform_train),
        batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        datasets.CIFAR100('./data', train=False, download=True,
                          transform=transform_test),
        batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Optimiseur — identique au paper ─────────────────────────
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.lr_milestones,
        gamma=config.lr_gamma,
    )
    criterion = nn.CrossEntropyLoss()

    # ── Entraînement ────────────────────────────────────────────
    best_acc  = 0.0
    save_name = f'hcnn{"_plus" if args.plus else ""}_cifar100_best.pth'

    print(f"\nDébut entraînement — {config.epochs} époques, "
          f"batch={config.batch_size}, lr={config.lr}\n")

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss   = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = model(data).argmax(dim=1)
                    correct += (pred == target).sum().item()

            acc        = 100. * correct / len(test_loader.dataset)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Époque {epoch:3d}/{config.epochs} | "
                  f"Perte : {epoch_loss:.4f} | "
                  f"Précision Test : {acc:.2f}%")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_name)

    print(f"\nEntraînement terminé.")
    print(f"Meilleure précision : {best_acc:.2f}%")
    print(f"Poids sauvegardés   : {save_name}")
    print(f"Paper target        : {'62.01%' if not args.plus else '63.19%'}")


if __name__ == '__main__':
    main()
