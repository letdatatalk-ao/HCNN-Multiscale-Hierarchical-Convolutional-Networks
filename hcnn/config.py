# hcnn/config.py
"""
Configuration hCNN — CORRIGÉE.

CORRECTIONS vs version précédente :
  lr           : 0.01 → 0.25   (25× trop petit avant, fidèle au paper)
  weight_decay : 3e-4 → 2e-4   (fidèle à l2_reg du code TF2 original)

Paper / code TF2 (hCNN_cifar10.py ancien étudiant) :
  SGD lr=0.25, steps=[40,80,120,160,200]
  lr_values=[0.25, 0.025, 0.0025, 0.00025, 2.5e-5, 2.5e-6]
  weight_decay (l2_reg) = 2e-4
"""

from dataclasses import dataclass


@dataclass
class HCNNConfig:
    v_j:           int   = 16
    v_1:           int   = 7
    v_2:           int   = 11
    Q:             int   = 9    # filtres séparables : 9=hCNN, 32=hCNN+
    num_classes:   int   = 10
    batch_size:    int   = 50
    epochs:        int   = 240
    lr:            float = 0.25
    momentum:      float = 0.9
    weight_decay:  float = 2e-4
    lr_milestones: tuple = (40, 80, 120, 160, 200)
    lr_gamma:      float = 0.1


def get_default_config():
    """hCNN CIFAR-10 — Q=9, paper Table 1."""
    return HCNNConfig(Q=9)


def get_cifar100_config():
    """hCNN CIFAR-100 — Q=9, paper Table 2. v_1=11 pour kernel 11×11."""
    return HCNNConfig(
        v_j=16, v_1=11, v_2=11,
        Q=9,
        num_classes=100,
        batch_size=50, epochs=240,
        lr=0.25, momentum=0.9, weight_decay=2e-4,
        lr_milestones=(40, 80, 120, 160, 200),
        lr_gamma=0.1,
    )
