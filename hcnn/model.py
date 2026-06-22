# hcnn/model.py
"""
Modèles HCNN et HCNNPlus — CORRIGÉS.

CORRECTIONS :
  1. forward() adapté à la vraie sortie 5D des blocs (plus de (N,H,W,K,1))
  2. Réduction finale : 3× reduce_mean sur les axes corrects
     → fidèle à hCNN_models.py TF2 (ancien étudiant)

Correspondance avec hCNN_models.py TF2 :
    reduce_mean axis=1 × 3 :
      (N, H, W, n3, n4) → mean(H) → (N, W, n3, n4)
                        → mean(W) → (N, n3, n4)
                        → mean(n3) → (N, n4)
      puis Dense(n4 → num_classes)
"""

import torch
import torch.nn as nn
from .blocks import FirstBlock, Block, FirstBlockPlus, BlockPlus
from .config import get_default_config


class HCNN(nn.Module):
    """hCNN — Q filtres séparables configurables (paper Q=9 → 0.098M)."""

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = get_default_config()
        v_j = config.v_j
        v_1 = config.v_1
        v_2 = config.v_2
        Q   = config.Q            # ← lu depuis config
        l2  = config.weight_decay

        self.first  = FirstBlock(v_j, v_1, v_2, Q=Q, l2_reg=l2)
        self.block1 = Block(v_j, v_1, v_2, subsample=True,  Q=Q, l2_reg=l2)
        self.block2 = Block(v_j, v_1, v_2, subsample=False, Q=Q, l2_reg=l2)
        self.block3 = Block(v_j, v_1, v_2, subsample=True,  Q=Q, l2_reg=l2)
        self.block4 = Block(v_j, v_1, v_2, subsample=False, Q=Q, l2_reg=l2)
        self.fc     = nn.LazyLinear(config.num_classes)

    def forward(self, x):
        # x: (N, 3, H, W)
        x = self.first(x)    # (N, H,   W,   n3_a, n4_a)   5D
        x = self.block1(x)   # (N, H/2, W/2, n3_b, n4_b)   5D
        x = self.block2(x)   # (N, H/2, W/2, n3_c, n4_c)   5D
        x = self.block3(x)   # (N, H/4, W/4, n3_d, n4_d)   5D
        x = self.block4(x)   # (N, H/4, W/4, n3_e, n4_e)   5D

        # 3× reduce_mean — correspondance TF2 Lambda(reduce_mean, axis=1) × 3
        # x: (N, H, W, n3, n4)
        x = x.mean(dim=1)    # mean H → (N, W,  n3, n4)
        x = x.mean(dim=1)    # mean W → (N, n3, n4)
        x = x.mean(dim=1)    # mean n3 → (N, n4)

        return self.fc(x)    # (N, num_classes)  logits bruts


class HCNNPlus(nn.Module):
    """hCNN(+) — Q filtres séparables configurables (paper Q=32 → 0.34M)."""

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = get_default_config()
        v_j = config.v_j
        v_1 = config.v_1
        v_2 = config.v_2
        Q   = config.Q            # ← lu depuis config
        l2  = config.weight_decay

        self.first  = FirstBlockPlus(v_j, v_1, v_2, Q=Q, l2_reg=l2)
        self.block1 = BlockPlus(v_j, v_1, v_2, subsample=True,  Q=Q, l2_reg=l2)
        self.block2 = BlockPlus(v_j, v_1, v_2, subsample=False, Q=Q, l2_reg=l2)
        self.block3 = BlockPlus(v_j, v_1, v_2, subsample=True,  Q=Q, l2_reg=l2)
        self.block4 = BlockPlus(v_j, v_1, v_2, subsample=False, Q=Q, l2_reg=l2)
        self.fc     = nn.LazyLinear(config.num_classes)

    def forward(self, x):
        x = self.first(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.mean(dim=1).mean(dim=1).mean(dim=1)
        return self.fc(x)
