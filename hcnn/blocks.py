# hcnn/blocks.py
"""
Blocs hCNN — CORRIGÉS pour fonctionner avec la vraie sortie 6D de HConv4D.

HConv4D retourne maintenant (N, H', W', n3', n4', n5) — 6D intact.
reduce_dim_mean() réduit UN seul axe attribut (dim=3 = n3').
Les blocs reçoivent et transmettent des tenseurs 5D (après reduce).

Correspondance avec hCNN_models.py (TF2, ancien étudiant) :
    first_block() → FirstBlock
    block()       → Block
"""

import torch
import torch.nn as nn
from .layers import HConv2D, HConv3D, HConv4D, HConv4DPlus


def reduce_dim_mean(x, dim=3):
    """
    Réduit UN axe attribut par moyenne.
    Correspondance : reduce_dim_mean(y, 3) du code TF2 original.
    In  : tenseur 6D  (N, H, W, n3, n4, n5)
    Out : tenseur 5D  (N, H, W, n4, n5)   [mean sur dim=3]
    """
    assert x.dim() == 6, f"reduce_dim_mean attend 6D, reçu {x.shape}"
    return x.mean(dim=dim)   # réduit n3 → (N, H, W, n4, n5)


# ─────────────────────────────────────────────────────────────────
# FirstBlock
#
# Correspondance TF2 first_block() :
#   hconv2d → hconv3d → hconv4d(st=(1,1,2,2)) → reduce_dim_mean(dim=3)
#
# Shapes (v_j=16, v_1=7, v_2=11) :
#   in        (N, 3,  H, W)
#   hconv2d   (N, 16, H, W)       → permute → (N, H, W, 16)
#   hconv3d   (N, H, W, 16, 16)
#   hconv4d   (N, H, W, 8,  8, 16)   6D
#   reduce    (N, H, W, 8, 16)       5D  [mean sur n3'=8]
# ─────────────────────────────────────────────────────────────────
class FirstBlock(nn.Module):
    def __init__(self, v_j=16, v_1=7, v_2=11, Q=9, l2_reg=2e-4):
        super().__init__()
        self.conv2d = HConv2D(3, v_j, k=3, s=1)
        self.conv3d = HConv3D(v_j, v_j, spatial_k=3, attr_k=v_2,
                              spatial_s=1, attr_s=1)
        self.conv4d = HConv4D(n5=v_j, Q=Q,
                               ksize_sp=3, ksize_ch1=v_1, ksize_ch2=v_2,
                               stride_sp=1, stride_ch1=2, stride_ch2=2,
                               l2_reg=l2_reg)

    def forward(self, x):
        # x: (N, 3, H, W)
        x = self.conv2d(x)             # (N, v_j, H, W)
        x = x.permute(0, 2, 3, 1)     # (N, H, W, v_j)
        x = self.conv3d(x)             # (N, H, W, v_j, v_j)
        x = self.conv4d(x)             # (N, H, W, n3', n4', v_j)  6D
        x = reduce_dim_mean(x, dim=3)  # (N, H, W, n4', v_j)       5D
        return x


# ─────────────────────────────────────────────────────────────────
# Block
#
# Correspondance TF2 block() :
#   hconv4d(st=(2,2,2,2) ou (1,1,2,2)) → reduce_dim_mean(3)
#   → hconv4d(st=(1,1,2,2)) → reduce_dim_mean(3)
#
# In  : (N, H,  W,  n3, n4)   5D attrs-last
# Out : (N, H', W', n3'', n4'')  5D attrs-last
# ─────────────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, v_j=16, v_1=7, v_2=11, subsample=False,
                 Q=9, l2_reg=2e-4):
        super().__init__()
        st_sp = 2 if subsample else 1
        kw = dict(n5=v_j, Q=Q, ksize_sp=3, ksize_ch1=v_1, ksize_ch2=v_2,
                  stride_ch1=2, stride_ch2=2, l2_reg=l2_reg)
        self.conv4d_1 = HConv4D(stride_sp=st_sp, **kw)
        self.conv4d_2 = HConv4D(stride_sp=1,     **kw)

    def forward(self, x):
        # x: (N, H, W, n3, n4)   5D
        x = self.conv4d_1(x)          # (N, H', W', n3', n4', v_j)  6D
        x = reduce_dim_mean(x, dim=3) # (N, H', W', n4', v_j)       5D
        x = self.conv4d_2(x)          # (N, H', W', n3'',n4'',v_j)  6D
        x = reduce_dim_mean(x, dim=3) # (N, H', W', n4'', v_j)      5D
        return x


# ─────────────────────────────────────────────────────────────────
# FirstBlockPlus / BlockPlus — hCNN(+), avec ELU intermédiaire
# ─────────────────────────────────────────────────────────────────
class FirstBlockPlus(nn.Module):
    def __init__(self, v_j=16, v_1=7, v_2=11, Q=32, l2_reg=2e-4):
        super().__init__()
        self.conv2d = HConv2D(3, v_j, k=3, s=1)
        self.conv3d = HConv3D(v_j, v_j, spatial_k=3, attr_k=v_2,
                              spatial_s=1, attr_s=1)
        self.conv4d = HConv4DPlus(n5=v_j, Q=Q,
                                   ksize_sp=3, ksize_ch1=v_1, ksize_ch2=v_2,
                                   stride_sp=1, stride_ch1=2, stride_ch2=2,
                                   l2_reg=l2_reg)

    def forward(self, x):
        x = self.conv2d(x)
        x = x.permute(0, 2, 3, 1)
        x = self.conv3d(x)
        x = self.conv4d(x)
        x = reduce_dim_mean(x, dim=3)
        return x


class BlockPlus(nn.Module):
    def __init__(self, v_j=16, v_1=7, v_2=11, subsample=False,
                 Q=32, l2_reg=2e-4):
        super().__init__()
        st_sp = 2 if subsample else 1
        kw = dict(n5=v_j, Q=Q, ksize_sp=3, ksize_ch1=v_1, ksize_ch2=v_2,
                  stride_ch1=2, stride_ch2=2, l2_reg=l2_reg)
        self.conv4d_1 = HConv4DPlus(stride_sp=st_sp, **kw)
        self.conv4d_2 = HConv4DPlus(stride_sp=1,     **kw)

    def forward(self, x):
        x = self.conv4d_1(x)
        x = reduce_dim_mean(x, dim=3)
        x = self.conv4d_2(x)
        x = reduce_dim_mean(x, dim=3)
        return x
