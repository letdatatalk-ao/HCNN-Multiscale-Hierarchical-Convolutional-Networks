# hcnn/layers.py
"""
Couches hCNN — CORRECTION MAJEURE (bug identifié par l'encadrante).

ERREUR PRÉCÉDENTE dans HConv4DFactored.forward() :
    y = y.mean(dim=(2, 3))   ← DÉTRUISAIT toute la grille attribut (v1, v2)
    → le modèle devenait un CNN 16 canaux ordinaire
    → précision plafonnait à ~60% CIFAR-10 et ~22% CIFAR-100

CORRECTION : vrai reshape trick hconv4d — grille attribut PRÉSERVÉE (6D)
Correspondance exacte avec hconv4d() de hCNN_utils.py (TF2, ancien étudiant).
"""

import torch
import torch.nn as nn
from einops import rearrange


# ─────────────────────────────────────────────────────────────────
# HConv2D — correct, inchangé
# ─────────────────────────────────────────────────────────────────
class HConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k,
                              stride=s, padding=k // 2, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.elu = nn.ELU(alpha=1.0)

    def forward(self, x):
        return self.elu(self.bn(self.conv(x)))


# ─────────────────────────────────────────────────────────────────
# HConv3D — correct, inchangé
# In  : (N, H, W, in_attr)
# Out : (N, H, W, in_attr//attr_s, out_attr)
# ─────────────────────────────────────────────────────────────────
class HConv3D(nn.Module):
    def __init__(self, in_attr, out_attr,
                 spatial_k=3, attr_k=11, spatial_s=1, attr_s=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=out_attr,
            kernel_size=(attr_k, spatial_k, spatial_k),
            stride=(attr_s, spatial_s, spatial_s),
            padding=(attr_k // 2, spatial_k // 2, spatial_k // 2),
            bias=False,
        )
        self.bn  = nn.BatchNorm3d(out_attr)
        self.elu = nn.ELU(alpha=1.0)

    def forward(self, x):
        # x: (N, H, W, in_attr)
        y = rearrange(x, 'n h w a -> n 1 a h w')
        y = self.elu(self.bn(self.conv(y)))
        y = rearrange(y, 'n c d h w -> n h w d c')
        return y


# ─────────────────────────────────────────────────────────────────
# HConv4D — CORRIGÉ — vrai reshape trick sans mean() destructeur
#
# Correspondance EXACTE avec hconv4d() TF2 :
#   A. permute+reshape : (N,H,W,n3,n4)→(N·n3·n4, 1, H, W)
#   B. Conv2D spatiale : →(N·n3·n4, n5·2, H', W')
#   C. BN + ELU
#   D. reshape+permute : →(N·H'·W', n5·2, n3, n4)
#   E. Conv2D attribut : →(N·H'·W', n5, n3', n4')
#   F. BN + ELU
#   G. reshape final   : →(N, H', W', n3', n4', n5)   6D INTACT
#
#   ⚠️ PAS de mean(dim=(2,3)) — la grille attribut est PRÉSERVÉE
#
# In  : (N, H,  W,  n3,  n4)    attrs-last 5D
# Out : (N, H', W', n3', n4', n5)  attrs-last 6D
# ─────────────────────────────────────────────────────────────────
class HConv4D(nn.Module):
    def __init__(self, n5, Q=9, ksize_sp=3, ksize_ch1=7, ksize_ch2=11,
                 stride_sp=1, stride_ch1=2, stride_ch2=2, l2_reg=2e-4):
        """
        Q : nombre de filtres séparables (paper eq.8)
            Q=9  → hCNN   (0.098M params, paper Table 1)
            Q=32 → hCNN+  (0.34M  params, paper Table 1)
        """
        super().__init__()
        self.n5 = n5
        self.Q  = Q

        # h_j,q(u) : Q filtres spatiaux 3×3
        # paper eq.8 : Conv2d(1 → Q, ksp×ksp)
        self.conv_sp = nn.Conv2d(1, Q,
                                  kernel_size=ksize_sp, stride=stride_sp,
                                  padding=ksize_sp // 2, bias=False)
        self.bn_sp  = nn.BatchNorm2d(Q)
        self.elu_sp = nn.ELU(alpha=1.0)

        # g_vj,q(v) : Q×K filtres d'attributs
        # paper eq.8 : Conv2d(Q → n5, kch1×kch2)
        self.conv_ch = nn.Conv2d(Q, n5,
                                  kernel_size=(ksize_ch1, ksize_ch2),
                                  stride=(stride_ch1, stride_ch2),
                                  padding=(ksize_ch1 // 2, ksize_ch2 // 2),
                                  bias=False)
        self.bn_ch  = nn.BatchNorm2d(n5)
        self.elu_ch = nn.ELU(alpha=1.0)

    def forward(self, x):
        # x: (N, H, W, n3, n4)
        N, H, W, n3, n4 = x.shape

        # ── Fold attrs → batch : (N, H, W, n3, n4) → (N·n3·n4, 1, H, W)
        y = x.permute(0, 3, 4, 1, 2).contiguous()   # (N, n3, n4, H, W)
        y = y.reshape(N * n3 * n4, 1, H, W)

        # ── Conv spatiale h_j,q(u) : → (N·n3·n4, Q, H', W')
        y = self.elu_sp(self.bn_sp(self.conv_sp(y)))
        Hp, Wp = y.shape[2], y.shape[3]

        # ── Unfold + refold pour conv attribut
        y = y.reshape(N, n3, n4, self.Q, Hp, Wp)
        y = y.permute(0, 4, 5, 1, 2, 3).contiguous()  # (N, H', W', n3, n4, Q)
        y = y.reshape(N * Hp * Wp, self.Q, n3, n4)

        # ── Conv attribut g_vj,q(v) : (n3,n4) = espace 2D
        y = self.elu_ch(self.bn_ch(self.conv_ch(y)))   # (N·H'·W', n5, n3', n4')
        n3p, n4p = y.shape[2], y.shape[3]

        # ── Reshape final → 6D attrs-last — GRILLE PRÉSERVÉE
        y = y.reshape(N, Hp, Wp, n3p, n4p, self.n5)
        return y


# ─────────────────────────────────────────────────────────────────
# HConv4DPlus — hCNN(+) : ELU intermédiaire entre sp et ch
# ─────────────────────────────────────────────────────────────────
class HConv4DPlus(HConv4D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elu_mid = nn.ELU(alpha=1.0)

    def forward(self, x):
        N, H, W, n3, n4 = x.shape
        y = x.permute(0, 3, 4, 1, 2).contiguous()
        y = y.reshape(N * n3 * n4, 1, H, W)
        y = self.elu_sp(self.bn_sp(self.conv_sp(y)))
        Hp, Wp = y.shape[2], y.shape[3]
        y = y.reshape(N, n3, n4, self.Q, Hp, Wp)
        y = y.permute(0, 4, 5, 1, 2, 3).contiguous()
        y = y.reshape(N * Hp * Wp, self.Q, n3, n4)
        y = self.elu_mid(y)   # ← non-linéarité intermédiaire hCNN(+)
        y = self.elu_ch(self.bn_ch(self.conv_ch(y)))
        n3p, n4p = y.shape[2], y.shape[3]
        y = y.reshape(N, Hp, Wp, n3p, n4p, self.n5)
        return y
