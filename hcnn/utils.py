# hcnn/utils.py
"""
Utilitaires hCNN.

LEVIER 1 — count_params_paper()
  Reproduit la méthode de comptage du paper (Section 5.2) :
  "The number of free parameters is the number of parameters
   of the convolution kernels w_vj"
  → Conv weights uniquement, sans BatchNorm, sans bias.

  Résultat sur notre architecture :
    HCNN    (Q=9)  : 0.0962M  ≈ paper 0.098M
    HCNNPlus(Q=32) : 0.3388M  ≈ paper 0.34M
"""

import torch.nn as nn


def reduce_dim_mean(x, dim):
    return x.mean(dim=dim, keepdim=False)


class ReduceDimMean(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return reduce_dim_mean(x, self.dim)


def count_parameters(model):
    """Comptage standard — inclut tous les paramètres (conv + BN + bias)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_params_paper(model):
    """
    LEVIER 1 — Comptage fidèle à la méthode du paper (Section 5.2).

    Compte UNIQUEMENT les poids des couches de convolution (Conv2d, Conv3d,
    Conv1d, Linear), sans BatchNorm et sans bias — exactement comme le paper.

    Paper: "The number of free parameters of the original architecture is
            the number of parameters of the convolution kernels w_vj"

    Exemple (avec Leviers 1+2+3) :
      HCNN    (Q=9)  : 96 217 params = 0.0962M  (paper : 0.098M)
      HCNNPlus(Q=32) : 338 825 params = 0.3388M  (paper : 0.34M)
    """
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            total += m.weight.numel()   # poids seulement, pas bias
        elif isinstance(m, nn.Linear):
            total += m.weight.numel()   # poids seulement, pas bias
        # BatchNorm (weight + bias) intentionnellement exclus
    return total


def param_breakdown(model):
    """Affiche le détail des paramètres par couche (deux méthodes)."""
    conv_total, bn_total, bias_total = 0, 0, 0
    print(f"{'Couche':50s} {'Standard':>12} {'Paper (conv)':>14}")
    print("─" * 80)
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            w = m.weight.numel()
            b = m.bias.numel() if m.bias is not None else 0
            conv_total += w
            bias_total += b
            print(f"  {name:48s} {w+b:>12,} {w:>14,}")
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            n = sum(p.numel() for p in m.parameters())
            bn_total += n
    print("─" * 80)
    full = conv_total + bn_total + bias_total
    print(f"  {'Conv weights':48s} {conv_total:>12,} {conv_total:>14,}")
    print(f"  {'BatchNorm':48s} {bn_total:>12,} {'(exclu)':>14}")
    print(f"  {'Bias':48s} {bias_total:>12,} {'(exclu)':>14}")
    print(f"  {'TOTAL':48s} {full:>12,} {conv_total:>14,}")
    print(f"\n  count_parameters()  = {full:,}  = {full/1e6:.4f}M")
    print(f"  count_params_paper()= {conv_total:,}  = {conv_total/1e6:.4f}M")
    return full, conv_total
