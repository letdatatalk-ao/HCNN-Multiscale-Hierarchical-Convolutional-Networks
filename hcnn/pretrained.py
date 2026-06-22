# hcnn/pretrained.py
"""
Pretrained hCNN models — Jacobsen et al., ICML 2017.
PyTorch reimplementation — overfitter007/hcnn-pytorch.

Usage:
    from hcnn.pretrained import hcnn_cifar10, hcnn_plus_cifar10, hcnn_cifar100

    model = hcnn_cifar10(Q=9,  pretrained=True)   # 89.34% CIFAR-10
    model = hcnn_cifar10(Q=32, pretrained=True)   # 91.63% CIFAR-10
    model = hcnn_plus_cifar10(pretrained=True)    # 91.77% CIFAR-10
    model = hcnn_cifar100(pretrained=True)        # 63.53% CIFAR-100
"""

import torch
from .model  import HCNN, HCNNPlus
from .config import HCNNConfig

# ── HuggingFace Hub URLs ─────────────────────────────────────────
_HF_BASE = "https://huggingface.co/overfitter007/hcnn-pytorch/resolve/main"

MODEL_URLS = {
    "hcnn_q9_cifar10":    f"{_HF_BASE}/weights/hcnn_q9_cifar10_best.pth",
    "hcnn_q32_cifar10":   f"{_HF_BASE}/weights/hcnn_q32_cifar10_best.pth",
    "hcnn_plus_q32_cifar10": f"{_HF_BASE}/weights/hcnn_plus_q32_cifar10_best.pth",
    "hcnn_q32_cifar100":  f"{_HF_BASE}/weights/hcnn_q32_cifar100_best.pth",
}


def _load_weights(model, url):
    """Télécharge et charge les poids depuis HuggingFace."""
    print(f"Downloading pretrained weights...")
    state = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=True)
    model.load_state_dict(state)
    print("Weights loaded successfully.")
    return model


def _warm_up(model):
    """Initialise LazyLinear avec un forward pass vide."""
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))
    return model


def hcnn_cifar10(Q=9, pretrained=True):
    """
    hCNN for CIFAR-10.

    Args:
        Q (int)        : separable filters — 9 (hCNN base) or 32
        pretrained (bool): load pretrained weights from HuggingFace

    Returns:
        HCNN model

    Accuracy:
        Q=9  → 89.34%
        Q=32 → 91.63%

    Example:
        from hcnn.pretrained import hcnn_cifar10
        model = hcnn_cifar10(Q=9, pretrained=True)
        model.eval()
    """
    model = _warm_up(HCNN(HCNNConfig(Q=Q, num_classes=10)))
    if pretrained:
        key = f"hcnn_q{Q}_cifar10"
        if key not in MODEL_URLS:
            raise ValueError(
                f"No pretrained weights for Q={Q} on CIFAR-10. "
                f"Available Q values: 9, 32"
            )
        _load_weights(model, MODEL_URLS[key])
    return model


def hcnn_plus_cifar10(pretrained=True):
    """
    hCNN+ for CIFAR-10 (Q=32, with intermediate ELU).

    Returns:
        HCNNPlus model — accuracy: 91.77%

    Example:
        from hcnn.pretrained import hcnn_plus_cifar10
        model = hcnn_plus_cifar10(pretrained=True)
        model.eval()
    """
    model = _warm_up(HCNNPlus(HCNNConfig(Q=32, num_classes=10)))
    if pretrained:
        _load_weights(model, MODEL_URLS["hcnn_plus_q32_cifar10"])
    return model


def hcnn_cifar100(Q=32, pretrained=True):
    """
    hCNN for CIFAR-100.

    Args:
        Q (int)        : separable filters (32 recommended)
        pretrained (bool): load pretrained weights from HuggingFace

    Returns:
        HCNN model — accuracy: 63.53%

    Example:
        from hcnn.pretrained import hcnn_cifar100
        model = hcnn_cifar100(pretrained=True)
        model.eval()
    """
    model = _warm_up(HCNN(HCNNConfig(Q=Q, v_1=11, v_2=11, num_classes=100)))
    if pretrained:
        key = f"hcnn_q{Q}_cifar100"
        if key not in MODEL_URLS:
            raise ValueError(
                f"No pretrained weights for Q={Q} on CIFAR-100. "
                f"Available Q values: 32"
            )
        _load_weights(model, MODEL_URLS[key])
    return model