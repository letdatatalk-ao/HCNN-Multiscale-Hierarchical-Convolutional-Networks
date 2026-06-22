# hcnn/pretrained.py
"""
Pretrained hCNN models — Jacobsen et al., ICML 2017.
PyTorch reimplementation.

Usage:
    from hcnn.pretrained import hcnn_cifar10, hcnn_plus_cifar10

    model = hcnn_cifar10(pretrained=True)   # Q=9,  91.34% on CIFAR-10
    model = hcnn_cifar10(Q=32, pretrained=True)  # Q=32, 91.63%
    model = hcnn_plus_cifar10(pretrained=True)   # Q=32, 91.77%
    model = hcnn_cifar100(pretrained=True)       # Q=32, 63.53%
"""

import torch
from .model  import HCNN, HCNNPlus
from .config import HCNNConfig

# ── HuggingFace Hub URLs ─────────────────────────────────────────
# Format: huggingface.co/{username}/{repo}/resolve/main/{filename}
# À remplacer par vos vraies URLs après upload sur HuggingFace
_HF_BASE = "https://huggingface.co/overfitter007/hcnn-pytorch/resolve/main/weights"

MODEL_URLS = {
    "hcnn_cifar10_Q9":   f"{_HF_BASE}/hcnn_cifar10_Q9.pth",
    "hcnn_cifar10_Q32":  f"{_HF_BASE}/hcnn_cifar10_Q32.pth",
    "hcnn_plus_cifar10": f"{_HF_BASE}/hcnn_plus_cifar10_Q32.pth",
    "hcnn_cifar100_Q32": f"{_HF_BASE}/hcnn_cifar100_Q32.pth",
}


def _load_weights(model, url):
    """Télécharge et charge les poids depuis HuggingFace."""
    print(f"Downloading pretrained weights from:\n  {url}")
    state = torch.hub.load_state_dict_from_url(
        url,
        map_location="cpu",
        progress=True,
    )
    model.load_state_dict(state)
    print("Weights loaded successfully.")
    return model


def hcnn_cifar10(Q=9, pretrained=True):
    """
    hCNN for CIFAR-10.

    Args:
        Q         : number of separable filters (9=hCNN, 32=hCNN Q=32)
        pretrained: if True, load pretrained weights

    Returns:
        HCNN model  — accuracy: 89.34% (Q=9) / 91.63% (Q=32)

    Example:
        model = hcnn_cifar10(Q=9, pretrained=True)
        model.eval()
    """
    config = HCNNConfig(Q=Q, num_classes=10)
    model  = HCNN(config)

    # Warm up LazyLinear
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))

    if pretrained:
        key = f"hcnn_cifar10_Q{Q}"
        if key not in MODEL_URLS:
            raise ValueError(
                f"No pretrained weights for Q={Q}. "
                f"Available: {list(MODEL_URLS.keys())}"
            )
        _load_weights(model, MODEL_URLS[key])

    return model


def hcnn_plus_cifar10(pretrained=True):
    """
    hCNN+ for CIFAR-10 (Q=32, with intermediate ELU).

    Returns:
        HCNNPlus model — accuracy: 91.77%

    Example:
        model = hcnn_plus_cifar10(pretrained=True)
        model.eval()
    """
    config = HCNNConfig(Q=32, num_classes=10)
    model  = HCNNPlus(config)

    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))

    if pretrained:
        _load_weights(model, MODEL_URLS["hcnn_plus_cifar10"])

    return model


def hcnn_cifar100(Q=32, pretrained=True):
    """
    hCNN for CIFAR-100.

    Args:
        Q         : number of separable filters (32 recommended)
        pretrained: if True, load pretrained weights

    Returns:
        HCNN model — accuracy: 63.53% (Q=32)

    Example:
        model = hcnn_cifar100(pretrained=True)
        model.eval()
    """
    config = HCNNConfig(Q=Q, v_1=11, v_2=11, num_classes=100)
    model  = HCNN(config)

    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))

    if pretrained:
        key = f"hcnn_cifar100_Q{Q}"
        if key not in MODEL_URLS:
            raise ValueError(
                f"No pretrained weights for CIFAR-100 Q={Q}."
            )
        _load_weights(model, MODEL_URLS[key])

    return model
