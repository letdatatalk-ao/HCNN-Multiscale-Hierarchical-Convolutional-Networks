# hcnn/__init__.py
"""
hCNN — Multiscale Hierarchical Convolutional Networks
PyTorch reimplementation of Jacobsen et al., ICML 2017
https://arxiv.org/abs/1703.04140

Quick start:
    from hcnn import HCNN, HCNNPlus, get_default_config
    model = HCNN(get_default_config())   # hCNN Q=9, CIFAR-10
"""
from .model  import HCNN, HCNNPlus
from .config import HCNNConfig, get_default_config, get_cifar100_config
from .utils  import count_parameters, count_params_paper, param_breakdown

__version__ = "1.0.0"
__all__ = [
    "HCNN", "HCNNPlus",
    "HCNNConfig", "get_default_config", "get_cifar100_config",
    "count_parameters", "count_params_paper", "param_breakdown",
]
