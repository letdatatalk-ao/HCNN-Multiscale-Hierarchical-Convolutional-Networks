# tests/test_all.py
"""
Tests hCNN — Leviers 1+2+3 appliqués.

Résultats attendus :
  count_parameters()   HCNN   = ~0.1032M  (conv + BN + bias)
  count_params_paper() HCNN   = ~0.0962M  (conv only — méthode paper)
  paper target                = ~0.098M   ("approximately", arrondi paper)

  count_parameters()   HCNN+  = ~0.3605M
  count_params_paper() HCNN+  = ~0.3388M
  paper target                = ~0.34M
"""

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hcnn import HCNN, HCNNPlus, get_default_config
from hcnn.utils import count_parameters, count_params_paper


def test_forward():
    print("── test_forward ─────────────────────────────────────────")
    for ModelClass in [HCNN, HCNNPlus]:
        config = get_default_config()
        model  = ModelClass(config)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(2, 3, 32, 32))
        assert y.shape == (2, 10), \
            f"Erreur {ModelClass.__name__}: {y.shape} ≠ (2, 10)"
        print(f"  ✓ {ModelClass.__name__:10s} forward → {tuple(y.shape)}")


def test_params():
    """
    LEVIER 1 — Vérifie les deux méthodes de comptage.
    count_params_paper() doit être ≈ 0.098M (paper) pour HCNN.
    """
    print("── test_params ──────────────────────────────────────────")
    config = get_default_config()

    # ── HCNN (Q=9) ─────────────────────────────────────────────
    model = HCNN(config)
    n_full  = count_parameters(model)
    n_paper = count_params_paper(model)

    print(f"  HCNN  count_parameters()   = {n_full/1e6:.4f}M  (conv+BN+bias)")
    print(f"  HCNN  count_params_paper() = {n_paper/1e6:.4f}M  (conv only — méthode paper)")
    print(f"  HCNN  paper target         = 0.098M  (arrondi 'approximately')")
    print(f"  HCNN  écart paper          = {(n_paper/1e6 - 0.098)*1000:+.1f}K params")

    # LEVIER 2+3 : count_params_paper doit être < 0.10M
    assert n_paper / 1e6 < 0.10, \
        f"count_params_paper HCNN trop grand: {n_paper/1e6:.4f}M (attendu < 0.10M)"
    # Tolérance arrondi paper ±5%
    assert abs(n_paper / 1e6 - 0.098) < 0.008, \
        f"count_params_paper HCNN hors tolérance paper: {n_paper/1e6:.4f}M"

    print()

    # ── HCNNPlus (Q=32) ────────────────────────────────────────
    model_plus = HCNNPlus(config)
    np_full  = count_parameters(model_plus)
    np_paper = count_params_paper(model_plus)

    print(f"  HCNN+ count_parameters()   = {np_full/1e6:.4f}M  (conv+BN+bias)")
    print(f"  HCNN+ count_params_paper() = {np_paper/1e6:.4f}M  (conv only — méthode paper)")
    print(f"  HCNN+ paper target         = 0.34M")
    print(f"  HCNN+ écart paper          = {(np_paper/1e6 - 0.34)*1000:+.1f}K params")

    assert abs(np_paper / 1e6 - 0.34) < 0.025, \
        f"count_params_paper HCNNPlus hors tolérance: {np_paper/1e6:.4f}M"


def test_gradient_flow():
    print("── test_gradient_flow ───────────────────────────────────")
    for ModelClass in [HCNN, HCNNPlus]:
        config = get_default_config()
        model  = ModelClass(config)
        model.train()
        out  = model(torch.randn(2, 3, 32, 32))
        loss = torch.nn.CrossEntropyLoss()(out, torch.randint(0, 10, (2,)))
        loss.backward()
        has_grad = all(p.grad is not None
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, f"Gradients manquants dans {ModelClass.__name__}"
        print(f"  ✓ {ModelClass.__name__:10s} gradients OK  loss={loss.item():.4f}")


def test_levier2_kernel():
    """Vérifie que le FirstBlock utilise bien le kernel K/4 × K/2 = 4×8."""
    print("── test_levier2 (FirstBlock kernel K/4×K/2) ────────────")
    from hcnn.blocks import FirstBlock
    config = get_default_config()
    v_j = config.v_j  # 16

    fb = FirstBlock(v_j=v_j, Q=9)
    k1 = fb.conv4d.conv_g.kernel_size[0]
    k2 = fb.conv4d.conv_g.kernel_size[1]
    expected_k1 = v_j // 4   # 4
    expected_k2 = v_j // 2   # 8

    assert k1 == expected_k1, f"kernel_h {k1} ≠ K//4={expected_k1}"
    assert k2 == expected_k2, f"kernel_w {k2} ≠ K//2={expected_k2}"

    g_params = fb.conv4d.conv_g.weight.numel()
    expected  = 9 * v_j * (v_j//4) * (v_j//2)  # 4608

    print(f"  ✓ FirstBlock conv_g kernel = ({k1},{k2}) = (K/4, K/2) ✓")
    print(f"  ✓ conv_g params = {g_params} = Q×K×(K/4)×(K/2) = 9×16×4×8 ✓")
    assert g_params == expected, f"conv_g params {g_params} ≠ {expected}"


def test_downsampling():
    print("── test_downsampling ────────────────────────────────────")
    for ModelClass in [HCNN, HCNNPlus]:
        config = get_default_config()
        model  = ModelClass(config)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(2, 3, 32, 32))
        assert y.shape == (2, config.num_classes)
        print(f"  ✓ {ModelClass.__name__:10s} 32×32 → {tuple(y.shape)}")


if __name__ == '__main__':
    print("hCNN — Tests Leviers 1+2+3\n")
    test_forward()
    print()
    test_params()
    print()
    test_gradient_flow()
    print()
    test_levier2_kernel()
    print()
    test_downsampling()
    print("\n✓ Tous les tests passent.")
