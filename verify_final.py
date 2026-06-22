import torch
from hcnn import HCNN, HCNNPlus, get_default_config, get_cifar100_config
from hcnn.config import HCNNConfig
from hcnn.utils import count_parameters, count_params_paper

print("=" * 55)
print("TEST 1 — Q parametrable")
print("=" * 55)
for Q in [9, 16, 32]:
    m = HCNN(HCNNConfig(Q=Q))
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 32, 32))
    n = count_params_paper(m)
    print(f"  Q={Q:2d}  out={tuple(y.shape)}  params={n/1e6:.4f}M")

print()
print("=" * 55)
print("TEST 2 — CIFAR-100")
print("=" * 55)
c100 = get_cifar100_config()
m100 = HCNN(c100)
m100.eval()
with torch.no_grad():
    y100 = m100(torch.randn(2, 3, 32, 32))
print(f"  CIFAR-100  out={tuple(y100.shape)}  params={count_params_paper(m100)/1e6:.4f}M")

print()
print("=" * 55)
print("TEST 3 — Gradient flow")
print("=" * 55)
m = HCNN(get_default_config())
m.train()
out  = m(torch.randn(4, 3, 32, 32))
loss = torch.nn.CrossEntropyLoss()(out, torch.randint(0, 10, (4,)))
loss.backward()
has_grad = all(p.grad is not None for p in m.parameters() if p.requires_grad)
print(f"  Gradients OK : {has_grad}")
print(f"  Loss         : {loss.item():.4f}")

print()
print("=" * 55)
print("TEST 4 — HCNNPlus")
print("=" * 55)
mp = HCNNPlus(HCNNConfig(Q=32))
mp.eval()
with torch.no_grad():
    yp = mp(torch.randn(2, 3, 32, 32))
print(f"  HCNNPlus Q=32  out={tuple(yp.shape)}  params={count_params_paper(mp)/1e6:.4f}M")

print()
if has_grad and y.shape == (2,10) and y100.shape == (2,100):
    print("OK  TOUS LES TESTS PASSENT")
else:
    print("ERREUR  certains tests ont echoue")