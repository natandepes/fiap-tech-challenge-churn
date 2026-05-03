import torch

from churn_nn.models.mlp import ChurnMLP


def test_forward_shape():
    model = ChurnMLP(input_dim=39)  # dimensão típica após OHE
    model.eval()
    x = torch.randn(16, 39)  # batch de 16 amostras
    with torch.no_grad():
        out = model(x)
    assert out.shape == (16, 1)


def test_forward_sem_nan():
    model = ChurnMLP(input_dim=39)
    model.eval()
    x = torch.randn(8, 39)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any()
