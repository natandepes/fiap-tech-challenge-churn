import pytest
import torch

from churn_nn.data.preprocessing import build_preprocessor, load_data
from churn_nn.models.mlp import ChurnMLP

DATA_PATH = "data/raw/telco-churn.csv"


@pytest.fixture(scope="module")
def input_dim() -> int:
    df = load_data(DATA_PATH)
    X = df.drop(columns=["customerID", "Churn"])
    return build_preprocessor().fit_transform(X).shape[1]


def test_forward_shape(input_dim):
    model = ChurnMLP(input_dim)
    model.eval()
    x = torch.randn(16, input_dim)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (16, 1)


def test_forward_sem_nan(input_dim):
    model = ChurnMLP(input_dim)
    model.eval()
    x = torch.randn(8, input_dim)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any()


def test_pipeline_csv_para_probabilidade(input_dim):
    """Fluxo completo: CSV bruto → preprocessor → MLP → probabilidades em [0, 1]."""
    df = load_data(DATA_PATH)
    X = df.drop(columns=["customerID", "Churn"])
    preprocessor = build_preprocessor()
    X_t = preprocessor.fit_transform(X)

    model = ChurnMLP(input_dim)
    model.eval()
    x = torch.tensor(X_t[:10], dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))

    assert probs.shape == (10, 1)
    assert (probs >= 0).all() and (probs <= 1).all()
