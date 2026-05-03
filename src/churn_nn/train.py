# src/churn_nn/train.py
import logging
import random
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from churn_nn.data.preprocessing import build_preprocessor, load_data
from churn_nn.models.mlp import ChurnMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

SEED = 42
DATA_PATH = "data/raw/telco-churn.csv"
MODELS_DIR = Path("models")
MLFLOW_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "telco-churn"
THRESHOLD = 0.4
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def main() -> None:
    set_seeds()
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_data(DATA_PATH)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=SEED
    )
    logger.info(
        "Split: treino=%d  val=%d  teste=%d", len(X_train), len(X_val), len(X_test)
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    logger.info("Preprocessor salvo em models/preprocessor.pkl")

    input_dim = X_train_t.shape[1]
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])

    def make_loader(X_arr, y_arr, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1),
        )
        generator = torch.Generator().manual_seed(SEED)
        return DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator
        )

    train_loader = make_loader(X_train_t, y_train, shuffle=True)
    val_loader = make_loader(X_val_t, y_val, shuffle=False)

    model = ChurnMLP(input_dim)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    mlflow_dataset = mlflow.data.from_pandas(
        pd.read_csv(DATA_PATH),
        source=DATA_PATH,
        name="telco-churn",
        targets="Churn",
    )

    with mlflow.start_run(run_name="mlp-pytorch"):
        mlflow.set_tags({
            "stage": "candidate",
            "model_type": "neural_network",
            "git_commit": git_commit,
        })
        mlflow.log_input(mlflow_dataset, context="training")
        mlflow.log_params(
            {
                "model": "MLP",
                "hidden_dims": "64-32",
                "dropout_rate": 0.3,
                "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "pos_weight": round(float(pos_weight.item()), 4),
                "threshold": THRESHOLD,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "seed": SEED,
            }
        )

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_weights: dict = {}

        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_loader.dataset)  # type: ignore[arg-type]

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += criterion(model(xb), yb).item() * len(xb)
            val_loss /= len(val_loader.dataset)  # type: ignore[arg-type]

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                logger.info(
                    "Early stopping na época %d — melhor val_loss: %.4f",
                    epoch,
                    best_val_loss,
                )
                break

        model.load_state_dict(best_weights)
        torch.save(model.state_dict(), MODELS_DIR / "mlp_best.pt")
        logger.info("Modelo salvo em models/mlp_best.pt")

        # Avaliação no test set com as 5 métricas do Canvas
        test_loader = make_loader(X_test_t, y_test, shuffle=False)
        model.eval()
        all_probs: list[float] = []
        all_labels: list[float] = []
        with torch.no_grad():
            for xb, yb in test_loader:
                probs = torch.sigmoid(model(xb))
                all_probs.extend(probs.cpu().numpy().flatten().tolist())
                all_labels.extend(yb.cpu().numpy().flatten().tolist())

        y_prob = np.array(all_probs)
        y_pred = (y_prob >= THRESHOLD).astype(int)
        y_true = np.array(all_labels).astype(int)

        test_metrics = {
            "test_auc_roc": roc_auc_score(y_true, y_prob),
            "test_pr_auc": average_precision_score(y_true, y_prob),
            "test_f1": f1_score(y_true, y_pred),
            "test_recall": recall_score(y_true, y_pred),
            "test_precision": precision_score(y_true, y_pred),
        }
        mlflow.log_metrics(test_metrics)
        for name, value in test_metrics.items():
            logger.info("%s: %.4f", name, value)

        mlflow.pytorch.log_model(model, name="model")
        mlflow.log_artifact(str(MODELS_DIR / "mlp_best.pt"))


if __name__ == "__main__":
    main()
