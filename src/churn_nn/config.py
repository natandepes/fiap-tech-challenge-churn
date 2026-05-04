import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH: str = str(_PROJECT_ROOT / "data" / "raw" / "telco-churn.csv")
MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", _PROJECT_ROOT / "models"))

THRESHOLD: float = 0.4
SEED: int = 42
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 100
PATIENCE: int = 10
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
