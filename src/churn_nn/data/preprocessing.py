import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# gender, PhoneService removidos (Cramér's V ≈ 0 com target)
# TotalCharges removido (r=0.83 com tenure)
# tenure omitido: sum(monthly_x + one_year_x + two_year_x) == tenure, coluna redundante
NUMERIC_FEATURES = [
    "MonthlyCharges",
    "monthly_x_tenure",
    "one_year_x_tenure",
    "two_year_x_tenure",
]
CATEGORICAL_FEATURES = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def _add_interaction_terms(X: pd.DataFrame) -> pd.DataFrame:
    """Contract × tenure: cada tipo de contrato recebe um coeficiente separado."""
    X = X.copy()
    is_monthly = (X["Contract"] == "Month-to-month").astype(float)
    is_one_year = (X["Contract"] == "One year").astype(float)
    is_two_year = (X["Contract"] == "Two year").astype(float)
    X["monthly_x_tenure"] = is_monthly * X["tenure"]
    X["one_year_x_tenure"] = is_one_year * X["tenure"]
    X["two_year_x_tenure"] = is_two_year * X["tenure"]
    return X


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    logger.info("Carregadas %d linhas de %s", len(df), path)
    return df


def build_preprocessor() -> Pipeline:
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="if_binary", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("interactions", FunctionTransformer(_add_interaction_terms, validate=False)),  # noqa: E501
            ("transform", column_transformer),
        ]
    )
