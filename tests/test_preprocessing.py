from churn_nn.data.preprocessing import build_preprocessor, load_data

DATA_PATH = "data/raw/telco-churn.csv"


def test_load_data_total_charges_sem_nulos():
    df = load_data(DATA_PATH)
    assert df["TotalCharges"].isna().sum() == 0


def test_load_data_churn_binario():
    df = load_data(DATA_PATH)
    assert set(df["Churn"].unique()).issubset({0, 1})


def test_build_preprocessor_shape_saida():
    df = load_data(DATA_PATH)
    X = df.drop(columns=["customerID", "Churn"])
    preprocessor = build_preprocessor()
    X_t = preprocessor.fit_transform(X)
    assert X_t.ndim == 2
    assert X_t.shape[0] == len(df)
    assert X_t.shape[1] > 19  # OHE expande colunas categóricas
