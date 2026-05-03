import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

DATA_PATH = "data/raw/telco-churn.csv"

SCHEMA = DataFrameSchema(
    {
        "customerID": Column(str),
        "gender": Column(str, pa.Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, pa.Check.isin([0, 1])),
        "Partner": Column(str, pa.Check.isin(["Yes", "No"])),
        "Dependents": Column(str, pa.Check.isin(["Yes", "No"])),
        "tenure": Column(int, pa.Check.ge(0)),
        "MonthlyCharges": Column(float, pa.Check.ge(0)),
        "Churn": Column(str, pa.Check.isin(["Yes", "No"])),
    }
)


def test_schema_csv_bruto():
    df = pd.read_csv(DATA_PATH)
    SCHEMA.validate(df)
