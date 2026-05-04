from typing import Literal

from pydantic import BaseModel, Field

Contract = Literal["Month-to-month", "One year", "Two year"]
InternetService = Literal["DSL", "Fiber optic", "No"]
PaymentMethod = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
YesNo = Literal["Yes", "No"]
YesNoPhone = Literal["Yes", "No", "No phone service"]
YesNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    SeniorCitizen: Literal[0, 1]
    gender: Literal["Male", "Female"]
    Partner: YesNo
    Dependents: YesNo
    PhoneService: YesNo
    MultipleLines: YesNoPhone
    InternetService: InternetService
    OnlineSecurity: YesNoInternet
    OnlineBackup: YesNoInternet
    DeviceProtection: YesNoInternet
    TechSupport: YesNoInternet
    StreamingTV: YesNoInternet
    StreamingMovies: YesNoInternet
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod


class PredictionResponse(BaseModel):
    churn: bool
    probability: float
    threshold: float
    model_version: str
