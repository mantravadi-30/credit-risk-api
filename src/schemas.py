from pydantic import BaseModel, Field
from typing import List


class Applicant(BaseModel):

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ge=0
    )

    age: int = Field(
        gt=0,
        le=120
    )

    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ge=0
    )

    DebtRatio: float = Field(
        ge=0
    )

    MonthlyIncome: float | None = Field(
        default=None,
        ge=0
    )

    NumberOfOpenCreditLinesAndLoans: int = Field(
        ge=0
    )

    NumberOfTimes90DaysLate: int = Field(
        ge=0
    )

    NumberRealEstateLoansOrLines: int = Field(
        ge=0
    )

    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ge=0
    )

    NumberOfDependents: float | None = Field(
        default=None,
        ge=0
    )


class PredictionResponse(BaseModel):

    default_probability: float

    risk_band: str

    shap_factors: List[dict]


class BatchResponse(BaseModel):

    predictions: List[PredictionResponse]