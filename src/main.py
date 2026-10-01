from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import shap

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    Applicant,
    PredictionResponse,
    BatchResponse
)


# ============================================================
# ARTIFACT PATH
# ============================================================

ARTIFACTS_DIR = (
    Path(__file__).resolve().parent.parent / "artifacts"
)


# ============================================================
# GLOBAL OBJECTS
# ============================================================

model = None
imputer = None
explainer = None


# ============================================================
# LOAD MODEL AT STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model
    global imputer
    global explainer

    print("Loading model...")

    model = joblib.load(
        ARTIFACTS_DIR / "xgboost_model.pkl"
    )

    imputer = joblib.load(
        ARTIFACTS_DIR / "imputer.pkl"
    )

    explainer = shap.TreeExplainer(model)

    print("Model loaded successfully!")

    yield

    print("Shutting down API...")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Credit Risk API",
    description="Credit default prediction using XGBoost and SHAP",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(applicant: Applicant):

    data = applicant.model_dump()

    # Convert API field names back to dataset field names

    data[
        "NumberOfTime30-59DaysPastDueNotWorse"
    ] = data.pop(
        "NumberOfTime30_59DaysPastDueNotWorse"
    )

    data[
        "NumberOfTime60-89DaysPastDueNotWorse"
    ] = data.pop(
        "NumberOfTime60_89DaysPastDueNotWorse"
    )

    df = pd.DataFrame([data])

    # -----------------------------
    # Feature engineering
    # -----------------------------

    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )

    df["IncomePerDependent"] = (
        df["MonthlyIncome"]
        / (df["NumberOfDependents"] + 1)
    )

    df["CreditLinesPerAge"] = (
        df["NumberOfOpenCreditLinesAndLoans"]
        / (df["age"] + 1)
    )

    df["PastDueRate"] = (
        df["TotalPastDue"]
        / (df["NumberOfOpenCreditLinesAndLoans"] + 1)
    )

    # -----------------------------
    # Exact training feature order
    # -----------------------------

    feature_columns = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
        "TotalPastDue",
        "IncomePerDependent",
        "CreditLinesPerAge",
        "PastDueRate"
    ]

    df = df[feature_columns]

    # -----------------------------
    # Apply training imputer
    # -----------------------------

    df = pd.DataFrame(
        imputer.transform(df),
        columns=feature_columns
    )

    return df


# ============================================================
# RISK BAND
# ============================================================

def get_risk_band(probability):

    if probability < 0.20:
        return "Low"

    elif probability < 0.50:
        return "Medium"

    else:
        return "High"


# ============================================================
# SHAP EXPLANATION
# ============================================================

def get_shap_factors(features):

    shap_values = explainer.shap_values(
        features
    )

    values = shap_values[0]

    explanation = pd.DataFrame({
        "feature": features.columns,
        "value": features.iloc[0].values,
        "shap_value": values
    })

    explanation["abs_shap"] = (
        explanation["shap_value"].abs()
    )

    explanation = explanation.sort_values(
        "abs_shap",
        ascending=False
    )

    top5 = explanation.head(5)

    factors = []

    for _, row in top5.iterrows():

        if row["shap_value"] > 0:
            direction = "increases risk"
        else:
            direction = "decreases risk"

        factors.append({
            "feature": row["feature"],
            "value": float(row["value"]),
            "shap_value": float(row["shap_value"]),
            "direction": direction
        })

    return factors


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(applicant: Applicant):

    features = prepare_features(
        applicant
    )

    probability = model.predict_proba(
        features
    )[0, 1]

    risk_band = get_risk_band(
        probability
    )

    shap_factors = get_shap_factors(
        features
    )

    return {
        "default_probability": round(
            float(probability),
            4
        ),
        "risk_band": risk_band,
        "shap_factors": shap_factors
    }


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post(
    "/predict/batch",
    response_model=BatchResponse
)
def predict_batch(
    applicants: list[Applicant]
):

    predictions = []

    for applicant in applicants:

        features = prepare_features(
            applicant
        )

        probability = model.predict_proba(
            features
        )[0, 1]

        risk_band = get_risk_band(
            probability
        )

        shap_factors = get_shap_factors(
            features
        )

        predictions.append({
            "default_probability": round(
                float(probability),
                4
            ),
            "risk_band": risk_band,
            "shap_factors": shap_factors
        })

    return {
        "predictions": predictions
    }