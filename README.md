# 💳 Credit Risk Prediction API

An end-to-end machine learning system that predicts the probability of loan default and explains _why_ — using XGBoost for prediction, SHAP for per-decision interpretability, FastAPI for serving, Streamlit for the dashboard, and Docker for reproducible deployment.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-brightgreen)
![SHAP](https://img.shields.io/badge/SHAP-0.52.0-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Table of contents

- [Abstract](#abstract)
- [Results](#results)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Feature engineering](#feature-engineering)
- [Data splitting](#data-splitting)
- [Model](#model)
- [Explainability](#explainability-with-shap)
- [Risk bands](#risk-bands)
- [API reference](#api-reference)
- [Streamlit dashboard](#streamlit-dashboard)
- [Validation & testing](#validation--testing)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Running with Docker](#running-with-docker)
- [Key engineering decisions](#key-engineering-decisions)
- [Known limitations & future improvements](#known-limitations--future-improvements)
- [Author](#author)

---

## Abstract

Lending decisions carry real financial and human consequences, which makes a "black box" risk score a liability, not just a technical shortcoming — a rejected applicant and the underwriter signing off both deserve to know _why_. This project predicts the probability that a loan applicant will experience serious financial distress using a gradient-boosted tree ensemble (XGBoost) trained on the **Give Me Some Credit** dataset, and pairs every prediction with a SHAP (SHapley Additive exPlanations) breakdown of the features that drove it. The system is split into a stateless FastAPI scoring service and a Streamlit dashboard that consumes it over HTTP — the same separation a production loan-origination pipeline would use, so the model can be scaled, versioned, and monitored independently of the UI.

## Results

Evaluated on a held-out test set (15% of 150,000 applicants, stratified by outcome):

| Metric           | Validation | Test   |
| ---------------- | ---------- | ------ |
| **ROC-AUC**      | 0.8711     | 0.8636 |
| **PR-AUC**       | 0.4047     | 0.4094 |
| **KS statistic** | 0.5874     | 0.5718 |

3-fold stratified cross-validation on the training set gave a best ROC-AUC of **0.8652**, confirming the held-out numbers aren't a lucky split.

Test confusion matrix:

```
[[16943, 4053],
 [  358, 1146]]
```

**Why four metrics instead of accuracy:** the dataset is roughly 93/7 imbalanced (97,982 vs. 7,018 in training), so accuracy alone is close to meaningless — predicting "no default" every time would score ~93% while being useless. ROC-AUC measures ranking quality across all thresholds; PR-AUC is the more honest metric under imbalance because it doesn't credit the model for trivially easy true negatives; and the **KS statistic** — the maximum separation between the cumulative score distributions of defaulters vs. non-defaulters — is the metric credit risk teams actually lead with. KS above 0.3 is considered a usable scorecard; **0.57 is a strong result.**

## Architecture

```text
                    Applicant Data
                          │
                          ▼
                ┌────────────────────┐
                │ Feature Engineering │
                └──────────┬─────────┘
                           ▼
                ┌────────────────────┐
                │  Median Imputation  │   (fitted on train split only)
                └──────────┬─────────┘
                           ▼
                ┌────────────────────┐
                │   XGBoost Model     │
                └─────────┬──────────┘
                     ┌─────┴─────┐
                     ▼           ▼
          Default Probability   SHAP TreeExplainer
                     │           │
                     ▼           ▼
                Risk Band    Per-feature
                     │        explanation
                     └─────┬─────┘
                           ▼
                      FastAPI
              (GET /health, POST /predict,
                 POST /predict/batch)
                           │
                           ▼  HTTP (JSON)
                     Streamlit UI
              (single applicant + batch CSV)
```

The Streamlit app never imports the model directly — it only calls the API over HTTP. This means the model can be redeployed, rolled back, or scaled without touching the UI, and the UI can be swapped out without retraining anything.

## Dataset

The project uses the **Give Me Some Credit** dataset. The prediction target is whether an applicant experienced serious financial distress (delinquency) within two years.

Original applicant-level attributes:

- Revolving utilization of unsecured lines
- Age
- Debt ratio
- Monthly income
- Number of open credit lines and loans
- Number of 30–59 / 60–89 day past-due incidents
- Number of 90+ day late incidents
- Number of real estate loans or lines
- Number of dependents

## Feature engineering

Four additional features were engineered from the raw attributes above, computed **before** imputation and included in the final 14-feature model input:

| Feature              | Definition                                                                    |
| -------------------- | ----------------------------------------------------------------------------- |
| `TotalPastDue`       | 30–59 days past due + 60–89 days past due + 90+ days late                     |
| `IncomePerDependent` | `MonthlyIncome / (NumberOfDependents + 1)` — the `+1` avoids division by zero |
| `CreditLinesPerAge`  | `NumberOfOpenCreditLinesAndLoans / (age + 1)`                                 |
| `PastDueRate`        | `TotalPastDue / (NumberOfOpenCreditLinesAndLoans + 1)`                        |

All four are derived purely from raw applicant inputs — none depend on the target or on information unavailable at the moment of underwriting.

## Data splitting

Stratified split to preserve class distribution across sets:

| Split      | Share | Rows    |
| ---------- | ----- | ------- |
| Train      | 70%   | 105,000 |
| Validation | 15%   | 22,500  |
| Test       | 15%   | 22,500  |

`random_state=42` for reproducibility. The `SimpleImputer` (median strategy) is fit **only** on the training split and reused unmodified on validation, test, and every live inference request — preventing information from val/test, or from a real applicant at serving time, from leaking into what counts as a "typical" value.

## Model

**Algorithm:** XGBoost — chosen because the problem is structured/tabular with nonlinear interactions between financial variables (e.g., high utilization _combined with_ recent delinquency compounding risk beyond either alone), which plays to tree ensembles' strengths over linear models or deep learning on a feature set this small.

**Class imbalance:** handled via `scale_pos_weight = 13.96` rather than synthetic oversampling (SMOTE) — reweighting the loss makes the rare class matter more during training without fabricating applicant records.

**Hyperparameter tuning:** `RandomizedSearchCV`, 20 parameter combinations, 3-fold stratified cross-validation, scored on ROC-AUC. Best CV ROC-AUC: **0.8652**.

Final parameters:

```
subsample          = 0.7
reg_lambda         = 10
reg_alpha          = 1
n_estimators       = 300
min_child_weight   = 5
max_depth          = 5
learning_rate      = 0.03
gamma              = 0
colsample_bytree   = 0.8
```

The non-trivial `reg_lambda`/`reg_alpha` and shallow `max_depth=5` were selected by the search itself, not hand-picked — evidence the search is actively penalizing complexity rather than overfitting to the training split.

## Explainability with SHAP

Every prediction — single or batch — returns the top features that drove it, computed via `shap.TreeExplainer`. This explainer computes **exact** Shapley values for tree ensembles in polynomial time, which is what makes real-time, per-request explanations feasible inside an API call; the model-agnostic `KernelExplainer` would be exponentially slower and unusable at request latency.

Top global SHAP features (based on 5,000 test samples):

| Feature                                | Mean Absolute SHAP |
| -------------------------------------- | ------------------ |
| `RevolvingUtilizationOfUnsecuredLines` | 0.7282             |
| `PastDueRate`                          | 0.5430             |
| `TotalPastDue`                         | 0.3014             |
| `age`                                  | 0.1661             |
| `CreditLinesPerAge`                    | 0.1485             |

## Risk bands

| Probability | Risk band |
| ----------- | --------- |
| < 0.20      | Low       |
| 0.20 – 0.50 | Medium    |
| ≥ 0.50      | High      |

These are **project-defined business-rule thresholds, not statistically calibrated decision boundaries** — a reasonable demo default, not a regulatory-grade cutoff, and that distinction is stated explicitly rather than implying false precision.

## API reference

### `GET /health`

Returns service and model-load status.

```json
{ "status": "healthy", "model_loaded": true }
```

### `POST /predict`

Scores a single applicant.

**Request:**

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.95,
  "age": 45,
  "NumberOfTime30_59DaysPastDueNotWorse": 2,
  "DebtRatio": 1.4,
  "MonthlyIncome": 5000,
  "NumberOfOpenCreditLinesAndLoans": 8,
  "NumberOfTimes90DaysLate": 1,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60_89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2
}
```

**Response:**

```json
{
  "default_probability": 0.9265,
  "risk_band": "High",
  "shap_factors": [
    {
      "feature": "RevolvingUtilizationOfUnsecuredLines",
      "value": 0.95,
      "shap_value": 1.42,
      "direction": "increases risk"
    },
    {
      "feature": "NumberOfTimes90DaysLate",
      "value": 1,
      "shap_value": 0.68,
      "direction": "increases risk"
    },
    {
      "feature": "PastDueRate",
      "value": 0.33,
      "shap_value": 0.51,
      "direction": "increases risk"
    },
    {
      "feature": "DebtRatio",
      "value": 1.4,
      "shap_value": 0.29,
      "direction": "increases risk"
    },
    {
      "feature": "age",
      "value": 45,
      "shap_value": -0.11,
      "direction": "decreases risk"
    }
  ]
}
```

Invalid input (missing field, out-of-range value) returns `422` naming the exact offending field — never a raw stack trace.

### `POST /predict/batch`

Accepts a JSON array of applicants, returns predictions in the same order. Backs the dashboard's CSV upload flow.

## Streamlit dashboard

- **Single applicant** — form-based input calling `/predict`, rendering a risk gauge and a color-coded SHAP factor breakdown (red = increases risk, green = decreases risk).
- **Batch CSV** — upload a CSV, score every row via `/predict/batch`, download results.
- **Live API status indicator** — calls `/health` so the dashboard never silently fails against a down backend.
- The frontend never loads the model directly, only talks to FastAPI — keeping the presentation layer and model-serving layer fully separate.
- `API_URL` is read from the environment (defaulting to `http://127.0.0.1:8000` locally), which is what lets the same code run unmodified both locally and in Docker Compose, where the Streamlit container reaches the API container by its service name instead of `localhost`.

## Validation & testing

Input validation via Pydantic: age bounded to (0, 120], financial values non-negative, required fields enforced, invalid types rejected — all returning clean `422`s.

```bash
python3 -m pytest tests/ -v
```

Four tests, currently **4 passed**:

1. A valid request returns a well-formed `200`.
2. A request missing a required field returns `422`.
3. An out-of-range value (age = 150) returns `422`.
4. **Directional sanity check** — raising `RevolvingUtilizationOfUnsecuredLines` from 0.02 to 0.95, holding everything else constant, must increase predicted risk. This is the test that proves the model behaves correctly, not just that the code runs.

## Project structure

```
credit-risk-api/
│
├── artifacts/
│   ├── imputer.pkl
│   ├── metrics.json
│   ├── shap_summary.png
│   └── xgboost_model.pkl
│
├── notebooks/
│   └── day1_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── main.py            # FastAPI app
│   └── schemas.py          # Pydantic request/response models
│
├── tests/
│   └── test_api.py
│
├── app.py                   # Streamlit dashboard
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.streamlit
├── requirements.txt
├── test_applicants.csv
├── .dockerignore
└── .gitignore
```

## Getting started

```bash
git clone https://github.com/mantravadi-30/credit-risk-api.git
cd credit-risk-api

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# backend
python3 -m uvicorn src.main:app --reload
# interactive docs at http://localhost:8000/docs

# frontend (separate terminal)
source venv/bin/activate
streamlit run app.py
```

## Running with Docker

```bash
docker compose up --build
```

- FastAPI: http://localhost:8000 (Swagger docs at `/docs`)
- Streamlit: http://localhost:8501

`docker-compose.yml` sets `API_URL=http://api:8000` for the Streamlit container, using Docker's internal service-name DNS rather than `localhost` — easy to get wrong, and it silently breaks container-to-container calls if missed.

## Key engineering decisions

**Why XGBoost?**
The dataset is structured/tabular with nonlinear relationships between financial variables — exactly where gradient-boosted trees outperform linear models, without the data volume or structure (sequential, spatial) that would justify deep learning.

**Why `scale_pos_weight` over SMOTE?**
SMOTE synthesizes interpolated applicants in feature space — a weaker fit for tree models and a risk of unrealistic synthetic profiles in a credit context. Reweighting the loss achieves the same goal without fabricating data.

**Why fit the imputer only on training data?**
Fitting on the full dataset before splitting would leak validation/test statistics into training; the same leak would occur at serving time if a new applicant's "typical" value were computed from a window including itself.

**Why `shap.TreeExplainer`, not `KernelExplainer`?**
Exact, fast Shapley values for tree ensembles — the only choice that makes synchronous, per-request explanations viable inside an API call.

**Why separate FastAPI and Streamlit?**
FastAPI is the model-serving layer; Streamlit is the presentation layer. This separation makes the API independently usable by other clients and lets the model be redeployed without touching the UI.

## Known limitations & future improvements

- Risk-band thresholds (0.20 / 0.50) are business-rule defaults, not calibrated against a real cost matrix (cost of a missed default vs. cost of wrongly rejecting a good applicant).
- No model monitoring or drift detection — a real deployment should track input-distribution and performance drift post-launch, since applicant populations shift over time.
- No CI/CD pipeline yet — next step is a GitHub Actions workflow running `pytest` on every push.
- No authentication on the API — fine for a demo, not for a production credit decision endpoint.
- Batch scoring is all-or-nothing on a malformed row — a production endpoint might isolate and report per-row failures instead.
- Further items: probability calibration, more extensive integration testing, cloud deployment.

## Author

**Niharika Mantravadi**
B.Tech Computer Science Engineering — Data Science
_(add LinkedIn / email / portfolio link here)_

## License

MIT
