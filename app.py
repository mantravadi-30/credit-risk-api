import os

import streamlit as st
import pandas as pd
import requests


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# API CONFIGURATION
# ============================================================

# Docker Compose will use http://api:8000
# Local Streamlit can use http://127.0.0.1:8000

API_URL = os.getenv(
    "API_URL",
    "http://api:8000"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    """
    <style>

    body {
        background-color: #f7f9fc;
    }

    .main-title {
        font-size: 42px;
        font-weight: 750;
        color: #172033;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        color: #667085;
        margin-bottom: 22px;
    }

    .status {
        background-color: #ecfdf3;
        border: 1px solid #c7f0d8;
        color: #16834a;
        padding: 10px 15px;
        border-radius: 10px;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #172033;
        margin-top: 12px;
        margin-bottom: 18px;
    }

    .prediction-card {
        background-color: white;
        border: 1px solid #e4e7ec;
        border-radius: 18px;
        padding: 28px;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.06);
        min-height: 315px;
    }

    .prediction-title {
        font-size: 20px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 20px;
    }

    .circle-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 10px;
    }

    .risk-circle {
        width: 190px;
        height: 190px;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .risk-circle-inner {
        width: 145px;
        height: 145px;
        border-radius: 50%;
        background-color: white;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .risk-number {
        font-size: 32px;
        font-weight: 750;
        color: #172033;
    }

    .risk-caption {
        font-size: 13px;
        color: #667085;
        margin-top: 2px;
    }

    .risk-label {
        text-align: center;
        font-size: 18px;
        font-weight: 700;
        margin-top: 12px;
    }

    .factor-title {
        font-size: 19px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 22px;
    }

    .factor {
        margin-bottom: 20px;
    }

    .factor-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
    }

    .factor-name {
        font-size: 14px;
        font-weight: 600;
        color: #344054;
    }

    .factor-shap {
        font-size: 13px;
        font-weight: 700;
    }

    .factor-description {
        font-size: 12px;
        color: #98a2b3;
        margin-bottom: 6px;
    }

    .bar-bg {
        width: 100%;
        height: 9px;
        background-color: #edf1f5;
        border-radius: 20px;
        overflow: hidden;
    }

    .bar-up {
        height: 100%;
        background-color: #e76f51;
        border-radius: 20px;
    }

    .bar-down {
        height: 100%;
        background-color: #2a9d8f;
        border-radius: 20px;
    }

    .info {
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 14px 18px;
        color: #475467;
        font-size: 14px;
        margin-top: 18px;
    }

    .batch-card {
        background-color: white;
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.05);
        margin-bottom: 18px;
    }

    </style>
    """
)


# ============================================================
# HEADER
# ============================================================

st.html(
    """
    <div class="main-title">
        💳 Credit Risk Prediction
    </div>

    <div class="subtitle">
        AI-powered credit default prediction using XGBoost
        with SHAP-based explanations.
    </div>
    """
)


# ============================================================
# API CONNECTION
# ============================================================

def check_api():

    urls = []

    # First try the environment variable.
    if API_URL:
        urls.append(API_URL)

    # Docker service name.
    if "http://api:8000" not in urls:
        urls.append("http://api:8000")

    # Local machine.
    if "http://127.0.0.1:8000" not in urls:
        urls.append("http://127.0.0.1:8000")

    for url in urls:

        try:

            response = requests.get(
                f"{url}/health",
                timeout=3
            )

            if response.status_code == 200:

                return url

        except requests.exceptions.RequestException:

            continue

    return None


ACTIVE_API_URL = check_api()


if ACTIVE_API_URL:

    st.html(
        """
        <div class="status">
            🟢 API Connected
        </div>
        """
    )

else:

    st.error(
        "🔴 API Offline. Make sure the FastAPI server is running."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "👤 Single Applicant",
        "📁 Batch CSV"
    ]
)


# ============================================================
# SINGLE APPLICANT
# ============================================================

with tab1:

    st.html(
        """
        <div class="section-title">
            Applicant Information
        </div>
        """
    )

    # --------------------------------------------------------
    # ROW 1
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        utilization = st.number_input(
            "Credit Utilization",
            min_value=0.0,
            value=0.20,
            step=0.01
        )

    with col2:

        debt_ratio = st.number_input(
            "Debt Ratio",
            min_value=0.0,
            value=0.30,
            step=0.01
        )

    with col3:

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    with col4:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45,
            step=1
        )

    # --------------------------------------------------------
    # ROW 2
    # --------------------------------------------------------

    col5, col6, col7, col8 = st.columns(4)

    with col5:

        open_credit_lines = st.number_input(
            "Open Credit Lines",
            min_value=0,
            value=8,
            step=1
        )

    with col6:

        past_due_30_59 = st.number_input(
            "30–59 Days Past Due",
            min_value=0,
            value=0,
            step=1
        )

    with col7:

        past_due_60_89 = st.number_input(
            "60–89 Days Past Due",
            min_value=0,
            value=0,
            step=1
        )

    with col8:

        late_90 = st.number_input(
            "90 Days Late",
            min_value=0,
            value=0,
            step=1
        )

    # --------------------------------------------------------
    # ROW 3
    # --------------------------------------------------------

    col9, col10 = st.columns(2)

    with col9:

        real_estate_loans = st.number_input(
            "Real Estate Loans",
            min_value=0,
            value=1,
            step=1
        )

    with col10:

        dependents = st.number_input(
            "Number of Dependents",
            min_value=0.0,
            value=1.0,
            step=1.0
        )

    st.write("")

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔮 Predict Credit Risk",
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        if ACTIVE_API_URL is None:

            st.error(
                "FastAPI is not connected."
            )

        else:

            applicant = {

                "RevolvingUtilizationOfUnsecuredLines":
                    utilization,

                "age":
                    age,

                "NumberOfTime30_59DaysPastDueNotWorse":
                    past_due_30_59,

                "DebtRatio":
                    debt_ratio,

                "MonthlyIncome":
                    monthly_income,

                "NumberOfOpenCreditLinesAndLoans":
                    open_credit_lines,

                "NumberOfTimes90DaysLate":
                    late_90,

                "NumberRealEstateLoansOrLines":
                    real_estate_loans,

                "NumberOfTime60_89DaysPastDueNotWorse":
                    past_due_60_89,

                "NumberOfDependents":
                    dependents
            }

            try:

                response = requests.post(
                    f"{ACTIVE_API_URL}/predict",
                    json=applicant,
                    timeout=30
                )

                if response.status_code == 200:

                    result = response.json()

                    probability = float(
                        result["default_probability"]
                    )

                    risk_band = result[
                        "risk_band"
                    ]

                    factors = result[
                        "shap_factors"
                    ]

                    # ------------------------------------------------
                    # RISK COLOR
                    # ------------------------------------------------

                    if risk_band == "Low":

                        risk_color = "#2a9d8f"

                    elif risk_band == "Medium":

                        risk_color = "#f4a261"

                    else:

                        risk_color = "#e76f51"

                    percentage = probability * 100

                    # ------------------------------------------------
                    # RISK ASSESSMENT
                    # ------------------------------------------------

                    st.html(
                        """
                        <div class="section-title">
                            Risk Assessment
                        </div>
                        """
                    )

                    left, right = st.columns(
                        [0.9, 1.4]
                    )

                    # ==============================================
                    # RISK SCORE
                    # ==============================================

                    with left:

                        st.html(
                            f"""
                            <div class="prediction-card">

                                <div class="prediction-title">
                                    Default Risk
                                </div>

                                <div class="circle-container">

                                    <div
                                        class="risk-circle"
                                        style="
                                            background:
                                            conic-gradient(
                                                {risk_color}
                                                {percentage}%,
                                                #edf1f5
                                                {percentage}%
                                            );
                                        "
                                    >

                                        <div class="risk-circle-inner">

                                            <div class="risk-number">
                                                {percentage:.0f}%
                                            </div>

                                            <div class="risk-caption">
                                                default risk
                                            </div>

                                        </div>

                                    </div>

                                </div>

                                <div
                                    class="risk-label"
                                    style="color:{risk_color};"
                                >
                                    {risk_band} Risk
                                </div>

                            </div>
                            """
                        )

                    # ==============================================
                    # SHAP
                    # ==============================================

                    with right:

                        st.html(
                            """
                            <div class="prediction-card">

                                <div class="factor-title">
                                    🔍 Top Factors — SHAP
                                </div>
                            """
                        )

                        if factors:

                            max_shap = max(
                                abs(
                                    float(
                                        factor["shap_value"]
                                    )
                                )
                                for factor in factors
                            )

                            if max_shap == 0:

                                max_shap = 1

                            for factor in factors:

                                feature = factor[
                                    "feature"
                                ]

                                value = float(
                                    factor["value"]
                                )

                                shap_value = float(
                                    factor["shap_value"]
                                )

                                direction = factor[
                                    "direction"
                                ]

                                width = min(
                                    (
                                        abs(shap_value)
                                        / max_shap
                                    ) * 100,
                                    100
                                )

                                if shap_value > 0:

                                    bar_class = "bar-up"
                                    symbol = "↑"
                                    factor_color = "#e76f51"

                                else:

                                    bar_class = "bar-down"
                                    symbol = "↓"
                                    factor_color = "#2a9d8f"

                                st.html(
                                    f"""
                                    <div class="factor">

                                        <div class="factor-top">

                                            <span
                                                class="factor-name"
                                            >
                                                {feature}
                                            </span>

                                            <span
                                                class="factor-shap"
                                                style="
                                                    color:
                                                    {factor_color};
                                                "
                                            >
                                                {symbol}
                                                {abs(shap_value):.3f}
                                            </span>

                                        </div>

                                        <div class="factor-description">
                                            Value: {value:.4f}
                                            · {direction}
                                        </div>

                                        <div class="bar-bg">

                                            <div
                                                class="{bar_class}"
                                                style="
                                                    width:
                                                    {width:.1f}%;
                                                "
                                            ></div>

                                        </div>

                                    </div>
                                    """
                                )

                        st.html(
                            """
                            </div>
                            """
                        )

                    # ------------------------------------------------
                    # EXPLANATION
                    # ------------------------------------------------

                    st.html(
                        """
                        <div class="info">
                            <b>How to read this:</b>
                            SHAP values show how individual features
                            contributed to this applicant's prediction.
                            Red factors push predicted risk upward,
                            while green factors push it downward.
                        </div>
                        """
                    )

                else:

                    st.error(
                        f"API returned "
                        f"{response.status_code}"
                    )

                    try:

                        st.json(
                            response.json()
                        )

                    except ValueError:

                        st.write(
                            response.text
                        )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Could not connect to API: {e}"
                )


# ============================================================
# BATCH CSV
# ============================================================

with tab2:

    st.html(
        """
        <div class="section-title">
            Batch Credit Risk Prediction
        </div>
        """
    )

    st.html(
        """
        <div class="batch-card">
            Upload a CSV containing multiple applicants.
            Each applicant will be sent to the FastAPI
            batch prediction endpoint.
        </div>
        """
    )

    st.write("")

    uploaded_file = st.file_uploader(
        "Upload applicant CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(
            uploaded_file
        )

        st.subheader(
            "Uploaded Applicants"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        st.html(
            f"""
            <div class="info">
                📊 <b>{len(df)}</b> applicants loaded.
            </div>
            """
        )

        st.write("")

        predict_batch_button = st.button(
            "🚀 Predict All Applicants",
            type="primary",
            use_container_width=True
        )

        if predict_batch_button:

            if ACTIVE_API_URL is None:

                st.error(
                    "FastAPI is not connected."
                )

            else:

                records = (
                    df.where(
                        pd.notna(df),
                        None
                    )
                    .to_dict(
                        orient="records"
                    )
                )

                try:

                    response = requests.post(
                        f"{ACTIVE_API_URL}/predict/batch",
                        json=records,
                        timeout=60
                    )

                    if response.status_code == 200:

                        result = response.json()

                        predictions = pd.DataFrame(
                            result["predictions"]
                        )

                        st.subheader(
                            "Prediction Results"
                        )

                        st.dataframe(
                            predictions,
                            use_container_width=True
                        )

                        csv_data = predictions.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "⬇️ Download Results",
                            csv_data,
                            "credit_risk_predictions.csv",
                            "text/csv",
                            use_container_width=True
                        )

                    else:

                        st.error(
                            f"API returned "
                            f"{response.status_code}"
                        )

                        try:

                            st.json(
                                response.json()
                            )

                        except ValueError:

                            st.write(
                                response.text
                            )

                except requests.exceptions.RequestException as e:

                    st.error(
                        f"Could not connect to API: {e}"
                    )