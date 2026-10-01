from fastapi.testclient import TestClient

from src.main import app


def valid_applicant():

    return {
        "RevolvingUtilizationOfUnsecuredLines": 0.20,
        "age": 45,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.30,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1
    }


def test_valid_request():

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json=valid_applicant()
        )

        assert response.status_code == 200

        data = response.json()

        assert "default_probability" in data
        assert "risk_band" in data
        assert "shap_factors" in data

        assert 0 <= data["default_probability"] <= 1


def test_missing_field():

    applicant = valid_applicant()

    del applicant["age"]

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json=applicant
        )

        assert response.status_code == 422


def test_age_out_of_range():

    applicant = valid_applicant()

    applicant["age"] = 150

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json=applicant
        )

        assert response.status_code == 422


def test_higher_utilization_increases_risk():

    low_risk_applicant = valid_applicant()
    high_risk_applicant = valid_applicant()

    low_risk_applicant[
        "RevolvingUtilizationOfUnsecuredLines"
    ] = 0.02

    high_risk_applicant[
        "RevolvingUtilizationOfUnsecuredLines"
    ] = 0.95

    with TestClient(app) as client:

        low_response = client.post(
            "/predict",
            json=low_risk_applicant
        )

        high_response = client.post(
            "/predict",
            json=high_risk_applicant
        )

        assert low_response.status_code == 200
        assert high_response.status_code == 200

        low_probability = (
            low_response.json()["default_probability"]
        )

        high_probability = (
            high_response.json()["default_probability"]
        )

        assert high_probability > low_probability