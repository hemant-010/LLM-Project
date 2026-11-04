"""End-to-end integration tests verifying frontend-backend contract compatibility."""
import pytest
from fastapi.testclient import TestClient
from api.app import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_frontend_health_contract(client):
    """Verify that Header.jsx receives expected health fields."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "models" in data
    assert "logistic_regression" in data["models"]
    assert "xgboost" in data["models"]
    assert "environment" in data
    assert "gemini_configured" in data["environment"]


def test_frontend_prediction_contract_logistic_regression(client):
    """Verify PredictionResult.jsx receives all fields for Logistic Regression."""
    payload = {
        "title": "Tesla recalls 1.5 million vehicles in US due to software failure",
        "ticker": "TSLA",
        "description": "NHTSA issues power steering defect notice.",
        "model_family": "logistic_regression",
        "use_llm": False
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert data["model_family"] == "logistic_regression"
    
    # Event verification
    event = data["event"]
    assert event["ticker"] == "TSLA"
    assert event["title"] == payload["title"]
    assert event["event_type"] is not None
    assert event["event_severity"] is not None
    assert event["classification_source"] is not None

    # Short interest verification
    si = data["short_interest"]
    assert si["status"] == "predicted"
    assert si["direction"] in {"increase", "decrease", "no_significant_change"}
    assert si["model_type"] == "LogisticRegression"
    assert "probabilities" in si
    probs = si["probabilities"]
    assert "increase" in probs
    assert "decrease" in probs
    assert "no_significant_change" in probs
    assert sum(probs.values()) == pytest.approx(1.0, rel=1e-2)

    # Stock price verification (limitation banner)
    assert "stock_price" in data
    assert "disclaimer" in data


def test_frontend_prediction_contract_xgboost(client):
    """Verify PredictionResult.jsx receives all fields for XGBoost."""
    payload = {
        "title": "Boeing 737 Max 9 aircraft grounded following mid-air breach",
        "ticker": "BA",
        "description": "Emergency airworthiness directive issued.",
        "model_family": "xgboost",
        "use_llm": False
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert data["model_family"] == "xgboost"
    si = data["short_interest"]
    assert si["status"] == "predicted"
    assert si["model_type"] == "XGBoost"
    assert si["direction"] in {"increase", "decrease", "no_significant_change"}
    assert si["confidence"] > 0.0


def test_frontend_error_handling_contract(client):
    """Verify that empty inputs return standard error format that PredictionForm displays."""
    res = client.post("/api/predict", json={"title": "  ", "ticker": "TSLA"})
    assert res.status_code in {400, 422}
    data = res.json()
    assert data["status"] == "error"
    assert "message" in data or "errors" in data

