"""Comprehensive integration and unit tests for the FastAPI REST API."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.app import app, get_predictor


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check_endpoint(client):
    """Verify /api/health returns 200 OK and model readiness."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data
    assert data["models"]["logistic_regression"] == "ready"
    assert data["models"]["xgboost"] == "ready"
    assert "environment" in data
    assert isinstance(data["environment"]["models_cached"], int)


def test_predict_endpoint_logistic_regression(client):
    """Verify POST /api/predict with default Logistic Regression model."""
    payload = {
        "title": "Major Automotive Manufacturer Announces Safety Recall for 500k Vehicles",
        "ticker": "F",
        "description": "Brake defect prompts nationwide product recall.",
        "model_family": "logistic_regression",
        "use_llm": False
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert data["model_family"] == "logistic_regression"
    
    # Event metadata
    event = data["event"]
    assert event["ticker"] == "F"
    assert event["title"] == payload["title"]
    assert event["event_type"] == "product_recall"
    assert event["classification_source"] == "rule_based_fallback"
    
    # Short interest prediction
    si = data["short_interest"]
    assert si["status"] == "predicted"
    assert si["direction"] in {"increase", "decrease", "no_significant_change"}
    assert si["model_type"] == "LogisticRegression"
    assert si["confidence"] > 0.0
    assert "probabilities" in si
    probs = si["probabilities"]
    assert set(probs.keys()) == {"increase", "decrease", "no_significant_change"}
    assert pytest.approx(sum(probs.values()), 0.01) == 1.0


def test_predict_endpoint_xgboost(client):
    """Verify POST /api/predict with XGBoost model."""
    payload = {
        "title": "CEO liquidates 30% of company equity in massive insider sale",
        "ticker": "XYZ",
        "description": "Executive filings show unplanned open market sales.",
        "model_family": "xgboost",
        "use_llm": False
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert data["model_family"] == "xgboost"
    
    event = data["event"]
    assert event["ticker"] == "XYZ"
    assert event["event_type"] == "insider_selling"
    
    si = data["short_interest"]
    assert si["status"] == "predicted"
    assert si["model_type"] == "XGBoost"
    assert si["direction"] in {"increase", "decrease", "no_significant_change"}
    assert "probabilities" in si


def test_predict_validation_missing_title(client):
    """Verify validation error when title is missing."""
    response = client.post("/api/predict", json={"ticker": "AAPL"})
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert "Input validation error" in data["message"]


def test_predict_validation_whitespace_title(client):
    """Verify error when title is only whitespace."""
    response = client.post("/api/predict", json={"title": "   ", "ticker": "AAPL"})
    assert response.status_code in {400, 422}
    data = response.json()
    assert data["status"] == "error"


def test_predict_validation_short_title(client):
    """Verify error when title is shorter than 3 characters."""
    response = client.post("/api/predict", json={"title": "AB", "ticker": "AAPL"})
    assert response.status_code == 422


def test_predict_validation_invalid_model_family(client):
    """Verify validation error when unsupported model family is requested."""
    payload = {
        "title": "Company reports record Q3 earnings and profits",
        "model_family": "random_forest"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"


def test_model_in_memory_caching(client):
    """Verify that models are cached in memory and not reloaded repeatedly."""
    predictor = get_predictor()
    initial_cache_size = len(predictor._artifact_cache)
    assert initial_cache_size >= 2  # LR and XGBoost preloaded

    # Run predictions multiple times
    for _ in range(3):
        res = client.post("/api/predict", json={
            "title": "Biotech company receives FDA breakthrough therapy designation",
            "model_family": "logistic_regression"
        })
        assert res.status_code == 200

    # Cache count should remain constant
    assert len(predictor._artifact_cache) == initial_cache_size


@patch("feature_engineering.gemini_classifier.classify_event_with_gemini")
@patch("feature_engineering.gemini_context_analyzer.analyze_context_with_gemini")
def test_predict_with_mocked_gemini(mock_context, mock_classify, client):
    """Verify Gemini enrichment integration when enabled and mocked."""
    headline = "Federal Trade Commission launches formal antitrust investigation"
    mock_classify.return_value = {
        "event_category": "fraud_investigation",
        "confidence": 0.95,
        "explanation": "Federal antitrust probe initiated.",
        "evidence": ["antitrust investigation"],
        "secondary_categories": [],
        "is_calibrated": False
    }
    mock_context.return_value = {
        "severity": "high",
        "severity_rationale": "Formal federal regulatory enforcement action.",
        "context_summary": "FTC scrutiny over antitrust practices.",
        "risk_factors": ["regulatory fines", "forced structural remedies"],
        "supporting_evidence": ["antitrust investigation"],
        "insider_nuance": "",
        "context_flags": "Federal regulatory investigation"
    }

    payload = {
        "title": headline,
        "ticker": "GOOGL",
        "description": "Antitrust enforcement action announced in Washington.",
        "model_family": "logistic_regression",
        "use_llm": True
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    event = data["event"]
    assert event["event_type"] == "fraud_investigation"
    assert event["event_severity"] == "high"
    assert event["classification_source"] == "gemini_llm"
    assert event["explanation"] == "Federal antitrust probe initiated."
    assert "antitrust investigation" in event["evidence"]


def test_error_handling_sanitization(client, monkeypatch):
    """Verify internal server errors return sanitized responses without leaking traces."""
    predictor = get_predictor()
    
    def raise_runtime(*args, **kwargs):
        raise RuntimeError("Sensitive internal database path: /var/secrets/key.pem")

    monkeypatch.setattr(predictor, "predict_event", raise_runtime)

    payload = {
        "title": "Corporate earnings update announced today",
        "model_family": "logistic_regression"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 500
    data = response.json()
    assert data["status"] == "error"
    # Ensure no secrets or internal paths leak to client
    assert "/var/secrets" not in str(data)
    assert "key.pem" not in str(data)
    assert "internal server error" in data.get("message", "").lower() or "failed to generate prediction" in data.get("message", "").lower()
