"""Milestone 4 Integration Test Suite: End-to-End System Validation.

Covers:
- End-to-end Short Interest Prediction (Logistic Regression & XGBoost).
- Probability distribution verification (3-class sum to ~1.0).
- Fallback behavior when Gemini is requested offline.
- Input validation (missing title, blank title, invalid model family).
- Persistent Investor Notification Store (deduplication, unread count, mark read, persistence across restarts).
- SEC Form 4 extraction data source discrimination (verified cache vs historical vs fallback).
"""
import pytest
from starlette.testclient import TestClient

from api.app import app
from alerts.notification_store import NotificationStore


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_e2e_prediction_logistic_regression(client):
    """Verify complete LR prediction workflow via REST API."""
    payload = {
        "title": "Tesla faces federal recall of 1.5M vehicles over power steering defect",
        "ticker": "TSLA",
        "description": "NHTSA notices indicate widespread steering component investigations.",
        "model_family": "logistic_regression",
        "use_llm": False
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model_family"] == "logistic_regression"
    
    # Short interest prediction structure
    si = data["short_interest"]
    assert si["status"] == "predicted"
    assert si["direction"] in ["increase", "decrease", "no_significant_change"]
    probs = si["probabilities"]
    assert set(probs.keys()) == {"increase", "decrease", "no_significant_change"}
    prob_sum = sum(probs.values())
    assert 0.99 <= prob_sum <= 1.01
    assert si["confidence"] == probs[si["direction"]]


def test_e2e_prediction_xgboost(client):
    """Verify complete XGBoost prediction workflow via REST API."""
    payload = {
        "title": "Tesla faces federal recall of 1.5M vehicles over power steering defect",
        "ticker": "TSLA",
        "description": "NHTSA notices indicate widespread steering component investigations.",
        "model_family": "xgboost",
        "use_llm": False
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model_family"] == "xgboost"
    
    si = data["short_interest"]
    assert si["status"] == "predicted"
    assert si["direction"] in ["increase", "decrease", "no_significant_change"]
    probs = si["probabilities"]
    assert set(probs.keys()) == {"increase", "decrease", "no_significant_change"}
    prob_sum = sum(probs.values())
    assert 0.99 <= prob_sum <= 1.01


def test_gemini_fallback_behavior(client):
    """Verify that when use_llm is requested, the system gracefully falls back to deterministic heuristics."""
    payload = {
        "title": "Boeing aircraft inspection finds loose bolts on door plug",
        "ticker": "BA",
        "description": "FAA halts operations pending complete structural review.",
        "model_family": "logistic_regression",
        "use_llm": True  # Will test graceful fallback if offline
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # Event metadata should be populated
    assert data["event"]["event_type"] is not None
    assert data["short_interest"]["status"] == "predicted"


def test_prediction_input_validation(client):
    """Verify validation on empty or invalid inputs."""
    # Empty title
    res_empty = client.post("/api/predict", json={"title": "   ", "model_family": "logistic_regression"})
    assert res_empty.status_code in [400, 422]

    # Invalid model family
    res_invalid_family = client.post("/api/predict", json={"title": "Valid title", "model_family": "random_forest"})
    assert res_invalid_family.status_code in [400, 422]


def test_persistent_notification_store(tmp_path):
    """Verify SQLite persistent notification store functionality."""
    db_file = tmp_path / "test_notifications.db"
    store = NotificationStore(db_path=db_file)

    # 1. Create notification
    notif1 = store.create_notification(
        title="TSLA CEO Discretionary Sale",
        message="Elon Musk sold 1.25M shares",
        ticker="TSLA",
        severity="significant",
        category="insider_selling"
    )
    assert notif1 is not None
    assert notif1["ticker"] == "TSLA"
    assert notif1["is_read"] is False
    assert store.get_unread_count() == 1

    # 2. Duplicate prevention
    dup = store.create_notification(
        title="TSLA CEO Discretionary Sale",
        message="Elon Musk sold 1.25M shares",
        ticker="TSLA",
        severity="significant",
        category="insider_selling"
    )
    assert dup is None  # Duplicate rejected
    assert store.get_unread_count() == 1

    # 3. Create second notification
    notif2 = store.create_notification(
        title="AAPL CFO Option Sale",
        message="Luca Maestri executed tax withholding",
        ticker="AAPL",
        severity="routine",
        category="insider_selling"
    )
    assert notif2 is not None
    assert store.get_unread_count() == 2

    # 4. Mark single as read
    store.mark_as_read(notif1["id"])
    assert store.get_unread_count() == 1

    # 5. Persistence across new instance (simulating server restart)
    store_restart = NotificationStore(db_path=db_file)
    assert store_restart.get_unread_count() == 1
    all_notifs = store_restart.get_notifications()
    assert len(all_notifs) == 2

    # 6. Mark all as read
    store_restart.mark_as_read()
    assert store_restart.get_unread_count() == 0

    # 7. Delete all
    store_restart.delete_notification()
    assert len(store_restart.get_notifications()) == 0


def test_api_notification_endpoints(client):
    """Verify REST API notification routes."""
    # Fetch notifications
    res = client.get("/api/notifications")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "unread_count" in data
    assert isinstance(data["notifications"], list)

    # Create custom notification
    create_payload = {
        "title": "Integration Test Alert",
        "message": "Testing automated notification ingestion.",
        "ticker": "NVDA",
        "severity": "requires_review",
        "category": "system"
    }
    create_res = client.post("/api/notifications", json=create_payload)
    assert create_res.status_code == 200
    created_data = create_res.json()
    assert created_data["status"] in ["success", "duplicate"]

    # Mark read
    mark_res = client.post("/api/notifications/mark-read", json={})
    assert mark_res.status_code == 200
    assert mark_res.json()["status"] == "success"
    assert mark_res.json()["unread_count"] == 0


def test_sec_service_data_source_discrimination(client):
    """Verify that transactions distinguish data sources."""
    res = client.get("/api/insider-transactions?ticker=TSLA&limit=2")
    assert res.status_code == 200
    txs = res.json()["transactions"]
    assert len(txs) > 0
    for tx in txs:
        assert "data_source" in tx
        assert tx["data_source"] in ["verified_cache", "historical_archive", "live_edgar", "fallback_record"]

