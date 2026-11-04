"""Tests for Milestone 3: Significant Insider Selling Detection & Investor Notifications."""
import pytest
from fastapi.testclient import TestClient

from api.app import app
from alerts.insider_analyzer import (
    calculate_percentage_sold,
    is_c_suite_role,
    analyze_insider_transaction
)
from preprocessing.sec_form4_service import get_sec_service


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_calculate_percentage_sold():
    # Calculation with shares_owned_before
    assert calculate_percentage_sold(1000, shares_owned_before=10000) == 10.0
    # Calculation with shares_owned_after: 2500 / (2500 + 7500) = 25%
    assert calculate_percentage_sold(2500, shares_owned_after=7500) == 25.0
    # Missing / None
    assert calculate_percentage_sold(None) is None
    assert calculate_percentage_sold(1000, shares_owned_before=0) is None


def test_is_c_suite_role():
    assert is_c_suite_role("Chief Executive Officer") is True
    assert is_c_suite_role("Senior VP, CFO") is True
    assert is_c_suite_role("President and COO") is True
    assert is_c_suite_role("Outside Director") is False
    assert is_c_suite_role(None) is False


def test_analyze_significant_discretionary_sale():
    """Verify large C-suite discretionary sale is categorized as 'significant'."""
    tx = {
        "ticker": "TSLA",
        "insider_name": "Senior Executive",
        "insider_role": "Chief Financial Officer",
        "shares_sold": 50_000,
        "price_per_share": 200.0,
        "shares_owned_after": 100_000, # 50,000 / 150,000 = 33.3%
        "is_10b5_1": False,
        "transaction_date": "2024-02-01"
    }
    analysis = analyze_insider_transaction(tx)
    assert analysis["significance_category"] == "significant"
    assert analysis["is_c_suite"] is True
    assert analysis["is_10b5_1"] is False
    assert analysis["percentage_holdings_sold"] == pytest.approx(33.33, 0.1)
    assert any("Discretionary transaction" in f for f in analysis["risk_factors"])


def test_analyze_routine_10b5_1_sale():
    """Verify modest pre-scheduled 10b5-1 sale is categorized as 'routine'."""
    tx = {
        "ticker": "AAPL",
        "insider_name": "Executive Officer",
        "insider_role": "Vice President",
        "shares_sold": 2_000,
        "price_per_share": 180.0,
        "shares_owned_after": 98_000, # 2,000 / 100,000 = 2%
        "is_10b5_1": True,
        "transaction_date": "2024-01-15"
    }
    analysis = analyze_insider_transaction(tx)
    assert analysis["significance_category"] == "routine"
    assert analysis["is_10b5_1"] is True
    assert any("Rule 10b5-1" in f for f in analysis["positive_factors"])


def test_api_insider_transactions_endpoint(client):
    """Verify GET /api/insider-transactions returns parsed Form 4 records."""
    res = client.get("/api/insider-transactions?ticker=TSLA&limit=5")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["ticker"] == "TSLA"
    assert len(data["transactions"]) >= 1
    tx = data["transactions"][0]
    assert "insider_name" in tx
    assert "insider_role" in tx
    assert "significance_category" in tx
    assert "percentage_holdings_sold" in tx
    assert "is_10b5_1" in tx


def test_api_analyze_insider_endpoint(client):
    """Verify POST /api/analyze-insider performs evaluation."""
    payload = {
        "ticker": "NVDA",
        "insider_name": "Director",
        "insider_role": "Independent Director",
        "shares_sold": 5000,
        "price_per_share": 800.0,
        "shares_owned_after": 45000,
        "is_10b5_1": True
    }
    res = client.post("/api/analyze-insider", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    analysis = data["analysis"]
    assert analysis["ticker"] == "NVDA"
    assert analysis["percentage_holdings_sold"] == 10.0
    assert analysis["is_10b5_1"] is True


def test_api_insider_alerts_endpoint(client):
    """Verify GET /api/insider-alerts returns institutional alert feed."""
    res = client.get("/api/insider-alerts")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "alerts" in data
    assert isinstance(data["alerts"], list)
    assert "disclaimer" in data

