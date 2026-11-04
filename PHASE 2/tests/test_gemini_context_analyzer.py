import pytest
from unittest.mock import MagicMock, patch
from feature_engineering.context_analyzer import analyze_context, analyze_context_rule_based
from feature_engineering.gemini_context_analyzer import analyze_context_with_gemini

def test_context_disabled_by_default(monkeypatch):
    """Verify context analyzer defaults to rule-based logic without LLM."""
    monkeypatch.delenv("USE_GEMINI_CONTEXT", raising=False)
    row = {
        "event_type": "insider_selling",
        "shares_sold": 1000,
        "shares_owned_before": 10000,
        "insider_role": "CEO",
        "is_10b5_1": True,
        "title": "CEO sells shares",
        "description": "Routine diversification transaction."
    }
    res = analyze_context(row)
    assert res["context_source"] == "rule_based_fallback"
    assert res["percentage_holdings_sold"] == 10.0
    assert res["event_severity"] == "medium"
    assert res["insider_is_10b5_1"] is True
    assert "marked as a 10b5-1 plan" in res["context_flags"]

@patch("feature_engineering.gemini_context_analyzer.get_gemini_client")
def test_gemini_context_mocked_success(mock_get_client):
    """Mock Gemini structured response for a routine 10b5-1 planned sale."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """{
        "severity": "low",
        "severity_rationale": "Transaction was executed pursuant to a pre-arranged Rule 10b5-1 trading plan.",
        "context_summary": "Routine pre-planned insider sale for diversification.",
        "risk_factors": ["Routine portfolio rebalancing"],
        "supporting_evidence": ["pursuant to a pre-arranged Rule 10b5-1 trading plan"],
        "insider_nuance": "CEO executed pre-planned scheduled transaction."
    }"""
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    row = {
        "event_type": "insider_selling",
        "shares_sold": 5000,
        "shares_owned_before": 100000,
        "transaction_value": 250000.0,
        "insider_role": "Chief Executive Officer",
        "is_10b5_1": True,
        "title": "CEO executes planned sale",
        "description": "Sale was executed pursuant to a pre-arranged Rule 10b5-1 trading plan adopted in 2024."
    }

    res = analyze_context(row, use_llm=True)
    assert res["context_source"] == "gemini_llm"
    assert res["event_severity"] == "low"
    assert res["percentage_holdings_sold"] == 5.0
    assert res["transaction_value"] == 250000.0
    assert res["insider_is_10b5_1"] is True
    assert "pursuant to a pre-arranged Rule 10b5-1 trading plan" in res["supporting_evidence"]
    assert "pre-arranged Rule 10b5-1 trading plan" in res["severity_rationale"]
    assert "marked as a 10b5-1 plan" in res["context_flags"]

@patch("feature_engineering.gemini_context_analyzer.get_gemini_client")
def test_gemini_context_preserves_unknown_denominator(mock_get_client):
    """Preserve unknown values when holding denominator is unavailable without assuming negative."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """{
        "severity": "unknown",
        "severity_rationale": "Holding denominator is unavailable; financial impact cannot be determined.",
        "context_summary": "Insider sale reported without total share holdings baseline.",
        "risk_factors": ["Missing share baseline"],
        "supporting_evidence": ["sold 20,000 shares"],
        "insider_nuance": "Holding percentage unavailable."
    }"""
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    row = {
        "event_type": "insider_selling",
        "shares_sold": 20000,
        "shares_owned_before": None,
        "shares_owned_after": None,
        "is_10b5_1": "unknown",
        "title": "Director sold 20,000 shares",
        "description": "Filing indicates 20,000 shares disposed."
    }

    res = analyze_context(row, use_llm=True)
    assert res["context_source"] == "gemini_llm"
    assert res["event_severity"] == "unknown"
    assert res["percentage_holdings_sold"] is None
    assert res["insider_is_10b5_1"] is None
    assert "10b5-1 plan status is unknown" in res["context_flags"]

@patch("feature_engineering.gemini_context_analyzer.get_gemini_client")
def test_gemini_context_high_severity_discretionary(mock_get_client):
    """High severity unhedged discretionary insider selling (>25%) by key executive."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """{
        "severity": "high",
        "severity_rationale": "CEO disposed of 40% of equity holdings outside any 10b5-1 plan.",
        "context_summary": "Large discretionary liquidation by senior executive.",
        "risk_factors": ["Large unhedged sale", "No 10b5-1 plan protection"],
        "supporting_evidence": ["discretionary liquidation of 40% of personal holdings"],
        "insider_nuance": "CEO unhedged discretionary sale."
    }"""
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    row = {
        "event_type": "insider_selling",
        "shares_sold": 40000,
        "shares_owned_before": 100000,
        "transaction_value": 5000000.0,
        "insider_role": "CEO",
        "is_10b5_1": False,
        "title": "CEO conducts discretionary liquidation",
        "description": "Filing confirms discretionary liquidation of 40% of personal holdings without 10b5-1 plan."
    }

    res = analyze_context(row, use_llm=True)
    assert res["context_source"] == "gemini_llm"
    assert res["event_severity"] == "high"
    assert res["percentage_holdings_sold"] == 40.0
    assert res["insider_is_10b5_1"] is False
    assert "confirmed not under a 10b5-1 plan" in res["context_flags"]

@patch("feature_engineering.gemini_context_analyzer.get_gemini_client")
def test_gemini_context_filters_hallucinated_evidence(mock_get_client):
    """Evidence phrases not present in input text must be filtered out."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """{
        "severity": "medium",
        "severity_rationale": "Analyst downgraded firm following earnings.",
        "context_summary": "Analyst downgrades rating.",
        "risk_factors": ["Lower valuation"],
        "supporting_evidence": ["lowered rating to Neutral", "whistleblower alleged illegal accounting"],
        "insider_nuance": ""
    }"""
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    row = {
        "event_type": "analyst_downgrade",
        "title": "Major broker lowered rating to Neutral",
        "description": "Target price revised to $45."
    }

    res = analyze_context(row, use_llm=True)
    assert res["context_source"] == "gemini_llm"
    assert "lowered rating to Neutral" in res["supporting_evidence"]
    # Hallucinated phrase not in text must be filtered
    assert "whistleblower alleged illegal accounting" not in res["supporting_evidence"]

def test_gemini_context_fallback_on_missing_api_key(monkeypatch):
    """Verify fallback to rule-based when GEMINI_API_KEY is not configured."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    row = {
        "event_type": "bankruptcy_distress",
        "title": "Company files Chapter 11 bankruptcy petition",
        "description": "Operations suspended."
    }
    res = analyze_context(row, use_llm=True)
    assert res["context_source"] == "rule_based_fallback"
    assert res["event_severity"] == "high"

@patch("feature_engineering.gemini_context_analyzer.get_gemini_client")
def test_gemini_context_fallback_on_api_exception(mock_get_client):
    """Verify fallback to rule-based when Gemini raises API error."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API rate limit or connection failure")
    mock_get_client.return_value = mock_client

    row = {
        "event_type": "product_recall",
        "title": "Company announces product recall",
        "description": "Safety defects identified."
    }
    res = analyze_context(row, use_llm=True)
    assert res["context_source"] == "rule_based_fallback"
    assert res["event_severity"] == "medium"

def test_gemini_context_empty_text_falls_back():
    """Verify empty text falls back cleanly to rule-based."""
    row = {
        "event_type": "insider_selling",
        "shares_sold": 500,
        "shares_owned_before": 10000,
        "title": "",
        "description": ""
    }
    res = analyze_context(row, use_llm=True)
    assert res["context_source"] == "rule_based_fallback"
    assert res["percentage_holdings_sold"] == 5.0
    assert res["event_severity"] == "low"

