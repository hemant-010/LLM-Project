import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from feature_engineering.build_features import enrich_events

@pytest.fixture
def sample_events_df():
    return pd.DataFrame([
        {
            "ticker": "AAPL",
            "title": "CEO executes planned sale of common stock",
            "description": "Chief Executive Officer executed sale of shares pursuant to Rule 10b5-1 trading plan.",
            "published_at": "2024-01-15T14:30:00Z",
            "source": "SEC EDGAR",
            "url": "https://sec.gov/filing/1",
            "shares_sold": 5000,
            "shares_owned_before": 100000,
            "transaction_value": 750000.0,
            "insider_role": "CEO",
            "is_10b5_1": True,
            "transaction_code": "S"
        },
        {
            "ticker": "TSLA",
            "title": "Major Automotive Manufacturer Announces Safety Recall for 500k Vehicles",
            "description": "Brake defect prompts nationwide product recall.",
            "published_at": "2024-01-16T09:00:00Z",
            "source": "Reuters",
            "url": "https://reuters.com/recall",
            "shares_sold": None,
            "shares_owned_before": None,
            "transaction_value": None,
            "insider_role": None,
            "is_10b5_1": None,
            "transaction_code": ""
        }
    ])

def test_enrich_events_fallback_workflow(sample_events_df):
    """Verify deterministic fallback workflow when use_llm=False."""
    enriched = enrich_events(sample_events_df, use_llm=False)

    assert len(enriched) == 2

    # Required fields
    required_cols = [
        "event_category", "classification_source", "confidence",
        "context_summary", "severity", "severity_rationale",
        "risk_factors", "supporting_evidence"
    ]
    for col in required_cols:
        assert col in enriched.columns, f"Missing required column: {col}"

    # Backward compatibility fields
    compat_cols = [
        "event_type", "event_classification_confidence", "event_severity",
        "percentage_holdings_sold", "transaction_value", "insider_is_10b5_1",
        "context_flags", "model_text", "event_id", "all_detected_event_types"
    ]
    for col in compat_cols:
        assert col in enriched.columns, f"Missing backward-compatible column: {col}"

    # Record 1: Insider selling fallback
    r1 = enriched.iloc[0]
    assert r1["event_category"] == "insider_selling"
    assert r1["event_type"] == "insider_selling"
    assert r1["classification_source"] == "rule_based_fallback"
    assert r1["context_source"] == "rule_based_fallback"
    assert r1["percentage_holdings_sold"] == 5.0
    assert r1["transaction_value"] == 750000.0
    assert r1["insider_is_10b5_1"] is True
    assert "marked as a 10b5-1 plan" in r1["context_flags"]

    # Record 2: Product recall fallback and preservation of unknowns
    r2 = enriched.iloc[1]
    assert r2["event_category"] == "product_recall"
    assert r2["event_type"] == "product_recall"
    assert r2["severity"] == "medium"
    assert pd.isna(r2["percentage_holdings_sold"])
    assert pd.isna(r2["transaction_value"])
    assert r2["insider_is_10b5_1"] is None

@patch("feature_engineering.gemini_context_analyzer.get_gemini_client")
@patch("feature_engineering.gemini_classifier.get_gemini_client")
def test_enrich_events_gemini_enabled_workflow(mock_class_client_getter, mock_ctx_client_getter, sample_events_df):
    """Verify end-to-end integration when Gemini LLM is enabled."""
    mock_class_client = MagicMock()
    mock_class_resp = MagicMock()
    mock_class_resp.text = """{
        "event_category": "insider_selling",
        "confidence": 0.95,
        "explanation": "Executive executed scheduled sale under Rule 10b5-1.",
        "evidence": ["executed sale of shares pursuant to Rule 10b5-1 trading plan"],
        "secondary_categories": []
    }"""
    mock_class_client.models.generate_content.return_value = mock_class_resp
    mock_class_client_getter.return_value = mock_class_client

    mock_ctx_client = MagicMock()
    mock_ctx_resp = MagicMock()
    mock_ctx_resp.text = """{
        "severity": "low",
        "severity_rationale": "Transaction was conducted under pre-established Rule 10b5-1 schedule.",
        "context_summary": "Routine pre-planned insider trade by CEO.",
        "risk_factors": ["Routine diversification"],
        "supporting_evidence": ["pursuant to Rule 10b5-1 trading plan"],
        "insider_nuance": "CEO executed scheduled transaction."
    }"""
    mock_ctx_client.models.generate_content.return_value = mock_ctx_resp
    mock_ctx_client_getter.return_value = mock_ctx_client

    enriched = enrich_events(sample_events_df.iloc[:1], use_llm=True)
    assert len(enriched) == 1

    r = enriched.iloc[0]
    assert r["event_category"] == "insider_selling"
    assert r["classification_source"] == "gemini_llm"
    assert r["confidence"] == 0.95
    assert r["event_classification_confidence"] == 0.95
    assert r["severity"] == "low"
    assert r["event_severity"] == "low"
    assert r["context_source"] == "gemini_llm"
    assert r["context_summary"] == "Routine pre-planned insider trade by CEO."
    assert "Routine diversification" in r["risk_factors"]
    assert any("Rule 10b5-1 trading plan" in e for e in r["supporting_evidence"])
    assert r["percentage_holdings_sold"] == 5.0
    assert r["transaction_value"] == 750000.0

def test_enrich_events_respects_env_flags(monkeypatch, sample_events_df):
    """Verify that when env flags are disabled, enrich_events defaults to fallback."""
    monkeypatch.setenv("USE_GEMINI_CLASSIFIER", "false")
    monkeypatch.setenv("USE_GEMINI_CONTEXT", "false")

    enriched = enrich_events(sample_events_df.iloc[:1])
    r = enriched.iloc[0]
    assert r["classification_source"] == "rule_based_fallback"
    assert r["context_source"] == "rule_based_fallback"

@patch("feature_engineering.gemini_context_analyzer.get_gemini_client")
@patch("feature_engineering.gemini_classifier.get_gemini_client")
def test_enrich_events_partial_gemini_fallback(mock_class_client_getter, mock_ctx_client_getter, sample_events_df):
    """Verify pipeline resilience when classifier uses Gemini but context analyzer fails."""
    mock_class_client = MagicMock()
    mock_class_resp = MagicMock()
    mock_class_resp.text = """{
        "event_category": "product_recall",
        "confidence": 0.90,
        "explanation": "Safety recall announced.",
        "evidence": ["Safety Recall for 500k Vehicles"],
        "secondary_categories": []
    }"""
    mock_class_client.models.generate_content.return_value = mock_class_resp
    mock_class_client_getter.return_value = mock_class_client

    # Context analyzer raises API failure
    mock_ctx_client = MagicMock()
    mock_ctx_client.models.generate_content.side_effect = Exception("503 Backend unavailable")
    mock_ctx_client_getter.return_value = mock_ctx_client

    enriched = enrich_events(sample_events_df.iloc[1:2], use_llm=True)
    r = enriched.iloc[0]

    assert r["classification_source"] == "gemini_llm"
    assert r["context_source"] == "rule_based_fallback"
    assert r["event_category"] == "product_recall"
    assert r["severity"] == "medium"
    assert "Safety Recall for 500k Vehicles" in r["supporting_evidence"]

def test_enrich_events_empty_dataframe():
    """Verify empty input dataframe returns empty output dataframe."""
    empty_df = pd.DataFrame()
    enriched = enrich_events(empty_df)
    assert enriched.empty

