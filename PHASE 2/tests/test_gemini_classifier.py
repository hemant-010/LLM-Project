import pytest
from unittest.mock import MagicMock, patch
from google.genai.errors import APIError
from feature_engineering.event_classifier import classify_event
from feature_engineering.gemini_classifier import (
    classify_event_with_gemini,
    verify_evidence_substrings
)

def test_gemini_disabled_by_default():
    """Verify Gemini is disabled by default and uses rule-based classification."""
    res = classify_event("CEO sold 5,000 shares of common stock")
    assert res.source == "rule_based_fallback"
    assert res.event_type == "insider_selling"
    assert res.is_calibrated is False

    # Verify 3-tuple backward compatibility
    category, confidence, matches = res
    assert category == "insider_selling"
    assert isinstance(confidence, float)
    assert "insider_selling" in matches

def test_evidence_substring_verification():
    full_text = "Federal prosecutors initiated a criminal probe into accounting irregularities at the firm."

    # Exact substring
    verified = verify_evidence_substrings(["criminal probe", "accounting irregularities"], full_text)
    assert verified == ["criminal probe", "accounting irregularities"]

    # Non-existent hallucinated phrase
    rejected = verify_evidence_substrings(["bribe paid to offshore accounts"], full_text)
    assert rejected == []

    # Mixed
    mixed = verify_evidence_substrings(["criminal probe", "CEO was arrested"], full_text)
    assert mixed == ["criminal probe"]

@patch("feature_engineering.gemini_classifier.get_gemini_client")
def test_gemini_mocked_success(mock_get_client):
    """Mock Gemini SDK returning valid structured JSON with exact substring evidence."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """{
        "event_category": "fraud_investigation",
        "confidence": 0.92,
        "explanation": "DOJ initiated criminal probe into accounting practices.",
        "evidence": ["criminal probe", "accounting practices"],
        "secondary_categories": []
    }"""
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    title = "DOJ announces criminal probe into firm"
    desc = "Investigation focuses on accounting practices."

    res = classify_event(title, desc, use_llm=True)
    assert res.source == "gemini_llm"
    assert res.event_type == "fraud_investigation"
    assert res.confidence == 0.92
    assert "criminal probe" in res.evidence
    assert res.is_calibrated is False

@patch("feature_engineering.gemini_classifier.get_gemini_client")
def test_gemini_rejects_hallucinated_evidence(mock_get_client):
    """If Gemini provides evidence not found in text, reject and fall back to rule-based."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Evidence text NOT in the input article
    mock_response.text = """{
        "event_category": "product_recall",
        "confidence": 0.88,
        "explanation": "Defective battery caused recall.",
        "evidence": ["spontaneous battery combustion caused massive recall"],
        "secondary_categories": []
    }"""
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    title = "Company updates warranty terms for electronics"
    desc = "New terms apply starting next month."

    # Because evidence is not a substring, gemini_classifier rejects it and falls back to regex
    res = classify_event(title, desc, use_llm=True)
    assert res.source == "rule_based_fallback"

def test_gemini_fallback_on_missing_api_key(monkeypatch):
    """Verify clean fallback to regex when GEMINI_API_KEY is not set."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    res = classify_event("Analyst downgrades stock to sell", use_llm=True)
    assert res.source == "rule_based_fallback"
    assert res.event_type == "analyst_downgrade"

@patch("feature_engineering.gemini_classifier.get_gemini_client")
def test_gemini_fallback_on_api_error(mock_get_client):
    """Verify clean fallback when Gemini API raises an error or rate limit."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 ResourceExhausted: rate limit exceeded")
    mock_get_client.return_value = mock_client

    res = classify_event("Company files for Chapter 11 bankruptcy", use_llm=True)
    assert res.source == "rule_based_fallback"
    assert res.event_type == "bankruptcy_distress"

