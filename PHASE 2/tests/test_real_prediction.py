import pytest
from models.predictor import DualPredictor

def test_raw_headline_dict_prediction():
    """Verify that a raw headline dictionary without categorical flags does not raise ValueError."""
    predictor = DualPredictor()
    raw_dict = {
        "title": "Federal Trade Commission launches formal antitrust probe into tech conglomerate",
        "ticker": "AMZN"
    }
    res = predictor.predict(raw_dict)
    assert "short_interest" in res
    assert res["short_interest"]["status"] == "predicted"
    assert res["short_interest"]["direction"] in {"increase", "decrease", "no_significant_change"}
    assert "probabilities" in res["short_interest"]

def test_xgboost_dual_predictor_prediction():
    """Verify that XGBoost model can be selected and executed via DualPredictor."""
    predictor = DualPredictor()
    raw_dict = {
        "title": "Automaker recalls 800,000 vehicles due to critical steering mechanism defect",
        "ticker": "GM",
        "event_type": "product_recall",
        "event_severity": "high",
        "event_classification_confidence": 0.90
    }
    res = predictor.predict(raw_dict, model_family="xgboost")
    assert res["short_interest"]["status"] == "predicted"
    assert res["short_interest"]["model_type"] == "XGBoost"
    assert res["short_interest"]["direction"] == "increase"
    assert res["short_interest"]["probabilities"]["increase"] > 0.50

def test_predict_event_end_to_end_convenience():
    """Verify end-to-end event enrichment + prediction on a raw headline."""
    predictor = DualPredictor()
    out = predictor.predict_event(
        title="FDA rejects novel oncology drug application citing trial data anomalies",
        ticker="BIIB",
        description="Complete response letter highlights severe safety concerns",
        use_llm=False,
        model_family="logistic_regression"
    )
    assert "event" in out
    assert "predictions" in out
    assert out["event"]["ticker"] == "BIIB"
    assert out["predictions"]["short_interest"]["status"] == "predicted"
    assert out["predictions"]["short_interest"]["direction"] in {"increase", "decrease", "no_significant_change"}
