import pytest
from pathlib import Path
from models.predictor import DualPredictor

def test_predictor_not_trained_behavior(tmp_path):
    """Verify that pointing to an empty model directory returns status='not_trained'."""
    empty_model_dir = tmp_path / "no_models"
    empty_model_dir.mkdir(parents=True, exist_ok=True)

    predictor = DualPredictor(model_dir=empty_model_dir)
    dummy_row = {
        "model_text": "CEO sold 10,000 shares",
        "event_type": "insider_selling",
        "insider_role": "CEO",
        "transaction_code": "S",
        "event_severity": "high",
        "percentage_holdings_sold": 20.0,
        "transaction_value": 1500000.0,
        "event_classification_confidence": 0.85
    }

    preds = predictor.predict(dummy_row)
    si_pred = preds["short_interest"]
    sp_pred = preds["stock_price"]

    # Verify short-interest status
    assert si_pred["status"] == "not_trained"
    assert si_pred["direction"] is None
    assert si_pred["probabilities"] is None
    assert "Model artifact not found" in si_pred["message"]

    # Verify stock-price status
    assert sp_pred["status"] == "not_trained"
    assert sp_pred["direction"] is None
    assert sp_pred["probabilities"] is None
    assert "Model artifact not found" in sp_pred["message"]

