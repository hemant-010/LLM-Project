import pytest
import pandas as pd
import joblib
from pathlib import Path
from models.train_models import main
from config import MODEL_DIR

def create_synthetic_training_dataset(path: Path, n_rows: int = 40, mixed_horizons: bool = True):
    rows = []
    for i in range(n_rows):
        htype = ("5_sessions_full_exposure" if i < n_rows // 2 else "5_sessions_post_close_drift") if mixed_horizons else "5_sessions_full_exposure"
        rows.append({
            "model_text": f"Insider transaction filing corporate update sequence {i}",
            "event_type": "insider_selling" if i % 2 == 0 else "positive_earnings",
            "insider_role": "CEO" if i % 2 == 0 else "Director",
            "transaction_code": "S",
            "event_severity": "high" if i % 2 == 0 else "low",
            "percentage_holdings_sold": 15.0 if i % 2 == 0 else 2.0,
            "transaction_value": 500000.0,
            "event_classification_confidence": 0.80,
            "published_at": f"2024-01-{i%25 + 1:02d} 10:00:00+00:00",
            "short_interest_label": "increase" if i % 3 == 0 else ("decrease" if i % 3 == 1 else "no_significant_change"),
            "stock_price_label": "decrease" if i % 3 == 0 else ("increase" if i % 3 == 1 else "no_significant_change"),
            "horizon_type": htype
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df

def test_mixed_horizons_rejected_by_default(tmp_path):
    csv_path = tmp_path / "mixed_train.csv"
    create_synthetic_training_dataset(csv_path, n_rows=40, mixed_horizons=True)

    with pytest.raises(ValueError, match="Incompatible stock-price horizons detected"):
        main(csv_path, min_rows=15, price_horizon=None, allow_mixed_price_horizons=False, output_dir=tmp_path)

def test_explicit_horizon_training_succeeds(tmp_path):
    test_model_dir = tmp_path / "models"
    test_model_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tmp_path / "mixed_train.csv"
    create_synthetic_training_dataset(csv_path, n_rows=40, mixed_horizons=True)

    main(csv_path, min_rows=15, price_horizon="5_sessions_full_exposure", output_dir=test_model_dir)
    sp_artifact = joblib.load(test_model_dir / "stock_price_model.joblib")
    si_artifact = joblib.load(test_model_dir / "short_interest_model.joblib")

    assert sp_artifact["horizon"] == "five trading sessions (5_sessions_full_exposure)"
    assert si_artifact["horizon"] == "next eligible short-interest reporting period"

