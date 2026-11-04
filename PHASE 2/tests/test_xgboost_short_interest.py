"""Tests for XGBoost Short-Interest Model: Leakage Prevention and Artifact Isolation."""
import json
import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from models.train_xgboost_short_interest import (
    STRUCTURED_FEATURE_ALLOWLIST,
    STRUCTURED_CATEGORICAL_FEATURES,
    STRUCTURED_NUMERICAL_FEATURES,
    TARGET_COLUMN,
    TARGET_CLASSES,
    build_structured_preprocessor,
    StructuredXGBoostPipeline,
    train_and_evaluate_xgboost,
    DEFAULT_OUTPUT_DIR
)


def test_structured_feature_allowlist_excludes_leakage_and_identifiers():
    """Verify that feature allowlist excludes text, identifiers, dates, and future outcomes."""
    prohibited_features = [
        # Text features
        "headline_text", "model_text", "title",
        # Identifiers
        "event_id", "ticker", "company",
        # Dates and Timestamps
        "date", "published_at", "news_date", "published_date",
        # Target and future outcomes
        "short_interest_label", "short_interest_after", "short_interest_delta",
        "settlement_date_after", "dissemination_date_after", "split"
    ]

    for col in prohibited_features:
        assert col not in STRUCTURED_FEATURE_ALLOWLIST, f"Prohibited column {col} found in XGBoost feature allowlist!"

    # Ensure all allowlisted features are classified as either categorical or numerical
    assert set(STRUCTURED_FEATURE_ALLOWLIST) == set(STRUCTURED_CATEGORICAL_FEATURES + STRUCTURED_NUMERICAL_FEATURES)


def test_preprocessor_fitting_and_transformation_isolation():
    """Verify preprocessor fits on train-only data and transforms test data cleanly."""
    train_data = pd.DataFrame({
        "event_type": ["insider_selling", "positive_earnings", "clinical_trial"],
        "event_category": ["corporate", "earnings", "regulatory"],
        "event_severity": ["high", "low", "medium"],
        "insider_role": ["CEO", "Director", "None"],
        "transaction_code": ["S", "P", "None"],
        "is_multi_ticker": [False, True, False],
        "percentage_holdings_sold": [12.5, np.nan, 0.0],
        "transaction_value": [100000.0, np.nan, 0.0],
        "event_classification_confidence": [0.85, 0.90, 0.75],
        "n_tokens": [10, 15, 8],
        "headline_ticker_count": [1, 2, 1],
        "short_interest_before": [500000.0, 1200000.0, 300000.0]
    })

    test_data = pd.DataFrame({
        "event_type": ["unknown_new_event"],  # Unseen category
        "event_category": ["corporate"],
        "event_severity": ["high"],
        "insider_role": ["UnknownRole"],      # Unseen category
        "transaction_code": ["X"],            # Unseen category
        "is_multi_ticker": [True],
        "percentage_holdings_sold": [np.nan],
        "transaction_value": [np.nan],
        "event_classification_confidence": [0.80],
        "n_tokens": [12],
        "headline_ticker_count": [1],
        "short_interest_before": [450000.0]
    })

    preprocessor = build_structured_preprocessor()
    X_train = preprocessor.fit_transform(train_data[STRUCTURED_FEATURE_ALLOWLIST])
    assert X_train.shape[0] == 3
    assert not np.isnan(X_train).any(), "NaN values detected after train preprocessing!"

    # Transform unseen test data without refitting
    X_test = preprocessor.transform(test_data[STRUCTURED_FEATURE_ALLOWLIST])
    assert X_test.shape[0] == 1
    assert X_test.shape[1] == X_train.shape[1]
    assert not np.isnan(X_test).any(), "NaN values detected after test preprocessing!"


def test_baseline_artifacts_preserved_during_xgboost_training(tmp_path):
    """Verify that training XGBoost does not overwrite Logistic Regression artifacts."""
    baseline_joblib = DEFAULT_OUTPUT_DIR / "short_interest_model.joblib"
    baseline_metrics = DEFAULT_OUTPUT_DIR / "short_interest_baseline_metrics.json"

    # Record baseline state if present
    baseline_joblib_mtime = baseline_joblib.stat().st_mtime if baseline_joblib.exists() else None
    baseline_metrics_mtime = baseline_metrics.stat().st_mtime if baseline_metrics.exists() else None

    # Create synthetic dataset with train/val/test splits
    rows = []
    for split in ["train", "val", "test"]:
        n = 30 if split == "train" else 15
        for i in range(n):
            rows.append({
                "split": split,
                "title": f"Headline title {i} for {split}",
                "event_type": "insider_selling" if i % 2 == 0 else "positive_earnings",
                "event_category": "corporate" if i % 2 == 0 else "earnings",
                "event_severity": "high" if i % 2 == 0 else "low",
                "insider_role": "CEO" if i % 2 == 0 else "Director",
                "transaction_code": "S" if i % 2 == 0 else "P",
                "is_multi_ticker": False,
                "percentage_holdings_sold": 5.0,
                "transaction_value": 50000.0,
                "event_classification_confidence": 0.8,
                "n_tokens": 10,
                "headline_ticker_count": 1,
                "short_interest_before": 100000.0,
                "short_interest_label": "increase" if i % 3 == 0 else ("decrease" if i % 3 == 1 else "no_significant_change")
            })

    syn_df = pd.DataFrame(rows)
    syn_parquet = tmp_path / "synthetic_dataset.parquet"
    syn_df.to_parquet(syn_parquet)

    custom_out = tmp_path / "artifacts"
    custom_report = tmp_path / "xgboost_report.md"

    # Run training targeting custom output directory
    metrics = train_and_evaluate_xgboost(
        dataset_path=syn_parquet,
        output_dir=custom_out,
        report_path=custom_report,
        random_seed=42
    )

    # Verify custom artifacts were created
    assert (custom_out / "short_interest_xgboost_model.joblib").exists()
    assert (custom_out / "short_interest_xgboost_metrics.json").exists()
    assert custom_report.exists()

    # Verify baseline production artifacts were untouched
    if baseline_joblib_mtime is not None:
        assert baseline_joblib.stat().st_mtime == baseline_joblib_mtime, "Baseline joblib was modified!"
    if baseline_metrics_mtime is not None:
        assert baseline_metrics.stat().st_mtime == baseline_metrics_mtime, "Baseline metrics JSON was modified!"


def test_structured_xgboost_pipeline_end_to_end_inference(tmp_path):
    """Verify inference pipeline can be pickled/unpickled and outputs valid string predictions."""
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    train_data = pd.DataFrame({
        "event_type": ["insider_selling", "positive_earnings", "clinical_trial"] * 5,
        "event_category": ["corporate", "earnings", "regulatory"] * 5,
        "event_severity": ["high", "low", "medium"] * 5,
        "insider_role": ["CEO", "Director", "None"] * 5,
        "transaction_code": ["S", "P", "None"] * 5,
        "is_multi_ticker": [False, True, False] * 5,
        "percentage_holdings_sold": [12.5, 0.0, 0.0] * 5,
        "transaction_value": [100000.0, 0.0, 0.0] * 5,
        "event_classification_confidence": [0.85, 0.90, 0.75] * 5,
        "n_tokens": [10, 15, 8] * 5,
        "headline_ticker_count": [1, 2, 1] * 5,
        "short_interest_before": [500000.0, 1200000.0, 300000.0] * 5,
        "short_interest_label": ["increase", "decrease", "no_significant_change"] * 5
    })

    prep = build_structured_preprocessor()
    X_mat = prep.fit_transform(train_data[STRUCTURED_FEATURE_ALLOWLIST])

    le = LabelEncoder()
    y_enc = le.fit_transform(train_data["short_interest_label"])

    model = xgb.XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X_mat, y_enc)

    pipeline = StructuredXGBoostPipeline(preprocessor=prep, label_encoder=le, model=model)

    # Test prediction
    preds = pipeline.predict(train_data)
    assert len(preds) == len(train_data)
    assert set(preds).issubset(set(TARGET_CLASSES))

    # Test probability prediction
    proba = pipeline.predict_proba(train_data)
    assert proba.shape == (len(train_data), len(TARGET_CLASSES))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    # Test serialization and reload
    art_path = tmp_path / "test_xgb.joblib"
    joblib.dump(pipeline, art_path)
    loaded = joblib.load(art_path)
    preds_loaded = loaded.predict(train_data)
    np.testing.assert_array_equal(preds, preds_loaded)

