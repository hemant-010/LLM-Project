"""Tests for Controlled Feature-Ablation Study: Feature Definition Integrity & Artifact Isolation."""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from models.run_feature_ablation import (
    CONFIGURATIONS,
    build_ablation_preprocessor,
    run_ablation_experiments,
    DEFAULT_OUTPUT_DIR
)


def test_ablation_configurations_feature_definitions():
    """Verify that all 6 ablation configurations conform strictly to requirements and contain no leakage."""
    assert len(CONFIGURATIONS) == 6, f"Expected 6 configurations, found {len(CONFIGURATIONS)}"

    prohibited_cols = {
        "event_id", "ticker", "company", "date", "published_at", "news_date", "published_date",
        "short_interest_after", "short_interest_delta", "settlement_date_after", "dissemination_date_after", "split"
    }

    for cfg_key, cfg in CONFIGURATIONS.items():
        all_cols = set(cfg["text_cols"] + cfg["cat_cols"] + cfg["num_cols"])
        leakage = all_cols.intersection(prohibited_cols)
        assert len(leakage) == 0, f"Configuration {cfg_key} contains prohibited columns: {leakage}"

    # Config 1: Structured only (no text)
    assert len(CONFIGURATIONS["1_structured_only"]["text_cols"]) == 0
    assert len(CONFIGURATIONS["1_structured_only"]["cat_cols"]) == 6
    assert len(CONFIGURATIONS["1_structured_only"]["num_cols"]) == 6

    # Config 2: Headline only (TF-IDF, no structured)
    assert CONFIGURATIONS["2_headline_only"]["text_cols"] == ["headline_text"]
    assert len(CONFIGURATIONS["2_headline_only"]["cat_cols"]) == 0
    assert len(CONFIGURATIONS["2_headline_only"]["num_cols"]) == 0

    # Config 3: Headline + Event classification
    assert CONFIGURATIONS["3_headline_plus_event_classification"]["text_cols"] == ["headline_text"]
    assert set(CONFIGURATIONS["3_headline_plus_event_classification"]["cat_cols"]) == {"event_type", "event_category", "event_severity"}
    assert CONFIGURATIONS["3_headline_plus_event_classification"]["num_cols"] == ["event_classification_confidence"]

    # Config 4: Headline + Insider transactions
    assert CONFIGURATIONS["4_headline_plus_insider_transactions"]["text_cols"] == ["headline_text"]
    assert set(CONFIGURATIONS["4_headline_plus_insider_transactions"]["cat_cols"]) == {"insider_role", "transaction_code"}
    assert set(CONFIGURATIONS["4_headline_plus_insider_transactions"]["num_cols"]) == {"percentage_holdings_sold", "transaction_value"}

    # Config 5: Headline + Baseline
    assert CONFIGURATIONS["5_headline_plus_baseline"]["text_cols"] == ["headline_text"]
    assert len(CONFIGURATIONS["5_headline_plus_baseline"]["cat_cols"]) == 0
    assert CONFIGURATIONS["5_headline_plus_baseline"]["num_cols"] == ["short_interest_before"]

    # Config 6: Full feature set
    assert CONFIGURATIONS["6_full_feature_set"]["text_cols"] == ["headline_text"]
    assert len(CONFIGURATIONS["6_full_feature_set"]["cat_cols"]) == 6
    assert len(CONFIGURATIONS["6_full_feature_set"]["num_cols"]) == 6


def test_ablation_preprocessors_train_only_fitting():
    """Verify preprocessors for all 6 configurations fit on train and transform unseen test data."""
    train_df = pd.DataFrame({
        "headline_text": ["Apple reports strong quarterly earnings", "FDA approves novel therapy", "CEO sells shares"],
        "event_type": ["positive_earnings", "clinical_trial", "insider_selling"],
        "event_category": ["earnings", "regulatory", "corporate"],
        "event_severity": ["high", "medium", "low"],
        "insider_role": ["None", "None", "CEO"],
        "transaction_code": ["None", "None", "S"],
        "is_multi_ticker": [False, False, True],
        "percentage_holdings_sold": [0.0, np.nan, 15.0],
        "transaction_value": [0.0, np.nan, 500000.0],
        "event_classification_confidence": [0.90, 0.85, 0.70],
        "n_tokens": [10, 8, 12],
        "headline_ticker_count": [1, 1, 2],
        "short_interest_before": [1000000.0, 500000.0, 750000.0]
    })

    test_df = pd.DataFrame({
        "headline_text": ["Unknown company faces regulatory investigation"],
        "event_type": ["investigation"],
        "event_category": ["legal"],
        "event_severity": ["critical"],
        "insider_role": ["CFO"],
        "transaction_code": ["P"],
        "is_multi_ticker": [False],
        "percentage_holdings_sold": [np.nan],
        "transaction_value": [np.nan],
        "event_classification_confidence": [0.65],
        "n_tokens": [11],
        "headline_ticker_count": [1],
        "short_interest_before": [300000.0]
    })

    for cfg_key, cfg in CONFIGURATIONS.items():
        preprocessor = build_ablation_preprocessor(
            text_cols=cfg["text_cols"],
            cat_cols=cfg["cat_cols"],
            num_cols=cfg["num_cols"],
            max_text_features=100,
            min_df=1
        )
        cols_needed = cfg["text_cols"] + cfg["cat_cols"] + cfg["num_cols"]
        X_tr = preprocessor.fit_transform(train_df[cols_needed])
        X_te = preprocessor.transform(test_df[cols_needed])

        assert X_tr.shape[0] == 3
        assert X_te.shape[0] == 1
        assert X_tr.shape[1] == X_te.shape[1]


def test_ablation_preserves_baseline_and_xgboost_artifacts(tmp_path):
    """Verify that running ablation study does not touch baseline or XGBoost artifacts."""
    lr_joblib = DEFAULT_OUTPUT_DIR / "short_interest_model.joblib"
    lr_metrics = DEFAULT_OUTPUT_DIR / "short_interest_baseline_metrics.json"
    xgb_joblib = DEFAULT_OUTPUT_DIR / "short_interest_xgboost_model.joblib"
    xgb_metrics = DEFAULT_OUTPUT_DIR / "short_interest_xgboost_metrics.json"

    mtimes = {
        p: p.stat().st_mtime for p in [lr_joblib, lr_metrics, xgb_joblib, xgb_metrics] if p.exists()
    }

    # Synthetic dataset
    rows = []
    for split in ["train", "val", "test"]:
        n = 20 if split == "train" else 10
        for i in range(n):
            rows.append({
                "split": split,
                "title": f"Title headline {i} for {split}",
                "headline_text": f"Headline text {i} for {split}",
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
                "short_interest_label": "increase" if i % 3 == 0 else ("decrease" if i % 3 == 1 else "no_significant_change"),
                "event_id": f"ev_{split}_{i}"
            })

    syn_parquet = tmp_path / "synthetic_dataset.parquet"
    pd.DataFrame(rows).to_parquet(syn_parquet)

    custom_out = tmp_path / "artifacts"
    custom_report = tmp_path / "ablation_report.md"

    res = run_ablation_experiments(
        dataset_path=syn_parquet,
        output_dir=custom_out,
        report_path=custom_report,
        random_seed=42
    )

    assert (custom_out / "short_interest_ablation_metrics.json").exists()
    assert custom_report.exists()
    assert len(res["ablation_results"]) == 6

    # Verify existing artifacts were untouched
    for p, mt in mtimes.items():
        assert p.stat().st_mtime == mt, f"Artifact {p.name} was modified by ablation experiment!"
