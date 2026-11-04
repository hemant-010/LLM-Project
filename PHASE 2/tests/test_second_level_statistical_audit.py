"""Tests for Second-Level Statistical Audit: Holm Correction, Event Reconciliation & Artifact Isolation."""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from models.evaluate_second_level_statistical_audit import (
    holm_bonferroni_correction,
    run_second_level_audit,
    DEFAULT_OUTPUT_DIR
)


def test_holm_bonferroni_correction_calculation():
    """Verify Holm-Bonferroni step-down adjustment against manual calculation."""
    # Test case with 4 p-values
    # Sorted: p = [0.01, 0.02, 0.04, 0.10]
    # m = 4
    # rank 1: 4 * 0.01 = 0.04
    # rank 2: 3 * 0.02 = 0.06
    # rank 3: 2 * 0.04 = 0.08
    # rank 4: 1 * 0.10 = 0.10
    raw_p = [0.04, 0.01, 0.10, 0.02]
    expected_adj = [0.08, 0.04, 0.10, 0.06]

    res = holm_bonferroni_correction(raw_p)
    np.testing.assert_allclose(res, expected_adj, atol=1e-4)

    # Edge cases
    assert holm_bonferroni_correction([]) == []
    assert holm_bonferroni_correction([0.03]) == [0.03]
    # Cap at 1.0
    assert holm_bonferroni_correction([0.6, 0.7]) == [1.0, 1.0]


def test_event_count_reconciliation_logic():
    """Verify reconciliation of single-ticker vs novel corporate catalyst filters."""
    train_df = pd.DataFrame({
        "title": ["Apple Reports Earnings", "Syndicated Roundup Friday"],
        "is_multi_ticker": [False, True],
        "headline_ticker_count": [1, 5]
    })

    test_df = pd.DataFrame({
        "title": [
            "Apple Reports Earnings",       # Single ticker, but seen in train
            "Tesla Unveils New Vehicle",    # Single ticker, novel
            "Pfizer Receives Approval",     # Single ticker, novel
            "Syndicated Roundup Friday",    # Multi ticker, seen in train
            "Midday Market Mover Wrap"      # Multi ticker, novel
        ],
        "is_multi_ticker": [False, False, False, True, True],
        "headline_ticker_count": [1, 1, 1, 4, 3]
    })

    single_mask = (test_df["is_multi_ticker"] == False) & (test_df["headline_ticker_count"] <= 1)
    train_titles = set(train_df["title"].dropna().unique())
    novel_mask = single_mask & (~test_df["title"].isin(train_titles))

    n_single = single_mask.sum()
    n_novel = novel_mask.sum()
    n_diff = n_single - n_novel

    assert n_single == 3
    assert n_novel == 2
    assert n_diff == 1
    # Exactly one event (Apple Reports Earnings) is in the difference
    assert test_df[single_mask & (~novel_mask)]["title"].iloc[0] == "Apple Reports Earnings"


def test_second_level_audit_preserves_all_existing_artifacts(tmp_path):
    """Verify that second-level audit creates separate artifacts and never touches existing ones."""
    existing_artifacts = [
        "short_interest_model.joblib",
        "short_interest_baseline_metrics.json",
        "short_interest_xgboost_model.joblib",
        "short_interest_xgboost_metrics.json",
        "short_interest_ablation_metrics.json",
        "short_interest_statistical_robustness_metrics.json"
    ]

    mtimes = {}
    for art in existing_artifacts:
        p = DEFAULT_OUTPUT_DIR / art
        if p.exists():
            mtimes[art] = p.stat().st_mtime

    # Generate synthetic parquet dataset
    rows = []
    for split in ["train", "val", "test"]:
        n = 30 if split == "train" else 20
        for i in range(n):
            rows.append({
                "split": split,
                "ticker": f"TICK{i % 4}",
                "title": f"Headline title {i % 8} for {split}",
                "headline_text": f"Headline text {i % 8} for {split}",
                "event_type": "insider_selling" if i % 2 == 0 else "positive_earnings",
                "event_category": "corporate" if i % 2 == 0 else "earnings",
                "event_severity": "high" if i % 2 == 0 else "low",
                "insider_role": "CEO" if i % 2 == 0 else "Director",
                "transaction_code": "S" if i % 2 == 0 else "P",
                "is_multi_ticker": (i % 3 == 0),
                "percentage_holdings_sold": 5.0,
                "transaction_value": 50000.0,
                "event_classification_confidence": 0.8,
                "n_tokens": 10,
                "headline_ticker_count": 3 if (i % 3 == 0) else 1,
                "short_interest_before": 100000.0 * (i + 1),
                "short_interest_label": "increase" if i % 3 == 0 else ("decrease" if i % 3 == 1 else "no_significant_change"),
                "event_id": f"ev_{split}_{i}",
            })

    syn_parquet = tmp_path / "synthetic_dataset.parquet"
    pd.DataFrame(rows).to_parquet(syn_parquet)

    custom_out = tmp_path / "artifacts"
    custom_report = tmp_path / "second_level_report.md"

    metrics = run_second_level_audit(
        dataset_path=syn_parquet,
        output_dir=custom_out,
        report_path=custom_report,
        n_bootstrap=50,
        random_seed=42
    )

    assert (custom_out / "short_interest_second_level_statistical_audit_metrics.json").exists()
    assert custom_report.exists()
    assert "primary_multiple_testing_correction" in metrics
    assert "top_20_ticker_investigation" in metrics
    assert "event_count_reconciliation" in metrics

    # Verify existing artifacts were untouched
    for art, mt in mtimes.items():
        p = DEFAULT_OUTPUT_DIR / art
        assert p.stat().st_mtime == mt, f"Artifact {art} was modified by second-level audit!"

