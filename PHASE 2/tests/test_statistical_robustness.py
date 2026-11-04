"""Tests for Statistical Robustness & Paired Cluster-Bootstrap Evaluation."""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from models.evaluate_statistical_robustness import (
    fast_compute_metrics,
    build_cluster_index_map,
    paired_cluster_bootstrap,
    run_statistical_robustness_analysis,
    DEFAULT_OUTPUT_DIR
)


def test_fast_compute_metrics_matches_sklearn():
    """Verify that vectorized confusion-matrix metrics match scikit-learn exactly."""
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1, 2], size=5000, p=[0.35, 0.45, 0.20])
    y_pred = rng.choice([0, 1, 2], size=5000, p=[0.30, 0.50, 0.20])

    acc, macro_f1, per_class = fast_compute_metrics(y_true, y_pred, n_classes=3)

    sk_acc = accuracy_score(y_true, y_pred)
    sk_macro_f1 = f1_score(y_true, y_pred, average="macro")
    sk_per_class = f1_score(y_true, y_pred, average=None)

    assert np.isclose(acc, sk_acc, atol=1e-6)
    assert np.isclose(macro_f1, sk_macro_f1, atol=1e-6)
    assert np.allclose(per_class, sk_per_class, atol=1e-6)


def test_build_cluster_index_map_properties():
    """Verify cluster mapping creates exhaustive, non-overlapping partitions of row indices."""
    clusters = np.array(["title_A", "title_B", "title_A", "title_C", "title_B", "title_A"])
    unique_clusters, cluster_map = build_cluster_index_map(clusters)

    assert len(unique_clusters) == 3
    assert set(unique_clusters) == {"title_A", "title_B", "title_C"}

    all_indices = np.concatenate(cluster_map)
    assert len(all_indices) == len(clusters)
    assert set(all_indices) == set(range(len(clusters)))

    # Verify cluster memberships
    idx_A = np.where(unique_clusters == "title_A")[0][0]
    assert set(cluster_map[idx_A]) == {0, 2, 5}


def test_paired_cluster_bootstrap_identical_models():
    """When models are identical, difference must be 0 and marked not significant."""
    y_true = np.array([0, 1, 2, 0, 1, 2] * 20)
    y_pred = np.array([0, 1, 1, 0, 1, 0] * 20)
    clusters = np.array([f"headline_{i % 10}" for i in range(len(y_true))])

    res = paired_cluster_bootstrap(
        y_true, y_pred, y_pred, clusters, n_bootstrap=200, random_seed=42
    )

    assert res["accuracy_diff"]["point_estimate"] == 0.0
    assert res["macro_f1_diff"]["point_estimate"] == 0.0
    assert res["accuracy_diff"]["statistically_significant"] is False
    assert res["macro_f1_diff"]["statistically_significant"] is False


def test_paired_cluster_bootstrap_superior_model():
    """When Model A strictly dominates Model B, difference must be positive and significant."""
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1, 2], size=1000)
    # Model A is perfect
    y_pred_a = y_true.copy()
    # Model B has high error rate
    y_pred_b = rng.choice([0, 1, 2], size=1000)
    clusters = np.array([f"head_{i % 100}" for i in range(1000)])

    res = paired_cluster_bootstrap(
        y_true, y_pred_a, y_pred_b, clusters, n_bootstrap=200, random_seed=42
    )

    assert res["accuracy_diff"]["point_estimate"] > 0.3
    assert res["macro_f1_diff"]["point_estimate"] > 0.3
    assert res["accuracy_diff"]["ci_95_lower"] > 0.0
    assert res["accuracy_diff"]["statistically_significant"] is True
    assert res["macro_f1_diff"]["statistically_significant"] is True


def test_statistical_robustness_preserves_existing_artifacts(tmp_path):
    """Verify that running robustness analysis does not overwrite any existing baseline artifacts."""
    existing_artifacts = [
        "short_interest_model.joblib",
        "short_interest_baseline_metrics.json",
        "short_interest_xgboost_model.joblib",
        "short_interest_xgboost_metrics.json",
        "short_interest_ablation_metrics.json"
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
                "ticker": f"TICK{i % 5}",
                "title": f"Headline title {i % 10} for {split}",
                "headline_text": f"Headline text {i % 10} for {split}",
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
                "event_id": f"ev_{split}_{i}",
                "classification_source": "rule_based_fallback",
                "context_source": "rule_based_fallback",
                "short_interest_before_dissemination_source": "official_finra"
            })

    syn_parquet = tmp_path / "synthetic_dataset.parquet"
    pd.DataFrame(rows).to_parquet(syn_parquet)

    custom_out = tmp_path / "artifacts"
    custom_report = tmp_path / "robustness_report.md"

    metrics = run_statistical_robustness_analysis(
        dataset_path=syn_parquet,
        output_dir=custom_out,
        report_path=custom_report,
        n_bootstrap=50,
        random_seed=42
    )

    assert (custom_out / "short_interest_statistical_robustness_metrics.json").exists()
    assert custom_report.exists()
    assert "comparisons" in metrics
    assert "subgroup_robustness" in metrics

    # Verify existing artifacts were not touched
    for art, mt in mtimes.items():
        p = DEFAULT_OUTPUT_DIR / art
        assert p.stat().st_mtime == mt, f"Artifact {art} was modified by robustness analysis!"

