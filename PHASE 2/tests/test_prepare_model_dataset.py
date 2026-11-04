"""Unit tests for Phase 2 short-interest model dataset preparation."""
import pytest
import pandas as pd
from pathlib import Path
from preprocessing.prepare_model_dataset import (
    build_model_dataset,
    save_model_dataset,
    LEAKAGE_COLUMNS
)

@pytest.fixture
def mock_labeled_events_parquet(tmp_path):
    records = [
        # Train records (2018)
        {
            "event_id": "ev_001",
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "title": "Apple Reports Record Q1 Earnings and Revenue",
            "published_at": "2018-02-01T10:00:00Z",
            "short_interest_before": 50000000.0,
            "short_interest_before_date": "2018-01-15",
            "short_interest_before_dissemination": "2018-01-24",
            "short_interest_before_dissemination_source": "finra_rule_4560_schedule",
            "short_interest_after": 55000000.0,
            "short_interest_after_date": "2018-02-15",
            "short_interest_after_dissemination": "2018-02-27",
            "short_interest_after_dissemination_source": "finra_rule_4560_schedule",
            "short_interest_gap_days": 31,
            "short_interest_change_pct": 10.0,
            "short_interest_label": "increase",
            "phase1_label_short_interest_legacy": "increase"
        },
        # Val records (early 2020)
        {
            "event_id": "ev_002",
            "ticker": "TSLA",
            "company": "Tesla Inc.",
            "title": "Tesla CEO Sells 10,000 Shares of Common Stock",
            "published_at": "2020-01-15T14:00:00Z",
            "short_interest_before": 80000000.0,
            "short_interest_before_date": "2019-12-31",
            "short_interest_before_dissemination": "2020-01-10",
            "short_interest_before_dissemination_source": "finra_rule_4560_schedule",
            "short_interest_after": 82000000.0,
            "short_interest_after_date": "2020-01-31",
            "short_interest_after_dissemination": "2020-02-11",
            "short_interest_after_dissemination_source": "finra_rule_4560_schedule",
            "short_interest_gap_days": 31,
            "short_interest_change_pct": 2.5,
            "short_interest_label": "no_significant_change",
            "phase1_label_short_interest_legacy": "decrease"
        },
        # Test records (mid 2020)
        {
            "event_id": "ev_003",
            "ticker": "NVDA",
            "company": "NVIDIA Corp.",
            "title": "Analyst Downgrades NVIDIA to Neutral on Valuation",
            "published_at": "2020-04-20T09:30:00Z",
            "short_interest_before": 12000000.0,
            "short_interest_before_date": "2020-03-31",
            "short_interest_before_dissemination": "2020-04-09",
            "short_interest_before_dissemination_source": "finra_rule_4560_schedule",
            "short_interest_after": 10000000.0,
            "short_interest_after_date": "2020-04-30",
            "short_interest_after_dissemination": "2020-05-11",
            "short_interest_after_dissemination_source": "finra_rule_4560_schedule",
            "short_interest_gap_days": 30,
            "short_interest_change_pct": -16.67,
            "short_interest_label": "decrease",
            "phase1_label_short_interest_legacy": "decrease"
        }
    ]
    df = pd.DataFrame(records)
    pq_path = tmp_path / "mock_labeled.parquet"
    df.to_parquet(pq_path, index=False)
    return pq_path

def test_build_model_dataset_purges_leakage(mock_labeled_events_parquet):
    """Verify all leakage columns are completely purged from final training data."""
    df, report = build_model_dataset(
        mock_labeled_events_parquet,
        train_end_date="2019-11-30",
        val_end_date="2020-02-29"
    )

    for col in LEAKAGE_COLUMNS:
        assert col not in df.columns, f"Leakage column {col} was found in model dataset!"

    assert report["target_leakage_check"]["is_leakage_free"] == True
    assert len(report["target_leakage_check"]["leakage_columns_remaining_in_features"]) == 0

def test_chronological_splits_integrity(mock_labeled_events_parquet):
    """Verify strict chronological assignment to train, val, test with zero overlap."""
    df, report = build_model_dataset(
        mock_labeled_events_parquet,
        train_end_date="2019-11-30",
        val_end_date="2020-02-29"
    )

    assert "split" in df.columns
    splits = df.set_index("event_id")["split"].to_dict()

    assert splits["ev_001"] == "train"
    assert splits["ev_002"] == "val"
    assert splits["ev_003"] == "test"

    train_dates = pd.to_datetime(df[df["split"] == "train"]["published_at"]).dt.date
    val_dates = pd.to_datetime(df[df["split"] == "val"]["published_at"]).dt.date
    test_dates = pd.to_datetime(df[df["split"] == "test"]["published_at"]).dt.date

    assert train_dates.max() < val_dates.min()
    assert val_dates.max() < test_dates.min()

def test_event_id_and_target_preserved(mock_labeled_events_parquet):
    """Verify original event IDs and target labels are perfectly retained."""
    df, _ = build_model_dataset(mock_labeled_events_parquet)

    assert "event_id" in df.columns
    assert sorted(df["event_id"].tolist()) == ["ev_001", "ev_002", "ev_003"]

    assert "short_interest_label" in df.columns
    labels = df.set_index("event_id")["short_interest_label"].to_dict()
    assert labels["ev_001"] == "increase"
    assert labels["ev_002"] == "no_significant_change"
    assert labels["ev_003"] == "decrease"

def test_candidate_features_populated(mock_labeled_events_parquet):
    """Verify feature pipeline outputs valid candidate features."""
    df, _ = build_model_dataset(mock_labeled_events_parquet)

    required_features = [
        "model_text", "event_type", "event_category", "event_severity",
        "severity", "event_classification_confidence", "confidence",
        "short_interest_before", "short_interest_before_date"
    ]
    for feat in required_features:
        assert feat in df.columns, f"Required feature {feat} missing!"

def test_save_model_dataset_artifacts(mock_labeled_events_parquet, tmp_path):
    """Verify parquet, CSV, and markdown audit report are properly created."""
    df, report = build_model_dataset(mock_labeled_events_parquet)
    out_dir = tmp_path / "model_out"

    pq_p, csv_p, rpt_p = save_model_dataset(df, report, output_dir=out_dir, file_stem="test_dataset")

    assert pq_p.exists()
    assert csv_p.exists()
    assert rpt_p.exists()

    df_loaded = pd.read_parquet(pq_p)
    assert len(df_loaded) == len(df)
    assert "short_interest_label" in df_loaded.columns
    assert "split" in df_loaded.columns

