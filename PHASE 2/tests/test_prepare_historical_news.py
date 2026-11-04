"""Unit tests for Phase 2 historical news dataset preparation."""
import pytest
import pandas as pd
from pathlib import Path
from preprocessing.prepare_historical_news import (
    extract_historical_news,
    save_extracted_dataset
)

@pytest.fixture
def mock_phase1_parquet(tmp_path):
    """Create a small mock Phase 1 parquet dataset for deterministic testing."""
    records = [
        # Out-of-bounds early date (2017-12-28)
        {
            "title": "Early News Headline",
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "news_date": 20171228,
            "n_tokens": 10,
            "label_regsho_volume": "decrease",
            "label_short_interest": None
        },
        # In-bounds start date (2017-12-29) - Syndicated across two tickers
        {
            "title": "Market Wrap: Tech Stocks Rally",
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "news_date": 20171229,
            "n_tokens": 12,
            "label_regsho_volume": "increase",
            "label_short_interest": "increase"
        },
        {
            "title": "Market Wrap: Tech Stocks Rally",
            "ticker": "MSFT",
            "company": "Microsoft Corp.",
            "news_date": 20171229,
            "n_tokens": 12,
            "label_regsho_volume": "increase",
            "label_short_interest": "decrease"
        },
        # In-bounds single-ticker news (2019-05-15)
        {
            "title": "Tesla CEO Buys 10,000 Shares of Common Stock",
            "ticker": "TSLA",
            "company": "Tesla Inc.",
            "news_date": 20190515,
            "n_tokens": 15,
            "label_regsho_volume": "decrease",
            "label_short_interest": "increase"
        },
        # In-bounds end date (2020-06-11)
        {
            "title": "Final Boundary News Event",
            "ticker": "NVDA",
            "company": "NVIDIA Corp.",
            "news_date": 20200611,
            "n_tokens": 8,
            "label_regsho_volume": "increase",
            "label_short_interest": "decrease"
        },
        # Out-of-bounds late date (2020-06-12)
        {
            "title": "Late News Event After Cutoff",
            "ticker": "AMZN",
            "company": "Amazon.com Inc.",
            "news_date": 20200612,
            "n_tokens": 9,
            "label_regsho_volume": "decrease",
            "label_short_interest": None
        }
    ]
    df = pd.DataFrame(records)
    parquet_path = tmp_path / "sitp_mock.parquet"
    df.to_parquet(parquet_path, index=False)
    return parquet_path

def test_extract_historical_news_date_filtering(mock_phase1_parquet):
    """Verify that records outside the start/end date range are strictly excluded."""
    df, report = extract_historical_news(
        source_path=mock_phase1_parquet,
        start_date="2017-12-29",
        end_date="2020-06-11"
    )

    # 4 of 6 records are within the [2017-12-29, 2020-06-11] window
    assert len(df) == 4
    assert report["total_rows"] == 4
    assert report["date_min"] == "2017-12-29"
    assert report["date_max"] == "2020-06-11"
    assert "2017-12-28" not in df["date"].values
    assert "2020-06-12" not in df["date"].values

def test_date_only_semantics_preserved(mock_phase1_parquet):
    """Verify parsed date columns strictly preserve date-only format without invented times."""
    df, _ = extract_historical_news(mock_phase1_parquet)

    for col in ["date", "published_date", "published_at"]:
        assert col in df.columns
        for val in df[col]:
            assert isinstance(val, str)
            # Must strictly match YYYY-MM-DD
            assert len(val) == 10
            assert val[4] == "-" and val[7] == "-"
            assert ":" not in val  # No invented time

def test_unique_event_id_generation(mock_phase1_parquet):
    """Verify every extracted record receives a non-empty, unique event_id."""
    df, _ = extract_historical_news(mock_phase1_parquet)

    assert "event_id" in df.columns
    assert df["event_id"].nunique() == len(df)
    assert all(eid.startswith("ev_p1_") for eid in df["event_id"])

def test_multi_ticker_headline_identification(mock_phase1_parquet):
    """Verify syndication tracking flags multi-ticker headlines and preserves counts."""
    df, report = extract_historical_news(mock_phase1_parquet)

    # 'Market Wrap: Tech Stocks Rally' appears on AAPL and MSFT
    rally_rows = df[df["title"] == "Market Wrap: Tech Stocks Rally"]
    assert len(rally_rows) == 2
    assert all(rally_rows["is_multi_ticker"] == True)
    assert all(rally_rows["headline_ticker_count"] == 2)

    # 'Tesla CEO Buys 10,000 Shares...' is unique to TSLA
    tsla_rows = df[df["ticker"] == "TSLA"]
    assert len(tsla_rows) == 1
    assert tsla_rows.iloc[0]["is_multi_ticker"] == False
    assert tsla_rows.iloc[0]["headline_ticker_count"] == 1

    assert report["multi_ticker_headlines_count"] == 1
    assert report["multi_ticker_records_count"] == 2

def test_legacy_labels_segregated_from_phase2_ground_truth(mock_phase1_parquet):
    """Verify legacy Phase 1 labels are renamed and not exposed as Phase 2 targets."""
    df, _ = extract_historical_news(mock_phase1_parquet)

    # Main label columns must NOT exist
    assert "label_short_interest" not in df.columns
    assert "label_regsho_volume" not in df.columns
    assert "short_interest_label" not in df.columns

    # Legacy reference columns preserved
    assert "phase1_label_short_interest_legacy" in df.columns
    assert "phase1_label_regsho_volume_legacy" in df.columns

def test_save_extracted_dataset(mock_phase1_parquet, tmp_path):
    """Verify extracted dataset saves valid parquet and CSV artifacts."""
    df, _ = extract_historical_news(mock_phase1_parquet)
    out_dir = tmp_path / "processed"

    p_path, c_path = save_extracted_dataset(df, output_dir=out_dir, file_stem="test_news")

    assert p_path.exists()
    assert c_path.exists()

    df_loaded_parquet = pd.read_parquet(p_path)
    assert len(df_loaded_parquet) == len(df)
    assert "event_id" in df_loaded_parquet.columns

    df_loaded_csv = pd.read_csv(c_path)
    assert len(df_loaded_csv) == len(df)
    assert "event_id" in df_loaded_csv.columns

