"""Unit tests for historical FINRA short-interest collection workflow."""
import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from preprocessing.collect_historical_short_interest import (
    load_target_universe,
    get_top_tickers_by_event_count,
    run_ticker_collection,
    run_cycle_collection,
    save_collected_observations
)

@pytest.fixture
def mock_news_parquet(tmp_path):
    df = pd.DataFrame([
        {"ticker": "AAPL", "date": "2018-01-05"},
        {"ticker": "AAPL", "date": "2018-01-10"},
        {"ticker": "MSFT", "date": "2019-03-15"},
        {"ticker": "NVDA", "date": "2020-05-20"}
    ])
    p = tmp_path / "mock_news.parquet"
    df.to_parquet(p, index=False)
    return p

def test_load_target_universe(mock_news_parquet):
    tickers, min_d, max_d = load_target_universe(mock_news_parquet)
    assert sorted(tickers) == ["AAPL", "MSFT", "NVDA"]
    assert min_d == "2018-01-05"
    assert max_d == "2020-05-20"

def test_get_top_tickers_by_event_count(mock_news_parquet):
    top = get_top_tickers_by_event_count(mock_news_parquet, top_n=2)
    assert len(top) == 2
    assert top[0] == "AAPL"  # 2 events

def test_run_ticker_collection_mocked(tmp_path):
    """Verify ticker collection loop handles SUCCESS and EMPTY_DATASET cleanly."""
    raw_dir = tmp_path / "raw"
    ckpt_file = raw_dir / "ckpt.json"

    def mock_fetch(start_date, end_date, ticker, raw_dir):
        if ticker == "FAIL":
            return {"status": "ERROR", "message": "API 500 error", "normalized_df": pd.DataFrame()}
        elif ticker == "EMPTY":
            return {"status": "EMPTY_DATASET", "message": "No records", "normalized_df": pd.DataFrame()}
        else:
            df = pd.DataFrame([{
                "ticker": ticker,
                "settlement_date": "2018-01-15",
                "short_interest": 100000.0,
                "dissemination_date": "2018-01-24",
                "dissemination_date_source": "finra_rule_4560_schedule",
                "issue_name": f"{ticker} Corp",
                "previous_short_interest": 90000.0,
                "change_pct": 11.1,
                "days_to_cover": 1.2,
                "market_category": "Q"
            }])
            return {"status": "SUCCESS", "normalized_df": df}

    with patch("preprocessing.collect_historical_short_interest.fetch_consolidated_short_interest", side_effect=mock_fetch):
        df, report = run_ticker_collection(
            tickers=["AAPL", "EMPTY", "FAIL"],
            raw_dir=raw_dir,
            checkpoint_file=ckpt_file,
            delay_between_calls=0
        )

    assert report["requested_tickers"] == 3
    assert report["successful_tickers"] == 1
    assert report["empty_tickers"] == 1
    assert report["failed_tickers"] == 1
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"

    # Verify checkpoint persistence
    assert ckpt_file.exists()
    with open(ckpt_file) as f:
        ckpt_data = json.load(f)
    assert "AAPL" in ckpt_data["completed"]
    assert "EMPTY" in ckpt_data["empty"]
    assert "FAIL" in ckpt_data["failed"]

def test_run_ticker_collection_resumability(tmp_path):
    """Verify previously completed tickers are skipped on resume."""
    raw_dir = tmp_path / "raw"
    interim_dir = raw_dir / "interim_tickers"
    interim_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = raw_dir / "ckpt.json"

    # Create previous interim file for AAPL
    prev_df = pd.DataFrame([{
        "ticker": "AAPL",
        "settlement_date": "2018-01-15",
        "short_interest": 500000.0,
        "dissemination_date": "2018-01-24"
    }])
    prev_df.to_parquet(interim_dir / "AAPL.parquet", index=False)

    with open(ckpt_file, "w") as f:
        json.dump({"completed": {"AAPL": 1}, "empty": [], "failed": {}}, f)

    fetch_mock = MagicMock()
    with patch("preprocessing.collect_historical_short_interest.fetch_consolidated_short_interest", fetch_mock):
        df, report = run_ticker_collection(
            tickers=["AAPL"],
            raw_dir=raw_dir,
            checkpoint_file=ckpt_file,
            delay_between_calls=0
        )

    # AAPL was already completed, so fetch should NOT have been called
    fetch_mock.assert_not_called()
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"

def test_run_cycle_collection_mocked(tmp_path):
    """Verify cycle collection aggregates and filters to target tickers."""
    raw_dir = tmp_path / "raw"
    ckpt_file = raw_dir / "ckpt_cycle.json"

    cycle_records = pd.DataFrame([
        {"ticker": "AAPL", "settlement_date": "2018-01-15", "short_interest": 1000.0, "dissemination_date": "2018-01-24"},
        {"ticker": "OTHER", "settlement_date": "2018-01-15", "short_interest": 2000.0, "dissemination_date": "2018-01-24"}
    ])

    with patch("preprocessing.collect_historical_short_interest.fetch_consolidated_short_interest", return_value={"status": "SUCCESS", "normalized_df": cycle_records}):
        df, report = run_cycle_collection(
            target_tickers={"AAPL"},
            cycles=["2018-01-15"],
            raw_dir=raw_dir,
            checkpoint_file=ckpt_file,
            delay_between_calls=0
        )

    # Filtered only to target universe AAPL
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"
    assert report["total_records_all_securities"] == 2
    assert report["total_records_target_universe"] == 1
    assert report["target_tickers_found"] == 1

def test_save_collected_observations(tmp_path):
    df = pd.DataFrame([{"ticker": "AAPL", "settlement_date": "2018-01-15", "short_interest": 100.0}])
    pq_path, csv_path = save_collected_observations(df, output_dir=tmp_path, file_stem="test_si")

    assert pq_path.exists()
    assert csv_path.exists()
    assert len(pd.read_parquet(pq_path)) == 1

