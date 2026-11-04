import pytest
import pandas as pd
from pathlib import Path
from preprocessing.sec_to_events import convert
from feature_engineering.context_analyzer import analyze_context

def test_10b5_1_three_state_conversion(tmp_path):
    input_csv = tmp_path / "raw_tx.csv"
    output_csv = tmp_path / "events.csv"

    df_raw = pd.DataFrame([
        {
            "ticker": "AAPL", "filing_date": "2024-01-15", "transaction_date": "2024-01-14",
            "transaction_code": "S", "acquired_disposed": "D", "shares": 1000.0,
            "price_per_share": 180.0, "shares_owned_after": 9000.0, "insider_roles": "Chief Executive Officer",
            "filing_url": "https://sec.gov/1", "mentions_10b5_1": True
        },
        {
            "ticker": "GOOGL", "filing_date": "2024-01-15", "transaction_date": "2024-01-14",
            "transaction_code": "S", "acquired_disposed": "D", "shares": 2000.0,
            "price_per_share": 140.0, "shares_owned_after": 8000.0, "insider_roles": "Director",
            "filing_url": "https://sec.gov/2", "mentions_10b5_1": False
        },
        {
            "ticker": "MSFT", "filing_date": "2024-01-15", "transaction_date": "2024-01-14",
            "transaction_code": "S", "acquired_disposed": "D", "shares": 500.0,
            "price_per_share": 390.0, "shares_owned_after": 4500.0, "insider_roles": "Officer",
            "filing_url": "https://sec.gov/3", "mentions_10b5_1": None
        }
    ])
    df_raw.to_csv(input_csv, index=False)

    events = convert(str(input_csv), str(output_csv))
    assert len(events) == 3

    val_true = events.loc[events["ticker"] == "AAPL", "is_10b5_1"].iloc[0]
    val_false = events.loc[events["ticker"] == "GOOGL", "is_10b5_1"].iloc[0]
    val_unknown = events.loc[events["ticker"] == "MSFT", "is_10b5_1"].iloc[0]

    assert val_true is True
    assert val_false is False
    assert val_unknown == "unknown"

def test_10b5_1_downstream_context_analyzer():
    # 1. True case
    row_true = {
        "event_type": "insider_selling", "is_10b5_1": True,
        "shares_sold": 1000, "shares_owned_before": 10000, "insider_role": "CEO"
    }
    res_true = analyze_context(row_true)
    assert res_true["insider_is_10b5_1"] is True
    assert "marked as a 10b5-1 plan" in res_true["context_flags"]

    # 2. False case (must not be converted to None or unknown!)
    row_false = {
        "event_type": "insider_selling", "is_10b5_1": False,
        "shares_sold": 1000, "shares_owned_before": 10000, "insider_role": "Director"
    }
    res_false = analyze_context(row_false)
    assert res_false["insider_is_10b5_1"] is False
    assert "confirmed not under a 10b5-1 plan" in res_false["context_flags"]

    # 3. Unknown case
    row_unknown = {
        "event_type": "insider_selling", "is_10b5_1": "unknown",
        "shares_sold": 1000, "shares_owned_before": 10000
    }
    res_unknown = analyze_context(row_unknown)
    assert res_unknown["insider_is_10b5_1"] is None
    assert "10b5-1 plan status is unknown" in res_unknown["context_flags"]

