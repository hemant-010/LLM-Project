import pytest
import pandas as pd
from pathlib import Path
from preprocessing.merge_labels import main

def test_merge_labels_clean_target_columns(tmp_path):
    si_file = tmp_path / "si.csv"
    price_file = tmp_path / "price.csv"
    out_file = tmp_path / "merged.csv"

    df_si = pd.DataFrame([
        {"event_id": "ev1", "ticker": "AAPL", "title": "Headline 1", "short_interest_label": "increase"},
        {"event_id": "ev2", "ticker": "TSLA", "title": "Headline 2", "short_interest_label": "decrease"},
        {"event_id": "ev3", "ticker": "MSFT", "title": "Headline 3", "short_interest_label": "no_significant_change"},
    ])
    df_si.to_csv(si_file, index=False)

    df_price = pd.DataFrame([
        {"event_id": "ev1", "ticker": "AAPL", "stock_price_label": "decrease"},
        {"event_id": "ev2", "ticker": "TSLA", "stock_price_label": "increase"},
        {"event_id": "ev4", "ticker": "NVDA", "stock_price_label": "increase"},
    ])
    df_price.to_csv(price_file, index=False)

    merged = main(str(si_file), str(price_file), str(out_file))

    cols = list(merged.columns)
    assert cols.count("short_interest_label") == 1
    assert cols.count("stock_price_label") == 1
    assert not any(c.endswith("_dup") for c in cols)
    assert len(merged) == 2
    assert set(merged["event_id"]) == {"ev1", "ev2"}

def test_merge_labels_validates_required_columns(tmp_path):
    si_file = tmp_path / "si_invalid.csv"
    price_file = tmp_path / "price_invalid.csv"
    out_file = tmp_path / "out.csv"

    # Missing short_interest_label
    pd.DataFrame([{"event_id": "ev1"}]).to_csv(si_file, index=False)
    pd.DataFrame([{"event_id": "ev1", "stock_price_label": "increase"}]).to_csv(price_file, index=False)

    with pytest.raises(ValueError, match="missing 'short_interest_label'"):
        main(str(si_file), str(price_file), str(out_file))

