import pytest
import pandas as pd
from datetime import time
from preprocessing.build_price_labels import determine_baseline_and_target_dates

# 10 consecutive trading days around MLK Day 2024 (Jan 15 holiday)
TRADING_DAYS = [
    pd.Timestamp(d) for d in [
        "2024-01-12", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19",
        "2024-01-22", "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-26"
    ]
]

def test_pre_market_timing():
    # Tuesday Jan 16, 08:30 ET
    dt = pd.to_datetime("2024-01-16 13:30:00+00:00")
    d0, d5, info = determine_baseline_and_target_dates(dt, TRADING_DAYS)
    assert str(d0.date()) == "2024-01-12"
    assert str(d5.date()) == "2024-01-22"
    assert info["horizon_type"] == "5_sessions_full_exposure"
    assert info["includes_announcement_day"] is True
    assert info["first_trading_session"] == "2024-01-16"

def test_intraday_post_close_timing():
    # Tuesday Jan 16, 11:30 ET
    dt = pd.to_datetime("2024-01-16 16:30:00+00:00")
    d0, d5, info = determine_baseline_and_target_dates(dt, TRADING_DAYS, intraday_baseline="post_close")
    assert str(d0.date()) == "2024-01-16"
    assert str(d5.date()) == "2024-01-23"
    assert info["horizon_type"] == "5_sessions_post_close_drift"
    assert info["includes_announcement_day"] is False
    assert info["first_trading_session"] == "2024-01-17"

def test_after_hours_timing():
    # Tuesday Jan 16, 17:30 ET
    dt = pd.to_datetime("2024-01-16 22:30:00+00:00")
    d0, d5, info = determine_baseline_and_target_dates(dt, TRADING_DAYS)
    assert str(d0.date()) == "2024-01-16"
    assert str(d5.date()) == "2024-01-23"
    assert info["horizon_type"] == "5_sessions_full_exposure"
    assert info["includes_announcement_day"] is False

def test_weekend_and_holiday_timing():
    # Saturday Jan 13, 14:00 ET
    dt_sat = pd.to_datetime("2024-01-13 19:00:00+00:00")
    d0_sat, d5_sat, info_sat = determine_baseline_and_target_dates(dt_sat, TRADING_DAYS)
    assert str(d0_sat.date()) == "2024-01-12"
    assert str(d5_sat.date()) == "2024-01-22"

    # MLK Monday Jan 15, 12:00 ET
    dt_mlk = pd.to_datetime("2024-01-15 17:00:00+00:00")
    d0_mlk, d5_mlk, info_mlk = determine_baseline_and_target_dates(dt_mlk, TRADING_DAYS)
    assert str(d0_mlk.date()) == "2024-01-12"
    assert str(d5_mlk.date()) == "2024-01-22"

def test_session_count_is_strictly_five():
    """Verify that every regime spans exactly 5 trading sessions between d0 and d5."""
    timestamps = [
        "2024-01-16 13:00:00+00:00", # Pre-market
        "2024-01-16 16:00:00+00:00", # Intraday
        "2024-01-16 22:00:00+00:00", # After-hours
        "2024-01-13 18:00:00+00:00", # Weekend
        "2024-01-15 18:00:00+00:00", # Holiday
    ]
    for ts in timestamps:
        d0, d5, info = determine_baseline_and_target_dates(pd.to_datetime(ts), TRADING_DAYS)
        active_sessions = [d for d in TRADING_DAYS if d.date() > d0.date() and d.date() <= d5.date()]
        assert len(active_sessions) == 5
        assert info["horizon_sessions"] == 5

