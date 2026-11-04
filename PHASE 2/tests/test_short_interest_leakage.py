import pytest
import pandas as pd
from pathlib import Path
from preprocessing.label_short_interest import label_events, _to_market_release_time, get_official_publication_date

def test_short_interest_prevents_look_ahead_bias(tmp_path):
    events_file = tmp_path / "events.csv"
    obs_file = tmp_path / "obs.csv"

    # Obs 1: Dec 31 settlement, public Jan 11 (SI = 10,000,000)
    # Obs 2: Jan 15 settlement, public Jan 26 (SI = 12,000,000)
    # Obs 3: Jan 31 settlement, public Feb 12 (SI = 10,200,000)
    df_obs = pd.DataFrame([
        {"ticker": "TEST", "settlement_date": "2023-12-31", "dissemination_date": "2024-01-11 18:00:00", "short_interest": 10000000},
        {"ticker": "TEST", "settlement_date": "2024-01-15", "dissemination_date": "2024-01-26 18:00:00", "short_interest": 12000000},
        {"ticker": "TEST", "settlement_date": "2024-01-31", "dissemination_date": "2024-02-12 18:00:00", "short_interest": 10200000},
    ])
    df_obs.to_csv(obs_file, index=False)

    # Event A: 2024-01-20 (between Jan 15 settlement and Jan 26 dissemination)
    # Under prior flawed code: picked Jan 15 settlement (LOOK-AHEAD BIAS)
    # Under fixed code: must pick Dec 31 settlement (public Jan 11 <= Jan 20)
    # Event B: 2024-01-28 (after Jan 26 dissemination)
    # Must pick Jan 15 settlement (public Jan 26 <= Jan 28)
    df_events = pd.DataFrame([
        {"event_id": "ev_A", "ticker": "TEST", "published_at": "2024-01-20 10:00:00+00:00"},
        {"event_id": "ev_B", "ticker": "TEST", "published_at": "2024-01-28 14:00:00+00:00"},
    ])
    df_events.to_csv(events_file, index=False)

    res = label_events(events_file, obs_file, threshold_pct=5.0)
    assert len(res) == 2

    row_a = res[res["event_id"] == "ev_A"].iloc[0]
    row_b = res[res["event_id"] == "ev_B"].iloc[0]

    # Verify Event A chose Dec 31 settlement (31 days gap <= 45 days)
    assert row_a["short_interest_before_date"] == "2023-12-31"
    assert row_a["short_interest_after_date"] == "2024-01-31"
    assert row_a["short_interest_gap_days"] == 31
    assert row_a["short_interest_label"] == "no_significant_change"

    # Verify Event B chose Jan 15 settlement (16 days gap <= 45 days)
    assert row_b["short_interest_before_date"] == "2024-01-15"
    assert row_b["short_interest_after_date"] == "2024-01-31"
    assert row_b["short_interest_gap_days"] == 16
    assert row_b["short_interest_label"] == "decrease"

def test_market_holiday_awareness_avoids_early_dissemination():
    """Verify that market holidays (e.g. Good Friday) push the dissemination date forward."""
    # March 28, 2024 is the settlement date. March 29, 2024 is Good Friday (Market Closed).
    # Raw BDay(7) gives 2024-04-08 (1 day early).
    # Market holiday calendar must give 2024-04-09.
    pub_date = get_official_publication_date("2024-03-28", bday_lag=7)
    assert str(pub_date) == "2024-04-09"

    # Backward compatibility release time helper
    dissem_ts = _to_market_release_time("2024-03-28", bday_lag=7, release_hour_et=16)
    assert str(dissem_ts.date()) == "2024-04-09"

def test_date_only_publication_conservatively_excludes_same_day_events(tmp_path):
    """Verify that date-only publication data conservatively excludes events on publication date."""
    events_file = tmp_path / "events.csv"
    obs_file = tmp_path / "obs.csv"

    # Obs 1: Dec 31 settlement, date-only publication Jan 11 (2024-01-11)
    # Obs 2: Jan 15 settlement, date-only publication Jan 26 (2024-01-26)
    # Obs 3: Jan 31 settlement, date-only publication Feb 12 (2024-02-12)
    pd.DataFrame([
        {"ticker": "TEST", "settlement_date": "2023-12-31", "dissemination_date": "2024-01-11", "short_interest": 10000000},
        {"ticker": "TEST", "settlement_date": "2024-01-15", "dissemination_date": "2024-01-26", "short_interest": 12000000},
        {"ticker": "TEST", "settlement_date": "2024-01-31", "dissemination_date": "2024-02-12", "short_interest": 15000000},
    ]).to_csv(obs_file, index=False)

    pd.DataFrame([
        # Event 1: Occurred on Jan 26 (the publication date of Obs 2).
        # Without verified timestamp, it MUST NOT use Obs 2 as baseline; must use Obs 1!
        {"event_id": "ev_same_day", "ticker": "TEST", "published_at": "2024-01-26 16:30:00+00:00"},
        # Event 2: Occurred on Jan 27 (day after publication date).
        # Safely uses Obs 2 as baseline!
        {"event_id": "ev_next_day", "ticker": "TEST", "published_at": "2024-01-27 10:00:00+00:00"},
    ]).to_csv(events_file, index=False)

    res = label_events(events_file, obs_file)
    assert len(res) == 2

    row_same_day = res[res["event_id"] == "ev_same_day"].iloc[0]
    row_next_day = res[res["event_id"] == "ev_next_day"].iloc[0]

    # Same day event: Obs 2 was excluded, so baseline is Dec 31 settlement (Obs 1)
    assert row_same_day["short_interest_before_date"] == "2023-12-31"
    assert row_same_day["short_interest_after_date"] == "2024-01-31"

    # Next day event: Obs 2 is now fully prior, so baseline is Jan 15 settlement (Obs 2)
    assert row_next_day["short_interest_before_date"] == "2024-01-15"
    assert row_next_day["short_interest_after_date"] == "2024-01-31"

def test_dissemination_source_recording(tmp_path):
    """Verify that dissemination date sources are audited and recorded in labeled records."""
    events_file = tmp_path / "events.csv"
    obs_file = tmp_path / "obs.csv"

    pd.DataFrame([
        {"ticker": "AAPL", "settlement_date": "2024-01-15", "short_interest": 50000000},
        {"ticker": "AAPL", "settlement_date": "2024-01-31", "short_interest": 55000000},
    ]).to_csv(obs_file, index=False)

    pd.DataFrame([
        {"event_id": "ev1", "ticker": "AAPL", "published_at": "2024-01-26 10:00:00+00:00"}
    ]).to_csv(events_file, index=False)

    # When no dissemination column is passed, source should be finra_rule_4560_schedule
    res = label_events(events_file, obs_file)
    assert len(res) == 1
    row = res.iloc[0]
    assert row["short_interest_before_dissemination_source"] == "finra_rule_4560_schedule"
    assert row["short_interest_after_dissemination_source"] == "finra_rule_4560_schedule"

def test_observation_gap_within_limit_succeeds(tmp_path):
    """Verify that observations within the settlement gap limit (default: 45 days) are labeled."""
    events_file = tmp_path / "events.csv"
    obs_file = tmp_path / "obs.csv"

    # Gap between 2024-01-15 and 2024-01-31 is 16 calendar days <= 45 days
    pd.DataFrame([
        {"ticker": "AAPL", "settlement_date": "2024-01-15", "dissemination_date": "2024-01-25 18:00:00", "short_interest": 50000000},
        {"ticker": "AAPL", "settlement_date": "2024-01-31", "dissemination_date": "2024-02-12 18:00:00", "short_interest": 55000000},
    ]).to_csv(obs_file, index=False)

    pd.DataFrame([
        {"event_id": "ev_valid", "ticker": "AAPL", "published_at": "2024-01-26 12:00:00+00:00"}
    ]).to_csv(events_file, index=False)

    res, audit = label_events(events_file, obs_file, threshold_pct=5.0, max_settlement_gap_days=45, return_audit=True)
    assert len(res) == 1
    assert len(audit) == 0
    assert res.iloc[0]["short_interest_gap_days"] == 16
    assert res.iloc[0]["short_interest_label"] == "increase"

def test_observation_gap_exceeding_limit_skipped(tmp_path):
    """Verify that observations with a settlement gap exceeding the threshold are skipped."""
    events_file = tmp_path / "events.csv"
    obs_file = tmp_path / "obs.csv"

    # Multi-month gap: Jan 15 to May 15 (121 calendar days > default 45 days)
    pd.DataFrame([
        {"ticker": "STALE", "settlement_date": "2024-01-15", "dissemination_date": "2024-01-25 18:00:00", "short_interest": 1000000},
        {"ticker": "STALE", "settlement_date": "2024-05-15", "dissemination_date": "2024-05-25 18:00:00", "short_interest": 2000000},
    ]).to_csv(obs_file, index=False)

    pd.DataFrame([
        {"event_id": "ev_gap", "ticker": "STALE", "published_at": "2024-01-26 10:00:00+00:00"}
    ]).to_csv(events_file, index=False)

    # 1. Skipped under default 45-day limit
    res, audit = label_events(events_file, obs_file, max_settlement_gap_days=45, return_audit=True)
    assert len(res) == 0
    assert len(audit) == 1
    assert "excessive_observation_gap" in audit[0]["reason"]
    assert audit[0]["gap_days"] == 121
    assert audit[0]["event_id"] == "ev_gap"

    # 2. Configurable threshold test: when max_settlement_gap_days=150, comparison is permitted
    res_allowed, audit_allowed = label_events(events_file, obs_file, max_settlement_gap_days=150, return_audit=True)
    assert len(res_allowed) == 1
    assert len(audit_allowed) == 0
    assert res_allowed.iloc[0]["short_interest_gap_days"] == 121

def test_missing_observations_skipped_cleanly(tmp_path):
    """Verify that missing baseline, missing outcome, or unknown tickers are skipped and audited."""
    events_file = tmp_path / "events.csv"
    obs_file = tmp_path / "obs.csv"

    pd.DataFrame([
        {"ticker": "KNOWN", "settlement_date": "2024-02-15", "dissemination_date": "2024-02-26 18:00:00", "short_interest": 1000000},
    ]).to_csv(obs_file, index=False)

    pd.DataFrame([
        # 1. Event prior to any disseminated observation (missing baseline)
        {"event_id": "ev_no_base", "ticker": "KNOWN", "published_at": "2024-01-10 10:00:00+00:00"},
        # 2. Event after all observations (missing outcome)
        {"event_id": "ev_no_outcome", "ticker": "KNOWN", "published_at": "2024-03-01 10:00:00+00:00"},
        # 3. Ticker not in observations
        {"event_id": "ev_unknown_ticker", "ticker": "UNKNOWN", "published_at": "2024-02-20 10:00:00+00:00"},
    ]).to_csv(events_file, index=False)

    res, audit = label_events(events_file, obs_file, return_audit=True)
    assert len(res) == 0
    assert len(audit) == 3

    reasons = {a["event_id"]: a["reason"] for a in audit}
    assert "missing_baseline" in reasons["ev_no_base"]
    assert "missing_outcome" in reasons["ev_no_outcome"]
    assert "no_observations_for_ticker" in reasons["ev_unknown_ticker"]
