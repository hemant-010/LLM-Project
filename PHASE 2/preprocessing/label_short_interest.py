"""Create next-observation short-interest direction labels from verified FINRA observations.

FINRA Rule 4560 & Exchange Reporting Architecture:
1. Schedule:
   - Reporting occurs twice monthly (mid-month ~15th and month-end settlement dates).
   - Firm Reporting Deadline: Broker-dealers submit reports to FINRA by 6:00 PM ET on the
     2nd business day following the settlement date.
   - Public Dissemination Date: Aggregate short interest is disseminated on the 7th
     market business day following the settlement date.
2. Dissemination Channels (Exchange-Listed vs. OTC):
   - Exchange-Listed Equities (NYSE, NASDAQ, Cboe): FINRA transfers compiled reports to
     each stock's primary listing exchange; the exchange officially disseminates after market close.
   - OTC Equities: FINRA serves as the primary dissemination agent directly via FINRA.org.
3. Market Holiday Calendars & Grounded Publication Dates:
   - The official calendar is governed by US market closures (Good Friday, MLK Day, etc.).
   - Dissemination dates are calendar dates established by FINRA's official reporting schedule.
   - For date-only publication data, events occurring on the publication date itself are
     conservatively excluded from using that observation as baseline unless an exact release
     timestamp has been verified. This eliminates intraday look-ahead bias on release days.
4. Maximum Observation Gap Safeguard:
   - A configurable maximum settlement gap (default: 45 calendar days, ~2 bi-monthly reporting cycles)
     is enforced to prevent pairing stale observations across historical lapses.
"""
import argparse
import bisect
from datetime import datetime, date
from pathlib import Path
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday,
    USMartinLutherKingJr, USPresidentsDay, USMemorialDay,
    USLaborDay, USThanksgivingDay, GoodFriday
)
from pandas.tseries.offsets import CustomBusinessDay

class USMarketCalendar(AbstractHolidayCalendar):
    """Official NYSE and FINRA US equity market holiday schedule."""
    rules = [
        Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2022-01-01"),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]

MARKET_BDAY = CustomBusinessDay(calendar=USMarketCalendar())

def get_official_publication_date(settlement_date, bday_lag: int = 7) -> date | None:
    """Derive official FINRA Rule 4560 publication calendar date (7 market business days after settlement)."""
    if pd.isna(settlement_date):
        return None
    ts = pd.to_datetime(settlement_date)
    d = ts.date()
    ts_market = pd.Timestamp(year=d.year, month=d.month, day=d.day)
    target_dt = ts_market + (MARKET_BDAY * bday_lag)
    return target_dt.date()

def _to_market_release_time(dt_val, bday_lag: int = 7, release_hour_et: int = 16):
    """Derive estimated public dissemination timestamp for backward compatibility."""
    if pd.isna(dt_val):
        return pd.NaT
    pub_date = get_official_publication_date(dt_val, bday_lag=bday_lag)
    if pub_date is None:
        return pd.NaT
    ts_et = pd.Timestamp(year=pub_date.year, month=pub_date.month, day=pub_date.day, tz="America/New_York").replace(hour=release_hour_et, minute=0, second=0)
    return ts_et.tz_convert("UTC")

def resolve_dissemination_timestamps(
    obs: pd.DataFrame,
    dissemination_col: str | None = None,
    bday_lag: int = 7
) -> pd.DataFrame:
    """Resolve publication dates and record whether verified intraday timestamps exist.

    Distinguishes date-only publication dates from verified intraday timestamps,
    recording the source of each dissemination date for research auditability.
    """
    obs = obs.copy()
    candidate_cols = [dissemination_col] if dissemination_col else [
        "dissemination_date", "exchange_dissemination_date", "finra_dissemination_date",
        "publication_date", "release_date"
    ]
    found_col = next((c for c in candidate_cols if c and c in obs.columns), None)

    if found_col:
        raw_vals = obs[found_col]
        parsed_dates = []
        is_timestamped = []
        sources = []

        for val in raw_vals:
            if pd.isna(val):
                parsed_dates.append(pd.NaT)
                is_timestamped.append(False)
                sources.append("missing")
                continue

            ts = pd.to_datetime(val, utc=True, errors="coerce")
            raw_str = str(val).strip()

            if pd.isna(ts):
                parsed_dates.append(pd.NaT)
                is_timestamped.append(False)
                sources.append("unparseable")
            # If the raw string does not contain an explicit time indicator (colon or 'T'), it is date-only
            elif ":" not in raw_str and "T" not in raw_str:
                parsed_dates.append(ts.date())
                is_timestamped.append(False)
                sources.append(f"dataset_provided_date_only ({found_col})")
            elif ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ("00:00:00" in raw_str or "T00:00" in raw_str):
                # Explicit midnight timestamp is typically date-only normalization
                parsed_dates.append(ts.date())
                is_timestamped.append(False)
                sources.append(f"dataset_provided_date_only ({found_col})")
            else:
                # Verified intraday timestamp with non-zero time
                parsed_dates.append(ts)
                is_timestamped.append(True)
                sources.append(f"verified_timestamp ({found_col})")

        obs["dissemination_date"] = parsed_dates
        obs["dissemination_is_timestamped"] = is_timestamped
        obs["dissemination_date_source"] = sources
    else:
        # Grounded in FINRA Rule 4560 official calendar
        obs["dissemination_date"] = obs["settlement_date"].map(lambda d: get_official_publication_date(d, bday_lag))
        obs["dissemination_is_timestamped"] = False
        obs["dissemination_date_source"] = "finra_rule_4560_schedule"

    return obs

def _load_data(path_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load DataFrame from path (parquet or csv) or return copy if already DataFrame."""
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df.copy()
    p = Path(path_or_df)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)

def label_events(
    events_path: str | Path | pd.DataFrame,
    observations_path: str | Path | pd.DataFrame,
    threshold_pct: float = 5.0,
    max_settlement_gap_days: int = 45,
    dissemination_col: str | None = None,
    bday_lag: int = 7,
    return_audit: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]]:
    """Label events based on next eligible FINRA short-interest observation.

    Parameters
    ----------
    events_path : str | Path
        Path to events CSV (must contain ticker, published_at).
    observations_path : str | Path
        Path to FINRA observations CSV (must contain ticker, settlement_date, short_interest).
    threshold_pct : float
        Percentage change threshold for 3-class labeling (+/- 5%).
    max_settlement_gap_days : int
        Maximum allowed calendar days between baseline and outcome settlement dates (default: 45).
    dissemination_col : str | None
        Explicit dissemination column name if present in observations.
    bday_lag : int
        Business day lag for estimated dissemination (default: 7 market days).
    return_audit : bool
        If True, returns a 2-tuple: (labeled_df, skipped_audit_records).

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, list[dict]]
        Labeled events dataframe (and optional list of skipped audit dicts).
    """
    events = _load_data(events_path)
    obs = _load_data(observations_path)

    events["published_at"] = pd.to_datetime(events["published_at"], utc=True, errors="coerce")
    obs["settlement_date"] = pd.to_datetime(obs["settlement_date"], errors="coerce", utc=True)
    obs["short_interest"] = pd.to_numeric(obs["short_interest"], errors="coerce")

    obs = resolve_dissemination_timestamps(
        obs, dissemination_col=dissemination_col, bday_lag=bday_lag
    )

    obs = obs.dropna(subset=["settlement_date", "dissemination_date", "short_interest"])
    obs = obs[obs["short_interest"] > 0].sort_values("settlement_date").reset_index(drop=True)
    obs["_clean_ticker"] = obs["ticker"].astype(str).str.strip().str.upper()

    # Pre-index observations per ticker with fast lookup metadata
    obs_by_ticker = {}
    for t, grp in obs.groupby("_clean_ticker"):
        has_ts = bool(grp["dissemination_is_timestamped"].any()) if "dissemination_is_timestamped" in grp else False
        rows_list = grp.to_dict(orient="records")
        settle_dates = [s.date() if hasattr(s, "date") else pd.to_datetime(s).date() for s in grp["settlement_date"]]
        dissem_dates = [
            d if isinstance(d, date) else (d.date() if hasattr(d, "date") else pd.to_datetime(d).date())
            for d in grp["dissemination_date"]
        ]
        obs_by_ticker[t] = {
            "rows": rows_list,
            "has_ts": has_ts,
            "settle_dates": settle_dates,
            "dissem_dates": dissem_dates
        }

    rows = []
    skipped = []

    events_records = events.to_dict(orient="records")

    for idx, event_dict in enumerate(events_records):
        event_id = event_dict.get("event_id", f"idx_{idx}")
        pub_at_raw = event_dict.get("published_at")

        if pd.isna(pub_at_raw) or str(pub_at_raw).strip() == "" or str(pub_at_raw) == "NaT":
            skipped.append({
                "event_id": event_id,
                "ticker": event_dict.get("ticker"),
                "published_at": pub_at_raw,
                "reason": "missing_or_invalid_published_at"
            })
            continue

        event_ts = pd.to_datetime(pub_at_raw, utc=True)
        if pd.isna(event_ts):
            skipped.append({
                "event_id": event_id,
                "ticker": event_dict.get("ticker"),
                "published_at": str(pub_at_raw),
                "reason": "missing_or_invalid_published_at"
            })
            continue

        ticker = str(event_dict.get("ticker", "")).strip().upper()
        if not ticker or ticker == "NAN":
            skipped.append({
                "event_id": event_id,
                "ticker": event_dict.get("ticker"),
                "published_at": str(event_ts),
                "reason": "missing_ticker"
            })
            continue

        hist_meta = obs_by_ticker.get(ticker)
        if hist_meta is None or not hist_meta["rows"]:
            skipped.append({
                "event_id": event_id,
                "ticker": ticker,
                "published_at": str(event_ts),
                "reason": "no_observations_for_ticker"
            })
            continue

        # Event date in America/New_York (preserve exact calendar date if date-only input)
        if "published_date" in event_dict and pd.notna(event_dict["published_date"]):
            event_date = pd.to_datetime(event_dict["published_date"]).date()
        elif "date" in event_dict and pd.notna(event_dict["date"]):
            event_date = pd.to_datetime(event_dict["date"]).date()
        else:
            if event_ts.hour == 0 and event_ts.minute == 0 and event_ts.second == 0 and event_ts.tzinfo:
                event_date = event_ts.date()
            else:
                event_date = event_ts.tz_convert("America/New_York").date() if event_ts.tzinfo else event_ts.date()

        rows_list = hist_meta["rows"]
        settle_dates = hist_meta["settle_dates"]
        dissem_dates = hist_meta["dissem_dates"]

        if hist_meta["has_ts"]:
            # Intraday timestamped observations present: inspect exact dissemination timestamp
            before = None
            for r_idx, r_obs in enumerate(rows_list):
                dissem_val = r_obs["dissemination_date"]
                if pd.isna(dissem_val):
                    continue
                if r_obs.get("dissemination_is_timestamped"):
                    is_prior = dissem_val <= event_ts
                else:
                    pub_d = dissem_dates[r_idx]
                    is_prior = pub_d < event_date
                if is_prior:
                    before = r_obs
                else:
                    break

            after = None
            for r_idx, r_obs in enumerate(rows_list):
                if settle_dates[r_idx] > event_date:
                    after = r_obs
                    break
        else:
            # Fast binary search on date-only publication & settlement schedules
            before_idx = bisect.bisect_left(dissem_dates, event_date) - 1
            before = rows_list[before_idx] if before_idx >= 0 else None

            after_idx = bisect.bisect_right(settle_dates, event_date)
            after = rows_list[after_idx] if after_idx < len(rows_list) else None

        if before is None:
            skipped.append({
                "event_id": event_id,
                "ticker": ticker,
                "published_at": str(event_ts),
                "reason": "missing_baseline_observation_before_event"
            })
            continue

        if after is None:
            skipped.append({
                "event_id": event_id,
                "ticker": ticker,
                "published_at": str(event_ts),
                "reason": "missing_outcome_observation_after_event"
            })
            continue

        before_si = float(before["short_interest"])
        after_si = float(after["short_interest"])

        if before_si <= 0:
            skipped.append({
                "event_id": event_id,
                "ticker": ticker,
                "published_at": str(event_ts),
                "reason": "invalid_baseline_short_interest_non_positive"
            })
            continue

        before_settle = before["settlement_date"].date() if hasattr(before["settlement_date"], "date") else pd.to_datetime(before["settlement_date"]).date()
        after_settle = after["settlement_date"].date() if hasattr(after["settlement_date"], "date") else pd.to_datetime(after["settlement_date"]).date()

        gap_days = (after_settle - before_settle).days
        if max_settlement_gap_days is not None and gap_days > max_settlement_gap_days:
            skipped.append({
                "event_id": event_id,
                "ticker": ticker,
                "published_at": str(event_ts),
                "reason": f"excessive_observation_gap: gap of {gap_days} calendar days ({before_settle} to {after_settle}) exceeds max {max_settlement_gap_days} days",
                "gap_days": gap_days,
                "baseline_settlement": str(before_settle),
                "outcome_settlement": str(after_settle)
            })
            continue

        change_pct = (after_si - before_si) / before_si * 100
        label = "increase" if change_pct > threshold_pct else "decrease" if change_pct < -threshold_pct else "no_significant_change"

        r = event_dict.copy()
        r.update({
            "short_interest_before": before_si,
            "short_interest_before_date": str(before_settle),
            "short_interest_before_dissemination": str(before["dissemination_date"]),
            "short_interest_before_dissemination_source": str(before.get("dissemination_date_source", "unknown")),
            "short_interest_after": after_si,
            "short_interest_after_date": str(after_settle),
            "short_interest_after_dissemination": str(after["dissemination_date"]),
            "short_interest_after_dissemination_source": str(after.get("dissemination_date_source", "unknown")),
            "short_interest_gap_days": gap_days,
            "short_interest_change_pct": round(change_pct, 4),
            "short_interest_label": label
        })
        rows.append(r)

    out = pd.DataFrame(rows)
    if skipped:
        print(f"INFO: {len(rows)} event(s) labeled, {len(skipped)} event(s) skipped:")
        reason_counts = pd.Series([s["reason"].split(":")[0] for s in skipped]).value_counts().to_dict()
        for r_name, cnt in reason_counts.items():
            print(f"  - {r_name}: {cnt}")
    else:
        print(f"INFO: All {len(rows)} event(s) successfully labeled.")

    if return_audit:
        return out, skipped
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--output", default="data/processed/events_with_short_interest_labels.csv")
    p.add_argument("--output-parquet", default="data/processed/events_with_short_interest_labels.parquet")
    p.add_argument("--threshold-pct", type=float, default=5.0)
    p.add_argument("--max-settlement-gap-days", type=int, default=45, help="Maximum calendar days between baseline and outcome settlement dates (default: 45)")
    p.add_argument("--audit-output", default="data/processed/skipped_short_interest_labels_audit.csv", help="Optional path to write CSV audit log of skipped events")
    p.add_argument("--dissemination-col", default=None, help="Explicit column name for dissemination/publication dates")
    p.add_argument("--bday-lag", type=int, default=7, help="Exchange business-day lag for public dissemination (default: 7)")
    a = p.parse_args()

    result, audit = label_events(
        a.events, a.observations,
        threshold_pct=a.threshold_pct,
        max_settlement_gap_days=a.max_settlement_gap_days,
        dissemination_col=a.dissemination_col,
        bday_lag=a.bday_lag,
        return_audit=True
    )
    if a.output:
        out_p = Path(a.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix == ".parquet":
            result.to_parquet(out_p, index=False)
        else:
            result.to_csv(out_p, index=False)
        print(f"Wrote {len(result)} labelled rows to {a.output}")

    if a.output_parquet:
        pq_p = Path(a.output_parquet)
        pq_p.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(pq_p, index=False)
        print(f"Wrote {len(result)} labelled rows to {a.output_parquet}")

    if a.audit_output and audit:
        audit_p = Path(a.audit_output)
        audit_p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit).to_csv(audit_p, index=False)
        print(f"Wrote {len(audit)} skipped audit record(s) to {a.audit_output}")
