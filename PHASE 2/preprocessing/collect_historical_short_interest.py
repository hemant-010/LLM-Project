"""Historical FINRA Consolidated Short-Interest Collection Workflow for Phase 2.

Collects, normalizes, and audits official FINRA short-interest observations for the
historical equity universe (2017-12-29 through 2020-07-15).

Key Architectural Principles:
1. Target Universe Grounding:
   - Reads target tickers and dates from `data/processed/historical_news_2017_2020.parquet`.
   - Extends end settlement date to `2020-07-15` to ensure valid post-event outcome
     observations for all news events occurring up to `2020-06-11`.
2. Resumability & Checkpointing:
   - Tracks download progress in checkpoint files (`.finra_*_checkpoint.json`).
   - Interrupted runs resume from the exact point of interruption without re-downloading
     completed data.
3. Raw API Persistence:
   - Stores original, unaltered JSON responses from FINRA under `data/raw/market/finra/`.
4. Dissemination Fidelity:
   - Grounds publication dates in FINRA Rule 4560 calendar schedule (date-only string).
   - Zero fabricated release timestamps.
5. Dual Collection Strategies:
   - Ticker-Mode: Queries ticker by ticker (ideal for pilot test of 10 tickers).
   - Cycle-Mode: Queries semi-monthly settlement cycles (optimal for full market universe
     in ~62 API calls without hitting per-ticker rate limits).
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# Ensure PHASE 2 root is in sys.path
_phase2_root = Path(__file__).resolve().parent.parent
if str(_phase2_root) not in sys.path:
    sys.path.insert(0, str(_phase2_root))

from config import ROOT, DATA_RAW, DATA_PROCESSED
from preprocessing.finra_short_interest_collector import (
    fetch_consolidated_short_interest,
    normalize_finra_records,
    get_access_token,
    get_finra_credentials
)

DEFAULT_NEWS_PARQUET = DATA_PROCESSED / "historical_news_2017_2020.parquet"
DEFAULT_RAW_DIR = DATA_RAW / "market" / "finra"
DEFAULT_PROCESSED_DIR = DATA_PROCESSED
EARLIEST_SETTLEMENT = "2017-12-29"
LATEST_SETTLEMENT = "2020-07-15"

# Known 62 semi-monthly settlement dates from 2017-12-29 to 2020-07-15
HISTORICAL_CYCLES_2017_2020 = [
    "2017-12-29", "2018-01-12", "2018-01-31", "2018-02-15", "2018-02-28",
    "2018-03-15", "2018-03-29", "2018-04-13", "2018-04-30", "2018-05-15",
    "2018-05-31", "2018-06-15", "2018-06-29", "2018-07-13", "2018-07-31",
    "2018-08-15", "2018-08-31", "2018-09-14", "2018-09-28", "2018-10-15",
    "2018-10-31", "2018-11-15", "2018-11-30", "2018-12-14", "2018-12-31",
    "2019-01-15", "2019-01-31", "2019-02-15", "2019-02-28", "2019-03-15",
    "2019-03-29", "2019-04-15", "2019-04-30", "2019-05-15", "2019-05-31",
    "2019-06-14", "2019-06-28", "2019-07-15", "2019-07-31", "2019-08-15",
    "2019-08-30", "2019-09-13", "2019-09-30", "2019-10-15", "2019-10-31",
    "2019-11-15", "2019-11-29", "2019-12-13", "2019-12-31", "2020-01-15",
    "2020-01-31", "2020-02-14", "2020-02-28", "2020-03-13", "2020-03-31",
    "2020-04-15", "2020-04-30", "2020-05-15", "2020-05-29", "2020-06-15",
    "2020-06-30", "2020-07-15"
]

def load_target_universe(news_path: str | Path = DEFAULT_NEWS_PARQUET) -> tuple[list[str], str, str]:
    """Load target tickers and date range from extracted historical news dataset."""
    p = Path(news_path)
    if not p.exists():
        raise FileNotFoundError(f"News parquet dataset not found at {p}")
    df = pd.read_parquet(p, columns=["ticker", "date"])
    tickers = sorted(df["ticker"].dropna().unique().tolist())
    min_date = str(df["date"].min())
    max_date = str(df["date"].max())
    return tickers, min_date, max_date

def get_top_tickers_by_event_count(
    news_path: str | Path = DEFAULT_NEWS_PARQUET,
    top_n: int = 10
) -> list[str]:
    """Return top N tickers by news event volume for pilot testing."""
    p = Path(news_path)
    df = pd.read_parquet(p, columns=["ticker"])
    top = df["ticker"].value_counts().head(top_n).index.tolist()
    return top

def run_ticker_collection(
    tickers: list[str],
    start_date: str = EARLIEST_SETTLEMENT,
    end_date: str = LATEST_SETTLEMENT,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    checkpoint_file: str | Path | None = None,
    delay_between_calls: float = 0.5
) -> tuple[pd.DataFrame, dict]:
    """Collect FINRA short interest ticker-by-ticker with resumable checkpointing."""
    raw_d = Path(raw_dir)
    raw_d.mkdir(parents=True, exist_ok=True)
    interim_dir = raw_d / "interim_tickers"
    interim_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(checkpoint_file) if checkpoint_file else raw_d / ".finra_ticker_checkpoint.json"
    checkpoint = {
        "completed": {},
        "empty": [],
        "failed": {}
    }
    if ckpt_path.exists():
        try:
            with open(ckpt_path) as f:
                checkpoint = json.load(f)
        except Exception:
            pass

    completed_tickers = set(checkpoint.get("completed", {}).keys())
    empty_tickers = set(checkpoint.get("empty", []))
    failed_tickers = checkpoint.get("failed", {})

    all_dfs = []

    # Reload previously completed interim parquet files
    for t in completed_tickers:
        t_pq = interim_dir / f"{t}.parquet"
        if t_pq.exists():
            try:
                all_dfs.append(pd.read_parquet(t_pq))
            except Exception:
                pass

    total = len(tickers)
    for idx, ticker in enumerate(tickers):
        clean_t = str(ticker).strip().upper()
        if clean_t in completed_tickers or clean_t in empty_tickers:
            continue

        print(f"[{idx+1}/{total}] Fetching {clean_t} ({start_date} -> {end_date})...", end=" ", flush=True)
        res = fetch_consolidated_short_interest(
            start_date=start_date,
            end_date=end_date,
            ticker=clean_t,
            raw_dir=raw_d
        )

        status = res["status"]
        if status == "SUCCESS":
            df_norm = res["normalized_df"]
            records_cnt = len(df_norm)
            print(f"SUCCESS: {records_cnt} records")
            # Save interim parquet
            t_pq = interim_dir / f"{clean_t}.parquet"
            df_norm.to_parquet(t_pq, index=False)
            all_dfs.append(df_norm)
            checkpoint.setdefault("completed", {})[clean_t] = records_cnt
            failed_tickers.pop(clean_t, None)
        elif status == "EMPTY_DATASET":
            print("EMPTY_DATASET: 0 records")
            checkpoint.setdefault("empty", []).append(clean_t)
            failed_tickers.pop(clean_t, None)
        else:
            err_msg = res.get("message", "Unknown error")
            print(f"FAILED: {err_msg}")
            failed_tickers[clean_t] = err_msg

        # Update checkpoint
        checkpoint["failed"] = failed_tickers
        checkpoint["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        if delay_between_calls > 0:
            time.sleep(delay_between_calls)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(
            subset=["ticker", "settlement_date"]
        ).sort_values(["ticker", "settlement_date"]).reset_index(drop=True)
    else:
        combined_df = normalize_finra_records([])

    report = {
        "mode": "ticker",
        "requested_tickers": len(tickers),
        "successful_tickers": len(checkpoint.get("completed", {})),
        "empty_tickers": len(checkpoint.get("empty", [])),
        "failed_tickers": len(checkpoint.get("failed", {})),
        "total_records": len(combined_df),
        "date_min": str(combined_df["settlement_date"].min()) if not combined_df.empty else None,
        "date_max": str(combined_df["settlement_date"].max()) if not combined_df.empty else None,
        "unique_settlement_dates": int(combined_df["settlement_date"].nunique()) if not combined_df.empty else 0,
        "failure_details": checkpoint.get("failed", {})
    }
    return combined_df, report

def run_cycle_collection(
    target_tickers: set[str] | None = None,
    cycles: list[str] = HISTORICAL_CYCLES_2017_2020,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    checkpoint_file: str | Path | None = None,
    delay_between_calls: float = 0.5
) -> tuple[pd.DataFrame, dict]:
    """Collect FINRA short interest cycle-by-cycle across all securities with resumable checkpointing."""
    raw_d = Path(raw_dir)
    raw_d.mkdir(parents=True, exist_ok=True)
    interim_dir = raw_d / "interim_cycles"
    interim_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(checkpoint_file) if checkpoint_file else raw_d / ".finra_cycle_checkpoint.json"
    checkpoint = {
        "completed_cycles": {},
        "failed_cycles": {}
    }
    if ckpt_path.exists():
        try:
            with open(ckpt_path) as f:
                checkpoint = json.load(f)
        except Exception:
            pass

    completed_cycles = set(checkpoint.get("completed_cycles", {}).keys())
    failed_cycles = checkpoint.get("failed_cycles", {})

    all_dfs = []

    # Reload previously completed cycles
    for cyc in completed_cycles:
        cyc_pq = interim_dir / f"cycle_{cyc}.parquet"
        if cyc_pq.exists():
            try:
                all_dfs.append(pd.read_parquet(cyc_pq))
            except Exception:
                pass

    total = len(cycles)
    for idx, cycle_date in enumerate(cycles):
        if cycle_date in completed_cycles:
            continue

        print(f"[{idx+1}/{total}] Fetching settlement cycle {cycle_date}...", end=" ", flush=True)
        res = fetch_consolidated_short_interest(
            start_date=cycle_date,
            end_date=cycle_date,
            ticker=None,
            raw_dir=raw_d
        )

        status = res["status"]
        if status in {"SUCCESS", "EMPTY_DATASET"}:
            df_norm = res["normalized_df"]
            rec_cnt = len(df_norm)
            print(f"SUCCESS: {rec_cnt} securities")
            # Save interim cycle parquet
            cyc_pq = interim_dir / f"cycle_{cycle_date}.parquet"
            df_norm.to_parquet(cyc_pq, index=False)
            all_dfs.append(df_norm)
            checkpoint.setdefault("completed_cycles", {})[cycle_date] = rec_cnt
            failed_cycles.pop(cycle_date, None)
        else:
            err_msg = res.get("message", "Unknown error")
            print(f"FAILED: {err_msg}")
            failed_cycles[cycle_date] = err_msg

        checkpoint["failed_cycles"] = failed_cycles
        checkpoint["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        if delay_between_calls > 0:
            time.sleep(delay_between_calls)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(
            subset=["ticker", "settlement_date"]
        ).sort_values(["ticker", "settlement_date"]).reset_index(drop=True)
    else:
        full_df = normalize_finra_records([])

    # Filter to target universe if provided
    if target_tickers is not None and not full_df.empty:
        clean_target = {t.strip().upper() for t in target_tickers}
        filtered_df = full_df[full_df["ticker"].isin(clean_target)].reset_index(drop=True)
    else:
        filtered_df = full_df

    found_tickers = set(filtered_df["ticker"].unique()) if not filtered_df.empty else set()
    missing_tickers = (set(target_tickers) - found_tickers) if target_tickers else set()

    report = {
        "mode": "cycle",
        "total_cycles_requested": len(cycles),
        "completed_cycles": len(checkpoint.get("completed_cycles", {})),
        "failed_cycles": len(checkpoint.get("failed_cycles", {})),
        "total_records_all_securities": len(full_df),
        "total_records_target_universe": len(filtered_df),
        "target_tickers_requested": len(target_tickers) if target_tickers else None,
        "target_tickers_found": len(found_tickers),
        "target_tickers_missing": len(missing_tickers),
        "date_min": str(filtered_df["settlement_date"].min()) if not filtered_df.empty else None,
        "date_max": str(filtered_df["settlement_date"].max()) if not filtered_df.empty else None,
        "unique_settlement_dates": int(filtered_df["settlement_date"].nunique()) if not filtered_df.empty else 0,
        "failed_cycle_details": checkpoint.get("failed_cycles", {})
    }
    return filtered_df, report

def save_collected_observations(
    df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_PROCESSED_DIR,
    file_stem: str = "finra_short_interest_2017_2020"
) -> tuple[Path, Path]:
    """Save normalized short-interest observations to parquet and CSV."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"{file_stem}.parquet"
    csv_path = out_dir / f"{file_stem}.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    return parquet_path, csv_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect historical FINRA consolidated short interest for Phase 2.")
    parser.add_argument("--pilot", type=int, default=None, help="Run pilot collection on top N tickers (e.g. 10)")
    parser.add_argument("--mode", choices=["ticker", "cycle"], default="cycle", help="Collection mode: 'cycle' (fastest, recommended) or 'ticker'")
    parser.add_argument("--news-path", default=DEFAULT_NEWS_PARQUET, help="Path to historical news parquet")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API requests in seconds")
    parser.add_argument("--output-stem", default=None, help="Stem name for output parquet/CSV files")
    args = parser.parse_args()

    print("=" * 70)
    print("FINRA Historical Consolidated Short-Interest Collector")
    print("=" * 70)

    target_tickers, min_d, max_d = load_target_universe(args.news_path)
    print(f"Target News Universe: {len(target_tickers):,} tickers ({min_d} to {max_d})")
    print(f"Required Observation Period: {EARLIEST_SETTLEMENT} to {LATEST_SETTLEMENT}")

    if args.pilot is not None:
        pilot_n = args.pilot
        pilot_tickers = get_top_tickers_by_event_count(args.news_path, top_n=pilot_n)
        print(f"\n--- Running Pilot Collection ({pilot_n} tickers) ---")
        print(f"Pilot Tickers: {pilot_tickers}")
        df_pilot, report = run_ticker_collection(
            tickers=pilot_tickers,
            start_date=EARLIEST_SETTLEMENT,
            end_date=LATEST_SETTLEMENT,
            delay_between_calls=args.delay
        )
        stem = args.output_stem or f"finra_short_interest_pilot_{pilot_n}"
        p_path, c_path = save_collected_observations(df_pilot, file_stem=stem)
        print("\n--- Pilot Results ---")
        print(f"Successful Tickers: {report['successful_tickers']}/{report['requested_tickers']}")
        print(f"Empty Tickers:      {report['empty_tickers']}")
        print(f"Failed Tickers:     {report['failed_tickers']}")
        print(f"Total Observations: {report['total_records']:,}")
        print(f"Date Coverage:      {report['date_min']} to {report['date_max']} ({report['unique_settlement_dates']} settlement dates)")
        print(f"Saved: {p_path} ({p_path.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"\n--- Running Full Collection (Mode: {args.mode}) ---")
        if args.mode == "cycle":
            df_full, report = run_cycle_collection(
                target_tickers=set(target_tickers),
                cycles=HISTORICAL_CYCLES_2017_2020,
                delay_between_calls=args.delay
            )
        else:
            df_full, report = run_ticker_collection(
                tickers=target_tickers,
                start_date=EARLIEST_SETTLEMENT,
                end_date=LATEST_SETTLEMENT,
                delay_between_calls=args.delay
            )

        stem = args.output_stem or "finra_short_interest_2017_2020"
        p_path, c_path = save_collected_observations(df_full, file_stem=stem)
        print("\n--- Full Collection Results ---")
        print(f"Total Records Target Universe: {report['total_records_target_universe']:,}")
        print(f"Target Tickers Found:          {report['target_tickers_found']:,}/{report['target_tickers_requested']:,}")
        print(f"Target Tickers Missing:        {report['target_tickers_missing']:,}")
        print(f"Date Coverage:                 {report['date_min']} to {report['date_max']} ({report['unique_settlement_dates']} settlement dates)")
        print(f"Completed Cycles / Tickers:    {report.get('completed_cycles', report.get('successful_tickers'))}")
        print(f"Failed Cycles / Tickers:       {report.get('failed_cycles', report.get('failed_tickers'))}")
        print(f"Saved: {p_path} ({p_path.stat().st_size / (1024*1024):.2f} MB)")

    print("=" * 70)

if __name__ == "__main__":
    main()
