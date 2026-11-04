"""Historical News Dataset Preparation for Phase 2.

Extracts, normalizes, and audits news events from `PHASE 1/sitp_dataset.parquet` (read-only)
for the supported FINRA short-interest date range (2017-12-29 through 2020-06-11).

Key Principles:
1. Read-Only Reference:
   - Does not modify Phase 1 or its underlying parquet file.
2. Date-Only Semantics:
   - Preserves exact calendar date without inventing intraday release timestamps.
3. Event Identifiers:
   - Assigns a unique, stable `event_id` to every event.
4. Headline Syndication Auditing:
   - Flags identical headlines spanning multiple tickers (e.g., market roundups)
     and calculates ticker frequency per headline without silent deletion.
5. Ground Truth Hygiene:
   - Segregates legacy Phase 1 labels as audit-only reference columns; does NOT
     treat Phase 1 labels as Phase 2 ground truth.
"""
import sys
from pathlib import Path

# Ensure PHASE 2 root is in sys.path when executed directly
_phase2_root = Path(__file__).resolve().parent.parent
if str(_phase2_root) not in sys.path:
    sys.path.insert(0, str(_phase2_root))

import pandas as pd
from config import ROOT, DATA_PROCESSED

DEFAULT_SOURCE_PATH = ROOT.parent / "PHASE 1" / "sitp_dataset.parquet"
DEFAULT_START_DATE = "2017-12-29"
DEFAULT_END_DATE = "2020-06-11"

def extract_historical_news(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE
) -> tuple[pd.DataFrame, dict]:
    """Extract and normalize historical events from Phase 1 parquet within date range.

    Parameters
    ----------
    source_path : str | Path
        Path to Phase 1 sitp_dataset.parquet.
    start_date : str
        Start date inclusive (YYYY-MM-DD). Default: "2017-12-29".
    end_date : str
        End date inclusive (YYYY-MM-DD). Default: "2020-06-11".

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Normalized dataframe and comprehensive data-quality report dict.
    """
    source_p = Path(source_path)
    if not source_p.exists():
        raise FileNotFoundError(f"Phase 1 dataset not found at {source_p}")

    df_raw = pd.read_parquet(source_p)

    # Convert filter dates to integer YYYYMMDD
    start_int = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
    end_int = int(pd.to_datetime(end_date).strftime("%Y%m%d"))

    # Extract slice
    mask = (df_raw["news_date"] >= start_int) & (df_raw["news_date"] <= end_int)
    df = df_raw[mask].copy().reset_index(drop=True)

    # Parse date column preserving date-only semantics
    dates_dt = pd.to_datetime(df["news_date"].astype(str), format="%Y%m%d")
    date_strs = dates_dt.dt.strftime("%Y-%m-%d")

    df["date"] = date_strs
    df["published_date"] = date_strs
    df["published_at"] = date_strs  # Date-only string, no invented time

    # Clean ticker & text
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["title"] = df["title"].astype(str).str.strip()
    df["company"] = df["company"].fillna("").astype(str).str.strip()

    # Assign unique event_id
    df["event_id"] = [f"ev_p1_{i+1:07d}" for i in range(len(df))]

    # Identify headlines across multiple tickers
    title_ticker_counts = df.groupby("title")["ticker"].nunique()
    df["headline_ticker_count"] = df["title"].map(title_ticker_counts)
    df["is_multi_ticker"] = df["headline_ticker_count"] > 1

    # Preserve legacy labels as explicitly segregated reference columns
    if "label_regsho_volume" in df.columns:
        df["phase1_label_regsho_volume_legacy"] = df["label_regsho_volume"]
        df = df.drop(columns=["label_regsho_volume"])

    if "label_short_interest" in df.columns:
        df["phase1_label_short_interest_legacy"] = df["label_short_interest"]
        df = df.drop(columns=["label_short_interest"])

    # Reorder columns
    col_order = [
        "event_id",
        "ticker",
        "company",
        "date",
        "published_date",
        "published_at",
        "news_date",
        "title",
        "n_tokens",
        "headline_ticker_count",
        "is_multi_ticker",
        "phase1_label_short_interest_legacy",
        "phase1_label_regsho_volume_legacy"
    ]
    existing_cols = [c for c in col_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + remaining_cols]

    # Compute data-quality audit report
    unique_titles = df["title"].nunique()
    multi_ticker_titles = (title_ticker_counts > 1).sum()
    multi_ticker_rows = df["is_multi_ticker"].sum()

    report = {
        "total_rows": len(df),
        "unique_tickers": df["ticker"].nunique(),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "unique_dates": df["date"].nunique(),
        "records_per_year": dates_dt.dt.year.value_counts().sort_index().to_dict(),
        "missing_values": df[["event_id", "ticker", "date", "title"]].isnull().sum().to_dict(),
        "exact_duplicates": int(df.duplicated().sum()),
        "ticker_date_title_duplicates": int(df.duplicated(subset=["ticker", "date", "title"]).sum()),
        "unique_headlines": unique_titles,
        "multi_ticker_headlines_count": int(multi_ticker_titles),
        "multi_ticker_headlines_pct": float(round((multi_ticker_titles / unique_titles) * 100, 2)) if unique_titles else 0.0,
        "multi_ticker_records_count": int(multi_ticker_rows),
        "multi_ticker_records_pct": float(round((multi_ticker_rows / len(df)) * 100, 2)) if len(df) else 0.0,
        "top_syndicated_headlines": title_ticker_counts.nlargest(10).to_dict()
    }

    return df, report

def save_extracted_dataset(
    df: pd.DataFrame,
    output_dir: str | Path = DATA_PROCESSED,
    file_stem: str = "historical_news_2017_2020"
) -> tuple[Path, Path]:
    """Save extracted historical news dataset to parquet and CSV in output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"{file_stem}.parquet"
    csv_path = out_dir / f"{file_stem}.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    return parquet_path, csv_path

def main() -> None:
    """Run historical news extraction and display data-quality audit report."""
    print("=" * 70)
    print("Phase 2 Historical News Preparation from Phase 1 Parquet")
    print("=" * 70)
    df, report = extract_historical_news()

    parquet_p, csv_p = save_extracted_dataset(df)

    print(f"\nExtraction Complete:")
    print(f"  Total records:      {report['total_rows']:,}")
    print(f"  Unique tickers:     {report['unique_tickers']:,}")
    print(f"  Date range:         {report['date_min']} to {report['date_max']} ({report['unique_dates']} unique dates)")
    print(f"  Exact duplicates:   {report['exact_duplicates']}")
    print(f"  [ticker, date, title] duplicates: {report['ticker_date_title_duplicates']}")
    print("\nRecords per year:")
    for yr, cnt in report["records_per_year"].items():
        print(f"  {yr}: {cnt:,}")

    print("\nHeadline Syndication Audit:")
    print(f"  Unique headlines:   {report['unique_headlines']:,}")
    print(f"  Multi-ticker titles:{report['multi_ticker_headlines_count']:,} ({report['multi_ticker_headlines_pct']}%)")
    print(f"  Multi-ticker records:{report['multi_ticker_records_count']:,} ({report['multi_ticker_records_pct']}%)")

    print("\nSaved Artifacts:")
    print(f"  Parquet: {parquet_p} ({parquet_p.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  CSV:     {csv_p} ({csv_p.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 70)

if __name__ == "__main__":
    main()
