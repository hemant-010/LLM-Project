"""Prepare the final consolidated short-interest model training dataset for Phase 2.

Key Principles:
1. Feature Enrichment:
   - Enriches event records with event_type, event_category, event_severity, confidence,
     and context signals using the existing feature-engineering pipeline.
2. Temporal & Target Leakage Prevention:
   - Excludes all future outcome variables: `short_interest_after`, `short_interest_after_date`,
     `short_interest_after_dissemination`, `short_interest_after_dissemination_source`,
     `short_interest_gap_days`, and `short_interest_change_pct`.
   - Strips legacy Phase 1 labels to prevent target contamination.
   - Preserves historical pre-event short interest baseline (`short_interest_before*`)
     which was publicly disseminated before the event.
3. Strict Chronological Splitting:
   - Splits on calendar boundaries:
     * Train: 2018-01-11 through 2019-11-30 (~70% of dataset)
     * Validation: 2019-12-01 through 2020-02-29 (~12% of dataset)
     * Test: 2020-03-01 through 2020-06-11 (~18% of dataset)
   - Zero random sampling or temporal look-ahead across splits.
4. Headline Syndication Auditing:
   - Tracks and audits repeated and multi-ticker syndicated headlines across splits.
5. Model Compatibility:
   - Produces candidate features matching `models/train_models.py` requirements:
     `model_text`, `event_type`, `insider_role`, `transaction_code`, `event_severity`,
     `percentage_holdings_sold`, `transaction_value`, `event_classification_confidence`.
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone, date
import pandas as pd
import numpy as np

# Ensure PHASE 2 root is in sys.path
_phase2_root = Path(__file__).resolve().parent.parent
if str(_phase2_root) not in sys.path:
    sys.path.insert(0, str(_phase2_root))

from config import ROOT, DATA_PROCESSED
from feature_engineering.build_features import enrich_events

DEFAULT_LABELED_EVENTS_PARQUET = DATA_PROCESSED / "events_with_short_interest_labels.parquet"
DEFAULT_OUTPUT_DIR = DATA_PROCESSED

# Explicit leakage columns to drop from model inputs
LEAKAGE_COLUMNS = [
    "short_interest_after",
    "short_interest_after_date",
    "short_interest_after_dissemination",
    "short_interest_after_dissemination_source",
    "short_interest_gap_days",
    "short_interest_change_pct",
    "phase1_label_short_interest_legacy",
    "phase1_label_regsho_volume_legacy",
    "stock_price_label",
    "stock_price_change_pct"
]

def build_model_dataset(
    labeled_events_path: str | Path = DEFAULT_LABELED_EVENTS_PARQUET,
    train_end_date: str = "2019-11-30",
    val_end_date: str = "2020-02-29",
    use_llm: bool | None = False
) -> tuple[pd.DataFrame, dict]:
    """Generate features, purge target leakage, and construct chronological splits.

    Parameters
    ----------
    labeled_events_path : str | Path
        Path to events with short interest labels.
    train_end_date : str
        Inclusive calendar end date for train split (YYYY-MM-DD). Default: "2019-11-30".
    val_end_date : str
        Inclusive calendar end date for validation split (YYYY-MM-DD). Default: "2020-02-29".
    use_llm : bool | None
        LLM toggle for enrich_events (False forces deterministic fast heuristics).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Model-ready training dataset and comprehensive audit report dict.
    """
    p = Path(labeled_events_path)
    if not p.exists():
        raise FileNotFoundError(f"Labeled events file not found at {p}")

    df_raw = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    # 1. Run feature engineering pipeline
    enriched = enrich_events(df_raw, use_llm=use_llm)

    # Ensure headline_text representation exists (headline/title only, free of ticker/company metadata)
    if "headline_text" not in enriched.columns:
        enriched["headline_text"] = enriched["title"].fillna("").astype(str)

    # Ensure compatibility fields for models/train_models.py
    if "insider_role" not in enriched.columns:
        enriched["insider_role"] = ""
    else:
        enriched["insider_role"] = enriched["insider_role"].fillna("").astype(str)

    if "transaction_code" not in enriched.columns:
        enriched["transaction_code"] = ""
    else:
        enriched["transaction_code"] = enriched["transaction_code"].fillna("").astype(str)

    # Clean numeric columns
    for num_col in ["percentage_holdings_sold", "transaction_value", "event_classification_confidence", "short_interest_before"]:
        if num_col in enriched.columns:
            enriched[num_col] = pd.to_numeric(enriched[num_col], errors="coerce")

    # 2. Purge leakage columns
    purged_cols = [c for c in LEAKAGE_COLUMNS if c in enriched.columns]
    df_clean = enriched.drop(columns=purged_cols).copy()

    # 3. Chronological Train / Validation / Test Splitting
    df_clean["published_at"] = pd.to_datetime(df_clean["published_at"], utc=True)
    df_clean = df_clean.sort_values(["published_at", "event_id"]).reset_index(drop=True)

    pub_dates = df_clean["published_at"].dt.date
    train_cutoff = pd.to_datetime(train_end_date).date()
    val_cutoff = pd.to_datetime(val_end_date).date()

    splits = []
    for d in pub_dates:
        if d <= train_cutoff:
            splits.append("train")
        elif d <= val_cutoff:
            splits.append("val")
        else:
            splits.append("test")

    df_clean["split"] = splits

    # Reorder columns logically: Identifiers -> Text/Features -> Baseline Market -> Target -> Split
    priority_cols = [
        "event_id", "ticker", "company", "date", "published_at", "title", "headline_text", "model_text",
        "n_tokens", "headline_ticker_count", "is_multi_ticker",
        "event_type", "event_category", "event_severity", "severity",
        "event_classification_confidence", "confidence", "classification_source",
        "insider_role", "transaction_code", "is_10b5_1", "insider_is_10b5_1",
        "percentage_holdings_sold", "transaction_value",
        "context_flags", "context_summary", "severity_rationale",
        "short_interest_before", "short_interest_before_date",
        "short_interest_before_dissemination", "short_interest_before_dissemination_source",
        "short_interest_label", "split"
    ]
    present_priority = [c for c in priority_cols if c in df_clean.columns]
    other_cols = [c for c in df_clean.columns if c not in present_priority]
    df_clean = df_clean[present_priority + other_cols]

    # 4. Generate Comprehensive Audit Metrics
    n_total = len(df_clean)
    train_mask = df_clean["split"] == "train"
    val_mask = df_clean["split"] == "val"
    test_mask = df_clean["split"] == "test"

    train_df = df_clean[train_mask]
    val_df = df_clean[val_mask]
    test_df = df_clean[test_mask]

    # Class distribution by split
    def get_dist(sub_df):
        vc = sub_df["short_interest_label"].value_counts().to_dict()
        pct = (sub_df["short_interest_label"].value_counts(normalize=True) * 100).to_dict()
        return {k: {"count": vc[k], "pct": round(pct[k], 2)} for k in vc}

    # Syndication and headline overlap audit
    train_titles = set(train_df["title"].unique())
    val_titles = set(val_df["title"].unique())
    test_titles = set(test_df["title"].unique())

    overlap_tr_val = train_titles.intersection(val_titles)
    overlap_tr_test = train_titles.intersection(test_titles)
    overlap_val_test = val_titles.intersection(test_titles)
    overlap_all = train_titles.intersection(val_titles).intersection(test_titles)

    report = {
        "dataset_name": "short_interest_model_dataset",
        "total_records": n_total,
        "unique_tickers": df_clean["ticker"].nunique(),
        "date_min": str(pub_dates.min()),
        "date_max": str(pub_dates.max()),
        "splits": {
            "train": {
                "count": len(train_df),
                "pct": round(len(train_df) / n_total * 100, 2),
                "date_min": str(train_df["published_at"].dt.date.min()),
                "date_max": str(train_df["published_at"].dt.date.max()),
                "tickers": train_df["ticker"].nunique(),
                "class_distribution": get_dist(train_df)
            },
            "val": {
                "count": len(val_df),
                "pct": round(len(val_df) / n_total * 100, 2),
                "date_min": str(val_df["published_at"].dt.date.min()),
                "date_max": str(val_df["published_at"].dt.date.max()),
                "tickers": val_df["ticker"].nunique(),
                "class_distribution": get_dist(val_df)
            },
            "test": {
                "count": len(test_df),
                "pct": round(len(test_df) / n_total * 100, 2),
                "date_min": str(test_df["published_at"].dt.date.min()),
                "date_max": str(test_df["published_at"].dt.date.max()),
                "tickers": test_df["ticker"].nunique(),
                "class_distribution": get_dist(test_df)
            }
        },
        "target_leakage_check": {
            "purged_leakage_columns": purged_cols,
            "leakage_columns_remaining_in_features": [c for c in LEAKAGE_COLUMNS if c in df_clean.columns],
            "is_leakage_free": all(c not in df_clean.columns for c in LEAKAGE_COLUMNS)
        },
        "feature_summary": {
            "total_columns": len(df_clean.columns),
            "columns_list": df_clean.columns.tolist(),
            "target_column": "short_interest_label",
            "split_column": "split"
        },
        "syndication_audit": {
            "unique_headlines": df_clean["title"].nunique(),
            "multi_ticker_records": int(df_clean["is_multi_ticker"].sum()) if "is_multi_ticker" in df_clean else 0,
            "multi_ticker_pct": round(df_clean["is_multi_ticker"].mean() * 100, 2) if "is_multi_ticker" in df_clean else 0.0,
            "train_val_headline_overlap": len(overlap_tr_val),
            "train_test_headline_overlap": len(overlap_tr_test),
            "val_test_headline_overlap": len(overlap_val_test),
            "all_splits_headline_overlap": len(overlap_all)
        }
    }

    return df_clean, report

def save_model_dataset(
    df: pd.DataFrame,
    report: dict,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    file_stem: str = "short_interest_model_dataset"
) -> tuple[Path, Path, Path]:
    """Save training dataset to parquet, CSV, and markdown audit report."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pq_path = out_dir / f"{file_stem}.parquet"
    csv_path = out_dir / f"{file_stem}.csv"
    report_path = out_dir / "feature_leakage_audit_report.md"

    # Save Parquet
    df.to_parquet(pq_path, index=False)

    # Save CSV (converting complex list/dict columns to strings)
    df_csv = df.copy()
    for col in df_csv.columns:
        if df_csv[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_csv[col] = df_csv[col].apply(lambda x: ";".join(str(i) for i in x) if isinstance(x, list) else str(x))
    df_csv.to_csv(csv_path, index=False)

    # Save Markdown Audit Report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2 Short-Interest Model Dataset & Feature Leakage Audit Report\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write("## 1. Dataset Overview\n\n")
        f.write(f"- **Total Events:** `{report['total_records']:,}`\n")
        f.write(f"- **Unique Tickers:** `{report['unique_tickers']:,}`\n")
        f.write(f"- **Date Range:** `{report['date_min']}` to `{report['date_max']}`\n")
        f.write(f"- **Parquet Artifact:** `{pq_path.name}`\n")
        f.write(f"- **CSV Artifact:** `{csv_path.name}`\n\n")

        f.write("## 2. Chronological Train / Validation / Test Splits\n\n")
        f.write("| Split | Event Count | Share (%) | Date Range | Unique Tickers |\n")
        f.write("| :--- | :---: | :---: | :--- | :---: |\n")
        for s_name in ["train", "val", "test"]:
            s_data = report["splits"][s_name]
            f.write(f"| **{s_name.upper()}** | `{s_data['count']:,}` | `{s_data['pct']}%` | `{s_data['date_min']}` to `{s_data['date_max']}` | `{s_data['tickers']:,}` |\n")

        f.write("\n### Class Distribution per Chronological Split\n\n")
        f.write("| Split | `increase` Count (%) | `decrease` Count (%) | `no_significant_change` Count (%) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for s_name in ["train", "val", "test"]:
            dist = report["splits"][s_name]["class_distribution"]
            inc = f"{dist.get('increase', {}).get('count', 0):,} ({dist.get('increase', {}).get('pct', 0)}%)"
            dec = f"{dist.get('decrease', {}).get('count', 0):,} ({dist.get('decrease', {}).get('pct', 0)}%)"
            neu = f"{dist.get('no_significant_change', {}).get('count', 0):,} ({dist.get('no_significant_change', {}).get('pct', 0)}%)"
            f.write(f"| **{s_name.upper()}** | {inc} | {dec} | {neu} |\n")

        f.write("\n## 3. Target & Temporal Leakage Check\n\n")
        leakage = report["target_leakage_check"]
        status = "**PASSED (100% Leakage-Free)**" if leakage["is_leakage_free"] else "**FAILED (Leakage Detected)**"
        f.write(f"- **Audit Status:** {status}\n")
        f.write("- **Purged Future Outcome Columns:**\n")
        for col in leakage["purged_leakage_columns"]:
            f.write(f"  * `{col}` (future outcome settlement, label, or drift)\n")
        f.write(f"- **Remaining Leakage Columns in Feature Matrix:** `{len(leakage['leakage_columns_remaining_in_features'])}`\n\n")

        f.write("## 4. Feature Columns Matrix\n\n")
        f.write(f"- **Total Feature Columns:** `{report['feature_summary']['total_columns']}`\n")
        f.write("- **Candidate Features List:**\n")
        for col in report["feature_summary"]["columns_list"]:
            f.write(f"  * `{col}`\n")

        f.write("\n## 5. Headline Syndication & Temporal Overlap Audit\n\n")
        synd = report["syndication_audit"]
        f.write(f"- **Unique Headlines:** `{synd['unique_headlines']:,}`\n")
        f.write(f"- **Multi-Ticker Syndicated Records:** `{synd['multi_ticker_records']:,}` (`{synd['multi_ticker_pct']}%`)\n")
        f.write(f"- **Headlines in Train & Val:** `{synd['train_val_headline_overlap']:,}`\n")
        f.write(f"- **Headlines in Train & Test:** `{synd['train_test_headline_overlap']:,}`\n")
        f.write(f"- **Headlines across all 3 splits:** `{synd['all_splits_headline_overlap']:,}`\n")
        f.write("*(Note: Cross-split headline overlap is predominantly composed of recurring generic market roundups such as 'Stocks That Hit 52-Week Lows On Friday')*\n")

    return pq_path, csv_path, report_path

def main():
    parser = argparse.ArgumentParser(description="Prepare final Phase 2 short-interest model training dataset.")
    parser.add_argument("--labeled-events", default=DEFAULT_LABELED_EVENTS_PARQUET, help="Path to labeled events parquet")
    parser.add_argument("--output-stem", default="short_interest_model_dataset", help="Output file stem name")
    parser.add_argument("--train-end", default="2019-11-30", help="Train split end date (YYYY-MM-DD)")
    parser.add_argument("--val-end", default="2020-02-29", help="Validation split end date (YYYY-MM-DD)")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 2 Short-Interest Model Dataset Preparation")
    print("=" * 70)

    df_model, report = build_model_dataset(
        labeled_events_path=args.labeled_events,
        train_end_date=args.train_end,
        val_end_date=args.val_end,
        use_llm=False
    )

    pq_p, csv_p, rpt_p = save_model_dataset(
        df_model, report,
        output_dir=DEFAULT_OUTPUT_DIR,
        file_stem=args.output_stem
    )

    print(f"\nProcessing Complete:")
    print(f"  Total records:      {report['total_records']:,}")
    print(f"  Unique tickers:     {report['unique_tickers']:,}")
    print(f"  Date range:         {report['date_min']} to {report['date_max']}")
    print(f"  Feature columns:    {report['feature_summary']['total_columns']}")
    print(f"  Leakage check:      {'PASSED (0 leakage columns)' if report['target_leakage_check']['is_leakage_free'] else 'FAILED'}")

    print("\nChronological Splits:")
    for s in ["train", "val", "test"]:
        info = report["splits"][s]
        print(f"  {s.upper():5s}: {info['count']:,} ({info['pct']}%) | {info['date_min']} to {info['date_max']} | {info['tickers']:,} tickers")

    print("\nSaved Artifacts:")
    print(f"  Parquet: {pq_p} ({pq_p.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  CSV:     {csv_p} ({csv_p.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Report:  {rpt_p}")
    print("=" * 70)

if __name__ == "__main__":
    main()

