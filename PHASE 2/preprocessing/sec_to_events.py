"""Convert parsed SEC Form 4 rows into filing-derived event rows."""
import argparse
import pandas as pd

def convert(input_path, output_path):
    df = pd.read_csv(input_path)
    required = {"ticker","filing_date","transaction_date","transaction_code","shares",
                "price_per_share","shares_owned_after","insider_roles","filing_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SEC transaction CSV missing columns: {sorted(missing)}")
    df = df[(df.transaction_code == "S") & (df.shares.notna())].copy()
    if "acquired_disposed" in df.columns:
        # A Form 4 transaction marked D is a disposition; retain only those for this sale experiment.
        df = df[df.acquired_disposed.astype(str).str.upper() == "D"].copy()
    df["shares_owned_after_num"] = pd.to_numeric(df["shares_owned_after"], errors="coerce")
    df["shares_owned_before"] = df["shares"] + df["shares_owned_after_num"]
    df["shares_owned_after"] = df["shares_owned_after_num"]
    df["shares_sold"] = pd.to_numeric(df["shares"], errors="coerce")
    df["transaction_value"] = df["shares_sold"] * pd.to_numeric(df["price_per_share"], errors="coerce")
    df["title"] = "Insider Form 4 reports a sale of shares at " + df["ticker"].astype(str)
    df["description"] = (
        df["insider_roles"].fillna("Reporting owner") + " reported sale of " +
        df["shares_sold"].fillna(0).round(0).astype(int).astype(str) +
        " shares. Transaction date: " + df["transaction_date"].astype(str) +
        ". Filing date: " + df["filing_date"].astype(str) + "."
    )
    df["published_at"] = pd.to_datetime(df["filing_date"], errors="coerce", utc=True)
    df["source"] = "SEC EDGAR Form 4"
    df["url"] = df["filing_url"]
    df["insider_role"] = df["insider_roles"]

    def parse_10b5_1(val):
        if pd.isna(val):
            return "unknown"
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        return "unknown"

    raw_series = df["mentions_10b5_1"] if "mentions_10b5_1" in df.columns else (
        df["is_10b5_1"] if "is_10b5_1" in df.columns else pd.Series("unknown", index=df.index)
    )
    df["is_10b5_1"] = raw_series.map(parse_10b5_1)
    df["source_type"] = "sec_form4_generated_event"
    keep = ["title","description","ticker","published_at","source","url","insider_role",
            "transaction_code","shares_sold","shares_owned_before","shares_owned_after",
            "transaction_value","is_10b5_1","source_type","transaction_date"]
    out = df[keep].dropna(subset=["published_at"])
    out.to_csv(output_path,index=False)
    print(f"Wrote {len(out)} SEC-derived event(s) to {output_path}")
    print("This is filing-derived data, not independent news coverage.")
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/processed/sec_form4_transactions.csv")
    p.add_argument("--output",default="data/raw/news/sec_form4_events.csv")
    a=p.parse_args()
    convert(a.input,a.output)
