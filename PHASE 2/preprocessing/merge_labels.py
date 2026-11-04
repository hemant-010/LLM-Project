"""Merge independently generated price and short-interest labels by stable event_id."""
import argparse
import pandas as pd

def main(si_path, price_path, output):
    si = pd.read_csv(si_path)
    price = pd.read_csv(price_path)
    if "event_id" not in si.columns or "event_id" not in price.columns:
        raise ValueError("Both labelled files must contain event_id. Run the feature pipeline first.")
    if "short_interest_label" not in si.columns:
        raise ValueError("Short-interest file is missing 'short_interest_label' column.")
    if "stock_price_label" not in price.columns:
        raise ValueError("Stock-price file is missing 'stock_price_label' column.")

    base = si.drop_duplicates("event_id")
    if "stock_price_label" in base.columns:
        base = base.drop(columns=["stock_price_label"])

    price_subset = price[["event_id", "stock_price_label"]].drop_duplicates("event_id")
    merged = base.merge(price_subset, on="event_id", how="inner")

    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    merged.to_csv(output, index=False)
    print(f"Wrote {len(merged)} rows with both targets to {output}")
    return merged

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--short-interest",required=True)
    p.add_argument("--stock-price",required=True)
    p.add_argument("--output",default="data/processed/training_data.csv")
    a=p.parse_args()
    main(a.short_interest,a.stock_price,a.output)
