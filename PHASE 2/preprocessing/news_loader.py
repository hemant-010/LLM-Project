from pathlib import Path
import pandas as pd

REQUIRED = {"title"}

def load_news(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"News CSV is missing required columns: {sorted(missing)}")
    defaults = {
        "description": "", "ticker": "", "published_at": "",
        "source": "", "url": "", "insider_role": "",
        "transaction_code": "", "shares_sold": None,
        "shares_owned_before": None, "shares_owned_after": None,
        "transaction_value": None, "is_10b5_1": None
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    for col in ["title", "description", "ticker", "source", "url",
                "insider_role", "transaction_code"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    for col in ["shares_sold", "shares_owned_before", "shares_owned_after", "transaction_value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df[df["title"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["ticker", "published_at", "title"])
    return df.reset_index(drop=True)
