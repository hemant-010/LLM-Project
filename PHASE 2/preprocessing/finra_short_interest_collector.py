"""Official FINRA Consolidated Short Interest Data Collector for Phase 2.

Architectural Principles:
1. Endpoint & Authentication:
   - Uses FINRA Identity Platform (FIP) OAuth 2.0 to obtain bearer access tokens.
   - Queries official API: `https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest`.
2. Coverage & Retention Limits:
   - FINRA maintains a rolling 5-year retention window for online API queries.
   - The collector explicitly distinguishes historical data outside this window
     (status: UNAVAILABLE_HISTORICAL_DATA) from an empty query (status: EMPTY_DATASET).
3. Dissemination Grounding:
   - Derives verified publication timestamps via official FINRA Rule 4560 schedule
     using exchange-holiday calendars (7 market business days after settlement, 16:00 ET).
4. Raw Audit Persistence:
   - Saves original, unaltered raw JSON responses to `data/raw/market/finra/`.
5. Normalization:
   - Produces clean records compatible with `preprocessing/label_short_interest.py`:
     `ticker`, `settlement_date`, `short_interest`, `dissemination_date`.
"""
import os
import json
import base64
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv
from config import ROOT, DATA_RAW
from preprocessing.label_short_interest import _to_market_release_time, get_official_publication_date

load_dotenv(ROOT / ".env")

FIP_TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
FINRA_API_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
FINRA_EARLIEST_SETTLEMENT_DATE = "2017-12-29"
FINRA_HISTORICAL_RETENTION_YEARS = 8.8  # Empirically verified coverage back to 2017-12-29
DEFAULT_RAW_DIR = DATA_RAW / "market" / "finra"

def get_finra_credentials() -> tuple[str, str]:
    """Retrieve FINRA Client ID and Secret from environment variables."""
    client_id = (
        os.getenv("FINRA_CLIENT_ID") or
        os.getenv("FINRA_API_KEY") or
        ""
    ).strip()
    client_secret = (
        os.getenv("FINRA_CLIENT_SECRET") or
        os.getenv("FINRA_API_SECRET") or
        ""
    ).strip()
    return client_id, client_secret

def get_access_token(
    client_id: str | None = None,
    client_secret: str | None = None,
    token_url: str = FIP_TOKEN_URL,
    session: requests.Session | None = None,
    timeout: float = 30.0
) -> str:
    """Acquire an OAuth 2.0 bearer token from FINRA Identity Platform (FIP)."""
    if not client_id or not client_secret:
        cid, csec = get_finra_credentials()
        client_id = client_id or cid
        client_secret = client_secret or csec

    if not client_id or not client_secret:
        raise ValueError(
            "Missing FINRA API credentials. Set FINRA_CLIENT_ID and FINRA_CLIENT_SECRET in .env."
        )

    creds = f"{client_id}:{client_secret}".encode("utf-8")
    basic_auth = base64.b64encode(creds).decode("utf-8")

    headers = {
        "Authorization": f"Basic {basic_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}

    s = session or requests.Session()
    resp = s.post(token_url, headers=headers, data=data, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"FINRA authentication failed (HTTP {resp.status_code}): {resp.text}"
        )

    token_data = resp.json()
    token = token_data.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token present in FINRA auth response: {token_data}")
    return token

def check_historical_availability(start_date: str | None) -> tuple[bool, str]:
    """Check whether a requested start date falls within FINRA API's supported historical coverage.

    Note on Metadata vs. Empirical Database Availability:
    The official metadata description notes that data is available for 'one rolling year'.
    However, the live API database retains historical observations back to 2017-12-29 (~8.8 years).
    Dates prior to 2017-12-29 are unavailable via the live API.
    """
    if not start_date:
        return True, "No start date provided; querying active retention window."
    try:
        req_dt = pd.to_datetime(start_date, utc=True)
        earliest_dt = pd.to_datetime(FINRA_EARLIEST_SETTLEMENT_DATE, utc=True)
        if req_dt < earliest_dt:
            msg = (
                f"Requested start date {req_dt.date()} is prior to FINRA API's earliest available "
                f"historical record ({FINRA_EARLIEST_SETTLEMENT_DATE}). Consolidated short-interest data "
                f"prior to {FINRA_EARLIEST_SETTLEMENT_DATE} is unavailable via the live API and requires external historical archives."
            )
            return False, msg
        return True, f"Date {req_dt.date()} is within the supported historical API coverage ({FINRA_EARLIEST_SETTLEMENT_DATE} to present)."
    except Exception as e:
        return True, f"Unparseable date '{start_date}': {e}"

def build_query_payload(
    start_date: str | None = None,
    end_date: str | None = None,
    ticker: str | None = None,
    limit: int = 5000,
    offset: int = 0
) -> dict:
    """Construct JSON filter payload for FINRA Query API."""
    compare_filters = []

    if start_date and end_date:
        compare_filters.append({
            "compareType": "GREATER_THAN_EQUAL",
            "fieldName": "settlementDate",
            "fieldValue": start_date
        })
        compare_filters.append({
            "compareType": "LESS_THAN_EQUAL",
            "fieldName": "settlementDate",
            "fieldValue": end_date
        })
    elif start_date:
        compare_filters.append({
            "compareType": "GREATER_THAN_EQUAL",
            "fieldName": "settlementDate",
            "fieldValue": start_date
        })
    elif end_date:
        compare_filters.append({
            "compareType": "LESS_THAN_EQUAL",
            "fieldName": "settlementDate",
            "fieldValue": end_date
        })

    if ticker:
        clean_ticker = str(ticker).strip().upper()
        compare_filters.append({
            "compareType": "EQUAL",
            "fieldName": "symbolCode",
            "fieldValue": clean_ticker
        })

    payload = {
        "compareFilters": compare_filters,
        "limit": limit,
        "offset": offset
    }
    return payload

def normalize_finra_records(records: list[dict]) -> pd.DataFrame:
    """Normalize raw FINRA consolidatedShortInterest records into standard schema."""
    if not records:
        return pd.DataFrame(columns=[
            "ticker", "settlement_date", "short_interest", "dissemination_date",
            "issue_name", "previous_short_interest", "change_pct", "days_to_cover",
            "market_category"
        ])

    rows = []
    for r in records:
        ticker = str(
            r.get("symbolCode") or
            r.get("issueSymbolIdentifier") or
            r.get("ticker") or
            ""
        ).strip().upper()
        settlement = str(r.get("settlementDate") or "").strip()
        si_raw = (
            r.get("currentShortPositionQuantity")
            if r.get("currentShortPositionQuantity") is not None
            else r.get("currentShortShareNumber")
            if r.get("currentShortShareNumber") is not None
            else r.get("short_interest")
        )

        try:
            short_interest = float(si_raw) if si_raw is not None else None
        except (ValueError, TypeError):
            short_interest = None

        prev_si = (
            r.get("previousShortPositionQuantity")
            if r.get("previousShortPositionQuantity") is not None
            else r.get("previousShortShareNumber")
        )
        try:
            prev_short_interest = float(prev_si) if prev_si is not None else None
        except (ValueError, TypeError):
            prev_short_interest = None

        chg_pct = r.get("changePercent")
        try:
            change_pct = float(chg_pct) if chg_pct is not None else None
        except (ValueError, TypeError):
            change_pct = None

        dtc = (
            r.get("daysToCoverQuantity")
            if r.get("daysToCoverQuantity") is not None
            else r.get("daysToCoverNumber")
        )
        try:
            days_to_cover = float(dtc) if dtc is not None else None
        except (ValueError, TypeError):
            days_to_cover = None

        # Derive official publication date via FINRA Rule 4560 schedule
        pub_date = get_official_publication_date(settlement, bday_lag=7)
        pub_date_str = str(pub_date) if pub_date is not None else None

        rows.append({
            "ticker": ticker,
            "settlement_date": settlement,
            "short_interest": short_interest,
            "dissemination_date": pub_date_str,
            "dissemination_date_source": "finra_rule_4560_schedule",
            "issue_name": r.get("issueName") or "",
            "previous_short_interest": prev_short_interest,
            "change_pct": change_pct,
            "days_to_cover": days_to_cover,
            "market_category": r.get("marketClassCode") or r.get("marketCategoryCode") or ""
        })

    return pd.DataFrame(rows)

def fetch_consolidated_short_interest(
    start_date: str | None = None,
    end_date: str | None = None,
    ticker: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    limit: int = 5000,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    timeout: float = 45.0,
    raw_dir: str | Path | None = None,
    session: requests.Session | None = None
) -> dict:
    """Fetch consolidated short interest from FINRA API with pagination and rate-limit handling.

    Returns
    -------
    dict:
        status: "SUCCESS" | "UNAVAILABLE_HISTORICAL_DATA" | "EMPTY_DATASET" | "ERROR"
        raw_records: list of raw dicts
        normalized_df: pd.DataFrame with normalized columns
        raw_file_path: Path to saved raw response or None
        total_records: int
        message: str
    """
    is_available, coverage_msg = check_historical_availability(start_date)
    if not is_available:
        return {
            "status": "UNAVAILABLE_HISTORICAL_DATA",
            "raw_records": [],
            "normalized_df": normalize_finra_records([]),
            "raw_file_path": None,
            "total_records": 0,
            "message": coverage_msg
        }

    s = session or requests.Session()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    cid, csec = get_finra_credentials()
    effective_cid = client_id or cid
    effective_csec = client_secret or csec

    if effective_cid and effective_csec:
        try:
            token = get_access_token(effective_cid, effective_csec, session=s, timeout=timeout)
            headers["Authorization"] = f"Bearer {token}"
        except Exception as e:
            return {
                "status": "ERROR",
                "raw_records": [],
                "normalized_df": normalize_finra_records([]),
                "raw_file_path": None,
                "total_records": 0,
                "message": f"Authentication failure: {e}"
            }

    all_records = []
    offset = 0

    while True:
        payload = build_query_payload(
            start_date=start_date,
            end_date=end_date,
            ticker=ticker,
            limit=limit,
            offset=offset
        )

        success = False
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                resp = s.post(FINRA_API_URL, headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    page_records = data if isinstance(data, list) else data.get("data", [])
                    all_records.extend(page_records)
                    success = True
                    break
                elif resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", backoff_factor * (2 ** attempt)))
                    time.sleep(retry_after)
                    continue
                elif resp.status_code in {500, 502, 503, 504}:
                    time.sleep(backoff_factor * (2 ** attempt))
                    continue
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                    break
            except Exception as ex:
                last_error = str(ex)
                if attempt < max_retries:
                    time.sleep(backoff_factor * (2 ** attempt))
                    continue

        if not success:
            return {
                "status": "ERROR",
                "raw_records": all_records,
                "normalized_df": normalize_finra_records(all_records),
                "raw_file_path": None,
                "total_records": len(all_records),
                "message": f"Failed fetching offset {offset}: {last_error}"
            }

        if len(page_records) < limit:
            break
        offset += limit

    if not all_records:
        return {
            "status": "EMPTY_DATASET",
            "raw_records": [],
            "normalized_df": normalize_finra_records([]),
            "raw_file_path": None,
            "total_records": 0,
            "message": "Query completed successfully, but no short interest records were returned for the given criteria."
        }

    # Save raw responses to disk
    target_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    date_tag = f"{start_date or 'all'}_to_{end_date or 'all'}"
    ticker_tag = f"{ticker}_" if ticker else ""
    raw_path = target_dir / f"finra_short_interest_raw_{ticker_tag}{date_tag}_{ts_tag}.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "start_date": start_date,
                "end_date": end_date,
                "ticker": ticker,
                "record_count": len(all_records),
                "endpoint": FINRA_API_URL
            },
            "records": all_records
        }, f, indent=2)

    df_normalized = normalize_finra_records(all_records)

    return {
        "status": "SUCCESS",
        "raw_records": all_records,
        "normalized_df": df_normalized,
        "raw_file_path": str(raw_path),
        "total_records": len(all_records),
        "message": f"Successfully retrieved {len(all_records)} record(s) from FINRA API."
    }

def main():
    import argparse
    p = argparse.ArgumentParser(description="Collect consolidated short-interest data from official FINRA API.")
    p.add_argument("--start-date", default=None, help="Start settlement date (YYYY-MM-DD)")
    p.add_argument("--end-date", default=None, help="End settlement date (YYYY-MM-DD)")
    p.add_argument("--ticker", default=None, help="Filter by ticker symbol (e.g. AAPL)")
    p.add_argument("--output", default="data/processed/finra_short_interest.csv", help="Normalized output CSV path")
    p.add_argument("--raw-dir", default=None, help="Directory to store raw API JSON responses")
    args = p.parse_args()

    print(f"INFO: Querying FINRA consolidated short interest ({args.start_date or 'earliest'} -> {args.end_date or 'latest'})...")
    res = fetch_consolidated_short_interest(
        start_date=args.start_date,
        end_date=args.end_date,
        ticker=args.ticker,
        raw_dir=args.raw_dir
    )

    print(f"Status: {res['status']}")
    print(f"Message: {res['message']}")
    if res["status"] == "SUCCESS":
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res["normalized_df"].to_csv(out_path, index=False)
        print(f"Wrote {len(res['normalized_df'])} normalized records to {out_path}")
        if res["raw_file_path"]:
            print(f"Saved raw response to {res['raw_file_path']}")

if __name__ == "__main__":
    main()

