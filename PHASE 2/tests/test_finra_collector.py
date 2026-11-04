import pytest
import json
import base64
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
from preprocessing.finra_short_interest_collector import (
    get_access_token,
    build_query_payload,
    normalize_finra_records,
    check_historical_availability,
    fetch_consolidated_short_interest,
    FINRA_HISTORICAL_RETENTION_YEARS,
    FINRA_EARLIEST_SETTLEMENT_DATE
)

def test_finra_token_acquisition():
    """Verify OAuth2 Basic Authentication token acquisition against FIP endpoint."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token": "mocked_bearer_token_xyz",
        "token_type": "Bearer",
        "expires_in": 3600
    }
    mock_session.post.return_value = mock_resp

    client_id = "test_client_id"
    client_secret = "test_client_secret"

    token = get_access_token(client_id, client_secret, session=mock_session)
    assert token == "mocked_bearer_token_xyz"

    # Verify authorization header
    call_args = mock_session.post.call_args
    headers = call_args[1]["headers"]
    expected_b64 = base64.b64encode(b"test_client_id:test_client_secret").decode("utf-8")
    assert headers["Authorization"] == f"Basic {expected_b64}"

def test_finra_missing_credentials_raises(monkeypatch):
    """Verify ValueError is raised if FINRA credentials are not provided."""
    monkeypatch.delenv("FINRA_CLIENT_ID", raising=False)
    monkeypatch.delenv("FINRA_API_KEY", raising=False)
    monkeypatch.delenv("FINRA_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("FINRA_API_SECRET", raising=False)

    with pytest.raises(ValueError, match="Missing FINRA API credentials"):
        get_access_token(client_id="", client_secret="")

def test_finra_build_query_payload():
    """Verify JSON query payload builder for single-date, range-dates, and ticker filters."""
    payload = build_query_payload(
        start_date="2024-01-01",
        end_date="2024-01-31",
        ticker="aapl",
        limit=100,
        offset=200
    )
    assert payload["limit"] == 100
    assert payload["offset"] == 200
    assert len(payload["compareFilters"]) == 3

    # Date filters (GREATER_THAN_EQUAL and LESS_THAN_EQUAL)
    gte_filter = next(f for f in payload["compareFilters"] if f["compareType"] == "GREATER_THAN_EQUAL")
    assert gte_filter["fieldName"] == "settlementDate"
    assert gte_filter["fieldValue"] == "2024-01-01"

    lte_filter = next(f for f in payload["compareFilters"] if f["compareType"] == "LESS_THAN_EQUAL")
    assert lte_filter["fieldName"] == "settlementDate"
    assert lte_filter["fieldValue"] == "2024-01-31"

    # Ticker filter
    ticker_filter = next(f for f in payload["compareFilters"] if f["fieldName"] == "symbolCode")
    assert ticker_filter["compareType"] == "EQUAL"
    assert ticker_filter["fieldValue"] == "AAPL"

def test_finra_normalization_and_dissemination():
    """Verify raw FINRA API response normalization into columns required by label_short_interest.py."""
    raw_records = [
        {
            "issueSymbolIdentifier": "TSLA",
            "issueName": "TESLA INC",
            "settlementDate": "2024-01-15",
            "currentShortShareNumber": 85000000,
            "previousShortShareNumber": 80000000,
            "changePercent": 6.25,
            "daysToCoverNumber": 1.8,
            "marketCategoryCode": "Q"
        }
    ]
    df = normalize_finra_records(raw_records)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "TSLA"
    assert row["settlement_date"] == "2024-01-15"
    assert row["short_interest"] == 85000000.0
    assert row["previous_short_interest"] == 80000000.0
    assert row["change_pct"] == 6.25
    assert row["days_to_cover"] == 1.8

    # Verify official dissemination date is present and grounded
    assert pd.notna(row["dissemination_date"])
    # 2024-01-15 was MLK Day (holiday). Settlement normalized to 7 market days = 2024-01-24 (date-only)
    assert row["dissemination_date"] == "2024-01-24"
    assert row["dissemination_date_source"] == "finra_rule_4560_schedule"

def test_finra_fetch_with_pagination(tmp_path):
    """Verify multi-page pagination loop fetches all records and increments offset."""
    mock_session = MagicMock()

    # Page 1: 3 records (limit=3) -> triggers next page
    page1 = [
        {"issueSymbolIdentifier": "AAA", "settlementDate": "2024-01-15", "currentShortShareNumber": 1000},
        {"issueSymbolIdentifier": "BBB", "settlementDate": "2024-01-15", "currentShortShareNumber": 2000},
        {"issueSymbolIdentifier": "CCC", "settlementDate": "2024-01-15", "currentShortShareNumber": 3000},
    ]
    # Page 2: 1 record (< limit) -> ends pagination
    page2 = [
        {"issueSymbolIdentifier": "DDD", "settlementDate": "2024-01-15", "currentShortShareNumber": 4000},
    ]

    resp1 = MagicMock()
    resp1.status_code = 200
    resp1.json.return_value = page1

    resp2 = MagicMock()
    resp2.status_code = 200
    resp2.json.return_value = page2

    mock_session.post.side_effect = [resp1, resp2]

    with patch("preprocessing.finra_short_interest_collector.get_access_token", return_value="test_token"):
        res = fetch_consolidated_short_interest(
            start_date="2024-01-15",
            end_date="2024-01-15",
            client_id="id",
            client_secret="sec",
            limit=3,
            raw_dir=tmp_path,
            session=mock_session
        )

    assert res["status"] == "SUCCESS"
    assert res["total_records"] == 4
    assert len(res["normalized_df"]) == 4
    assert res["raw_file_path"] is not None
    assert Path(res["raw_file_path"]).exists()

def test_finra_unavailable_historical_data():
    """Verify that dates prior to FINRA's earliest available record return UNAVAILABLE_HISTORICAL_DATA."""
    # Date from 2016 is prior to 2017-12-29
    res = fetch_consolidated_short_interest(start_date="2016-06-15")
    assert res["status"] == "UNAVAILABLE_HISTORICAL_DATA"
    assert res["total_records"] == 0
    assert "earliest available historical record" in res["message"]

def test_finra_historical_availability_boundaries():
    """Verify check_historical_availability boundary behavior."""
    # Prior to earliest date
    avail, msg = check_historical_availability("2017-12-28")
    assert not avail
    assert "prior to FINRA API's earliest available" in msg

    # Exactly earliest date
    avail, msg = check_historical_availability("2017-12-29")
    assert avail
    assert "within the supported historical API coverage" in msg

    # Recent date
    avail, msg = check_historical_availability("2024-01-15")
    assert avail

    # None / empty date
    avail, msg = check_historical_availability(None)
    assert avail

def test_finra_empty_dataset_distinction(tmp_path):
    """Verify that a valid recent query with no matching records returns EMPTY_DATASET."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = []
    mock_session.post.return_value = mock_resp

    recent_date = (pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=6)).strftime("%Y-%m-%d")

    with patch("preprocessing.finra_short_interest_collector.get_access_token", return_value="test_token"):
        res = fetch_consolidated_short_interest(
            start_date=recent_date,
            client_id="id",
            client_secret="sec",
            raw_dir=tmp_path,
            session=mock_session
        )

    assert res["status"] == "EMPTY_DATASET"
    assert res["total_records"] == 0
    assert "no short interest records were returned" in res["message"].lower()

def test_finra_rate_limit_retry(tmp_path):
    """Verify HTTP 429 rate-limit response triggers retry and succeeds."""
    mock_session = MagicMock()

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0.01"}

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = [
        {"issueSymbolIdentifier": "RETRY", "settlementDate": "2024-02-15", "currentShortShareNumber": 500}
    ]

    mock_session.post.side_effect = [resp_429, resp_200]

    with patch("preprocessing.finra_short_interest_collector.get_access_token", return_value="test_token"):
        res = fetch_consolidated_short_interest(
            start_date="2024-02-15",
            client_id="id",
            client_secret="sec",
            raw_dir=tmp_path,
            session=mock_session,
            backoff_factor=0.01
        )

    assert res["status"] == "SUCCESS"
    assert res["total_records"] == 1
    assert res["normalized_df"].iloc[0]["ticker"] == "RETRY"

