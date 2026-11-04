import pytest
import json
import pandas as pd
from pathlib import Path
from preprocessing.news_loader import load_news
from feature_engineering.build_features import enrich_events
from models.predictor import DualPredictor
from alerts.alert_generator import build_alert, render_text
from scripts.run_pipeline import main as run_pipeline_main

def test_alert_generation_structure():
    row = {
        "ticker": "AAPL",
        "title": "CEO disposed of shares under Rule 10b5-1 plan",
        "published_at": "2024-01-15T12:00:00Z",
        "event_type": "insider_selling",
        "event_severity": "low",
        "event_classification_confidence": 0.85,
        "context_flags": "Transaction is marked as a 10b5-1 plan; interpret with additional context.",
        "source": "SEC EDGAR",
        "url": "https://sec.gov/filing/1"
    }
    dummy_preds = {
        "short_interest": {"status": "not_trained", "message": "Model artifact not found"},
        "stock_price": {"status": "not_trained", "message": "Model artifact not found"}
    }
    alert = build_alert(row, dummy_preds)

    assert alert["ticker"] == "AAPL"
    assert alert["headline"] == row["title"]
    assert alert["event_type"] == "insider_selling"
    assert alert["event_severity"] == "low"
    assert "disclaimer" in alert
    assert "Research output, not investment advice" in alert["disclaimer"]
    assert alert["short_interest_prediction"]["status"] == "not_trained"

    rendered = render_text(alert)
    assert "[AAPL]" in rendered
    assert "Event: insider_selling" in rendered
    assert "NOTE:" in rendered

def test_pipeline_end_to_end_execution(tmp_path, monkeypatch):
    input_csv = tmp_path / "news_input.csv"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch config.DATA_PROCESSED so test output writes to tmp_path
    monkeypatch.setattr("scripts.run_pipeline.DATA_PROCESSED", processed_dir)

    df_news = pd.DataFrame([
        {
            "title": "Major Automotive Manufacturer Announces Safety Recall for 500k Vehicles",
            "description": "Brake defect prompts nationwide product recall.",
            "ticker": "F",
            "published_at": "2024-02-10 10:30:00+00:00",
            "source": "Reuters",
            "url": "https://reuters.com/1"
        },
        {
            "title": "Biotech Company Faces Federal Fraud Investigation",
            "description": "DOJ and SEC probe accounting irregularities.",
            "ticker": "BIOT",
            "published_at": "2024-02-11 14:00:00+00:00",
            "source": "Bloomberg",
            "url": "https://bloomberg.com/2"
        }
    ])
    df_news.to_csv(input_csv, index=False)

    run_pipeline_main(str(input_csv))

    csv_out = processed_dir / "event_analysis.csv"
    jsonl_out = processed_dir / "investor_alerts.jsonl"

    assert csv_out.exists()
    assert jsonl_out.exists()

    df_analysis = pd.read_csv(csv_out)
    assert len(df_analysis) == 2
    assert "event_id" in df_analysis.columns
    assert "event_type" in df_analysis.columns

    with jsonl_out.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert lines[0]["ticker"] == "F"
    assert lines[0]["event_type"] == "product_recall"
    assert lines[1]["ticker"] == "BIOT"
    assert lines[1]["event_type"] == "fraud_investigation"

