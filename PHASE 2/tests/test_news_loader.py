import pytest
import pandas as pd
from pathlib import Path
from preprocessing.news_loader import load_news

def test_load_news_valid_csv(tmp_path):
    csv_file = tmp_path / "valid_news.csv"
    df = pd.DataFrame([
        {
            "title": "Acme Corp Announces Q3 Earnings",
            "description": "Revenue surged 15% year over year.",
            "ticker": "ACME",
            "published_at": "2024-01-15 10:00:00+00:00",
            "source": "PR Newswire",
            "url": "https://example.com/news/1"
        }
    ])
    df.to_csv(csv_file, index=False)

    loaded = load_news(csv_file)
    assert len(loaded) == 1
    assert loaded.loc[0, "title"] == "Acme Corp Announces Q3 Earnings"
    assert loaded.loc[0, "ticker"] == "ACME"
    assert pd.notna(loaded.loc[0, "published_at"])
    # Default columns should be populated
    assert "shares_sold" in loaded.columns
    assert "is_10b5_1" in loaded.columns

def test_load_news_missing_required_column(tmp_path):
    csv_file = tmp_path / "missing_title.csv"
    df = pd.DataFrame([
        {"ticker": "ACME", "description": "No title here"}
    ])
    df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_news(csv_file)

def test_load_news_deduplication(tmp_path):
    csv_file = tmp_path / "duplicates.csv"
    df = pd.DataFrame([
        {"title": "Duplicate headline", "ticker": "XYZ", "published_at": "2024-02-01 12:00:00+00:00"},
        {"title": "Duplicate headline", "ticker": "XYZ", "published_at": "2024-02-01 12:00:00+00:00"},
        {"title": "Unique headline", "ticker": "XYZ", "published_at": "2024-02-01 12:00:00+00:00"},
    ])
    df.to_csv(csv_file, index=False)

    loaded = load_news(csv_file)
    assert len(loaded) == 2
    assert set(loaded["title"]) == {"Duplicate headline", "Unique headline"}

def test_load_news_empty_title_dropped(tmp_path):
    csv_file = tmp_path / "empty_titles.csv"
    df = pd.DataFrame([
        {"title": "", "ticker": "AAA"},
        {"title": "   ", "ticker": "BBB"},
        {"title": "Valid title", "ticker": "CCC"},
    ])
    df.to_csv(csv_file, index=False)

    loaded = load_news(csv_file)
    assert len(loaded) == 1
    assert loaded.iloc[0]["ticker"] == "CCC"

