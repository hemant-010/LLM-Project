"""Tests for model feature allowlist, leakage protection, and train-only preprocessing."""
import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from models.feature_pipeline import (
    MODEL_FEATURE_ALLOWLIST,
    COMPARATIVE_FEATURE_ALLOWLIST,
    TEXT_FEATURES,
    COMPARATIVE_TEXT_FEATURES,
    DEFAULT_TEXT_FEATURE,
    COMPOSITE_TEXT_FEATURE,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    SPLIT_COLUMN,
    EXCLUDED_COLUMNS_AUDIT,
    sanitize_to_headline_only,
    ensure_text_representations,
    validate_feature_inputs,
    build_feature_preprocessor,
    fit_preprocessor_train_only,
    transform_split,
    get_novel_catalyst_mask,
    prepare_splits_and_transforms
)

@pytest.fixture
def sample_features_df():
    """Construct a synthetic multi-split dataset following the Phase 2 schema."""
    rows = []
    # Train rows
    for i in range(20):
        headline = f"Earnings Announcement for Company {i%5}" if i < 15 else "Weekly Market Roundup Highs"
        rows.append({
            "event_id": f"train_{i}",
            "ticker": f"TICK{i%5}",
            "company": f"Company {i%5}",
            "date": "2018-05-15",
            "published_at": f"2018-05-15T10:{i:02d}:00Z",
            "title": headline,
            "headline_text": headline,
            "model_text": f"Title: {headline} | Company: Company {i%5} | Ticker: TICK{i%5}",
            "n_tokens": 15 + i,
            "headline_ticker_count": 1 if i < 15 else 5,
            "is_multi_ticker": (i >= 15),
            "event_type": "earnings_announcement" if i < 15 else "market_roundup",
            "event_category": "corporate_action" if i < 15 else "market_summary",
            "event_severity": "high" if i % 2 == 0 else "medium",
            "insider_role": "CEO" if i % 3 == 0 else "None",
            "transaction_code": "S" if i % 3 == 0 else "None",
            "percentage_holdings_sold": 10.5 if i % 3 == 0 else 0.0,
            "transaction_value": 250000.0 if i % 3 == 0 else 0.0,
            "event_classification_confidence": 0.90,
            "short_interest_before": 1000000.0 + i * 50000,
            "short_interest_before_date": "2018-04-30",
            "short_interest_before_dissemination": "2018-05-10",
            "short_interest_before_dissemination_source": "finra_rule_4560_schedule",
            "short_interest_label": "increase" if i % 2 == 0 else "decrease",
            "split": "train"
        })

    # Val rows
    for i in range(10):
        headline = f"Val Clinical Update {i}"
        rows.append({
            "event_id": f"val_{i}",
            "ticker": f"TICK{i%5}",
            "company": f"Company {i%5}",
            "date": "2020-01-15",
            "published_at": f"2020-01-15T12:{i:02d}:00Z",
            "title": headline,
            "headline_text": headline,
            "model_text": f"Title: {headline} | Company: Company {i%5} | Ticker: TICK{i%5}",
            "n_tokens": 18,
            "headline_ticker_count": 1,
            "is_multi_ticker": False,
            "event_type": "clinical_trial",
            "event_category": "clinical",
            "event_severity": "high",
            "insider_role": "None",
            "transaction_code": "None",
            "percentage_holdings_sold": 0.0,
            "transaction_value": 0.0,
            "event_classification_confidence": 0.85,
            "short_interest_before": 2000000.0,
            "short_interest_before_date": "2019-12-31",
            "short_interest_before_dissemination": "2020-01-10",
            "short_interest_before_dissemination_source": "finra_rule_4560_schedule",
            "short_interest_label": "no_significant_change",
            "split": "val"
        })

    # Test rows (some novel single-ticker catalysts, some repeated generic headlines)
    for i in range(10):
        is_generic = (i % 2 == 0)
        headline = "Weekly Market Roundup Highs" if is_generic else f"Novel Biotech FDA Approval Catalyst {i}"
        rows.append({
            "event_id": f"test_{i}",
            "ticker": f"TEST{i}",
            "company": f"Test Bio {i}",
            "date": "2020-04-20",
            "published_at": f"2020-04-20T14:{i:02d}:00Z",
            "title": headline,
            "headline_text": headline,
            "model_text": f"Title: {headline} | Company: Test Bio {i} | Ticker: TEST{i}",
            "n_tokens": 20,
            "headline_ticker_count": 10 if is_generic else 1,
            "is_multi_ticker": is_generic,
            "event_type": "market_roundup" if is_generic else "regulatory_approval",
            "event_category": "market_summary" if is_generic else "regulatory",
            "event_severity": "low" if is_generic else "high",
            "insider_role": "None",
            "transaction_code": "None",
            "percentage_holdings_sold": 0.0,
            "transaction_value": 0.0,
            "event_classification_confidence": 0.95,
            "short_interest_before": 3000000.0,
            "short_interest_before_date": "2020-03-31",
            "short_interest_before_dissemination": "2020-04-09",
            "short_interest_before_dissemination_source": "finra_rule_4560_schedule",
            "short_interest_label": "increase" if i % 2 == 0 else "decrease",
            "split": "test"
        })

    return pd.DataFrame(rows)

def test_feature_allowlist_composition():
    """Verify that allowlist contains only intentional features with headline_text as default."""
    assert DEFAULT_TEXT_FEATURE == "headline_text"
    assert COMPOSITE_TEXT_FEATURE == "model_text"
    assert TEXT_FEATURES == ["headline_text"]
    assert COMPARATIVE_TEXT_FEATURES == ["model_text"]

    expected_cat = ["event_type", "event_category", "event_severity", "insider_role", "transaction_code", "is_multi_ticker"]
    assert sorted(CATEGORICAL_FEATURES) == sorted(expected_cat)

    expected_num = ["percentage_holdings_sold", "transaction_value", "event_classification_confidence", "n_tokens", "headline_ticker_count", "short_interest_before"]
    assert sorted(NUMERICAL_FEATURES) == sorted(expected_num)

    assert len(MODEL_FEATURE_ALLOWLIST) == len(TEXT_FEATURES) + len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES)
    assert "headline_text" in MODEL_FEATURE_ALLOWLIST
    assert "model_text" not in MODEL_FEATURE_ALLOWLIST  # model_text excluded from default allowlist

def test_identifiers_and_dates_in_excluded_audit():
    """Verify high-cardinality identifiers, dates, and target leakage are in excluded audit."""
    assert "ticker" in EXCLUDED_COLUMNS_AUDIT
    assert "company" in EXCLUDED_COLUMNS_AUDIT
    assert "event_id" in EXCLUDED_COLUMNS_AUDIT
    assert "published_at" in EXCLUDED_COLUMNS_AUDIT
    assert "date" in EXCLUDED_COLUMNS_AUDIT
    assert "short_interest_label" in EXCLUDED_COLUMNS_AUDIT
    assert "short_interest_after" in EXCLUDED_COLUMNS_AUDIT
    assert "model_text" in EXCLUDED_COLUMNS_AUDIT  # Excluded from default to resolve identity leakage

    for excluded in EXCLUDED_COLUMNS_AUDIT:
        assert excluded not in MODEL_FEATURE_ALLOWLIST, f"{excluded} must not be in MODEL_FEATURE_ALLOWLIST!"

def test_sanitize_to_headline_only():
    """Verify that sanitize_to_headline_only strips out company and ticker metadata tags."""
    composite_1 = "Title: FDA Grants Fast Track Designation | Company: BioPharma Corp | Ticker: BPHT"
    cleaned_1 = sanitize_to_headline_only(composite_1)
    assert cleaned_1 == "FDA Grants Fast Track Designation"
    assert "BioPharma" not in cleaned_1
    assert "BPHT" not in cleaned_1
    assert "Company:" not in cleaned_1
    assert "Ticker:" not in cleaned_1

    composite_2 = "Title: CEO Resigns Effective Immediately | Ticker: XYZ | Company: XYZ Holdings"
    cleaned_2 = sanitize_to_headline_only(composite_2)
    assert cleaned_2 == "CEO Resigns Effective Immediately"
    assert "XYZ" not in cleaned_2

    plain_headline = "Raymond James Upgrades Alcentra Capital to Market Perform"
    assert sanitize_to_headline_only(plain_headline) == plain_headline

def test_ensure_text_representations_purges_ticker_and_company():
    """Verify ensure_text_representations creates headline_text without ticker/company tags."""
    df_raw = pd.DataFrame([{
        "title": "Quarterly Revenue Exceeds Guidance by 15%",
        "company": "NVIDIA Corp",
        "ticker": "NVDA",
        "model_text": "Title: Quarterly Revenue Exceeds Guidance by 15% | Company: NVIDIA Corp | Ticker: NVDA"
    }])

    df_ensured = ensure_text_representations(df_raw)
    assert "headline_text" in df_ensured.columns
    assert "model_text" in df_ensured.columns

    headline = df_ensured["headline_text"].iloc[0]
    assert headline == "Quarterly Revenue Exceeds Guidance by 15%"
    assert "NVDA" not in headline
    assert "NVIDIA" not in headline
    assert "Company:" not in headline
    assert "Ticker:" not in headline

def test_default_preprocessor_prevents_identity_leakage_in_vocabulary():
    """Verify that default preprocessor TF-IDF vocabulary contains no ticker/company tokens."""
    train_df = pd.DataFrame([
        {
            "title": "Hardware Shipment Growth Reported",
            "headline_text": "Hardware Shipment Growth Reported",
            "model_text": "Title: Hardware Shipment Growth Reported | Company: Apple Inc | Ticker: AAPL",
            "ticker": "AAPL",
            "company": "Apple Inc",
            "event_type": "earnings_announcement",
            "event_category": "corporate_action",
            "event_severity": "high",
            "insider_role": "None",
            "transaction_code": "None",
            "is_multi_ticker": False,
            "percentage_holdings_sold": 0.0,
            "transaction_value": 0.0,
            "event_classification_confidence": 0.9,
            "n_tokens": 10,
            "headline_ticker_count": 1,
            "short_interest_before": 1000000.0,
            "short_interest_label": "increase",
            "split": "train"
        },
        {
            "title": "Cloud Computing Infrastructure Expansion",
            "headline_text": "Cloud Computing Infrastructure Expansion",
            "model_text": "Title: Cloud Computing Infrastructure Expansion | Company: Microsoft Corp | Ticker: MSFT",
            "ticker": "MSFT",
            "company": "Microsoft Corp",
            "event_type": "product_launch",
            "event_category": "corporate_action",
            "event_severity": "medium",
            "insider_role": "None",
            "transaction_code": "None",
            "is_multi_ticker": False,
            "percentage_holdings_sold": 0.0,
            "transaction_value": 0.0,
            "event_classification_confidence": 0.85,
            "n_tokens": 10,
            "headline_ticker_count": 1,
            "short_interest_before": 2000000.0,
            "short_interest_label": "decrease",
            "split": "train"
        }
    ])

    # 1. Default preprocessor (headline_text): ticker and company tokens must NOT be present
    prep_default = build_feature_preprocessor(use_headline_only=True, min_df=1)
    fit_preprocessor_train_only(prep_default, train_df, use_headline_only=True)
    vocab_default = set(prep_default.named_transformers_["text"].vocabulary_.keys())

    assert "aapl" not in vocab_default
    assert "apple" not in vocab_default
    assert "msft" not in vocab_default
    assert "microsoft" not in vocab_default
    assert "shipment" in vocab_default
    assert "cloud" in vocab_default

    # 2. Comparative preprocessor (model_text): company and ticker tokens ARE present
    prep_comparative = build_feature_preprocessor(use_headline_only=False, min_df=1)
    fit_preprocessor_train_only(prep_comparative, train_df, use_headline_only=False)
    vocab_comparative = set(prep_comparative.named_transformers_["text"].vocabulary_.keys())

    assert "aapl" in vocab_comparative
    assert "apple" in vocab_comparative
    assert "msft" in vocab_comparative
    assert "microsoft" in vocab_comparative

def test_validate_feature_inputs_rejects_leakage(sample_features_df):
    """Verify that any prohibited target leakage column is detected and rejected."""
    df_leak = sample_features_df.copy()
    df_leak["short_interest_after"] = 500000.0

    with pytest.raises(ValueError, match="Prohibited target/temporal leakage columns detected"):
        validate_feature_inputs(df_leak)

def test_validate_feature_inputs_detects_future_short_interest(sample_features_df):
    """Verify that baseline short interest with future dissemination date raises an error."""
    df_future = sample_features_df.copy()
    df_future.loc[0, "short_interest_before_dissemination"] = "2018-05-20"
    df_future.loc[0, "published_at"] = "2018-05-15T10:00:00Z"

    with pytest.raises(ValueError, match="Pre-event baseline short-interest availability violation"):
        validate_feature_inputs(df_future)

def test_fit_preprocessor_train_only_rejection(sample_features_df):
    """Verify fit_preprocessor_train_only rejects datasets containing non-train rows."""
    prep = build_feature_preprocessor(use_headline_only=True)
    with pytest.raises(ValueError, match="fit_preprocessor_train_only must be called exclusively on 'train' split"):
        fit_preprocessor_train_only(prep, sample_features_df)

def test_train_only_preprocessing_isolation(sample_features_df):
    """Verify that fitting on train only does not absorb vocabulary or stats from val or test."""
    train_df = sample_features_df[sample_features_df["split"] == "train"]
    val_df = sample_features_df[sample_features_df["split"] == "val"]
    test_df = sample_features_df[sample_features_df["split"] == "test"]

    prep = build_feature_preprocessor(use_headline_only=True)
    fitted_prep = fit_preprocessor_train_only(prep, train_df, use_headline_only=True)

    text_vec = fitted_prep.named_transformers_["text"]
    train_vocab = set(text_vec.vocabulary_.keys())

    assert "clinical" not in train_vocab
    assert "biotech" not in train_vocab

    X_val = transform_split(fitted_prep, val_df, use_headline_only=True)
    X_test = transform_split(fitted_prep, test_df, use_headline_only=True)

    assert set(text_vec.vocabulary_.keys()) == train_vocab

    scaler = fitted_prep.named_transformers_["num"].named_steps["scale"]
    var_before = scaler.var_.copy()
    _ = transform_split(fitted_prep, test_df, use_headline_only=True)
    np.testing.assert_array_equal(scaler.var_, var_before)

def test_novel_catalyst_mask(sample_features_df):
    """Verify that secondary evaluation mask identifies novel single-ticker corporate events."""
    train_df = sample_features_df[sample_features_df["split"] == "train"]
    test_df = sample_features_df[sample_features_df["split"] == "test"]

    novel_mask = get_novel_catalyst_mask(test_df, train_df)

    assert novel_mask.sum() == 5
    for idx, is_novel in novel_mask.items():
        row = test_df.loc[idx]
        if is_novel:
            assert not row["is_multi_ticker"]
            assert "Novel Biotech" in row["title"]
        else:
            assert row["is_multi_ticker"] or row["title"] in train_df["title"].values

def test_prepare_splits_and_transforms_e2e(sample_features_df):
    """Verify complete end-to-end split transformation pipeline using headline_text."""
    res = prepare_splits_and_transforms(sample_features_df, use_headline_only=True)

    assert "X_train" in res and "X_val" in res and "X_test" in res
    assert "y_train" in res and "y_val" in res and "y_test" in res
    assert "novel_catalyst_mask_test" in res

    assert res["X_train"].shape[0] == 20
    assert res["X_val"].shape[0] == 10
    assert res["X_test"].shape[0] == 10
    assert len(res["y_train"]) == 20
    assert len(res["y_val"]) == 10
    assert len(res["y_test"]) == 10
    assert res["novel_catalyst_mask_test"].sum() == 5
    assert res["metadata"]["feature_dim"] > 0
    assert res["metadata"]["text_representation"] == "headline_text"
