"""Explicit feature allowlist, leakage protection, and train-only preprocessing pipeline for Phase 2.

Key Design Principles:
1. Explicit Feature Allowlist:
   - Only validated, causal pre-event features are passed to the model.
   - Text features: Headline/title only (`headline_text`) is the DEFAULT text input,
     free of company names or ticker symbol identity tags.
     The composite representation (`model_text`) is preserved for comparative benchmarking.
   - Categorical features: Taxonomical event type, category, severity, insider role, transaction code, and syndication flag.
   - Numerical features: Insider percentage sold, transaction value, classification confidence, token counts, headline ticker counts, and pre-event short interest baseline.
2. Deliberate Treatment of Identifiers and Timestamps:
   - High-cardinality nominal identifiers (`ticker`, `company`) are strictly excluded from dummy encoding to prevent overfitting, memorization of historical returns, and failure on unseen tickers.
   - Identity leakage via text inputs is resolved: `headline_text` strips out all company name and ticker tags.
   - Absolute timestamps (`published_at`, `date`, `published_date`, `news_date`, `short_interest_before_date`, etc.) are excluded to prevent temporal boundary overfitting.
   - Chronological boundaries are enforced exclusively via dataset partitioning (`train`, `val`, `test`).
3. Strict Train-Only Fitting:
   - Text vectorizers (TF-IDF), numerical imputers/scalers, and categorical encoders are fit EXCLUSIVELY on the training split (`split == 'train'`).
   - Validation and test splits are transformed using the frozen preprocessor state. Zero refitting on validation/test.
4. Pre-Event Baseline Grounding:
   - Pre-event short interest baseline (`short_interest_before`) is admitted only because its public dissemination date strictly precedes the event date (`short_interest_before_dissemination < published_at`).
5. Secondary Evaluation for Headline Repetition:
   - Evaluates performance on both:
     a) Primary Benchmark: Full chronological test holdout (measuring raw stream performance).
     b) Novel Catalyst Benchmark: Deduplicated single-ticker events with novel headlines (testing uninflated generalization on unseen corporate catalysts).
"""
import re
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Text Representations
DEFAULT_TEXT_FEATURE: str = "headline_text"          # Headline/title only, free of ticker/company identity tags
COMPOSITE_TEXT_FEATURE: str = "model_text"           # Preserved composite representation for comparison

# Default Text Feature in Allowlist
TEXT_FEATURES: List[str] = [
    DEFAULT_TEXT_FEATURE
]

# Preserved comparative text features
COMPARATIVE_TEXT_FEATURES: List[str] = [
    COMPOSITE_TEXT_FEATURE
]

# Categorical Features (Unchanged)
CATEGORICAL_FEATURES: List[str] = [
    "event_type",
    "event_category",
    "event_severity",
    "insider_role",
    "transaction_code",
    "is_multi_ticker"
]

# Numerical Features (Unchanged)
NUMERICAL_FEATURES: List[str] = [
    "percentage_holdings_sold",
    "transaction_value",
    "event_classification_confidence",
    "n_tokens",
    "headline_ticker_count",
    "short_interest_before"
]

# Default Allowlist
MODEL_FEATURE_ALLOWLIST: List[str] = TEXT_FEATURES + CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Allowlist for comparative model using composite model_text
COMPARATIVE_FEATURE_ALLOWLIST: List[str] = COMPARATIVE_TEXT_FEATURES + CATEGORICAL_FEATURES + NUMERICAL_FEATURES

TARGET_COLUMN: str = "short_interest_label"
SPLIT_COLUMN: str = "split"

# Explicit audit of excluded columns and reasons
EXCLUDED_COLUMNS_AUDIT: Dict[str, str] = {
    "event_id": "Unique primary key; excluded to prevent row memorization and indexing leakage.",
    "ticker": "High-cardinality identifier (4,447 unique values); excluded from one-hot encoding and purged from default headline_text to prevent identity leakage, sparsity explosion, and memorization of past firm returns.",
    "company": "Free-text company name; excluded from one-hot encoding and purged from default headline_text to prevent firm identity leakage.",
    "date": "Calendar date string; excluded to prevent decision trees/linear models from learning calendar boundary cutoffs.",
    "published_at": "UTC timestamp; excluded to prevent temporal boundary overfitting. Time is handled via chronological splitting.",
    "published_date": "Date string; excluded to prevent temporal boundary overfitting.",
    "news_date": "Integer/epoch date; excluded to prevent temporal boundary overfitting.",
    "title": "Raw headline string; standardized into primary default text feature 'headline_text'.",
    "model_text": "Composite text representation (title + description or metadata tags); preserved for comparative baseline experiments, but excluded from default model input to eliminate ticker/company identity leakage.",
    "severity": "Duplicate alias of event_severity.",
    "confidence": "Duplicate alias of event_classification_confidence.",
    "classification_source": "Internal pipeline provenance metadata ('heuristic_rule', 'llm_gemini').",
    "context_source": "Internal pipeline provenance metadata.",
    "classification_explanation": "Qualitative textual explanation intended for human audit.",
    "context_summary": "Qualitative textual summary intended for human audit.",
    "severity_rationale": "Qualitative textual rationale intended for human audit.",
    "all_detected_event_types": "Internal diagnostic list of secondary heuristic matches.",
    "risk_factors": "Semi-structured diagnostic list.",
    "supporting_evidence": "Semi-structured diagnostic list.",
    "insider_nuance": "Qualitative diagnostic note.",
    "context_flags": "Semi-structured diagnostic flag string.",
    "insider_is_10b5_1": "SEC Form 4 10b5-1 flag; unpopulated for general financial news.",
    "is_10b5_1": "Duplicate alias of insider_is_10b5_1.",
    "short_interest_before_date": "Pre-event settlement date timestamp; excluded to prevent date threshold overfitting.",
    "short_interest_before_dissemination": "Pre-event dissemination date timestamp; audit only.",
    "short_interest_before_dissemination_source": "Data provenance audit string.",
    "split": "Dataset partition label ('train', 'val', 'test'); excluded from features.",
    "short_interest_label": "Target variable ground truth; excluded to prevent direct target leakage.",
    # Future outcome columns (prohibited leakage)
    "short_interest_after": "Future short interest observation; strict target leakage.",
    "short_interest_after_date": "Future settlement date; strict target leakage.",
    "short_interest_after_dissemination": "Future dissemination date; strict target leakage.",
    "short_interest_after_dissemination_source": "Future provenance; strict target leakage.",
    "short_interest_gap_days": "Future outcome duration; strict target leakage.",
    "short_interest_change_pct": "Continuous future outcome delta; strict target leakage.",
    "phase1_label_short_interest_legacy": "Legacy Phase 1 label; target contamination.",
    "phase1_label_regsho_volume_legacy": "Legacy Phase 1 label; target contamination.",
    "stock_price_label": "Alternative target label; target contamination.",
    "stock_price_change_pct": "Stock price outcome delta; target contamination."
}

# Strict future outcome leakage columns prohibited from any modeling dataset
PROHIBITED_LEAKAGE_COLUMNS: List[str] = [
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

def sanitize_to_headline_only(text: str) -> str:
    """Extract strictly the headline/title string, stripping any company or ticker metadata tags.

    Example:
    "Title: Apple Beats Q1 | Company: Apple Inc. | Ticker: AAPL" -> "Apple Beats Q1"
    """
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()

    # If text is structured with "Title: ... | Company: ... | Ticker: ...", isolate Title
    if " | Company:" in cleaned or " | Ticker:" in cleaned:
        # Match Title: <title> before first metadata pipe
        m = re.search(r"^(?:Title:\s*)?(.*?)(?:\s*\|\s*(?:Company|Ticker):)", cleaned, re.IGNORECASE)
        if m:
            cleaned = m.group(1).strip()
        else:
            # Fallback: take substring before the first metadata pipe
            cleaned = re.split(r"\s*\|\s*(?:Company|Ticker):", cleaned, flags=re.IGNORECASE)[0].strip()

    # Remove leading "Title: " prefix if present
    if cleaned.lower().startswith("title:"):
        cleaned = cleaned[6:].strip()

    return cleaned

def ensure_text_representations(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure both `headline_text` (default) and `model_text` (composite) exist on df.

    - `headline_text` contains headline/title only, free from company and ticker tags.
    - `model_text` preserves the composite representation for comparative benchmarking.
    """
    df_out = df.copy()

    # 1. Headline-only representation
    if "headline_text" in df_out.columns:
        df_out["headline_text"] = df_out["headline_text"].apply(sanitize_to_headline_only)
    elif "title" in df_out.columns:
        df_out["headline_text"] = df_out["title"].apply(sanitize_to_headline_only)
    elif "model_text" in df_out.columns:
        df_out["headline_text"] = df_out["model_text"].apply(sanitize_to_headline_only)
    else:
        df_out["headline_text"] = ""

    # 2. Composite representation
    if "model_text" not in df_out.columns:
        if "title" in df_out.columns and "company" in df_out.columns and "ticker" in df_out.columns:
            df_out["model_text"] = [
                f"Title: {t} | Company: {c} | Ticker: {tk}"
                for t, c, tk in zip(df_out["title"], df_out["company"], df_out["ticker"])
            ]
        elif "title" in df_out.columns:
            df_out["model_text"] = df_out["title"].fillna("").astype(str)
        else:
            df_out["model_text"] = df_out["headline_text"]

    return df_out

def validate_feature_inputs(
    df: pd.DataFrame,
    require_target: bool = True,
    require_split: bool = True,
    use_headline_only: bool = True
) -> None:
    """Validate that input DataFrame conforms strictly to feature allowlist specifications."""
    active_allowlist = MODEL_FEATURE_ALLOWLIST if use_headline_only else COMPARATIVE_FEATURE_ALLOWLIST
    missing_features = [f for f in active_allowlist if f not in df.columns]

    # If only the text feature is missing, check if it can be derived
    if missing_features == [DEFAULT_TEXT_FEATURE] and ("title" in df.columns or "model_text" in df.columns):
        missing_features = []

    if missing_features:
        raise ValueError(f"Input DataFrame is missing required allowlist features: {missing_features}")

    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(f"Input DataFrame is missing required target column: '{TARGET_COLUMN}'")

    if require_split and SPLIT_COLUMN not in df.columns:
        raise ValueError(f"Input DataFrame is missing required split column: '{SPLIT_COLUMN}'")

    # Verify no prohibited future outcome leakage columns are present
    prohibited_present = [col for col in PROHIBITED_LEAKAGE_COLUMNS if col in df.columns]
    if prohibited_present:
        raise ValueError(f"Prohibited target/temporal leakage columns detected in dataset: {prohibited_present}")

    # Verify pre-event baseline availability if dissemination timestamp is present
    if "short_interest_before_dissemination" in df.columns and "published_at" in df.columns:
        diss_dates = pd.to_datetime(df["short_interest_before_dissemination"]).dt.date
        pub_dates = pd.to_datetime(df["published_at"]).dt.date
        invalid_mask = diss_dates >= pub_dates
        if invalid_mask.any():
            n_invalid = int(invalid_mask.sum())
            raise ValueError(f"Pre-event baseline short-interest availability violation: {n_invalid} records have dissemination date >= published date.")

def build_feature_preprocessor(
    use_headline_only: bool = True,
    max_text_features: int = 30000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2
) -> ColumnTransformer:
    """Construct an unfitted scikit-learn ColumnTransformer conforming strictly to allowlist.

    Parameters
    ----------
    use_headline_only : bool, default=True
        If True (default), trains on `headline_text` (headline only, free of ticker/company identity leakage).
        If False, trains on `model_text` (composite representation, preserved for comparative experiments).
    max_text_features : int, default=30000
    ngram_range : Tuple[int, int], default=(1, 2)
    min_df : int, default=2
    """
    text_col = DEFAULT_TEXT_FEATURE if use_headline_only else COMPOSITE_TEXT_FEATURE

    text_transformer = TfidfVectorizer(
        max_features=max_text_features,
        ngram_range=ngram_range,
        min_df=min_df
    )

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

    numerical_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)),
        ("scale", StandardScaler(with_mean=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, text_col),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numerical_transformer, NUMERICAL_FEATURES)
        ],
        remainder="drop"  # Strictly drop any non-allowlist columns if passed
    )

    return preprocessor

def fit_preprocessor_train_only(
    preprocessor: ColumnTransformer,
    train_df: pd.DataFrame,
    use_headline_only: bool = True
) -> ColumnTransformer:
    """Fit the feature preprocessor EXCLUSIVELY on training data."""
    if SPLIT_COLUMN in train_df.columns:
        splits = train_df[SPLIT_COLUMN].unique()
        if len(splits) != 1 or splits[0] != "train":
            raise ValueError(f"fit_preprocessor_train_only must be called exclusively on 'train' split! Found splits: {splits}")

    # Ensure text representations exist and are sanitized
    df_prepared = ensure_text_representations(train_df)
    active_allowlist = MODEL_FEATURE_ALLOWLIST if use_headline_only else COMPARATIVE_FEATURE_ALLOWLIST

    X_train_df = df_prepared[active_allowlist].copy()

    # Ensure categorical columns are properly cast as strings
    for col in CATEGORICAL_FEATURES:
        X_train_df[col] = X_train_df[col].fillna("missing").astype(str)

    # Ensure numerical columns are properly numeric
    for col in NUMERICAL_FEATURES:
        X_train_df[col] = pd.to_numeric(X_train_df[col], errors="coerce")

    # Fit exclusively on train
    preprocessor.fit(X_train_df)
    return preprocessor

def transform_split(
    fitted_preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    use_headline_only: bool = True
) -> Any:
    """Apply fitted preprocessor to a dataset split without refitting or mutating state."""
    if not hasattr(fitted_preprocessor, "transformers_"):
        raise RuntimeError("transform_split called with unfitted preprocessor!")

    df_prepared = ensure_text_representations(df)
    active_allowlist = MODEL_FEATURE_ALLOWLIST if use_headline_only else COMPARATIVE_FEATURE_ALLOWLIST

    X_df = df_prepared[active_allowlist].copy()

    for col in CATEGORICAL_FEATURES:
        X_df[col] = X_df[col].fillna("missing").astype(str)

    for col in NUMERICAL_FEATURES:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    return fitted_preprocessor.transform(X_df)

def get_novel_catalyst_mask(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame
) -> pd.Series:
    """Identify single-ticker events in test with headlines completely unseen in training.

    Used for secondary evaluation to measure true out-of-sample corporate catalyst
    generalization, free from repetitive generic market roundup inflation.
    """
    train_titles = set(train_df["title"].dropna().unique())
    is_single_ticker = ~test_df["is_multi_ticker"].astype(bool)
    is_novel_headline = ~test_df["title"].isin(train_titles)
    return is_single_ticker & is_novel_headline

def prepare_splits_and_transforms(
    df: pd.DataFrame,
    use_headline_only: bool = True,
    max_text_features: int = 30000
) -> Dict[str, Any]:
    """Execute complete end-to-end split partitioning and train-only transformation.

    Returns dictionary containing:
    - preprocessor: fitted ColumnTransformer
    - X_train, X_val, X_test: transformed feature matrices
    - y_train, y_val, y_test: target arrays
    - novel_catalyst_mask_test: boolean mask for secondary test evaluation
    - metadata: split counts and diagnostics
    """
    df_prepared = ensure_text_representations(df)
    validate_feature_inputs(df_prepared, require_target=True, require_split=True, use_headline_only=use_headline_only)

    train_df = df_prepared[df_prepared[SPLIT_COLUMN] == "train"].copy()
    val_df = df_prepared[df_prepared[SPLIT_COLUMN] == "val"].copy()
    test_df = df_prepared[df_prepared[SPLIT_COLUMN] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(f"Invalid split counts: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Build and fit preprocessor on train only
    preprocessor = build_feature_preprocessor(use_headline_only=use_headline_only, max_text_features=max_text_features)
    preprocessor = fit_preprocessor_train_only(preprocessor, train_df, use_headline_only=use_headline_only)

    # Transform each split without refitting
    X_train = transform_split(preprocessor, train_df, use_headline_only=use_headline_only)
    X_val = transform_split(preprocessor, val_df, use_headline_only=use_headline_only)
    X_test = transform_split(preprocessor, test_df, use_headline_only=use_headline_only)

    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    novel_mask = get_novel_catalyst_mask(test_df, train_df)

    active_allowlist = MODEL_FEATURE_ALLOWLIST if use_headline_only else COMPARATIVE_FEATURE_ALLOWLIST

    return {
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "novel_catalyst_mask_test": novel_mask,
        "metadata": {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_test_novel_catalysts": int(novel_mask.sum()),
            "feature_dim": X_train.shape[1],
            "text_representation": DEFAULT_TEXT_FEATURE if use_headline_only else COMPOSITE_TEXT_FEATURE,
            "allowlist_features": active_allowlist
        }
    }
