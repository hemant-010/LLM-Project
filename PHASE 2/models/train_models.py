"""Train baseline classifiers from a verified labelled event dataset.

Supports:
1. Short-Interest Classification with explicit feature allowlist and train-only preprocessing.
2. Chronological Train / Val / Test splitting and secondary novel catalyst evaluation.
3. Backward compatibility for legacy synthetic fixtures and stock-price horizon validation.
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODEL_DIR
from models.feature_pipeline import (
    MODEL_FEATURE_ALLOWLIST,
    COMPARATIVE_FEATURE_ALLOWLIST,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TEXT_FEATURES,
    DEFAULT_TEXT_FEATURE,
    COMPOSITE_TEXT_FEATURE,
    TARGET_COLUMN,
    SPLIT_COLUMN,
    ensure_text_representations,
    build_feature_preprocessor,
    fit_preprocessor_train_only,
    transform_split,
    get_novel_catalyst_mask
)

CAT = ["event_type", "insider_role", "transaction_code", "event_severity"]
NUM = ["percentage_holdings_sold", "transaction_value", "event_classification_confidence"]
TARGETS = {
    "short_interest_label": ("short_interest_model.joblib", "next eligible short-interest reporting period"),
    "stock_price_label": ("stock_price_model.joblib", "five trading sessions"),
}
VALID = {"increase", "decrease", "no_significant_change"}

def make_model(use_extended_features: bool = False, use_headline_only: bool = True):
    """Construct a pipeline with feature preprocessor and classifier."""
    if use_extended_features:
        prep = build_feature_preprocessor(use_headline_only=use_headline_only)
    else:
        text_col = DEFAULT_TEXT_FEATURE if use_headline_only else COMPOSITE_TEXT_FEATURE
        prep = ColumnTransformer([
            ("text", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False))
            ]), NUM)
        ])
    return Pipeline([
        ("features", prep),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])

def main(input_path, min_rows: int = 100, price_horizon: str | None = None,
         allow_mixed_price_horizons: bool = False, use_composite_text: bool = False,
         output_dir: Path | None = None):
    p = Path(input_path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df = ensure_text_representations(df)

    # Determine which targets exist in the dataset
    available_targets = {t: v for t, v in TARGETS.items() if t in df.columns}
    if not available_targets:
        raise ValueError(f"Training data missing any valid target columns: {list(TARGETS.keys())}")

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["published_at"]).sort_values("published_at")

    target_dir = Path(output_dir) if output_dir is not None else MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    for target, (filename, horizon) in available_targets.items():
        subset = df[df[target].isin(VALID)].copy()
        target_horizon = horizon

        # Guard against mixing incompatible stock-price horizon types
        if target == "stock_price_label":
            if price_horizon is not None:
                if "horizon_type" not in subset.columns:
                    raise ValueError(f"Cannot filter by price_horizon '{price_horizon}': 'horizon_type' column missing from dataset.")
                subset = subset[subset["horizon_type"] == price_horizon].copy()
                if subset.empty:
                    raise ValueError(f"No stock-price rows found matching selected horizon '{price_horizon}'.")
                target_horizon = f"five trading sessions ({price_horizon})"
            elif "horizon_type" in subset.columns:
                unique_horizons = sorted(subset["horizon_type"].dropna().unique().tolist())
                if len(unique_horizons) > 1 and not allow_mixed_price_horizons:
                    raise ValueError(
                        f"Incompatible stock-price horizons detected in training data: {unique_horizons}. "
                        "Mixing full-exposure labels with post-close drift confounds initial market reaction with drift. "
                        "Specify --price-horizon to train on a single homogeneous horizon (e.g. --price-horizon 5_sessions_full_exposure), "
                        "or pass --allow-mixed-price-horizons to proceed intentionally."
                    )
                elif len(unique_horizons) == 1:
                    target_horizon = f"five trading sessions ({unique_horizons[0]})"
                elif allow_mixed_price_horizons:
                    target_horizon = f"five trading sessions (mixed: {', '.join(unique_horizons)})"

        if len(subset) < min_rows:
            raise ValueError(f"{target}: need at least {min_rows} verified labelled rows; found {len(subset)}")

        use_extended = (target == "short_interest_label" and all(col in subset.columns for col in MODEL_FEATURE_ALLOWLIST))

        # Check for explicit chronological split column
        if SPLIT_COLUMN in subset.columns and set(subset[SPLIT_COLUMN].unique()).issuperset({"train", "test"}):
            train = subset[subset[SPLIT_COLUMN] == "train"].copy()
            val = subset[subset[SPLIT_COLUMN] == "val"].copy() if "val" in subset[SPLIT_COLUMN].values else None
            test = subset[subset[SPLIT_COLUMN] == "test"].copy()
        else:
            # Chronological holdout fallback: keep latest 20% for testing
            cut = max(1, int(len(subset) * 0.8))
            train, test = subset.iloc[:cut].copy(), subset.iloc[cut:].copy()
            val = None

        if train[target].nunique() < 2 or test.empty:
            raise ValueError(f"{target}: insufficient class diversity or test rows.")

        model = make_model(use_extended_features=use_extended, use_headline_only=(not use_composite_text))

        # Train model strictly on training split
        model.fit(train, train[target])

        # Primary test evaluation
        pred_test = model.predict(test)
        acc_test = round(accuracy_score(test[target], pred_test), 4)
        f1_test = round(f1_score(test[target], pred_test, average="macro"), 4)

        print(f"\n=== {target} ===")
        print("Horizon Specification:", target_horizon)
        print("Feature Allowlist:", "Extended (13 causal features)" if use_extended else "Standard baseline")
        print("Rows:", len(train), "train /", len(test), "test" + (f" / {len(val)} val" if val is not None else ""))
        print("Primary Test Accuracy:", acc_test)
        print("Primary Test Macro F1:", f1_test)
        print(classification_report(test[target], pred_test, zero_division=0))

        # Secondary evaluation on novel corporate catalysts (if title and multi_ticker columns present)
        novel_metrics = None
        if "title" in test.columns and "is_multi_ticker" in test.columns and "title" in train.columns:
            novel_mask = get_novel_catalyst_mask(test, train)
            if novel_mask.sum() > 0:
                test_novel = test[novel_mask]
                pred_novel = model.predict(test_novel)
                acc_novel = round(accuracy_score(test_novel[target], pred_novel), 4)
                f1_novel = round(f1_score(test_novel[target], pred_novel, average="macro"), 4)
                print(f"--- Secondary Evaluation: Novel Single-Ticker Corporate Catalysts ---")
                print(f"Novel Catalyst Rows: {len(test_novel)} ({len(test_novel)/len(test)*100:.1f}% of test set)")
                print(f"Novel Catalyst Accuracy: {acc_novel}")
                print(f"Novel Catalyst Macro F1: {f1_novel}")
                novel_metrics = {
                    "novel_catalyst_rows": len(test_novel),
                    "accuracy": acc_novel,
                    "macro_f1": f1_novel
                }

        artifact_payload = {
            "model": model,
            "horizon": target_horizon,
            "target": target,
            "trained_rows": len(train),
            "test_rows": len(test),
            "use_extended_features": use_extended,
            "primary_test_metrics": {"accuracy": acc_test, "macro_f1": f1_test},
            "secondary_novel_catalyst_metrics": novel_metrics
        }
        if "published_at" in train.columns and not train["published_at"].isna().all():
            artifact_payload["latest_train_date"] = str(train.published_at.max())
            artifact_payload["test_start_date"] = str(test.published_at.min())

        joblib.dump(artifact_payload, target_dir / filename)
        print("Saved:", target_dir / filename)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--price-horizon", default=None, help="Explicitly select a single stock-price horizon type")
    p.add_argument("--allow-mixed-price-horizons", action="store_true", help="Explicitly permit mixing multiple stock-price horizon types")
    p.add_argument("--use-composite-text", action="store_true", help="Use composite model_text instead of default headline_text")
    p.add_argument("--min-rows", type=int, default=100, help="Minimum verified labelled rows required for training (default: 100)")
    a = p.parse_args()
    main(a.input, min_rows=a.min_rows, price_horizon=a.price_horizon,
         allow_mixed_price_horizons=a.allow_mixed_price_horizons,
         use_composite_text=a.use_composite_text)
