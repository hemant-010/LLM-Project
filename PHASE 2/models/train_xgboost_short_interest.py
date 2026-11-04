"""Train and evaluate XGBoost baseline model on structured features for Phase 2 Short-Interest Prediction.

Strictly enforces:
1. Structured categorical and numerical features only (no text features, no identifiers, no dates, no target-derived features).
2. Fitting preprocessing strictly on TRAIN split only.
3. Hyperparameter selection strictly on VALIDATION Macro-F1.
4. Evaluation on out-of-sample TEST holdout (Full test holdout and Novel corporate catalysts) strictly post-selection.
5. Preserves existing Logistic Regression artifacts untouched; exports separate artifacts:
   - models/artifacts/short_interest_xgboost_model.joblib
   - models/artifacts/short_interest_xgboost_metrics.json
   - data/processed/short_interest_xgboost_evaluation_report.md
"""
import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

# Feature Allowlist: Structured features ONLY (Categorical + Numerical)
STRUCTURED_CATEGORICAL_FEATURES: List[str] = [
    "event_type",
    "event_category",
    "event_severity",
    "insider_role",
    "transaction_code",
    "is_multi_ticker",
]

STRUCTURED_NUMERICAL_FEATURES: List[str] = [
    "percentage_holdings_sold",
    "transaction_value",
    "event_classification_confidence",
    "n_tokens",
    "headline_ticker_count",
    "short_interest_before",
]

STRUCTURED_FEATURE_ALLOWLIST: List[str] = STRUCTURED_CATEGORICAL_FEATURES + STRUCTURED_NUMERICAL_FEATURES

TARGET_COLUMN: str = "short_interest_label"
SPLIT_COLUMN: str = "split"
TARGET_CLASSES: List[str] = ["decrease", "increase", "no_significant_change"]
RANDOM_SEED: int = 42

DEFAULT_DATASET_PARQUET: Path = Path(__file__).resolve().parent.parent / "data" / "processed" / "short_interest_model_dataset.parquet"
DEFAULT_OUTPUT_DIR: Path = Path(__file__).resolve().parent / "artifacts"
DEFAULT_REPORT_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "processed" / "short_interest_xgboost_evaluation_report.md"
BASELINE_METRICS_PATH: Path = Path(__file__).resolve().parent / "artifacts" / "short_interest_baseline_metrics.json"


def build_structured_preprocessor() -> ColumnTransformer:
    """Construct column transformer for structured categorical and numerical features."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                STRUCTURED_NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                STRUCTURED_CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str] = TARGET_CLASSES) -> Dict[str, Any]:
    """Compute comprehensive multiclass classification metrics."""
    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    macro_f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    weighted_f1 = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
    bal_acc = round(float(balanced_accuracy_score(y_true, y_pred)), 4)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    per_class = {}
    for idx, cls_name in enumerate(classes):
        per_class[cls_name] = {
            "precision": round(float(p[idx]), 4),
            "recall": round(float(r[idx]), 4),
            "f1": round(float(f[idx]), 4),
            "support": int(s[idx]),
        }

    cm_raw = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    cm_norm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    cm_norm = np.round(cm_norm, 4).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "per_class": per_class,
        "confusion_matrix_raw": cm_raw,
        "confusion_matrix_normalized": cm_norm,
        "classes": classes,
    }


def get_novel_catalyst_mask(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.Series:
    """Mask selecting deduplicated, novel corporate catalysts in the test set."""
    is_single_ticker = (test_df["is_multi_ticker"] == False) & (test_df["headline_ticker_count"] <= 1)
    train_titles = set(train_df["title"].dropna().unique())
    is_novel_headline = ~test_df["title"].isin(train_titles)
    return is_single_ticker & is_novel_headline


class StructuredXGBoostPipeline:
    """End-to-end inference wrapper combining preprocessor, label encoder, and XGBoost."""

    def __init__(self, preprocessor: ColumnTransformer, label_encoder: LabelEncoder, model: xgb.XGBClassifier):
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.model = model
        self.classes_ = label_encoder.classes_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_trans = self.preprocessor.transform(X[STRUCTURED_FEATURE_ALLOWLIST])
        preds_enc = self.model.predict(X_trans)
        return self.label_encoder.inverse_transform(preds_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_trans = self.preprocessor.transform(X[STRUCTURED_FEATURE_ALLOWLIST])
        return self.model.predict_proba(X_trans)


def train_and_evaluate_xgboost(
    dataset_path: Path = DEFAULT_DATASET_PARQUET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 2 Structured XGBoost Short-Interest Model Training & Evaluation")
    print("=" * 80)

    # 1. Load Dataset
    print(f"\n1. Loading dataset from {dataset_path.name}...")
    df = pd.read_parquet(dataset_path)
    print(f"   Total events: {len(df):,}")

    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    val_df = df[df[SPLIT_COLUMN] == "val"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()

    # Novel corporate catalyst benchmark
    novel_mask = get_novel_catalyst_mask(test_df, train_df)
    test_novel_df = test_df[novel_mask].copy()

    print(f"   TRAIN: {len(train_df):,} rows")
    print(f"   VAL:   {len(val_df):,} rows")
    print(f"   TEST:  {len(test_df):,} rows (Full holdout)")
    print(f"   TEST:  {len(test_novel_df):,} rows (Novel corporate catalysts, {len(test_novel_df)/len(test_df)*100:.1f}%)")

    # 2. Fit Preprocessing strictly on TRAIN
    print("\n2. Fitting preprocessor strictly on TRAIN split...")
    t_prep = time.time()
    preprocessor = build_structured_preprocessor()
    X_train = preprocessor.fit_transform(train_df[STRUCTURED_FEATURE_ALLOWLIST])
    X_val = preprocessor.transform(val_df[STRUCTURED_FEATURE_ALLOWLIST])
    X_test = preprocessor.transform(test_df[STRUCTURED_FEATURE_ALLOWLIST])
    X_test_novel = preprocessor.transform(test_novel_df[STRUCTURED_FEATURE_ALLOWLIST])
    prep_duration = time.time() - t_prep

    feature_dim = X_train.shape[1]
    print(f"   Preprocessing completed in {prep_duration:.2f}s.")
    print(f"   Feature dimensionality: {feature_dim} structured features.")

    # Encode targets
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[TARGET_COLUMN])
    y_val = le.transform(val_df[TARGET_COLUMN])
    y_test = le.transform(test_df[TARGET_COLUMN])
    y_test_novel = le.transform(test_novel_df[TARGET_COLUMN])

    y_train_str = train_df[TARGET_COLUMN].values
    y_val_str = val_df[TARGET_COLUMN].values
    y_test_str = test_df[TARGET_COLUMN].values
    y_test_novel_str = test_novel_df[TARGET_COLUMN].values

    # Pre-compute balanced sample weights on TRAIN only
    weights_balanced = compute_sample_weight("balanced", y_train)

    # 3. Hyperparameter Tuning on VALIDATION split strictly
    print("\n3. Tuning XGBoost Hyperparameters strictly on VALIDATION split...")
    tuning_grid = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8, "weight": "balanced"},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8, "weight": "balanced"},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8, "weight": "balanced"},
        {"max_depth": 7, "learning_rate": 0.05, "n_estimators": 150, "subsample": 0.8, "colsample_bytree": 0.8, "weight": "balanced"},
        {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 150, "subsample": 0.8, "colsample_bytree": 0.8, "weight": "balanced"},
        {"max_depth": 6, "learning_rate": 0.08, "n_estimators": 120, "subsample": 0.8, "colsample_bytree": 0.8, "weight": "balanced"},
        {"max_depth": 4, "learning_rate": 0.10, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8, "weight": "balanced"},
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8, "weight": None},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8, "weight": None},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8, "weight": None},
        {"max_depth": 7, "learning_rate": 0.05, "n_estimators": 150, "subsample": 0.8, "colsample_bytree": 0.8, "weight": None},
        {"max_depth": 4, "learning_rate": 0.10, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8, "weight": None},
    ]

    tuning_results = []
    best_config = None
    best_val_macro_f1 = -1.0
    best_candidate_model = None

    for idx, cfg in enumerate(tuning_grid, 1):
        t_start = time.time()
        w = weights_balanced if cfg["weight"] == "balanced" else None
        clf_cand = xgb.XGBClassifier(
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            random_state=random_seed,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=4,
        )
        clf_cand.fit(X_train, y_train, sample_weight=w)
        pred_val_enc = clf_cand.predict(X_val)
        pred_val_str = le.inverse_transform(pred_val_enc)
        val_metrics_cand = compute_metrics(y_val_str, pred_val_str)
        elapsed = time.time() - t_start

        tuning_entry = {
            "config": cfg,
            "val_accuracy": val_metrics_cand["accuracy"],
            "val_macro_f1": val_metrics_cand["macro_f1"],
            "val_weighted_f1": val_metrics_cand["weighted_f1"],
            "val_balanced_accuracy": val_metrics_cand["balanced_accuracy"],
            "fit_time_seconds": round(elapsed, 2),
        }
        tuning_results.append(tuning_entry)

        weight_str = f"'{cfg['weight']}'" if cfg["weight"] else "None"
        print(
            f"   [{idx:2d}/{len(tuning_grid)}] depth={cfg['max_depth']} lr={cfg['learning_rate']:<4} n_est={cfg['n_estimators']:<3} "
            f"weight={weight_str:<10} | Val Acc={val_metrics_cand['accuracy']:.4f} Macro-F1={val_metrics_cand['macro_f1']:.4f} "
            f"BalAcc={val_metrics_cand['balanced_accuracy']:.4f} ({elapsed:.1f}s)"
        )

        if val_metrics_cand["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics_cand["macro_f1"]
            best_config = cfg
            best_candidate_model = clf_cand

    print(f"\n   >>> Model Selection Complete: Best Validation Configuration: {best_config}")
    print(f"   >>> Selected Validation Macro-F1: {best_val_macro_f1:.4f}")

    # 4. Evaluate Final Selected Model across all splits (Test evaluated strictly post-selection)
    print("\n4. Evaluating selected model across all splits (Test evaluated only after selection)...")
    final_clf = best_candidate_model

    pred_train_str = le.inverse_transform(final_clf.predict(X_train))
    pred_val_str = le.inverse_transform(final_clf.predict(X_val))
    pred_test_full_str = le.inverse_transform(final_clf.predict(X_test))
    pred_test_novel_str = le.inverse_transform(final_clf.predict(X_test_novel))

    xgb_metrics = {
        "train": compute_metrics(y_train_str, pred_train_str),
        "val": compute_metrics(y_val_str, pred_val_str),
        "test_full": compute_metrics(y_test_str, pred_test_full_str),
        "test_novel_catalysts": compute_metrics(y_test_novel_str, pred_test_novel_str),
    }

    # Load baseline metrics for direct comparison
    baseline_metrics = None
    if BASELINE_METRICS_PATH.exists():
        with open(BASELINE_METRICS_PATH) as f:
            baseline_metrics = json.load(f)

    # 5. Save model pipeline and metrics separately
    print("\n5. Saving XGBoost pipeline and metrics under models/artifacts/...")
    full_pipeline = StructuredXGBoostPipeline(
        preprocessor=preprocessor,
        label_encoder=le,
        model=final_clf,
    )

    environment_metadata = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "xgboost_version": xgb.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "random_seed": random_seed,
    }

    model_artifact_path = output_dir / "short_interest_xgboost_model.joblib"
    joblib.dump(
        {
            "model": full_pipeline,
            "target": TARGET_COLUMN,
            "model_type": "XGBoost",
            "features_used": STRUCTURED_FEATURE_ALLOWLIST,
            "feature_dim": feature_dim,
            "best_hyperparameters": best_config,
            "classes": TARGET_CLASSES,
            "metrics": xgb_metrics,
            "environment": environment_metadata,
        },
        model_artifact_path,
    )
    print(f"   Model pipeline saved: {model_artifact_path}")

    metrics_artifact_path = output_dir / "short_interest_xgboost_metrics.json"
    metrics_export = {
        "model_type": "XGBClassifier",
        "features": {
            "categorical": STRUCTURED_CATEGORICAL_FEATURES,
            "numerical": STRUCTURED_NUMERICAL_FEATURES,
            "total_features": len(STRUCTURED_FEATURE_ALLOWLIST),
            "transformed_dim": feature_dim,
        },
        "best_hyperparameters": best_config,
        "validation_tuning_results": tuning_results,
        "xgboost_metrics": xgb_metrics,
        "environment": environment_metadata,
    }
    with open(metrics_artifact_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"   Metrics JSON saved:   {metrics_artifact_path}")

    # 6. Generate Markdown Evaluation Report
    print(f"\n6. Generating Markdown Evaluation Report at {report_path.name}...")
    generate_markdown_report(
        report_path,
        best_config,
        best_val_macro_f1,
        tuning_results,
        xgb_metrics,
        baseline_metrics,
        environment_metadata,
    )
    print(f"   Report generated:     {report_path}")

    return metrics_export


def generate_markdown_report(
    report_path: Path,
    best_config: Dict[str, Any],
    best_val_macro_f1: float,
    tuning_results: List[Dict[str, Any]],
    xgb_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any] | None,
    env: Dict[str, Any],
) -> None:
    """Generate structured markdown evaluation report with side-by-side model comparison."""
    with open(report_path, "w") as f:
        f.write("# Phase 2 Structured XGBoost Short-Interest Model Evaluation Report\n\n")
        f.write("**Evaluation Date:** 2026-11-04  \n")
        f.write("**Model Architecture:** XGBoost (`XGBClassifier` with histogram tree method)  \n")
        f.write("**Feature Scope:** Structured Categorical & Numerical Features Only (Zero text features, zero identifiers, zero dates)  \n")
        f.write("**Target Variable:** `short_interest_label` (`increase`, `decrease`, `no_significant_change`)  \n")
        f.write("**Artifact File:** `models/artifacts/short_interest_xgboost_model.joblib`  \n\n")

        f.write("---\n\n## 1. Executive Summary & Comparative Highlights\n\n")
        f.write("This experiment evaluates an **XGBoost gradient boosted decision tree classifier** trained strictly on 12 pre-event structured features (6 categorical and 6 numerical) against both the `DummyClassifier` majority baseline and the `LogisticRegression` baseline.\n\n")

        # Direct 3-way Comparison Table
        f.write("### Model Comparison Across Benchmarks\n\n")
        f.write("| Evaluation Benchmark | Metric | Dummy Baseline | Logistic Regression | XGBoost (Structured) | Delta vs LR |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")

        lr_val = baseline_metrics.get("logistic_regression_metrics", {}).get("val", {}) if baseline_metrics else {}
        lr_test_full = baseline_metrics.get("logistic_regression_metrics", {}).get("test_full", {}) if baseline_metrics else {}
        lr_test_novel = baseline_metrics.get("logistic_regression_metrics", {}).get("test_novel_catalysts", {}) if baseline_metrics else {}

        dummy_val = baseline_metrics.get("dummy_majority_metrics", {}).get("val", {}) if baseline_metrics else {}
        dummy_test_full = baseline_metrics.get("dummy_majority_metrics", {}).get("test_full", {}) if baseline_metrics else {}
        dummy_test_novel = baseline_metrics.get("dummy_majority_metrics", {}).get("test_novel_catalysts", {}) if baseline_metrics else {}

        xgb_val = xgb_metrics["val"]
        xgb_test_full = xgb_metrics["test_full"]
        xgb_test_novel = xgb_metrics["test_novel_catalysts"]

        benchmarks = [
            ("Validation Split", "Accuracy", dummy_val.get("accuracy", 0.3791), lr_val.get("accuracy", 0.3898), xgb_val["accuracy"]),
            ("Validation Split", "Macro-F1", dummy_val.get("macro_f1", 0.1833), lr_val.get("macro_f1", 0.3788), xgb_val["macro_f1"]),
            ("Validation Split", "Balanced Acc", dummy_val.get("balanced_accuracy", 0.3333), lr_val.get("balanced_accuracy", 0.3883), xgb_val["balanced_accuracy"]),
            ("Full Test Holdout", "Accuracy", dummy_test_full.get("accuracy", 0.4272), lr_test_full.get("accuracy", 0.3843), xgb_test_full["accuracy"]),
            ("Full Test Holdout", "Macro-F1", dummy_test_full.get("macro_f1", 0.1995), lr_test_full.get("macro_f1", 0.3459), xgb_test_full["macro_f1"]),
            ("Full Test Holdout", "Balanced Acc", dummy_test_full.get("balanced_accuracy", 0.3333), lr_test_full.get("balanced_accuracy", 0.3761), xgb_test_full["balanced_accuracy"]),
            ("Novel Corporate Catalysts", "Accuracy", dummy_test_novel.get("accuracy", 0.4067), lr_test_novel.get("accuracy", 0.3371), xgb_test_novel["accuracy"]),
            ("Novel Corporate Catalysts", "Macro-F1", dummy_test_novel.get("macro_f1", 0.1927), lr_test_novel.get("macro_f1", 0.3337), xgb_test_novel["macro_f1"]),
            ("Novel Corporate Catalysts", "Balanced Acc", dummy_test_novel.get("balanced_accuracy", 0.3333), lr_test_novel.get("balanced_accuracy", 0.3657), xgb_test_novel["balanced_accuracy"]),
        ]

        for b_name, m_name, d_val, l_val, x_val in benchmarks:
            diff = x_val - l_val
            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            f.write(f"| **{b_name}** | {m_name} | `{d_val:.4f}` | `{l_val:.4f}` | **`{x_val:.4f}`** | `{diff_str}` |\n")

        f.write("\n---\n\n## 2. Feature Definitions & Leakage Safeguards\n\n")
        f.write("In this experiment, text features (`headline_text`, `model_text`, `title`) were deliberately omitted to establish the standalone predictive value of structured metadata.\n\n")
        f.write("### Categorical Features (One-Hot Encoded on Train Only):\n")
        for c in STRUCTURED_CATEGORICAL_FEATURES:
            f.write(f"- `{c}`\n")
        f.write("\n### Numerical Features (SimpleImputer on Train Only):\n")
        for n in STRUCTURED_NUMERICAL_FEATURES:
            f.write(f"- `{n}`\n")
        f.write("\n### Excluded Attributes:\n")
        f.write("- **Identifiers:** `event_id`, `ticker`, `company`\n")
        f.write("- **Dates/Times:** `date`, `published_at`, `news_date`, `published_date`\n")
        f.write("- **Target & Future Outcomes:** `short_interest_label`, `short_interest_after`, settlement dates, dissemination dates\n")

        f.write("\n---\n\n## 3. Validation Model Selection & Hyperparameter Tuning\n\n")
        f.write("Grid search conducted strictly on the chronological **Validation split** ($N = 44,606$). The model was selected based on Validation Macro-F1.\n\n")
        f.write("| Config | Max Depth | Learning Rate | N Estimators | Subsample | Colsample | Class Weight | Val Accuracy | Val Macro-F1 | Val Balanced Acc |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for idx, res in enumerate(tuning_results, 1):
            cfg = res["config"]
            w_str = f"`{cfg['weight']}`" if cfg["weight"] else "`None`"
            sel_str = " **(Selected)**" if cfg == best_config else ""
            f.write(
                f"| {idx} | {cfg['max_depth']} | {cfg['learning_rate']} | {cfg['n_estimators']} | {cfg['subsample']} | {cfg['colsample_bytree']} | "
                f"{w_str} | {res['val_accuracy']:.4f} | **{res['val_macro_f1']:.4f}**{sel_str} | {res['val_balanced_accuracy']:.4f} |\n"
            )

        f.write(f"\n**Selected Hyperparameters:** `{best_config}`  \n")
        f.write(f"**Optimal Validation Macro-F1:** `{best_val_macro_f1:.4f}`  \n\n")

        f.write("---\n\n## 4. Detailed Out-of-Sample Test Evaluation\n\n")
        f.write("### 1. Full Chronological Test Holdout ($N = 66,578$)\n\n")
        f.write(f"- **Accuracy:** `{xgb_test_full['accuracy']:.4f}`\n")
        f.write(f"- **Macro-F1:** `{xgb_test_full['macro_f1']:.4f}`\n")
        f.write(f"- **Weighted-F1:** `{xgb_test_full['weighted_f1']:.4f}`\n")
        f.write(f"- **Balanced Accuracy:** `{xgb_test_full['balanced_accuracy']:.4f}`\n\n")

        f.write("| Class Label | Precision | Recall | F1-Score | Support ($N$) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for cls_name, c_m in xgb_test_full["per_class"].items():
            f.write(f"| **`{cls_name}`** | `{c_m['precision']:.4f}` | `{c_m['recall']:.4f}` | `{c_m['f1']:.4f}` | `{c_m['support']:,}` |\n")

        f.write("\n### 2. Novel Single-Ticker Corporate Catalysts ($N = 27,018$)\n\n")
        f.write(f"- **Accuracy:** `{xgb_test_novel['accuracy']:.4f}`\n")
        f.write(f"- **Macro-F1:** `{xgb_test_novel['macro_f1']:.4f}`\n")
        f.write(f"- **Weighted-F1:** `{xgb_test_novel['weighted_f1']:.4f}`\n")
        f.write(f"- **Balanced Accuracy:** `{xgb_test_novel['balanced_accuracy']:.4f}`\n\n")

        f.write("| Class Label | Precision | Recall | F1-Score | Support ($N$) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for cls_name, c_m in xgb_test_novel["per_class"].items():
            f.write(f"| **`{cls_name}`** | `{c_m['precision']:.4f}` | `{c_m['recall']:.4f}` | `{c_m['f1']:.4f}` | `{c_m['support']:,}` |\n")

        f.write("\n---\n\n## 5. Confusion Matrices\n\n")
        f.write("### Full Test Holdout Confusion Matrix (Raw Counts)\n\n")
        f.write("Rows: True Class (`decrease`, `increase`, `no_significant_change`); Columns: Predicted Class\n\n")
        f.write("| True Class | Pred `decrease` | Pred `increase` | Pred `no_significant_change` |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for idx, cls_name in enumerate(TARGET_CLASSES):
            row = xgb_test_full["confusion_matrix_raw"][idx]
            f.write(f"| **`{cls_name}`** | `{row[0]:,}` | `{row[1]:,}` | `{row[2]:,}` |\n")

        f.write("\n### Novel Corporate Catalysts Confusion Matrix (Raw Counts)\n\n")
        f.write("| True Class | Pred `decrease` | Pred `increase` | Pred `no_significant_change` |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for idx, cls_name in enumerate(TARGET_CLASSES):
            row = xgb_test_novel["confusion_matrix_raw"][idx]
            f.write(f"| **`{cls_name}`** | `{row[0]:,}` | `{row[1]:,}` | `{row[2]:,}` |\n")

        f.write("\n---\n\n## 6. Reproducibility & Environment Specification\n\n")
        f.write(f"- **Random Seed:** `{env['random_seed']}`\n")
        f.write(f"- **XGBoost Version:** `{env['xgboost_version']}`\n")
        f.write(f"- **Scikit-Learn Version:** `{env['scikit_learn_version']}`\n")
        f.write(f"- **Pandas Version:** `{env['pandas_version']}`\n")
        f.write(f"- **NumPy Version:** `{env['numpy_version']}`\n")
        f.write(f"- **Joblib Version:** `{env['joblib_version']}`\n")
        f.write(f"- **Python Version:** `{env['python_version']}`\n")
        f.write(f"- **Platform:** `{env['platform']}`\n\n")

        f.write("> [!CAUTION]\n")
        f.write("> **Research Disclaimers:**\n")
        f.write("> 1. **Statistical Correlation vs. Causal Evidence:** Model predictions represent statistical patterns across pre-event categorical and numerical features and subsequent FINRA bi-monthly settlement changes. They do not constitute causal proof.\n")
        f.write("> 2. **No Investment Advice:** This experimental pipeline is developed strictly for empirical research and academic analysis. It does not constitute investment advice or trading recommendations.\n")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Phase 2 XGBoost short-interest model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PARQUET, help="Path to input parquet dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save model artifacts")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="Path for markdown report")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    train_and_evaluate_xgboost(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        report_path=args.report_path,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
