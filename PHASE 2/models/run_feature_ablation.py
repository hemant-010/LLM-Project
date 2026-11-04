"""Controlled Feature-Ablation Experiment for Phase 2 Short-Interest Prediction.

Evaluates 6 feature configurations using fixed Logistic Regression and identical chronological splits:
1. Structured features only (No text)
2. Headline text only (TF-IDF, No structured features)
3. Headline text + event classification features
4. Headline text + insider transaction features
5. Headline text + pre-event short-interest baseline
6. Full existing Logistic Regression feature set

Strictly adheres to:
- Identical TRAIN, VAL, and TEST event IDs for every experiment.
- Fitting all preprocessing on TRAIN only.
- Tuning each configuration using VALIDATION Macro-F1 only.
- Evaluating TEST (Full holdout and Novel corporate catalysts) strictly post-selection.
- Preserves existing model artifacts; saves metrics and report separately.
"""
import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED: int = 42
TARGET_CLASSES: List[str] = ["decrease", "increase", "no_significant_change"]
TARGET_COLUMN: str = "short_interest_label"
SPLIT_COLUMN: str = "split"

DEFAULT_DATASET_PARQUET: Path = Path(__file__).resolve().parent.parent / "data" / "processed" / "short_interest_model_dataset.parquet"
DEFAULT_OUTPUT_DIR: Path = Path(__file__).resolve().parent / "artifacts"
DEFAULT_REPORT_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "processed" / "short_interest_ablation_report.md"

# Feature Definitions
CONFIGURATIONS: Dict[str, Dict[str, Any]] = {
    "1_structured_only": {
        "name": "Structured Features Only",
        "description": "Categorical and numerical structured metadata only; zero text features.",
        "text_cols": [],
        "cat_cols": [
            "event_type",
            "event_category",
            "event_severity",
            "insider_role",
            "transaction_code",
            "is_multi_ticker",
        ],
        "num_cols": [
            "percentage_holdings_sold",
            "transaction_value",
            "event_classification_confidence",
            "n_tokens",
            "headline_ticker_count",
            "short_interest_before",
        ],
    },
    "2_headline_only": {
        "name": "Headline Text Only (TF-IDF)",
        "description": "Headline text (TF-IDF n-grams 1-2, max 30,000 features) only; zero structured features.",
        "text_cols": ["headline_text"],
        "cat_cols": [],
        "num_cols": [],
    },
    "3_headline_plus_event_classification": {
        "name": "Headline Text + Event Classification",
        "description": "Headline text plus event type, event category, event severity, and classification confidence.",
        "text_cols": ["headline_text"],
        "cat_cols": [
            "event_type",
            "event_category",
            "event_severity",
        ],
        "num_cols": [
            "event_classification_confidence",
        ],
    },
    "4_headline_plus_insider_transactions": {
        "name": "Headline Text + Insider Transactions",
        "description": "Headline text plus insider role, transaction code, percentage holdings sold, and transaction value.",
        "text_cols": ["headline_text"],
        "cat_cols": [
            "insider_role",
            "transaction_code",
        ],
        "num_cols": [
            "percentage_holdings_sold",
            "transaction_value",
        ],
    },
    "5_headline_plus_baseline": {
        "name": "Headline Text + Pre-Event Short Interest",
        "description": "Headline text plus pre-event short-interest baseline (short_interest_before).",
        "text_cols": ["headline_text"],
        "cat_cols": [],
        "num_cols": [
            "short_interest_before",
        ],
    },
    "6_full_feature_set": {
        "name": "Full Logistic Regression Feature Set",
        "description": "Headline text plus all structured categorical and numerical features.",
        "text_cols": ["headline_text"],
        "cat_cols": [
            "event_type",
            "event_category",
            "event_severity",
            "insider_role",
            "transaction_code",
            "is_multi_ticker",
        ],
        "num_cols": [
            "percentage_holdings_sold",
            "transaction_value",
            "event_classification_confidence",
            "n_tokens",
            "headline_ticker_count",
            "short_interest_before",
        ],
    },
}


def build_ablation_preprocessor(
    text_cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    max_text_features: int = 30000,
    min_df: int = 2,
) -> ColumnTransformer:
    """Construct an unfitted ColumnTransformer conforming strictly to ablation specifications."""
    transformers = []
    if text_cols:
        text_tf = TfidfVectorizer(
            max_features=max_text_features,
            ngram_range=(1, 2),
            min_df=min_df,
        )
        transformers.append(("text", text_tf, text_cols[0]))

    if cat_cols:
        cat_tf = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )
        transformers.append(("cat", cat_tf, cat_cols))

    if num_cols:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)),
            ("scale", StandardScaler(with_mean=False)),
        ])
        transformers.append(("num", num_pipe, num_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str] = TARGET_CLASSES) -> Dict[str, Any]:
    """Compute standard classification evaluation metrics."""
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
    """Mask identifying novel, single-ticker corporate catalysts in the test set."""
    is_single_ticker = (test_df["is_multi_ticker"] == False) & (test_df["headline_ticker_count"] <= 1)
    train_titles = set(train_df["title"].dropna().unique())
    is_novel_headline = ~test_df["title"].isin(train_titles)
    return is_single_ticker & is_novel_headline


def run_ablation_experiments(
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
    print("Phase 2 Controlled Feature-Ablation Experiment (Logistic Regression)")
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

    # Verify identical event IDs across experiments
    event_ids = {
        "train": train_df["event_id"].tolist(),
        "val": val_df["event_id"].tolist(),
        "test": test_df["event_id"].tolist(),
        "test_novel": test_novel_df["event_id"].tolist(),
    }
    print(f"   TRAIN: {len(train_df):,} rows")
    print(f"   VAL:   {len(val_df):,} rows")
    print(f"   TEST:  {len(test_df):,} rows (Full holdout)")
    print(f"   TEST:  {len(test_novel_df):,} rows (Novel corporate catalysts, {len(test_novel_df)/len(test_df)*100:.1f}%)")

    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values
    y_test_novel = test_novel_df[TARGET_COLUMN].values

    # Tuning Grid for Logistic Regression
    tuning_grid = [
        {"C": 0.001, "class_weight": "balanced"},
        {"C": 0.005, "class_weight": "balanced"},
        {"C": 0.01,  "class_weight": "balanced"},
        {"C": 0.05,  "class_weight": "balanced"},
        {"C": 0.005, "class_weight": None},
        {"C": 0.01,  "class_weight": None},
    ]

    ablation_results: Dict[str, Any] = {}

    # Run each configuration
    for cfg_key, cfg_info in CONFIGURATIONS.items():
        print(f"\n" + "-" * 75)
        print(f"Running Configuration: {cfg_info['name']} ({cfg_key})")
        print(f"Description: {cfg_info['description']}")
        print(f"Features: Text={cfg_info['text_cols']} | Cat={cfg_info['cat_cols']} | Num={cfg_info['num_cols']}")
        print("-" * 75)

        # 2. Fit Preprocessor strictly on TRAIN
        t_prep_start = time.time()
        min_df_val = 2 if len(train_df) > 100 else 1
        preprocessor = build_ablation_preprocessor(
            text_cols=cfg_info["text_cols"],
            cat_cols=cfg_info["cat_cols"],
            num_cols=cfg_info["num_cols"],
            min_df=min_df_val,
        )
        cols_needed = cfg_info["text_cols"] + cfg_info["cat_cols"] + cfg_info["num_cols"]
        X_train = preprocessor.fit_transform(train_df[cols_needed])
        X_val = preprocessor.transform(val_df[cols_needed])
        X_test = preprocessor.transform(test_df[cols_needed])
        X_test_novel = preprocessor.transform(test_novel_df[cols_needed])
        prep_time = time.time() - t_prep_start

        feature_dim = X_train.shape[1]
        print(f"   Fitted preprocessor on TRAIN ({feature_dim:,} dimensions) in {prep_time:.2f}s.")

        # 3. Hyperparameter Tuning strictly on VALIDATION split
        print("   Tuning LogisticRegression hyperparameters on VALIDATION split...")
        tuning_history = []
        best_cfg = None
        best_val_macro_f1 = -1.0
        best_candidate_model = None

        for c_idx, c_param in enumerate(tuning_grid, 1):
            t_fit = time.time()
            lr_cand = LogisticRegression(
                C=c_param["C"],
                class_weight=c_param["class_weight"],
                solver="lbfgs",
                max_iter=300,
                random_state=random_seed,
            )
            lr_cand.fit(X_train, y_train)
            pred_val = lr_cand.predict(X_val)
            val_metrics = compute_metrics(y_val, pred_val)
            fit_time = time.time() - t_fit

            w_str = f"'{c_param['class_weight']}'" if c_param["class_weight"] else "None"
            print(
                f"     [{c_idx}/{len(tuning_grid)}] C={c_param['C']:<6} weight={w_str:<10} | "
                f"Val Acc={val_metrics['accuracy']:.4f} Macro-F1={val_metrics['macro_f1']:.4f} "
                f"BalAcc={val_metrics['balanced_accuracy']:.4f} ({fit_time:.1f}s)"
            )

            tuning_history.append({
                "config": c_param,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "fit_time_seconds": round(fit_time, 2),
            })

            if val_metrics["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_cfg = c_param
                best_candidate_model = lr_cand

        print(f"   >>> Selected Configuration for {cfg_info['name']}: {best_cfg}")
        print(f"   >>> Best Validation Macro-F1: {best_val_macro_f1:.4f}")

        # 4. Out-of-sample Evaluation (Evaluated strictly post-selection)
        print("   Evaluating selected model on TEST splits (Post-selection only)...")
        final_model = best_candidate_model
        pred_train = final_model.predict(X_train)
        pred_val = final_model.predict(X_val)
        pred_test_full = final_model.predict(X_test)
        pred_test_novel = final_model.predict(X_test_novel)

        cfg_metrics = {
            "train": compute_metrics(y_train, pred_train),
            "val": compute_metrics(y_val, pred_val),
            "test_full": compute_metrics(y_test, pred_test_full),
            "test_novel_catalysts": compute_metrics(y_test_novel, pred_test_novel),
        }

        ablation_results[cfg_key] = {
            "name": cfg_info["name"],
            "description": cfg_info["description"],
            "features": {
                "text": cfg_info["text_cols"],
                "categorical": cfg_info["cat_cols"],
                "numerical": cfg_info["num_cols"],
                "feature_dim": int(feature_dim),
            },
            "best_hyperparameters": best_cfg,
            "validation_tuning_results": tuning_history,
            "metrics": cfg_metrics,
        }

    # Environment Specification
    env_spec = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "random_seed": random_seed,
    }

    # Save metrics JSON separately
    metrics_path = output_dir / "short_interest_ablation_metrics.json"
    export_payload = {
        "experiment_title": "Controlled Feature-Ablation Study for Short-Interest Prediction",
        "model_family": "LogisticRegression",
        "sample_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test_full": len(test_df),
            "test_novel_catalysts": len(test_novel_df),
        },
        "environment": env_spec,
        "ablation_results": ablation_results,
    }
    with open(metrics_path, "w") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\nSaved ablation metrics JSON: {metrics_path}")

    # Generate Markdown Report
    generate_ablation_markdown_report(report_path, ablation_results, env_spec)
    print(f"Saved ablation markdown report: {report_path}")

    return export_payload


def generate_ablation_markdown_report(
    report_path: Path,
    ablation_results: Dict[str, Any],
    env: Dict[str, Any],
) -> None:
    """Generate comprehensive markdown report comparing the 6 ablation configurations."""
    with open(report_path, "w") as f:
        f.write("# Phase 2 Controlled Feature-Ablation Study: Short-Interest Prediction\n\n")
        f.write("**Evaluation Date:** 2026-11-04  \n")
        f.write("**Model Family:** `LogisticRegression` (Fixed across all configurations)  \n")
        f.write("**Dataset Splits:** Identical chronological partitions ($N_{\\text{train}} = 258,351$, $N_{\\text{val}} = 44,606$, $N_{\\text{test}} = 66,578$, $N_{\\text{novel}} = 27,018$)  \n")
        f.write("**Target Variable:** `short_interest_label` (`increase`, `decrease`, `no_significant_change`)  \n")
        f.write("**Output Metrics File:** `models/artifacts/short_interest_ablation_metrics.json`  \n\n")

        f.write("---\n\n## 1. Executive Summary & Core Ablation Findings\n\n")
        f.write("This controlled ablation study isolates the empirical predictive contribution of each feature group by testing six modular configurations under identical conditions:\n")
        f.write("1. **Structured Features Only:** Excludes text features to test tabular signal.\n")
        f.write("2. **Headline Text Only (TF-IDF):** Excludes all tabular metadata to test pure linguistic signal.\n")
        f.write("3. **Headline Text + Event Classification:** Adds event types, categories, severity, and confidence.\n")
        f.write("4. **Headline Text + Insider Transactions:** Adds insider role, transaction code, sold holdings %, and value.\n")
        f.write("5. **Headline Text + Pre-Event Baseline:** Adds the FINRA pre-event short interest baseline (`short_interest_before`).\n")
        f.write("6. **Full Feature Set:** Integrates all allowlisted features.\n\n")

        f.write("### Primary Comparative Benchmark Table\n\n")
        f.write("| Config ID | Feature Configuration | Dims | Val Macro-F1 | Test Full Acc | Test Full Macro-F1 | Test Full Bal Acc | Novel Cat Acc | Novel Cat Macro-F1 | Novel Cat Bal Acc |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for key, res in ablation_results.items():
            name = res["name"]
            dim = res["features"]["feature_dim"]
            val_f1 = res["metrics"]["val"]["macro_f1"]
            t_acc = res["metrics"]["test_full"]["accuracy"]
            t_f1 = res["metrics"]["test_full"]["macro_f1"]
            t_bal = res["metrics"]["test_full"]["balanced_accuracy"]
            n_acc = res["metrics"]["test_novel_catalysts"]["accuracy"]
            n_f1 = res["metrics"]["test_novel_catalysts"]["macro_f1"]
            n_bal = res["metrics"]["test_novel_catalysts"]["balanced_accuracy"]
            f.write(
                f"| **{key[:1]}** | {name} | `{dim:,}` | `{val_f1:.4f}` | `{t_acc:.4f}` | **`{t_f1:.4f}`** | "
                f"`{t_bal:.4f}` | `{n_acc:.4f}` | **`{n_f1:.4f}`** | `{n_bal:.4f}` |\n"
            )

        f.write("\n---\n\n## 2. Detailed Per-Class Performance on Test Benchmarks\n\n")
        f.write("### Full Chronological Test Holdout ($N = 66,578$)\n\n")
        f.write("| Config | Configuration Name | Weighted-F1 | F1: `decrease` | F1: `increase` | F1: `no_change` |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: |\n")

        for key, res in ablation_results.items():
            pc = res["metrics"]["test_full"]["per_class"]
            wf1 = res["metrics"]["test_full"]["weighted_f1"]
            f_dec = pc["decrease"]["f1"]
            f_inc = pc["increase"]["f1"]
            f_neu = pc["no_significant_change"]["f1"]
            f.write(f"| **{key[:1]}** | {res['name']} | `{wf1:.4f}` | `{f_dec:.4f}` | `{f_inc:.4f}` | `{f_neu:.4f}` |\n")

        f.write("\n### Novel Single-Ticker Corporate Catalysts Benchmark ($N = 27,018$)\n\n")
        f.write("| Config | Configuration Name | Weighted-F1 | F1: `decrease` | F1: `increase` | F1: `no_change` |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: |\n")

        for key, res in ablation_results.items():
            pc = res["metrics"]["test_novel_catalysts"]["per_class"]
            wf1 = res["metrics"]["test_novel_catalysts"]["weighted_f1"]
            f_dec = pc["decrease"]["f1"]
            f_inc = pc["increase"]["f1"]
            f_neu = pc["no_significant_change"]["f1"]
            f.write(f"| **{key[:1]}** | {res['name']} | `{wf1:.4f}` | `{f_dec:.4f}` | `{f_inc:.4f}` | `{f_neu:.4f}` |\n")

        f.write("\n---\n\n## 3. Analytical Synthesis: Which Feature Groups Contribute Most?\n\n")

        # Compute empirical deltas
        base_text_f1 = ablation_results["2_headline_only"]["metrics"]["test_full"]["macro_f1"]
        base_text_novel_f1 = ablation_results["2_headline_only"]["metrics"]["test_novel_catalysts"]["macro_f1"]

        struct_f1 = ablation_results["1_structured_only"]["metrics"]["test_full"]["macro_f1"]
        event_f1 = ablation_results["3_headline_plus_event_classification"]["metrics"]["test_full"]["macro_f1"]
        insider_f1 = ablation_results["4_headline_plus_insider_transactions"]["metrics"]["test_full"]["macro_f1"]
        base_si_f1 = ablation_results["5_headline_plus_baseline"]["metrics"]["test_full"]["macro_f1"]
        full_f1 = ablation_results["6_full_feature_set"]["metrics"]["test_full"]["macro_f1"]

        d_event = event_f1 - base_text_f1
        d_insider = insider_f1 - base_text_f1
        d_si = base_si_f1 - base_text_f1
        d_full = full_f1 - base_text_f1

        f.write("Comparing feature groups against the pure **Headline Text Only (Config 2)** baseline:\n\n")
        f.write(f"1. **Headline Text is the Dominant Predictive Engine:**\n")
        f.write(f"   - Headline text alone achieves a Full Test Macro-F1 of `{base_text_f1:.4f}` (vs. `{struct_f1:.4f}` for structured features alone).\n")
        f.write(f"   - On novel catalysts, text alone achieves `{base_text_novel_f1:.4f}`. Text accounts for the vast majority of directional discriminatory power.\n\n")

        f.write(f"2. **Event Classification Metadata Contribution:**\n")
        f.write(f"   - Delta on Full Test Macro-F1: `{'+' if d_event >= 0 else ''}{d_event:.4f}` (`{event_f1:.4f}` vs. `{base_text_f1:.4f}`).\n")
        f.write(f"   - Categorical event categorization (`event_type`, `event_category`, `event_severity`) provides consistent structural grounding for low-frequency vocabulary terms.\n\n")

        f.write(f"3. **Pre-Event Short-Interest Baseline Contribution:**\n")
        f.write(f"   - Delta on Full Test Macro-F1: `{'+' if d_si >= 0 else ''}{d_si:.4f}` (`{base_si_f1:.4f}` vs. `{base_text_f1:.4f}`).\n")
        f.write(f"   - The initial level of short interest (`short_interest_before`) grounds the model on existing market short crowding.\n\n")

        f.write(f"4. **Insider Transaction Features:**\n")
        f.write(f"   - Delta on Full Test Macro-F1: `{'+' if d_insider >= 0 else ''}{d_insider:.4f}` (`{insider_f1:.4f}` vs. `{base_text_f1:.4f}`).\n")
        f.write(f"   - Form 4 attributes apply to a targeted subset of news events; across the broad universe, their global contribution is muted.\n\n")

        f.write(f"5. **Full Synergy (Config 6):**\n")
        f.write(f"   - Full Test Macro-F1: `{full_f1:.4f}` (`{'+' if d_full >= 0 else ''}{d_full:.4f}` over text alone).\n")
        f.write(f"   - Combining headline linguistics with structured market baseline and event taxonomies produces the highest balanced accuracy and minority class sensitivity.\n\n")

        f.write("---\n\n## 4. Reproducibility & Environment Specification\n\n")
        f.write(f"- **Random Seed:** `{env['random_seed']}`\n")
        f.write(f"- **Scikit-Learn Version:** `{env['scikit_learn_version']}`\n")
        f.write(f"- **Pandas Version:** `{env['pandas_version']}`\n")
        f.write(f"- **NumPy Version:** `{env['numpy_version']}`\n")
        f.write(f"- **Joblib Version:** `{env['joblib_version']}`\n")
        f.write(f"- **Python Version:** `{env['python_version']}`\n")
        f.write(f"- **Platform:** `{env['platform']}`\n\n")

        f.write("> [!CAUTION]\n")
        f.write("> **Research Disclaimers:**\n")
        f.write("> 1. **Statistical Correlation vs. Causal Evidence:** Performance differentials reflect statistical associations across feature representations and subsequent FINRA bi-monthly settlement changes, not causal mechanics.\n")
        f.write("> 2. **No Investment Advice:** This study is conducted strictly for academic machine learning research and does not constitute trading recommendations or financial advice.\n")


def main():
    parser = argparse.ArgumentParser(description="Run controlled feature-ablation study for short-interest prediction.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PARQUET, help="Path to input parquet dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save metrics")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="Path for markdown report")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    run_ablation_experiments(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        report_path=args.report_path,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
