"""Train, tune, and evaluate baseline short-interest prediction models for Phase 2.

Protocol:
1. Feature Pipeline:
   - Uses `headline_text` as the primary text feature (headline only, free of ticker/company identity leakage).
   - Preprocessing (TF-IDF vectorizer, OneHotEncoder, StandardScaler, SimpleImputer) fit strictly on TRAIN only.
2. Baselines:
   - DummyClassifier (majority class: 'increase').
   - LogisticRegression baseline with hyperparameter tuning on VALIDATION.
3. Model Selection:
   - Tunes C and class_weight strictly on the VALIDATION split.
   - Evaluates on the TEST split only after model selection is finalized.
4. Evaluation Benchmarks:
   - Primary: Full chronological test holdout (N = 66,578).
   - Secondary: Novel single-ticker corporate-catalyst benchmark (N = 27,018, is_multi_ticker == False and title not in train).
5. Reporting:
   - Accuracy, macro-F1, weighted-F1, balanced accuracy, per-class precision/recall/F1, confusion matrices.
   - Saves artifacts to models/artifacts/ and data/processed/.
"""
import sys
import json
import time
import platform
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.pipeline import Pipeline

# Ensure Phase 2 root in path
_phase2_root = Path(__file__).resolve().parent.parent
if str(_phase2_root) not in sys.path:
    sys.path.insert(0, str(_phase2_root))

from config import MODEL_DIR, DATA_PROCESSED
from models.feature_pipeline import (
    prepare_splits_and_transforms,
    DEFAULT_TEXT_FEATURE,
    MODEL_FEATURE_ALLOWLIST
)

RANDOM_SEED = 42
TARGET_CLASSES = ["decrease", "increase", "no_significant_change"]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=TARGET_CLASSES, zero_division=0
    )

    per_class = {}
    for i, cls_name in enumerate(TARGET_CLASSES):
        per_class[cls_name] = {
            "precision": round(float(precisions[i]), 4),
            "recall": round(float(recalls[i]), 4),
            "f1": round(float(f1s[i]), 4),
            "support": int(supports[i])
        }

    cm_raw = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES).tolist()
    cm_norm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES, normalize="true").tolist()
    cm_norm_rounded = [[round(val, 4) for val in row] for row in cm_norm]

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "per_class": per_class,
        "confusion_matrix_raw": cm_raw,
        "confusion_matrix_normalized": cm_norm_rounded,
        "classes": TARGET_CLASSES
    }

def run_training_and_evaluation(
    dataset_path: Path = DATA_PROCESSED / "short_interest_model_dataset.parquet",
    output_dir: Path = MODEL_DIR,
    report_dir: Path = DATA_PROCESSED
) -> Dict[str, Any]:
    print("=" * 80)
    print("Phase 2 Short-Interest Baseline Model Training & Evaluation")
    print("=" * 80)

    # 1. Load dataset and prepare splits
    print(f"\n1. Loading dataset from {dataset_path.name}...")
    df = pd.read_parquet(dataset_path) if dataset_path.suffix == ".parquet" else pd.read_csv(dataset_path)
    print(f"   Total events: {len(df):,}")

    # Inspect class distributions
    class_dists = {}
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        vc = sub["short_interest_label"].value_counts().to_dict()
        n = len(sub)
        class_dists[s] = {k: {"count": vc.get(k, 0), "pct": round(vc.get(k, 0) / n * 100, 2)} for k in TARGET_CLASSES}
        print(f"   {s.upper():5s} (N={n:,}): " + ", ".join([f"{k}: {vc.get(k,0):,} ({vc.get(k,0)/n*100:.1f}%)" for k in TARGET_CLASSES]))

    print(f"\n2. Executing train-only preprocessing with default text feature '{DEFAULT_TEXT_FEATURE}'...")
    t0 = time.time()
    split_data = prepare_splits_and_transforms(df, use_headline_only=True, max_text_features=30000)
    print(f"   Preprocessing completed in {time.time()-t0:.2f}s.")
    print(f"   Feature dimensionality: {split_data['metadata']['feature_dim']:,} features.")

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_val = split_data["X_val"]
    y_val = split_data["y_val"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    novel_mask_test = split_data["novel_catalyst_mask_test"].values
    X_test_novel = X_test[novel_mask_test]
    y_test_novel = y_test[novel_mask_test]

    print(f"   Train samples: {X_train.shape[0]:,}")
    print(f"   Validation samples: {X_val.shape[0]:,}")
    print(f"   Test samples (Full holdout): {X_test.shape[0]:,}")
    print(f"   Test samples (Novel corporate catalysts): {X_test_novel.shape[0]:,} ({X_test_novel.shape[0]/X_test.shape[0]*100:.1f}%)")

    # 3. Train Majority Class Dummy Baseline
    print("\n3. Training Majority-Class Dummy Baseline...")
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy.fit(X_train, y_train)
    dummy_majority_class = str(dummy.predict(X_train[:1])[0])
    print(f"   Majority class predicted: '{dummy_majority_class}'")

    dummy_metrics = {
        "train": compute_metrics(y_train, dummy.predict(X_train)),
        "val": compute_metrics(y_val, dummy.predict(X_val)),
        "test_full": compute_metrics(y_test, dummy.predict(X_test)),
        "test_novel_catalysts": compute_metrics(y_test_novel, dummy.predict(X_test_novel))
    }
    print(f"   Dummy Validation: Acc={dummy_metrics['val']['accuracy']:.4f}, Macro-F1={dummy_metrics['val']['macro_f1']:.4f}, BalAcc={dummy_metrics['val']['balanced_accuracy']:.4f}")
    print(f"   Dummy Full Test:  Acc={dummy_metrics['test_full']['accuracy']:.4f}, Macro-F1={dummy_metrics['test_full']['macro_f1']:.4f}, BalAcc={dummy_metrics['test_full']['balanced_accuracy']:.4f}")
    print(f"   Dummy Novel Cat:  Acc={dummy_metrics['test_novel_catalysts']['accuracy']:.4f}, Macro-F1={dummy_metrics['test_novel_catalysts']['macro_f1']:.4f}, BalAcc={dummy_metrics['test_novel_catalysts']['balanced_accuracy']:.4f}")

    # 4. Hyperparameter Tuning on VALIDATION split
    print("\n4. Tuning LogisticRegression Hyperparameters strictly on VALIDATION split...")
    tuning_grid = [
        {"C": 0.001, "class_weight": "balanced"},
        {"C": 0.005, "class_weight": "balanced"},
        {"C": 0.01,  "class_weight": "balanced"},
        {"C": 0.05,  "class_weight": "balanced"},
        {"C": 0.1,   "class_weight": "balanced"},
        {"C": 1.0,   "class_weight": "balanced"},
        {"C": 0.005, "class_weight": None},
        {"C": 0.01,  "class_weight": None},
        {"C": 0.1,   "class_weight": None},
        {"C": 1.0,   "class_weight": None},
    ]

    tuning_results = []
    best_config = None
    best_val_macro_f1 = -1.0

    for idx, cfg in enumerate(tuning_grid, 1):
        t_start = time.time()
        lr_cand = LogisticRegression(
            C=cfg["C"],
            class_weight=cfg["class_weight"],
            solver="lbfgs",
            max_iter=300,
            random_state=RANDOM_SEED
        )
        lr_cand.fit(X_train, y_train)
        pred_val_cand = lr_cand.predict(X_val)
        val_metrics_cand = compute_metrics(y_val, pred_val_cand)
        elapsed = time.time() - t_start

        tuning_entry = {
            "config": cfg,
            "val_accuracy": val_metrics_cand["accuracy"],
            "val_macro_f1": val_metrics_cand["macro_f1"],
            "val_weighted_f1": val_metrics_cand["weighted_f1"],
            "val_balanced_accuracy": val_metrics_cand["balanced_accuracy"],
            "fit_time_seconds": round(elapsed, 2)
        }
        tuning_results.append(tuning_entry)

        weight_str = f"'{cfg['class_weight']}'" if cfg["class_weight"] else "None"
        print(f"   [{idx:2d}/{len(tuning_grid)}] C={cfg['C']:<6} weight={weight_str:<10} | Val Acc={val_metrics_cand['accuracy']:.4f} "
              f"Macro-F1={val_metrics_cand['macro_f1']:.4f} BalAcc={val_metrics_cand['balanced_accuracy']:.4f} ({elapsed:.1f}s)")

        # Select by highest Validation Macro-F1 (balanced sensitivity across all 3 classes)
        if val_metrics_cand["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics_cand["macro_f1"]
            best_config = cfg

    print(f"\n   >>> Model Selection Complete: Best Validation Configuration: {best_config}")
    print(f"   >>> Selected Validation Macro-F1: {best_val_macro_f1:.4f}")

    # 5. Fit Selected Final Model & Evaluate on TEST
    print("\n5. Fitting selected model configuration on TRAIN split...")
    selected_lr = LogisticRegression(
        C=best_config["C"],
        class_weight=best_config["class_weight"],
        solver="lbfgs",
        max_iter=500,
        random_state=RANDOM_SEED
    )
    t_fit = time.time()
    selected_lr.fit(X_train, y_train)
    fit_duration = time.time() - t_fit
    print(f"   Fitted in {fit_duration:.2f}s.")

    print("\n6. Evaluating selected model across all splits (Test evaluated only after selection)...")
    pred_train = selected_lr.predict(X_train)
    pred_val = selected_lr.predict(X_val)
    pred_test_full = selected_lr.predict(X_test)
    pred_test_novel = selected_lr.predict(X_test_novel)

    lr_metrics = {
        "train": compute_metrics(y_train, pred_train),
        "val": compute_metrics(y_val, pred_val),
        "test_full": compute_metrics(y_test, pred_test_full),
        "test_novel_catalysts": compute_metrics(y_test_novel, pred_test_novel)
    }

    # Print Summary Table
    print("\n" + "=" * 80)
    print("EVALUATION PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Split / Benchmark':<30} | {'Model':<15} | {'Accuracy':<9} | {'Macro-F1':<9} | {'Bal Acc':<9} | {'Gain vs Dummy':<14}")
    print("-" * 95)

    comparisons = [
        ("Validation Set", "Dummy (Majority)", dummy_metrics["val"], None),
        ("Validation Set", "LogisticRegression", lr_metrics["val"], dummy_metrics["val"]),
        ("Test Set (Full Holdout)", "Dummy (Majority)", dummy_metrics["test_full"], None),
        ("Test Set (Full Holdout)", "LogisticRegression", lr_metrics["test_full"], dummy_metrics["test_full"]),
        ("Test Set (Novel Catalysts)", "Dummy (Majority)", dummy_metrics["test_novel_catalysts"], None),
        ("Test Set (Novel Catalysts)", "LogisticRegression", lr_metrics["test_novel_catalysts"], dummy_metrics["test_novel_catalysts"]),
    ]

    for split_label, model_label, m, d in comparisons:
        gain_str = "-"
        if d is not None:
            f1_diff = m["macro_f1"] - d["macro_f1"]
            gain_str = f"+{f1_diff:.4f} F1"
        print(f"{split_label:<30} | {model_label:<15} | {m['accuracy']:<9.4f} | {m['macro_f1']:<9.4f} | {m['balanced_accuracy']:<9.4f} | {gain_str:<14}")

    print("\nDetailed Per-Class Performance on Full Test Set (LogisticRegression):")
    for cls_name, vals in lr_metrics["test_full"]["per_class"].items():
        print(f"  {cls_name:<22}: Precision={vals['precision']:.4f}, Recall={vals['recall']:.4f}, F1={vals['f1']:.4f} (N={vals['support']:,})")

    print("\nDetailed Per-Class Performance on Novel Corporate Catalysts (LogisticRegression):")
    for cls_name, vals in lr_metrics["test_novel_catalysts"]["per_class"].items():
        print(f"  {cls_name:<22}: Precision={vals['precision']:.4f}, Recall={vals['recall']:.4f}, F1={vals['f1']:.4f} (N={vals['support']:,})")

    # 7. Construct Full Pipeline and Save Artifacts
    print("\n7. Saving model pipeline, parameters, metrics, and markdown evaluation report...")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Build end-to-end inference Pipeline compatible with DualPredictor
    full_pipeline = Pipeline([
        ("features", split_data["preprocessor"]),
        ("classifier", selected_lr)
    ])

    environment_metadata = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "random_seed": RANDOM_SEED
    }

    artifact_payload = {
        "model": full_pipeline,
        "target": "short_interest_label",
        "horizon": "next eligible short-interest reporting period",
        "model_type": "LogisticRegression",
        "text_feature": DEFAULT_TEXT_FEATURE,
        "best_hyperparameters": best_config,
        "feature_dim": split_data["metadata"]["feature_dim"],
        "classes": TARGET_CLASSES,
        "metrics": {
            "dummy_baseline": dummy_metrics,
            "logistic_regression": lr_metrics
        },
        "environment": environment_metadata
    }

    joblib_path = output_dir / "short_interest_model.joblib"
    joblib.dump(artifact_payload, joblib_path)
    print(f"   Model pipeline saved: {joblib_path}")

    # Save metrics JSON
    metrics_json_path = output_dir / "short_interest_baseline_metrics.json"
    metrics_export = {
        "model_type": "LogisticRegression",
        "text_feature": DEFAULT_TEXT_FEATURE,
        "best_hyperparameters": best_config,
        "validation_tuning_results": tuning_results,
        "dummy_majority_metrics": dummy_metrics,
        "logistic_regression_metrics": lr_metrics,
        "class_distributions": class_dists,
        "environment": environment_metadata
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"   Metrics JSON saved:   {metrics_json_path}")

    # Generate Markdown Evaluation Report
    report_md_path = report_dir / "short_interest_baseline_evaluation_report.md"
    _generate_markdown_report(report_md_path, metrics_export, artifact_payload)
    print(f"   Evaluation report:    {report_md_path}")
    print("=" * 80)

    return metrics_export

def _generate_markdown_report(report_path: Path, metrics: Dict[str, Any], artifact: Dict[str, Any]) -> None:
    lr_m = metrics["logistic_regression_metrics"]
    dum_m = metrics["dummy_majority_metrics"]
    tuning = metrics["validation_tuning_results"]
    best_cfg = metrics["best_hyperparameters"]
    env = metrics["environment"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2 Baseline Short-Interest Prediction Model Evaluation Report\n\n")
        f.write("> **Research Disclaimer:** This report documents experimental statistical classification models for academic and algorithmic research purposes only. Predictive scores and directional probabilities represent empirical pattern correlations and **do not constitute causal evidence, trading recommendations, or investment advice**.\n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Task:** Predict short-interest direction (`increase`, `decrease`, `no_significant_change`) at next FINRA reporting settlement.\n")
        f.write(f"- **Primary Text Representation:** `{metrics['text_feature']}` (Headline/title only, free of company name or ticker identity tags).\n")
        f.write(f"- **Selected Model:** `LogisticRegression(C={best_cfg['C']}, class_weight='{best_cfg['class_weight']}', solver='lbfgs')`.\n")
        f.write(f"- **Feature Dimension:** `{artifact['feature_dim']:,}` sparse features (TF-IDF 1-2 ngrams, one-hot event taxonomy, scaled continuous signals).\n")
        f.write(f"- **Model Artifact:** `models/artifacts/short_interest_model.joblib`\n\n")

        f.write("## 2. Dataset Partitions & Class Distributions\n\n")
        f.write("| Split | Event Count | Calendar Range | `increase` (%) | `decrease` (%) | `no_significant_change` (%) |\n")
        f.write("| :--- | :---: | :--- | :---: | :---: | :---: |\n")
        for s in ["train", "val", "test"]:
            dist = metrics["class_distributions"][s]
            f.write(f"| **{s.upper()}** | `{sum(v['count'] for v in dist.values()):,}` | ")
            if s == "train":
                f.write("`2018-01-11` to `2019-11-30` | ")
            elif s == "val":
                f.write("`2019-12-01` to `2020-02-29` | ")
            else:
                f.write("`2020-03-01` to `2020-06-11` | ")
            f.write(f"{dist['increase']['pct']}% | {dist['decrease']['pct']}% | {dist['no_significant_change']['pct']}% |\n")

        f.write("\n## 3. Validation Model Selection & Hyperparameter Tuning\n\n")
        f.write("Hyperparameter grid search conducted strictly on the chronological **Validation split** (N = 44,606). The model was selected based on Validation Macro-F1 to guarantee balanced sensitivity across all three classes.\n\n")
        f.write("| C | `class_weight` | Validation Accuracy | Validation Macro-F1 | Validation Weighted-F1 | Validation Balanced Accuracy |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for t in tuning:
            cfg = t["config"]
            w_str = f"`{cfg['class_weight']}`" if cfg["class_weight"] else "`None`"
            is_best = "**(Selected)**" if cfg == best_cfg else ""
            f.write(f"| {cfg['C']} | {w_str} | {t['val_accuracy']:.4f} | **{t['val_macro_f1']:.4f}** | {t['val_weighted_f1']:.4f} | {t['val_balanced_accuracy']:.4f} {is_best} |\n")

        f.write(f"\n**Selection Conclusion:** Configuration `C = {best_cfg['C']}` with `class_weight = '{best_cfg['class_weight']}'` achieved the highest Macro-F1 (`{best_cfg}` on Validation). This configuration was frozen for out-of-sample Test evaluation.\n\n")

        f.write("## 4. Test Set Evaluation & Benchmark Comparison\n\n")
        f.write("Evaluated on the out-of-sample Test holdout only after selection was finalized:\n")
        f.write("1. **Full Chronological Holdout (N = 66,578):** Unconstrained real-world news stream.\n")
        f.write("2. **Secondary Novel Catalyst Benchmark (N = 27,018):** Single-ticker corporate announcements (`is_multi_ticker == False` and `title not in train`), eliminating repeated headline memorization.\n\n")

        f.write("### Benchmark Comparison vs Majority-Class Baseline\n\n")
        f.write("| Split / Benchmark | Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy | Gain vs Majority |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")

        rows = [
            ("Validation Split", "Dummy (Majority)", dum_m["val"], "-"),
            ("Validation Split", "LogisticRegression", lr_m["val"], f"+{lr_m['val']['macro_f1'] - dum_m['val']['macro_f1']:.4f} Macro-F1"),
            ("Test Split (Full Holdout)", "Dummy (Majority)", dum_m["test_full"], "-"),
            ("Test Split (Full Holdout)", "LogisticRegression", lr_m["test_full"], f"+{lr_m['test_full']['macro_f1'] - dum_m['test_full']['macro_f1']:.4f} Macro-F1"),
            ("Test Split (Novel Catalysts)", "Dummy (Majority)", dum_m["test_novel_catalysts"], "-"),
            ("Test Split (Novel Catalysts)", "LogisticRegression", lr_m["test_novel_catalysts"], f"+{lr_m['test_novel_catalysts']['macro_f1'] - dum_m['test_novel_catalysts']['macro_f1']:.4f} Macro-F1"),
        ]
        for s_lbl, m_lbl, data, gain in rows:
            f.write(f"| **{s_lbl}** | {m_lbl} | {data['accuracy']:.4f} | {data['macro_f1']:.4f} | {data['weighted_f1']:.4f} | {data['balanced_accuracy']:.4f} | {gain} |\n")

        f.write("\n### Per-Class Detailed Performance (LogisticRegression)\n\n")
        f.write("#### 1. Full Test Holdout (N = 66,578)\n\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for cls_name, c_data in lr_m["test_full"]["per_class"].items():
            f.write(f"| `{cls_name}` | {c_data['precision']:.4f} | {c_data['recall']:.4f} | {c_data['f1']:.4f} | {c_data['support']:,} |\n")

        f.write("\n#### 2. Novel Single-Ticker Corporate Catalysts (N = 27,018)\n\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for cls_name, c_data in lr_m["test_novel_catalysts"]["per_class"].items():
            f.write(f"| `{cls_name}` | {c_data['precision']:.4f} | {c_data['recall']:.4f} | {c_data['f1']:.4f} | {c_data['support']:,} |\n")

        f.write("\n## 5. Confusion Matrices\n\n")
        f.write("### Full Test Holdout Confusion Matrix (Raw Counts / Row-Normalized)\n\n")
        f.write("Rows represent true labels (`decrease`, `increase`, `no_significant_change`); columns represent predictions.\n\n")
        f.write("| True Class | Pred `decrease` | Pred `increase` | Pred `no_significant_change` |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for i, cls_name in enumerate(TARGET_CLASSES):
            raw_row = lr_m["test_full"]["confusion_matrix_raw"][i]
            norm_row = lr_m["test_full"]["confusion_matrix_normalized"][i]
            f.write(f"| **`{cls_name}`** | `{raw_row[0]:,}` ({norm_row[0]*100:.1f}%) | `{raw_row[1]:,}` ({norm_row[1]*100:.1f}%) | `{raw_row[2]:,}` ({norm_row[2]*100:.1f}%) |\n")

        f.write("\n### Novel Corporate Catalysts Confusion Matrix\n\n")
        f.write("| True Class | Pred `decrease` | Pred `increase` | Pred `no_significant_change` |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for i, cls_name in enumerate(TARGET_CLASSES):
            raw_row = lr_m["test_novel_catalysts"]["confusion_matrix_raw"][i]
            norm_row = lr_m["test_novel_catalysts"]["confusion_matrix_normalized"][i]
            f.write(f"| **`{cls_name}`** | `{raw_row[0]:,}` ({norm_row[0]*100:.1f}%) | `{raw_row[1]:,}` ({norm_row[1]*100:.1f}%) | `{raw_row[2]:,}` ({norm_row[2]*100:.1f}%) |\n")

        f.write("\n## 6. Reproducibility & Environment Specification\n\n")
        f.write(f"- **Random Seed:** `{env['random_seed']}`\n")
        f.write(f"- **Python Version:** `{env['python_version'].split()[0]}`\n")
        f.write(f"- **Platform:** `{env['platform']}`\n")
        f.write(f"- **Scikit-Learn Version:** `{env['scikit_learn_version']}`\n")
        f.write(f"- **Pandas Version:** `{env['pandas_version']}`\n")
        f.write(f"- **NumPy Version:** `{env['numpy_version']}`\n")
        f.write(f"- **Joblib Version:** `{env['joblib_version']}`\n")

if __name__ == "__main__":
    run_training_and_evaluation()
