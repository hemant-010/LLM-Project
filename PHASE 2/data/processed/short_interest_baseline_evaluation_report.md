# Phase 2 Baseline Short-Interest Prediction Model Evaluation Report

> **Research Disclaimer:** This report documents experimental statistical classification models for academic and algorithmic research purposes only. Predictive scores and directional probabilities represent empirical pattern correlations and **do not constitute causal evidence, trading recommendations, or investment advice**.

## 1. Executive Summary

- **Task:** Predict short-interest direction (`increase`, `decrease`, `no_significant_change`) at next FINRA reporting settlement.
- **Primary Text Representation:** `headline_text` (Headline/title only, free of company name or ticker identity tags).
- **Selected Model:** `LogisticRegression(C=0.005, class_weight='balanced', solver='lbfgs')`.
- **Feature Dimension:** `30,032` sparse features (TF-IDF 1-2 ngrams, one-hot event taxonomy, scaled continuous signals).
- **Model Artifact:** `models/artifacts/short_interest_model.joblib`

## 2. Dataset Partitions & Class Distributions

| Split | Event Count | Calendar Range | `increase` (%) | `decrease` (%) | `no_significant_change` (%) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **TRAIN** | `258,351` | `2018-01-11` to `2019-11-30` | 39.15% | 35.4% | 25.45% |
| **VAL** | `44,606` | `2019-12-01` to `2020-02-29` | 37.91% | 36.49% | 25.6% |
| **TEST** | `66,578` | `2020-03-01` to `2020-06-11` | 42.72% | 36.75% | 20.54% |

## 3. Validation Model Selection & Hyperparameter Tuning

Hyperparameter grid search conducted strictly on the chronological **Validation split** (N = 44,606). The model was selected based on Validation Macro-F1 to guarantee balanced sensitivity across all three classes.

| C | `class_weight` | Validation Accuracy | Validation Macro-F1 | Validation Weighted-F1 | Validation Balanced Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.001 | `balanced` | 0.3937 | **0.3781** | 0.3843 | 0.3886  |
| 0.005 | `balanced` | 0.3898 | **0.3788** | 0.3835 | 0.3883 **(Selected)** |
| 0.01 | `balanced` | 0.3876 | **0.3787** | 0.3830 | 0.3873  |
| 0.05 | `balanced` | 0.3833 | **0.3747** | 0.3778 | 0.3867  |
| 0.1 | `balanced` | 0.3719 | **0.3702** | 0.3727 | 0.3775  |
| 1.0 | `balanced` | 0.3698 | **0.3662** | 0.3695 | 0.3725  |
| 0.005 | `None` | 0.4199 | **0.3179** | 0.3459 | 0.3777  |
| 0.01 | `None` | 0.4200 | **0.3254** | 0.3539 | 0.3784  |
| 0.1 | `None` | 0.4082 | **0.3513** | 0.3754 | 0.3742  |
| 1.0 | `None` | 0.4015 | **0.3517** | 0.3727 | 0.3706  |

**Selection Conclusion:** Configuration `C = 0.005` with `class_weight = 'balanced'` achieved the highest Macro-F1 (`{'C': 0.005, 'class_weight': 'balanced'}` on Validation). This configuration was frozen for out-of-sample Test evaluation.

## 4. Test Set Evaluation & Benchmark Comparison

Evaluated on the out-of-sample Test holdout only after selection was finalized:
1. **Full Chronological Holdout (N = 66,578):** Unconstrained real-world news stream.
2. **Secondary Novel Catalyst Benchmark (N = 27,018):** Single-ticker corporate announcements (`is_multi_ticker == False` and `title not in train`), eliminating repeated headline memorization.

### Benchmark Comparison vs Majority-Class Baseline

| Split / Benchmark | Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy | Gain vs Majority |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation Split** | Dummy (Majority) | 0.3791 | 0.1833 | 0.2084 | 0.3333 | - |
| **Validation Split** | LogisticRegression | 0.3898 | 0.3788 | 0.3835 | 0.3883 | +0.1955 Macro-F1 |
| **Test Split (Full Holdout)** | Dummy (Majority) | 0.4272 | 0.1995 | 0.2557 | 0.3333 | - |
| **Test Split (Full Holdout)** | LogisticRegression | 0.3843 | 0.3459 | 0.3628 | 0.3761 | +0.1464 Macro-F1 |
| **Test Split (Novel Catalysts)** | Dummy (Majority) | 0.4067 | 0.1927 | 0.2352 | 0.3333 | - |
| **Test Split (Novel Catalysts)** | LogisticRegression | 0.3371 | 0.3337 | 0.3344 | 0.3657 | +0.1410 Macro-F1 |

### Per-Class Detailed Performance (LogisticRegression)

#### 1. Full Test Holdout (N = 66,578)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `decrease` | 0.3824 | 0.1455 | 0.2108 | 24,466 |
| `increase` | 0.4717 | 0.5816 | 0.5209 | 28,440 |
| `no_significant_change` | 0.2472 | 0.4013 | 0.3059 | 13,672 |

#### 2. Novel Single-Ticker Corporate Catalysts (N = 27,018)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `decrease` | 0.3718 | 0.2161 | 0.2733 | 10,060 |
| `increase` | 0.4602 | 0.3337 | 0.3869 | 10,988 |
| `no_significant_change` | 0.2475 | 0.5472 | 0.3408 | 5,970 |

## 5. Confusion Matrices

### Full Test Holdout Confusion Matrix (Raw Counts / Row-Normalized)

Rows represent true labels (`decrease`, `increase`, `no_significant_change`); columns represent predictions.

| True Class | Pred `decrease` | Pred `increase` | Pred `no_significant_change` |
| :--- | :---: | :---: | :---: |
| **`decrease`** | `3,560` (14.5%) | `12,320` (50.4%) | `8,586` (35.1%) |
| **`increase`** | `3,772` (13.3%) | `16,540` (58.2%) | `8,128` (28.6%) |
| **`no_significant_change`** | `1,977` (14.5%) | `6,208` (45.4%) | `5,487` (40.1%) |

### Novel Corporate Catalysts Confusion Matrix

| True Class | Pred `decrease` | Pred `increase` | Pred `no_significant_change` |
| :--- | :---: | :---: | :---: |
| **`decrease`** | `2,174` (21.6%) | `2,854` (28.4%) | `5,032` (50.0%) |
| **`increase`** | `2,418` (22.0%) | `3,667` (33.4%) | `4,903` (44.6%) |
| **`no_significant_change`** | `1,255` (21.0%) | `1,448` (24.2%) | `3,267` (54.7%) |

## 6. Reproducibility & Environment Specification

- **Random Seed:** `42`
- **Python Version:** `3.14.2`
- **Platform:** `macOS-27.0.1-arm64-arm-64bit-Mach-O`
- **Scikit-Learn Version:** `1.9.1`
- **Pandas Version:** `3.1.0`
- **NumPy Version:** `2.5.4`
- **Joblib Version:** `1.6.0`
