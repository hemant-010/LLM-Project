# Phase 2 Structured XGBoost Short-Interest Model Evaluation Report

**Evaluation Date:** 2026-11-04  
**Model Architecture:** XGBoost (`XGBClassifier` with histogram tree method)  
**Feature Scope:** Structured Categorical & Numerical Features Only (Zero text features, zero identifiers, zero dates)  
**Target Variable:** `short_interest_label` (`increase`, `decrease`, `no_significant_change`)  
**Artifact File:** `models/artifacts/short_interest_xgboost_model.joblib`  

---

## 1. Executive Summary & Comparative Highlights

This experiment evaluates an **XGBoost gradient boosted decision tree classifier** trained strictly on 12 pre-event structured features (6 categorical and 6 numerical) against both the `DummyClassifier` majority baseline and the `LogisticRegression` baseline.

### Model Comparison Across Benchmarks

| Evaluation Benchmark | Metric | Dummy Baseline | Logistic Regression | XGBoost (Structured) | Delta vs LR |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Validation Split** | Accuracy | `0.3791` | `0.3898` | **`0.3532`** | `-0.0366` |
| **Validation Split** | Macro-F1 | `0.1833` | `0.3788` | **`0.3447`** | `-0.0341` |
| **Validation Split** | Balanced Acc | `0.3333` | `0.3883` | **`0.3765`** | `-0.0118` |
| **Full Test Holdout** | Accuracy | `0.4272` | `0.3843` | **`0.3263`** | `-0.0580` |
| **Full Test Holdout** | Macro-F1 | `0.1995` | `0.3459` | **`0.3195`** | `-0.0264` |
| **Full Test Holdout** | Balanced Acc | `0.3333` | `0.3761` | **`0.3692`** | `-0.0069` |
| **Novel Corporate Catalysts** | Accuracy | `0.4067` | `0.3371` | **`0.3119`** | `-0.0252` |
| **Novel Corporate Catalysts** | Macro-F1 | `0.1927` | `0.3337` | **`0.3069`** | `-0.0268` |
| **Novel Corporate Catalysts** | Balanced Acc | `0.3333` | `0.3657` | **`0.3595`** | `-0.0062` |

---

## 2. Feature Definitions & Leakage Safeguards

In this experiment, text features (`headline_text`, `model_text`, `title`) were deliberately omitted to establish the standalone predictive value of structured metadata.

### Categorical Features (One-Hot Encoded on Train Only):
- `event_type`
- `event_category`
- `event_severity`
- `insider_role`
- `transaction_code`
- `is_multi_ticker`

### Numerical Features (SimpleImputer on Train Only):
- `percentage_holdings_sold`
- `transaction_value`
- `event_classification_confidence`
- `n_tokens`
- `headline_ticker_count`
- `short_interest_before`

### Excluded Attributes:
- **Identifiers:** `event_id`, `ticker`, `company`
- **Dates/Times:** `date`, `published_at`, `news_date`, `published_date`
- **Target & Future Outcomes:** `short_interest_label`, `short_interest_after`, settlement dates, dissemination dates

---

## 3. Validation Model Selection & Hyperparameter Tuning

Grid search conducted strictly on the chronological **Validation split** ($N = 44,606$). The model was selected based on Validation Macro-F1.

| Config | Max Depth | Learning Rate | N Estimators | Subsample | Colsample | Class Weight | Val Accuracy | Val Macro-F1 | Val Balanced Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 3 | 0.05 | 100 | 0.8 | 0.8 | `balanced` | 0.3559 | **0.3290** | 0.3881 |
| 2 | 4 | 0.05 | 100 | 0.8 | 0.8 | `balanced` | 0.3551 | **0.3319** | 0.3867 |
| 3 | 6 | 0.05 | 100 | 0.8 | 0.8 | `balanced` | 0.3522 | **0.3371** | 0.3808 |
| 4 | 7 | 0.05 | 150 | 0.8 | 0.8 | `balanced` | 0.3532 | **0.3447** **(Selected)** | 0.3765 |
| 5 | 8 | 0.03 | 150 | 0.8 | 0.8 | `balanced` | 0.3523 | **0.3433** | 0.3766 |
| 6 | 6 | 0.08 | 120 | 0.8 | 0.8 | `balanced` | 0.3517 | **0.3420** | 0.3758 |
| 7 | 4 | 0.1 | 100 | 0.8 | 0.8 | `balanced` | 0.3545 | **0.3353** | 0.3832 |
| 8 | 3 | 0.05 | 100 | 0.8 | 0.8 | `None` | 0.4159 | **0.3164** | 0.3730 |
| 9 | 4 | 0.05 | 100 | 0.8 | 0.8 | `None` | 0.4160 | **0.3191** | 0.3735 |
| 10 | 6 | 0.05 | 100 | 0.8 | 0.8 | `None` | 0.4139 | **0.3276** | 0.3736 |
| 11 | 7 | 0.05 | 150 | 0.8 | 0.8 | `None` | 0.4099 | **0.3392** | 0.3728 |
| 12 | 4 | 0.1 | 100 | 0.8 | 0.8 | `None` | 0.4147 | **0.3316** | 0.3751 |

**Selected Hyperparameters:** `{'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 150, 'subsample': 0.8, 'colsample_bytree': 0.8, 'weight': 'balanced'}`  
**Optimal Validation Macro-F1:** `0.3447`  

---

## 4. Detailed Out-of-Sample Test Evaluation

### 1. Full Chronological Test Holdout ($N = 66,578$)

- **Accuracy:** `0.3263`
- **Macro-F1:** `0.3195`
- **Weighted-F1:** `0.3213`
- **Balanced Accuracy:** `0.3692`

| Class Label | Precision | Recall | F1-Score | Support ($N$) |
| :--- | :---: | :---: | :---: | :---: |
| **`decrease`** | `0.3651` | `0.1752` | `0.2368` | `24,466` |
| **`increase`** | `0.4983` | `0.3177` | `0.3880` | `28,440` |
| **`no_significant_change`** | `0.2290` | `0.6149` | `0.3337` | `13,672` |

### 2. Novel Single-Ticker Corporate Catalysts ($N = 27,018$)

- **Accuracy:** `0.3119`
- **Macro-F1:** `0.3069`
- **Weighted-F1:** `0.3003`
- **Balanced Accuracy:** `0.3595`

| Class Label | Precision | Recall | F1-Score | Support ($N$) |
| :--- | :---: | :---: | :---: | :---: |
| **`decrease`** | `0.3632` | `0.2072` | `0.2638` | `10,060` |
| **`increase`** | `0.4687` | `0.2272` | `0.3061` | `10,988` |
| **`no_significant_change`** | `0.2411` | `0.6442` | `0.3509` | `5,970` |

---

## 5. Confusion Matrices

### Full Test Holdout Confusion Matrix (Raw Counts)

Rows: True Class (`decrease`, `increase`, `no_significant_change`); Columns: Predicted Class

| True Class | Pred `decrease` | Pred `increase` | Pred `no_significant_change` |
| :--- | :---: | :---: | :---: |
| **`decrease`** | `4,286` | `6,166` | `14,014` |
| **`increase`** | `5,117` | `9,034` | `14,289` |
| **`no_significant_change`** | `2,336` | `2,929` | `8,407` |

### Novel Corporate Catalysts Confusion Matrix (Raw Counts)

| True Class | Pred `decrease` | Pred `increase` | Pred `no_significant_change` |
| :--- | :---: | :---: | :---: |
| **`decrease`** | `2,084` | `1,883` | `6,093` |
| **`increase`** | `2,477` | `2,497` | `6,014` |
| **`no_significant_change`** | `1,177` | `947` | `3,846` |

---

## 6. Reproducibility & Environment Specification

- **Random Seed:** `42`
- **XGBoost Version:** `3.4.1`
- **Scikit-Learn Version:** `1.9.1`
- **Pandas Version:** `3.1.0`
- **NumPy Version:** `2.5.4`
- **Joblib Version:** `1.6.0`
- **Python Version:** `3.14.2`
- **Platform:** `macOS-27.0.1-arm64-arm-64bit-Mach-O`

> [!CAUTION]
> **Research Disclaimers:**
> 1. **Statistical Correlation vs. Causal Evidence:** Model predictions represent statistical patterns across pre-event categorical and numerical features and subsequent FINRA bi-monthly settlement changes. They do not constitute causal proof.
> 2. **No Investment Advice:** This experimental pipeline is developed strictly for empirical research and academic analysis. It does not constitute investment advice or trading recommendations.
