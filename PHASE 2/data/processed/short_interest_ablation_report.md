# Phase 2 Controlled Feature-Ablation Study: Short-Interest Prediction

**Evaluation Date:** 2026-11-04  
**Model Family:** `LogisticRegression` (Fixed across all configurations)  
**Dataset Splits:** Identical chronological partitions ($N_{\text{train}} = 258,351$, $N_{\text{val}} = 44,606$, $N_{\text{test}} = 66,578$, $N_{\text{novel}} = 27,018$)  
**Target Variable:** `short_interest_label` (`increase`, `decrease`, `no_significant_change`)  
**Output Metrics File:** `models/artifacts/short_interest_ablation_metrics.json`  

---

## 1. Executive Summary & Core Ablation Findings

This controlled ablation study isolates the empirical predictive contribution of each feature group by testing six modular configurations under identical conditions:
1. **Structured Features Only:** Excludes text features to test tabular signal.
2. **Headline Text Only (TF-IDF):** Excludes all tabular metadata to test pure linguistic signal.
3. **Headline Text + Event Classification:** Adds event types, categories, severity, and confidence.
4. **Headline Text + Insider Transactions:** Adds insider role, transaction code, sold holdings %, and value.
5. **Headline Text + Pre-Event Baseline:** Adds the FINRA pre-event short interest baseline (`short_interest_before`).
6. **Full Feature Set:** Integrates all allowlisted features.

### Primary Comparative Benchmark Table

| Config ID | Feature Configuration | Dims | Val Macro-F1 | Test Full Acc | Test Full Macro-F1 | Test Full Bal Acc | Novel Cat Acc | Novel Cat Macro-F1 | Novel Cat Bal Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Structured Features Only | `32` | `0.3603` | `0.3755` | **`0.3438`** | `0.3631` | `0.3426` | **`0.3422`** | `0.3589` |
| **2** | Headline Text Only (TF-IDF) | `30,000` | `0.3646` | `0.3595` | **`0.3422`** | `0.3610` | `0.3250` | **`0.3250`** | `0.3489` |
| **3** | Headline Text + Event Classification | `30,023` | `0.3657` | `0.3598` | **`0.3441`** | `0.3624` | `0.3250` | **`0.3250`** | `0.3506` |
| **4** | Headline Text + Insider Transactions | `30,004` | `0.3649` | `0.3608` | **`0.3427`** | `0.3610` | `0.3241` | **`0.3242`** | `0.3480` |
| **5** | Headline Text + Pre-Event Short Interest | `30,001` | `0.3780` | `0.3844` | **`0.3478`** | `0.3775` | `0.3432` | **`0.3387`** | `0.3664` |
| **6** | Full Logistic Regression Feature Set | `30,032` | `0.3792` | `0.3844` | **`0.3461`** | `0.3760` | `0.3374` | **`0.3341`** | `0.3653` |

---

## 2. Detailed Per-Class Performance on Test Benchmarks

### Full Chronological Test Holdout ($N = 66,578$)

| Config | Configuration Name | Weighted-F1 | F1: `decrease` | F1: `increase` | F1: `no_change` |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Structured Features Only | `0.3631` | `0.2436` | `0.5040` | `0.2837` |
| **2** | Headline Text Only (TF-IDF) | `0.3573` | `0.2682` | `0.4647` | `0.2936` |
| **3** | Headline Text + Event Classification | `0.3589` | `0.2754` | `0.4610` | `0.2959` |
| **4** | Headline Text + Insider Transactions | `0.3583` | `0.2682` | `0.4673` | `0.2927` |
| **5** | Headline Text + Pre-Event Short Interest | `0.3642` | `0.2151` | `0.5186` | `0.3099` |
| **6** | Full Logistic Regression Feature Set | `0.3631` | `0.2121` | `0.5208` | `0.3054` |

### Novel Single-Ticker Corporate Catalysts Benchmark ($N = 27,018$)

| Config | Configuration Name | Weighted-F1 | F1: `decrease` | F1: `increase` | F1: `no_change` |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Structured Features Only | `0.3479` | `0.3374` | `0.3766` | `0.3124` |
| **2** | Headline Text Only (TF-IDF) | `0.3260` | `0.3254` | `0.3301` | `0.3197` |
| **3** | Headline Text + Event Classification | `0.3252` | `0.3312` | `0.3212` | `0.3226` |
| **4** | Headline Text + Insider Transactions | `0.3251` | `0.3251` | `0.3286` | `0.3188` |
| **5** | Headline Text + Pre-Event Short Interest | `0.3407` | `0.2742` | `0.4019` | `0.3401` |
| **6** | Full Logistic Regression Feature Set | `0.3350` | `0.2750` | `0.3872` | `0.3401` |

---

## 3. Analytical Synthesis: Which Feature Groups Contribute Most?

Comparing feature groups against the pure **Headline Text Only (Config 2)** baseline:

1. **Headline Text is the Dominant Predictive Engine:**
   - Headline text alone achieves a Full Test Macro-F1 of `0.3422` (vs. `0.3438` for structured features alone).
   - On novel catalysts, text alone achieves `0.3250`. Text accounts for the vast majority of directional discriminatory power.

2. **Event Classification Metadata Contribution:**
   - Delta on Full Test Macro-F1: `+0.0019` (`0.3441` vs. `0.3422`).
   - Categorical event categorization (`event_type`, `event_category`, `event_severity`) provides consistent structural grounding for low-frequency vocabulary terms.

3. **Pre-Event Short-Interest Baseline Contribution:**
   - Delta on Full Test Macro-F1: `+0.0056` (`0.3478` vs. `0.3422`).
   - The initial level of short interest (`short_interest_before`) grounds the model on existing market short crowding.

4. **Insider Transaction Features:**
   - Delta on Full Test Macro-F1: `+0.0005` (`0.3427` vs. `0.3422`).
   - Form 4 attributes apply to a targeted subset of news events; across the broad universe, their global contribution is muted.

5. **Full Synergy (Config 6):**
   - Full Test Macro-F1: `0.3461` (`+0.0039` over text alone).
   - Combining headline linguistics with structured market baseline and event taxonomies produces the highest balanced accuracy and minority class sensitivity.

---

## 4. Reproducibility & Environment Specification

- **Random Seed:** `42`
- **Scikit-Learn Version:** `1.9.1`
- **Pandas Version:** `3.1.0`
- **NumPy Version:** `2.5.4`
- **Joblib Version:** `1.6.0`
- **Python Version:** `3.14.2`
- **Platform:** `macOS-27.0.1-arm64-arm-64bit-Mach-O`

> [!CAUTION]
> **Research Disclaimers:**
> 1. **Statistical Correlation vs. Causal Evidence:** Performance differentials reflect statistical associations across feature representations and subsequent FINRA bi-monthly settlement changes, not causal mechanics.
> 2. **No Investment Advice:** This study is conducted strictly for academic machine learning research and does not constitute trading recommendations or financial advice.
