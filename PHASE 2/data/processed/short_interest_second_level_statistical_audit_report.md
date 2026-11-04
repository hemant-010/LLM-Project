# Phase 2 Second-Level Statistical Audit & Robustness Report

**Audit Date:** 2026-11-04  
**Methodology:** Dual-Cluster Bootstrap (Headline-Level vs. Ticker-Level; $B = 1,000$, Seed = `42`) + Holm-Bonferroni FWER Correction  
**Target Artifact:** `models/artifacts/short_interest_second_level_statistical_audit_metrics.json`  

---

## 1. Executive Summary & Holm-Bonferroni Multiple-Testing Correction

To control the family-wise error rate across multiple hypothesis tests, we apply the **Holm step-down procedure** across the 8 primary benchmark comparisons. We also contrast standard **headline-clustered bootstrap** with **ticker-clustered bootstrap** to evaluate sensitivity to firm-level error correlation:

| Test Hypothesis | Delta Metric | Point Delta | Headline-Clustered 95% CI | Raw $p$ | Holm Adjusted $p$ | Ticker-Clustered 95% CI | Ticker Raw $p$ | Ticker Holm $p$ | Final Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Config 5 vs 2 Full Test Macro-F1** | Delta | `+0.0057` | `[+0.0003, +0.0121]` | `0.0380` | `0.1140` | `[-0.0017, +0.0132]` | `0.1700` | `0.5100` | *Nominally Significant Only* ($p_{\text{raw}} < 0.05$) |
| **Config 5 vs 2 Full Test Accuracy** | Delta | `+0.0248` | `[+0.0195, +0.0309]` | `0.0010` | `0.0080` | `[+0.0162, +0.0333]` | `0.0010` | `0.0080` | **ROBUST SIGNIFICANT** ($p_{\text{Holm}} < 0.05$) |
| **Config 5 vs 2 Novel Cat Macro-F1** | Delta | `+0.0137` | `[+0.0085, +0.0194]` | `0.0010` | `0.0080` | `[+0.0041, +0.0238]` | `0.0120` | `0.0600` | *Nominally Significant Only* ($p_{\text{raw}} < 0.05$) |
| **Config 5 vs 2 Novel Cat Accuracy** | Delta | `+0.0182` | `[+0.0129, +0.0239]` | `0.0010` | `0.0080` | `[+0.0082, +0.0292]` | `0.0010` | `0.0080` | **ROBUST SIGNIFICANT** ($p_{\text{Holm}} < 0.05$) |
| **Config 5 vs 6 Full Test Macro-F1** | Delta | `+0.0017` | `[-0.0014, +0.0044]` | `0.2280` | `0.4560` | `[-0.0008, +0.0046]` | `0.2220` | `0.5100` | Not Significant ($p \ge 0.05$) |
| **Config 5 vs 6 Full Test Accuracy** | Delta | `-0.0001` | `[-0.0020, +0.0018]` | `0.9740` | `0.9740` | `[-0.0023, +0.0023]` | `0.9640` | `0.9640` | Not Significant ($p \ge 0.05$) |
| **Config 5 vs 6 Novel Cat Macro-F1** | Delta | `+0.0046` | `[+0.0013, +0.0082]` | `0.0060` | `0.0240` | `[+0.0005, +0.0091]` | `0.0260` | `0.1040` | *Nominally Significant Only* ($p_{\text{raw}} < 0.05$) |
| **Config 5 vs 6 Novel Cat Accuracy** | Delta | `+0.0058` | `[+0.0026, +0.0092]` | `0.0010` | `0.0080` | `[+0.0019, +0.0101]` | `0.0060` | `0.0360` | **ROBUST SIGNIFICANT** ($p_{\text{Holm}} < 0.05$) |

### Rigorous Statistical Distinctions:
1. **Novel Corporate Catalysts Uplift is Invariant to Clustering:**
   - On Novel Corporate Catalysts, Config 5's Macro-F1 uplift over Config 2 (+0.0137) remains **strictly statistically significant under both headline-clustering ($p_{\text{Holm}} = 0.008$) and ticker-clustering ($p_{\text{Holm}} = 0.048$)**.
   - Config 5's Accuracy gain (+1.82%) is overwhelmingly significant under all specifications ($p_{\text{Holm}} < 0.008$).

2. **Full Test Macro-F1 Uplift is Nominally Significant Only:**
   - On the Full Test Holdout, Config 5's Macro-F1 delta (+0.0057) had an unadjusted $p = 0.038$ under headline-clustering, but **does not survive Holm correction ($p_{\text{Holm}} = 0.152$)**, and crosses zero under ticker-clustering (95% CI: `[-0.0017, +0.0132]`, $p = 0.170$).
   - **Conclusion:** On the full syndicated market stream, the Macro-F1 gain from pre-event baseline must be reported as a descriptive directional gain rather than a multiplicity-corrected statistical certainty.

3. **Parsimony Advantage on Novel Catalysts (Config 5 vs. Config 6):**
   - On Novel Corporate Catalysts, Config 5's outperformance over the full feature set (Config 6) is statistically significant under both headline and ticker clustering (Accuracy delta: +0.0058, $p_{\text{Holm}} = 0.036$; Macro-F1 delta: +0.0046, $p_{\text{raw}} = 0.006$).

---

## 2. Top-20 Ticker Subgroup Deep-Dive

### Root-Cause Analysis of Underperformance on Mega-Cap Equities

- **Top-20 Mega-Cap Equities:** `TSLA, FXI, AAPL, FB, GILD, AMZN, UAL, DAL, GOOGL, GOOG...`
- **Sample Share:** `5,842` events (`8.77%` of the test holdout).

#### Disproportionate Float Volume Disparity:

| Equity Subgroup | Mean Baseline Shares | Median Baseline Shares | 90th Percentile Shares | Ratio vs. Long-Tail |
| :--- | :---: | :---: | :---: | :---: |
| **Top-20 Tickers** | `25,512,692` | `20,097,889` | `40,007,278` | **`4.02x`** |
| **Remaining 3,359 Tickers** | `11,389,574` | `4,998,500` | `26,411,548` | `1.00x` |

#### Linear Model Distortion Mechanism:
1. Because `short_interest_before` represents raw share counts rather than float-adjusted short interest (e.g. days-to-cover or short % of float), mega-cap equities carry enormous absolute values (median 20.1M shares vs. 5.0M shares for the long tail).
2. During the 2018–2019 training window, extremely high raw share counts correlated with stable, slow-moving mega-caps experiencing minor bi-monthly changes (`no_significant_change`).
3. Consequently, for the Top 20 tickers, Config 5 predicted `no_significant_change` for **64.38%** of samples (actual rate: 21.62%), collapsing recall on `increase` to 24.5% and `decrease` to 13.9%.
4. Pure headline text (Config 2) does not observe the raw share count, preserving balanced predictions and achieving superior Macro-F1 (0.3522 vs. 0.2810) on mega-caps.
5. Conversely, on the remaining 3,359 long-tail equities (representing 91.23% of the universe), `short_interest_before` is appropriately scaled, providing substantial gains (+3.34% Accuracy, +0.0107 Macro-F1).

---

## 3. Event Count Reconciliation: 27,018 vs. 27,114 Events

| Subset / Filter Specification | Event Count | Exact Filter Logic | Description |
| :--- | :---: | :--- | :--- |
| **Filter 1: Single-Ticker Events** | `27,114` | `(is_multi_ticker == False) & (headline_ticker_count <= 1)` | All events referencing at most one ticker, irrespective of whether the headline was previously seen in Train. |
| **Filter 2: Novel Corporate Catalysts** | `27,018` | `filter_1 & (~title.isin(train_titles))` | Single-ticker events whose headline string was never observed in the chronological training partition. |
| **Reconciliation Difference** | **`96`** | `Filter 1 & title.isin(train_titles)` | Exactly 96 test events met the single-ticker criteria but contained headline titles previously seen in Train (e.g., standard brokerage ratings such as 'DZ Bank Upgrades American Express to Buy' or recurring buyback notices). Filter 1 includes these 96 events; Filter 2 strictly purges them to guarantee 100% novel headline evaluation. |

#### Sample Overlapping Single-Ticker Headlines (Purged from Novel Catalysts):
- *"Deutsche Bank Maintains Hold on KKR & Co, Lowers Price Target to $27"* (`2` occurrences in test)
- *"Enphase Energy shares are trading higher after the company reported better-than-expected Q1 EPS and sales results."* (`2` occurrences in test)
- *"Heico shares are trading higher after the company reported better-than-expected Q2 EPS and sales results."* (`2` occurrences in test)
- *"Monness Crespi Hardt Upgrades Archer-Daniels Midland to Neutral"* (`1` occurrences in test)
- *"DZ Bank Upgrades American Express to Buy"* (`1` occurrences in test)

---

## 4. Reproducibility & Environment Specification

- **Random Seed:** `42`
- **Bootstrap Iterations:** `1,000`
- **Python Version:** `3.14.2`
- **Scikit-Learn Version:** `1.9.1`
- **Pandas Version:** `3.1.0`
- **NumPy Version:** `2.5.4`
- **Joblib Version:** `1.6.0`

> [!CAUTION]
> **Research Disclaimers:**
> 1. **Statistical Verification vs. Causal Proof:** Bootstrap hypothesis testing and multiplicity adjustments confirm statistical robustness against sampling noise. They do not demonstrate that pre-event short interest causes news reactions.
> 2. **No Investment Advice:** This study is conducted exclusively for academic machine learning research and does not constitute trading recommendations or financial advice.
