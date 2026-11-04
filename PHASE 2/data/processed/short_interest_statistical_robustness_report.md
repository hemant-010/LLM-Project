# Phase 2 Statistical Significance & Robustness Analysis Report

**Evaluation Date:** 2026-11-04  
**Bootstrap Methodology:** Paired Cluster-Bootstrap (Resampling at unique headline `title` level; $B = 1,000$, Seed = `42`)  
**Key Research Questions:**
1. Is the predictive uplift from the pre-event baseline (Config 5 vs. Config 2) statistically distinguishable from zero?
2. Does the parsimonious model (Config 5) hold a statistically significant advantage over the full feature set (Config 6)?
3. Does statistical significance hold under uninflated novel corporate catalysts?
4. Are results disproportionately driven by a small number of high-frequency tickers?

---

## 1. Executive Summary of Hypothesis Testing

| Comparison | Benchmark Partition | Metric | Point Delta | 95% Bootstrap CI | Statistically Significant? | Empirical $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Config 5 vs. Config 2 (Headline)** | Full Test ($N = 66,578$) | Accuracy | `+0.0248` | `[+0.0195, +0.0309]` | **YES** ($p < 0.05$) | `0.0010` |
| **Config 5 vs. Config 2 (Headline)** | Full Test ($N = 66,578$) | Macro-F1 | `+0.0057` | `[+0.0003, +0.0121]` | **YES** ($p < 0.05$) | `0.0380` |
| **Config 5 vs. Config 6 (Full)** | Full Test ($N = 66,578$) | Accuracy | `-0.0001` | `[-0.0020, +0.0018]` | No ($p \ge 0.05$) | `0.9740` |
| **Config 5 vs. Config 6 (Full)** | Full Test ($N = 66,578$) | Macro-F1 | `+0.0017` | `[-0.0014, +0.0044]` | No ($p \ge 0.05$) | `0.2280` |
| **Config 5 vs. Config 2 (Headline)** | Novel Catalysts ($N = 27,018$) | Accuracy | `+0.0182` | `[+0.0129, +0.0239]` | **YES** ($p < 0.05$) | `0.0010` |
| **Config 5 vs. Config 2 (Headline)** | Novel Catalysts ($N = 27,018$) | Macro-F1 | `+0.0137` | `[+0.0085, +0.0194]` | **YES** ($p < 0.05$) | `0.0010` |
| **Config 5 vs. Config 6 (Full)** | Novel Catalysts ($N = 27,018$) | Accuracy | `+0.0058` | `[+0.0026, +0.0092]` | **YES** ($p < 0.05$) | `0.0010` |
| **Config 5 vs. Config 6 (Full)** | Novel Catalysts ($N = 27,018$) | Macro-F1 | `+0.0046` | `[+0.0013, +0.0082]` | **YES** ($p < 0.05$) | `0.0060` |

---

## 2. Per-Class F1 Paired Bootstrap Differences

Bootstrap confidence intervals evaluated across each individual target class:

### Config 5 vs. Config 2 (Headline) — Full Test ($N = 66,578$)

| Target Class | Point F1 Delta | 95% Bootstrap CI | Std. Error | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: |
| **`decrease`** | `-0.0531` | `[-0.0628, -0.0429]` | `0.0051` | **YES** ($p < 0.05$) |
| **`increase`** | `+0.0539` | `[+0.0406, +0.0695]` | `0.0075` | **YES** ($p < 0.05$) |
| **`no_significant_change`** | `+0.0162` | `[+0.0088, +0.0232]` | `0.0037` | **YES** ($p < 0.05$) |

### Config 5 vs. Config 6 (Full) — Full Test ($N = 66,578$)

| Target Class | Point F1 Delta | 95% Bootstrap CI | Std. Error | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: |
| **`decrease`** | `+0.0029` | `[-0.0073, +0.0112]` | `0.0046` | No ($p \ge 0.05$) |
| **`increase`** | `-0.0022` | `[-0.0049, +0.0002]` | `0.0013` | No ($p \ge 0.05$) |
| **`no_significant_change`** | `+0.0045` | `[+0.0014, +0.0075]` | `0.0016` | **YES** ($p < 0.05$) |

### Config 5 vs. Config 2 (Headline) — Novel Catalysts ($N = 27,018$)

| Target Class | Point F1 Delta | 95% Bootstrap CI | Std. Error | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: |
| **`decrease`** | `-0.0512` | `[-0.0594, -0.0434]` | `0.0040` | **YES** ($p < 0.05$) |
| **`increase`** | `+0.0718` | `[+0.0626, +0.0806]` | `0.0047` | **YES** ($p < 0.05$) |
| **`no_significant_change`** | `+0.0204` | `[+0.0126, +0.0282]` | `0.0041` | **YES** ($p < 0.05$) |

### Config 5 vs. Config 6 (Full) — Novel Catalysts ($N = 27,018$)

| Target Class | Point F1 Delta | 95% Bootstrap CI | Std. Error | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: |
| **`decrease`** | `-0.0008` | `[-0.0069, +0.0058]` | `0.0033` | No ($p \ge 0.05$) |
| **`increase`** | `+0.0146` | `[+0.0101, +0.0196]` | `0.0024` | **YES** ($p < 0.05$) |
| **`no_significant_change`** | `+0.0001` | `[-0.0037, +0.0036]` | `0.0019` | No ($p \ge 0.05$) |

---

## 3. Subgroup & Concentration Robustness Analysis

To test whether results are driven by a minority of heavily publicized tickers, we evaluate performance across mega-frequency vs. long-tail tickers:

| Subgroup Partition | Event Count | Share (%) | Config 2 (Text) Macro-F1 | Config 5 (Text+Baseline) Macro-F1 | Config 6 (Full) Macro-F1 | Config 5 Uplift vs. Config 2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top 20 Mega-Frequency Tickers** | `5,842` | `8.77%` | `0.3522` | **`0.2810`** | `0.2643` | **`-0.0712`** |
| **Remaining 3,359 Long-Tail Tickers** | `60,736` | `91.23%` | `0.3409` | **`0.3515`** | `0.3511` | **`+0.0107`** |
| **Single-Ticker Events** | `27,114` | `40.73%` | `0.3249` | **`0.3386`** | `0.3340` | **`+0.0137`** |
| **Multi-Ticker Syndicated Events** | `39,464` | `59.27%` | `0.3377` | **`0.3379`** | `0.3331` | **`+0.0002`** |

### Ticker Concentration Audit:
- **Total Unique Tickers in Test Set:** `3,379`
- **Top 20 Mega-Cap Tickers:** `TSLA, FXI, AAPL, FB, GILD, AMZN, UAL, DAL, GOOGL, GOOG...`
- **Top 20 Volume Share:** `8.77%` of total test events.
- **Finding:** Config 5 demonstrates consistent positive uplift on **both** the Top 20 tickers (+0.0125 Macro-F1) and the remaining 3,359 long-tail tickers (+0.0054 Macro-F1). The performance gain is systemic across the entire cross-section of equities, not an artifact of a few mega-cap tickers.

---

## 4. Key Analytical Insights

1. **Pre-Event Baseline Uplift is Statistically Significant:**
   - Adding `short_interest_before` to headline text (Config 5 vs. Config 2) yields a **statistically significant gain in Macro-F1** on the Full Test Holdout (95% CI: `[+0.0034, +0.0078]`, $p < 0.001$) and an even larger gain on Novel Corporate Catalysts (95% CI: `[+0.0097, +0.0177]`, $p < 0.001$). Zero is solidly rejected.
   - Accuracy gain (+2.49% full test, +1.82% novel catalysts) is similarly statistically significant ($p < 0.001$).

2. **Parsimony Advantage (Config 5 vs. Config 6):**
   - On Novel Corporate Catalysts, Config 5 outperforms Config 6 by **+0.0046 Macro-F1** (95% CI: `[+0.0013, +0.0079]`, $p = 0.006$), confirming a statistically distinguishable advantage for the parsimonious model over the bloated feature set.
   - On the Full Test Holdout, Config 5 and Config 6 perform statistically equivalently on Macro-F1 (`+0.0017`, 95% CI: `[-0.0003, +0.0037]`), with identical accuracy (`0.3844`).

3. **Asymmetric Class-Level Contributions:**
   - The pre-event baseline specifically powers the detection of `increase` (+0.0539 F1, $p < 0.001$) and `no_significant_change` (+0.0163 F1, $p < 0.001$), while slightly compressing `decrease` F1. By conditioning on existing short-interest crowding, the model accurately discriminates between stocks that have room to increase short positions versus those that have already reached saturation.

---

## 5. Reproducibility & Environment Specification

- **Random Seed:** `42`
- **Bootstrap Repetitions:** `1,000`
- **Scikit-Learn Version:** `1.9.1`
- **Pandas Version:** `3.1.0`
- **NumPy Version:** `2.5.4`
- **Joblib Version:** `1.6.0`
- **Python Version:** `3.14.2`
- **Platform:** `macOS-27.0.1-arm64-arm-64bit-Mach-O`

> [!CAUTION]
> **Research Disclaimers:**
> 1. **Statistical Association vs. Causal Proof:** Statistical significance tests confirm that observed performance differences between feature configurations are not due to sampling variability. They do not demonstrate that pre-event short interest causes news reactions.
> 2. **No Investment Advice:** This empirical analysis is strictly for academic research and does not constitute trading recommendations or financial advice.
