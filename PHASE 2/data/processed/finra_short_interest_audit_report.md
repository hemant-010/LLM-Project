# Data-Quality Audit Report: FINRA Consolidated Short Interest (2017–2020)

**Dataset Inspected:** `data/processed/finra_short_interest_2017_2020.parquet`  
**Companion File:** `data/processed/finra_short_interest_2017_2020.csv`  
**Target Reference:** `data/processed/historical_news_2017_2020.parquet`  
**Audit Date:** November 4, 2026  
**Status:** **PASSED ALL DATA INTEGRITY CHECKS**

---

### Executive Summary

| Audit Dimension | Result | Status | Notes |
| :--- | :--- | :---: | :--- |
| **Total Observations** | `256,219` | PASS | Filtered strictly for the target historical equity news universe |
| **Duplicates (`ticker + date`)** | `0` (0.00%) | PASS | Every observation is unique |
| **Missing / Invalid Values** | `0` (0.00%) | PASS | 0 null, NaN, or non-numeric values in core fields |
| **Negative / Zero Values** | `0` (0.00%) | PASS | 100% of short interest values are strictly positive (`> 0`) |
| **Unique Settlement Dates** | `62` cycles | PASS | Continuous semi-monthly cycles from `2017-12-29` to `2020-07-15` |
| **Ticker Universe Coverage** | `4,497 / 4,542` (`99.01%`) | PASS | 4,497 tickers with verified short-interest reporting |
| **Missing News Tickers** | `45` (`0.99%`) | PASS | Account for only 488 news events (0.13%); pre-2018 acquisitions/delistings |
| **Unexpected Ticker Symbols** | `0` | PASS | All symbols conform to uppercase alphanumeric standard (`^[A-Z0-9]+$`) |
| **Dissemination Fidelity** | `100%` grounded | PASS | 256,219 records sourced from `finra_rule_4560_schedule` (0 fabricated times) |
| **Look-Ahead Lag Consistency**| `9–12` calendar days | PASS | Exactly 7 market business days after settlement; 0 instances of `dissemination <= settlement` |

---

### 1. Duplicates & Record Uniqueness

* **Exact Duplicate Rows:** `0`
* **Duplicates on `[ticker, settlement_date]`:** `0`
* **Finding:** Every single observation is uniquely indexed by ticker symbol and settlement date.

---

### 2. Missing & Invalid Short-Interest Values

* **Null Count:** `0` (0.00%)
* **NaN Count:** `0` (0.00%)
* **Non-Numeric Representations:** `0` (0.00%)
* **Data Types:** `short_interest` is stored as standard `float64` (`int64` compatible).
* **Finding:** No missing or malformed values exist in the observation corpus.

---

### 3. Settlement Date Coverage & Cycle Count

* **Unique Settlement Dates:** `62` cycles
* **Earliest Settlement Date:** `2017-12-29`
* **Latest Settlement Date:** `2020-07-15`
* **Temporal Continuity:** 
  * 2017: 1 cycle (`2017-12-29`)
  * 2018: 24 cycles (100% complete)
  * 2019: 24 cycles (100% complete)
  * 2020: 13 cycles (through `2020-07-15`)
* **Finding:** Exactly covers all semi-monthly regulatory reporting periods corresponding to the `2017-12-29` to `2020-06-11` news corpus, including post-June outcome observations.

---

### 4. Observations per Settlement Cycle

| Metric | Value |
| :--- | :--- |
| **Minimum Count** | `3,804` securities (`2020-07-15`) |
| **Maximum Count** | `4,455` securities (`2017-12-29`) |
| **Mean Count** | `4,132.6` securities / cycle |
| **Median Count** | `4,124.0` securities / cycle |
| **Standard Deviation** | `196.95` securities |

* **Finding:** Observation density across cycles is highly stable, reflecting expected natural corporate lifecycle changes (mergers, acquisitions, bankruptcies, and IPO additions) over the 2.5-year span.

---

### 5. Ticker Observation Depth & Missing Cycles

* **Total Tickers:** `4,497`
* **Mean Cycles / Ticker:** `56.98` (out of 62 cycles)
* **Median Cycles / Ticker:** `62.0`
* **Coverage Breakdown:**
  * **All 62 cycles present:** `3,688` tickers (**82.01%**)
  * **>= 60 cycles present:** `3,757` tickers (**83.54%**)
  * **>= 50 cycles present:** `3,888` tickers (**86.46%**)
  * **< 10 cycles present:** `87` tickers (**1.93%**)
* **Finding:** Over 82% of tickers have full, uninterrupted bi-weekly observation histories for the entire 2.5-year window. Tickers with fewer cycles reflect historical corporate actions (e.g. IPOs debuting in late 2019 or early 2018 acquisitions).

---

### 6. Negative and Zero Values

* **Negative Values (`< 0`):** `0`
* **Zero Values (`== 0`):** `0`
* **Positive Values (`> 0`):** `256,219` (100.0%)
* **Range:** Min `1.0` share to Max `371,418,295.0` shares.
* **Finding:** Zero invalid non-positive short interest values.

---

### 7. Ticker Symbol Formatting

* **Non-alphanumeric Symbols:** `0`
* **Symbols with length > 5:** `0`
* **Symbols with special characters (`.`, `-`, `/`):** `0`
* **Finding:** Tickers strictly adhere to standard exchange symbol conventions.

---

### 8. Dissemination-Date Provenance & Consistency

* **Source Attribution:** 100% attributed to `finra_rule_4560_schedule`.
* **Dissemination Lag:**
  * Minimum calendar day lag: `9` days
  * Maximum calendar day lag: `12` days
  * Mean calendar day lag: `10.57` days
* **Calendar Rule Compliance:** Exactly matches FINRA's 7-market-business-day dissemination schedule factoring in weekend and market holiday shifts (Martin Luther King Jr. Day, Presidents' Day, Good Friday, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas).
* **Look-Ahead Verification:**
  * Number of records where `dissemination_date <= settlement_date`: **`0`**
  * Number of records with date-only semantics (`YYYY-MM-DD`): **`256,219` (100%)**
  * Zero fabricated release times.

---

### 9. Comparison: Collected Tickers vs. Historical News Universe

| Category | Count | Share (%) | Notes |
| :--- | :---: | :---: | :--- |
| **Total News Tickers** | `4,542` | 100.00% | Derived from `historical_news_2017_2020.parquet` |
| **Short Interest Tickers** | `4,497` | **99.01%** | Tickers with confirmed official short-interest reporting |
| **Missing Tickers** | `45` | **0.99%** | Pre-2018 acquired/delisted companies (e.g. `WFM`, `BBRY`, `SPLS`, `GMCR`, `HOT`, `KITE`, `STJ`, `BHI`) |
| **News Events in Missing Tickers** | `488` | **0.13%** | 99.87% of all news events in the dataset have active short-interest tracking |
| **Extra Tickers in SI** | `0` | 0.00% | Precisely aligned with target universe |

