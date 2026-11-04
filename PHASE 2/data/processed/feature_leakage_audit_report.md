# Phase 2 Short-Interest Model Dataset & Pre-Training Feature Audit Report

**Audit Date:** 2026-11-04  
**Audit Status:** **APPROVED FOR TRAINING (Identity Leakage Resolution Verified)**  
**Target Variable:** `short_interest_label` (`increase`, `decrease`, `no_significant_change`)  
**Data Artifacts Audited:**
- `data/processed/short_interest_model_dataset.parquet` (369,535 rows, 40 columns, 28.1 MB)
- `data/processed/short_interest_model_dataset.csv` (369,535 rows, 40 columns, 188.5 MB)

---

## 1. Executive Summary & Verification Checklist

| Requirement | Audit Finding | Status |
| :--- | :--- | :---: |
| **1. Explicit Feature Allowlist** | 13 causal pre-event features defined; all identifiers, future outcomes, and audit columns excluded. | **PASSED** |
| **2. Ticker/Company Text Leakage Resolution** | `model_text` composite representation identified as leaking company/ticker tokens. Created `headline_text` (headline-only) and established it as the default text input. `model_text` preserved for comparative benchmarking. | **PASSED** |
| **3. Identifier Dummy Encoding Exclusion** | `ticker` (4,447 categories) and `company` excluded from one-hot encoding; absolute timestamps excluded from features. | **PASSED** |
| **4. Train-Only Preprocessing** | TF-IDF vocabulary, categorical encodings, and numerical scalers/imputers fitted strictly on `split == 'train'`. | **PASSED** |
| **5. Out-of-Sample Isolation** | `val` and `test` splits transformed using frozen preprocessor parameters without refitting or state mutation. | **PASSED** |
| **6. Strict Chronological Splits** | Partitioned strictly by calendar date: Train (69.91%), Val (12.07%), Test (18.02%) with 0 temporal overlap. | **PASSED** |
| **7. Headline Syndication Audit** | 16.11% of Test events share recurring headlines with Train (generic market roundups). Dual-benchmark evaluation protocol established. | **PASSED** |
| **8. Pre-Event Baseline Grounding** | `short_interest_before` dissemination date strictly precedes event date for 100% of records (`min_gap = 1 day`). | **PASSED** |
| **9. Automated Regression Tests** | 11 unit tests in `tests/test_model_feature_allowlist.py`; entire 84-test suite passing. | **PASSED** |
| **10. Model Training Restraint** | Zero models trained; pipelines ready in `models/feature_pipeline.py` and `models/train_models.py`. | **PASSED** |
| **11. Phase 1 Immutability** | `PHASE 1/` codebase and datasets untouched (read-only). | **PASSED** |

---

## 2. Ticker & Company Identity Leakage Audit & Resolution

### Inspection of `model_text`
1. **Original Formulation:** In the baseline feature pipeline, `model_text` was formed by concatenating title and description, or formatted as `"Title: <title> | Company: <company> | Ticker: <ticker>"`.
2. **Identity Leakage Hazard:** Even though `ticker` and `company` were excluded from one-hot dummy encoding columns, embedding them directly into the text input fed to `TfidfVectorizer` leaked nominal firm identities directly into the vocabulary (e.g. unigrams/bigrams such as `aapl`, `apple inc`, `msft`, `microsoft corp`). The NLP classifier could thus memorize past short-interest drift associated with specific companies, failing to generalize to unseen tickers in production.

### Resolution Protocol
1. **Introduction of `headline_text` (Default Input):**
   - Created a second, sanitized text representation containing strictly the headline/title string (`headline_text`).
   - Implemented `sanitize_to_headline_only()` to automatically strip any trailing or embedded metadata tags (`| Company: ...`, `| Ticker: ...`, `Title:`).
   - Set `headline_text` as the primary, default text feature in `MODEL_FEATURE_ALLOWLIST`.
2. **Preservation of `model_text` for Comparative Benchmarking:**
   - The composite representation (`model_text`) is preserved in `COMPARATIVE_FEATURE_ALLOWLIST`.
   - Researchers can toggle `use_headline_only=False` to benchmark the unconstrained composite baseline against the leak-free headline model.
3. **Empirical Verification:**
   - Unit test `test_default_preprocessor_prevents_identity_leakage_in_vocabulary` verifies that for events referencing Apple (`AAPL`) and Microsoft (`MSFT`), the default preprocessor's TF-IDF vocabulary contains neither `aapl`, `apple`, `msft`, nor `microsoft`, while the comparative preprocessor does.

---

## 3. Explicit Model Feature Allowlist (13 Core Features)

The model input space is strictly restricted to 13 pre-event features in `models/feature_pipeline.py`:

### Text Feature (1 column - Default: Headline Only)
- `headline_text` (Default): Clean headline/title only, free of company name or ticker tags. Transformed via `TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2)`.
- *Note:* `model_text` is preserved in `COMPARATIVE_FEATURE_ALLOWLIST` for baseline comparison only.

### Categorical Features (6 columns - Unchanged)
- `event_type`: Primary taxonomical classification (e.g. `earnings_announcement`, `product_recall`, `clinical_trial`, `insider_selling`).
- `event_category`: Macro category (e.g. `financial_reporting`, `corporate_action`, `clinical`, `legal_regulatory`).
- `event_severity`: Expert severity rating (`low`, `medium`, `high`, `critical`).
- `insider_role`: Form 4 insider status (`CEO`, `Director`, `Officer`, `10% Owner`, `None`).
- `transaction_code`: SEC Form 4 transaction type (`P`, `S`, `M`, `A`, `None`).
- `is_multi_ticker`: Boolean indicator for whether the news event references multiple tickers (syndication flag).

*Categorical Encoding:* `OneHotEncoder(handle_unknown="ignore", sparse_output=True)`, fit exclusively on Train.

### Numerical Features (6 columns - Unchanged)
- `percentage_holdings_sold`: Percentage of holdings disposed in insider filing.
- `transaction_value`: Dollar value of insider transaction.
- `event_classification_confidence`: Model confidence score for event taxonomy.
- `n_tokens`: Headline token length.
- `headline_ticker_count`: Number of extracted stock tickers in the event text.
- `short_interest_before`: Historical pre-event short-interest share volume from FINRA.

*Numerical Preprocessing:* `SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)` followed by `StandardScaler(with_mean=False)`.

---

## 4. Comprehensive Exclusion Audit (27 Non-Feature Columns)

| Column | Category | Exclusion Rationale |
| :--- | :--- | :--- |
| `model_text` | Text (Composite) | Preserved for comparative baseline benchmarking, but excluded from default model input to eliminate ticker/company identity leakage. |
| `event_id` | Identifier | Primary row key. Excluded to prevent hash memorization and row order leakage. |
| `ticker` | Identifier | High-cardinality nominal symbol (4,447 unique values). Excluded from dummy encoding and default text input to avoid sparsity explosion and failure on unseen tickers. Market volume is captured continuously via `short_interest_before`. |
| `company` | Identifier | Free-text entity name. Excluded from dummy encoding and default text input. |
| `date` | Timestamp | Calendar date. Excluded to prevent linear or tree models from splitting on arbitrary date boundaries. |
| `published_at` | Timestamp | UTC timestamp. Excluded to prevent temporal boundary overfitting. Splitting is handled via chronological dataset slicing. |
| `published_date` | Timestamp | Redundant calendar date string. Excluded. |
| `news_date` | Timestamp | Integer date representation. Excluded. |
| `title` | Text (Raw) | Standardized into primary default text feature `headline_text`. |
| `severity` | Duplicate | Alias of `event_severity`. |
| `confidence` | Duplicate | Alias of `event_classification_confidence`. |
| `classification_source` | Audit Metadata | Internal provenance flag (`heuristic_rule`, `llm_gemini`). |
| `context_source` | Audit Metadata | Internal provenance flag. |
| `classification_explanation` | Qualitative Audit | Free-form diagnostic explanation for human review. |
| `context_summary` | Qualitative Audit | Free-form summary for human review. |
| `severity_rationale` | Qualitative Audit | Free-form diagnostic rationale for human review. |
| `all_detected_event_types` | Diagnostic List | Secondary heuristic match list. |
| `risk_factors` | Diagnostic List | Qualitative risk factor list. |
| `supporting_evidence` | Diagnostic List | Qualitative supporting excerpt list. |
| `insider_nuance` | Diagnostic Note | Qualitative Form 4 nuance note. |
| `context_flags` | Diagnostic String | Qualitative heuristic flag string. |
| `insider_is_10b5_1` / `is_10b5_1` | Form 4 Flag | Unpopulated for general financial news headlines; omitted to prevent all-null sparsity. |
| `short_interest_before_date` | Timestamp | Settlement date of prior short interest. Excluded. |
| `short_interest_before_dissemination` | Timestamp | Dissemination date of prior short interest. Excluded. |
| `short_interest_before_dissemination_source` | Provenance | Source authority string (`finra_rule_4560_schedule`). |
| `split` | Dataset Partition | Slicing metadata (`train`, `val`, `test`). Excluded from model inputs. |
| `short_interest_label` | Target Ground Truth | Target label. Excluded from input feature matrix to prevent direct target leakage. |

### Prohibited Future Outcome Columns (Strict Zero-Tolerance Leakage Check)
All verified absent:
- `short_interest_after`, `short_interest_after_date`, `short_interest_after_dissemination`, `short_interest_after_dissemination_source`
- `short_interest_gap_days`, `short_interest_change_pct`
- `phase1_label_short_interest_legacy`, `phase1_label_regsho_volume_legacy`
- `stock_price_label`, `stock_price_change_pct`

---

## 5. Chronological Splits & Headline Repetition Dual-Evaluation

### Chronological Partitions (Unchanged)
| Split | Event Count | Share (%) | Calendar Date Range | Unique Tickers | `increase` (%) | `decrease` (%) | `no_significant_change` (%) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **TRAIN** | `258,351` | 69.91% | `2018-01-11` to `2019-11-30` | 4,326 | 39.15% | 35.40% | 25.45% |
| **VAL** | `44,606` | 12.07% | `2019-12-01` to `2020-02-29` | 3,272 | 37.91% | 36.49% | 25.60% |
| **TEST** | `66,578` | 18.02% | `2020-03-01` to `2020-06-11` | 3,379 | 42.72% | 36.75% | 20.54% |

### Dual-Evaluation Protocol
1. **Primary Benchmark (Full Chronological Holdout):**
   - **Sample Size:** `66,578` events.
   - Evaluates performance on raw streaming market news, including multi-ticker syndicated market wraps.
2. **Secondary Benchmark (Deduplicated Novel Corporate Catalysts):**
   - **Filter Criteria:** `is_multi_ticker == False` and `title not in train_titles`.
   - **Sample Size:** `27,018` events (40.58% of test split; 99.65% of all single-ticker test events).
   - Strictly tests model ability to predict short-interest direction on genuine, novel corporate catalysts (earnings releases, FDA approvals, executive departures) free of repeated headline token memorization.
