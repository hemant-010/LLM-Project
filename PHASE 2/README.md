# Phase 2: News-driven Investor Intelligence System

This is a modular Phase 2 workspace. It does not modify Phase 1.

## Pipeline

1. Ingest financial-news rows from CSV (and structured insider transactions from SEC Form 4).
2. Classify the event category.
3. Extract context and estimate event severity.
4. Build a feature row for each event.
5. Run two independent models:
   - short-interest direction
   - stock-price direction
6. Generate an evidence-based investor alert.

## Setup (macOS / VS Code)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and replace the example email with an address you control.

Convert parsed Form 4 rows into event records:

```bash
python -m preprocessing.sec_to_events
python -m scripts.run_pipeline --input data/raw/news/sec_form4_events.csv
```

## Run the rule-based pipeline (no trained model required)

Put a CSV at `data/raw/news/news_input.csv`. Required column: `title`.
Recommended columns: `description`, `ticker`, `published_at`, `source`, `url`.

For an insider transaction, also provide any available structured fields:
`insider_role`, `transaction_code`, `shares_sold`, `shares_owned_before`,
`shares_owned_after`, `transaction_value`, `is_10b5_1`.

Then run:

```bash
python -m scripts.run_pipeline --input data/raw/news/news_input.csv
```

The pipeline writes:
- `data/processed/event_analysis.csv`
- `data/processed/investor_alerts.jsonl`

If a prediction model has not been trained, the output explicitly says `not_trained`.
It does not invent predictions.

## SEC Form 4 collector

```bash
python -m preprocessing.sec_form4_collector --ticker AAPL --limit 10
```

This stores raw filings in `data/raw/sec/` and parsed transactions in
`data/processed/sec_form4_transactions.csv`. Inspect source filings and footnotes;
transaction code `S` alone does not prove a discretionary open-market sale.

## Train the two prediction heads

Training data must be historical, verified, and labelled using information available
after each event. See `data/processed/training_data_schema.csv` for the expected columns.

```bash
python -m models.train_models --input data/processed/training_data.csv
```

The trainer performs a chronological holdout when `published_at` is present, reports
classification metrics, and saves separate model artifacts. Do not use random row splits
for final financial backtests. For stronger validation, use ticker-aware and time-aware
splits, and ensure repeated or near-duplicate headlines cannot leak across splits.

## Generate stock-price labels (optional helper)

```bash
python -m preprocessing.build_price_labels --input data/processed/events_with_tickers.csv
```

This uses yfinance to calculate five-trading-session market-adjusted returns versus SPY.
It is an experimental label helper, not a guaranteed source of survivorship-bias-free data.

## Short-interest labels

Supply a verified historical observations CSV with columns:
`ticker, settlement_date, short_interest`.
Then create short-interest labels from the enriched event file:

```bash
python -m preprocessing.label_short_interest --events data/processed/event_analysis.csv --observations path/to/verified_short_interest_observations.csv --output data/processed/events_with_short_interest_labels.csv
```

Create stock-price labels:

```bash
python -m preprocessing.build_price_labels --input data/processed/event_analysis.csv --output data/processed/events_with_price_labels.csv
```

Merge the two target sets:

```bash
python -m preprocessing.merge_labels --short-interest data/processed/events_with_short_interest_labels.csv --stock-price data/processed/events_with_price_labels.csv --output data/processed/training_data.csv
```
Short interest is reported periodically, not continuously. Do not substitute daily short-sale
volume for short interest.

## Important research safeguards

- Predictions are hypotheses, not investment advice.
- A Form 4 sale is not automatically bearish.
- Keep publication/filing time separate from transaction time.
- Do not use future information in features.
- Evaluate price and short-interest targets separately.
- Include a neutral class and define horizons/thresholds before evaluating.
- The initial rule-based classifier is a transparent prototype, not a trained NLP model.
- Confidence scores are model probabilities only after proper calibration; they are not
  guarantees of correctness.

## Backend REST API Server

Start the REST API server:

```bash
cd "PHASE 2"
python -m api.main
# Or directly via uvicorn:
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Endpoints
- `GET /api/health`: Health status and pre-loaded model availability.
- `POST /api/predict`: Real-time news event classification and short-interest prediction.

