# Short Interest Trend Prediction — React Investor Dashboard

Modern financial intelligence interface for forecasting FINRA short-interest trends from corporate news events and regulatory filings.

## Architecture

- **Engine:** React 18 + Vite + Tailwind CSS v4 + Lucide Icons.
- **Backend:** FastAPI REST API (`http://localhost:8000`).
- **Proxy:** Configured in `vite.config.js` (`/api` -> `http://127.0.0.1:8000`).

## Running the Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at: `http://localhost:3000`

## Production Build

```bash
cd frontend
npm run build
npm run preview
```

## Features

1. **Live System Health Monitoring:** Auto-pings `/api/health` displaying Logistic Regression, XGBoost, and Gemini status.
2. **Event Evaluation Form:** Input title, ticker, and optional filing context.
3. **Model Selection:** Logistic Regression (TF-IDF 30k + Categoricals) or XGBoost (Calibrated Gradient Boosted Trees).
4. **Gemini LLM Enrichment:** Toggle for LLM evidence extraction with automatic deterministic regex fallback.
5. **Multi-Class Probability Distribution:** Visual bars for `increase`, `decrease`, and `no_significant_change`.
6. **Event Risk Context:** Event category, severity, classification confidence, verbatim text quotes, and context flags.
7. **Session History:** In-memory session tracking with instant reload.
8. **Experimental Stock-Price Notice:** Clearly marked 16-sample legacy fixture warning banner.
9. **Insider Selling Preview:** Reserved section for Milestone 3 SEC Form 4 monitoring.

