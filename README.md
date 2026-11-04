# Short Interest Trend Prediction with Large Language Models

An end-to-end quantitative financial intelligence and machine learning platform predicting FINRA short-interest movements and screening significant corporate insider selling from financial news events and SEC EDGAR Form 4 filings.

---

## Architecture Overview

```
                                      ┌─────────────────────────────────────┐
                                      │        React Investor Dashboard     │
                                      │  (Vite, Tailwind CSS, Lucide Icons) │
                                      └──────────────────┬──────────────────┘
                                                         │
                                               HTTP / JSON REST API
                                                         │
                                                         ▼
                                      ┌─────────────────────────────────────┐
                                      │       FastAPI Backend Application   │
                                      │   (api/app.py - In-Memory Caching)  │
                                      └──────┬───────────────────────┬──────┘
                                             │                       │
                     ┌───────────────────────┘                       └────────────────────────┐
                     ▼                                                                        ▼
   ┌───────────────────────────────────┐                                    ┌───────────────────────────────────┐
   │     Dual Machine Learning Engine  │                                    │   SEC Form 4 Insider Intelligence │
   │  • Logistic Regression (30k TF-IDF)│                                    │  • SEC EDGAR XML Ownership Parser │
   │  • XGBoost (Categorical & Numeric)│                                    │  • Rule 10b5-1 Plan Discrimination│
   │  • Gemini Grounded Enrichment     │                                    │  • Multi-Dimensional Significance │
   │  • Deterministic Rule Fallbacks   │                                    │  • Persistent SQLite Store        │
   └───────────────────────────────────┘                                    └───────────────────────────────────┘
```

---

## Repository Structure

```
├── README.md                           # Root documentation & architecture guide
├── .gitignore                          # Global exclusion rules for secrets & large datasets
│
├── PHASE 1/                            # Phase 1 Foundational Research & Reproduction
│   ├── research paper.pdf              # Academic research foundation
│   ├── stage1_reproduction.ipynb       # Phase 1 Jupyter reproduction notebook
│   ├── sitp_dataset.parquet            # Consolidated Stage 1 dataset
│   ├── experiment_config.json          # Flan-T5 model configuration
│   └── stage 1 report.pdf              # Stage 1 summary report
│
├── PHASE 2/                            # Production ML Models, REST API & Alert Systems
│   ├── api/
│   │   └── app.py                      # FastAPI REST application (Health, Predict, Form 4, Alerts)
│   ├── models/
│   │   ├── predictor.py                # DualPredictor singleton with model caching
│   │   └── artifacts/                  # Trained production models & audit metrics
│   │       ├── production/             # Validated Logistic Regression production model (1.8MB)
│   │       └── short_interest_xgboost_model.joblib # Production XGBoost model (3.1MB)
│   ├── alerts/
│   │   ├── insider_analyzer.py         # Multi-dimensional Form 4 significance & 10b5-1 engine
│   │   ├── notification_store.py       # Persistent SQLite notification store & deduplication
│   │   └── alert_generator.py          # Notification message formatter
│   ├── preprocessing/
│   │   ├── sec_form4_service.py        # SEC Form 4 retrieval & data provenance layer
│   │   └── sec_form4_collector.py      # SEC EDGAR XML ownership collector
│   ├── feature_engineering/
│   │   ├── gemini_classifier.py        # Grounded Gemini event classifier
│   │   ├── gemini_context_analyzer.py  # Gemini context & severity extractor
│   │   └── build_features.py           # Unified feature enrichment pipeline
│   ├── tests/                          # 130 passing unit, integration & E2E tests
│   ├── Dockerfile                      # Production container specification
│   ├── Procfile                        # ASGI process runner for PaaS deployment
│   ├── render.yaml                     # Render Blueprint specification
│   └── requirements.txt                # Python dependencies
│
└── frontend/                           # React 18 Investor Dashboard
    ├── src/
    │   ├── App.jsx                     # Core application orchestrator
    │   ├── components/                 # UI components
    │   │   ├── Header.jsx              # Status badges & persistent notification drawer
    │   │   ├── StatsOverview.jsx       # Real-time session KPI cards
    │   │   ├── PredictionForm.jsx      # Financial news input & model switcher
    │   │   ├── PredictionResult.jsx    # 3-class probability distribution visualizer
    │   │   ├── InsiderSellingCard.jsx  # Interactive Form 4 screener & alert feed
    │   │   └── PredictionHistory.jsx   # Session prediction history
    ├── vercel.json                     # Vercel SPA routing configuration
    └── package.json                    # Frontend dependencies
```

---

## Quickstart

### 1. Backend Setup & Run

```bash
cd "PHASE 2"

# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Configure environment (optional, defaults provided)
cp .env.example .env

# 3. Launch the FastAPI server
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

API documentation will be available at `http://localhost:8000/docs`.

### 2. Frontend Setup & Run

```bash
cd frontend

# 1. Install dependencies
npm install

# 2. Launch Vite development server
npm run dev
```

Dashboard will open at `http://localhost:3000`.

---

## Automated Test Suite

Run all 130 unit, functional, statistical audit, and end-to-end integration tests:

```bash
cd "PHASE 2"
pytest tests/
```

Expected result:
```
======================= 130 passed, 7 warnings in ~27s =======================
```

---

## Production Deployment

- **Frontend:** Pre-configured for **Vercel** via `frontend/vercel.json`. Set `VITE_API_BASE` in Vercel project environment variables.
- **Backend:** Pre-configured for **Render** via `PHASE 2/render.yaml` (with persistent disk mount for SQLite) or **Railway** via `PHASE 2/Dockerfile`.
