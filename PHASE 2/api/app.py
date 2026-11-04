"""FastAPI REST API application for Short Interest Trend Prediction."""
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path

# Ensure PHASE 2 directory is in sys.path
PHASE2_ROOT = Path(__file__).resolve().parent.parent
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config import ROOT, MODEL_DIR
from models.predictor import DualPredictor
from alerts.insider_analyzer import analyze_insider_transaction
from alerts.notification_store import get_notification_store
from preprocessing.sec_form4_service import get_sec_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api")

# Shared Predictor instance
_predictor: Optional[DualPredictor] = None

def get_predictor() -> DualPredictor:
    """Retrieve or initialize the singleton DualPredictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = DualPredictor(model_dir=MODEL_DIR)
        _predictor.preload()
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for model pre-loading and warmup."""
    logger.info("Initializing Short Interest Prediction API...")
    predictor = get_predictor()
    preload_status = predictor.preload()
    logger.info(f"Model preload status: {preload_status}")
    app.state.predictor = predictor

    # Initialize & seed persistent notification store
    notif_store = get_notification_store()
    sec_service = get_sec_service()
    seed_alerts = []
    for sym in ["TSLA", "AAPL", "AMZN", "GM", "NVDA"]:
        for tx in sec_service.get_transactions_for_ticker(ticker=sym, limit=5):
            if tx.get("significance_category") in {"significant", "requires_review"}:
                seed_alerts.append(tx)
    notif_store.seed_initial_alerts(seed_alerts)
    app.state.notification_store = notif_store

    yield
    logger.info("Shutting down Short Interest Prediction API...")


app = FastAPI(
    title="Short Interest Trend Prediction API",
    description="REST API for financial news event classification and dual-horizon market prediction (Short-Interest & Stock-Price).",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for React frontend
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
allowed_origins = [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    title: str = Field(
        ...,
        description="News headline or event title to classify and predict.",
        min_length=3,
        max_length=1000
    )
    ticker: Optional[str] = Field(
        default=None,
        description="Stock ticker symbol (optional).",
        max_length=15
    )
    description: Optional[str] = Field(
        default="",
        description="Detailed news body or SEC filing context (optional).",
        max_length=10000
    )
    model_family: Literal["logistic_regression", "xgboost"] = Field(
        default="logistic_regression",
        description="Machine learning model family for short-interest prediction."
    )
    use_llm: Optional[bool] = Field(
        default=False,
        description="Whether to use Gemini LLM for event enrichment (requires GEMINI_API_KEY). Defaults to False (deterministic rule-based heuristics)."
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Event headline 'title' cannot be empty or purely whitespace.")
        return trimmed

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        trimmed = v.strip().upper()
        return trimmed if trimmed else None

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: Optional[str]) -> str:
        return v.strip() if v else ""


class EventMetadataResponse(BaseModel):
    ticker: Optional[str] = None
    title: str
    event_type: Optional[str] = None
    event_severity: Optional[str] = None
    event_classification_confidence: Optional[float] = None
    classification_source: Optional[str] = None
    explanation: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)
    context_flags: Optional[str] = None


class ShortInterestPredictionResponse(BaseModel):
    status: str
    direction: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    horizon: str
    model_type: str


class StockPricePredictionResponse(BaseModel):
    status: str
    direction: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    horizon: str


class PredictionResponse(BaseModel):
    status: str = "success"
    model_family: str
    event: EventMetadataResponse
    short_interest: ShortInterestPredictionResponse
    stock_price: StockPricePredictionResponse
    disclaimer: str = (
        "Research output, not investment advice. Event classification and predicted short-interest "
        "trends may be incorrect. Past statistical patterns do not guarantee future market behavior."
    )


# ---------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_details = []
    for error in exc.errors():
        field = " -> ".join([str(loc) for loc in error.get("loc", [])])
        msg = error.get("msg", "Invalid value")
        error_details.append(f"{field}: {msg}")
    
    logger.warning(f"Request validation failure on {request.url.path}: {error_details}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Input validation error",
            "errors": error_details
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning(f"Value error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "error",
            "message": str(exc)
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log internal error without exposing sensitive details or paths to users
    logger.error(f"Internal server error processing {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An internal server error occurred while processing the request."
        }
    )


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/api/health", tags=["Health"])
def health_check():
    """Health check endpoint reflecting predictor and model readiness."""
    predictor = getattr(app.state, "predictor", None) or get_predictor()
    
    lr_path = predictor._resolve_model_path("short_interest_model.joblib")
    xgb_path = predictor._resolve_model_path("short_interest_xgboost_model.joblib")
    sp_path = predictor._resolve_model_path("stock_price_model.joblib")
    
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    has_gemini = bool(gemini_key and not any(p in gemini_key.lower() for p in ["your-api-key", "placeholder", "example"]))
    
    return {
        "status": "healthy",
        "service": "Short Interest Trend Prediction API",
        "models": {
            "logistic_regression": "ready" if lr_path.exists() else "not_found",
            "xgboost": "ready" if xgb_path.exists() else "not_found",
            "stock_price": "ready" if sp_path.exists() else "not_found"
        },
        "environment": {
            "gemini_configured": has_gemini,
            "models_cached": len(predictor._artifact_cache)
        }
    }


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_trend(req: PredictionRequest):
    """Enrich an incoming financial news event and predict short-interest trend direction."""
    t0 = time.time()
    predictor = getattr(app.state, "predictor", None) or get_predictor()

    try:
        raw_result = predictor.predict_event(
            title=req.title,
            ticker=req.ticker or "",
            description=req.description or "",
            use_llm=req.use_llm,
            model_family=req.model_family
        )
    except Exception as e:
        logger.error(f"Prediction engine error for ticker='{req.ticker}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate prediction from the machine learning engine."
        )

    event_data = raw_result.get("event", {})
    preds = raw_result.get("predictions", {})
    si_pred = preds.get("short_interest", {})
    sp_pred = preds.get("stock_price", {})

    elapsed_ms = round((time.time() - t0) * 1000, 2)
    logger.info(
        f"Predicted trend for ticker='{req.ticker}' family='{req.model_family}' "
        f"direction='{si_pred.get('direction')}' in {elapsed_ms}ms"
    )

    return PredictionResponse(
        status="success",
        model_family=req.model_family,
        event=EventMetadataResponse(
            ticker=event_data.get("ticker"),
            title=event_data.get("title", req.title),
            event_type=event_data.get("event_type"),
            event_severity=event_data.get("event_severity"),
            event_classification_confidence=event_data.get("event_classification_confidence"),
            classification_source=event_data.get("classification_source"),
            explanation=event_data.get("explanation"),
            evidence=event_data.get("evidence") or [],
            context_flags=event_data.get("context_flags")
        ),
        short_interest=ShortInterestPredictionResponse(
            status=si_pred.get("status", "unknown"),
            direction=si_pred.get("direction"),
            probabilities=si_pred.get("probabilities"),
            confidence=si_pred.get("confidence"),
            horizon=si_pred.get("horizon", "next eligible short-interest reporting period"),
            model_type=si_pred.get("model_type", "LogisticRegression")
        ),
        stock_price=StockPricePredictionResponse(
            status=sp_pred.get("status", "unknown"),
            direction=sp_pred.get("direction"),
            probabilities=sp_pred.get("probabilities"),
            confidence=sp_pred.get("confidence"),
            horizon=sp_pred.get("horizon", "five trading sessions")
        )
    )


# ---------------------------------------------------------
# Milestone 3: Insider Selling & Alert Endpoints
# ---------------------------------------------------------

class InsiderAnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    insider_name: Optional[str] = "Corporate Insider"
    insider_role: Optional[str] = "Officer/Director"
    shares_sold: float = Field(..., gt=0)
    price_per_share: Optional[float] = None
    transaction_value: Optional[float] = None
    shares_owned_after: Optional[float] = None
    shares_owned_before: Optional[float] = None
    percentage_holdings_sold: Optional[float] = None
    is_10b5_1: Optional[bool] = None
    transaction_date: Optional[str] = None
    filing_url: Optional[str] = None

    @field_validator("ticker")
    @classmethod
    def clean_ticker(cls, v: str) -> str:
        return v.strip().upper()


@app.get("/api/insider-transactions", tags=["Insider Selling"])
def list_insider_transactions(ticker: str = "TSLA", limit: int = 10):
    """Retrieve and analyze SEC Form 4 insider transactions for a ticker."""
    sec_service = get_sec_service()
    transactions = sec_service.get_transactions_for_ticker(ticker=ticker, limit=limit)
    return {
        "status": "success",
        "ticker": ticker.upper(),
        "total_returned": len(transactions),
        "transactions": transactions
    }


@app.post("/api/analyze-insider", tags=["Insider Selling"])
def evaluate_insider_transaction(req: InsiderAnalysisRequest):
    """Perform forensic evaluation of an insider transaction and compute significance."""
    analysis = analyze_insider_transaction(req.model_dump())
    return {
        "status": "success",
        "analysis": analysis
    }


@app.get("/api/insider-alerts", tags=["Insider Selling"])
def list_insider_alerts():
    """Retrieve recent significant and review-recommended insider selling notifications across core institutional tickers."""
    sec_service = get_sec_service()
    alerts = []
    
    # Screen across core institutional coverage
    coverage_tickers = ["TSLA", "AAPL", "AMZN", "GM", "NVDA"]
    for sym in coverage_tickers:
        txs = sec_service.get_transactions_for_ticker(ticker=sym, limit=5)
        for tx in txs:
            if tx.get("significance_category") in {"significant", "requires_review"}:
                alerts.append(tx)

    # Sort significant first
    alerts.sort(key=lambda x: (x["significance_category"] != "significant", -x["risk_score"]))

    return {
        "status": "success",
        "total_alerts": len(alerts),
        "alerts": alerts,
        "disclaimer": (
            "Research output, not investment advice. Insider sales occur for estate, tax, and planned "
            "liquidity reasons and are not automated signals of fundamental business weakness."
        )
    }


# ---------------------------------------------------------
# Milestone 4: Persistent Investor Notification Endpoints
# ---------------------------------------------------------

class NotificationCreateRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    message: str = Field(..., min_length=3, max_length=2000)
    ticker: Optional[str] = Field(default=None, max_length=10)
    severity: Literal["significant", "requires_review", "routine", "info"] = "requires_review"
    category: str = "insider_selling"
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("ticker")
    @classmethod
    def clean_ticker(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().upper() if v else None


class NotificationMarkReadRequest(BaseModel):
    notification_id: Optional[str] = None  # None marks all as read


@app.get("/api/notifications", tags=["Notifications"])
def get_notifications(limit: int = 50, unread_only: bool = False):
    """Retrieve persistent notifications and current unread count."""
    store = getattr(app.state, "notification_store", None) or get_notification_store()
    notifs = store.get_notifications(limit=limit, unread_only=unread_only)
    unread_count = store.get_unread_count()
    return {
        "status": "success",
        "unread_count": unread_count,
        "total_returned": len(notifs),
        "notifications": notifs
    }


@app.post("/api/notifications", tags=["Notifications"])
def create_notification(req: NotificationCreateRequest):
    """Create a persistent notification with automated duplicate prevention."""
    store = getattr(app.state, "notification_store", None) or get_notification_store()
    created = store.create_notification(
        title=req.title,
        message=req.message,
        ticker=req.ticker,
        severity=req.severity,
        category=req.category,
        metadata=req.metadata
    )
    if created is None:
        return {
            "status": "duplicate",
            "message": "Notification with identical fingerprint already exists. Skipped insertion."
        }
    return {
        "status": "success",
        "notification": created
    }


@app.post("/api/notifications/mark-read", tags=["Notifications"])
def mark_notifications_read(req: NotificationMarkReadRequest):
    """Mark a single notification or all notifications as read."""
    store = getattr(app.state, "notification_store", None) or get_notification_store()
    updated = store.mark_as_read(notif_id=req.notification_id)
    return {
        "status": "success",
        "updated_count": updated,
        "unread_count": store.get_unread_count()
    }


@app.delete("/api/notifications", tags=["Notifications"])
def clear_notifications(notification_id: Optional[str] = None):
    """Delete a single notification or clear all notifications."""
    store = getattr(app.state, "notification_store", None) or get_notification_store()
    deleted = store.delete_notification(notif_id=notification_id)
    return {
        "status": "success",
        "deleted_count": deleted,
        "unread_count": store.get_unread_count()
    }


