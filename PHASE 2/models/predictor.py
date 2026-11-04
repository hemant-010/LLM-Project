import sys
from pathlib import Path
import joblib
import pandas as pd
from config import MODEL_DIR

# Ensure StructuredXGBoostPipeline is deserializable if pickled under __main__
try:
    import models.train_xgboost_short_interest as txgb
    if "__main__" in sys.modules and not hasattr(sys.modules["__main__"], "StructuredXGBoostPipeline"):
        sys.modules["__main__"].StructuredXGBoostPipeline = txgb.StructuredXGBoostPipeline
except Exception:
    pass

class DualPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR, prefer_production: bool = True):
        self.model_dir = Path(model_dir)
        self.prefer_production = prefer_production
        self._artifact_cache: dict[str, dict] = {}

    def clear_cache(self) -> None:
        """Clear the in-memory artifact cache."""
        self._artifact_cache.clear()

    def _resolve_model_path(self, filename: str) -> Path:
        if self.prefer_production:
            prod_path = self.model_dir / "production" / filename
            if prod_path.exists():
                return prod_path
        return self.model_dir / filename

    def _load_artifact(self, path: Path) -> dict:
        cache_key = str(path.resolve())
        if cache_key not in self._artifact_cache:
            self._artifact_cache[cache_key] = joblib.load(path)
        return self._artifact_cache[cache_key]

    def preload(self) -> dict[str, str]:
        """Pre-load model artifacts into memory for fast zero-latency inference."""
        status = {}
        for fn in ["short_interest_model.joblib", "short_interest_xgboost_model.joblib", "stock_price_model.joblib"]:
            p = self._resolve_model_path(fn)
            if p.exists():
                try:
                    self._load_artifact(p)
                    status[fn] = "ready"
                except Exception as e:
                    status[fn] = f"error: {e}"
            else:
                status[fn] = "not_found"
        return status

    def _one(self, filename: str, row: dict) -> dict:
        path = self._resolve_model_path(filename)
        if not path.exists():
            return {"status":"not_trained", "direction":None, "probabilities":None,
                    "message":f"Model artifact not found: {filename}"}
        artifact = self._load_artifact(path)
        model = artifact["model"]
        frame = pd.DataFrame([row])
        # Ensure text representations exist
        if "headline_text" not in frame.columns:
            if "title" in frame.columns and frame["title"].iloc[0] is not None:
                frame["headline_text"] = str(frame["title"].iloc[0])
            elif "model_text" in frame.columns and frame["model_text"].iloc[0] is not None:
                frame["headline_text"] = str(frame["model_text"].iloc[0])
            else:
                frame["headline_text"] = ""
        if "model_text" not in frame.columns:
            frame["model_text"] = frame["headline_text"]

        # Ensure standard categorical and numerical features exist with proper dtypes
        for c in ["event_type", "event_category", "event_severity", "insider_role", "transaction_code", "is_multi_ticker"]:
            if c not in frame.columns:
                frame[c] = ""
            else:
                frame[c] = frame[c].fillna("").astype(str)
        for n in ["percentage_holdings_sold", "transaction_value", "event_classification_confidence",
                  "short_interest_before", "n_tokens", "headline_ticker_count"]:
            if n not in frame.columns:
                frame[n] = float("nan")
            else:
                frame[n] = pd.to_numeric(frame[n], errors="coerce")
        probabilities = model.predict_proba(frame)[0]
        classes = list(getattr(model, "classes_", artifact.get("classes", [])))
        order = probabilities.argsort()[::-1]
        return {
            "status":"predicted",
            "direction":str(classes[order[0]]),
            "probabilities":{str(classes[i]):round(float(probabilities[i]),4) for i in range(len(classes))},
            "confidence":round(float(probabilities[order[0]]),4),
            "horizon":artifact.get("horizon","unspecified"),
            "model_type":artifact.get("model_type", "LogisticRegression"),
        }

    def predict(self, row: dict, model_family: str = "logistic_regression") -> dict:
        si_artifact = "short_interest_xgboost_model.joblib" if model_family.lower() == "xgboost" else "short_interest_model.joblib"
        return {
            "short_interest": self._one(si_artifact, row),
            "stock_price": self._one("stock_price_model.joblib", row),
        }

    def predict_event(
        self,
        title: str,
        ticker: str = "",
        description: str = "",
        use_llm: bool | None = None,
        model_family: str = "logistic_regression"
    ) -> dict:
        """End-to-end convenience method: enriches a raw headline and runs dual prediction."""
        from feature_engineering.build_features import enrich_events
        raw_df = pd.DataFrame([{
            "title": title,
            "ticker": ticker,
            "description": description
        }])
        enriched_df = enrich_events(raw_df, use_llm=use_llm)
        row = enriched_df.iloc[0].to_dict()
        preds = self.predict(row, model_family=model_family)
        return {
            "event": {
                "ticker": ticker or None,
                "title": title,
                "event_type": row.get("event_type"),
                "event_severity": row.get("event_severity"),
                "event_classification_confidence": row.get("event_classification_confidence"),
                "classification_source": row.get("classification_source"),
                "explanation": row.get("classification_explanation"),
                "evidence": row.get("supporting_evidence"),
                "context_flags": row.get("context_flags")
            },
            "predictions": preds
        }
