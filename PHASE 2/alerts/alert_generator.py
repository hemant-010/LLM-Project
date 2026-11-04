import json
from datetime import datetime, timezone

def build_alert(row: dict, predictions: dict) -> dict:
    def summary(name, result):
        if result["status"] == "predicted":
            return {
                "direction": result["direction"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "horizon": result["horizon"],
            }
        return {"status":result["status"],"message":result["message"]}
    return {
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "ticker":row.get("ticker") or None,
        "headline":row.get("title"),
        "published_at":str(row.get("published_at") or ""),
        "event_type":row.get("event_type"),
        "event_severity":row.get("event_severity"),
        "event_classification_confidence":row.get("event_classification_confidence"),
        "context_flags":row.get("context_flags"),
        "short_interest_prediction":summary("short_interest",predictions["short_interest"]),
        "stock_price_prediction":summary("stock_price",predictions["stock_price"]),
        "source":row.get("source"),
        "url":row.get("url"),
        "disclaimer":"Research output, not investment advice. Event classification and predicted market direction may be wrong."
    }

def render_text(alert: dict) -> str:
    si = alert["short_interest_prediction"]
    sp = alert["stock_price_prediction"]
    return (
        f"[{alert.get('ticker') or 'UNKNOWN'}] {alert.get('headline')}\\n"
        f"Event: {alert.get('event_type')} | Severity: {alert.get('event_severity')}\\n"
        f"Short-interest: {si.get('direction') or si.get('status')}\\n"
        f"Stock-price: {sp.get('direction') or sp.get('status')}\\n"
        f"Context: {alert.get('context_flags')}\\n"
        f"Source: {alert.get('url')}\\n"
        f"NOTE: {alert['disclaimer']}"
    )
