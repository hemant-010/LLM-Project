"""Compute interpretable context features; not a causal or investment-risk score."""
import os
import pandas as pd

def _number(value):
    try:
        x = float(value)
        return x if pd.notna(x) else None
    except (TypeError, ValueError):
        return None

def compute_context_metrics(row: dict) -> dict:
    """Compute quantitative holding metrics and extract 10b5-1 status."""
    category = row.get("event_type", "other")
    shares = _number(row.get("shares_sold"))
    before = _number(row.get("shares_owned_before"))
    after = _number(row.get("shares_owned_after"))
    value = _number(row.get("transaction_value"))
    pct = (shares / before * 100) if shares is not None and before and before > 0 else None
    if pct is None and shares is not None and after is not None and shares + after > 0:
        pct = shares / (shares + after) * 100

    val_10b5 = row.get("is_10b5_1")
    if isinstance(val_10b5, bool):
        is_10b5 = val_10b5
    elif pd.isna(val_10b5):
        is_10b5 = None
    else:
        raw_10b5 = str(val_10b5).strip().lower()
        if raw_10b5 in {"1", "true", "yes", "y"}:
            is_10b5 = True
        elif raw_10b5 in {"0", "false", "no", "n"}:
            is_10b5 = False
        else:
            is_10b5 = None

    return {
        "event_type": category,
        "shares_sold": shares,
        "shares_owned_before": before,
        "shares_owned_after": after,
        "transaction_value": value,
        "percentage_holdings_sold": round(pct, 4) if pct is not None else None,
        "insider_is_10b5_1": is_10b5,
    }

def analyze_context_rule_based(row: dict, metrics: dict | None = None) -> dict:
    """Deterministic rule-based context and severity analyzer."""
    if metrics is None:
        metrics = compute_context_metrics(row)

    category = metrics["event_type"]
    pct = metrics["percentage_holdings_sold"]
    value = metrics["transaction_value"]
    is_10b5 = metrics["insider_is_10b5_1"]
    role = str(row.get("insider_role") or "").lower()

    flags = []
    if category == "insider_selling":
        if pct is None:
            severity = "unknown"
            flags.append("Insider holding denominator unavailable; sale significance cannot be quantified.")
        elif pct >= 25:
            severity = "high"
        elif pct >= 10:
            severity = "medium"
        else:
            severity = "low"
        if is_10b5 is True:
            flags.append("Transaction is marked as a 10b5-1 plan; interpret with additional context.")
        elif is_10b5 is False:
            flags.append("Transaction is confirmed not under a 10b5-1 plan.")
        elif is_10b5 is None:
            flags.append("10b5-1 plan status is unknown from the extracted fields.")
        if "ceo" in role or "chief executive" in role:
            flags.append("Reporting person is identified as CEO.")
        if value is None:
            flags.append("Transaction value unavailable.")
    elif category in {"fraud_investigation", "bankruptcy_distress"}:
        severity = "high"
    elif category in {"analyst_downgrade", "earnings_miss", "product_recall"}:
        severity = "medium"
    elif category in {"positive_earnings", "merger_acquisition"}:
        severity = "context_dependent"
    else:
        severity = "unknown"

    return {
        "percentage_holdings_sold": pct,
        "transaction_value": value,
        "insider_is_10b5_1": is_10b5,
        "event_severity": severity,
        "context_flags": " | ".join(flags),
        "severity_rationale": "",
        "context_summary": "",
        "risk_factors": [],
        "supporting_evidence": [],
        "insider_nuance": "",
        "context_source": "rule_based_fallback",
    }

def analyze_context(
    row: dict,
    use_llm: bool | None = None,
    classification_result = None
) -> dict:
    """Analyze event context and severity with optional Gemini LLM routing.

    Gemini LLM is disabled by default (use_llm=False or unset) and must be explicitly enabled
    via parameter (use_llm=True) or USE_GEMINI_CONTEXT=true environment variable.
    """
    if classification_result is not None:
        if hasattr(classification_result, "event_type"):
            row["event_type"] = classification_result.event_type
        if hasattr(classification_result, "source"):
            row["classification_source"] = classification_result.source
        if hasattr(classification_result, "evidence") and classification_result.evidence:
            row["classification_evidence"] = classification_result.evidence

    if use_llm is None:
        enable_llm = os.getenv("USE_GEMINI_CONTEXT", "").strip().lower() in {"1", "true", "yes"}
    else:
        enable_llm = bool(use_llm)
    metrics = compute_context_metrics(row)

    if enable_llm:
        from feature_engineering.gemini_context_analyzer import analyze_context_with_gemini
        llm_res = analyze_context_with_gemini(row, metrics)
        if llm_res is not None:
            return {
                "percentage_holdings_sold": metrics["percentage_holdings_sold"],
                "transaction_value": metrics["transaction_value"],
                "insider_is_10b5_1": metrics["insider_is_10b5_1"],
                "event_severity": llm_res["severity"],
                "context_flags": llm_res["context_flags"],
                "severity_rationale": llm_res.get("severity_rationale", ""),
                "context_summary": llm_res.get("context_summary", ""),
                "risk_factors": llm_res.get("risk_factors", []),
                "supporting_evidence": llm_res.get("supporting_evidence", []),
                "insider_nuance": llm_res.get("insider_nuance", ""),
                "context_source": "gemini_llm",
            }

    return analyze_context_rule_based(row, metrics)
