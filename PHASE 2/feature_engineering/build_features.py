"""Feature engineering pipeline for corporate event enrichment with LLM & heuristic modules."""
import os
import hashlib
import pandas as pd
from feature_engineering.event_classifier import classify_event
from feature_engineering.context_analyzer import analyze_context

def enrich_events(df: pd.DataFrame, use_llm: bool | None = None) -> pd.DataFrame:
    """Enrich corporate news and filing records with classification and context features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with corporate events (must contain title, description, etc.).
    use_llm : bool | None
        Explicit toggle for LLM features. If True, enables Gemini for both classification
        and context analysis. If False, forces deterministic rule-based fallbacks.
        If None (default), respects environment variables (USE_GEMINI_CLASSIFIER,
        USE_GEMINI_CONTEXT, or USE_GEMINI).

    Returns
    -------
    pd.DataFrame
        Enriched dataframe with all original columns preserved plus classification
        and context features.
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    for record in df.to_dict(orient="records"):
        # 1. Run event classification first
        class_res = classify_event(
            title=str(record.get("title") or ""),
            description=str(record.get("description") or ""),
            transaction_code=str(record.get("transaction_code") or ""),
            use_llm=use_llm
        )

        event_category = class_res.event_type
        record["event_type"] = event_category
        record["event_category"] = event_category
        record["event_classification_confidence"] = class_res.confidence
        record["confidence"] = class_res.confidence
        record["classification_source"] = class_res.source
        record["classification_explanation"] = getattr(class_res, "explanation", "")
        record["all_detected_event_types"] = ";".join(class_res.all_detected)

        # 2. Pass classification result and available event data into context analysis
        context_res = analyze_context(
            row=record,
            use_llm=use_llm,
            classification_result=class_res
        )
        record.update(context_res)

        # Ensure canonical severity and evidence aliases match requirements
        record["severity"] = record.get("event_severity")

        # Merge classification evidence into supporting_evidence if context had none
        supporting_evidence = record.get("supporting_evidence") or []
        class_evidence = getattr(class_res, "evidence", []) or []
        combined_evidence = list(dict.fromkeys(supporting_evidence + class_evidence))
        record["supporting_evidence"] = combined_evidence

        # 3. Model text and event id for backward compatibility
        record["headline_text"] = str(record.get("title") or "").strip()
        record["model_text"] = (str(record.get("title") or "") + " " +
                                str(record.get("description") or "")).strip()
        if not record.get("event_id"):
            identity = "|".join([
                str(record.get("ticker") or "").upper(),
                str(record.get("published_at") or ""),
                str(record.get("title") or "")
            ])
            record["event_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        rows.append(record)

    return pd.DataFrame(rows)
