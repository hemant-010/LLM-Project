"""Event classification module supporting Gemini LLM extraction with deterministic rule-based fallback."""
import os
import re

PATTERNS = {
    "insider_selling": [r"\bceo\b.{0,80}\b(sell|sells|sold|selling)\b",
        r"\b(insider|director|executive|founder)\b.{0,100}\b(sell|sells|sold|selling|disposed)\b",
        r"\bform\s*4\b", r"\binsider.{0,30}sale\b"],
    "analyst_downgrade": [r"\b(downgrade|downgrades|downgraded|cut to sell|cut to hold)\b",
        r"\blowered.{0,30}(rating|price target)\b"],
    "earnings_miss": [r"\b(misses|missed|below expectations|loss widens|unexpected loss)\b",
        r"\bearnings.{0,50}(miss|disappoint|below)\b"],
    "fraud_investigation": [r"\b(fraud|accounting scandal|sec investigation|criminal probe|investigation into)\b"],
    "product_recall": [r"\b(recall|recalls|recalled|safety defect)\b"],
    "merger_acquisition": [r"\b(merger|acquisition|acquire|acquires|buyout|takeover)\b"],
    "bankruptcy_distress": [r"\b(bankruptcy|chapter 11|insolvency|default|debt restructuring)\b"],
    "positive_earnings": [r"\b(beat estimates|beats expectations|record profit|earnings surge|raises guidance)\b"]
}

class EventClassificationResult:
    """Rich classification result container supporting transparent 3-tuple backward compatibility."""
    def __init__(
        self,
        event_type: str,
        confidence: float,
        all_detected: list[str],
        explanation: str = "",
        evidence: list[str] | None = None,
        source: str = "rule_based_fallback",
        is_calibrated: bool = False
    ):
        self.event_type = event_type
        self.category = event_type
        self.confidence = confidence
        self.all_detected = all_detected
        self.all_detected_event_types = all_detected
        self.explanation = explanation
        self.evidence = evidence or []
        self.source = source
        self.is_calibrated = is_calibrated

    def __iter__(self):
        yield self.event_type
        yield self.confidence
        yield self.all_detected

    def __getitem__(self, idx):
        return [self.event_type, self.confidence, self.all_detected][idx]

    def __repr__(self):
        return (f"EventClassificationResult(event_type={self.event_type!r}, "
                f"confidence={self.confidence}, source={self.source!r})")

def classify_event_rule_based(title: str, description: str = "", transaction_code: str = "") -> tuple[str, float, list[str]]:
    """Deterministic regex rule-based classifier."""
    text = f"{title} {description}".lower()
    matches = []
    for category, patterns in PATTERNS.items():
        hits = [p for p in patterns if re.search(p, text, re.I)]
        if hits: matches.append((category, len(hits)))
    if transaction_code.upper() == "S" and not any(x[0] == "insider_selling" for x in matches):
        matches.append(("insider_selling", 1))
    if not matches:
        return "other", 0.35, []
    matches.sort(key=lambda x: x[1], reverse=True)
    category, count = matches[0]
    confidence = min(0.55 + 0.15 * (count - 1), 0.90)
    return category, confidence, [x[0] for x in matches]

def classify_event(
    title: str,
    description: str = "",
    transaction_code: str = "",
    use_llm: bool | None = None
) -> EventClassificationResult:
    """Classify corporate financial events.

    Gemini LLM is disabled by default (use_llm=False or unset) and must be explicitly enabled
    via parameter (use_llm=True) or USE_GEMINI_CLASSIFIER=true environment variable.
    """
    if use_llm is None:
        enable_llm = os.getenv("USE_GEMINI_CLASSIFIER", "").strip().lower() in {"1", "true", "yes"}
    else:
        enable_llm = bool(use_llm)

    if enable_llm:
        from feature_engineering.gemini_classifier import classify_event_with_gemini
        llm_res = classify_event_with_gemini(title=title, description=description)
        if llm_res is not None:
            category = llm_res["event_category"]
            matches = [category] + [c for c in llm_res.get("secondary_categories", []) if c != category]
            if transaction_code.upper() == "S" and "insider_selling" not in matches:
                matches.append("insider_selling")
            return EventClassificationResult(
                event_type=category,
                confidence=llm_res["confidence"],
                all_detected=matches,
                explanation=llm_res.get("explanation", ""),
                evidence=llm_res.get("evidence", []),
                source="gemini_llm",
                is_calibrated=False
            )

    # Deterministic regex fallback
    cat, conf, matches = classify_event_rule_based(title, description, transaction_code)
    return EventClassificationResult(
        event_type=cat,
        confidence=conf,
        all_detected=matches,
        explanation="Classified via deterministic keyword heuristics.",
        evidence=[],
        source="rule_based_fallback",
        is_calibrated=False
    )
