"""Gemini-powered financial event classifier using the official google-genai SDK.

Architectural Principles:
1. Grounds classification strictly on input text; rejects ungrounded evidence.
2. Returns structured JSON containing category, confidence, explanation, and evidence.
3. Completely separate from rule-based classification logic.
4. Handles rate limits, timeouts, API errors, and missing keys gracefully.
"""
import os
import time
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError
from config import ROOT, EVENT_TYPES

load_dotenv(ROOT / ".env")

EventCategory = Literal[
    "insider_selling",
    "analyst_downgrade",
    "earnings_miss",
    "fraud_investigation",
    "product_recall",
    "merger_acquisition",
    "bankruptcy_distress",
    "positive_earnings",
    "other"
]

class GeminiClassificationResponse(BaseModel):
    event_category: EventCategory = Field(
        description="The primary corporate event category best matching the text from the allowed list."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Uncalibrated model confidence score between 0.0 and 1.0 reflecting how clearly the headline matches."
    )
    explanation: str = Field(
        description="Concise rationale explaining why this category was selected based strictly on the text."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Exact quote(s) or key phrases extracted verbatim as substrings from the input text supporting the classification."
    )
    secondary_categories: list[EventCategory] = Field(
        default_factory=list,
        description="Any secondary event categories detected from the allowed list."
    )

def get_gemini_client() -> genai.Client | None:
    """Initialize and return google-genai Client if GEMINI_API_KEY is configured."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    placeholders = ["your-api-key", "your_actual_api_key", "your_api_key", "placeholder", "example"]
    if not api_key or any(p in api_key.lower() for p in placeholders):
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini client: {e}")
        return None

def verify_evidence_substrings(evidence: list[str], full_text: str) -> list[str]:
    """Verify that all evidence phrases are exact verbatim substrings of the input text."""
    lower_full = full_text.lower()
    verified = []
    for phrase in evidence:
        p = phrase.strip()
        if p and p.lower() in lower_full:
            verified.append(p)
    return verified

def classify_event_with_gemini(
    title: str,
    description: str = "",
    model: str | None = None,
    max_retries: int = 2,
    timeout_seconds: float = 10.0
) -> dict | None:
    """Classify corporate event using Gemini LLM with strict grounding and structured output.

    Returns dict with keys: event_category, confidence, explanation, evidence,
    secondary_categories, or None on failure (triggering fallback).
    """
    if model is None:
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    client = get_gemini_client()
    if client is None:
        return None

    full_text = f"{title} {description}".strip()
    if not full_text:
        return None

    prompt = f"""You are a financial NLP intelligence system classifying corporate events for academic research.
Allowed categories:
- insider_selling
- analyst_downgrade
- earnings_miss
- fraud_investigation
- product_recall
- merger_acquisition
- bankruptcy_distress
- positive_earnings
- other

Grounding and Evidence Rules:
1. Base your classification strictly on the provided text. Never assume or fabricate unstated facts.
2. If the text denies, refutes, or rumors an event without confirmation, classify as 'other'.
3. Every item in 'evidence' MUST be an exact verbatim substring present in the text.
4. If no event clearly matches, output 'other' with an explanation.

Text to classify:
Title: {title}
Description: {description}
"""

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiClassificationResponse,
                    temperature=0.0,
                )
            )
            raw_json = response.text
            parsed = GeminiClassificationResponse.model_validate_json(raw_json)

            # Strict substring verification of evidence quotes
            verified_evidence = verify_evidence_substrings(parsed.evidence, full_text)

            # Anti-fabrication check: if category is not 'other', require at least one verified substring
            if parsed.event_category != "other" and not verified_evidence:
                print(f"Notice: Gemini categorized as '{parsed.event_category}' but evidence was not found verbatim in input text. Rejecting classification as ungrounded.")
                return None

            return {
                "event_category": parsed.event_category,
                "confidence": round(float(parsed.confidence), 4),
                "explanation": parsed.explanation,
                "evidence": verified_evidence,
                "secondary_categories": [c for c in parsed.secondary_categories if c in EVENT_TYPES],
                "is_calibrated": False
            }

        except (APIError, Exception) as e:
            if attempt < max_retries:
                time.sleep(1.0 * (2 ** attempt))
                continue
            print(f"Notice: Gemini classification failed after {max_retries + 1} attempt(s): {e}")
            return None

    return None

