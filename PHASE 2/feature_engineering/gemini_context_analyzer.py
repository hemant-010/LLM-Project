"""Gemini-powered financial context and severity analyzer using the official google-genai SDK.

Architectural Principles:
1. Analyzes corporate event context, supporting evidence, risk factors, and severity.
2. Uses Gemini structured JSON output via Pydantic schema.
3. For insider selling: evaluates executive role, transaction value, percentage sold,
   and 10b5-1 plan status without assuming every sale is negative.
4. Preserves unknown values; never assumes missing information implies negative signals.
5. Verifies all supporting evidence strings are verbatim substrings of input text.
6. Completely decoupled from rule-based fallback logic.
"""
import os
import time
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError
from config import ROOT
from feature_engineering.gemini_classifier import get_gemini_client, verify_evidence_substrings

load_dotenv(ROOT / ".env")

SeverityLevel = Literal["low", "medium", "high", "critical", "unknown"]

class GeminiContextAnalysisResponse(BaseModel):
    severity: SeverityLevel = Field(
        description="Assessed event severity from the allowed levels: low, medium, high, critical, unknown."
    )
    severity_rationale: str = Field(
        description="Clear reasoning explaining why this severity level was assigned based strictly on the evidence."
    )
    context_summary: str = Field(
        description="Concise synthesis of the corporate context and circumstances."
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Specific financial, operational, governance, or market risk factors identified."
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim quote(s) or key phrases extracted as exact substrings from the text."
    )
    insider_nuance: str = Field(
        default="",
        description="Specific analysis of insider transaction characteristics (role, holding %, size, 10b5-1 status)."
    )

def analyze_context_with_gemini(
    row: dict,
    metrics: dict,
    model: str | None = None,
    max_retries: int = 2
) -> dict | None:
    """Analyze event context and severity using Gemini LLM with structured output."""
    client = get_gemini_client()
    if client is None:
        return None

    title = str(row.get("title") or "").strip()
    description = str(row.get("description") or "").strip()
    full_text = f"{title} {description}".strip()
    if not full_text:
        return None

    if model is None:
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    event_type = row.get("event_type", "other")
    pct_sold = metrics.get("percentage_holdings_sold")
    val = metrics.get("transaction_value")
    is_10b5 = metrics.get("insider_is_10b5_1")
    role = row.get("insider_role") or "Unspecified"

    prompt = f"""You are a senior financial analyst and forensic governance researcher.
Analyze the following corporate event context and evaluate its severity for investors.

Event Details:
- Title: {title}
- Description: {description}
- Event Type: {event_type}
- Classification Source: {row.get('classification_source', 'heuristic')}
- Classification Quotes: {row.get('classification_evidence') or 'None'}
- Insider Role: {role}
- Percentage of Holdings Sold: {pct_sold if pct_sold is not None else 'Unknown / Not provided'}
- Transaction Value ($): {f'${val:,.2f}' if val is not None else 'Unknown / Not provided'}
- Confirmed 10b5-1 Pre-planned Trading Plan: {'Yes (True)' if is_10b5 is True else ('No (False - Discretionary)' if is_10b5 is False else 'Unknown / Insufficient filing evidence')}
- Repeated Selling / Pattern: {row.get('repeated_selling', 'Not specified')}

Analytical Guidelines:
1. Severity Levels:
   - 'low': Routine administrative filings, modest planned sales, routine 10b5-1 trades, minor recalls, or immaterial notices.
   - 'medium': Meaningful analyst downgrades, moderate earnings misses, noticeable insider sales (10-25%), or notable product recalls.
   - 'high': Large unhedged discretionary insider selling (>25%) by key officers (CEO/CFO), repeated unhedged liquidation, severe earnings misses, formal SEC/DOJ fraud investigations, or credit distress.
   - 'critical': Imminent Chapter 11 bankruptcy filing, catastrophic fraud scandal, severe systemic insolvency, or total operational halt.
   - 'unknown': Context or holding denominator is insufficient to quantify financial impact.

2. Insider Selling Rules:
   - Do NOT assume every insider sale is negative.
   - Pre-planned Rule 10b5-1 sales reflect routine diversification or tax management and should generally be assessed as lower risk.
   - If holding percentage is unknown, state that the holding denominator is unavailable rather than assuming a large liquidation.
   - Consider executive role, transaction size, percentage of holdings sold, repeated selling, and 10b5-1 status.
   - Never assume missing data indicates fraud or panic selling.

3. Evidence Grounding:
   - Every item in 'supporting_evidence' MUST be an exact verbatim substring present in the text.
"""

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiContextAnalysisResponse,
                    temperature=0.0,
                )
            )
            raw_json = response.text
            parsed = GeminiContextAnalysisResponse.model_validate_json(raw_json)

            # Substring verification
            verified_evidence = verify_evidence_substrings(parsed.supporting_evidence, full_text)
            if row.get("classification_evidence"):
                class_verified = verify_evidence_substrings(row["classification_evidence"], full_text)
                for item in class_verified:
                    if item not in verified_evidence:
                        verified_evidence.append(item)

            flags = []
            if parsed.insider_nuance:
                flags.append(parsed.insider_nuance)
            if parsed.severity_rationale:
                flags.append(parsed.severity_rationale)
            if is_10b5 is True:
                flags.append("Transaction is marked as a 10b5-1 plan; interpret with additional context.")
            elif is_10b5 is False:
                flags.append("Transaction is confirmed not under a 10b5-1 plan.")
            elif is_10b5 is None and event_type == "insider_selling":
                flags.append("10b5-1 plan status is unknown from the extracted fields.")

            return {
                "severity": parsed.severity,
                "severity_rationale": parsed.severity_rationale,
                "context_summary": parsed.context_summary,
                "risk_factors": parsed.risk_factors,
                "supporting_evidence": verified_evidence,
                "insider_nuance": parsed.insider_nuance,
                "context_flags": " | ".join([f for f in flags if f])
            }

        except (APIError, Exception) as e:
            if attempt < max_retries:
                time.sleep(1.0 * (2 ** attempt))
                continue
            print(f"Notice: Gemini context analysis failed after {max_retries + 1} attempt(s): {e}")
            return None

    return None
