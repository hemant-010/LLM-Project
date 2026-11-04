"""Significant Insider Selling Analysis and Investor Notification Engine.

Principles:
1. Objective quantitative evaluation: evaluates holding percentage, transaction size, and officer role.
2. 10b5-1 Plan Awareness: pre-planned Rule 10b5-1 transactions reflect routine diversification/tax planning
   and are distinguished from opportunistic discretionary liquidations.
3. Categorization: differentiates 'significant', 'requires_review', and 'routine' transactions.
4. Non-fabrication & Missing Data Handling: preserves unknown denominators without assuming worst-case defaults.
5. Research Guardrails: explicit research disclaimers that insider sales are not automatic proof of distress or bearish signals.
"""
import re
from typing import Dict, Any, List, Optional
import math


def calculate_percentage_sold(
    shares_sold: Optional[float],
    shares_owned_before: Optional[float] = None,
    shares_owned_after: Optional[float] = None
) -> Optional[float]:
    """Calculate percentage of insider holdings liquidated in transaction."""
    if shares_sold is None or shares_sold <= 0:
        return None

    denom = None
    if shares_owned_before is not None and shares_owned_before > 0:
        denom = shares_owned_before
    elif shares_owned_after is not None and (shares_sold + shares_owned_after) > 0:
        denom = shares_sold + shares_owned_after

    if denom is not None and denom > 0:
        pct = (shares_sold / denom) * 100.0
        return round(pct, 2)
    return None


C_SUITE_PATTERN = re.compile(
    r"\b(ceo|chief executive|cfo|chief financial|coo|chief operating|cto|chief technology|president|chairman|founder|general counsel)\b",
    re.IGNORECASE
)

def is_c_suite_role(role: Optional[str]) -> bool:
    """Determine if insider role represents core corporate executive leadership."""
    if not role:
        return False
    return bool(C_SUITE_PATTERN.search(role))


def analyze_insider_transaction(tx: Dict[str, Any]) -> Dict[str, Any]:
    """Perform multi-dimensional significance analysis on an insider transaction.

    Returns:
    --------
    dict containing:
        - significance_category: 'significant' | 'requires_review' | 'routine'
        - percentage_holdings_sold: float | None
        - transaction_value: float | None
        - is_c_suite: bool
        - is_10b5_1: bool | None
        - risk_score: float (0.0 to 1.0)
        - risk_factors: list[str]
        - positive_factors: list[str]
        - summary: str
        - alert_title: str
    """
    shares = tx.get("shares") or tx.get("shares_sold")
    try:
        shares_sold = float(shares) if shares is not None else None
    except (ValueError, TypeError):
        shares_sold = None

    price = tx.get("price_per_share")
    try:
        price_per_share = float(price) if price is not None else None
    except (ValueError, TypeError):
        price_per_share = None

    val = tx.get("estimated_value") or tx.get("transaction_value")
    try:
        transaction_value = float(val) if val is not None else None
    except (ValueError, TypeError):
        transaction_value = None

    if transaction_value is None and shares_sold is not None and price_per_share is not None:
        transaction_value = round(shares_sold * price_per_share, 2)

    before = tx.get("shares_owned_before")
    after = tx.get("shares_owned_after")
    try:
        before_num = float(before) if before is not None else None
    except (ValueError, TypeError):
        before_num = None
    try:
        after_num = float(after) if after is not None else None
    except (ValueError, TypeError):
        after_num = None

    pct_sold = tx.get("percentage_holdings_sold")
    if pct_sold is None:
        pct_sold = calculate_percentage_sold(shares_sold, before_num, after_num)
    else:
        try:
            pct_sold = round(float(pct_sold), 2)
        except (ValueError, TypeError):
            pct_sold = calculate_percentage_sold(shares_sold, before_num, after_num)

    # 10b5-1 plan detection
    raw_10b5 = tx.get("mentions_10b5_1") if tx.get("mentions_10b5_1") is not None else tx.get("is_10b5_1")
    if isinstance(raw_10b5, bool):
        is_10b5_1 = raw_10b5
    elif raw_10b5 is None:
        is_10b5_1 = None
    else:
        s = str(raw_10b5).strip().lower()
        if s in {"true", "1", "yes", "y"}:
            is_10b5_1 = True
        elif s in {"false", "0", "no", "n"}:
            is_10b5_1 = False
        else:
            is_10b5_1 = None

    role = str(tx.get("insider_roles") or tx.get("insider_role") or "Unspecified Officer/Director")
    c_suite = is_c_suite_role(role)
    ticker = str(tx.get("ticker") or "UNKNOWN").upper()
    owner_name = str(tx.get("insider_names") or tx.get("insider_name") or "Corporate Insider")

    # Risk Factor Aggregation
    risk_factors = []
    positive_factors = []
    base_score = 0.20

    # 1. Holding percentage evaluation
    if pct_sold is not None:
        if pct_sold >= 50.0:
            base_score += 0.40
            risk_factors.append(f"Major equity disposition: insider liquidated {pct_sold:.1f}% of entire holding.")
        elif pct_sold >= 25.0:
            base_score += 0.25
            risk_factors.append(f"Substantial holding reduction: insider liquidated {pct_sold:.1f}% of holding (>=25%).")
        elif pct_sold >= 10.0:
            base_score += 0.10
            risk_factors.append(f"Noticeable holding reduction: insider sold {pct_sold:.1f}% of holding (10-25%).")
        else:
            positive_factors.append(f"Modest holding reduction: sale represents only {pct_sold:.1f}% of total holding (<10%).")
    else:
        risk_factors.append("Holding denominator unavailable; percentage of equity liquidated cannot be verified.")

    # 2. Transaction value evaluation
    if transaction_value is not None:
        if transaction_value >= 10_000_000:
            base_score += 0.25
            risk_factors.append(f"High-magnitude transaction: total disposition value is ${transaction_value:,.2f} (>= $10M).")
        elif transaction_value >= 3_000_000:
            base_score += 0.15
            risk_factors.append(f"Substantial transaction value: ${transaction_value:,.2f} (>= $3M).")
        elif transaction_value < 500_000:
            positive_factors.append(f"Routine transaction value: ${transaction_value:,.2f} (< $500k).")
    else:
        risk_factors.append("Total transaction value could not be confirmed from filing fields.")

    # 3. Executive Role evaluation
    if c_suite:
        base_score += 0.15
        risk_factors.append(f"High-visibility executive: transaction executed by key officer ({role}).")
    else:
        positive_factors.append(f"Standard non-executive or outside director role: {role}.")

    # 4. 10b5-1 Plan Impact (Key mitigating factor)
    if is_10b5_1 is True:
        base_score -= 0.30
        positive_factors.append("Confirmed Rule 10b5-1 trading plan: transaction was pre-scheduled under SEC affirmative defense rules.")
    elif is_10b5_1 is False:
        base_score += 0.15
        risk_factors.append("Discretionary transaction: filing indicates trade was NOT under a Rule 10b5-1 pre-scheduled plan.")
    else:
        risk_factors.append("10b5-1 plan status unspecified in extracted filing footnotes.")

    # Normalize score between 0.05 and 0.95
    normalized_score = max(0.05, min(0.95, round(base_score, 2)))

    # Categorization Decision Rules
    # Significant: High pct (>=25%) or large value (>= $5M) by C-suite without 10b5-1, or extreme liquidation (>=50%)
    if (pct_sold is not None and pct_sold >= 40.0) or \
       (normalized_score >= 0.65) or \
       (c_suite and is_10b5_1 is False and transaction_value is not None and transaction_value >= 3_000_000):
        category = "significant"
        alert_title = f"SIGNIFICANT INSIDER SALE: [{ticker}] {owner_name} ({role})"
    elif normalized_score >= 0.40 or (pct_sold is not None and pct_sold >= 10.0) or (is_10b5_1 is False and c_suite):
        category = "requires_review"
        alert_title = f"INSIDER SALE (REVIEW RECOMMENDED): [{ticker}] {owner_name}"
    else:
        category = "routine"
        alert_title = f"ROUTINE INSIDER SALE: [{ticker}] {owner_name}"

    summary = (
        f"Insider {owner_name} ({role}) reported sale of "
        f"{f'{shares_sold:,.0f}' if shares_sold is not None else 'unspecified'} shares of {ticker} "
        f"{f'valued at ${transaction_value:,.2f}' if transaction_value is not None else ''}. "
        f"{f'Represents {pct_sold:.1f}% of total position. ' if pct_sold is not None else 'Total holding percentage unknown. '}"
        f"{'Executed pursuant to a pre-established Rule 10b5-1 plan.' if is_10b5_1 is True else ('Discretionary sale not marked as 10b5-1 plan.' if is_10b5_1 is False else 'Plan status unspecified.')}"
    )

    return {
        "ticker": ticker,
        "insider_name": owner_name,
        "insider_role": role,
        "is_c_suite": c_suite,
        "transaction_date": tx.get("transaction_date"),
        "filing_date": tx.get("filing_date"),
        "shares_sold": shares_sold,
        "price_per_share": price_per_share,
        "transaction_value": transaction_value,
        "shares_owned_after": after_num,
        "percentage_holdings_sold": pct_sold,
        "is_10b5_1": is_10b5_1,
        "filing_url": tx.get("filing_url") or "",
        "significance_category": category,
        "risk_score": normalized_score,
        "risk_factors": risk_factors,
        "positive_factors": positive_factors,
        "summary": summary,
        "alert_title": alert_title,
        "disclaimer": (
            "Research notification, not investment advice. Insider sales occur for routine tax planning, "
            "estate diversification, and scheduled compensation liquidity. Not indicative of corporate fraud or distress."
        )
    }
