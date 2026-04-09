"""
APEX Pattern Scorer — composite 0-100 scoring engine.

Combines three independent data signals into a single actionable score:

  ┌─────────────────────────────────────────────────────────┐
  │  Signal           Weight   Source                       │
  │  ─────────────────────────────────────────────────────  │
  │  Volume spike     0 – 30   Polygon intraday vol ratio   │
  │  Float quality    0 – 20   Finviz float / shares out    │
  │  LLM catalyst     0 – 40   Claude filing classification  │
  │  Bonus / halts    0 – 10   Halts, gap, short interest   │
  │  ─────────────────────────────────────────────────────  │
  │  TOTAL            0 – 100                               │
  └─────────────────────────────────────────────────────────┘

Example 90+ setup:
  Float < 3M (+26)  +  Volume > 15x (+26)  +  Clean catalyst LLM=38 (+38)
  + LULD halt (+5)  + premarket gap (+4)   = 99 (capped at 100)

Toxic deductions applied directly to the LLM sub-score mean an S-3 ATM
filing can push the LLM contribution negative (floored at 0), making a
90+ composite score impossible regardless of how high volume spikes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


# ── Volume sub-score: 0-30 ────────────────────────────────────────────────────
_VOLUME_TIERS = [
    (20.0, 30),
    (15.0, 26),
    (10.0, 21),
     (7.0, 15),
     (5.0,  9),
]

def _volume_sub(vol_ratio: float) -> int:
    for threshold, pts in _VOLUME_TIERS:
        if vol_ratio >= threshold:
            return pts
    return 0


# ── Float sub-score: 0-20 ─────────────────────────────────────────────────────
_FLOAT_TIERS = [
    (1_000_000,  20),
    (3_000_000,  17),
    (5_000_000,  14),
    (10_000_000, 10),
    (15_000_000,  6),
    (20_000_000,  2),
]

def _float_sub(float_shares: float | None) -> int:
    if float_shares is None:
        return 0   # unknown float → no contribution
    for threshold, pts in _FLOAT_TIERS:
        if float_shares < threshold:
            return pts
    return 0


# ── LLM sub-score: 0-40 (deductions applied for toxicity) ────────────────────
_TOXICITY_DEDUCTIONS: dict[str, int] = {
    "DILUTION_DEATH_SPIRAL":   20,
    "ATM_OFFERING":            15,
    "TOXIC_CONVERTIBLE":       12,
    "DISCOUNTED_PIPE":         10,
    "GOING_CONCERN":           10,
    "SHELF_REGISTRATION":       8,
    "WARRANT_RESET":            8,
    "WARRANT_EXERCISE":         6,
    "RESALE_REGISTRATION":      5,
}

_DILUTION_PENALTY = {
    "SEVERE": 10,
    "HIGH":    5,
    "MEDIUM":  2,
    "LOW":     0,
    "NONE":    0,
}

def _llm_sub(llm_result: dict[str, Any]) -> int:
    base     = int(llm_result.get("catalyst_score", 0) or 0)
    base     = max(0, min(40, base))   # clamp to declared range

    # Deduct for each detected toxic flag
    deduction = sum(
        _TOXICITY_DEDUCTIONS.get(flag, 3)   # default 3 pts for unknown flags
        for flag in (llm_result.get("toxicity_flags") or [])
    )

    # Additional dilution penalty
    deduction += _DILUTION_PENALTY.get(
        llm_result.get("dilution_risk", "NONE"), 0
    )

    # Bearish LLM sentiment: additional -8
    if llm_result.get("sentiment") == "BEARISH":
        deduction += 8

    return max(0, base - deduction)


# ── Bonus sub-score: 0-10 ─────────────────────────────────────────────────────
def _bonus_sub(
    quote: dict[str, Any],
    float_data: dict[str, Any],
    is_halted: bool = False,
) -> int:
    pts = 0

    # Active LULD halt is strong evidence of extreme momentum
    if is_halted:
        pts += 5

    # Pre-market / intraday gap
    premarket = float(quote.get("premarket_pct") or 0)
    if premarket >= 30:
        pts += 4
    elif premarket >= 15:
        pts += 2
    elif premarket >= 5:
        pts += 1

    # Short squeeze fuel
    short_pct = float(float_data.get("short_float_pct") or 0)
    if short_pct >= 30:
        pts += 3
    elif short_pct >= 20:
        pts += 2

    # Raw intraday momentum
    change_pct = float(quote.get("change_pct") or 0)
    if change_pct >= 50:
        pts += 2
    elif change_pct >= 25:
        pts += 1

    return min(pts, 10)   # cap bonus at 10


# ── Public API ────────────────────────────────────────────────────────────────

def score_setup(
    quote: dict[str, Any],
    float_data: dict[str, Any],
    llm_result: dict[str, Any],
    halt_tickers: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute the composite 0-100 score for a single setup.

    Parameters
    ----------
    quote        : polygon_client.get_quote() output
    float_data   : finviz_client.get_float_data() output
    llm_result   : filing_parser.classify_filing() output
    halt_tickers : list of tickers currently under LULD halt

    Returns a ready-to-render setup dict including score_breakdown.
    """
    ticker = quote.get("ticker", "???")

    is_halted = bool(halt_tickers and ticker in halt_tickers)

    vol_sub   = _volume_sub(float(quote.get("volume_ratio") or 0))
    fl_sub    = _float_sub(float_data.get("float_shares"))
    llm_s     = _llm_sub(llm_result)
    bonus     = _bonus_sub(quote, float_data, is_halted)

    raw_score = vol_sub + fl_sub + llm_s + bonus
    score     = max(0, min(100, raw_score))

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "ticker":         ticker,
        "company_name":   "",                       # filled by caller if available
        "filing_url":     "",                       # filled by caller if available
        "filed_at":       "",                       # filled by caller if available
        # ── Market data ───────────────────────────────────────────────────────
        "price":          quote.get("price"),
        "change_pct":     quote.get("change_pct"),
        "premarket_pct":  quote.get("premarket_pct"),
        "volume":         quote.get("volume"),
        "volume_ratio":   quote.get("volume_ratio"),
        "vwap":           quote.get("vwap"),
        "is_halted":      is_halted,
        # ── Float data ────────────────────────────────────────────────────────
        "float_shares":   float_data.get("float_shares"),
        "shares_out":     float_data.get("shares_out"),
        "short_pct":      float_data.get("short_float_pct"),
        "days_to_cover":  float_data.get("days_to_cover"),
        # ── LLM output ────────────────────────────────────────────────────────
        "catalyst_type":  llm_result.get("catalyst_type"),
        "sentiment":      llm_result.get("sentiment"),
        "dilution_risk":  llm_result.get("dilution_risk"),
        "toxicity_flags": llm_result.get("toxicity_flags", []),
        "bullish_signals":llm_result.get("bullish_signals", []),
        "llm_summary":    llm_result.get("summary"),
        # ── Score ─────────────────────────────────────────────────────────────
        "score":          score,
        "score_breakdown": {
            "volume_sub": vol_sub,
            "float_sub":  fl_sub,
            "llm_sub":    llm_s,
            "bonus_sub":  bonus,
        },
    }


def score_dataframe(setups: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of scored setup dicts into a sorted, display-ready DataFrame.
    Expands the score_breakdown dict into individual columns.
    """
    if not setups:
        return pd.DataFrame()

    df = pd.DataFrame(setups)

    # Expand score_breakdown sub-dict into flat columns
    if "score_breakdown" in df.columns:
        breakdown = pd.json_normalize(df["score_breakdown"])
        df = pd.concat(
            [df.drop(columns=["score_breakdown"]), breakdown],
            axis=1,
        )

    # Expand list columns to pipe-delimited strings for display
    for col in ("toxicity_flags", "bullish_signals"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: "|".join(x) if isinstance(x, list) else (x or "")
            )

    return df.sort_values("score", ascending=False).reset_index(drop=True)
