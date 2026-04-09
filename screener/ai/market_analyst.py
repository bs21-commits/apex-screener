"""
AI market analyst — uses free LLM (Groq / Ollama) via llm_client.

For each ticker, synthesizes:
  1. WHY it moved today (news + filing + technicals)
  2. Entry timing signals (VWAP relationship, RSI zone, volume pattern)
  3. Next-day outlook (continuation vs fade probability)

Results are cached per ticker per market day so the LLM isn't called repeatedly.
"""

from __future__ import annotations

import logging
from datetime import date

from screener.ai.llm_client import chat, is_available, backend_name

logger = logging.getLogger("apex.analyst")

# ── Per-session cache: {ticker: analysis_dict} ───────────────────────────────
_cache: dict[str, dict] = {}
_cache_date: date | None = None


def _day_cache(ticker: str) -> dict | None:
    global _cache_date
    today = date.today()
    if _cache_date != today:
        _cache.clear()
        _cache_date = today
    return _cache.get(ticker)


def _set_cache(ticker: str, result: dict) -> None:
    _cache[ticker] = result


# ── Entry timing helper ───────────────────────────────────────────────────────
def entry_signals(price: float, vwap: float | None, rsi: float | None,
                  volume_ratio: float, change_pct: float, short_pct: float | None) -> dict:
    """
    Rule-based entry timing signals — no API call needed.
    Returns: zone, action, reason, entry_note
    """
    signals = []
    action = "WATCH"
    zone = "NEUTRAL"

    # RSI zone
    if rsi is not None:
        if rsi < 35:
            signals.append("RSI oversold — potential reversal setup")
            zone = "OVERSOLD"
        elif 40 <= rsi <= 65:
            signals.append(f"RSI {rsi:.0f} — healthy momentum zone, not extended")
            zone = "HEALTHY"
            action = "BUY_ZONE"
        elif 65 < rsi <= 78:
            signals.append(f"RSI {rsi:.0f} — extended but not overbought, trail stops tight")
            zone = "EXTENDED"
        elif rsi > 78:
            signals.append(f"RSI {rsi:.0f} — overbought, wait for pullback to VWAP")
            zone = "OVERBOUGHT"
            action = "WAIT"

    # VWAP relationship
    if vwap and price and vwap > 0:
        pct_above_vwap = ((price - vwap) / vwap) * 100
        if price >= vwap:
            signals.append(f"Price above VWAP (+{pct_above_vwap:.1f}%) — bulls in control")
            if action != "WAIT":
                action = "BUY_ZONE"
        else:
            signals.append(f"Price below VWAP ({pct_above_vwap:.1f}%) — caution, potential breakdown")
            if action == "BUY_ZONE":
                action = "WATCH"

    # Volume confirmation
    if volume_ratio >= 10:
        signals.append(f"Volume {volume_ratio:.0f}× average — institutional/retail surge, high conviction")
    elif volume_ratio >= 5:
        signals.append(f"Volume {volume_ratio:.0f}× average — solid confirmation")
    else:
        signals.append(f"Volume only {volume_ratio:.1f}× average — weak confirmation, higher risk")
        if action == "BUY_ZONE":
            action = "WATCH"

    # Short squeeze fuel
    if short_pct and short_pct > 25:
        signals.append(f"Short interest {short_pct:.0f}% — squeeze fuel, can accelerate fast")

    # Entry note
    entry_notes = {
        "BUY_ZONE": "Consider entry on pullback to VWAP or 9 EMA. Set stop below day's low.",
        "WAIT":     "Wait for RSI to cool below 70 or price to reclaim VWAP before entry.",
        "WATCH":    "Monitor for volume surge or VWAP reclaim before committing.",
        "OVERSOLD": "Look for reversal candle (hammer/engulfing) before entry.",
    }

    return {
        "zone":       zone,
        "action":     action,
        "signals":    signals,
        "entry_note": entry_notes.get(action, ""),
    }


# ── Fast rule-based watchlist (no API call) ──────────────────────────────────
def rule_based_analysis(ticker: str, change_pct: float, volume_ratio: float,
                        rsi: float | None, price: float) -> dict:
    """
    Instant no-API analysis for bulk watchlist population.
    Uses RSI + volume + momentum rules to classify tomorrow_action.
    Claude analysis (via analyse_ticker) upgrades this when user requests it.
    """
    rsi_val = float(rsi) if rsi is not None else 55.0  # assume neutral if missing
    vol     = float(volume_ratio or 1)
    chg     = float(change_pct or 0)

    # Determine tomorrow action
    if rsi_val > 83 or chg > 120:
        action    = "AVOID"
        outlook   = f"Likely extended after {chg:+.0f}% move. RSI {rsi_val:.0f} overbought — gap-fill risk next session."
        entry     = "Avoid chasing. Watch for 20-30% pullback before re-entry."
        conf      = "MEDIUM"
        why       = f"Ran {chg:+.0f}% on {vol:.0f}× volume. Technically overbought on RSI {rsi_val:.0f}."
    elif 40 <= rsi_val <= 68 and vol >= 10 and chg >= 15:
        action    = "BUY_AT_OPEN"
        outlook   = f"Strong setup: {chg:+.0f}% on {vol:.0f}× volume with RSI in healthy zone ({rsi_val:.0f}). Premarket continuation likely if sector/news holds."
        entry     = f"Buy at open if gap up holds with volume. Wait for first 5-min candle to close green, then enter. Stop: below open candle low."
        conf      = "HIGH" if vol >= 20 else "MEDIUM"
        why       = f"Momentum breakout: {chg:+.0f}% gain on {vol:.0f}× average volume. RSI {rsi_val:.0f} has room to run."
    elif 40 <= rsi_val <= 72 and vol >= 5:
        action    = "WATCH_OPEN"
        outlook   = f"{chg:+.0f}% mover on {vol:.0f}× volume. RSI {rsi_val:.0f} in healthy zone — monitor first 15-min candle at open."
        entry     = f"Wait for VWAP hold at open. Enter if first 15-min candle is green with volume above 1M."
        conf      = "MEDIUM"
        why       = f"Volume-driven momentum: {chg:+.0f}% on {vol:.0f}× average volume."
    elif rsi_val < 35:
        action    = "WATCH_OPEN"
        outlook   = f"RSI {rsi_val:.0f} suggests oversold bounce potential. Watch for reversal candle."
        entry     = f"Look for hammer/engulfing candle on daily. Enter above yesterday's high."
        conf      = "LOW"
        why       = f"Possible oversold bounce after extended selloff. RSI {rsi_val:.0f}."
    else:
        action    = "AVOID"
        outlook   = f"Weak confirmation: {chg:+.0f}% on only {vol:.1f}× volume. No high-conviction setup."
        entry     = "Skip — volume insufficient for reliable follow-through."
        conf      = "LOW"
        why       = f"Low-volume {chg:+.0f}% move. Could be thin float or news fade."

    catalyst = ("TECHNICAL" if not any(k in why.lower() for k in ["news","filing","sec"])
                else "NEWS")

    risks = []
    if rsi_val > 70:    risks.append(f"RSI {rsi_val:.0f} — overbought, reversal risk")
    if vol < 5:         risks.append("Low volume ratio — breakout may not hold")
    if chg > 50:        risks.append("Large single-day move — gap-fill common next session")
    if not risks:       risks.append("Monitor pre-market volume for continuation signal")

    return {
        "why_moved":        why,
        "catalyst_type":    catalyst,
        "tomorrow_outlook": outlook,
        "tomorrow_action":  action,
        "ideal_entry":      entry,
        "confidence":       conf,
        "risk_factors":     risks,
        "_rule_based":      True,   # flag so UI can show "upgrade" button
    }


def auto_populate_watchlist(gainers: list[dict], top_n: int = 30) -> dict[str, dict]:
    """
    Run rule-based analysis on top_n gainers and return {ticker: analysis_dict}.
    Called automatically during scan so Tomorrow's Watchlist is pre-populated.
    Only processes gainers not already in the day cache.
    """
    results: dict[str, dict] = {}
    # Score: vol_ratio * change_pct — highest conviction first
    scored = sorted(gainers, key=lambda g: float(g.get("volume_ratio", 1)) * float(g.get("change_pct", 0)), reverse=True)
    for g in scored[:top_n]:
        t = g.get("ticker", "")
        if not t:
            continue
        cached = _day_cache(t)
        if cached:
            results[t] = cached
            continue
        analysis = rule_based_analysis(
            ticker       = t,
            change_pct   = float(g.get("change_pct", 0)),
            volume_ratio = float(g.get("volume_ratio", 1)),
            rsi          = g.get("rsi"),
            price        = float(g.get("price", 0)),
        )
        _set_cache(t, analysis)
        results[t] = analysis
    return results


# ── Claude: Why did it move + next-day outlook ────────────────────────────────
def analyse_ticker(ticker: str, change_pct: float, volume_ratio: float,
                   float_M: float | None, short_pct: float | None,
                   news_headlines: list[str], filing_summary: str = "") -> dict:
    """
    Ask Claude:
      1. Why did this stock move today?
      2. Is this continuation-worthy tomorrow?
      3. What's the risk?

    Returns dict with: why_moved, tomorrow_outlook, tomorrow_action,
                       confidence, risk_factors, catalyst_type
    """
    cached = _day_cache(ticker)
    if cached:
        return cached

    headlines_text = "\n".join(f"- {h}" for h in news_headlines[:8]) if news_headlines else "No news found."
    float_text = f"{float_M:.1f}M shares" if float_M else "unknown"
    si_text    = f"{short_pct:.1f}%" if short_pct else "unknown"

    prompt = f"""You are a professional day-trading analyst specializing in low-float momentum stocks.

Ticker: {ticker}
Today's move: {change_pct:+.1f}%
Volume: {volume_ratio:.1f}× average
Float: {float_text}
Short interest: {si_text}

Recent news headlines:
{headlines_text}

SEC filing context (if any):
{filing_summary or "No recent filing."}

Answer in JSON with exactly these keys:
{{
  "why_moved": "2-3 sentence explanation of what drove today's move",
  "catalyst_type": "ONE of: SEC_FILING | NEWS | SQUEEZE | TECHNICAL | UNKNOWN",
  "tomorrow_outlook": "2-3 sentence assessment of continuation potential for tomorrow",
  "tomorrow_action": "ONE of: BUY_AT_OPEN | WATCH_OPEN | AVOID | SHORT_CANDIDATE",
  "ideal_entry": "specific price action or level to watch for entry tomorrow (e.g. 'gap up above $X with volume', 'pullback to $Y VWAP')",
  "confidence": "HIGH | MEDIUM | LOW",
  "risk_factors": ["risk1", "risk2"]
}}

Rules:
- tomorrow_action BUY_AT_OPEN only if strong catalyst + float <10M + did NOT dump >40% from high today
- AVOID if news is one-day pump, no real catalyst, or stock >80% RSI with no short squeeze thesis
- SHORT_CANDIDATE if big gap up on no real catalyst, high dilution risk
- Return ONLY the JSON."""

    try:
        result = chat(prompt, max_tokens=600, expect_json=True)
        _set_cache(ticker, result)
        return result
    except Exception as exc:
        logger.debug(f"[analyst] {ticker}: {exc}")
        fallback = {
            "why_moved":        f"{change_pct:+.1f}% move on {volume_ratio:.0f}× volume. Catalyst unclear — check news.",
            "catalyst_type":    "UNKNOWN",
            "tomorrow_outlook": "Insufficient data for Claude analysis. Monitor pre-market volume and news.",
            "tomorrow_action":  "WATCH_OPEN",
            "ideal_entry":      "Wait for first 15-min candle to close, then assess VWAP relationship.",
            "confidence":       "LOW",
            "risk_factors":     ["No Claude analysis available", "Unknown catalyst"],
        }
        _set_cache(ticker, fallback)
        return fallback
