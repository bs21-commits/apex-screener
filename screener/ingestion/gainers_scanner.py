"""
Gainers Scanner — real-time low-float momentum detection.

Polls Polygon's top-gainers snapshot every scan cycle and runs each mover
through the same float + volume filters as the EDGAR pipeline, then generates
a scored setup dict identical in shape to the SEC-filing setups so the
dashboard can display both feeds in one unified table.

This is how stocks like UCAR (+444%) surface DURING the move, not after.
"""

from __future__ import annotations
import logging
try:
    from screener.ingestion.openbb_client import get_gainers
except Exception:
    from screener.ingestion.polygon_client import get_gainers
from screener.ingestion.finviz_client   import get_float_data, is_low_float
from screener.scoring.engine            import score_setup
from screener.config                    import MAX_FLOAT_SHARES, VOLUME_SPIKE_THRESHOLD

logger = logging.getLogger("apex.gainers")


def momentum_scenarios(price: float, float_shares: float | None,
                       volume_ratio: float, short_pct: float | None) -> dict:
    """
    Generate 3 intraday price scenarios based on float size, volume intensity,
    and short interest.  Returns {conservative, base, aggressive, basis}.

    Logic:
      float_mult   — smaller float → larger price impact per dollar of buying
      vol_mult     — higher vol ratio → more sustained buying pressure
      squeeze_mult — high short % adds covering fuel on top of organic buying
    """
    float_M    = (float_shares or 5_000_000) / 1_000_000
    float_mult = max(0.5, min(4.0, 8.0 / float_M))          # 2M float → 4×, 8M → 1×
    vol_mult   = min(2.5, (volume_ratio or 1) / 8)           # 20x vol → 2.5 mult
    si         = (short_pct or 0) / 100                      # decimal

    # Additional squeeze fuel when short interest > 20%
    squeeze_bonus = max(0.0, (si - 0.20) * 3)               # 30% SI → +0.30 bonus

    conservative = round(price * (1 + 0.08 * float_mult), 2)
    base         = round(price * (1 + 0.20 * float_mult * vol_mult), 2)
    aggressive   = round(price * (1 + 0.45 * float_mult * vol_mult * (1 + squeeze_bonus)), 2)

    return {
        "conservative": conservative,
        "base":         base,
        "aggressive":   aggressive,
        "basis":        "low-float squeeze model",
    }


def scan_gainers(halt_tickers: list[str] | None = None,
                 min_change_pct: float = 5.0) -> list[dict]:
    """
    Fetch today's top % gainers, filter to low-float names,
    score them, and return as setup dicts.

    Parameters
    ----------
    halt_tickers    : list of tickers with active LULD halts
    min_change_pct  : ignore stocks with less than this % gain today
    """
    halt_tickers = halt_tickers or []
    raw_gainers  = get_gainers(min_change_pct=min_change_pct)
    setups: list[dict] = []

    for quote in raw_gainers:
        ticker    = quote.get("ticker", "")
        vol_ratio = float(quote.get("volume_ratio") or 0)

        if vol_ratio < VOLUME_SPIKE_THRESHOLD:
            logger.debug(f"[gainers] {ticker} skip — vol_ratio {vol_ratio:.1f}x < {VOLUME_SPIKE_THRESHOLD}x")
            continue

        float_data = get_float_data(ticker)
        if not is_low_float(float_data):
            logger.debug(f"[gainers] {ticker} skip — not low float")
            continue

        # Score using same engine as EDGAR pipeline (llm_result=None → neutral LLM sub)
        llm_result = {
            "catalyst_type":  "MOMENTUM",
            "sentiment":      "BULLISH",
            "catalyst_score": 15,          # base score for pure price momentum
            "toxicity_flags": [],
            "bullish_signals": ["high_relative_volume", "low_float_squeeze"],
            "dilution_risk":  "NONE",
            "summary":        f"Pure momentum mover — {quote['change_pct']:+.1f}% today on {vol_ratio:.1f}x vol. No SEC catalyst detected; price action driven by volume and float dynamics.",
        }

        setup = score_setup(quote, float_data, llm_result, halt_tickers=halt_tickers)

        # Momentum scenario targets
        setup["scenarios"] = momentum_scenarios(
            price        = float(quote.get("price") or 0),
            float_shares = float_data.get("float_shares"),
            volume_ratio = vol_ratio,
            short_pct    = float_data.get("short_float_pct"),
        )

        setup["source"]       = "gainers"          # distinguish from EDGAR setups
        setup["company_name"] = ticker
        setup["filing_url"]   = ""
        setup["filed_at"]     = ""
        setup["form_type"]    = "MOMENTUM"

        float_M = (float_data.get('float_shares') or 0) / 1e6
        logger.info(
            f"[gainers] {ticker}  chg={quote['change_pct']:+.1f}%  "
            f"vol={vol_ratio:.1f}x  float={float_M:.1f}M  "
            f"score={setup['score']}  "
            f"bull={setup['scenarios']['base']}  bear={setup['scenarios']['conservative']}"
        )
        setups.append(setup)

    return setups
