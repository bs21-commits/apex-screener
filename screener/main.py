"""
APEX Low-Float Screener — main orchestration loop.

Run:  python -m screener.main

Each cycle:
  1. Fetch active LULD halts (Polygon)
  2. Poll EDGAR for new 8-K / 6-K / S-1 / S-3 / F-3 filings
  3. For each filing: float filter → volume filter → LLM classify → score
  4. Log any setup scoring >= HIGH_SCORE_THRESHOLD
  5. Retroactively fill T+1hr / T+24hr prices in the alert log

Set USE_MOCK_POLYGON=false and USE_MOCK_FINVIZ=false in .env to go live.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any

from screener.config import (
    EDGAR_POLL_INTERVAL_SEC,
    HIGH_SCORE_THRESHOLD,
    MAX_FLOAT_SHARES,
    VOLUME_SPIKE_THRESHOLD,
)
from screener.ingestion.edgar_client   import poll_once, get_mock_filings, fetch_filing_text
try:
    from screener.ingestion.openbb_client  import get_quote, get_luld_halts
except Exception:
    from screener.ingestion.polygon_client import get_quote, get_luld_halts
from screener.ingestion.finviz_client  import get_float_data, is_low_float
from screener.ingestion.gainers_scanner import scan_gainers
from screener.ai.filing_parser         import classify_filing
from screener.scoring.engine           import score_setup, score_dataframe
from screener.logger.backtest_logger   import log_alert, update_prices, backtest_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("apex.main")


def process_filing(filing: dict, halt_tickers: list[str]) -> dict | None:
    """
    Full single-filing pipeline:
      1. Float filter  — skip if float > MAX_FLOAT_SHARES
      2. Volume filter — skip if vol_ratio < VOLUME_SPIKE_THRESHOLD
      3. LLM classify  — send filing text to Claude
      4. Score         — composite 0-100
      5. Log           — write to CSV if score >= HIGH_SCORE_THRESHOLD

    Returns the scored setup dict, or None when filtered out.
    """
    ticker = filing.get("ticker", "")
    if not ticker:
        logger.debug(f"[pipeline] no ticker on filing — {filing.get('company_name')}")
        return None

    # ── Step 1: float filter ─────────────────────────────────────────────────
    float_data = get_float_data(ticker)
    if not is_low_float(float_data):
        logger.debug(
            f"[pipeline] {ticker} SKIP — float "
            f"{(float_data.get('float_shares') or 0):,.0f} > {MAX_FLOAT_SHARES:,}"
        )
        return None

    # ── Step 2: volume filter ────────────────────────────────────────────────
    quote = get_quote(ticker)
    vol_ratio = float(quote.get("volume_ratio") or 0)
    if vol_ratio < VOLUME_SPIKE_THRESHOLD:
        logger.debug(
            f"[pipeline] {ticker} SKIP — vol_ratio {vol_ratio:.1f}x "
            f"< {VOLUME_SPIKE_THRESHOLD}x"
        )
        return None

    # ── Step 3: LLM classification ───────────────────────────────────────────
    if not filing.get("full_text"):
        filing["full_text"] = fetch_filing_text(filing.get("filing_url", ""))

    llm_result = classify_filing(
        filing["full_text"],
        ticker=ticker,
        form_type=filing.get("form_type", ""),
        filing_url=filing.get("filing_url", ""),
    )

    # ── Step 4: composite score ──────────────────────────────────────────────
    setup = score_setup(quote, float_data, llm_result, halt_tickers=halt_tickers)

    # Attach filing metadata for the dashboard and logger
    setup["company_name"] = filing.get("company_name", "")
    setup["filing_url"]   = filing.get("filing_url", "")
    setup["filed_at"]     = filing.get("filed_at", "")
    setup["form_type"]    = filing.get("form_type", "")

    logger.info(
        f"[pipeline] {ticker}  score={setup['score']}  "
        f"vol={vol_ratio:.1f}x  float={setup['float_shares']:,.0f}  "
        f"halt={'YES' if setup['is_halted'] else 'no'}  "
        f"catalyst={setup['catalyst_type']}  sentiment={setup['sentiment']}  "
        f"dilution={setup['dilution_risk']}"
    )

    # ── Step 5: log high-scoring alerts ─────────────────────────────────────
    log_alert(setup, filing)

    return setup


def run_scan_cycle(use_mock: bool = True) -> list[dict[str, Any]]:
    """
    Execute one full scan cycle and return all scored setups that passed filters.

    Parameters
    ----------
    use_mock : if True, uses 3 pre-baked sample filings instead of live EDGAR
    """
    # ── Halt check ───────────────────────────────────────────────────────────
    halts        = get_luld_halts()
    halt_tickers = [h["ticker"] for h in halts]
    if halt_tickers:
        logger.info(f"[halts] {len(halts)} active — {halt_tickers}")

    # ── Filing poll ──────────────────────────────────────────────────────────
    filings = get_mock_filings() if use_mock else poll_once()
    setups: list[dict] = []

    for filing in filings:
        setup = process_filing(filing, halt_tickers)
        if setup:
            setups.append(setup)

    # ── Gainers scan (runs every cycle alongside EDGAR) ──────────────────────
    gainer_setups = scan_gainers(halt_tickers=halt_tickers)
    setups.extend(gainer_setups)
    if gainer_setups:
        logger.info(f"[gainers] {len(gainer_setups)} low-float movers added to results")

    # ── T+1hr / T+24hr price fill ────────────────────────────────────────────
    filled = update_prices(get_quote)
    if filled:
        logger.info(f"[logger] filled {filled} price cells")

    n_high = sum(1 for s in setups if s["score"] >= HIGH_SCORE_THRESHOLD)
    logger.info(
        f"[scan] done — {len(filings)} filings + {len(gainer_setups)} gainers, "
        f"{len(setups)} total passed filters, "
        f"{n_high} high-score alerts (>= {HIGH_SCORE_THRESHOLD})"
    )
    return setups


def run_forever(use_mock: bool = True) -> None:
    """Main loop — polls on EDGAR_POLL_INTERVAL_SEC cadence."""
    logger.info(
        f"APEX Low-Float Screener v2  |  "
        f"mock={'ON' if use_mock else 'OFF'}  |  "
        f"poll_interval={EDGAR_POLL_INTERVAL_SEC}s"
    )
    cycle = 0
    while True:
        cycle += 1
        logger.info(f"── Cycle {cycle} ── {datetime.now().strftime('%H:%M:%S')} ──")
        run_scan_cycle(use_mock=use_mock)

        # Print backtest summary every 10 cycles
        if cycle % 10 == 0:
            summary = backtest_summary()
            logger.info(f"[backtest] {summary}")

        logger.info(f"[main] sleeping {EDGAR_POLL_INTERVAL_SEC}s…")
        time.sleep(EDGAR_POLL_INTERVAL_SEC)


if __name__ == "__main__":
    run_forever(use_mock=False)
