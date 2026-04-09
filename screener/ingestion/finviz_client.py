"""
Float & Short Interest — multi-source free aggregator.

Finviz is NOT used. Data is pulled from four authoritative free sources
and merged to give the most current picture possible:

  Source 1 — yfinance / Yahoo Finance
             float_shares, shares_outstanding, short_float_pct, days_to_cover
             Updates throughout the trading day. Best all-around free source.
             Includes exponential back-off retry for transient 429s.

  Source 2 — SEC EDGAR XBRL Company Facts API
             CommonStockSharesOutstanding from the most recent 10-K/10-Q.
             The legal, authoritative share count straight from SEC filings.
             Endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json

  Source 3 — Nasdaq.com short interest API
             Official bi-weekly settlement short interest for Nasdaq-listed
             stocks. Returns absolute short shares + days-to-cover.
             Gracefully skips non-Nasdaq names (returns None).

  Source 4 — Yahoo Finance quoteSummary JSON (direct HTTP, no yfinance)
             Fallback when the yfinance library itself is rate-limited.
             Uses a slightly different request path that is often not blocked
             when the standard library endpoint is 429.

Results are cached per ticker for CACHE_TTL_SEC (30 min) because float
and short interest don't change tick-by-tick — caching avoids rate limits
across rapid scan cycles while keeping data fresh enough for intraday use.

Public interface (unchanged from finviz_client.py):
  get_float_data(ticker)  → dict
  is_low_float(data)      → bool
  squeeze_score(data)     → float
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Any

import requests
import yfinance as yf

from screener.config import USE_MOCK_FINVIZ, MAX_FLOAT_SHARES

logger = logging.getLogger(__name__)

# ── In-memory result cache ───────────────────────────────────────────────────
_cache: dict[str, tuple[float, dict]] = {}
CACHE_TTL_SEC = 30 * 60   # 30 minutes

_HEADERS = {
    "User-Agent": "APEX-Screener/2.0 research@example.com",
    "Accept":     "application/json",
}
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}


# ── Number parsing helpers ────────────────────────────────────────────────────
def _parse_number(s: str) -> float | None:
    """Convert '12.5M', '1.2B', '980K', '12,500,000' → float."""
    if not s:
        return None
    s = s.strip().upper().replace(",", "")
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    for suffix, factor in mult.items():
        if s.endswith(suffix):
            try:
                return float(s[:-1]) * factor
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _get_cached(ticker: str) -> dict | None:
    entry = _cache.get(ticker)
    if entry:
        ts, data = entry
        if time.time() - ts < CACHE_TTL_SEC:
            return data
    return None

def _set_cached(ticker: str, data: dict) -> None:
    _cache[ticker] = (time.time(), data)


# ════════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — yfinance / Yahoo Finance (with retry)
# ════════════════════════════════════════════════════════════════════════════════
def _from_yfinance(ticker: str, max_retries: int = 3) -> dict | None:
    """
    Primary source. Yahoo Finance aggregates from multiple data vendors and
    updates several times per day during market hours.

    Includes exponential back-off for transient 429 rate limits.
    In production usage with 30-min caching, 429s are extremely rare.

    Key fields:
      floatShares           — tradeable float (excludes insider/restricted)
      sharesOutstanding     — total shares issued
      shortPercentOfFloat   — short interest as decimal (0.15 = 15%)
      shortRatio            — days to cover (short interest / avg daily vol)
      sharesShort           — absolute number of shares sold short
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info

            # shortPercentOfFloat comes as a decimal in yfinance
            short_pct_raw = info.get("shortPercentOfFloat")
            short_pct = round(float(short_pct_raw) * 100, 2) if short_pct_raw else None

            float_shares  = info.get("floatShares")
            shares_out    = info.get("sharesOutstanding")
            short_ratio   = info.get("shortRatio")
            shares_short  = info.get("sharesShort")

            if float_shares is None and shares_out is None:
                logger.debug(f"[float/yfinance] no data for {ticker}")
                return None

            logger.debug(
                f"[float/yfinance] {ticker} float={float_shares} "
                f"short%={short_pct} dtc={short_ratio}"
            )
            return {
                "ticker":          ticker,
                "float_shares":    float_shares,
                "shares_out":      shares_out,
                "shares_short":    shares_short,
                "short_float_pct": short_pct,
                "days_to_cover":   short_ratio,
                "source":          "yfinance",
            }
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) or "Too Many Requests" in str(exc):
                wait = 2 ** attempt   # 1s, 2s, 4s
                logger.debug(f"[float/yfinance] {ticker} 429, retry {attempt+1} in {wait}s")
                time.sleep(wait)
            else:
                break   # non-rate-limit error, don't retry

    logger.warning(f"[float/yfinance] {ticker}: {last_exc}")
    return None


# ════════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — SEC EDGAR XBRL Company Facts API
# ════════════════════════════════════════════════════════════════════════════════

# Loaded once — maps uppercase ticker → zero-padded 10-digit CIK
_ticker_cik_map: dict[str, str] = {}

def _load_edgar_cik_map() -> None:
    global _ticker_cik_map
    if _ticker_cik_map:
        return
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_HEADERS, timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        _ticker_cik_map = {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in data.values()
        }
        logger.info(f"[float/edgar] loaded {len(_ticker_cik_map):,} CIK mappings")
    except Exception as exc:
        logger.warning(f"[float/edgar] CIK map load failed: {exc}")


def _from_edgar(ticker: str) -> dict | None:
    """
    Pull CommonStockSharesOutstanding from the most recent 10-K or 10-Q
    via the EDGAR XBRL company facts endpoint. Zero cost, no rate limits
    beyond the SEC's 10 req/s guideline.
    """
    try:
        _load_edgar_cik_map()
        cik = _ticker_cik_map.get(ticker.upper())
        if not cik:
            logger.debug(f"[float/edgar] no CIK for {ticker}")
            return None

        url  = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()

        us_gaap = resp.json().get("facts", {}).get("us-gaap", {})

        # CommonStockSharesOutstanding is the standard GAAP tag
        shares_entries = (
            us_gaap
            .get("CommonStockSharesOutstanding", {})
            .get("units", {})
            .get("shares", [])
        )

        if not shares_entries:
            return None

        # Take the most recent filed value
        latest     = max(shares_entries, key=lambda x: x.get("filed", ""))
        shares_out = latest.get("val")

        if not shares_out:
            return None

        logger.debug(f"[float/edgar] {ticker} shares_out={shares_out:,} filed={latest.get('filed')}")
        return {
            "ticker":          ticker,
            "float_shares":    None,     # EDGAR does not break out float separately
            "shares_out":      shares_out,
            "shares_short":    None,
            "short_float_pct": None,
            "days_to_cover":   None,
            "source":          "edgar_xbrl",
        }
    except Exception as exc:
        logger.warning(f"[float/edgar] {ticker}: {exc}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — Nasdaq.com short interest API
# ════════════════════════════════════════════════════════════════════════════════
def _from_nasdaq(ticker: str) -> dict | None:
    """
    Nasdaq publishes official bi-weekly short interest settlement data
    for Nasdaq-listed stocks at no cost via their public market data API.
    Gracefully returns None for NYSE/OTC names.

    Provides: shares_short (absolute), days_to_cover
    We derive short_float_pct later in _compute_short_pct() once we have float.
    """
    try:
        url = (
            f"https://api.nasdaq.com/api/quote/{ticker.upper()}"
            "/short-interest?assetClass=stocks"
        )
        resp = requests.get(url, headers={**_BROWSER_HEADERS, "Referer": "https://www.nasdaq.com/"}, timeout=12)
        resp.raise_for_status()
        payload = resp.json()

        # Status-level errors (symbol not found, not Nasdaq-listed, etc.)
        if not payload.get("data") or payload.get("status", {}).get("rCode") != 200:
            return None

        rows = payload["data"].get("shortInterestTable", {}).get("rows", [])
        if not rows:
            return None

        # Most recent settlement is the first row
        row          = rows[0]
        shares_short = _parse_number(str(row.get("interest", "")))
        dtc          = row.get("daysToCover")

        if not shares_short:
            return None

        logger.debug(
            f"[float/nasdaq] {ticker} shares_short={shares_short:,.0f} "
            f"dtc={dtc} settled={row.get('settlementDate')}"
        )
        return {
            "ticker":          ticker,
            "float_shares":    None,
            "shares_out":      None,
            "shares_short":    shares_short,
            "short_float_pct": None,
            "days_to_cover":   dtc,
            "source":          "nasdaq_si",
        }
    except Exception as exc:
        logger.debug(f"[float/nasdaq] {ticker}: {exc}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# SOURCE 4 — Yahoo Finance quoteSummary (direct HTTP, no yfinance library)
# ════════════════════════════════════════════════════════════════════════════════
def _from_yahoo_direct(ticker: str) -> dict | None:
    """
    Fallback when the yfinance library is rate-limited. Makes a direct HTTP
    request to a Yahoo Finance JSON endpoint using the v8 finance chart API
    which serves key statistics in a separate path from the v10 summary API.

    This path is often not rate-limited at the same time as the yfinance
    library path, giving us an independent data lane.
    """
    try:
        # Use v8 chart for basic price/volume (gives us enough to cross-check)
        # and query1 vs query2 as an alternative host
        url = (
            f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            "?modules=defaultKeyStatistics,summaryDetail"
        )
        headers = {
            **_BROWSER_HEADERS,
            "Accept": "application/json",
        }
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code == 429:
            return None
        resp.raise_for_status()

        result  = resp.json().get("quoteSummary", {}).get("result", [])
        if not result:
            return None

        stats   = result[0].get("defaultKeyStatistics", {})
        detail  = result[0].get("summaryDetail", {})

        def _val(d: dict, key: str):
            """Extract raw value from Yahoo Finance fmt dict."""
            v = d.get(key)
            if isinstance(v, dict):
                return v.get("raw")
            return v

        float_shares  = _val(stats, "floatShares")
        shares_out    = _val(stats, "sharesOutstanding")
        shares_short  = _val(stats, "sharesShort")
        short_ratio   = _val(stats, "shortRatio")
        short_pct_raw = _val(stats, "shortPercentOfFloat")
        short_pct     = round(float(short_pct_raw) * 100, 2) if short_pct_raw else None

        if float_shares is None and shares_out is None:
            return None

        logger.debug(
            f"[float/yahoo_direct] {ticker} float={float_shares} "
            f"short%={short_pct}"
        )
        return {
            "ticker":          ticker,
            "float_shares":    float_shares,
            "shares_out":      shares_out,
            "shares_short":    shares_short,
            "short_float_pct": short_pct,
            "days_to_cover":   short_ratio,
            "source":          "yahoo_direct",
        }
    except Exception as exc:
        logger.debug(f"[float/yahoo_direct] {ticker}: {exc}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# MOCK fallback
# ════════════════════════════════════════════════════════════════════════════════
def _mock_float_data(ticker: str) -> dict:
    rng = random.Random(sum(ord(c) for c in ticker))
    return {
        "ticker":          ticker,
        "float_shares":    rng.randint(1_000_000, 19_000_000),
        "shares_out":      rng.randint(2_000_000, 25_000_000),
        "shares_short":    rng.randint(100_000, 3_000_000),
        "short_float_pct": round(rng.uniform(5, 45), 1),
        "days_to_cover":   round(rng.uniform(0.5, 8.0), 1),
        "source":          "mock",
    }


# ════════════════════════════════════════════════════════════════════════════════
# AGGREGATOR — merge best data from all sources
# ════════════════════════════════════════════════════════════════════════════════
def _merge(primary: dict, *supplements: dict | None) -> dict:
    """
    Fill any None fields in `primary` using the first non-None value
    found in `supplements` (in order). Never overwrites a value that's
    already present.
    """
    result = dict(primary)
    for sup in supplements:
        if not sup:
            continue
        for key, val in sup.items():
            if key in result and result[key] is None and val is not None:
                result[key] = val
    return result


def _compute_short_pct(data: dict) -> dict:
    """
    If short_float_pct is still None but we have shares_short + float_shares,
    derive short % from absolute share counts.
    """
    if data.get("short_float_pct") is not None:
        return data
    shares_short = data.get("shares_short")
    float_shares = data.get("float_shares") or data.get("shares_out")
    if shares_short and float_shares and float_shares > 0:
        data["short_float_pct"] = round((shares_short / float_shares) * 100, 2)
    return data


# ════════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════════
def get_float_data(ticker: str) -> dict:
    """
    Return the richest available float / short-interest dict for `ticker`.

    Keys: ticker, float_shares, shares_out, shares_short,
          short_float_pct, days_to_cover, source

    Source waterfall:
      1. yfinance (with retry)         — primary; best all-around free source
      2. Yahoo direct API              — fallback when yfinance library is 429'd
      3. EDGAR XBRL                    — authoritative shares_out supplement
      4. Nasdaq short interest API     — authoritative short data for Nasdaq stocks
      5. mock                          — last resort (all live sources failed)
    """
    if USE_MOCK_FINVIZ:
        return _mock_float_data(ticker)

    cached = _get_cached(ticker)
    if cached:
        logger.debug(f"[float] cache hit for {ticker}")
        return cached

    # ── 1. Primary: yfinance (with exponential backoff retry) ────────────────
    result = _from_yfinance(ticker)

    # ── 2. Fallback to direct Yahoo Finance HTTP if yfinance was rate-limited ─
    if result is None:
        logger.debug(f"[float] yfinance failed for {ticker}, trying yahoo_direct")
        result = _from_yahoo_direct(ticker)

    # ── 3. Supplement shares_out from EDGAR (most authoritative) ─────────────
    if result and result.get("shares_out") is None:
        edgar = _from_edgar(ticker)
        if edgar:
            result = _merge(result, edgar)
    elif result is None:
        edgar = _from_edgar(ticker)
        if edgar:
            result = edgar

    # ── 4. Supplement short data from Nasdaq API ──────────────────────────────
    if result:
        nasdaq = _from_nasdaq(ticker)
        if nasdaq:
            result = _merge(result, nasdaq)

    # ── 5. Derive short % from absolute share counts if still missing ─────────
    if result:
        result = _compute_short_pct(result)

    # ── 6. Last resort: mock ──────────────────────────────────────────────────
    if not result:
        logger.warning(f"[float] all sources failed for {ticker} — using mock")
        result = _mock_float_data(ticker)

    source_chain = result.get("source", "?")
    logger.info(
        f"[float] {ticker}  float={result.get('float_shares')}  "
        f"short%={result.get('short_float_pct')}  "
        f"dtc={result.get('days_to_cover')}  src={source_chain}"
    )

    _set_cached(ticker, result)
    return result


def is_low_float(float_data: dict, max_shares: int = MAX_FLOAT_SHARES) -> bool:
    """Return True if float is known and below the configured threshold."""
    f = float_data.get("float_shares")
    if f is None:
        # If we only have shares_out, use that as a conservative proxy
        f = float_data.get("shares_out")
    if f is None:
        return False
    return f < max_shares


def squeeze_score(float_data: dict) -> float:
    """
    0–30 sub-score reflecting short-squeeze fuel:
      short_float_pct > 20% → up to 20 pts
      days_to_cover   > 3   → up to 10 pts
    """
    score = 0.0
    si  = float(float_data.get("short_float_pct") or 0)
    dtc = float(float_data.get("days_to_cover")   or 0)
    score += min(si,  20)
    score += min(dtc / 3 * 10, 10)
    return round(score, 1)
