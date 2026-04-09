"""
OpenBB client — replaces Polygon for price, gainers, and market data.

Provides identical public interface to polygon_client.py so the rest of the
screener works without changes.  Falls back to mock data on any failure.

OpenBB free-tier sources used:
  yfinance  — real-time quotes, historical OHLCV, top gainers
  sec       — SEC filing index (10-K / 10-Q URLs)
  benzinga  — news headlines + sentiment (requires free API key, optional)
"""

from __future__ import annotations

import logging
import requests
import random
import time
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# ── OpenBB singleton ──────────────────────────────────────────────────────────
try:
    import warnings
    warnings.filterwarnings("ignore")
    from openbb import obb
    _OBB_OK = True
except Exception as exc:
    logger.warning(f"[openbb] SDK unavailable: {exc}")
    _OBB_OK = False

# ── RSI (manual, no ta-lib dependency) ───────────────────────────────────────
def _rsi(closes: list[float], period: int = 14) -> float | None:
    """Wilder RSI from a list of closing prices."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_g  = sum(gains[:period]) / period
    avg_l  = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return round(100 - 100 / (1 + rs), 2)


# ── Mock helpers (identical to polygon_client mocks) ─────────────────────────
def _mock_quote(ticker: str) -> dict:
    rng = random.Random(sum(ord(c) for c in ticker) + int(time.time() // 300))
    price = round(rng.uniform(0.50, 15.00), 2)
    avg_vol = rng.randint(200_000, 2_000_000)
    vol_spike = round(rng.uniform(1.0, 20.0), 1)
    return {
        "ticker": ticker, "price": price,
        "change_pct": round(rng.uniform(-15, 80), 2),
        "premarket_pct": round(rng.uniform(-5, 50), 2),
        "volume": int(avg_vol * vol_spike), "avg_volume_30d": avg_vol,
        "volume_ratio": vol_spike, "vwap": round(price * rng.uniform(0.92, 1.08), 2),
        "rsi": round(rng.uniform(30, 85), 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

def _mock_gainers() -> list[dict]:
    movers = [
        ("UCAR", 4.21, 444.2, 18_500_000, 320_000),
        ("LTRY", 1.87, 182.5, 9_200_000,  180_000),
        ("SYTA", 3.44, 95.3,  4_100_000,  220_000),
        ("HUDI", 2.11, 73.8,  6_700_000,  150_000),
        ("MYNZ", 0.94, 61.2,  3_400_000,   90_000),
    ]
    return [{
        "ticker": sym, "price": price, "change_pct": chg,
        "premarket_pct": round(chg * 0.4, 1),
        "volume": vol, "avg_volume_30d": avg,
        "volume_ratio": round(vol / avg, 1),
        "vwap": round(price * 0.85, 2), "rsi": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    } for sym, price, chg, vol, avg in movers]

def _mock_halts() -> list[dict]:
    return []


# ── Public API ────────────────────────────────────────────────────────────────

def get_quote(ticker: str) -> dict:
    """Real-time quote with RSI attached."""
    if not _OBB_OK:
        return _mock_quote(ticker)
    try:
        q     = obb.equity.price.quote(ticker, provider="yfinance")
        r     = q.results[0] if q.results else None
        if not r:
            return _mock_quote(ticker)

        price      = float(getattr(r, "last_price", 0) or 0)
        prev_close = float(getattr(r, "prev_close", price) or price)
        change_pct = round(((price - prev_close) / prev_close) * 100, 2) if prev_close else 0
        volume     = int(getattr(r, "volume", 0) or 0)
        avg_vol    = int(getattr(r, "average_volume", 1) or 1)

        # RSI from 30d of daily closes
        rsi_val = None
        try:
            hist  = obb.equity.price.historical(ticker, interval="1d", provider="yfinance")
            closes = [float(x.close) for x in hist.results if x.close is not None]
            rsi_val = _rsi(closes)
        except Exception:
            pass

        return {
            "ticker":         ticker,
            "price":          price,
            "change_pct":     change_pct,
            "premarket_pct":  float(getattr(r, "pre_market_change_percent", 0) or 0),
            "volume":         volume,
            "avg_volume_30d": avg_vol,
            "volume_ratio":   round(volume / avg_vol, 1) if avg_vol else 0,
            "vwap":           float(getattr(r, "last_price", price) or price),
            "rsi":            rsi_val,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.warning(f"[openbb] quote {ticker}: {exc} — mock")
        return _mock_quote(ticker)


def get_gainers(min_change_pct: float = 5.0, limit: int = 250) -> list[dict]:
    """
    Full daily gainers list from Yahoo Finance screener API.
    Returns up to 250 stocks, sorted by % change descending.
    Falls back to OpenBB discovery if the direct API fails.
    """
    results = _yf_screener_gainers(min_change_pct=min_change_pct, limit=limit)
    if results:
        return results
    # Fallback: OpenBB discovery (smaller set)
    return _openbb_gainers(min_change_pct=min_change_pct, limit=limit)


def _yf_screener_gainers(min_change_pct: float, limit: int) -> list[dict]:
    """Direct hit to Yahoo Finance predefined screener — returns full market movers."""
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        params = {"count": min(limit, 250), "scrIds": "day_gainers", "formatted": "false"}
        r = requests.get(url, headers=headers, params=params, timeout=12)
        if r.status_code != 200:
            return []
        data      = r.json()
        all_quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])

        results = []
        for q in all_quotes:
            sym   = q.get("symbol", "")
            chg   = float(q.get("regularMarketChangePercent", 0) or 0)
            if chg < min_change_pct:
                continue
            price = float(q.get("regularMarketPrice", 0) or 0)
            vol   = int(q.get("regularMarketVolume", 0) or 0)
            avg_v = int(q.get("averageDailyVolume3Month", 1) or 1)
            mktcap= q.get("marketCap")
            results.append({
                "ticker":         sym,
                "price":          price,
                "change_pct":     round(chg, 2),
                "premarket_pct":  float(q.get("preMarketChangePercent", 0) or 0),
                "volume":         vol,
                "avg_volume_30d": avg_v,
                "volume_ratio":   round(vol / avg_v, 1) if avg_v else 0,
                "vwap":           float(q.get("regularMarketDayHigh", price) or price),
                "market_cap":     mktcap,
                "name":           q.get("shortName", sym),
                "rsi":            None,
                "timestamp":      datetime.now(timezone.utc).isoformat(),
            })

        logger.info(f"[yf-screener] {len(results)} gainers >= {min_change_pct}%")
        return sorted(results, key=lambda x: x["change_pct"], reverse=True)
    except Exception as exc:
        logger.warning(f"[yf-screener] failed: {exc}")
        return []


def _openbb_gainers(min_change_pct: float, limit: int) -> list[dict]:
    """Fallback: OpenBB discovery gainers (smaller set, ~200 stocks)."""
    if not _OBB_OK:
        return _mock_gainers()
    try:
        g = obb.equity.discovery.gainers(provider="yfinance")
        results = []
        for r in g.results[:limit]:
            sym = getattr(r, "symbol", "")
            chg = float(getattr(r, "percent_change", 0) or 0)
            if abs(chg) < 1: chg = round(chg * 100, 2)
            if chg < min_change_pct: continue
            price = float(getattr(r, "price", 0) or 0)
            vol   = int(getattr(r, "volume", 0) or 0)
            avg_v = int(getattr(r, "avg_volume", 0) or 0)
            vol_ratio = round(vol / avg_v, 1) if avg_v and avg_v > vol * 0.01 else 1.0
            results.append({
                "ticker": sym, "price": price, "change_pct": round(chg, 2),
                "premarket_pct": 0.0, "volume": vol, "avg_volume_30d": avg_v,
                "volume_ratio": vol_ratio, "vwap": price, "market_cap": None,
                "name": sym, "rsi": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        logger.info(f"[openbb-discovery] {len(results)} gainers >= {min_change_pct}%")
        return sorted(results, key=lambda x: x["change_pct"], reverse=True)
    except Exception as exc:
        logger.warning(f"[openbb] gainers: {exc} — mock")
        return _mock_gainers()


def get_luld_halts() -> list[dict]:
    """OpenBB has no LULD halt feed — returns empty (halts via SEC/exchange WS only)."""
    return []


def get_prev_close(ticker: str) -> dict:
    if not _OBB_OK:
        return {"ticker": ticker, "prev_close": 0, "prev_volume": 0}
    try:
        q = obb.equity.price.quote(ticker, provider="yfinance")
        r = q.results[0] if q.results else None
        return {
            "ticker":      ticker,
            "prev_close":  float(getattr(r, "prev_close", 0) or 0) if r else 0,
            "prev_volume": int(getattr(r, "volume", 0) or 0) if r else 0,
        }
    except Exception:
        return {"ticker": ticker, "prev_close": 0, "prev_volume": 0}


def get_sec_filings(ticker: str, form_type: str = "10-K", limit: int = 3) -> list[dict]:
    """Return recent SEC filing URLs for a ticker via OpenBB SEC provider."""
    if not _OBB_OK:
        return []
    try:
        f = obb.equity.fundamental.filings(ticker, form_type=form_type,
                                           provider="sec", limit=limit)
        return [
            {
                "ticker":    ticker,
                "form_type": getattr(r, "form_type", form_type),
                "date":      str(getattr(r, "date", "")),
                "url":       getattr(r, "report_url", "") or "",
            }
            for r in f.results
        ]
    except Exception as exc:
        logger.debug(f"[openbb] filings {ticker}: {exc}")
        return []


def batch_enrich_rsi(gainers: list[dict], period: int = 14) -> list[dict]:
    """
    Enrich a list of gainer dicts with RSI values using a single batch
    yfinance download (fast — one API call for all tickers).
    Falls back gracefully if yfinance is unavailable or download fails.
    """
    if not gainers:
        return gainers
    try:
        import yfinance as yf
        tickers = [g["ticker"] for g in gainers]
        # Batch download: 1 month of daily closes for all tickers at once
        raw = yf.download(
            tickers, period="1mo", interval="1d",
            progress=False, auto_adjust=True, threads=True,
        )
        if raw.empty:
            return gainers

        # yfinance returns multi-index columns for multiple tickers
        if hasattr(raw.columns, "levels"):
            close_df = raw["Close"]
        else:
            # single ticker — wrap so indexing is uniform
            close_df = raw[["Close"]].rename(columns={"Close": tickers[0]})

        rsi_map: dict[str, float | None] = {}
        for sym in tickers:
            try:
                closes = close_df[sym].dropna().tolist() if sym in close_df.columns else []
                rsi_map[sym] = _rsi(closes, period) if len(closes) >= period + 1 else None
            except Exception:
                rsi_map[sym] = None

        return [{**g, "rsi": rsi_map.get(g["ticker"])} for g in gainers]
    except Exception as exc:
        logger.warning(f"[batch_enrich_rsi] failed: {exc}")
        return gainers


def get_news(ticker: str, limit: int = 10) -> list[dict]:
    """Recent news headlines for a ticker."""
    if not _OBB_OK:
        return []
    try:
        n = obb.equity.news(symbol=ticker, limit=limit, provider="yfinance")
        return [
            {
                "title":     getattr(r, "title", ""),
                "url":       getattr(r, "url", ""),
                "published": str(getattr(r, "date", "")),
                "source":    getattr(r, "source", ""),
            }
            for r in n.results
        ]
    except Exception as exc:
        logger.debug(f"[openbb] news {ticker}: {exc}")
        return []
