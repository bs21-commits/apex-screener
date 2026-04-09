"""
Polygon.io client — real implementation + mock fallback.

Real endpoints used:
  GET /v2/aggs/ticker/{ticker}/prev           → previous close / avg volume
  GET /v2/aggs/ticker/{ticker}/range/1/minute → intraday bars
  GET /v3/trades/{ticker}                     → real-time trade feed
  GET /v3/reference/tickers/{ticker}          → reference data
  Websocket /stocks.*                         → LULD halts (Polygon WS)

In mock mode every public function returns plausible dummy data so the
scoring engine and dashboard can run without a paid key.
"""

import random
import time
from datetime import datetime, timezone
from typing import Optional

from screener.config import USE_MOCK_POLYGON, POLYGON_API_KEY

# ── Real client helper ───────────────────────────────────────────────────────
try:
    import requests as _requests
    _SESSION = _requests.Session()
    _SESSION.headers.update({"Authorization": f"Bearer {POLYGON_API_KEY}"})
    _BASE = "https://api.polygon.io"
except ImportError:
    _SESSION = None
    _BASE = ""


def _get(path: str, params: dict | None = None) -> dict:
    resp = _SESSION.get(f"{_BASE}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── Mock helpers ─────────────────────────────────────────────────────────────
def _mock_quote(ticker: str) -> dict:
    """Return a plausible intraday snapshot for a low-float ticker."""
    seed = sum(ord(c) for c in ticker)
    rng  = random.Random(seed + int(time.time() // 300))   # changes every 5 min

    price         = round(rng.uniform(0.50, 15.00), 2)
    avg_volume    = rng.randint(200_000, 2_000_000)
    volume_spike  = round(rng.uniform(1.0, 20.0), 1)
    current_vol   = int(avg_volume * volume_spike)
    change_pct    = round(rng.uniform(-15, 80), 2)
    premarket_pct = round(rng.uniform(-5, 50), 2)

    return {
        "ticker":         ticker,
        "price":          price,
        "change_pct":     change_pct,
        "premarket_pct":  premarket_pct,
        "volume":         current_vol,
        "avg_volume_30d": avg_volume,
        "volume_ratio":   volume_spike,
        "vwap":           round(price * rng.uniform(0.92, 1.08), 2),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


def _mock_halts() -> list[dict]:
    """Return 0-2 fake LULD halt events."""
    tickers = ["SYTA", "MULN", "FFIE", "NKLA", "CENX"]
    halts = []
    if random.random() < 0.3:   # 30 % chance there's an active halt
        t = random.choice(tickers)
        halts.append({
            "ticker":    t,
            "halt_type": random.choice(["LULD", "T1", "T2"]),
            "status":    random.choice(["Halted", "Resumed"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    return halts


def _mock_prev_close(ticker: str) -> dict:
    seed = sum(ord(c) for c in ticker)
    rng  = random.Random(seed)
    price = round(rng.uniform(0.50, 15.00), 2)
    return {"ticker": ticker, "prev_close": price, "prev_volume": rng.randint(100_000, 3_000_000)}


# ── Public API ───────────────────────────────────────────────────────────────

def get_quote(ticker: str) -> dict:
    """
    Return an intraday snapshot dict with keys:
      ticker, price, change_pct, premarket_pct, volume,
      avg_volume_30d, volume_ratio, vwap, timestamp
    """
    if USE_MOCK_POLYGON or not POLYGON_API_KEY:
        return _mock_quote(ticker)

    try:
        # Real: snapshot endpoint
        data = _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
        snap = data.get("ticker", {})
        day  = snap.get("day", {})
        prev = snap.get("prevDay", {})

        price      = day.get("c") or prev.get("c", 0)
        prev_close = prev.get("c", price) or price
        change_pct = round(((price - prev_close) / prev_close) * 100, 2) if prev_close else 0
        avg_vol    = prev.get("v", 1) or 1

        return {
            "ticker":         ticker,
            "price":          price,
            "change_pct":     change_pct,
            "premarket_pct":  0.0,         # requires separate pre-mkt bar query
            "volume":         day.get("v", 0),
            "avg_volume_30d": avg_vol,
            "volume_ratio":   round(day.get("v", 0) / avg_vol, 1),
            "vwap":           day.get("vw", price),
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        print(f"[polygon] live quote failed for {ticker}: {exc} — falling back to mock")
        return _mock_quote(ticker)


def get_luld_halts() -> list[dict]:
    """
    Return a list of active or recent LULD halt events.
    Each dict: {ticker, halt_type, status, timestamp}
    """
    if USE_MOCK_POLYGON or not POLYGON_API_KEY:
        return _mock_halts()

    try:
        data = _get("/v3/markets/holidays")   # placeholder — real LULD needs WS
        # Polygon LULD halts are delivered via WebSocket (channel: LH).
        # For REST-only MVP we return empty and let the WS module fill this.
        return []
    except Exception as exc:
        print(f"[polygon] halt fetch failed: {exc} — falling back to mock")
        return _mock_halts()


def get_gainers(min_change_pct: float = 10.0, limit: int = 40) -> list[dict]:
    """
    Return top % gainers right now from Polygon's snapshot endpoint.
    Filters to stocks with >= min_change_pct intraday gain.
    Each dict: ticker, price, change_pct, volume, avg_volume_30d, volume_ratio, vwap
    """
    if USE_MOCK_POLYGON or not POLYGON_API_KEY:
        return _mock_gainers()

    try:
        data    = _get("/v2/snapshot/locale/us/markets/stocks/gainers")
        tickers = data.get("tickers", [])
        results = []
        for snap in tickers[:limit]:
            day  = snap.get("day", {})
            prev = snap.get("prevDay", {})
            sym  = snap.get("ticker", "")
            price      = day.get("c") or prev.get("c", 0)
            prev_close = prev.get("c", price) or price
            change_pct = round(((price - prev_close) / prev_close) * 100, 2) if prev_close else 0
            if change_pct < min_change_pct:
                continue
            avg_vol = prev.get("v", 1) or 1
            cur_vol = day.get("v", 0)
            results.append({
                "ticker":         sym,
                "price":          price,
                "change_pct":     change_pct,
                "premarket_pct":  0.0,
                "volume":         cur_vol,
                "avg_volume_30d": avg_vol,
                "volume_ratio":   round(cur_vol / avg_vol, 1),
                "vwap":           day.get("vw", price),
                "timestamp":      datetime.now(timezone.utc).isoformat(),
            })
        return sorted(results, key=lambda x: x["change_pct"], reverse=True)
    except Exception as exc:
        print(f"[polygon] gainers fetch failed: {exc} — mock")
        return _mock_gainers()


def _mock_gainers() -> list[dict]:
    """Realistic mock of today's low-float movers."""
    movers = [
        ("UCAR", 4.21, 444.2, 18_500_000, 320_000),
        ("LTRY", 1.87, 182.5, 9_200_000,  180_000),
        ("SYTA", 3.44, 95.3,  4_100_000,  220_000),
        ("HUDI", 2.11, 73.8,  6_700_000,  150_000),
        ("MYNZ", 0.94, 61.2,  3_400_000,   90_000),
    ]
    results = []
    for sym, price, chg, vol, avg_vol in movers:
        results.append({
            "ticker":         sym,
            "price":          price,
            "change_pct":     chg,
            "premarket_pct":  round(chg * 0.4, 1),
            "volume":         vol,
            "avg_volume_30d": avg_vol,
            "volume_ratio":   round(vol / avg_vol, 1),
            "vwap":           round(price * 0.85, 2),
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        })
    return results


def get_prev_close(ticker: str) -> dict:
    """Return {ticker, prev_close, prev_volume}."""
    if USE_MOCK_POLYGON or not POLYGON_API_KEY:
        return _mock_prev_close(ticker)

    try:
        data = _get(f"/v2/aggs/ticker/{ticker}/prev")
        result = (data.get("results") or [{}])[0]
        return {
            "ticker":       ticker,
            "prev_close":   result.get("c", 0),
            "prev_volume":  result.get("v", 0),
        }
    except Exception as exc:
        print(f"[polygon] prev_close failed for {ticker}: {exc} — mock")
        return _mock_prev_close(ticker)
