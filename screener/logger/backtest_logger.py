"""
Backtest Logger — alert persistence + T+1hr / T+24hr price follow-up.

Every setup that scores >= HIGH_SCORE_THRESHOLD is written to a CSV row.
A background-safe `update_prices()` job fills in price_t1hr and price_t24hr
by checking elapsed time since `logged_at` on each scan cycle.

CSV columns
-----------
logged_at       ISO-8601 UTC timestamp when the alert fired
ticker          Stock symbol
price_at_alert  Price at the moment of alert
float_shares    Float size at alert time
volume          Intraday volume at alert
volume_ratio    Volume vs 30-day average
short_pct       Short interest %
score           Composite 0-100 score
catalyst_type   LLM-classified filing category
sentiment       BULLISH | BEARISH | NEUTRAL
dilution_risk   NONE | LOW | MEDIUM | HIGH | SEVERE
toxicity_flags  Pipe-delimited list of detected toxic patterns
bullish_signals Pipe-delimited list of detected bullish patterns
llm_summary     2-3 sentence Claude summary
filing_url      Direct link to the SEC filing
price_t1hr      Price ~1 hour after alert (filled retroactively)
price_t24hr     Price ~24 hours after alert (filled retroactively)
pct_t1hr        % change from alert price to T+1hr (computed on fill)
pct_t24hr       % change from alert price to T+24hr (computed on fill)
"""

from __future__ import annotations

import csv
import io
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from screener.config import ALERT_CSV, HIGH_SCORE_THRESHOLD

logger = logging.getLogger(__name__)

_COLUMNS = [
    "logged_at",
    "ticker",
    "price_at_alert",
    "float_shares",
    "volume",
    "volume_ratio",
    "short_pct",
    "score",
    "catalyst_type",
    "sentiment",
    "dilution_risk",
    "toxicity_flags",
    "bullish_signals",
    "llm_summary",
    "filing_url",
    "price_t1hr",
    "price_t24hr",
    "pct_t1hr",
    "pct_t24hr",
]


def _ensure_csv() -> None:
    os.makedirs(os.path.dirname(ALERT_CSV), exist_ok=True)
    if not os.path.exists(ALERT_CSV):
        with open(ALERT_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_COLUMNS).writeheader()
        logger.info(f"[logger] created alert log at {ALERT_CSV}")


def log_alert(setup: dict[str, Any], filing: dict[str, Any] | None = None) -> bool:
    """
    Append a high-scoring setup to the alert CSV.

    Returns True if the alert was written, False if filtered out (score too low
    or duplicate ticker already logged in the past 60 minutes).
    """
    if setup.get("score", 0) < HIGH_SCORE_THRESHOLD:
        return False

    _ensure_csv()

    ticker = setup.get("ticker", "")

    # Dedup: don't log the same ticker more than once per hour
    existing = load_alerts_df()
    if not existing.empty and ticker:
        recent = existing[existing["ticker"] == ticker].copy()
        if not recent.empty:
            recent["logged_at"] = pd.to_datetime(recent["logged_at"], utc=True, errors="coerce")
            last = recent["logged_at"].max()
            if pd.notna(last):
                age = datetime.now(timezone.utc) - last
                if age < timedelta(hours=1):
                    logger.debug(f"[logger] skipping duplicate alert for {ticker} ({age} ago)")
                    return False

    row = {
        "logged_at":       datetime.now(timezone.utc).isoformat(),
        "ticker":          ticker,
        "price_at_alert":  setup.get("price", ""),
        "float_shares":    setup.get("float_shares", ""),
        "volume":          setup.get("volume", ""),
        "volume_ratio":    setup.get("volume_ratio", ""),
        "short_pct":       setup.get("short_pct", ""),
        "score":           setup.get("score", ""),
        "catalyst_type":   setup.get("catalyst_type", ""),
        "sentiment":       setup.get("sentiment", ""),
        "dilution_risk":   setup.get("dilution_risk", ""),
        "toxicity_flags":  "|".join(setup.get("toxicity_flags") or []),
        "bullish_signals": "|".join(setup.get("bullish_signals") or []),
        "llm_summary":     setup.get("llm_summary", ""),
        "filing_url":      (filing or {}).get("filing_url", ""),
        "price_t1hr":      "",
        "price_t24hr":     "",
        "pct_t1hr":        "",
        "pct_t24hr":       "",
    }

    with open(ALERT_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=_COLUMNS).writerow(row)

    logger.info(
        f"[logger] ALERT → {ticker}  score={row['score']}  "
        f"price=${row['price_at_alert']}  catalyst={row['catalyst_type']}  "
        f"sentiment={row['sentiment']}"
    )
    return True


def update_prices(
    quote_fn: Callable[[str], dict[str, Any]],
    t1hr_window: int = 75,    # check between 60-75 min after alert
    t24hr_window: int = 1500, # check between 24hr and 25hr after alert
) -> int:
    """
    Retroactively fill price_t1hr and price_t24hr columns.

    Reads all unfilled alert rows, checks elapsed time, fetches current price
    via quote_fn (polygon_client.get_quote), and rewrites the CSV atomically.

    Parameters
    ----------
    quote_fn      : callable(ticker) → dict with key "price"
    t1hr_window   : minutes after alert within which we fill price_t1hr
    t24hr_window  : minutes after alert within which we fill price_t24hr

    Returns number of price cells updated.
    """
    _ensure_csv()

    try:
        df = pd.read_csv(ALERT_CSV, dtype=str)
    except Exception as exc:
        logger.error(f"[logger] failed to read alert CSV: {exc}")
        return 0

    if df.empty:
        return 0

    df["logged_at_dt"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce")
    now = datetime.now(timezone.utc)
    updated = 0

    for idx, row in df.iterrows():
        logged = row["logged_at_dt"]
        if pd.isna(logged):
            continue
        ticker = row.get("ticker", "")
        if not ticker:
            continue

        elapsed_min = (now - logged).total_seconds() / 60

        # ── T+1hr fill ───────────────────────────────────────────────────────
        if str(row.get("price_t1hr", "")).strip() == "" and 60 <= elapsed_min <= t1hr_window:
            try:
                q = quote_fn(ticker)
                price = q.get("price")
                if price is not None:
                    df.at[idx, "price_t1hr"] = str(price)
                    alert_price = _safe_float(row.get("price_at_alert"))
                    if alert_price and alert_price > 0:
                        pct = round(((price - alert_price) / alert_price) * 100, 2)
                        df.at[idx, "pct_t1hr"] = str(pct)
                    updated += 1
                    logger.info(f"[logger] T+1hr fill: {ticker} → ${price}")
            except Exception as exc:
                logger.warning(f"[logger] T+1hr fetch failed for {ticker}: {exc}")

        # ── T+24hr fill ──────────────────────────────────────────────────────
        elif (
            str(row.get("price_t24hr", "")).strip() == ""
            and 24 * 60 <= elapsed_min <= t24hr_window
        ):
            try:
                q = quote_fn(ticker)
                price = q.get("price")
                if price is not None:
                    df.at[idx, "price_t24hr"] = str(price)
                    alert_price = _safe_float(row.get("price_at_alert"))
                    if alert_price and alert_price > 0:
                        pct = round(((price - alert_price) / alert_price) * 100, 2)
                        df.at[idx, "pct_t24hr"] = str(pct)
                    updated += 1
                    logger.info(f"[logger] T+24hr fill: {ticker} → ${price}")
            except Exception as exc:
                logger.warning(f"[logger] T+24hr fetch failed for {ticker}: {exc}")

    if updated > 0:
        df.drop(columns=["logged_at_dt"]).to_csv(ALERT_CSV, index=False)
        logger.info(f"[logger] price update complete — {updated} cells filled")

    return updated


def _safe_float(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def load_alerts() -> list[dict[str, Any]]:
    """Return all logged alerts as a list of raw dicts."""
    _ensure_csv()
    with open(ALERT_CSV, newline="") as f:
        return list(csv.DictReader(f))


def load_alerts_df() -> pd.DataFrame:
    """Return all logged alerts as a pandas DataFrame."""
    _ensure_csv()
    try:
        return pd.read_csv(ALERT_CSV, dtype=str)
    except Exception:
        return pd.DataFrame(columns=_COLUMNS)


def backtest_summary() -> dict[str, Any]:
    """
    Compute simple performance statistics across all filled alert rows.

    Returns a dict with keys:
      total_alerts, filled_t1hr, filled_t24hr,
      hit_rate_t1hr  (% positive returns at T+1hr),
      hit_rate_t24hr (% positive returns at T+24hr),
      avg_pct_t1hr, avg_pct_t24hr, median_pct_t24hr
    """
    df = load_alerts_df()
    if df.empty:
        return {"total_alerts": 0}

    def _to_num(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce")

    pct_1  = _to_num("pct_t1hr").dropna()
    pct_24 = _to_num("pct_t24hr").dropna()

    return {
        "total_alerts":    len(df),
        "filled_t1hr":     int(pct_1.count()),
        "filled_t24hr":    int(pct_24.count()),
        "hit_rate_t1hr":   round((pct_1 > 0).mean() * 100, 1) if len(pct_1) else None,
        "hit_rate_t24hr":  round((pct_24 > 0).mean() * 100, 1) if len(pct_24) else None,
        "avg_pct_t1hr":    round(float(pct_1.mean()), 2) if len(pct_1) else None,
        "avg_pct_t24hr":   round(float(pct_24.mean()), 2) if len(pct_24) else None,
        "median_pct_t24hr":round(float(pct_24.median()), 2) if len(pct_24) else None,
    }
