"""
Tomorrow's Mover Predictor
==========================
Aggregates FORWARD-LOOKING signals — not just what moved today —
to predict tomorrow's biggest % gainers before the market opens.

Signal sources (all free / already integrated):
  1. NASDAQ earnings calendar  — stocks reporting tomorrow → binary catalyst
  2. SEC EDGAR 8-K filings     — fresh material-event disclosures from today
  3. OpenBB aggressive small caps — momentum small-float discovery
  4. Today's gainers (YF)      — continuation plays with RSI room
  5. OBB equity news           — company headlines from today

Sources you could add for significantly better predictions:
  • Intrinio API key  → unusual options (smart-money flow) — biggest alpha signal
  • Benzinga Pro key  → real-time PR newswire before moves happen
  • FMP API key       → earnings calendar with surprise estimates + guidance
  • FDA calendar RSS  → biotech binary events (single-session +100-300% movers)
  • Finviz Elite      → real-time short float + insider trades
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, timedelta

import requests

logger = logging.getLogger("apex.tomorrow")

try:
    from openbb import obb
    _OBB_OK = True
except Exception:
    _OBB_OK = False

from screener.ai.llm_client import chat as _llm_chat, is_available as _llm_ok, backend_name as _llm_backend

_HEADERS = {"User-Agent": "Mozilla/5.0 AppleWebKit/537.36"}
_SEC_HEADERS = {"User-Agent": "apex-screener research@apex.local"}


# ── 1. NASDAQ earnings calendar ───────────────────────────────────────────────
def _fetch_earnings_tomorrow() -> list[dict]:
    """Stocks reporting earnings tomorrow — high IV binary events."""
    tomorrow = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        r = requests.get(
            f"https://api.nasdaq.com/api/calendar/earnings?date={tomorrow}",
            headers={**_HEADERS, "Accept": "application/json"},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        rows = r.json().get("data", {}).get("rows", []) or []
        results = []
        for row in rows:
            sym = row.get("symbol", "")
            if not sym:
                continue
            # Strip market-cap string → float
            raw_cap = row.get("marketCap", "$0").replace("$", "").replace(",", "")
            try:
                mktcap = float(raw_cap)
            except ValueError:
                mktcap = 0
            results.append({
                "ticker":       sym,
                "name":         row.get("name", sym),
                "signal_type":  "EARNINGS_TOMORROW",
                "market_cap":   mktcap,
                "eps_forecast": row.get("epsForecast", "?"),
                "eps_last_yr":  row.get("lastYearEPS", "?"),
                "report_time":  row.get("time", ""),   # pre-market / after-hours
            })
        logger.info(f"[tomorrow] earnings tomorrow: {len(results)} stocks")
        return results
    except Exception as exc:
        logger.debug(f"[tomorrow] earnings fetch: {exc}")
        return []


# ── 2. SEC EDGAR 8-K filings filed today ─────────────────────────────────────
def _fetch_todays_8k_tickers() -> list[dict]:
    """Recent 8-K filers — material event disclosures that can carry momentum."""
    today = date.today().strftime("%Y-%m-%d")
    try:
        r = requests.get(
            f"https://efts.sec.gov/LATEST/search-index?forms=8-K"
            f"&dateRange=custom&startdt={today}&enddt={today}",
            headers=_SEC_HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            return []
        hits = r.json().get("hits", {}).get("hits", []) or []
        results = []
        seen = set()
        for hit in hits[:50]:
            src   = hit.get("_source", {})
            names = src.get("display_names", [])
            items = src.get("items", [])        # e.g. ['1.01', '8.01']
            for name_str in names:
                tickers = re.findall(r"\(([A-Z]{1,5})\)", name_str)
                for t in tickers:
                    if t in seen:
                        continue
                    seen.add(t)
                    # 8.01 = material definitive agreements, 1.01 = entry into agreement
                    # Filter to likely-material items
                    results.append({
                        "ticker":      t,
                        "name":        name_str.split("(")[0].strip(),
                        "signal_type": "SEC_8K_TODAY",
                        "items":       ", ".join(items),
                    })
        logger.info(f"[tomorrow] 8-K tickers today: {len(results)}")
        return results
    except Exception as exc:
        logger.debug(f"[tomorrow] 8-K fetch: {exc}")
        return []


# ── 3. OpenBB aggressive small caps ──────────────────────────────────────────
def _fetch_small_cap_momentum() -> list[dict]:
    """OpenBB aggressive small cap discovery — low float breakout setups."""
    if not _OBB_OK:
        return []
    try:
        r = obb.equity.discovery.aggressive_small_caps(provider="yfinance")
        results = []
        for x in r.results[:20]:
            sym = getattr(x, "symbol", "")
            if not sym:
                continue
            chg = float(getattr(x, "percent_change", 0) or 0)
            # Normalise — yfinance sometimes returns decimal, sometimes %
            if abs(chg) < 1:
                chg = round(chg * 100, 2)
            results.append({
                "ticker":      sym,
                "name":        getattr(x, "name", sym),
                "signal_type": "SMALL_CAP_MOMENTUM",
                "change_pct":  chg,
                "volume":      int(getattr(x, "volume", 0) or 0),
                "market_cap":  float(getattr(x, "market_cap", 0) or 0),
                "pe_forward":  getattr(x, "pe_forward", None),
            })
        logger.info(f"[tomorrow] small cap momentum: {len(results)}")
        return results
    except Exception as exc:
        logger.debug(f"[tomorrow] small caps: {exc}")
        return []


# ── 4. Today's gainers with RSI continuation room ─────────────────────────────
def _pick_continuation(raw_gainers: list[dict]) -> list[dict]:
    """Today's movers with healthy RSI and 2x+ volume — most likely to continue."""
    results = []
    for g in raw_gainers:
        rsi       = g.get("rsi")
        vol_ratio = float(g.get("volume_ratio") or 0)
        chg       = float(g.get("change_pct") or 0)
        # Continuation criteria: moved today, at least 2x volume, RSI not overbought
        if chg >= 5 and vol_ratio >= 2 and (rsi is None or float(rsi) <= 73):
            results.append({
                "ticker":       g["ticker"],
                "name":         g.get("name", g["ticker"]),
                "signal_type":  "CONTINUATION",
                "change_pct":   chg,
                "volume_ratio": vol_ratio,
                "rsi":          rsi,
                "market_cap":   g.get("market_cap"),
            })
    results.sort(key=lambda x: x["volume_ratio"] * x["change_pct"], reverse=True)
    logger.info(f"[tomorrow] continuation candidates: {len(results[:15])}")
    return results[:15]


# ── 5. Company news headlines (for context) ──────────────────────────────────
def _fetch_news_for_tickers(tickers: list[str], max_per_ticker: int = 2) -> dict[str, list[str]]:
    """Grab today's headlines for the top tickers to give Claude more context."""
    news_map: dict[str, list[str]] = {}
    if not _OBB_OK:
        return news_map
    for t in tickers[:12]:   # limit API calls
        try:
            n = obb.equity.news(symbol=t, limit=max_per_ticker, provider="yfinance")
            news_map[t] = [getattr(r, "title", "") for r in n.results]
            time.sleep(0.15)
        except Exception:
            pass
    return news_map


# ── Rule-based fallback ranker ────────────────────────────────────────────────
def _rule_based_picks(sorted_tickers: list[dict]) -> list[dict]:
    """
    Rank without Claude — uses signal count + type heuristics.
    Called automatically when Claude API is unavailable.
    """
    picks = []
    for rank, x in enumerate(sorted_tickers[:10], start=1):
        t       = x["ticker"]
        sigs    = list(set(x["signals"]))
        d       = x["details"]
        chg     = float(d.get("change_pct") or 0)
        vol     = float(d.get("volume_ratio") or 1)
        rsi     = d.get("rsi")
        cap     = float(d.get("market_cap") or 0)
        eps_est = d.get("eps_forecast", "")
        n_sigs  = len(sigs)

        # Confidence based on signal count + type
        if n_sigs >= 2 or "EARNINGS_TOMORROW" in sigs:
            conf = "HIGH" if (n_sigs >= 2 or cap < 5e8) else "MEDIUM"
        elif "CONTINUATION" in sigs and vol >= 5:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        # Determine catalyst label
        if n_sigs >= 2:
            catalyst = "MULTI_SIGNAL"
        elif "EARNINGS_TOMORROW" in sigs:
            catalyst = "EARNINGS"
        elif "SEC_8K_TODAY" in sigs:
            catalyst = "SEC_CATALYST"
        elif "CONTINUATION" in sigs:
            catalyst = "CONTINUATION"
        else:
            catalyst = "SMALL_CAP_MOMENTUM"

        # Action
        action = "BUY_AT_OPEN" if conf == "HIGH" else "WATCH_OPEN"

        # Why text
        why_parts = []
        if "EARNINGS_TOMORROW" in sigs:
            why_parts.append(f"Earnings tomorrow (EPS est {eps_est}) — guaranteed vol expansion.")
        if "SEC_8K_TODAY" in sigs:
            why_parts.append(f"Filed 8-K today — fresh material event not fully priced in.")
        if "CONTINUATION" in sigs:
            why_parts.append(f"Up {chg:+.0f}% today on {vol:.0f}× volume with RSI room remaining.")
        if "SMALL_CAP_MOMENTUM" in sigs:
            why_parts.append(f"Flagged as aggressive small-cap momentum play.")
        why = " ".join(why_parts) or f"Multiple signals: {', '.join(sigs)}."

        # Entry
        if "EARNINGS_TOMORROW" in sigs:
            entry = "Wait for gap direction at open, then enter on first 5-min candle close with volume."
        elif "CONTINUATION" in sigs:
            entry = f"Buy if opens above yesterday's close with volume ≥ 1M in first 5 min."
        else:
            entry = "Enter on breakout of first 15-min candle high with above-average volume."

        picks.append({
            "rank":           rank,
            "ticker":         t,
            "action":         action,
            "tomorrow_action": action,
            "catalyst":       catalyst,
            "why":            why,
            "entry":          entry,
            "stop":           "Stop below open candle low or -8% from entry, whichever is tighter.",
            "confidence":     conf,
            "risk":           "Rule-based signal only — upgrade API credits for Claude analysis.",
            "_source":        "RULE_BASED",
        })
    logger.info(f"[tomorrow] rule-based fallback: {len(picks)} picks")
    return picks


# ── Main: synthesise all signals with Claude ─────────────────────────────────
def predict_tomorrow(raw_gainers: list[dict]) -> list[dict]:
    """
    Fetch all forward-looking signals, deduplicate by ticker,
    then ask Claude to rank the top 8-10 picks for tomorrow's session.

    Returns list of pick dicts ordered by rank.
    """
    earnings   = _fetch_earnings_tomorrow()
    eight_ks   = _fetch_todays_8k_tickers()
    small_caps = _fetch_small_cap_momentum()
    cont       = _pick_continuation(raw_gainers)

    # ── Merge all signals per ticker ────────────────────────────────────────
    ticker_data: dict[str, dict] = {}
    for item in earnings + eight_ks + small_caps + cont:
        t = item.get("ticker", "")
        if not t or len(t) > 6:
            continue
        if t not in ticker_data:
            ticker_data[t] = {
                "ticker":   t,
                "name":     item.get("name", t),
                "signals":  [],
                "details":  {},
            }
        ticker_data[t]["signals"].append(item["signal_type"])
        ticker_data[t]["details"].update(
            {k: v for k, v in item.items() if k not in ("ticker", "signal_type", "name")}
        )

    if not ticker_data:
        logger.warning("[tomorrow] no signals gathered — all sources returned empty")
        return []

    # Sort: multi-signal tickers first, then by change_pct
    sorted_tickers = sorted(
        ticker_data.values(),
        key=lambda x: (len(x["signals"]), float(x["details"].get("change_pct", 0) or 0)),
        reverse=True,
    )

    # Fetch news for top candidates
    top_syms  = [x["ticker"] for x in sorted_tickers[:15]]
    news_map  = _fetch_news_for_tickers(top_syms)

    # ── Build Claude prompt ──────────────────────────────────────────────────
    lines = []
    for x in sorted_tickers[:30]:
        t   = x["ticker"]
        sig = ", ".join(sorted(set(x["signals"])))
        d   = x["details"]
        chg = d.get("change_pct")
        vol = d.get("volume_ratio")
        rsi = d.get("rsi")
        cap = d.get("market_cap", 0)
        eps = d.get("eps_forecast")
        time_str = d.get("report_time", "")
        headlines = "; ".join(h for h in news_map.get(t, []) if h)

        parts = [f"{t} [{sig}]"]
        if chg  is not None: parts.append(f"today={float(chg):+.0f}%")
        if vol  is not None: parts.append(f"vol={float(vol):.0f}x")
        if rsi  is not None: parts.append(f"rsi={float(rsi):.0f}")
        if cap  and float(cap) > 0:
            parts.append(f"mktcap=${float(cap)/1e6:.0f}M")
        if eps  is not None: parts.append(f"eps_est={eps}")
        if time_str:          parts.append(f"reports={time_str.replace('time-','')}")
        if headlines:         parts.append(f'news="{headlines[:100]}"')
        lines.append("  " + "  ".join(parts))

    tomorrow_str = (date.today() + timedelta(days=1)).strftime("%A %B %d")
    prompt = f"""You are a professional momentum and catalyst trader. Today is {date.today().strftime('%A %B %d, %Y')}.

Your job: identify the TOP 8 stocks most likely to make a BIG % move during regular market hours tomorrow ({tomorrow_str}).

Candidates with their signals:
{chr(10).join(lines)}

Signal legend:
  EARNINGS_TOMORROW  = binary earnings event → vol expansion guaranteed, direction uncertain but small caps often spike
  SEC_8K_TODAY       = filed a material event 8-K today → may not yet be priced in
  SMALL_CAP_MOMENTUM = discovered as aggressive small-cap by OpenBB → float/momentum setup
  CONTINUATION       = strong gainer today with RSI room and high volume → likely continues next session

Ranking criteria (most important first):
1. Multiple converging signals on same ticker = highest conviction
2. Small/micro cap (<$500M) earnings = most volatile; institutional can't front-run
3. 8-K with unusual items (agreements, product launches) + today's momentum
4. Continuation: RSI ≤ 70, volume ≥ 10x, moved ≥ 15% today
5. Pure small-cap discovery with strong today % and volume

Return ONLY a valid JSON array of exactly 8 objects (fewer if not enough quality setups):
[
  {{
    "rank": 1,
    "ticker": "XXX",
    "action": "BUY_AT_OPEN",
    "catalyst": "EARNINGS | OPTIONS_FLOW | SEC_CATALYST | CONTINUATION | SQUEEZE | MULTI_SIGNAL",
    "why": "2-sentence forward-looking reason why this will move BIG tomorrow",
    "entry": "exact regular-hours entry instruction (e.g. 'Buy first 5-min candle close above $X', 'Enter on gap-and-go above yesterday high with vol')",
    "stop": "where to cut the loss (e.g. 'Stop below open', 'Stop -8% from entry')",
    "confidence": "HIGH | MEDIUM | LOW",
    "risk": "the one biggest risk for this trade"
  }}
]

Rules:
- BUY_AT_OPEN = high conviction, act at open. WATCH_OPEN = wait for first 15-min candle confirmation.
- If a stock has earnings tomorrow and is small-cap, that is almost always HIGH or MEDIUM confidence for volatility.
- Do NOT include large-cap stocks (>$5B mktcap) unless they have multiple converging signals.
- Return ONLY the raw JSON array. No markdown, no explanation."""

    # ── Try LLM (Groq / Ollama) ──────────────────────────────────────────────
    if _llm_ok():
        try:
            picks = _llm_chat(prompt, max_tokens=1400, expect_json=True)
            for p in picks:
                p["_source"]         = f"LLM_{_llm_backend()}"
                p["tomorrow_action"] = p.get("action", "WATCH_OPEN")
            logger.info(f"[tomorrow] {_llm_backend()} returned {len(picks)} picks")
            return picks
        except Exception as exc:
            logger.warning(f"[tomorrow] LLM failed ({exc}) — falling back to rule-based ranking")

    # ── Rule-based fallback (no API required) ────────────────────────────────
    return _rule_based_picks(sorted_tickers)
