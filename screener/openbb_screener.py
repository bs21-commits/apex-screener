"""
APEX AI Stock Screener — powered by OpenBB + Claude.

What it does each run:
  1. Fetch today's top % gainers via OpenBB (yfinance provider)
  2. Filter to low-float names (<20M shares)
  3. Pull recent SEC filings (10-K / 10-Q) for each candidate
  4. Read filing text and extract risk factors / growth catalysts via Claude
  5. Compute RSI from 30d price history
  6. Output ranked "Best Stocks to Buy" list with buy thesis

Run:
  python -m screener.openbb_screener
  python -m screener.openbb_screener --tickers SYTA,UCAR,HUDI   # analyse specific tickers
  python -m screener.openbb_screener --min-change 20 --top 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("apex.screener")

from screener.ingestion.openbb_client  import get_gainers, get_quote, get_sec_filings, get_news
from screener.ingestion.finviz_client  import get_float_data, is_low_float
from screener.config                   import MAX_FLOAT_SHARES

# ── LLM client (free: Groq / Ollama) ──────────────────────────────────────────
from screener.ai.llm_client import chat as _llm_chat, is_available as _llm_ok

# ── Filing text fetcher (reuse EDGAR client) ──────────────────────────────────
from screener.ingestion.edgar_client import fetch_filing_text


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _analyse_filing(ticker: str, filing_url: str, form_type: str) -> dict:
    """
    Send filing text to Claude and extract:
      - key_risks       : top 3 risk factors
      - growth_catalysts: top 3 bullish catalysts
      - dilution_risk   : NONE / LOW / MEDIUM / HIGH / SEVERE
      - sentiment       : BULLISH / NEUTRAL / BEARISH
      - buy_thesis      : 1-sentence investment thesis
      - score_delta     : -10 to +10 adjustment to base score
    """
    if not filing_url:
        return _neutral_filing()

    text = fetch_filing_text(filing_url)
    if not text or len(text) < 200:
        return _neutral_filing()

    prompt = textwrap.dedent(f"""
        You are analysing a {form_type} SEC filing for {ticker}.

        Filing excerpt (first 6000 chars):
        {text[:6000]}

        Return a JSON object with exactly these keys:
        {{
          "key_risks": ["risk1", "risk2", "risk3"],
          "growth_catalysts": ["catalyst1", "catalyst2", "catalyst3"],
          "dilution_risk": "NONE|LOW|MEDIUM|HIGH|SEVERE",
          "sentiment": "BULLISH|NEUTRAL|BEARISH",
          "buy_thesis": "one sentence",
          "score_delta": <integer -10 to 10>
        }}

        Rules:
        - dilution_risk SEVERE if you see ATM offering, death-spiral convert, or shelf >50% of float
        - sentiment BULLISH only if genuine new revenue, contract, or product milestone
        - score_delta positive for clean balance sheet + real catalyst, negative for dilution/going concern
        - Return ONLY the JSON, no markdown, no explanation.
    """).strip()

    try:
        return _llm_chat(prompt, max_tokens=512, expect_json=True)
    except Exception as exc:
        logger.debug(f"[llm] {ticker} filing analysis failed: {exc}")
        return _neutral_filing()


def _neutral_filing() -> dict:
    return {
        "key_risks": [], "growth_catalysts": [],
        "dilution_risk": "UNKNOWN", "sentiment": "NEUTRAL",
        "buy_thesis": "No filing available for analysis.",
        "score_delta": 0,
    }


def _score_candidate(quote: dict, float_data: dict, filing_analysis: dict) -> int:
    """
    0-100 composite score:
      change_pct   0-25   (momentum today)
      volume_ratio 0-20   (relative volume)
      rsi          0-15   (technical confirmation: ideal 40-70)
      float        0-20   (smaller = better)
      filing       0-20   (Claude's score_delta + sentiment)
    """
    score = 0

    # Momentum (0-25)
    chg = float(quote.get("change_pct") or 0)
    score += min(25, max(0, int(chg / 4)))          # 100% gain → 25 pts

    # Relative volume (0-20)
    vr = float(quote.get("volume_ratio") or 0)
    score += min(20, int(vr * 1.5))                 # 13x vol → 20 pts

    # RSI (0-15) — reward 40-70 range (not overbought)
    rsi = quote.get("rsi")
    if rsi is not None:
        if 40 <= rsi <= 70:
            score += 15
        elif 30 <= rsi < 40 or 70 < rsi <= 80:
            score += 8
        elif rsi > 80:
            score += 2   # overbought — caution

    # Float (0-20)
    fl = float_data.get("float_shares") or float_data.get("shares_out") or 0
    if fl < 1_000_000:   score += 20
    elif fl < 3_000_000: score += 17
    elif fl < 5_000_000: score += 14
    elif fl < 10_000_000:score += 10
    elif fl < 15_000_000:score += 6
    elif fl < 20_000_000:score += 2

    # Filing Claude analysis (0-20)
    sentiment = filing_analysis.get("sentiment", "NEUTRAL")
    delta     = int(filing_analysis.get("score_delta", 0))
    dilution  = filing_analysis.get("dilution_risk", "UNKNOWN")
    base_llm  = {"BULLISH": 15, "NEUTRAL": 8, "BEARISH": 0}.get(sentiment, 8)
    dil_pen   = {"SEVERE": -15, "HIGH": -8, "MEDIUM": -3, "LOW": 0, "NONE": 0}.get(dilution, 0)
    score += max(0, min(20, base_llm + delta + dil_pen))

    return min(100, max(0, score))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN SCREENER
# ═════════════════════════════════════════════════════════════════════════════

def run_screener(tickers: list[str] | None = None,
                 min_change: float = 15.0,
                 top_n: int = 15,
                 include_filing: bool = True) -> list[dict]:
    """
    Full screening run. Returns ranked list of setup dicts.

    Parameters
    ----------
    tickers     : if provided, analyse these specific tickers instead of gainers
    min_change  : minimum % change today to include in gainers scan
    top_n       : cap number of candidates to analyse (saves API calls)
    include_filing : if False, skip SEC filing fetch (faster, lower quality)
    """
    print(f"\n{'='*60}")
    print(f"  APEX AI Stock Screener — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Powered by OpenBB + Claude")
    print(f"{'='*60}\n")

    # ── Step 1: Candidate list ────────────────────────────────────────────────
    if tickers:
        print(f"[1/4] Analysing {len(tickers)} specified tickers…")
        candidates = []
        for sym in tickers:
            q = get_quote(sym)
            candidates.append(q)
    else:
        print(f"[1/4] Fetching top gainers (≥{min_change}% today)…")
        candidates = get_gainers(min_change_pct=min_change, limit=100)
        print(f"      Found {len(candidates)} gainers — filtering to low-float…")

    # ── Step 2: Float filter ──────────────────────────────────────────────────
    print(f"[2/4] Checking float data…")
    low_float_candidates = []
    for q in candidates[:top_n * 3]:   # check up to 3× top_n to find enough
        ticker = q.get("ticker", "")
        if not ticker:
            continue
        float_data = get_float_data(ticker)
        if is_low_float(float_data):
            low_float_candidates.append((q, float_data))
            if len(low_float_candidates) >= top_n:
                break

    if not low_float_candidates:
        print("      No low-float candidates found today.")
        return []

    print(f"      {len(low_float_candidates)} low-float candidates pass filter")

    # ── Step 3: SEC filing + Claude analysis ──────────────────────────────────
    results = []
    for i, (quote, float_data) in enumerate(low_float_candidates):
        ticker = quote["ticker"]
        print(f"[3/4] Analysing {ticker} ({i+1}/{len(low_float_candidates)})…", end=" ")

        filing_analysis = _neutral_filing()
        filings = []

        if include_filing:
            # Try 10-Q first (more recent), fall back to 10-K
            for form in ["10-Q", "10-K"]:
                filings = get_sec_filings(ticker, form_type=form, limit=1)
                if filings:
                    break

            if filings:
                url = filings[0].get("url", "")
                filing_analysis = _analyse_filing(ticker, url, filings[0].get("form_type", ""))
                print(f"✓ {filing_analysis['sentiment']} | dilution={filing_analysis['dilution_risk']}")
            else:
                print("no filing found — momentum-only scoring")
        else:
            print("(filing skipped)")

        # RSI computation (if not already in quote)
        score = _score_candidate(quote, float_data, filing_analysis)

        results.append({
            "ticker":           ticker,
            "score":            score,
            "price":            quote.get("price"),
            "change_pct":       quote.get("change_pct"),
            "volume_ratio":     quote.get("volume_ratio"),
            "rsi":              quote.get("rsi"),
            "float_M":          round((float_data.get("float_shares") or float_data.get("shares_out") or 0) / 1e6, 2),
            "short_pct":        float_data.get("short_float_pct"),
            "sentiment":        filing_analysis["sentiment"],
            "dilution_risk":    filing_analysis["dilution_risk"],
            "buy_thesis":       filing_analysis["buy_thesis"],
            "growth_catalysts": filing_analysis["growth_catalysts"],
            "key_risks":        filing_analysis["key_risks"],
            "filing_date":      filings[0]["date"] if filings else None,
            "filing_url":       filings[0]["url"]  if filings else None,
            "form_type":        filings[0]["form_type"] if filings else "N/A",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def print_report(results: list[dict]) -> None:
    """Pretty-print the final ranked list."""
    if not results:
        print("\nNo qualifying setups found today.")
        return

    print(f"\n{'='*60}")
    print(f"  🏆 BEST STOCKS TO BUY — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*60}")

    for rank, r in enumerate(results, 1):
        score  = r["score"]
        ticker = r["ticker"]
        bar    = "█" * (score // 5) + "░" * (20 - score // 5)
        tier   = "🟢 STRONG BUY" if score >= 75 else "🟡 WATCH" if score >= 50 else "⚪ SKIP"

        print(f"\n#{rank}  {ticker:6s}  [{bar}] {score}/100  {tier}")
        print(f"    Price: ${r['price']:.2f}  Chg: {r['change_pct']:+.1f}%  "
              f"Vol: {r['volume_ratio']:.1f}×  RSI: {r['rsi'] or 'N/A'}  "
              f"Float: {r['float_M']:.1f}M  Short: {r['short_pct'] or '?'}%")
        print(f"    Sentiment: {r['sentiment']}  Dilution: {r['dilution_risk']}  "
              f"Filing: {r['form_type']} ({r['filing_date'] or 'N/A'})")

        if r["buy_thesis"] and r["buy_thesis"] != "No filing available for analysis.":
            print(f"    📌 Thesis: {r['buy_thesis']}")

        if r["growth_catalysts"]:
            print(f"    ✅ Catalysts:")
            for c in r["growth_catalysts"][:2]:
                print(f"       • {c}")

        if r["key_risks"]:
            print(f"    ⚠  Risks:")
            for rk in r["key_risks"][:2]:
                print(f"       • {rk}")

        if r["filing_url"]:
            print(f"    🔗 {r['filing_url']}")

    print(f"\n{'='*60}")
    print(f"  Disclaimer: Research only. Not financial advice.")
    print(f"{'='*60}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APEX AI Stock Screener")
    parser.add_argument("--tickers",    type=str,   default=None,
                        help="Comma-separated tickers to analyse (skips gainers scan)")
    parser.add_argument("--min-change", type=float, default=15.0,
                        help="Min %% change today for gainers scan (default 15)")
    parser.add_argument("--top",        type=int,   default=10,
                        help="Max candidates to analyse in depth (default 10)")
    parser.add_argument("--no-filing",  action="store_true",
                        help="Skip SEC filing fetch (faster, lower quality scores)")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    results = run_screener(
        tickers        = tickers,
        min_change     = args.min_change,
        top_n          = args.top,
        include_filing = not args.no_filing,
    )
    print_report(results)
