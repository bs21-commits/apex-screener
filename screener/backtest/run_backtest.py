"""
APEX Scoring Model — Historical Backtest

Replays the scoring model against known historical low-float movers and
measures whether a high score at signal time actually predicted future gains.

Methodology:
  1. Curated list of confirmed big-day movers, verified to have yfinance data
  2. Fetch daily OHLCV via yfinance for each ticker around the signal date
  3. Reconstruct what the score would have been at ~open on the signal day:
       - change_pct   = (open - prev_close) / prev_close * 100
       - volume_ratio = day volume / 30d avg volume
       - rsi          = Wilder RSI from 30 closes before signal date
       - float        = approximate float at event time (float changes slowly)
       - filing_delta = 0 (historical Claude calls not feasible at scale)
  4. Measure forward returns:
       - same-day  : close vs open
       - high-day  : intraday high vs open (best-case capture)
       - T+1day    : next close vs signal-day close
       - T+3day    : 3-day close vs signal-day close
  5. Aggregate by score bucket → hit rate / avg return

Run:
  python -m screener.backtest.run_backtest
  python -m screener.backtest.run_backtest --min-score 50
  python -m screener.backtest.run_backtest --export results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

import pandas as pd

# ── Historical setup list ─────────────────────────────────────────────────────
# Only tickers with verified yfinance data (still trading or recently delisted
# with archived data). signal_date = the major move day.
# float_M = approximate tradeable float on that date (in millions).
HISTORICAL_SETUPS = [
    # ticker   signal_date    catalyst                               float_M
    ("GME",  "2024-05-14",  "Roaring Kitty return tweet",            300.0),  # large — control
    ("GME",  "2021-01-27",  "WSB short squeeze peak",                69.0),   # large — control
    ("AMC",  "2021-06-02",  "Retail squeeze wave",                   450.0),  # large — control
    ("MVIS", "2021-05-07",  "Acquisition speculation",               14.0),
    ("MVIS", "2021-06-01",  "Microsoft HoloLens contract rumor",     14.0),
    ("CLOV", "2021-06-09",  "WSB target — short squeeze",            85.0),
    ("WISH", "2021-06-10",  "Reddit squeeze wave",                   180.0),
    ("SPCE", "2021-02-01",  "Virgin Galactic space tourism buzz",    170.0),
    ("ATOS", "2021-01-27",  "WSB wave + low price",                  120.0),
    ("WKHS", "2021-01-26",  "Government contract speculation",       55.0),
    ("SENS", "2021-02-09",  "Low float + options activity",          30.0),
    ("SNDL", "2021-02-10",  "Cannabis + WSB momentum",               800.0),  # large — control
    ("BBIG", "2021-09-01",  "NFT/crypto catalyst",                   35.0),
    ("PROG", "2021-11-03",  "Short squeeze — retail buying",         28.0),
    ("DWAC", "2021-10-21",  "Trump SPAC announcement",               25.0),
]


# ── OHLCV fetcher (yfinance direct — handles older data better) ───────────────
def _fetch_ohlcv(ticker: str, signal_date: str, lookback: int = 45) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        sig   = datetime.strptime(signal_date, "%Y-%m-%d")
        start = (sig - timedelta(days=lookback)).strftime("%Y-%m-%d")
        end   = (sig + timedelta(days=10)).strftime("%Y-%m-%d")
        df    = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df["date"] = df["date"].astype(str).str[:10]
        return df[["date","open","high","low","close","volume"]].copy()
    except Exception as exc:
        print(f"  [warn] {ticker} fetch: {exc}")
        return None


# ── RSI ───────────────────────────────────────────────────────────────────────
def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    d  = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    ag = sum(max(x,0) for x in d[:period]) / period
    al = sum(max(-x,0) for x in d[:period]) / period
    for i in range(period, len(d)):
        ag = (ag*(period-1) + max(d[i],0)) / period
        al = (al*(period-1) + max(-d[i],0)) / period
    return round(100 - 100/(1 + ag/al), 2) if al else 100.0


# ── Score reconstructor ───────────────────────────────────────────────────────
def _score(change_pct: float, vol_ratio: float, rsi: float | None, float_M: float) -> int:
    s = 0
    s += min(25, max(0, int(change_pct / 4)))
    s += min(20, int(vol_ratio * 1.5))
    if rsi is not None:
        s += 15 if 40 <= rsi <= 70 else 8 if (30 <= rsi < 40 or 70 < rsi <= 80) else 2 if rsi > 80 else 0
    f = float_M * 1e6
    s += 20 if f<1e6 else 17 if f<3e6 else 14 if f<5e6 else 10 if f<10e6 else 6 if f<15e6 else 2 if f<20e6 else 0
    return min(100, max(0, s))


# ── Main backtest ─────────────────────────────────────────────────────────────
def run_backtest(min_score: int = 0) -> pd.DataFrame:
    results = []

    for ticker, signal_date, catalyst, float_M in HISTORICAL_SETUPS:
        print(f"  {ticker:6s} {signal_date}  {catalyst[:38]:<38}…", end=" ", flush=True)

        df = _fetch_ohlcv(ticker, signal_date)
        if df is None or df.empty:
            print("no data"); continue

        # Find signal row (exact or nearest prior trading day)
        exact = df[df["date"] == signal_date]
        if exact.empty:
            df["dt"] = pd.to_datetime(df["date"])
            sig_dt   = pd.to_datetime(signal_date)
            prior    = df[df["dt"] <= sig_dt]
            if prior.empty:
                print("no prior date"); continue
            sig_idx = prior.index[-1]
        else:
            sig_idx = exact.index[0]

        if sig_idx == 0:
            print("no prev close"); continue

        sig        = df.iloc[sig_idx]
        prev_close = float(df.iloc[sig_idx-1]["close"])
        if prev_close == 0:
            print("zero prev close"); continue

        open_p  = float(sig["open"])
        close_p = float(sig["close"])
        high_p  = float(sig["high"])
        volume  = int(sig["volume"])
        actual_date = sig["date"]

        change_pct = round((open_p - prev_close) / prev_close * 100, 2)

        prior_rows = df.iloc[max(0, sig_idx-30):sig_idx]
        avg_vol   = int(prior_rows["volume"].mean()) if not prior_rows.empty else 1
        vol_ratio = round(volume / avg_vol, 1) if avg_vol else 1.0
        rsi_val   = _rsi(prior_rows["close"].tolist())

        score = _score(change_pct, vol_ratio, rsi_val, float_M)

        def _fwd(n):
            i = sig_idx + n
            return round((float(df.iloc[i]["close"]) - close_p) / close_p * 100, 2) if i < len(df) else None

        ret_same = round((close_p - open_p)  / open_p  * 100, 2) if open_p else None
        ret_high = round((high_p  - open_p)  / open_p  * 100, 2) if open_p else None

        if score >= min_score:
            row = {
                "ticker":      ticker,
                "signal_date": actual_date,
                "catalyst":    catalyst,
                "float_M":     float_M,
                "score":       score,
                "change_pct":  change_pct,
                "vol_ratio":   vol_ratio,
                "rsi":         rsi_val,
                "open":        round(open_p, 2),
                "close":       round(close_p, 2),
                "ret_same":    ret_same,
                "ret_high":    ret_high,
                "ret_1d":      _fwd(1),
                "ret_3d":      _fwd(3),
            }
            results.append(row)
            r1 = row["ret_1d"]
            print(f"score={score:3d}  same={ret_same:+6.1f}%  +1d={r1:+6.1f}%" if r1 is not None
                  else f"score={score:3d}  same={ret_same:+6.1f}%  +1d=N/A")
        else:
            print(f"score={score} (below {min_score})")

    return pd.DataFrame(results)


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(df: pd.DataFrame) -> None:
    if df.empty:
        print("\nNo results to report.")
        return

    print(f"\n{'='*72}")
    print(f"  APEX BACKTEST — {len(df)} setups  |  scoring excludes filing delta")
    print(f"{'='*72}")

    cols = ["ticker","signal_date","score","float_M","change_pct","vol_ratio","rsi",
            "ret_same","ret_high","ret_1d","ret_3d"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))

    print(f"\n{'─'*72}")
    print(f"  PERFORMANCE BY SCORE BUCKET")
    print(f"{'─'*72}")

    for lo, hi, label in [(70,100,"🟢 HIGH  (70-100)"),
                          (45, 69,"🟡 MED   (45-69)"),
                          (0,  44,"⚪ LOW   (0-44)")]:
        sub = df[(df["score"]>=lo)&(df["score"]<=hi)]
        if sub.empty: continue
        n = len(sub)
        print(f"\n  {label}  (n={n})")
        for col, name in [("ret_same","Same-day"),("ret_high","Intraday hi"),
                          ("ret_1d","T+1 Day"),("ret_3d","T+3 Days")]:
            vals = sub[col].dropna()
            if vals.empty: continue
            print(f"    {name:12s}  hit={round((vals>0).mean()*100):3.0f}%  "
                  f"avg={vals.mean():+6.1f}%  med={vals.median():+6.1f}%  "
                  f"best={vals.max():+6.1f}%  worst={vals.min():+6.1f}%")

    print(f"\n{'─'*72}")
    print(f"  OVERALL ({len(df)} setups)")
    for col, name in [("ret_same","Same-day"),("ret_high","Intraday hi"),
                      ("ret_1d","T+1 Day"),("ret_3d","T+3 Days")]:
        vals = df[col].dropna()
        if vals.empty: continue
        print(f"    {name:12s}  hit={round((vals>0).mean()*100):3.0f}%  "
              f"avg={vals.mean():+6.1f}%  med={vals.median():+6.1f}%  "
              f"best={vals.max():+6.1f}%  worst={vals.min():+6.1f}%")

    print(f"\n  Note: Scores here exclude live filing analysis — add ~0-15 pts")
    print(f"        for genuine SEC catalysts in production scoring.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=int, default=0)
    ap.add_argument("--export",    type=str, default=None)
    args = ap.parse_args()

    print(f"\nRunning backtest on {len(HISTORICAL_SETUPS)} historical setups…\n")
    df = run_backtest(min_score=args.min_score)
    if args.export and not df.empty:
        df.to_csv(args.export, index=False)
        print(f"\nExported → {args.export}")
    print_report(df)
