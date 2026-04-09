# Step 1 — path bootstrap + load .env
import sys, os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
try:
    from dotenv import load_dotenv
    load_dotenv(_os.path.join(_ROOT, ".env"), override=False)
except ImportError:
    pass
# Streamlit Cloud: push st.secrets into os.environ so all modules pick them up
try:
    import streamlit as _st_env
    for _k, _v in _st_env.secrets.items():
        if _k not in _os.environ:
            _os.environ[_k] = str(_v)
except Exception:
    pass

# Step 2 — Streamlit + page config FIRST
import streamlit as st
st.set_page_config(
    page_title="APEX Low-Float Screener",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
.block-container{padding-top:.6rem;padding-bottom:0}
[data-testid="stSidebar"]>div{padding-top:.8rem}
.score-pill{display:inline-block;padding:2px 10px;border-radius:12px;
  font-weight:700;font-size:.85rem;font-family:monospace}
.score-green {background:#0d3b1e;color:#00e676;border:1px solid #00e676}
.score-orange{background:#3b2200;color:#ffaa00;border:1px solid #ffaa00}
.score-gray  {background:#222;   color:#888;   border:1px solid #444}
.tox-card{background:#2d0000;border:1px solid #ff4444;border-radius:6px;
  padding:8px 12px;margin-bottom:4px;font-size:.8rem;color:#ff8888}
.bull-card{background:#002d10;border:1px solid #00c853;border-radius:6px;
  padding:8px 12px;margin-bottom:4px;font-size:.8rem;color:#69f0ae}
.halt-badge{background:#ff4444;color:#fff;font-weight:700;font-size:.75rem;
  padding:2px 7px;border-radius:4px;margin-right:4px}
.scenario-box{border-radius:8px;padding:12px 16px;margin-bottom:8px;text-align:center}
.sc-bear{background:#2d0000;border:1px solid #ff4444}
.sc-base{background:#1a1a00;border:1px solid #ffaa00}
.sc-bull{background:#002d10;border:1px solid #00e676}
.sc-label{font-size:.7rem;color:#aaa;text-transform:uppercase;letter-spacing:.08em}
.sc-price{font-size:1.5rem;font-weight:800;font-family:monospace;margin:4px 0}
.sc-pct  {font-size:.85rem;font-weight:600}
div[data-testid="stDataFrame"]{font-size:.8rem}
.source-badge{font-size:.68rem;padding:1px 6px;border-radius:4px;font-weight:600}
.src-gainer{background:#1a3a00;color:#76ff03;border:1px solid #76ff03}
.src-edgar {background:#003a4a;color:#40c4ff;border:1px solid #40c4ff}
.action-buy{background:#003a1a;color:#00e676;border:1px solid #00e676;
  border-radius:5px;padding:3px 8px;font-weight:700;font-size:.78rem}
.action-watch{background:#2a2200;color:#ffaa00;border:1px solid #ffaa00;
  border-radius:5px;padding:3px 8px;font-weight:700;font-size:.78rem}
.action-avoid{background:#2d0000;color:#ff4444;border:1px solid #ff4444;
  border-radius:5px;padding:3px 8px;font-weight:700;font-size:.78rem}
.tmr-card{border-radius:8px;padding:12px;margin-bottom:8px;border:1px solid #333}
.tmr-buy  {background:#071f0e;border-color:#00c853}
.tmr-watch{background:#1f1500;border-color:#ffaa00}
.tmr-avoid{background:#1f0000;border-color:#ff4444}
.disclaimer{background:#111;color:#555;border-radius:4px;
  padding:8px;font-size:.68rem;line-height:1.4}
</style>""", unsafe_allow_html=True)

# Step 3 — all other imports
import time
from datetime import datetime
import pandas as pd
from screener.main                      import run_scan_cycle
from screener.ingestion.openbb_client   import get_luld_halts, get_quote, get_sec_filings, get_gainers, get_news, batch_enrich_rsi
from screener.ingestion.gainers_scanner import momentum_scenarios
from screener.logger.backtest_logger    import load_alerts_df, backtest_summary
from screener.scoring.engine            import score_dataframe
from screener.openbb_screener           import _analyse_filing
from screener.ai.market_analyst         import analyse_ticker, entry_signals, auto_populate_watchlist
from screener.ai.tomorrow_predictor     import predict_tomorrow
from screener.ai.llm_client             import is_available as llm_available, backend_name as llm_backend


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"setups": [], "last_scan": None, "session_alerts": [],
             "deep_analysis": {}, "raw_gainers": [], "tmr_watchlist": [],
             "tomorrow_picks": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ APEX Screener")
    st.caption("Low-Float Momentum — SEC + Market Feed")
    st.divider()

    auto_refresh  = st.toggle("Auto-refresh (60s)", value=True)
    manual_scan   = st.button("▶ Scan Now", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**Filters**")
    min_score     = st.slider("Min Score",         0, 100,  0)
    max_float_m   = st.slider("Max Float (M sh.)", 1,  20, 20)
    min_vol_ratio = st.slider("Min Vol Ratio",     1,  20,  1)
    min_change    = st.slider("Min % Change Today",0, 200,  5)
    bullish_only  = st.checkbox("Bullish only (BUY ZONE signal only)", value=False)

    st.divider()
    halts = get_luld_halts()
    if halts:
        st.markdown("**🚨 Active Halts**")
        for h in halts:
            st.markdown(
                f"<span class='halt-badge'>{h['ticker']}</span>"
                f"<span style='font-size:.75rem;color:#aaa'> {h.get('halt_type','?')} — {h.get('status','?')}</span>",
                unsafe_allow_html=True)
    else:
        st.caption("No active halts")

    st.divider()
    st.markdown("<div class='disclaimer'>Research only. Not financial advice.</div>",
                unsafe_allow_html=True)


# ── Scan trigger ──────────────────────────────────────────────────────────────
should_scan = manual_scan or (st.session_state.last_scan is None)
if auto_refresh and st.session_state.last_scan:
    if (datetime.now() - st.session_state.last_scan).total_seconds() >= 60:
        should_scan = True

if should_scan:
    with st.spinner("Scanning market + SEC EDGAR…"):
        new_setups  = run_scan_cycle(use_mock=False)
        # "All Movers" always uses a fixed low threshold (2%) — sidebar filter applies to Today's Setups only
        raw_gainers = get_gainers(min_change_pct=2.0, limit=250)
        # Enrich with RSI via single batch yfinance download
        with st.spinner("Computing RSI for all movers…"):
            raw_gainers = batch_enrich_rsi(raw_gainers)
        # Auto-populate rule-based deep_analysis for today's movers (detail panel fallback)
        watchlist_batch = auto_populate_watchlist(raw_gainers, top_n=30)
        for t, analysis in watchlist_batch.items():
            if t not in st.session_state["deep_analysis"]:
                st.session_state["deep_analysis"][t] = analysis

        # Forward-looking multi-signal prediction for tomorrow
        with st.spinner("Predicting tomorrow's movers (earnings + 8-K + small caps + continuation)…"):
            tomorrow_picks = predict_tomorrow(raw_gainers)
        st.session_state["tomorrow_picks"] = tomorrow_picks

        st.session_state.setups      = new_setups
        st.session_state.raw_gainers = raw_gainers
        st.session_state.last_scan   = datetime.now()
        for s in new_setups:
            if s.get("score", 0) >= 80:
                entry = f"{s['ticker']} {s['score']}/100 @ {datetime.now().strftime('%H:%M:%S')}"
                if entry not in st.session_state.session_alerts:
                    st.session_state.session_alerts.append(entry)

setups       = st.session_state.setups or []
raw_gainers  = st.session_state.raw_gainers or []
halt_tickers = [h["ticker"] for h in halts]


# ── Headline metrics ──────────────────────────────────────────────────────────
st.markdown("## ⚡ APEX Low-Float Screener")
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    ts = st.session_state.last_scan
    st.metric("Last Scan", ts.strftime("%H:%M:%S") if ts else "—")
with m2:
    gainers_ct = sum(1 for s in setups if s.get("source") == "gainers")
    edgar_ct   = len(setups) - gainers_ct
    st.metric("Live Movers", gainers_ct, delta=f"+{edgar_ct} SEC catalysts" if edgar_ct else None)
with m3:
    n_high = sum(1 for s in setups if s.get("score", 0) >= 80)
    st.metric("High-Score (≥80)", n_high, delta=f"+{n_high}" if n_high else None)
with m4:
    st.metric("Active LULD Halts", len(halts))
with m5:
    alerts_df = load_alerts_df()
    st.metric("Alerts Logged", len(alerts_df))

st.divider()


# ── Apply sidebar filters ────────────────────────────────────────────────────
def _apply_filters(rows):
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.empty:
        return df
    if "float_shares" in df.columns:
        df = df[df["float_shares"].fillna(float("inf")) <= max_float_m * 1_000_000]
    if "volume_ratio" in df.columns:
        df = df[df["volume_ratio"].fillna(0) >= min_vol_ratio]
    if "score" in df.columns:
        df = df[df["score"] >= min_score]
    if "change_pct" in df.columns:
        df = df[df["change_pct"].fillna(0) >= min_change]
    if bullish_only and "sentiment" in df.columns:
        df = df[df["sentiment"] == "BULLISH"]
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ── Format helpers ────────────────────────────────────────────────────────────
def _pill(s):
    s = int(s or 0)
    cls = "score-green" if s >= 80 else "score-orange" if s >= 60 else "score-gray"
    return f"<span class='score-pill {cls}'>{s}</span>"

def _fmt_price(v):  return f"${float(v):.2f}" if pd.notna(v) else "?"
def _fmt_pct(v):    return f"{float(v):+.1f}%" if pd.notna(v) else "?"
def _fmt_vol(v):    return f"{float(v):.1f}×"  if pd.notna(v) else "?"
def _fmt_float(v):  return f"{float(v)/1e6:.1f}M" if pd.notna(v) and v else "?"
def _fmt_short(v):  return f"{float(v):.1f}%" if pd.notna(v) and v else "?"
def _source_badge(src):
    if src == "gainers":
        return "<span class='source-badge src-gainer'>🔥 MOVER</span>"
    return "<span class='source-badge src-edgar'>📡 SEC</span>"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
col_table, col_detail = st.columns([60, 40], gap="large")


# ── LEFT: Unified setups table ───────────────────────────────────────────────
with col_table:
    st.markdown("### 🔥 Today's Setups")
    st.caption("Live market movers + SEC filing catalysts — both feeds merged, sorted by score")

    df = _apply_filters(setups)

    if df.empty:
        if not setups:
            st.info("Click **▶ Scan Now** to pull the latest movers and filings.")
        else:
            st.warning("No setups match current filters. Try lowering Min Score or Min % Change.")
    else:
        # Build display table
        def _fmt_rsi(v):
            if pd.isna(v) or v is None: return "—"
            v = float(v)
            if v >= 80: return f"🔴{v:.0f}"
            if v >= 70: return f"🟡{v:.0f}"
            if v >= 40: return f"🟢{v:.0f}"
            return f"⚪{v:.0f}"

        display = pd.DataFrame({
            "":         df["source"].apply(_source_badge),
            "Halt":     df["ticker"].apply(lambda t: "🔴" if t in halt_tickers else ""),
            "Ticker":   df["ticker"],
            "Score":    df["score"],
            "Price":    df["price"].apply(_fmt_price),
            "Chg %":    df["change_pct"].apply(_fmt_pct),
            "Vol":      df["volume_ratio"].apply(_fmt_vol),
            "RSI":      df["rsi"].apply(_fmt_rsi) if "rsi" in df.columns else "—",
            "Float":    df["float_shares"].apply(_fmt_float),
            "Short %":  df["short_pct"].apply(_fmt_short),
            "Type":     df["form_type"].str.replace("_", " ", regex=False),
            "Sentiment":df["sentiment"],
        })

        def _row_bg(row):
            s = row["Score"]
            if s >= 80: return ["background:#071f0e"]*len(row)
            if s >= 60: return ["background:#1f1500"]*len(row)
            return [""]*len(row)

        styled = (display.style
            .apply(_row_bg, axis=1)
            .applymap(lambda v: "color:#00e676" if v == "BULLISH" else "color:#ff4444" if v == "BEARISH" else "color:#888",
                      subset=["Sentiment"])
            .format({"Score": "{:.0f}"}))

        st.dataframe(styled, use_container_width=True, hide_index=True,
                     height=min(55 + len(display) * 36, 480))

        selected = st.selectbox("Select ticker for detail →",
                                options=df["ticker"].tolist(), key="ticker_select")
        if selected:
            st.session_state["_selected_row"] = df[df["ticker"] == selected].iloc[0].to_dict()

    if st.session_state.session_alerts:
        with st.expander(f"🔔 Session Alerts ({len(st.session_state.session_alerts)})", expanded=False):
            for e in reversed(st.session_state.session_alerts[-30:]):
                st.caption(f"🔔 {e}")

    # ── All-movers watchlist ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 All Movers Today")

    if not raw_gainers:
        st.caption("No data yet — click ▶ Scan Now.")
    else:
        wdf = pd.DataFrame(raw_gainers)

        # Ensure required columns exist
        for col, default in [("volume_ratio", 1.0), ("market_cap", None),
                              ("name", None), ("rsi", None)]:
            if col not in wdf.columns:
                wdf[col] = default
        if wdf["name"].isna().all():
            wdf["name"] = wdf["ticker"]

        # ── Quick inline signal (no VWAP needed) ─────────────────────────────
        def _quick_signal(rsi, vol_ratio, chg_pct):
            rsi = float(rsi) if rsi is not None else None
            vol_ratio = float(vol_ratio or 1)
            chg_pct   = float(chg_pct or 0)
            if rsi is not None:
                if rsi > 82:          return "🔴 EXTENDED"
                if rsi > 72:          return "🟡 WATCH"
                if rsi >= 40 and vol_ratio >= 5:  return "🟢 BUY ZONE"
                if rsi < 35:          return "🔵 OVERSOLD"
            else:
                if vol_ratio >= 10 and chg_pct >= 20: return "🟢 BUY ZONE"
                if vol_ratio >= 5:    return "🟡 WATCH"
            return "⚪ WEAK"

        def _rsi_fmt(v):
            if v is None or pd.isna(v): return "—"
            v = float(v)
            if v >= 80: return f"🔴 {v:.0f}"
            if v >= 70: return f"🟡 {v:.0f}"
            if v >= 40: return f"🟢 {v:.0f}"
            return f"🔵 {v:.0f}"

        def _mktcap(v):
            if v is None or (hasattr(v, '__float__') and pd.isna(float(v))): return "?"
            try:
                v = float(v)
                if v >= 1e9:  return f"${v/1e9:.1f}B"
                if v >= 1e6:  return f"${v/1e6:.0f}M"
                return f"${v:.0f}"
            except Exception:
                return "?"

        # ── Deep-analysis action badge (if already analysed) ──────────────────
        def _tmr_badge(ticker):
            d = st.session_state["deep_analysis"].get(ticker)
            if not d: return ""
            a = d.get("tomorrow_action", "")
            m = {"BUY_AT_OPEN": "🟢 BUY OPEN", "WATCH_OPEN": "🟡 WATCH",
                 "AVOID": "🔴 AVOID", "SHORT_CANDIDATE": "🔴 SHORT"}
            return m.get(a, "")

        wdf["_signal"] = wdf.apply(
            lambda r: _quick_signal(r.get("rsi"), r.get("volume_ratio"), r.get("change_pct")), axis=1)
        wdf["_tmr"]    = wdf["ticker"].apply(_tmr_badge)

        # ── Apply ALL sidebar filters to this table ───────────────────────────
        display_df = wdf.copy()
        display_df = display_df[display_df["change_pct"].fillna(0)   >= min_change]
        display_df = display_df[display_df["volume_ratio"].fillna(0) >= min_vol_ratio]
        if bullish_only:
            # Bullish only = only stocks with BUY ZONE signal (healthy RSI + volume confirmed)
            display_df = display_df[display_df["_signal"].str.contains("BUY ZONE", na=False)]
        # Always strip pure weak/noise rows (unless user explicitly lowered all filters)
        if min_vol_ratio >= 2:
            display_df = display_df[~display_df["_signal"].str.contains("WEAK", na=False)]

        # ── Text search ──────────────────────────────────────────────────────
        search = st.text_input("🔍 Search ticker / name", placeholder="e.g. AXTI or Hut", key="movers_search")
        if search.strip():
            s = search.strip().upper()
            display_df = display_df[
                display_df["ticker"].str.contains(s, case=False, na=False) |
                display_df["name"].str.contains(search.strip(), case=False, na=False)
            ]

        display_df = display_df.reset_index(drop=True)
        total_raw  = len(wdf)
        total_show = len(display_df)

        active_filters = []
        if min_change  > 2:  active_filters.append(f"≥{min_change}% change")
        if min_vol_ratio > 1: active_filters.append(f"≥{min_vol_ratio}× vol")
        if bullish_only:      active_filters.append("BUY ZONE only")
        filter_str = " · ".join(active_filters) if active_filters else "no filters"
        st.caption(
            f"Showing **{total_show}** of {total_raw} movers today — filtered by: {filter_str}. "
            f"Select a ticker below to load the detail panel →"
        )

        if display_df.empty:
            st.info("No stocks match the current filters. Try lowering Min Vol Ratio or Min % Change in the sidebar.")
        else:
            w_display = pd.DataFrame({
                "#":         range(1, len(display_df) + 1),
                "Ticker":    display_df["ticker"],
                "Name":      display_df["name"].str[:22],
                "Price":     display_df["price"].apply(_fmt_price),
                "Chg %":     display_df["change_pct"].apply(_fmt_pct),
                "RSI":       display_df["rsi"].apply(_rsi_fmt),
                "Vol Ratio": display_df["volume_ratio"].apply(_fmt_vol),
                "Volume":    display_df["volume"].apply(
                                 lambda v: f"{int(v)/1e6:.1f}M" if pd.notna(v) and v else "?"),
                "Mkt Cap":   display_df["market_cap"].apply(_mktcap),
                "Signal":    display_df["_signal"],
                "AI Pick":   display_df["_tmr"],
            }).reset_index(drop=True)

            def _row_signal_color(row):
                sig = display_df.iloc[row.name]["_signal"] if row.name < len(display_df) else ""
                if "BUY ZONE"  in sig: return ["background:#071a0e"] * len(row)
                if "EXTENDED"  in sig: return ["background:#1a0a00"] * len(row)
                chg = float(display_df.iloc[row.name]["change_pct"]) if row.name < len(display_df) else 0
                if chg >= 30: return ["background:#0a1500"] * len(row)
                return [""] * len(row)

            styled_w = w_display.style.apply(_row_signal_color, axis=1)
            st.dataframe(styled_w, use_container_width=True, hide_index=True,
                         height=min(60 + len(w_display) * 35, 600))

        # Ticker selector — only from the filtered list
        if not display_df.empty:
            movers_pick = st.selectbox(
                "Select a mover for deep analysis →",
                options=[""] + display_df["ticker"].tolist(),
                key="movers_select",
                format_func=lambda t: (
                    f"{t} — {display_df.loc[display_df['ticker']==t,'name'].values[0][:25]}  "
                    f"({display_df.loc[display_df['ticker']==t,'change_pct'].values[0]:+.1f}%)"
                    if t else "— pick a ticker —"
                ),
            )
            if movers_pick:
                g_row = display_df[display_df["ticker"] == movers_pick].iloc[0].to_dict()
                synthetic = {
                    "ticker":          g_row["ticker"],
                    "score":           0,
                    "source":          "gainers",
                    "price":           float(g_row.get("price") or 0),
                    "change_pct":      float(g_row.get("change_pct") or 0),
                    "volume_ratio":    float(g_row.get("volume_ratio") or 1),
                    "rsi":             g_row.get("rsi"),
                    "float_shares":    None,
                    "short_pct":       None,
                    "vwap":            None,
                    "form_type":       "—",
                    "sentiment":       "NEUTRAL",
                    "score_breakdown": {},
                    "scenarios":       None,
                }
                st.session_state["_selected_row"] = synthetic


# ── RIGHT: Detail + Momentum Scenarios ───────────────────────────────────────
with col_detail:
    row = st.session_state.get("_selected_row")

    if not row:
        st.markdown("### Select a ticker →")
        st.caption("Click any row in the table to see price scenarios, score breakdown, and filing analysis.")
    else:
        ticker    = row.get("ticker", "?")
        score     = int(row.get("score", 0))
        sentiment = row.get("sentiment", "NEUTRAL")
        src       = row.get("source", "edgar")
        price_now = float(row.get("price") or 0)
        chg       = float(row.get("change_pct") or 0)

        score_cls = "score-green" if score >= 80 else "score-orange" if score >= 60 else "score-gray"
        s_color   = {"BULLISH":"#00e676","BEARISH":"#ff4444"}.get(sentiment,"#aaa")

        st.markdown(
            f"<h3 style='margin-bottom:2px'>${ticker} &nbsp;"
            f"<span class='score-pill {score_cls}'>{score}/100</span>&nbsp;"
            f"{_source_badge(src)}</h3>"
            f"<span style='color:{s_color};font-weight:600'>{sentiment}</span>"
            f"&emsp;|&emsp;<span style='color:#888'>{row.get('form_type','')}</span>",
            unsafe_allow_html=True)

        # Live price refresh
        live = get_quote(ticker)
        live_price = live.get("price", price_now)
        live_chg   = live.get("change_pct", chg)

        dm1, dm2, dm3, dm4 = st.columns(4)
        with dm1: st.metric("Live Price", f"${live_price:.2f}", delta=f"{live_chg:+.1f}%")
        with dm2: st.metric("Vol Ratio",  _fmt_vol(row.get("volume_ratio")))
        with dm3: st.metric("Float",      _fmt_float(row.get("float_shares")))
        with dm4: st.metric("Short %",    _fmt_short(row.get("short_pct")))

        st.divider()

        # ── Momentum Scenarios ────────────────────────────────────────────────
        st.markdown("**📈 Intraday Price Scenarios**")
        sc = row.get("scenarios") or momentum_scenarios(
            price        = live_price,
            float_shares = row.get("float_shares"),
            volume_ratio = float(row.get("volume_ratio") or 1),
            short_pct    = row.get("short_pct"),
        )

        bear_pct = round((sc["conservative"] / live_price - 1) * 100, 1) if live_price else 0
        base_pct = round((sc["base"]         / live_price - 1) * 100, 1) if live_price else 0
        bull_pct = round((sc["aggressive"]   / live_price - 1) * 100, 1) if live_price else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"<div class='scenario-box sc-bear'>"
                f"<div class='sc-label'>Conservative</div>"
                f"<div class='sc-price' style='color:#ff8888'>${sc['conservative']:.2f}</div>"
                f"<div class='sc-pct' style='color:#ff4444'>{bear_pct:+.1f}%</div>"
                f"</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(
                f"<div class='scenario-box sc-base'>"
                f"<div class='sc-label'>Base Case</div>"
                f"<div class='sc-price' style='color:#ffcc80'>${sc['base']:.2f}</div>"
                f"<div class='sc-pct' style='color:#ffaa00'>{base_pct:+.1f}%</div>"
                f"</div>", unsafe_allow_html=True)
        with c3:
            st.markdown(
                f"<div class='scenario-box sc-bull'>"
                f"<div class='sc-label'>Squeeze Target</div>"
                f"<div class='sc-price' style='color:#69f0ae'>${sc['aggressive']:.2f}</div>"
                f"<div class='sc-pct' style='color:#00e676'>{bull_pct:+.1f}%</div>"
                f"</div>", unsafe_allow_html=True)

        float_M = (row.get("float_shares") or 5_000_000) / 1_000_000
        si      = row.get("short_pct") or 0
        st.caption(
            f"Scenarios based on {float_M:.1f}M float · {row.get('volume_ratio',1):.1f}× vol · {si:.1f}% short interest. "
            f"Not a price guarantee — low-float stocks can reverse sharply.")

        st.divider()

        # ── Score breakdown ───────────────────────────────────────────────────
        st.markdown("**Score Breakdown**")
        bd = row.get("score_breakdown") or {}
        for label, val, max_val, color in [
            ("Volume",   bd.get("volume_sub", 0), 30, "#4fc3f7"),
            ("Float",    bd.get("float_sub",  0), 20, "#81c784"),
            ("Catalyst", bd.get("llm_sub",    0), 40, "#ffb74d"),
            ("Bonus",    bd.get("bonus_sub",  0), 10, "#ce93d8"),
        ]:
            pct = int((val / max_val) * 100) if max_val else 0
            st.markdown(
                f"<div style='margin-bottom:5px'>"
                f"<div style='display:flex;justify-content:space-between;font-size:.75rem;color:#aaa;margin-bottom:2px'>"
                f"<span>{label}</span><span>{val}/{max_val}</span></div>"
                f"<div style='background:#1a1a1a;border-radius:4px;height:7px'>"
                f"<div style='background:{color};width:{pct}%;height:7px;border-radius:4px'></div>"
                f"</div></div>", unsafe_allow_html=True)

        st.divider()

        # ── Claude analysis ───────────────────────────────────────────────────
        summary = row.get("llm_summary", "")
        if summary and not summary.startswith("["):
            st.markdown("**Claude's Analysis**")
            st.info(summary)

        # ── Toxicity / bullish signal cards ──────────────────────────────────
        flags = row.get("toxicity_flags") or []
        if isinstance(flags, str): flags = [f for f in flags.split("|") if f]
        if flags:
            st.markdown("**⚠ Toxicity Flags**")
            for flag in flags:
                st.markdown(f"<div class='tox-card'><b>{flag.replace('_',' ').title()}</b></div>",
                            unsafe_allow_html=True)

        bulls = row.get("bullish_signals") or []
        if isinstance(bulls, str): bulls = [b for b in bulls.split("|") if b]
        if bulls:
            st.markdown("**✓ Bullish Signals**")
            for sig in bulls:
                st.markdown(f"<div class='bull-card'><b>{sig.replace('_',' ').title()}</b></div>",
                            unsafe_allow_html=True)

        filing_url = row.get("filing_url", "")
        if filing_url and filing_url.startswith("https://www.sec.gov"):
            st.markdown(f"[📄 View SEC Filing]({filing_url})")

        # ── Entry timing signals ──────────────────────────────────────────────
        st.divider()
        st.markdown("**⏱ When to Enter**")
        signals = entry_signals(
            price        = live_price,
            vwap         = float(row.get("vwap") or live_price),
            rsi          = row.get("rsi"),
            volume_ratio = float(row.get("volume_ratio") or 1),
            change_pct   = live_chg,
            short_pct    = row.get("short_pct"),
        )
        action_colors = {"BUY_ZONE":"#00e676","WATCH":"#ffaa00","WAIT":"#ff8800","OVERSOLD":"#40c4ff"}
        ac = action_colors.get(signals["action"], "#aaa")
        st.markdown(f"<span style='font-size:1.1rem;font-weight:800;color:{ac}'>"
                    f"● {signals['action'].replace('_',' ')}</span> &nbsp;"
                    f"<span style='color:#666;font-size:.8rem'>{signals['zone']}</span>",
                    unsafe_allow_html=True)
        for sig in signals["signals"]:
            st.caption(f"• {sig}")
        if signals["entry_note"]:
            st.info(signals["entry_note"], icon="📍")

        st.divider()

        # ── OpenBB Deep Analysis (on-demand) ──────────────────────────────────
        st.markdown("**🤖 AI Deep Analysis**")
        deep = st.session_state["deep_analysis"].get(ticker)

        if deep:
            # ── Why it moved ──────────────────────────────────────────────────
            why = deep.get("why_moved") or deep.get("buy_thesis", "")
            if why and not why.startswith("No filing"):
                st.markdown("**💡 Why It Moved**")
                st.info(why)

            # ── Tomorrow's outlook ────────────────────────────────────────────
            tmr_action = deep.get("tomorrow_action", "WATCH_OPEN")
            tmr_colors = {"BUY_AT_OPEN":"#00e676","WATCH_OPEN":"#ffaa00",
                          "AVOID":"#ff4444","SHORT_CANDIDATE":"#ff8800"}
            tc = tmr_colors.get(tmr_action, "#aaa")
            outlook = deep.get("tomorrow_outlook","")
            if outlook:
                st.markdown("**📅 Tomorrow's Outlook**")
                st.markdown(f"<span style='font-weight:800;color:{tc};font-size:1rem'>"
                            f"→ {tmr_action.replace('_',' ')}</span>",
                            unsafe_allow_html=True)
                st.caption(outlook)

            ideal = deep.get("ideal_entry","")
            if ideal:
                st.markdown(f"**📍 Ideal Entry:** {ideal}")

            conf = deep.get("confidence","?")
            cat  = deep.get("catalyst_type","?")
            st.caption(f"Catalyst: **{cat}**  |  Confidence: **{conf}**")

            risks = deep.get("key_risks") or deep.get("risk_factors",[])
            if risks:
                with st.expander("⚠ Risk Factors"):
                    for r in risks:
                        st.markdown(f"<div class='tox-card'>{r}</div>", unsafe_allow_html=True)

            cats = deep.get("growth_catalysts", [])
            if cats:
                with st.expander("✅ Growth Catalysts"):
                    for c in cats:
                        st.markdown(f"<div class='bull-card'>{c}</div>", unsafe_allow_html=True)
        else:
            if st.button("🔍 Run Deep Analysis (Claude + News + SEC)", key=f"deep_{ticker}",
                         use_container_width=True):
                with st.spinner(f"Fetching news + SEC filing + running Claude for {ticker}…"):
                    # Get news
                    news = get_news(ticker, limit=8)
                    headlines = [n["title"] for n in news]
                    # Get SEC filing summary
                    filings = get_sec_filings(ticker, "10-Q", limit=1) or \
                              get_sec_filings(ticker, "10-K", limit=1)
                    filing_url_deep = filings[0]["url"] if filings else filing_url
                    form = filings[0]["form_type"] if filings else row.get("form_type","8-K")
                    sec_analysis = _analyse_filing(ticker, filing_url_deep, form)
                    filing_summary = sec_analysis.get("buy_thesis","")
                    # Run market analyst
                    float_M = (row.get("float_shares") or 0) / 1e6 or None
                    analysis = analyse_ticker(
                        ticker       = ticker,
                        change_pct   = live_chg,
                        volume_ratio = float(row.get("volume_ratio") or 1),
                        float_M      = float_M,
                        short_pct    = row.get("short_pct"),
                        news_headlines = headlines,
                        filing_summary = filing_summary,
                    )
                    # Merge SEC fields in
                    analysis["growth_catalysts"] = sec_analysis.get("growth_catalysts",[])
                    analysis["key_risks"] = (analysis.get("risk_factors",[]) +
                                             sec_analysis.get("key_risks",[]))
                    st.session_state["deep_analysis"][ticker] = analysis
                st.rerun()


# ── Tomorrow's Best Buys ─────────────────────────────────────────────────────
st.divider()
st.markdown("## 🎯 Tomorrow's Best Buys")

tomorrow_picks = st.session_state.get("tomorrow_picks", [])
_conf_color = {"HIGH": "#00e676", "MEDIUM": "#ffaa00", "LOW": "#888"}
_conf_dot   = {"HIGH": "●●●",    "MEDIUM": "●●○",     "LOW": "●○○"}
_cat_emoji  = {
    "EARNINGS":      "📢",
    "SEC_CATALYST":  "📄",
    "CONTINUATION":  "🔥",
    "SQUEEZE":       "💥",
    "MULTI_SIGNAL":  "⚡",
    "OPTIONS_FLOW":  "🎯",
}

if not tomorrow_picks:
    st.info("Click **▶ Scan Now** to generate picks — pulls earnings tomorrow, 8-K filings, small-cap momentum, and continuation setups.")
else:
    buys    = [p for p in tomorrow_picks if p.get("tomorrow_action") == "BUY_AT_OPEN"]
    watches = [p for p in tomorrow_picks if p.get("tomorrow_action") != "BUY_AT_OPEN"]

    n_high   = sum(1 for p in tomorrow_picks if p.get("confidence") == "HIGH")
    has_ai   = llm_available()
    backend  = llm_backend() if has_ai else "rule-based"
    is_ruled = all(p.get("_source") == "RULE_BASED" for p in tomorrow_picks)

    if is_ruled and not has_ai:
        st.warning(
            "⚡ **Rule-based picks** — no AI key found. "
            "Set **GROQ_API_KEY** free at [console.groq.com](https://console.groq.com) → App Settings → Secrets.",
            icon="⚠️",
        )
    st.caption(
        f"**{len(tomorrow_picks)} picks** · **{n_high} HIGH confidence** · "
        f"🧠 {backend} · "
        f"Sources: NASDAQ earnings · SEC 8-K filings · small-cap momentum · today's movers"
    )

    if buys:
        st.markdown(
            f"### 🟢 Buy at Open "
            f"<span style='color:#555;font-size:.8rem;font-weight:400'>({len(buys)} high-conviction)</span>",
            unsafe_allow_html=True,
        )
        cols = st.columns(min(len(buys), 3))
        for i, p in enumerate(buys):
            with cols[i % 3]:
                t        = p.get("ticker", "?")
                conf     = p.get("confidence", "MEDIUM")
                catalyst = p.get("catalyst", "")
                why      = p.get("why", "")
                entry    = p.get("entry", "Buy at open with volume confirmation")
                stop     = p.get("stop", "Stop below open candle low")
                risk     = p.get("risk", "")
                cat_icon = _cat_emoji.get(catalyst, "📊")
                cc       = _conf_color.get(conf, "#888")
                cd       = _conf_dot.get(conf, "●○○")
                st.markdown(
                    f"<div class='tmr-card tmr-buy'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>"
                    f"  <b style='font-size:1.3rem;color:#fff'>${t}</b>"
                    f"  <div style='text-align:right'>"
                    f"    <div style='color:{cc};font-size:.8rem;font-weight:700'>{cd} {conf}</div>"
                    f"    <div style='color:#666;font-size:.68rem'>{cat_icon} {catalyst}</div>"
                    f"  </div>"
                    f"</div>"
                    f"<div style='font-size:.82rem;color:#e8e8e8;margin-bottom:8px;line-height:1.5'>{why}</div>"
                    f"<div style='background:#0a2e18;border-radius:5px;padding:7px 10px;margin-bottom:6px'>"
                    f"  <div style='color:#69f0ae;font-size:.72rem;font-weight:700;margin-bottom:2px'>📍 ENTRY</div>"
                    f"  <div style='color:#b9f6ca;font-size:.76rem'>{entry}</div>"
                    f"</div>"
                    f"<div style='display:flex;gap:8px'>"
                    f"  <div style='background:#1a0a00;border-radius:4px;padding:4px 7px;flex:1'>"
                    f"    <div style='color:#ff7043;font-size:.68rem;font-weight:700'>🛑 STOP</div>"
                    f"    <div style='color:#ffab91;font-size:.72rem'>{stop}</div>"
                    f"  </div>"
                    f"  <div style='background:#111;border-radius:4px;padding:4px 7px;flex:2'>"
                    f"    <div style='color:#666;font-size:.68rem;font-weight:700'>⚠ RISK</div>"
                    f"    <div style='color:#888;font-size:.7rem'>{risk[:80]}</div>"
                    f"  </div>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    if watches:
        st.markdown(
            f"### 🟡 Watch at Open "
            f"<span style='color:#555;font-size:.8rem;font-weight:400'>({len(watches)} — wait for confirmation)</span>",
            unsafe_allow_html=True,
        )
        n_cols = min(len(watches), 4)
        cols   = st.columns(n_cols)
        for i, p in enumerate(watches):
            with cols[i % n_cols]:
                t      = p.get("ticker", "?")
                conf   = p.get("confidence", "MEDIUM")
                why    = p.get("why", "")
                entry  = p.get("entry", "Wait for 15-min candle confirmation")
                stop   = p.get("stop", "")
                cat    = p.get("catalyst", "")
                cc     = _conf_color.get(conf, "#888")
                cd     = _conf_dot.get(conf, "●○○")
                st.markdown(
                    f"<div class='tmr-card tmr-watch'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px'>"
                    f"  <b style='font-size:1.1rem;color:#fff'>${t}</b>"
                    f"  <span style='color:{cc};font-size:.75rem'>{cd} {conf}</span>"
                    f"</div>"
                    f"<div style='font-size:.72rem;color:#aaa;margin-bottom:5px'>{_cat_emoji.get(cat,'📊')} {cat}</div>"
                    f"<div style='font-size:.78rem;color:#ddd;margin-bottom:6px;line-height:1.4'>{why[:120]}</div>"
                    f"<div style='font-size:.73rem;color:#ffcc80'>📍 {entry}</div>"
                    f"{f'<div style=\"font-size:.7rem;color:#888;margin-top:3px\">🛑 {stop}</div>' if stop else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ── Backtest log ──────────────────────────────────────────────────────────────
st.divider()
with st.expander("📊 Backtest Performance Log", expanded=False):
    bdf     = load_alerts_df()
    summary = backtest_summary()
    if bdf.empty:
        st.caption("No alerts logged yet. Setups scoring ≥ 80 are auto-logged here.")
    else:
        bs1, bs2, bs3, bs4, bs5 = st.columns(5)
        with bs1: st.metric("Total Alerts", summary.get("total_alerts", 0))
        with bs2:
            hr1 = summary.get("hit_rate_t1hr")
            st.metric("T+1hr Hit Rate", f"{hr1}%" if hr1 is not None else "—")
        with bs3:
            avg1 = summary.get("avg_pct_t1hr")
            st.metric("Avg T+1hr Return", f"{avg1:+.1f}%" if avg1 is not None else "—")
        with bs4:
            hr24 = summary.get("hit_rate_t24hr")
            st.metric("T+24hr Hit Rate", f"{hr24}%" if hr24 is not None else "—")
        with bs5:
            med24 = summary.get("median_pct_t24hr")
            st.metric("Median T+24hr", f"{med24:+.1f}%" if med24 is not None else "—")

        show_cols = [c for c in ["logged_at","ticker","price_at_alert","score",
            "catalyst_type","sentiment","dilution_risk","price_t1hr","pct_t1hr",
            "price_t24hr","pct_t24hr"] if c in bdf.columns]
        st.dataframe(bdf[show_cols].sort_values("logged_at", ascending=False),
                     use_container_width=True, hide_index=True)


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(1)
    st.rerun()
