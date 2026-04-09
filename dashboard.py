import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime
from scanner import run_full_scan

st.set_page_config(
    page_title="APEX Sentiment Scanner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.disclaimer { background-color: #1a1a2e; color: #888; padding: 10px;
              border-radius: 6px; font-size: 0.75rem; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Controls")
    st.divider()
    min_score = st.slider("Min APEX Score", 0, 100, 10)
    show_breakout_only = st.toggle("Breakout flags only", False)
    show_ultra_only = st.toggle("Ultra breakout only", False)
    show_bullish_only = st.toggle("Bullish only", False)
    min_mentions = st.slider("Min mentions", 0, 20, 1)
    auto_refresh = st.toggle("Auto-refresh every 5 min", True)
    st.divider()
    st.caption("Data source: Finnhub + yfinance")
    st.caption("Human-only posts")
    st.caption("Bot filtered")
    st.caption("Works 24/7")
    st.divider()
    manual_scan = st.button("Scan Now", use_container_width=True)
    st.divider()
    st.markdown("""
    <div class='disclaimer'>
    For educational use only.
    Not financial advice. Never
    invest more than you can
    afford to lose.
    </div>
    """, unsafe_allow_html=True)

st.title("APEX Sentiment Scanner")
st.caption("Human-filtered social/news sentiment — Finnhub + yfinance — 24/7")

def fmt_num(val, suffix="", decimals=2, prefix=""):
    if val is None:
        return "N/A"
    try:
        return f"{prefix}{float(val):.{decimals}f}{suffix}"
    except Exception:
        return "N/A"

if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0
if "all_time_breakouts" not in st.session_state:
    st.session_state.all_time_breakouts = []

should_scan = False
if manual_scan:
    should_scan = True
elif st.session_state.last_scan_time is None:
    should_scan = True
elif auto_refresh:
    elapsed = (datetime.now() - st.session_state.last_scan_time).seconds
    if elapsed >= 300:
        should_scan = True

if should_scan:
    with st.spinner("Scanning Finnhub for human-filtered signals..."):
        df = run_full_scan()
        st.session_state.df = df
        st.session_state.last_scan_time = datetime.now()
        st.session_state.scan_count += 1
        if not df.empty:
            new_breakouts = df[df["breakout_flag"] == True]["ticker"].tolist()
            for t in new_breakouts:
                entry = f"{t} @ {datetime.now().strftime('%H:%M:%S')}"
                if entry not in st.session_state.all_time_breakouts:
                    st.session_state.all_time_breakouts.append(entry)

df = st.session_state.df

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    last = st.session_state.last_scan_time
    st.metric("Last Scan", last.strftime("%H:%M:%S") if last else "Never")
with col2:
    st.metric("Tickers Found", len(df) if not df.empty else 0)
with col3:
    ultra = len(df[df["ultra_breakout"] == True]) if not df.empty else 0
    st.metric("ULTRA Breakouts", ultra)
with col4:
    brk = len(df[df["breakout_flag"] == True]) if not df.empty else 0
    st.metric("Breakout Flags", brk)
with col5:
    status = datetime.now().strftime("%H:%M")
    st.metric("Scans Run", st.session_state.scan_count, delta=f"Now {status}")

st.divider()

if df.empty:
    st.info("Click Scan Now — works 24/7, no market hours needed.")
    st.stop()

filtered = df.copy()
filtered = filtered[filtered["apex_score"] >= min_score]
filtered = filtered[filtered["mentions"] >= min_mentions]
if show_ultra_only:
    filtered = filtered[filtered["ultra_breakout"] == True]
elif show_breakout_only:
    filtered = filtered[filtered["breakout_flag"] == True]
if show_bullish_only:
    filtered = filtered[filtered["avg_sentiment"] >= 0.15]

ultra_df = filtered[filtered["ultra_breakout"] == True]
if not ultra_df.empty:
    st.error("ULTRA BREAKOUT — Extreme human activity detected")
    for _, row in ultra_df.iterrows():
        c1,c2,c3,c4,c5,c6 = st.columns([1,2,2,2,2,2])
        with c1:
            st.markdown(f"## ${row['ticker']}")
        with c2:
            st.metric("APEX", f"{row['apex_score']}/100")
        with c3:
            st.metric("Squeeze Score", f"{row['avg_squeeze_score']}")
        with c4:
            st.metric("Velocity", f"{row['velocity_ratio']}x")
        with c5:
            st.metric("Bull %", f"{row['bull_pct']}%")
        with c6:
            st.metric("Last 15min", f"{row['last_15min']} posts")
    st.divider()

brk_df = filtered[
    (filtered["breakout_flag"] == True) &
    (filtered["ultra_breakout"] == False)
]
if not brk_df.empty:
    st.warning("BREAKOUT CANDIDATES — Potential intraday movers")
    for _, row in brk_df.iterrows():
        c1,c2,c3,c4,c5 = st.columns([1,2,2,2,2])
        with c1:
            st.markdown(f"### ${row['ticker']}")
        with c2:
            st.metric("APEX", f"{row['apex_score']}/100")
        with c3:
            st.metric("Velocity", f"{row['velocity_ratio']}x")
        with c4:
            st.metric("Bull %", f"{row['bull_pct']}%")
        with c5:
            st.metric("Squeeze Score", f"{row['avg_squeeze_score']}")
    st.divider()

st.subheader(f"All Signals — {len(filtered)} tickers")

if filtered.empty:
    st.warning("No tickers match filters. Lower the minimum score or mentions.")
else:
    display = filtered[[
        "ticker","price","change_pct","apex_score","mentions","rel_volume",
        "vwap_position_pct","sentiment_label","news_catalyst",
        "data_quality","source_count",
        "avg_squeeze_score","velocity_ratio","last_15min",
        "bull_pct","bear_pct","breakout_flag","ultra_breakout","last_seen"
    ]].copy()
    display.columns = [
        "Ticker","Price","Change %","APEX Score","Mentions","Rel Vol",
        "VWAP %","Sentiment","News Catalyst","Data Quality","Sources",
        "Squeeze Score","Velocity","Last 15min",
        "Bull %","Bear %","Breakout","ULTRA","Last Seen"
    ]

    def color_rows(row):
        if row["ULTRA"]:
            return ["background-color: #3d0000; color: #ff4444"] * len(row)
        elif row["Breakout"]:
            return ["background-color: #2d1500; color: #ff8800"] * len(row)
        elif row["APEX Score"] >= 60:
            return ["background-color: #001a00; color: #44ff44"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display.style.apply(color_rows, axis=1),
        use_container_width=True,
        height=400
    )

st.divider()
st.subheader("Ticker Detail")

if not filtered.empty:
    selected = st.selectbox(
        "Select ticker",
        options=filtered["ticker"].tolist(),
        index=0
    )
    row = filtered[filtered["ticker"] == selected].iloc[0]

    d1,d2,d3,d4,d5,d6 = st.columns(6)
    with d1:
        st.metric("APEX Score", f"{row['apex_score']}/100")
    with d2:
        st.metric("Price", fmt_num(row.get("price"), prefix="$"), delta=fmt_num(row.get("change_pct"), suffix="%"))
    with d3:
        st.metric("Rel Vol", fmt_num(row.get("rel_volume"), suffix="x"))
    with d4:
        st.metric("VWAP Position", fmt_num(row.get("vwap_position_pct"), suffix="%"))
    with d5:
        st.metric("Bull %", f"{row['bull_pct']}%")
    with d6:
        st.metric("Gap %", fmt_num(row.get("premarket_gap_pct"), suffix="%"))

    reasons = row.get("why_flagged", [])
    st.markdown("**Real-time score summary**")
    st.caption(row.get("score_summary", "Summary unavailable."))
    if reasons:
        st.markdown("**Why this ticker flagged**")
        for reason in reasons:
            st.caption(f"- {reason}")
    st.caption(f"Data quality: {row.get('data_quality', 'unknown')}")
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.metric("Projected 1D Move", f"{row.get('projected_1d_pct', 0)}%")
    with p2:
        st.metric("Projected 1D Price", fmt_num(row.get("projected_1d_price"), prefix="$"))
    with p3:
        st.metric("Projected 1W Move", f"{row.get('projected_1w_pct', 0)}%")
    with p4:
        st.metric("Projected 1W Price", fmt_num(row.get("projected_1w_price"), prefix="$"))
    with p5:
        st.metric("Forecast Confidence", str(row.get("projection_confidence", "low")).upper())
    freshest = row.get("freshest_post_mins", None)
    if freshest is not None:
        st.caption(f"Freshest source post: ~{freshest} minutes ago")
    st.caption(f"X posts captured this scan: {int(row.get('x_posts_count', 0) or 0)}")
    st.caption("Forecasts are model-derived scenarios from current momentum/sentiment inputs, not financial advice.")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=row["avg_sentiment"],
        domain={"x": [0,1], "y": [0,1]},
        title={"text": f"${selected} Sentiment"},
        gauge={
            "axis": {"range": [-1,1]},
            "bar": {"color": "green" if row["avg_sentiment"] > 0 else "red"},
            "steps": [
                {"range": [-1,-0.5], "color": "#3d0000"},
                {"range": [-0.5,0], "color": "#1a0000"},
                {"range": [0,0.5], "color": "#001a00"},
                {"range": [0.5,1], "color": "#003300"},
            ],
        }
    ))
    fig.update_layout(
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color":"white"}
    )
    st.plotly_chart(fig, use_container_width=True)

    messages = row.get("messages", [])
    if messages:
        st.subheader(f"Recent news/social messages about ${selected}")
        for msg in messages:
            icon = (
                "🟢" if msg["sentiment"] == "Bullish" else
                "🔴" if msg["sentiment"] == "Bearish" else "⚪"
            )
            with st.expander(
                f"{icon} @{msg['username']} "
                f"[{msg.get('source', 'source')}] "
                f"({msg['followers']} followers) — "
                f"{msg['age_hours']}h ago — "
                f"Squeeze: {msg['squeeze_score']}"
            ):
                st.write(msg["body"])
                st.markdown(f"[Open source link]({msg['url']})")

if st.session_state.all_time_breakouts:
    st.divider()
    st.subheader("Breakout History This Session")
    for entry in reversed(st.session_state.all_time_breakouts[-20:]):
        st.caption(f"🚨 {entry}")

if auto_refresh:
    time.sleep(1)
    st.rerun()
    