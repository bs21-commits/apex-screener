import requests
import pandas as pd
import re
import time
import logging
from datetime import datetime, timezone, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import os
import yfinance as yf
import json
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
analyzer = SentimentIntensityAnalyzer()

FALSE_POSITIVES = {
    "A","I","AM","AN","AS","AT","BE","BY","DO","GO","HE","IF","IN","IS","IT",
    "ME","MY","NO","OF","OK","ON","OR","PM","SO","TO","UP","US","WE","DD","OP",
    "PS","CEO","CFO","IPO","ETF","GDP","IMO","FYI","ATH","EOD","EPS","LOL",
    "SEC","FDA","AI","AR","ALL","ARE","FOR","THE","AND","BUT","NOT","YOU",
    "HIS","HER","THEY","THIS","THAT","WITH","HAVE","FROM","WILL","BEEN",
    "WERE","WHEN","YOUR","WHAT","THEIR","THERE","THAN","THEN","INTO","OVER",
    "ALSO","BACK","AFTER","WELL","JUST","ONLY","EVEN","MOST","MUCH","BOTH",
    "EACH","VERY","GOOD","REAL","LAST","LONG","NEXT","HIGH","CALL","PUTS",
    "CASH","BULL","BEAR","MOON","PUMP","DUMP","FOMO","YOLO","HOLD","SELL",
    "LOSS","GAIN","FUND","DEBT","RATE","BOND","COST","RISK","NEWS","TLDR",
    "GET","GOT","HAD","HAS","ITS","DID","CAN","MAY","NOW","BUY","LOW","TOP",
    "NEW","OLD","BIG","WAY","DAY","USE","MAN","OWN","TOO","ANY","FEW","NOR",
    "PER","TRY","SET","LET","PUT","SAY","SHE","HIM","WHO","HOW","WHY",
    "ETC","DIV","AVG","ATR","RSI","EMA","SMA","VWAP","TTM","YOY","QOQ"
}

SQUEEZE_KEYWORDS = [
    "short squeeze","gamma squeeze","squeeze play","low float","high short",
    "short interest","days to cover","ftd","failure to deliver",
    "float rotation","breakout","break out","breaking out","broke out",
    "halt","halted","circuit breaker","level 2","l2",
    "premarket","pre-market","pre market","after hours","gap up","gapping",
    "runner","running","ripping","rip","ripper","flying","mooning",
    "100%","200%","300%","10x","5x","double","triple",
    "penny","small cap","micro cap","nano cap","otc","pink sheet",
    "catalyst","news","pr","press release","fda","sec filing","8k",
    "accumulating","loading","adding","conviction",
    "calls","options","unusual options","sweep","flow"
]

WARNING_KEYWORDS = [
    "scam","fraud","bagholders","trap","fake","manipulation","paid promotion",
    "stay away","avoid","dump","dumping","dumped","dilution","diluting",
    "reverse split","naked short","halt risk","delist","delisted"
]

PROMO_PATTERNS = [
    "use my link","referral","dm me","join my","telegram group",
    "discord invite","guaranteed","risk free","sign up now",
    "click here","follow me","check bio","check profile",
    "easy money","free money","cant lose"
]

message_global = {}
FINNHUB_API_BASE = "https://finnhub.io/api/v1"
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
UNIVERSE_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".universe_cache.json")
REDDIT_USER_AGENT = "apex-scanner/1.0"
NITTER_RSS_SOURCES = [
    "https://nitter.net",
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.unixfox.eu",
    "https://nitter.rawbit.ninja",
]
YF_UNIVERSE_SEEDS = [
    "SOUN", "MVIS", "BBAI", "RGTI", "IONQ", "RKLB", "PLUG", "FUBO", "NIO",
    "LCID", "RIOT", "MARA", "NKLA", "SIRI", "BB", "GPRO", "TLRY", "PENN",
    "OPEN", "RUN", "SPWR", "CHPT", "JOBY", "SPCE", "CLOV", "WULF", "BTBT",
    "HUT", "CLSK", "UAVS", "ASTS", "MULN", "MSTR", "AUR", "QS", "XPEV",
    "PINS", "SNAP", "PTON", "HOOD", "AFRM", "UPST", "BARK", "EVGO", "APPH",
    "WISH", "TTOO", "SENS", "KULR", "RIVN", "ABEV", "M", "AAL", "CCL", "NCLH",
    "DAL", "UAL", "SAVE", "T", "VZ", "PFE", "INTC", "AMD", "GME", "AMC"
]
try:
    yf.set_tz_cache_location(os.path.join(os.path.dirname(__file__), ".yf_tz_cache"))
except Exception:
    pass

def load_cached_universe():
    try:
        with open(UNIVERSE_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        tickers = data.get("tickers", [])
        if isinstance(tickers, list) and tickers:
            return [t for t in tickers if isinstance(t, str)]
    except Exception:
        pass
    return []

def save_cached_universe(tickers):
    try:
        with open(UNIVERSE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"updated_at": datetime.now(timezone.utc).isoformat(), "tickers": tickers},
                f,
            )
    except Exception as e:
        logger.warning(f"Could not write universe cache: {e}")

def is_human_post(message):
    body = message.get("body", "")
    user = message.get("user", {})
    if len(body.strip()) < 20:
        return False, "too_short"
    if not user.get("avatar_url"):
        return False, "no_avatar"
    followers = user.get("followers", 0)
    following = user.get("following", 0)
    if followers < 3 and following < 3:
        return False, "no_social_graph"
    body_lower = body.lower()
    for pattern in PROMO_PATTERNS:
        if pattern in body_lower:
            return False, "promotional"
    cashtags = re.findall(r"\$[A-Z]{1,5}", body)
    if len(cashtags) > 5:
        return False, "too_many_tickers"
    words = body.split()
    if len(words) > 3:
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)
        if caps_ratio > 0.7:
            return False, "all_caps_bot"
    join_date = user.get("join_date", "")
    if join_date:
        try:
            joined = datetime.strptime(join_date[:10], "%Y-%m-%d")
            age_days = (datetime.now() - joined).days
            if age_days < 14:
                return False, "account_too_new"
        except Exception:
            pass
    return True, None

def get_squeeze_score(body):
    global message_global
    body_lower = body.lower()
    score = 0
    squeeze_hits = sum(1 for kw in SQUEEZE_KEYWORDS if kw in body_lower)
    score += min(40, squeeze_hits * 8)
    warning_hits = sum(1 for kw in WARNING_KEYWORDS if kw in body_lower)
    score -= warning_hits * 10
    pct_mentions = re.findall(r"\d+%", body)
    high_pct = [p for p in pct_mentions if int(p.replace("%","")) >= 50]
    score += min(20, len(high_pct) * 10)
    urgency = [
        "today","now","right now","this morning","premarket",
        "opening","at open","watching","loading","just bought",
        "getting in","adding here","strong entry","setting up"
    ]
    urgency_hits = sum(1 for u in urgency if u in body_lower)
    score += min(20, urgency_hits * 7)
    st_sent = message_global.get("entities", {}).get("sentiment", {})
    if st_sent and st_sent.get("basic") == "Bullish":
        score += 20
    return max(0, min(100, score))

def score_sentiment(text):
    bull_boosts = [
        "squeeze","moon","rocket","breakout","explosive","catalyst",
        "loading","accumulating","strong","conviction","runner","ripping"
    ]
    bear_boosts = [
        "dump","scam","avoid","fraud","short","puts","trap","dilut"
    ]
    tl = text.lower()
    boost = sum(0.06 for w in bull_boosts if w in tl)
    boost -= sum(0.06 for w in bear_boosts if w in tl)
    raw = analyzer.polarity_scores(text)["compound"]
    return round(max(-1.0, min(1.0, raw + boost)), 3)

def init_ticker(ticker):
    return {
        "ticker": ticker,
        "mentions": 0.0,
        "sentiment_scores": [],
        "squeeze_scores": [],
        "bull_count": 0,
        "bear_count": 0,
        "mention_times": [],
        "messages": [],
        "last_seen": "",
    }

def map_finnhub_sentiment_to_label(sentiment_payload):
    if not sentiment_payload:
        return ""
    sentiment = sentiment_payload.get("sentiment", {})
    bullish = sentiment.get("bullishPercent", 0) or 0
    bearish = sentiment.get("bearishPercent", 0) or 0
    if bullish > bearish:
        return "Bullish"
    if bearish > bullish:
        return "Bearish"
    return "Neutral"

def fetch_finnhub(endpoint, params=None):
    if not FINNHUB_API_KEY:
        logger.error("Missing FINNHUB_API_KEY in environment")
        return None
    q = dict(params or {})
    q["token"] = FINNHUB_API_KEY
    try:
        r = requests.get(f"{FINNHUB_API_BASE}/{endpoint}", params=q, timeout=12)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            logger.warning("Finnhub rate limited — waiting 5 seconds")
            time.sleep(5)
        else:
            logger.warning(f"Finnhub returned {r.status_code} for {endpoint}")
    except Exception as e:
        logger.error(f"Finnhub fetch error ({endpoint}): {e}")
    return None

def get_active_smallcap_universe(max_tickers=20):
    ranked = []
    chunk_size = 20
    for i in range(0, len(YF_UNIVERSE_SEEDS), chunk_size):
        chunk = YF_UNIVERSE_SEEDS[i:i + chunk_size]
        try:
            history = yf.download(
                tickers=chunk,
                period="2mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as e:
            logger.warning(f"yfinance chunk download failed: {e}")
            continue

        for ticker in chunk:
            try:
                if ticker not in history:
                    continue
                frame = history[ticker].dropna()
                if frame.empty or len(frame) < 25:
                    continue
                latest = frame.iloc[-1]
                close = float(latest.get("Close", 0) or 0)
                volume = float(latest.get("Volume", 0) or 0)
                avg_vol_20 = float(frame["Volume"].tail(20).mean() or 0)
                rel_volume = (volume / avg_vol_20) if avg_vol_20 > 0 else 0.0
                dollar_volume = close * volume
                if close <= 0 or close >= 20:
                    continue
                if volume < 500_000 or rel_volume < 1.5:
                    continue
                ranked.append((ticker, rel_volume, dollar_volume))
            except Exception:
                continue
        time.sleep(0.25)

    if not ranked:
        cached = load_cached_universe()
        if cached:
            logger.warning("No high-RVOL small caps found; using cached universe")
            return cached[:max_tickers]
        logger.warning("No high-RVOL small caps found; falling back to seed tickers")
        return YF_UNIVERSE_SEEDS[:max_tickers]

    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    picks = [t[0] for t in ranked[:max_tickers]]
    logger.info(f"Selected universe ({len(picks)}): {picks}")
    save_cached_universe(picks)
    return picks

def build_finnhub_messages_for_ticker(ticker):
    now_utc = datetime.now(timezone.utc)
    to_date = datetime.now(timezone.utc).date()
    from_date = to_date - timedelta(days=2)
    sentiment_payload = fetch_finnhub("news-sentiment", {"symbol": ticker}) or {}
    news_payload = fetch_finnhub(
        "company-news",
        {
            "symbol": ticker,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
        },
    ) or []

    st_label = map_finnhub_sentiment_to_label(sentiment_payload)
    buzz = sentiment_payload.get("buzz", {})
    buzz_articles = int(buzz.get("articlesInLastWeek", 0) or 0)
    if not isinstance(news_payload, list):
        news_payload = []

    messages = []
    for article in news_payload[:20]:
        headline = article.get("headline", "").strip()
        summary = article.get("summary", "").strip()
        body = f"{headline}. {summary}".strip().strip(".")
        if not body:
            continue
        source = article.get("source", "finnhub_news")
        created_unix = int(article.get("datetime", 0) or 0)
        if created_unix:
            article_dt = datetime.fromtimestamp(created_unix, tz=timezone.utc)
        else:
            article_dt = now_utc

        # Keep age-based weighting useful while preventing total drop-off on stale wires.
        effective_dt = max(article_dt, now_utc - timedelta(hours=2))
        created_at = effective_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Keep bot-filter logic intact by mapping news metadata into the same schema.
        message = {
            "body": body,
            "created_at": created_at,
            "symbols": [{"symbol": ticker}],
            "entities": {"sentiment": {"basic": st_label}},
            "user": {
                "username": source.lower().replace(" ", "_"),
                "avatar_url": "https://finnhub.io/favicon.ico",
                "followers": max(120, buzz_articles),
                "following": 10,
                "join_date": "2010-01-01",
            },
            "url": article.get("url", ""),
        }
        messages.append(message)

    # Ensure every ticker can still contribute when news feed is empty/rate-limited.
    if not messages:
        sentiment = sentiment_payload.get("sentiment", {})
        bullish = sentiment.get("bullishPercent", 0) or 0
        bearish = sentiment.get("bearishPercent", 0) or 0
        article_count = buzz.get("articlesInLastWeek", 0) or 0
        label = st_label or "Neutral"
        body_1 = (
            f"Finnhub social sentiment for ${ticker}: bullish {bullish}% and bearish "
            f"{bearish}% with {article_count} related articles this week."
        )
        body_2 = (
            f"${ticker} sentiment monitor update: crowd positioning is {label.lower()} "
            f"with {article_count} article mentions tracked this week."
        )
        for body in (body_1, body_2):
            messages.append({
            "body": body,
            "created_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbols": [{"symbol": ticker}],
            "entities": {"sentiment": {"basic": label}},
            "user": {
                "username": "finnhub_sentiment",
                "avatar_url": "https://finnhub.io/favicon.ico",
                "followers": 120,
                "following": 10,
                "join_date": "2010-01-01",
            },
            "url": f"https://finnhub.io/dashboard",
        })
    return messages

def build_finnhub_general_messages_for_ticker(ticker):
    now_utc = datetime.now(timezone.utc)
    categories = ["general", "forex", "crypto", "merger"]
    out = []
    for category in categories:
        payload = fetch_finnhub("news", {"category": category, "minId": 0}) or []
        if not isinstance(payload, list):
            continue
        for article in payload[:80]:
            headline = (article.get("headline") or "").strip()
            summary = (article.get("summary") or "").strip()
            text_blob = f"{headline} {summary}".upper()
            if f"${ticker}" not in text_blob and re.search(rf"\b{re.escape(ticker)}\b", text_blob) is None:
                continue
            created_unix = int(article.get("datetime", 0) or 0)
            if created_unix:
                created = datetime.fromtimestamp(created_unix, tz=timezone.utc)
            else:
                created = now_utc
            created = max(created, now_utc - timedelta(hours=6))
            body = f"{headline}. {summary}".strip().strip(".")
            if len(body) < 20:
                continue
            out.append({
                "body": body,
                "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "symbols": [{"symbol": ticker}],
                "entities": {"sentiment": {"basic": ""}},
                "user": {
                    "username": (article.get("source") or "finnhub").lower().replace(" ", "_"),
                    "avatar_url": "https://finnhub.io/favicon.ico",
                    "followers": 80,
                    "following": 12,
                    "join_date": "2010-01-01",
                },
                "url": article.get("url", ""),
                "source": f"finnhub_{category}",
            })
    return out[:12]

def build_yahoo_news_messages_for_ticker(ticker):
    now_utc = datetime.now(timezone.utc)
    messages = []
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []
    for item in items[:12]:
        title = (item.get("title") or "").strip()
        summary = (item.get("summary") or "").strip()
        body = f"{title}. {summary}".strip().strip(".")
        if len(body) < 20:
            continue
        provider = (
            item.get("publisher")
            or item.get("providerPublishTime")
            or "yahoo_news"
        )
        ts = int(item.get("providerPublishTime", 0) or 0)
        created = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else now_utc
        created = max(created, now_utc - timedelta(hours=8))
        url = item.get("link") or item.get("canonicalUrl", {}).get("url") or "https://finance.yahoo.com"
        messages.append({
            "body": body,
            "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbols": [{"symbol": ticker}],
            "entities": {"sentiment": {"basic": ""}},
            "user": {
                "username": str(provider).lower().replace(" ", "_"),
                "avatar_url": "https://s.yimg.com/rz/l/favicon.ico",
                "followers": 70,
                "following": 10,
                "join_date": "2010-01-01",
            },
            "url": url,
            "source": "yahoo_news",
        })
    return messages

def build_reddit_messages_for_ticker(ticker):
    now_utc = datetime.now(timezone.utc)
    query = f"%24{ticker}%20OR%20{ticker}"
    url = f"https://www.reddit.com/search.json?q={query}&sort=new&t=day&limit=15"
    headers = {"User-Agent": REDDIT_USER_AGENT}
    messages = []
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        posts = r.json().get("data", {}).get("children", [])
    except Exception:
        return []
    for p in posts:
        d = p.get("data", {})
        title = (d.get("title") or "").strip()
        selftext = (d.get("selftext") or "").strip()
        body = f"{title}. {selftext}".strip().strip(".")
        if len(body) < 20:
            continue
        ts = int(d.get("created_utc", 0) or 0)
        created = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else now_utc
        created = max(created, now_utc - timedelta(hours=3))
        author = d.get("author") or "reddit_user"
        ups = int(d.get("ups", 0) or 0)
        source_sub = d.get("subreddit") or "reddit"
        messages.append({
            "body": body,
            "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbols": [{"symbol": ticker}],
            "entities": {"sentiment": {"basic": ""}},
            "user": {
                "username": f"u_{author}",
                "avatar_url": "https://www.redditstatic.com/desktop2x/img/favicon/apple-icon-57x57.png",
                "followers": max(30, ups),
                "following": 15,
                "join_date": "2010-01-01",
            },
            "url": f"https://reddit.com{d.get('permalink', '')}",
            "source": f"reddit_{source_sub}",
        })
    return messages[:10]

def build_x_messages_for_ticker(ticker):
    now_utc = datetime.now(timezone.utc)
    query = f"%24{ticker}%20lang%3Aen%20-filter%3Areplies"
    for base in NITTER_RSS_SOURCES:
        rss_url = f"{base}/search/rss?f=tweets&q={query}"
        try:
            r = requests.get(rss_url, timeout=8)
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.text)
            channel = root.find("channel")
            if channel is None:
                continue
            items = channel.findall("item")
            if not items:
                continue
            messages = []
            for item in items[:12]:
                title = (item.findtext("title") or "").strip()
                desc = (item.findtext("description") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                author = title.split(" (")[0].replace("@", "").strip() if title else "x_user"
                author_l = author.lower()
                if "bot" in author_l or "alert" in author_l:
                    continue
                body = f"{title}. {desc}".strip().strip(".")
                if len(body) < 20:
                    continue
                try:
                    dt = parsedate_to_datetime(pub)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    created = dt.astimezone(timezone.utc)
                except Exception:
                    created = now_utc
                created = max(created, now_utc - timedelta(hours=3))
                messages.append({
                    "body": body,
                    "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "symbols": [{"symbol": ticker}],
                    "entities": {"sentiment": {"basic": ""}},
                    "user": {
                        "username": f"x_{author_l}",
                        "avatar_url": "https://abs.twimg.com/favicons/twitter.3.ico",
                        "followers": 60,
                        "following": 20,
                        "join_date": "2010-01-01",
                    },
                    "url": link or "https://x.com",
                    "source": "x_public_rss",
                })
            if messages:
                return messages
        except Exception:
            continue
    return []

def fetch_intraday_context(ticker):
    default = {
        "price": None,
        "change_pct": None,
        "rel_volume": None,
        "vwap_position_pct": None,
        "above_vwap": False,
        "premarket_gap_pct": None,
    }
    quote = fetch_finnhub("quote", {"symbol": ticker}) or {}
    if isinstance(quote, dict) and quote.get("c"):
        c = float(quote.get("c", 0) or 0)
        dp = float(quote.get("dp", 0) or 0)
        o = float(quote.get("o", 0) or 0)
        pc = float(quote.get("pc", 0) or 0)
        default["price"] = round(c, 4) if c > 0 else None
        default["change_pct"] = round(dp, 2)
        if o > 0 and c > 0:
            default["vwap_position_pct"] = round(((c - o) / o) * 100, 2)
            default["above_vwap"] = c >= o
        if pc > 0 and o > 0:
            default["premarket_gap_pct"] = round(((o - pc) / pc) * 100, 2)
    try:
        intraday = yf.download(
            tickers=ticker,
            period="1d",
            interval="1m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if intraday is None or intraday.empty:
            return default
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.droplevel(0)
        intraday = intraday.dropna()
        if intraday.empty:
            return default
        current = intraday.iloc[-1]
        first = intraday.iloc[0]
        price = float(current.get("Close", 0) or 0)
        open_price = float(first.get("Open", 0) or 0)
        if open_price <= 0:
            open_price = price or 1.0
        change_pct = ((price - open_price) / open_price) * 100
        typical = (intraday["High"] + intraday["Low"] + intraday["Close"]) / 3
        vol = intraday["Volume"].fillna(0)
        vol_sum = float(vol.sum() or 0)
        vwap = float(((typical * vol).sum() / vol_sum) if vol_sum > 0 else price)
        vwap_pos = ((price - vwap) / vwap * 100) if vwap > 0 else 0.0
        bars = max(1, len(intraday))
        avg_bar_vol = vol_sum / bars
        rel_volume = (float(current.get("Volume", 0) or 0) / avg_bar_vol) if avg_bar_vol > 0 else 0.0
        premarket_gap = 0.0
        try:
            daily = yf.download(
                tickers=ticker,
                period="5d",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            ).dropna()
            if not daily.empty and len(daily) >= 2:
                prev_close = float(daily.iloc[-2].get("Close", 0) or 0)
                today_open = float(daily.iloc[-1].get("Open", 0) or 0)
                if prev_close > 0:
                    premarket_gap = ((today_open - prev_close) / prev_close) * 100
        except Exception:
            pass
        return {
            "price": round(price, 4) if price > 0 else default.get("price"),
            "change_pct": default.get("change_pct") if default.get("change_pct") is not None else round(change_pct, 2),
            "rel_volume": round(rel_volume, 2) if rel_volume > 0 else default.get("rel_volume"),
            "vwap_position_pct": default.get("vwap_position_pct") if default.get("vwap_position_pct") is not None else round(vwap_pos, 2),
            "above_vwap": default.get("above_vwap") if default.get("vwap_position_pct") is not None else (bool(price > vwap) if vwap > 0 else False),
            "premarket_gap_pct": round(premarket_gap, 2) if premarket_gap else default.get("premarket_gap_pct"),
        }
    except Exception as e:
        logger.warning(f"Intraday context failed for {ticker}: {e}")
        return default

def build_score_summary(ticker, apex, rel_vol, vwap_pos, avg_sent, avg_squeeze, news_catalyst, breakout, ultra_breakout):
    parts = [
        f"${ticker} scores {apex}/100",
        f"with RVOL {rel_vol}x",
        f"and VWAP position {vwap_pos}%",
    ]
    if avg_sent >= 0.2:
        parts.append(f"sentiment is bullish ({avg_sent})")
    elif avg_sent <= -0.2:
        parts.append(f"sentiment is bearish ({avg_sent})")
    else:
        parts.append(f"sentiment is neutral ({avg_sent})")
    parts.append(f"squeeze intensity is {avg_squeeze}")
    if news_catalyst:
        parts.append("recent news catalyst is present")
    if ultra_breakout:
        parts.append("ULTRA breakout conditions are active")
    elif breakout:
        parts.append("breakout conditions are active")
    return ". ".join(parts) + "."

def project_price_scenarios(price, avg_sent, rel_vol, vwap_pos, avg_squeeze, bull_pct, breakout, ultra_breakout, news_catalyst):
    sentiment_edge = avg_sent * 8.0
    flow_edge = min(12.0, rel_vol * 2.0) - 2.0
    vwap_edge = max(-6.0, min(10.0, vwap_pos * 0.8))
    squeeze_edge = (avg_squeeze - 20.0) * 0.15
    bonus = 1.0 if news_catalyst else 0.0
    if breakout:
        bonus += 3.0
    if ultra_breakout:
        bonus += 6.0
    move_1d = max(-15.0, min(25.0, sentiment_edge + flow_edge + vwap_edge + squeeze_edge + bonus))
    move_1w = max(-30.0, min(60.0, move_1d * 2.2 + (bull_pct - 50.0) / 8.0))
    if price and price > 0:
        p1d = round(price * (1 + move_1d / 100.0), 4)
        p1w = round(price * (1 + move_1w / 100.0), 4)
    else:
        p1d = None
        p1w = None
    return round(move_1d, 2), round(move_1w, 2), p1d, p1w

def process_messages(messages, ticker_data, forced_ticker=None):
    global message_global
    now = datetime.now(timezone.utc).timestamp()
    for msg in messages:
        message_global = msg
        is_human, reason = is_human_post(msg)
        if not is_human:
            continue
        body = msg.get("body", "")
        created_at = msg.get("created_at", "")
        user = msg.get("user", {})
        username = user.get("username", "unknown")
        try:
            mt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
            mt = mt.replace(tzinfo=timezone.utc)
            age_hours = (now - mt.timestamp()) / 3600
        except Exception:
            age_hours = 1.0
        if age_hours > 12:
            continue
        if age_hours < 0.25:
            time_weight = 1.0
        elif age_hours < 1:
            time_weight = 0.85
        elif age_hours < 3:
            time_weight = 0.65
        elif age_hours < 6:
            time_weight = 0.40
        else:
            time_weight = 0.20
        followers = user.get("followers", 0)
        if followers >= 1000:
            user_weight = 1.5
        elif followers >= 100:
            user_weight = 1.2
        elif followers >= 20:
            user_weight = 1.0
        else:
            user_weight = 0.8
        if forced_ticker:
            tickers = [(forced_ticker, 1.0)]
        else:
            symbols = msg.get("symbols", [])
            tickers = [(s["symbol"], 1.0) for s in symbols
                       if s["symbol"] not in FALSE_POSITIVES
                       and len(s["symbol"]) <= 5]
            if not tickers:
                cashtags = re.findall(r"\$([A-Z]{1,5})\b", body)
                tickers = [(t, 0.8) for t in cashtags
                           if t not in FALSE_POSITIVES]
        if not tickers:
            continue
        sentiment = score_sentiment(body)
        squeeze_score = get_squeeze_score(body)
        final_weight = time_weight * user_weight
        st_sent = msg.get("entities", {}).get("sentiment", {})
        st_label = st_sent.get("basic", "") if st_sent else ""
        for ticker, ticker_weight in tickers:
            if ticker not in ticker_data:
                ticker_data[ticker] = init_ticker(ticker)
            td = ticker_data[ticker]
            td["mentions"] += ticker_weight * final_weight
            td["sentiment_scores"].append(sentiment)
            td["squeeze_scores"].append(squeeze_score)
            td["mention_times"].append(now)
            td["last_seen"] = datetime.now().strftime("%H:%M:%S")
            if st_label == "Bullish":
                td["bull_count"] += 1
            elif st_label == "Bearish":
                td["bear_count"] += 1
            if len(td["messages"]) < 8:
                td["messages"].append({
                    "body": body[:200],
                    "username": username,
                    "source": msg.get("source", "unknown"),
                    "followers": followers,
                    "age_hours": round(age_hours, 1),
                    "sentiment": st_label or (
                        "Bullish" if sentiment > 0.2 else
                        "Bearish" if sentiment < -0.2 else "Neutral"
                    ),
                    "squeeze_score": squeeze_score,
                    "url": msg.get("url") or f"https://finnhub.io"
                })

def calculate_signals(ticker_data, market_context=None):
    market_context = market_context or {}
    now = datetime.now(timezone.utc).timestamp()
    rows = []
    for ticker, data in ticker_data.items():
        if data["mentions"] < 1:
            continue
        scores = data["sentiment_scores"]
        avg_sent = round(sum(scores) / len(scores), 3) if scores else 0.0
        sq_scores = data["squeeze_scores"]
        avg_squeeze = round(sum(sq_scores) / len(sq_scores), 1) if sq_scores else 0.0
        recent = [t for t in data["mention_times"] if now - t < 1800]
        prior = [t for t in data["mention_times"] if 1800 <= now - t < 3600]
        last_15 = [t for t in data["mention_times"] if now - t < 900]
        vel_recent = len(recent)
        vel_prior = len(prior)
        vel_ratio = (vel_recent / vel_prior) if vel_prior > 0 else (2.0 if vel_recent > 0 else 0.0)
        total_labeled = data["bull_count"] + data["bear_count"]
        bull_pct = round(data["bull_count"] / total_labeled * 100, 1) if total_labeled > 0 else 50.0
        mention_score = min(25, data["mentions"] * 3)
        sentiment_score = max(0, avg_sent * 20)
        squeeze_component = avg_squeeze * 0.25
        velocity_score = min(20, vel_ratio * 8)
        bull_score = max(0, (bull_pct - 50) / 50 * 10)
        base_apex = min(100, round(
            mention_score + sentiment_score + squeeze_component +
            velocity_score + bull_score, 1))
        mkt = market_context.get(ticker, {})
        rel_vol = float(mkt.get("rel_volume", 0.0) or 0.0)
        vwap_pos = float(mkt.get("vwap_position_pct", 0.0) or 0.0)
        change_pct = float(mkt.get("change_pct", 0.0) or 0.0)
        above_vwap = bool(mkt.get("above_vwap", False))
        premarket_gap_pct = float(mkt.get("premarket_gap_pct", 0.0) or 0.0)
        news_catalyst = bool(data.get("messages"))
        rv_score = min(25.0, max(0.0, rel_vol / 5.0 * 25.0))
        vwap_momo_score = min(20.0, max(0.0, vwap_pos / 6.0 * 20.0))
        fin_sent_score = min(25.0, max(0.0, avg_sent * 25.0))
        news_bonus = 15.0 if news_catalyst else 0.0
        gap_bonus = 15.0 if premarket_gap_pct >= 10 else 0.0
        market_apex = rv_score + vwap_momo_score + fin_sent_score + news_bonus + gap_bonus
        apex = min(100, round(base_apex * 0.55 + market_apex * 0.45, 1))
        breakout = (
            apex >= 60 and
            rel_vol >= 3.0 and
            above_vwap and
            avg_sent > 0 and
            news_catalyst
        )
        ultra_breakout = (
            apex >= 80 and
            rel_vol >= 5.0 and
            vwap_pos >= 5.0 and
            avg_sent >= 0.25 and
            news_catalyst
        )
        why = []
        if rel_vol >= 3:
            why.append(f"RVOL {rel_vol}x")
        if above_vwap:
            why.append(f"Above VWAP by {round(vwap_pos,1)}%")
        if avg_sent > 0:
            why.append(f"Positive sentiment {avg_sent}")
        if news_catalyst:
            why.append("Recent news catalyst")
        if premarket_gap_pct >= 10:
            why.append(f"Gap up {premarket_gap_pct}%")
        source_count = len({
            (m.get("source") or m.get("username") or "").split("_")[0]
            for m in data.get("messages", [])
            if (m.get("source") or m.get("username"))
        })
        x_posts_count = sum(1 for m in data.get("messages", []) if str(m.get("source", "")).startswith("x_"))
        freshest_age_hours = min((m.get("age_hours", 999) for m in data.get("messages", []) if "age_hours" in m), default=999)
        freshest_post_mins = max(0, int(round(freshest_age_hours * 60))) if freshest_age_hours < 999 else None
        if rel_vol > 0 and source_count >= 3 and data["mentions"] >= 4:
            proj_conf = "high"
        elif source_count >= 2 and data["mentions"] >= 2:
            proj_conf = "medium"
        else:
            proj_conf = "low"
        summary_text = build_score_summary(
            ticker=ticker,
            apex=apex,
            rel_vol=round(rel_vol, 2),
            vwap_pos=round(vwap_pos, 2),
            avg_sent=avg_sent,
            avg_squeeze=avg_squeeze,
            news_catalyst=news_catalyst,
            breakout=breakout,
            ultra_breakout=ultra_breakout,
        )
        proj_1d_pct, proj_1w_pct, proj_1d_price, proj_1w_price = project_price_scenarios(
            price=float(mkt.get("price", 0.0) or 0.0),
            avg_sent=avg_sent,
            rel_vol=rel_vol,
            vwap_pos=vwap_pos,
            avg_squeeze=avg_squeeze,
            bull_pct=bull_pct,
            breakout=breakout,
            ultra_breakout=ultra_breakout,
            news_catalyst=news_catalyst,
        )
        sent_label = (
            "VERY BULLISH" if avg_sent >= 0.5 else
            "BULLISH" if avg_sent >= 0.2 else
            "NEUTRAL" if avg_sent >= -0.2 else
            "BEARISH" if avg_sent >= -0.5 else
            "VERY BEARISH"
        )
        rows.append({
            "ticker": ticker,
            "apex_score": apex,
            "mentions": round(data["mentions"], 1),
            "price": mkt.get("price"),
            "change_pct": mkt.get("change_pct"),
            "rel_volume": mkt.get("rel_volume"),
            "vwap_position_pct": mkt.get("vwap_position_pct"),
            "above_vwap": above_vwap,
            "news_catalyst": news_catalyst,
            "data_quality": (
                "full" if rel_vol > 0 and mkt.get("price", 0) > 0 else
                "partial" if news_catalyst else "fallback"
            ),
            "source_count": source_count,
            "x_posts_count": x_posts_count,
            "freshest_post_mins": freshest_post_mins,
            "projection_confidence": proj_conf,
            "premarket_gap_pct": mkt.get("premarket_gap_pct"),
            "avg_sentiment": avg_sent,
            "sentiment_label": sent_label,
            "avg_squeeze_score": avg_squeeze,
            "velocity_30m": vel_recent,
            "velocity_prior_30m": vel_prior,
            "velocity_ratio": round(vel_ratio, 2),
            "last_15min": len(last_15),
            "bull_pct": bull_pct,
            "bear_pct": round(100 - bull_pct, 1),
            "breakout_flag": breakout,
            "ultra_breakout": ultra_breakout,
            "why_flagged": why,
            "score_summary": summary_text,
            "projected_1d_pct": proj_1d_pct,
            "projected_1w_pct": proj_1w_pct,
            "projected_1d_price": proj_1d_price,
            "projected_1w_price": proj_1w_price,
            "last_seen": data["last_seen"],
            "messages": data["messages"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("apex_score", ascending=False).reset_index(drop=True)

def run_full_scan():
    ticker_data = {}
    logger.info("=== Starting APEX Finnhub Scan ===")
    if not FINNHUB_API_KEY:
        logger.error("Missing FINNHUB_API_KEY in environment")
        return pd.DataFrame()
    universe = get_active_smallcap_universe(max_tickers=20)
    logger.info(f"Scanning Finnhub news/sentiment for {len(universe)} symbols")

    for ticker in universe:
        msgs = []
        msgs.extend(build_finnhub_messages_for_ticker(ticker))
        msgs.extend(build_finnhub_general_messages_for_ticker(ticker))
        msgs.extend(build_yahoo_news_messages_for_ticker(ticker))
        msgs.extend(build_reddit_messages_for_ticker(ticker))
        msgs.extend(build_x_messages_for_ticker(ticker))
        process_messages(msgs, ticker_data, forced_ticker=ticker)
        time.sleep(0.3)

    market_context = {t: fetch_intraday_context(t) for t in ticker_data.keys()}
    result = calculate_signals(ticker_data, market_context=market_context)
    logger.info(f"=== Scan complete — {len(result)} tickers ===")
    return result

if __name__ == "__main__":
    print("\nRunning Finnhub scan...\n")
    df = run_full_scan()
    if df.empty:
        print("No signals found.")
    else:
        print(df[["ticker","apex_score","mentions","sentiment_label",
                   "avg_squeeze_score","velocity_ratio","bull_pct",
                   "breakout_flag","ultra_breakout"]].to_string(index=False))
                   