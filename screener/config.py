"""
Central configuration for the APEX Low-Float Screener.
All API keys and tuneable thresholds live here.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
POLYGON_API_KEY   = os.getenv("POLYGON_API_KEY", "")
FINVIZ_API_KEY    = os.getenv("FINVIZ_API_KEY", "")   # optional paid tier

# ── Mock mode ───────────────────────────────────────────────────────────────
# Set USE_MOCK=true in .env (or leave key blank) to use dummy data
USE_MOCK_POLYGON = os.getenv("USE_MOCK_POLYGON", "true").lower() == "true"
USE_MOCK_FINVIZ  = os.getenv("USE_MOCK_FINVIZ",  "true").lower() == "true"

# ── Screening Thresholds ────────────────────────────────────────────────────
MAX_FLOAT_SHARES        = 20_000_000   # only track stocks with float < 20M
VOLUME_SPIKE_THRESHOLD  = 5.0          # flag if current vol > 5x 30-day avg
HIGH_SCORE_THRESHOLD    = 80           # trigger backtest log at this score

# ── EDGAR Polling ───────────────────────────────────────────────────────────
EDGAR_POLL_INTERVAL_SEC = 60           # how often to check for new filings
EDGAR_FILING_TYPES      = ["8-K", "6-K", "S-1", "S-3", "F-3"]

# ── Claude Models ───────────────────────────────────────────────────────────
# Haiku: fast + cheap — used for high-volume filing classification
# Opus:  deep reasoning — reserved for ambiguous/complex filings
CLAUDE_MODEL_FAST  = "claude-haiku-4-5-20251001"   # primary classifier
CLAUDE_MODEL_DEEP  = "claude-opus-4-6"             # fallback for low-confidence
CLAUDE_MODEL       = CLAUDE_MODEL_FAST             # alias used by filing_parser

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
LOG_DIR   = os.path.join(BASE_DIR, "data", "logs")
ALERT_CSV = os.path.join(LOG_DIR, "alerts.csv")
