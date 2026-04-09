"""
Claude-powered SEC filing classifier.

Uses claude-haiku-4-5 (fast + cheap) with structured JSON output to
classify every new SEC filing as bullish, bearish, or neutral — and
specifically to flag toxic dilution traps before they trap traders.

Output schema (guaranteed valid JSON via output_config.format):
  catalyst_type  : filing category label (one of ~12 enum values)
  sentiment      : BULLISH | BEARISH | NEUTRAL
  toxicity_flags : list of detected red-flag patterns
  bullish_signals: list of detected positive patterns
  catalyst_score : 0-40 sub-score fed into the composite 0-100 screener score
  dilution_risk  : NONE | LOW | MEDIUM | HIGH | SEVERE
  summary        : 2-3 sentence plain-English filing summary
  raw_response   : full API response dict (for debugging)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

from screener.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MODEL_DEEP

logger = logging.getLogger(__name__)


# ── JSON output schema (enforced by structured outputs) ───────────────────────
_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "catalyst_type": {
            "type": "string",
            "enum": [
                "PRIVATE_PLACEMENT",
                "ATM_OFFERING",
                "SHELF_REGISTRATION",
                "WARRANT_EXERCISE",
                "CONVERTIBLE_NOTE",
                "MAJOR_CONTRACT",
                "REGULATORY_APPROVAL",
                "REVERSE_SPLIT",
                "MERGER_ACQUISITION",
                "SHARE_BUYBACK",
                "GOING_CONCERN",
                "OTHER",
            ],
        },
        "sentiment": {
            "type": "string",
            "enum": ["BULLISH", "BEARISH", "NEUTRAL"],
        },
        "toxicity_flags": {
            "type": "array",
            "items": {"type": "string"},
        },
        "bullish_signals": {
            "type": "array",
            "items": {"type": "string"},
        },
        "catalyst_score": {
            "type": "integer",   # 0-40; no min/max — structured outputs don't support them
        },
        "dilution_risk": {
            "type": "string",
            "enum": ["NONE", "LOW", "MEDIUM", "HIGH", "SEVERE"],
        },
        "summary": {
            "type": "string",
        },
    },
    "required": [
        "catalyst_type",
        "sentiment",
        "toxicity_flags",
        "bullish_signals",
        "catalyst_score",
        "dilution_risk",
        "summary",
    ],
    "additionalProperties": False,
}


# ── System prompt (stable — eligible for prompt caching) ─────────────────────
_SYSTEM_PROMPT = """You are an expert quantitative analyst specializing in event-driven \
momentum trading in low-float microcap stocks and ADRs.

Your task is to parse SEC filings and classify them for short-term trading setups. \
You are specifically hunting for two things:
  1. Clean catalysts that can ignite a momentum move.
  2. Toxic dilution traps that destroy shareholder value and should be AVOIDED.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BULLISH SIGNALS — look for these patterns:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PRIVATE_PLACEMENT_PREMIUM   — Shares sold at or above the current market price; no warrants.
  PRIVATE_PLACEMENT_AT_MARKET — Shares sold at market with clean terms (no ratchets, no resets).
  MAJOR_CONTRACT              — A significant new government or enterprise contract award.
  STRATEGIC_PARTNERSHIP       — Formal alliance with a large, established company.
  REGULATORY_APPROVAL         — FDA, FCC, DEA, or comparable regulatory clearance/approval.
  REVERSE_SPLIT_CATALYST      — Reverse split that reduces float, often precedes exchange uplisting.
  SHARE_BUYBACK               — Board-authorised buyback program.
  EARNINGS_BEAT               — Revenue or earnings materially above consensus estimates.
  UPLISTING                   — Exchange uplisting (OTC → Nasdaq/NYSE) or compliance restoration.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEARISH / TOXIC SIGNALS — flag these immediately:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ATM_OFFERING                — At-the-Market facility: allows the company to drip-sell shares
                                continuously into the open market at prevailing prices.
                                This is the #1 killer of microcap momentum runs.
  SHELF_REGISTRATION          — S-3 / F-3 shelf: registers a large block of shares for future
                                resale. Creates persistent overhead supply.
  WARRANT_EXERCISE            — Existing warrant holders converting warrants into common shares.
                                Immediately increases share count.
  DISCOUNTED_PIPE             — Private placement priced BELOW current market price. The discount
                                plus potential warrant coverage = instant dilution.
  TOXIC_CONVERTIBLE           — Convertible notes with variable conversion price, ratchets,
                                or full-ratchet anti-dilution. Death-spiral risk.
  WARRANT_RESET               — Anti-dilution reset or price-adjustment provisions on prior warrants.
  GOING_CONCERN               — Auditor or management going-concern warning.
  DILUTION_DEATH_SPIRAL       — Pattern of repeated variable-rate convertibles + ATM + warrant resets.
  RESALE_REGISTRATION         — Registration of shares held by prior investors for resale (creates
                                supply overhang even though no new shares are issued).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATALYST SCORE GUIDE (0 – 40 points):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  35 – 40 : Clean bullish catalyst, zero dilution flags, above-market terms.
  25 – 34 : Moderately bullish; minor concerns or unconfirmed details.
  15 – 24 : Neutral or mixed — some positive signals, some uncertainty.
   5 – 14 : Bearish elements dominate; dilutive financing present.
   0 –  4 : Severe toxicity — ATM, toxic converts, death spiral. Hard pass.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DILUTION RISK LEVELS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NONE   — No new shares being issued or registered.
  LOW    — Shares issued at market/premium, small count relative to float.
  MEDIUM — Some dilution but manageable (e.g., small warrant tranche).
  HIGH   — Significant new shares, discounted PIPE, or shelf registration.
  SEVERE — ATM + warrants + convertibles, or death-spiral pattern detected.

Return ONLY valid JSON matching the provided schema. No markdown, no preamble."""


# ── Client initialisation ────────────────────────────────────────────────────
def _get_client() -> anthropic.Anthropic:
    key = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to .env or export it in your shell."
        )
    return anthropic.Anthropic(api_key=key)


# ── Fallback result ───────────────────────────────────────────────────────────
def _fallback(ticker: str, reason: str) -> dict[str, Any]:
    return {
        "catalyst_type":  "OTHER",
        "sentiment":      "NEUTRAL",
        "toxicity_flags": [],
        "bullish_signals":[],
        "catalyst_score": 0,
        "dilution_risk":  "NONE",
        "summary":        f"[ERROR] Could not classify filing for {ticker}: {reason}",
        "raw_response":   None,
    }


# ── Core classification function ──────────────────────────────────────────────
def classify_filing(
    filing_text: str,
    ticker: str = "",
    form_type: str = "",
    filing_url: str = "",
    use_deep_model: bool = False,
) -> dict[str, Any]:
    """
    Send a filing excerpt to Claude and return a structured classification dict.

    Parameters
    ----------
    filing_text    : Extracted text from the SEC filing (truncated to ~8K chars).
    ticker         : Stock ticker symbol (context for the model).
    form_type      : Filing form type, e.g. "8-K", "S-3" (context for the model).
    filing_url     : SEC EDGAR URL (included in the prompt for traceability).
    use_deep_model : If True, uses claude-opus-4-6 for ambiguous/high-stakes filings.

    Returns
    -------
    dict matching the _OUTPUT_SCHEMA, plus a "raw_response" key for debugging.
    """
    if not filing_text or not filing_text.strip():
        return _fallback(ticker, "empty filing text")

    model = CLAUDE_MODEL_DEEP if use_deep_model else CLAUDE_MODEL

    user_content = (
        f"Analyze this SEC filing.\n\n"
        f"Ticker: {ticker or 'UNKNOWN'}\n"
        f"Form type: {form_type or 'UNKNOWN'}\n"
        f"Filing URL: {filing_url or 'N/A'}\n\n"
        f"--- FILING TEXT (first ~8,000 characters) ---\n\n"
        f"{filing_text.strip()}"
    )

    try:
        client = _get_client()

        # Use streaming for large inputs to avoid HTTP timeout.
        # output_config.format enforces valid JSON matching our schema.
        with client.messages.stream(
            model=model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},   # cache the large system prompt
                }
            ],
            output_config={
                "format": {
                    "type":   "json_schema",
                    "schema": _OUTPUT_SCHEMA,
                }
            },
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            final = stream.get_final_message()

        # Extract text block (structured output guarantees first block is text/JSON)
        raw_text = next(
            (b.text for b in final.content if b.type == "text"), ""
        )

        parsed: dict[str, Any] = json.loads(raw_text)

        # Clamp catalyst_score to [0, 40] since schema can't enforce it
        parsed["catalyst_score"] = max(0, min(40, int(parsed.get("catalyst_score", 0))))
        parsed["raw_response"] = {
            "model":        final.model,
            "stop_reason":  final.stop_reason,
            "input_tokens": final.usage.input_tokens,
            "output_tokens":final.usage.output_tokens,
            "cache_read":   getattr(final.usage, "cache_read_input_tokens", 0),
        }

        logger.info(
            f"[parser] {ticker} → {parsed['sentiment']} / {parsed['catalyst_type']} "
            f"score={parsed['catalyst_score']} dilution={parsed['dilution_risk']} "
            f"(in={final.usage.input_tokens} out={final.usage.output_tokens} "
            f"cache_read={parsed['raw_response']['cache_read']})"
        )

        # Auto-escalate to deep model when Haiku is uncertain and score is mid-range
        if (
            not use_deep_model
            and parsed["sentiment"] == "NEUTRAL"
            and 10 <= parsed["catalyst_score"] <= 25
        ):
            logger.info(f"[parser] {ticker} ambiguous — escalating to {CLAUDE_MODEL_DEEP}")
            return classify_filing(
                filing_text, ticker, form_type, filing_url, use_deep_model=True
            )

        return parsed

    except anthropic.AuthenticationError:
        logger.error("[parser] Anthropic API key invalid or missing")
        return _fallback(ticker, "authentication error")

    except anthropic.RateLimitError:
        logger.warning("[parser] Rate limited — returning neutral fallback")
        return _fallback(ticker, "rate limited")

    except json.JSONDecodeError as exc:
        logger.error(f"[parser] JSON parse error for {ticker}: {exc}")
        return _fallback(ticker, f"JSON parse error: {exc}")

    except Exception as exc:
        logger.error(f"[parser] Unexpected error for {ticker}: {exc}")
        return _fallback(ticker, str(exc))


# ── Batch helper (future use) ─────────────────────────────────────────────────
def classify_filings_batch(
    filings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Classify multiple filings sequentially.
    TODO Phase 3+: swap for Anthropic Batches API (50% cost reduction) when
    latency tolerance allows.
    """
    results = []
    for f in filings:
        result = classify_filing(
            filing_text=f.get("full_text", ""),
            ticker=f.get("ticker", ""),
            form_type=f.get("form_type", ""),
            filing_url=f.get("filing_url", ""),
        )
        result["filing_uid"] = f.get("uid", "")
        results.append(result)
    return results
