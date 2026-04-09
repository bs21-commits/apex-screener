"""
SEC EDGAR real-time filing monitor.

Polls the EDGAR RSS full-text search feed for new 8-K, 6-K, S-1, S-3,
and F-3 filings.  Returns structured filing dicts ready for the AI parser.

No API key required — EDGAR is a public SEC service.
SEC rate-limit: ≤ 10 req/s; we stay well under with our poll interval.
"""

import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Generator

import requests

from screener.config import EDGAR_FILING_TYPES, EDGAR_POLL_INTERVAL_SEC

logger = logging.getLogger(__name__)

_HEADERS = {
    # SEC requires a descriptive User-Agent with contact info
    "User-Agent": "APEX-Screener/2.0 research@example.com",
    "Accept-Encoding": "gzip, deflate",
}
_NS = {"atom": "http://www.w3.org/2005/Atom"}

# EDGAR ATOM RSS — one URL per form type, returns 40 most recent filings
_RSS_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type={form}&dateb=&owner=include&count=40"
    "&search_text=&output=atom"
)

# Track seen filings across polls (dedup by URL hash)
_seen: set[str] = set()


# ── Text extraction helpers ───────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace for cleaner LLM input."""
    # Remove <script> and <style> blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove XBRL/iXBRL inline data blocks (very noisy)
    html = re.sub(r"<ix:[^>]+>.*?</ix:[^>]+>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    # Collapse runs of whitespace / blank lines
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def _find_primary_doc_url(index_html: str, base_url: str) -> str | None:
    """
    Extract the primary document URL from an EDGAR filing index page.
    Returns an absolute URL or None.
    """
    patterns = [
        # Explicit "Complete submission text file" link
        r'href="(/Archives/[^"]+\.txt)"[^>]*>[^<]*Complete',
        # Primary document labeled htm/html
        r'href="(/Archives/[^"]+\.(htm|html))"[^>]*>[^<]*(?:Primary|Document|Complete)',
        # First .htm link in the table (fallback)
        r'href="(/Archives/edgar/data/[^"]+\.(htm|html))"',
    ]
    for pat in patterns:
        m = re.search(pat, index_html, re.IGNORECASE)
        if m:
            return f"https://www.sec.gov{m.group(1)}"
    return None


def fetch_filing_text(filing_url: str, max_chars: int = 8_000) -> str:
    """
    Download the filing index page, resolve the primary document, and
    extract plain text.  Truncates to max_chars to fit Claude's context.

    Handles:
      - EDGAR filing index pages (most common from RSS)
      - Direct .htm document links
      - XBRL / inline XBRL stripping
    """
    if not filing_url:
        return ""
    try:
        resp = requests.get(filing_url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        html = resp.text

        # If this looks like an index page (contains filing table), find the doc
        if "Filing Detail" in html or "Documents" in html or "<table" in html.lower():
            doc_url = _find_primary_doc_url(html, filing_url)
            if doc_url and doc_url != filing_url:
                try:
                    doc_resp = requests.get(doc_url, headers=_HEADERS, timeout=20)
                    doc_resp.raise_for_status()
                    html = doc_resp.text
                except Exception as inner:
                    logger.debug(f"[edgar] primary doc fetch failed ({inner}), using index text")

        text = _strip_html(html)
        return text[:max_chars]

    except Exception as exc:
        logger.warning(f"[edgar] text fetch failed for {filing_url}: {exc}")
        return ""


# ── RSS parsing ───────────────────────────────────────────────────────────────

def _entry_to_dict(entry: ET.Element, form_type: str) -> dict:
    """Convert an ATOM <entry> element into a standard filing dict."""
    def _text(tag: str) -> str:
        el = entry.find(f"atom:{tag}", _NS)
        return (el.text or "").strip() if el is not None else ""

    title     = _text("title")
    link_el   = entry.find("atom:link", _NS)
    link      = link_el.attrib.get("href", "") if link_el is not None else ""
    published = _text("updated") or _text("published")
    summary   = _text("summary")

    # Extract ticker from title format: "TICKER (FORM_TYPE) filed by Company Name"
    ticker = ""
    # Try parenthesised ticker first (most common)
    m = re.search(r"\(([A-Z0-9\-]{1,6})\)", title)
    if m:
        ticker = m.group(1)

    uid = hashlib.md5(link.encode()).hexdigest()

    return {
        "uid":          uid,
        "ticker":       ticker,
        "company_name": title.split("(")[0].strip(),
        "form_type":    form_type,
        "filed_at":     published,
        "filing_url":   link,
        "summary":      summary,
        "full_text":    "",   # populated lazily by fetch_filing_text()
    }


# ── Public API ────────────────────────────────────────────────────────────────

def poll_once(form_types: list[str] | None = None) -> list[dict]:
    """
    Fetch the latest 40 filings per form type from EDGAR's ATOM RSS.
    Returns only filings not seen in previous calls (dedup by URL hash).
    Thread-safe as long as only one thread calls poll_once().
    """
    form_types = form_types or EDGAR_FILING_TYPES
    new_filings: list[dict] = []

    for form in form_types:
        url = _RSS_URL.format(form=form.replace("-", ""))
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as exc:
            logger.warning(f"[edgar] RSS fetch failed for {form}: {exc}")
            continue

        for entry in root.findall("atom:entry", _NS):
            filing = _entry_to_dict(entry, form)
            if filing["uid"] not in _seen:
                _seen.add(filing["uid"])
                new_filings.append(filing)
                logger.info(
                    f"[edgar] NEW {form} — {filing['company_name']} "
                    f"({filing['ticker']}) @ {filing['filed_at']}"
                )

    return new_filings


def stream_filings(
    form_types: list[str] | None = None,
    poll_interval: int = EDGAR_POLL_INTERVAL_SEC,
) -> Generator[dict, None, None]:
    """
    Infinite generator that yields new filing dicts as they appear on EDGAR.
    Designed to run in a background thread.

    Usage:
        for filing in stream_filings():
            process(filing)
    """
    watched = form_types or EDGAR_FILING_TYPES
    logger.info(f"[edgar] streaming — forms={watched}, interval={poll_interval}s")
    while True:
        try:
            for filing in poll_once(watched):
                yield filing
        except Exception as exc:
            logger.error(f"[edgar] stream error: {exc}")
        time.sleep(poll_interval)


# ── Mock data for pipeline testing ────────────────────────────────────────────

def get_mock_filings(n: int = 3) -> list[dict]:
    """
    Return n realistic dummy filings that exercise all three classifier paths:
      1. Clean bullish private placement (should score high)
      2. Toxic ATM / shelf S-3 (should be flagged bearish)
      3. Major contract catalyst (should score high)
    """
    samples = [
        {
            "uid":          "mock_8k_001",
            "ticker":       "SYTA",
            "company_name": "Syra Health Corp",
            "form_type":    "8-K",
            "filed_at":     datetime.now(timezone.utc).isoformat(),
            "filing_url":   "https://www.sec.gov/Archives/edgar/data/0001/000001.htm",
            "summary":      "Entry into a Material Definitive Agreement",
            "full_text": (
                "Item 1.01. Entry into a Material Definitive Agreement.\n\n"
                "On April 8, 2026, Syra Health Corp (NASDAQ: SYTA) entered into a securities "
                "purchase agreement with certain institutional investors in connection with a "
                "PRIVATE PLACEMENT of 2,000,000 shares of common stock at $3.50 per share, "
                "representing a 12% PREMIUM to the last closing price of $3.12. "
                "The offering is expected to raise gross proceeds of approximately $7 million. "
                "No warrants are being issued in connection with this offering. The shares will "
                "be registered for resale on a registration statement to be filed within 30 days. "
                "Proceeds will fund expansion of the company's healthcare staffing platform and "
                "working capital. The transaction is expected to close on or about April 10, 2026."
            ),
        },
        {
            "uid":          "mock_s3_002",
            "ticker":       "MULN",
            "company_name": "Mullen Automotive Inc",
            "form_type":    "S-3",
            "filed_at":     datetime.now(timezone.utc).isoformat(),
            "filing_url":   "https://www.sec.gov/Archives/edgar/data/0002/000002.htm",
            "summary":      "Registration Statement — Shelf Registration",
            "full_text": (
                "MULLEN AUTOMOTIVE INC — FORM S-3 REGISTRATION STATEMENT\n\n"
                "Mullen Automotive Inc hereby registers for resale an aggregate of 50,000,000 "
                "shares of common stock pursuant to this SHELF REGISTRATION. "
                "The company has established an AT-THE-MARKET (ATM) OFFERING facility with "
                "H.C. Wainwright & Co. allowing the sale of up to $25,000,000 of common stock "
                "from time to time at prevailing market prices. "
                "Additionally, this registration statement covers 15,000,000 shares of common "
                "stock issuable upon EXERCISE OF OUTSTANDING WARRANTS issued in connection with "
                "prior private placements at exercise prices ranging from $0.10 to $0.50 per share. "
                "The company has variable-rate convertible notes outstanding with a conversion "
                "price equal to 85% of the volume weighted average price (VWAP) of the common "
                "stock. The company has received a going concern opinion from its independent "
                "registered public accounting firm."
            ),
        },
        {
            "uid":          "mock_8k_003",
            "ticker":       "FFIE",
            "company_name": "Faraday Future Intelligent Electric Inc",
            "form_type":    "8-K",
            "filed_at":     datetime.now(timezone.utc).isoformat(),
            "filing_url":   "https://www.sec.gov/Archives/edgar/data/0003/000003.htm",
            "summary":      "Entry into Material Definitive Agreement — Government Contract",
            "full_text": (
                "Item 1.01. Entry into a Material Definitive Agreement.\n\n"
                "Faraday Future Intelligent Electric Inc (Nasdaq: FFIE) today announced it has "
                "received a firm fixed-price CONTRACT AWARD from the United States Department of "
                "Energy for the delivery of 500 electric vehicles valued at approximately "
                "$45 million over 24 months. This award was made under a competitive procurement "
                "process. The contract contains no termination-for-convenience provisions that "
                "would allow unilateral cancellation without penalty. "
                "The company confirmed that this transaction does not involve the issuance of "
                "any new shares of common stock, warrants, convertible notes, or other dilutive "
                "securities. The company's current share count remains unchanged at 12,400,000 "
                "shares outstanding."
            ),
        },
    ]
    return samples[:n]
