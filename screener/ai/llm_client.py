"""
LLM Client — free-tier wrapper
================================
Priority order:
  1. Groq  (free 14,400 req/day — get key at console.groq.com, set GROQ_API_KEY)
  2. Ollama (fully local, zero accounts — install at ollama.com, run: ollama pull llama3)
  3. Rule-based fallback (no AI at all — always works)

Usage:
    from screener.ai.llm_client import chat

    result = chat("Your prompt here", expect_json=True)
    # returns parsed dict/list if expect_json=True, else string
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger("apex.llm")

# ── Groq ─────────────────────────────────────────────────────────────────────
_GROQ_KEY   = os.environ.get("GROQ_API_KEY", "")
_GROQ_MODEL = "llama-3.3-70b-versatile"   # free, fast, very capable
_groq_client = None

if _GROQ_KEY:
    try:
        from groq import Groq
        _groq_client = Groq(api_key=_GROQ_KEY)
        logger.info(f"[llm] Groq ready ({_GROQ_MODEL})")
    except Exception as e:
        logger.warning(f"[llm] Groq init failed: {e}")

# ── Ollama (local) ────────────────────────────────────────────────────────────
_OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
_ollama_ok    = False

try:
    import urllib.request
    urllib.request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=1)
    _ollama_ok = True
    logger.info(f"[llm] Ollama ready at {_OLLAMA_URL} (model: {_OLLAMA_MODEL})")
except Exception:
    pass


def _call_groq(prompt: str, max_tokens: int = 1400) -> str:
    msg = _groq_client.chat.completions.create(
        model=_GROQ_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return msg.choices[0].message.content.strip()


def _call_ollama(prompt: str, max_tokens: int = 1400) -> str:
    import urllib.request, json as _json
    payload = json.dumps({
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.2},
    }).encode()
    req = urllib.request.Request(
        f"{_OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return _json.loads(resp.read())["response"].strip()


def chat(prompt: str, max_tokens: int = 1400, expect_json: bool = False) -> str | dict | list:
    """
    Send a prompt to the best available free LLM.
    Returns raw string, or parsed JSON if expect_json=True.
    Raises RuntimeError if no LLM backend is available.
    """
    raw = None

    if _groq_client:
        try:
            raw = _call_groq(prompt, max_tokens)
            logger.debug("[llm] Groq response received")
        except Exception as exc:
            logger.warning(f"[llm] Groq failed: {exc}")

    if raw is None and _ollama_ok:
        try:
            raw = _call_ollama(prompt, max_tokens)
            logger.debug("[llm] Ollama response received")
        except Exception as exc:
            logger.warning(f"[llm] Ollama failed: {exc}")

    if raw is None:
        raise RuntimeError("No LLM backend available (set GROQ_API_KEY or run Ollama locally)")

    if not expect_json:
        return raw

    # Strip markdown fences and parse JSON
    clean = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    # Find the first [ or { to handle preamble text
    start = min(
        (clean.find("[") if "[" in clean else len(clean)),
        (clean.find("{") if "{" in clean else len(clean)),
    )
    clean = clean[start:]
    return json.loads(clean)


def is_available() -> bool:
    return bool(_groq_client or _ollama_ok)


def backend_name() -> str:
    if _groq_client:  return f"Groq ({_GROQ_MODEL})"
    if _ollama_ok:    return f"Ollama ({_OLLAMA_MODEL})"
    return "none"
