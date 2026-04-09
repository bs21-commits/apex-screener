"""
APEX Health Monitor — LangGraph Agent
======================================
Checks every data source and AI backend.
Reports what's broken, why, and how to fix it.
Runs via GitHub Actions on a schedule — no extra server needed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Annotated, TypedDict

import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

load_dotenv()

GROQ_KEY         = os.environ.get("GROQ_API_KEY", "")
APP_URL          = os.environ.get("APP_URL", "https://apex-screener.streamlit.app")
GITHUB_REPO      = os.environ.get("GITHUB_REPOSITORY", "bs21-commits/apex-screener")
GITHUB_TOKEN     = os.environ.get("GITHUB_TOKEN", "")

# ── State ─────────────────────────────────────────────────────────────────────
class MonitorState(TypedDict):
    messages:     Annotated[list, add_messages]
    check_results: dict   # {source: {ok, latency_ms, error}}
    issues:        list   # list of issue strings found
    fixed:         list   # list of auto-fixes applied


# ── Health check tools ────────────────────────────────────────────────────────
@tool
def check_yahoo_finance() -> dict:
    """Check if Yahoo Finance gainers API is returning stock data."""
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            "?scrIds=day_gainers&count=5&formatted=false",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        t = r.elapsed.total_seconds() * 1000
        if r.status_code != 200:
            return {"ok": False, "latency_ms": t, "error": f"HTTP {r.status_code}"}
        quotes = r.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
        if not quotes:
            return {"ok": False, "latency_ms": t, "error": "Empty quotes list"}
        return {"ok": True, "latency_ms": t, "count": len(quotes)}
    except Exception as e:
        return {"ok": False, "latency_ms": 0, "error": str(e)}


@tool
def check_sec_edgar() -> dict:
    """Check if SEC EDGAR 8-K filing search is accessible."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        r = requests.get(
            f"https://efts.sec.gov/LATEST/search-index?forms=8-K"
            f"&dateRange=custom&startdt={today}&enddt={today}",
            headers={"User-Agent": "apex-monitor health@apex.local"},
            timeout=10,
        )
        t = r.elapsed.total_seconds() * 1000
        if r.status_code != 200:
            return {"ok": False, "latency_ms": t, "error": f"HTTP {r.status_code}"}
        hits = r.json().get("hits", {}).get("hits", [])
        return {"ok": True, "latency_ms": t, "filings_today": len(hits)}
    except Exception as e:
        return {"ok": False, "latency_ms": 0, "error": str(e)}


@tool
def check_nasdaq_earnings() -> dict:
    """Check if NASDAQ earnings calendar API is working."""
    try:
        from datetime import timedelta
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        r = requests.get(
            f"https://api.nasdaq.com/api/calendar/earnings?date={tomorrow}",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=10,
        )
        t = r.elapsed.total_seconds() * 1000
        if r.status_code != 200:
            return {"ok": False, "latency_ms": t, "error": f"HTTP {r.status_code}"}
        rows = r.json().get("data", {}).get("rows", []) or []
        return {"ok": True, "latency_ms": t, "earnings_tomorrow": len(rows)}
    except Exception as e:
        return {"ok": False, "latency_ms": 0, "error": str(e)}


@tool
def check_groq_api() -> dict:
    """Check if Groq LLM API is responding with the saved API key."""
    if not GROQ_KEY:
        return {"ok": False, "latency_ms": 0, "error": "GROQ_API_KEY not set"}
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)
        t0 = time.time()
        msg = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
        latency = (time.time() - t0) * 1000
        return {"ok": True, "latency_ms": latency, "response": msg.choices[0].message.content}
    except Exception as e:
        return {"ok": False, "latency_ms": 0, "error": str(e)}


@tool
def check_streamlit_app() -> dict:
    """Check if the live Streamlit app is responding."""
    try:
        r = requests.get(APP_URL, timeout=15)
        t = r.elapsed.total_seconds() * 1000
        ok = r.status_code == 200
        return {"ok": ok, "latency_ms": t, "status_code": r.status_code}
    except Exception as e:
        return {"ok": False, "latency_ms": 0, "error": str(e)}


@tool
def create_github_issue(title: str, body: str) -> dict:
    """Create a GitHub issue to report a problem that needs attention."""
    if not GITHUB_TOKEN:
        return {"ok": False, "error": "GITHUB_TOKEN not set — issue not created"}
    try:
        r = requests.post(
            f"https://api.github.com/repos/{GITHUB_REPO}/issues",
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
            },
            json={"title": title, "body": body, "labels": ["monitoring", "bug"]},
            timeout=10,
        )
        if r.status_code == 201:
            return {"ok": True, "url": r.json().get("html_url")}
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── LangGraph nodes ───────────────────────────────────────────────────────────
TOOLS = [
    check_yahoo_finance,
    check_sec_edgar,
    check_nasdaq_earnings,
    check_groq_api,
    check_streamlit_app,
    create_github_issue,
]
TOOL_MAP = {t.name: t for t in TOOLS}


def run_checks(state: MonitorState) -> MonitorState:
    """Run all health checks in parallel and collect results."""
    print("🔍 Running health checks...")
    results = {}
    for check_fn in [check_yahoo_finance, check_sec_edgar,
                     check_nasdaq_earnings, check_groq_api, check_streamlit_app]:
        name = check_fn.name
        print(f"  checking {name}...", end=" ")
        result = check_fn.invoke({})
        results[name] = result
        status = "✅" if result.get("ok") else "❌"
        print(f"{status} {result}")

    issues = [
        f"**{name}** is DOWN: {r['error']}"
        for name, r in results.items()
        if not r.get("ok")
    ]

    return {**state, "check_results": results, "issues": issues}


def analyse_and_act(state: MonitorState) -> MonitorState:
    """Use Groq LLM to analyse failures and decide whether to create GitHub issues."""
    if not state["issues"]:
        print("✅ All systems healthy — no action needed.")
        return state

    if not GROQ_KEY:
        print(f"⚠️  {len(state['issues'])} issues found but no Groq key to analyse — printing:")
        for issue in state["issues"]:
            print(f"  • {issue}")
        return state

    llm = ChatGroq(api_key=GROQ_KEY, model="llama-3.3-70b-versatile", temperature=0)

    results_text = json.dumps(state["check_results"], indent=2)
    issues_text  = "\n".join(f"- {i}" for i in state["issues"])

    prompt = f"""You are a DevOps agent monitoring the APEX stock screener app ({APP_URL}).

Health check results:
{results_text}

Issues detected:
{issues_text}

For each issue, decide:
1. Is this a transient failure (API temporarily down, rate limit) or a persistent break?
2. Should we create a GitHub issue? Only create one if it looks persistent, not a 1-minute blip.
3. Write a clear GitHub issue title and body if needed.

Respond in JSON:
{{
  "assessment": "1-2 sentence overall health summary",
  "create_issues": [
    {{
      "title": "...",
      "body": "## Problem\\n...\\n## Impact\\n...\\n## Suggested Fix\\n..."
    }}
  ]
}}
Return ONLY the JSON."""

    try:
        response = llm.invoke([SystemMessage(content="You are a DevOps monitoring agent."),
                               HumanMessage(content=prompt)])
        import re
        raw = re.sub(r"^```[a-z]*\n?", "", response.content).rstrip("`").strip()
        decision = json.loads(raw)

        print(f"\n🤖 Agent assessment: {decision.get('assessment')}")

        fixed = []
        for issue_spec in decision.get("create_issues", []):
            result = create_github_issue.invoke({
                "title": issue_spec["title"],
                "body":  issue_spec["body"],
            })
            if result.get("ok"):
                print(f"  📌 GitHub issue created: {result['url']}")
                fixed.append(f"Issue filed: {issue_spec['title']}")
            else:
                print(f"  ⚠️  Could not create issue: {result.get('error')}")

        return {**state, "fixed": fixed}

    except Exception as e:
        print(f"  ⚠️  LLM analysis failed: {e} — issues logged to console only")
        return state


def report(state: MonitorState) -> MonitorState:
    """Print final summary."""
    n_checks = len(state["check_results"])
    n_ok     = sum(1 for r in state["check_results"].values() if r.get("ok"))
    n_bad    = n_checks - n_ok

    print(f"\n{'='*50}")
    print(f"APEX Monitor — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*50}")
    print(f"  Checks passed : {n_ok}/{n_checks}")
    print(f"  Issues found  : {n_bad}")
    print(f"  Actions taken : {len(state['fixed'])}")
    if state["fixed"]:
        for f in state["fixed"]:
            print(f"    • {f}")
    print(f"{'='*50}\n")

    # Exit with error code if any checks failed (GitHub Actions marks the run as failed)
    if n_bad > 0:
        sys.exit(1)

    return state


# ── Build the graph ───────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(MonitorState)
    g.add_node("run_checks",     run_checks)
    g.add_node("analyse_and_act", analyse_and_act)
    g.add_node("report",         report)

    g.add_edge(START,            "run_checks")
    g.add_edge("run_checks",     "analyse_and_act")
    g.add_edge("analyse_and_act", "report")
    g.add_edge("report",         END)
    return g.compile()


def run():
    graph = build_graph()
    graph.invoke({
        "messages":      [],
        "check_results": {},
        "issues":        [],
        "fixed":         [],
    })


if __name__ == "__main__":
    run()
