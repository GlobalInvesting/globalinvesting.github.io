#!/usr/bin/env python3
"""
generate_narrative_signals.py — AI Narrative & Market Signals generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads available data from the public site repo and generates two outputs
via Groq (llama-3.3-70b-versatile):

  1. ai-analysis/index.json
     { "narrative": "...", "regime": "RISK-ON", "generated_at": "..." }

  2. ai-analysis/signals.json
     [ { "time": "HH:MM", "priority": "critical", "title": "...", "text": "..." } ]

DATA SOURCES used in build_context():
  - fx-performance/{CCY}.json    → spot rates, 1W/1M % change (ECB/Frankfurter)
  - extended-data/{CCY}.json     → bond yields (FRED)
  - rates/{CCY}.json             → CB policy rates with HIKING/CUTTING/ON HOLD trend
  - cot-data/{CCY}.json          → CFTC speculative positioning (net, long%, week)
  - economic-data/{CCY}.json     → CPI, PMI, GDP, unemployment
  - news-data/news.json          → top high-impact FX headlines
  - intraday-data/quotes.json    → VIX, SPX, Gold, WTI, DXY, MOVE, BTC, Nikkei,
                                   STOXX, US yields (2Y/5Y/10Y/30Y/3M),
                                   21 FX pairs — all updated every 15min via yfinance
  - sentiment-data/myfxbook.json → retail trader sentiment (long/short %) for 24 pairs,
                                   updated every 15min — used for contrarian signals
  - ai-analysis/{CCY}.json       → per-currency AI summaries (if available)
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
MODEL      = "llama-3.3-70b-versatile"
KEY_SWITCH_PAUSE = 8  # seconds to wait when rotating to next key


def load_groq_keys():
    """Load all available Groq API keys from environment variables."""
    keys = []
    for var in ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]:
        val = os.environ.get(var, "").strip()
        if val:
            keys.append(val)
    return keys


def mask_key(key):
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def check_key_status(key):
    """Quick check: returns 'ok', 'daily_limit', 'rate_limit', or 'invalid'."""
    try:
        import time as _time
        r = requests.post(
            GROQ_URL,
            json={"model": MODEL, "messages": [{"role": "user", "content": "test"}], "max_tokens": 1},
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            timeout=10,
        )
        if r.status_code == 401:
            return "invalid"
        if r.status_code == 429:
            try:
                body = r.json().get("error", {}).get("message", "").lower()
            except Exception:
                body = r.text.lower()
            return "daily_limit" if ("daily" in body or "quota" in body or "per day" in body) else "rate_limit"
        return "ok"
    except Exception:
        return "ok"
SITE_DIR     = Path(".")
CURRENCIES   = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
CB_LABELS    = {"USD":"Fed","EUR":"ECB","GBP":"BoE","JPY":"BoJ",
                "AUD":"RBA","CAD":"BoC","CHF":"SNB","NZD":"RBNZ"}


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def build_context() -> str:
    lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"Current time: {now}\n")

    # 0. Live FX spot rates (from fx-performance — ECB/Frankfurter daily)
    # rateNow is quoted as USD per 1 unit of foreign currency (or foreign per USD for JPY etc.)
    PAIR_MAP = {
        "EUR": ("EUR/USD", lambda r: round(1/r, 4) if r else None),
        "GBP": ("GBP/USD", lambda r: round(1/r, 4) if r else None),
        "JPY": ("USD/JPY", lambda r: round(r, 2)   if r else None),
        "AUD": ("AUD/USD", lambda r: round(1/r, 4) if r else None),
        "CAD": ("USD/CAD", lambda r: round(r, 4)   if r else None),
        "CHF": ("USD/CHF", lambda r: round(r, 4)   if r else None),
        "NZD": ("NZD/USD", lambda r: round(1/r, 4) if r else None),
    }
    # PAIR_INVERT: True = pair is quoted as foreign/USD (EUR/USD, GBP/USD, AUD/USD, NZD/USD)
    #              False = pair is quoted as USD/foreign (USD/JPY, USD/CAD, USD/CHF)
    # For inverted pairs: pair rate = 1/rateNow. If rateNow rises (USD stronger), pair falls.
    # So pair_chg% = -fxPerformance (fxPerf measures foreign ccy vs basket, not pair direction).
    # For non-inverted pairs: pair rate = rateNow. If rateNow rises (foreign stronger vs USD),
    # the pair rises too, so pair_chg% = actual rate change (computed directly from rates).
    PAIR_INVERT = {"EUR": True, "GBP": True, "AUD": True, "NZD": True,
                   "JPY": False, "CAD": False, "CHF": False}

    fx_lines = []
    for ccy, (pair, transform) in PAIR_MAP.items():
        d = load_json(SITE_DIR / "fx-performance" / f"{ccy}.json")
        if d:
            r_now  = d.get("rateNow")
            r_7d   = d.get("rate7dAgo")
            r_30d  = d.get("rate30dAgo")
            date   = d.get("dateNow", "")
            if r_now:
                spot = transform(r_now)
                invert = PAIR_INVERT.get(ccy, True)
                # Compute % change directly from actual rates for accuracy
                def pct_chg(r1, r0, inv):
                    if not r1 or not r0: return None
                    p1 = (1/r1) if inv else r1
                    p0 = (1/r0) if inv else r0
                    return round((p1 - p0) / p0 * 100, 2)
                chg7d  = pct_chg(r_now, r_7d,  invert)
                chg30d = pct_chg(r_now, r_30d, invert)
                chg_str = ""
                if chg7d is not None:
                    sign = "+" if chg7d >= 0 else ""
                    chg_str += f"  1W:{sign}{chg7d}%"
                if chg30d is not None:
                    sign = "+" if chg30d >= 0 else ""
                    chg_str += f"  1M:{sign}{chg30d}%"
                fx_lines.append(f"  {pair}: {spot} (date:{date}){chg_str}")
    if fx_lines:
        lines.append("=== Live FX Spot Rates (ECB/Frankfurter) ===")
        lines.extend(fx_lines)
        lines.append("  NOTE: Use these exact rates when referencing pair levels in the narrative.")
        lines.append("")

    # 0b. Key bond yields from extended-data (10Y, 2Y, 5Y where available)
    yield_lines = []
    YIELD_MAP = {"USD": "US 10Y", "EUR": "DE 10Y (Bund)", "GBP": "UK 10Y", "JPY": "JP 10Y (JGB)"}
    for ccy, label in YIELD_MAP.items():
        d = load_json(SITE_DIR / "extended-data" / f"{ccy}.json")
        if d and "data" in d:
            y10 = d["data"].get("bond10y")
            if y10 is not None:
                yield_lines.append(f"  {label}: {y10:.2f}%")
            # USD also has 2Y and 5Y from FRED DGS2/DGS5
            if ccy == "USD":
                y2 = d["data"].get("bond2y")
                y5 = d["data"].get("bond5y")
                if y2 is not None:
                    yield_lines.append(f"  US 2Y:  {y2:.2f}%")
                if y5 is not None:
                    yield_lines.append(f"  US 5Y:  {y5:.2f}%")
    if yield_lines:
        lines.append("=== Key Bond Yields ===")
        lines.extend(yield_lines)
        lines.append("")

    # 1. Currency strength ranking & regime
    # 1. Central bank rates with trend
    lines.append("=== Central Bank Policy Rates ===")
    for ccy in CURRENCIES:
        d = load_json(SITE_DIR / "rates" / f"{ccy}.json")
        if d and d.get("observations"):
            obs = d["observations"]
            rate = obs[0].get("value", "N/A")
            date = obs[0].get("date", "?")
            trend = ""
            if len(obs) >= 3:
                try:
                    r0, r2 = float(obs[0]["value"]), float(obs[2]["value"])
                    trend = " [CUTTING]" if r0 < r2 else " [HIKING]" if r0 > r2 else " [ON HOLD]"
                except (ValueError, KeyError):
                    pass
            lines.append(f"  {CB_LABELS.get(ccy,ccy)} ({ccy}): {rate}%  (as of {date}){trend}")
    lines.append("")

    # 2. CFTC COT — correct field names: netPosition, longPositions, shortPositions
    lines.append("=== CFTC Speculative Positioning (COT) ===")
    for ccy in CURRENCIES:
        d = load_json(SITE_DIR / "cot-data" / f"{ccy}.json")
        if d:
            net       = d.get("netPosition", None)
            long_pos  = d.get("longPositions", 0) or 0
            short_pos = d.get("shortPositions", 0) or 0
            total     = long_pos + short_pos
            long_pct  = f"{round(long_pos/total*100,1)}%" if total > 0 else "N/A"
            week_end  = d.get("weekEnding") or d.get("reportDate", "?")
            if net is not None:
                bias    = "NET LONG" if net > 0 else "NET SHORT" if net < 0 else "FLAT"
                net_str = f"{net:+,}"
            else:
                bias, net_str = "N/A", "N/A"
            lines.append(
                f"  {ccy}: {bias}, net={net_str}, long%={long_pct}  (week ending {week_end})"
            )
    lines.append("")

    # 3. Key macro indicators
    lines.append("=== Key Macro Indicators ===")
    econ_keys = [
        ("inflation", "CPI%"), ("unemployment", "Unemp%"),
        ("gdpGrowth", "GDP_growth"), ("manufacturingPMI", "Mfg_PMI"),
        ("servicesPMI", "Svc_PMI"), ("bond10y", "10Y_yield"),
    ]
    for ccy in ["USD", "EUR", "JPY", "AUD"]:
        d = load_json(SITE_DIR / "economic-data" / f"{ccy}.json")
        if d and "data" in d:
            econ = d["data"]
            parts = []
            for key, label in econ_keys:
                val = econ.get(key)
                if isinstance(val, (int, float)):
                    parts.append(f"{label}={val:.2f}")
            if parts:
                lines.append(f"  {ccy}: {', '.join(parts)}")
    lines.append("")

    # 4. News headlines — correct field name: 'cur' (not 'currency' or 'tag')
    news_raw = load_json(SITE_DIR / "news-data" / "news.json", default={})
    articles = news_raw.get("articles", news_raw.get("items", [])) \
               if isinstance(news_raw, dict) else news_raw

    high_impact = [
        a for a in articles
        if a.get("impact") == "high" and a.get("lang", "en") == "en"
    ][:8]

    if high_impact:
        lines.append("=== Latest FX Headlines (high-impact) ===")
        for item in high_impact:
            title  = (item.get("title") or "").strip()
            ccy    = item.get("cur") or item.get("currency") or ""   # 'cur' is the correct field
            time   = item.get("time", "")
            expand = (item.get("expand") or "").replace("\n", " ").strip()[:100]
            if title:
                tag      = f"[{ccy}] " if ccy else ""
                time_str = f"{time} " if time else ""
                lines.append(f"  • {time_str}{tag}{title}")
                if expand and len(expand) > 30:
                    lines.append(f"    → {expand}{'…' if len(expand)==100 else ''}")
        lines.append("")

    # 5. Intraday quotes — VIX, SPX, Gold, WTI, DXY, yields, BTC, FX pairs (yfinance via GitHub Action)
    intraday = load_json(SITE_DIR / "intraday-data" / "quotes.json")
    if intraday and intraday.get("quotes"):
        q = intraday["quotes"]
        updated = intraday.get("updated", "")
        lines.append(f"=== Intraday Market Quotes (yfinance, updated: {updated}) ===")

        # Cross-asset
        cross_asset = []
        for key, label, fmt_fn in [
            ("vix",   "VIX",          lambda v: f"{v:.1f}"),
            ("spx",   "S&P 500",      lambda v: f"{v:,.0f}"),
            ("gold",  "Gold (XAU)",   lambda v: f"${v:,.0f}"),
            ("wti",   "WTI Oil",      lambda v: f"${v:.1f}"),
            ("dxy",   "DXY",          lambda v: f"{v:.2f}"),
            ("move",  "MOVE Index",   lambda v: f"{v:.1f}"),
            ("btc",   "Bitcoin",      lambda v: f"${v:,.0f}"),
            ("nikkei","Nikkei 225",   lambda v: f"{v:,.0f}"),
            ("stoxx", "STOXX 50",     lambda v: f"{v:,.0f}"),
        ]:
            d = q.get(key)
            if d and d.get("close"):
                pct = d.get("pct", 0)
                sign = "+" if pct >= 0 else ""
                cross_asset.append(f"  {label}: {fmt_fn(d['close'])}  1D:{sign}{pct:.2f}%")
        if cross_asset:
            lines.extend(cross_asset)
            lines.append("")

        # Yields (intraday — más frescos que extended-data)
        yield_live = []
        for key, label in [("us10y","US 10Y"),("us2y","US 2Y"),("us5y","US 5Y"),("us30y","US 30Y"),("us3m","US 3M")]:
            d = q.get(key)
            if d and d.get("close"):
                pct = d.get("pct", 0)
                sign = "+" if pct >= 0 else ""
                yield_live.append(f"  {label}: {d['close']:.3f}%  1D:{sign}{pct:.2f}%")
        if yield_live:
            lines.append("  -- Intraday Yields (yfinance, more current than extended-data) --")
            lines.extend(yield_live)
            lines.append("")

        # FX pairs intraday (if available — requires workflow v2.2+)
        fx_intraday = []
        FX_PAIRS_DISPLAY = [
            ("eurusd","EUR/USD",5), ("gbpusd","GBP/USD",5), ("usdjpy","USD/JPY",3),
            ("audusd","AUD/USD",5), ("usdchf","USD/CHF",5), ("usdcad","USD/CAD",5),
            ("nzdusd","NZD/USD",5), ("gbpjpy","GBP/JPY",3), ("eurjpy","EUR/JPY",3),
        ]
        for key, label, dec in FX_PAIRS_DISPLAY:
            d = q.get(key)
            if d and d.get("close"):
                pct = d.get("pct", 0)
                sign = "+" if pct >= 0 else ""
                fx_intraday.append(f"  {label}: {d['close']:.{dec}f}  1D:{sign}{pct:.2f}%")
        if fx_intraday:
            lines.append("  -- FX Intraday Prices (yfinance, ~15min delay) --")
            lines.extend(fx_intraday)
            lines.append("")

    # 6. Myfxbook retail sentiment (community outlook, updated every 15min)
    mfx = load_json(SITE_DIR / "sentiment-data" / "myfxbook.json")
    if mfx and mfx.get("pairs"):
        updated = mfx.get("updated", "")
        lines.append(f"=== Retail FX Sentiment — Myfxbook Community Outlook (updated: {updated}) ===")
        lines.append("  (% of retail traders long vs short — contrarian signal: extremes often precede reversals)")
        lines.append("")

        # Classify extremes for signal generation
        extremes = []
        for p in mfx["pairs"]:
            sym   = p["sym"]
            lng   = p["long"]
            sht   = p["short"]
            dom   = max(lng, sht)
            side  = "LONG" if lng >= sht else "SHORT"
            bias  = f"{lng}% long / {sht}% short"
            note  = ""
            if dom >= 80:
                note = f"  ⚠ EXTREME {side} — contrarian reversal risk"
                extremes.append((sym, side, dom, lng, sht))
            elif dom >= 70:
                note = f"  ↑ Strong {side} bias"
            lines.append(f"  {sym}: {bias}{note}")

        if extremes:
            lines.append("")
            lines.append("  SENTIMENT EXTREMES (highest contrarian signal value):")
            for sym, side, dom, lng, sht in sorted(extremes, key=lambda x: -x[2]):
                lines.append(f"    {sym}: {dom}% {side} — retail heavily positioned, watch for squeeze")
        lines.append("")

    # 7. Per-currency AI summaries (if available)
    found = False
    for ccy in CURRENCIES:
        d = load_json(SITE_DIR / "ai-analysis" / f"{ccy}.json")
        if d and "summary" in d:
            if not found:
                lines.append("=== Per-Currency AI Summaries ===")
                found = True
            lines.append(f"  {ccy}: {d['summary'][:200]}")
    if found:
        lines.append("")

    return "\n".join(lines)


def call_groq(api_key: str, system: str, user: str, max_tokens: int = 800) -> str:
    """Call Groq API. Raises RuntimeError with DAILY_LIMIT or RATE_LIMIT tag on 429."""
    import time
    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.25,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # One retry on transient 429 (rate limit within key)
    for attempt in range(2):
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        if r.status_code == 429:
            try:
                body = r.json().get("error", {}).get("message", "").lower()
            except Exception:
                body = r.text.lower()
            if "daily" in body or "quota" in body or "per day" in body:
                raise RuntimeError("DAILY_LIMIT")
            # Transient rate limit — wait and retry once
            retry_after = int(r.headers.get("retry-after", 0))
            wait = retry_after if retry_after > 0 else 20
            print(f"  Groq rate limit — waiting {wait}s (attempt {attempt+1}/2)...")
            time.sleep(wait)
            continue
        if r.status_code == 401:
            raise RuntimeError("INVALID_KEY")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("RATE_LIMIT")


# ── Prompts ───────────────────────────────────────────────────────────────────

NARRATIVE_SYSTEM = """\
You are a senior FX strategist writing a live market snapshot for a professional trading terminal.
Your output appears at the very top of the screen — the first thing experienced traders read each session.

Write like a Bloomberg terminal snapshot or a top-desk morning note: specific, data-driven, present-tense.
Reference actual pairs, price levels, catalysts, and central bank dynamics based ONLY on the data you receive.

STYLE GUIDE — follow this tone and density exactly:
  "USD weakens as Fed signals patience — DXY retreats to [LEVEL]. EUR/USD firms at [LEVEL]; \
ECB rhetoric turns cautious. JPY supported by BoJ tightening signals; USD/JPY slides to [LEVEL]. \
Gold +0.4% on safe-haven demand amid equity weakness."

  "Risk-off tone dominates: equities slide, oil +2.1% on supply concerns. USD bid across \
the board. EUR/USD at [LEVEL], under pressure; GBP/USD vulnerable near [LEVEL]. JPY \
outperforms on safe-haven flows."

  "EUR/USD extends gains to [LEVEL] on softer US data — ECB holds firm. JPY bid on BoJ rate hike \
speculation; USD/JPY pressured to [LEVEL]. Commodity bloc supported: AUD/USD firm, gold higher — \
risk-on tone confirmed."

CRITICAL RULES:
- Use ONLY price levels, rates, and data explicitly provided in the market data below.
- NEVER invent, hallucinate, or copy price levels from examples above — [LEVEL] is a placeholder only.
- Describe pairs at their CURRENT level — use "at", "near", "trading at", "firming to", "retreating to".
- NEVER write "slips below X" or "breaks above X" using the current price — that implies X was just broken.
  Instead say "EUR/USD at 1.1517" or "EUR/USD firms at 1.1517" or "EUR/USD offered near 1.1517".
- 3-4 sentences. Target 380-480 characters total.
- Always reference at least 3 specific FX pairs with directional language and actual price levels from the data.
- Include a central bank or yield context (use intraday yields when available — they are more current).
- Include a cross-asset note (VIX, equities, oil, gold) if the data supports it.
- If retail sentiment shows an extreme (80%+ one-sided), briefly note it as a contrarian risk.
- Use em-dash (—) for flow. No bullet points. No "I". No disclaimers. No phrases like "as of".
- INTERNAL CONSISTENCY — STRICT AND MANDATORY:
  STEP 1 — Determine USD direction BEFORE writing. Choose exactly one:
    • RISK-OFF regime → USD is "bid" / "firm" / "broadly bid" / "strengthening"
    • RISK-ON regime  → USD is "offered" / "under pressure" / "weakening" / "broadly offered"
    • MIXED/NEUTRAL   → USD is "mixed" or "consolidating" — avoid strong directional language
  STEP 2 — Write the entire narrative using ONLY the direction chosen in Step 1.
  STEP 3 — Re-read the narrative. If ANY sentence contradicts the chosen USD direction, DELETE or rewrite it.
  FORBIDDEN: combining "USD bid" with "USD broadly offered" or "DXY weakening" in the same narrative.
  FORBIDDEN: ending a USD-bid narrative with "DXY weakening" or any bearish USD qualifier.
  EXAMPLE OF VIOLATION (never do this): "USD bid across the board … DXY weakening — USD broadly offered."
  EXAMPLE OF CORRECT (RISK-OFF): "USD bid across the board — DXY firms. EUR/USD pressured at 1.1517; GBP/USD offered near 1.3281."
  EXAMPLE OF CORRECT (RISK-ON): "USD broadly offered — DXY retreats. EUR/USD extends gains to 1.1517; GBP/USD firms near 1.3281."
- The 1W/1M % changes in the data reflect medium-term moves — do NOT use them to define today's direction.
  Infer today's USD direction SOLELY from: the regime label, CB stance (hiking/cutting/on hold), and pair levels.
- Regime must be exactly one of: RISK-ON, RISK-OFF, MIXED, NEUTRAL
- Respond ONLY with valid JSON — no markdown fences, no preamble, no explanation.

JSON format:
{
  "narrative": "...",
  "regime": "RISK-OFF"
}"""


SIGNALS_SYSTEM = """\
You are a senior FX risk analyst writing actionable alerts for a professional trading terminal.
Your output feeds the "Alerts & Market Signals" panel — read by active prop traders and fund managers.

Generate 4-5 specific, actionable market signals from the data provided.

PRIORITY:
  "critical" — key technical level broken, imminent CB decision, major macro surprise, intervention risk
  "warning"  — elevated risk, notable technical setup, positioning extreme at multi-year COT levels,
               significant macro divergence
  "info"     — regime note, cross-asset observation, VIX/MOVE level note, yield curve context

SIGNAL COMPOSITION — MANDATORY:
  - MINIMUM 2 signals must be macro or CB-driven (central bank stance, yield differentials, key data)
  - MINIMUM 1 signal must be technical (key level, breakout, range boundary from intraday data)
  - MAXIMUM 1 signal may reference retail sentiment (Myfxbook) — only when extreme (≥85%) AND
    it strongly DIVERGES from CFTC/COT institutional positioning. If retail and COT agree, skip it.
  - Cross-asset signals (VIX, MOVE, equities, gold) count as macro signals.

STYLE — write like a desk alert, not a summary:
  Title: "USD/JPY" | "ECB" | "Gold" | "CFTC — EUR" | "Risk Regime" | "BoJ Watch"
  Text: specific levels and catalysts from the data provided, 1-2 sentences, 80-230 chars.

CRITICAL RULES:
- Use ONLY price levels, rates, and data explicitly provided in the market data below.
- NEVER invent or hallucinate price levels not in the data.
- NEVER reference "spread", "confidence", "scoring", or "long/short" trade recommendations.
- Focus signals on: CB decisions, key technical levels, macro surprises, positioning extremes, regime shifts.
- CROSS-ASSET signals: Use VIX level, MOVE Index, and S&P 500 direction from intraday quotes to contextualize
  risk regime. VIX >25 = elevated fear. MOVE >120 = bond market stress.
- Sort: critical first, then warning, then info.
- No fluff. No "I think". Name pairs, CB names, actual levels from the data.
- Time: use the current UTC time provided for all signals.
- Respond ONLY with a valid JSON array — no markdown, no preamble, no extra text.

JSON format:
[
  { "time": "HH:MM", "priority": "critical", "title": "USD/JPY", "text": "..." }
]"""


def generate_narrative(api_key: str, context: str) -> dict:
    raw = call_groq(api_key, NARRATIVE_SYSTEM, f"Market data:\n\n{context}")
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    data = json.loads(raw)
    return {
        "narrative":    data.get("narrative", "Market data unavailable."),
        "regime":       data.get("regime", "NEUTRAL").upper(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def generate_signals(api_key: str, context: str) -> list:
    now_utc = datetime.now(timezone.utc).strftime("%H:%M")
    raw = call_groq(api_key, SIGNALS_SYSTEM, f"Current time UTC: {now_utc}\n\nMarket data:\n\n{context}", max_tokens=600)
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    start, end = raw.find("["), raw.rfind("]") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    signals = json.loads(raw)
    return [
        {
            "time":     s.get("time", now_utc),
            "priority": s.get("priority", "info").lower(),
            "title":    str(s.get("title", ""))[:40],
            "text":     str(s.get("text",  ""))[:300],
        }
        for s in signals if isinstance(s, dict)
    ]


def main():
    import time
    keys = load_groq_keys()
    if not keys:
        print("ERROR: No Groq API keys found (GROQ_API_KEY / GROQ_API_KEY_2 / GROQ_API_KEY_3).", file=sys.stderr)
        sys.exit(1)
    print(f"Groq keys available: {len(keys)}")

    out_dir = SITE_DIR / "ai-analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building context from available data files...")
    context = build_context()
    print(f"Context length: {len(context)} chars")

    if "--debug-context" in sys.argv:
        print("\n--- CONTEXT ---")
        print(context[:4000])
        print("--- END ---\n")

    def call_with_key_rotation(fn_name, fn, *args):
        """Call fn(api_key, *args) rotating through keys on DAILY_LIMIT."""
        key_idx = 0
        while key_idx < len(keys):
            key = keys[key_idx]
            try:
                print(f"  Groq Key {key_idx+1} ({mask_key(key)})...")
                return fn(key, *args)
            except RuntimeError as e:
                err = str(e)
                if "DAILY_LIMIT" in err or "RATE_LIMIT" in err:
                    label = "daily limit reached" if "DAILY_LIMIT" in err else "rate limit — rotating"
                    print(f"  ⛔ Key {key_idx+1} {label}")
                    key_idx += 1
                    if key_idx < len(keys):
                        print(f"  🔄 Switching to Key {key_idx+1} — pausing {KEY_SWITCH_PAUSE}s...")
                        time.sleep(KEY_SWITCH_PAUSE)
                    else:
                        print("  ⛔ All keys exhausted")
                        raise RuntimeError(f"{fn_name}: all Groq keys exhausted") from e
                else:
                    raise
        raise RuntimeError(f"{fn_name}: no keys available")

    narrative_file = out_dir / "index.json"
    signals_file   = out_dir / "signals.json"

    # Load existing content — kept if Groq is unavailable (keys exhausted/daily limit)
    existing_narrative = load_json(narrative_file)
    existing_signals   = load_json(signals_file, default=[])

    # Narrative
    print("Generating narrative via Groq...")
    try:
        narrative = call_with_key_rotation("narrative", generate_narrative, context)
        with open(narrative_file, "w") as f:
            json.dump(narrative, f, indent=2, ensure_ascii=False)
        print(f"  Regime:    {narrative['regime']}")
        print(f"  Narrative: {narrative['narrative']}")
    except Exception as e:
        print(f"  ⚠️  Groq unavailable for narrative: {e}", file=sys.stderr)
        if existing_narrative and existing_narrative.get("narrative", "").strip():
            # Keep previous — just update generated_at so the page knows it was checked
            existing_narrative["checked_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(narrative_file, "w") as f:
                json.dump(existing_narrative, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Preserved previous narrative (regime={existing_narrative.get('regime','?')})")
        else:
            with open(narrative_file, "w") as f:
                json.dump({
                    "narrative":    "Market data update in progress.",
                    "regime":       "NEUTRAL",
                    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }, f, indent=2)

    # Signals — brief pause between calls
    print("Pausing 8s before signals call...")
    time.sleep(8)
    print("Generating signals via Groq...")
    try:
        signals = call_with_key_rotation("signals", generate_signals, context)
        with open(signals_file, "w") as f:
            json.dump(signals, f, indent=2, ensure_ascii=False)
        print(f"  Generated {len(signals)} signals:")
        for s in signals:
            print(f"    [{s['priority'].upper()[:4]}] {s['title']}: {s['text'][:80]}")
    except Exception as e:
        print(f"  ⚠️  Groq unavailable for signals: {e}", file=sys.stderr)
        if existing_signals:
            print(f"  ✅ Preserved {len(existing_signals)} previous signals")
            # No rewrite needed — file already has them
        else:
            with open(signals_file, "w") as f:
                json.dump([], f)

    print("Done.")


if __name__ == "__main__":
    main()
