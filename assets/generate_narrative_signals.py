#!/usr/bin/env python3
"""
generate_narrative_signals.py — AI Narrative & Market Signals generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads available data from the public site repo and generates two outputs
via Gemini (primary editor) with Groq as coherence validator / fallback:

  1. ai-analysis/index.json
     { "narrative": "...", "regime": "RISK-ON", "generated_at": "..." }

  2. ai-analysis/signals.json
     [ { "time": "HH:MM", "priority": "critical", "title": "...", "text": "..." } ]

DATA SOURCES used in build_context():
  - intraday-data/quotes.json    → spot rates, 1W/1M % change (yfinance, ~5 min) [replaces fx-performance/]
  - extended-data/{CCY}.json     → bond yields (FRED)
  - rates/{CCY}.json             → CB policy rates with HIKING/CUTTING/ON HOLD trend
  - ois-rates/rates.json         → OIS overnight benchmarks (SOFR/€STR/SONIA/TONA/CORRA/SARON/AONIA/OCR)
                                   + pre-computed OIS carry differentials for 8 key pairs
  - cot-data/{CCY}.json          → CFTC speculative positioning (net, long%, week)
  - economic-data/ (REMOVED v7.24.1 — stale data, no incremental signal value)
  - news-data/news.json          → top high-impact FX headlines
  - rr-data/rr.json              → 25-delta Risk Reversals (Saxo Bank, 1M tenor, indicative mid)
                                   for 7 pairs: EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CAD,
                                   USD/CHF, EUR/JPY — options market directional bias
  - intraday-data/quotes.json    → VIX, SPX, Gold, WTI, DXY, MOVE, BTC, Nikkei,
                                   STOXX, US yields (2Y/5Y/10Y/30Y/3M),
                                   21 FX pairs — all updated every 15min via yfinance
  - intraday-data/quotes.json    → fx_etf_iv: CBOE FX implied volatility indices
                                   (^EUVIX/^BPVIX/^JYVIX/^AUDVIX/^USDVIX) for 5 pairs
  - intraday-data/quotes.json    → hv30: 30-day realized volatility (annualised) for 5 pairs
                                   Vol Risk Premium (VRP = IV − HV30) pre-computed per pair
  - sentiment-data/myfxbook.json → retail trader sentiment (long/short %) for 24 pairs,
                                   updated every 15min — used for contrarian signals
  - intraday-data/quotes.json    → correlations[] array: 60d Pearson + historical norm + z_score
                                   for 12 pairs incl. DXY/SPX and Gold/DXY regime signals
  - ai-analysis/context_snapshot.json → cross_pct (intraday % changes per symbol) and
                                   closed_symbols (list of symbols with no live data today due
                                   to bank holidays or market closure) — written by
                                   fetch_intraday_quotes.py PASO 6c each run.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
MODEL         = "llama-3.3-70b-versatile"
DRIVERS_MODEL = "llama-3.1-8b-instant"   # Smaller model for structured driver JSON — separate daily quota

# ── Gemini (primary editor) + Groq (coherence validator / fallback) ──────────
# Architecture (v7.54.0):
#   1. Gemini 3 Flash (PRIMARY): generates narrative + signals from scratch.
#      Superior reasoning, better instruction-following, larger context window
#      (1M tokens), and 64K output vs Groq's 8K — reduces retries and
#      directional-coherence errors that Python guards cannot fix deterministically.
#   2. Python guards (unchanged): deterministic cleaning layer between editor and output.
#   3. Groq (FALLBACK): when ALL Gemini keys are exhausted or fail, Groq takes over.
#      Also retained for: currency drivers, session context (cheaper, fine for structured notes).
#
# Why Gemini as primary works on free tier:
#   - 3 keys from 3 separate Google accounts = 3 × 250 RPD = 750 RPD (independent limits).
#   - Gemini makes fewer errors → fewer retries → actual consumption ~2 calls/run vs ~4 for Groq.
#   - 24 runs/day × 2 calls = ~48 calls/day — 15x headroom within 750 RPD.
#
# Model: gemini-3-flash-preview — upgraded from gemini-2.5-flash (v7.53.7).
# Free-tier limits (May 2026): ~250 RPD per project, 10 RPM.
# Keys: GEMINI_API_KEY, GEMINI_API_KEY_2 … GEMINI_API_KEY_5 (env vars).
GEMINI_URL        = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODEL      = "gemini-3-flash-preview"         # upgraded from gemini-2.5-flash
GEMINI_MODEL_LITE = "gemini-3.1-flash-lite-preview"  # faster/cheaper fallback option
KEY_SWITCH_PAUSE_DAILY  = 1   # pause when rotating off a daily-limit key (key is dead — no need to wait)
KEY_SWITCH_PAUSE_RATE   = 20  # pause when rotating off a rate-limit key (Groq RPM window = 60s; 20s gives meaningful backoff)

# ── Token budget guard ────────────────────────────────────────────────────────
# Tracks cumulative tokens consumed per run across ALL Groq calls in this process.
# When the budget is exceeded, optional repair retries (forbidden-language,
# token-missing, news-integration, MIXED-style) are skipped — they are quality
# improvements, not correctness requirements. The first-pass output is kept as-is.
#
# Budget per pool (signals: keys 3, 4, 5 with 5-key setup):
#   Signals primary call:    ~1,100 tokens
#   COT-OIS enforcement:     ~550 tokens/divergence × 2 typical = ~1,100
#   One optional retry:      ~700 tokens
#   Safety margin:           ~550 tokens
#   Signals target per run:  ~4,000 tokens
#   12 runs/day × 4,000 = 48,000/day — well within 500k/key/day × 2 keys.
#
# The hard cap is set at 40,000 per run (signals path only) — enough for:
#   • 3 full repair cycles (each ~1,100 tokens)
#   • 2 COT-OIS divergences
#   • 1 degenerate-response retry
# Any run that would exceed this has a quality problem in the prompt, not a
# budget problem — the cap reveals it rather than silently burning keys.
_SIGNALS_TOKEN_BUDGET = 40_000   # tokens per run — signals path only
_signals_tokens_used  = 0        # reset to 0 at process start; accumulated in generate_signals()

# G8 scorecard thresholds — populated by build_context() and consumed by generate_signals()
# for Python-side suppression of signals that fail the trigger thresholds.
_G8_SCORECARD: dict = {}  # keys: leader_avg, laggard_avg, spread, leader_ccy, laggard_ccy


def is_market_closed() -> tuple:
    """Mirror the FX session logic from dashboard.js updateSessions().

    FX market open window: Sunday 21:00 UTC → Friday 21:00 UTC.
    Closed window:
      - Saturday (js_day == 6), any hour
      - Sunday   (js_day == 0) before 21:00 UTC
      - Friday   (js_day == 5) at or after 21:00 UTC

    Also detects USD bank holidays from ff_calendar.json (written by
    fetch_ff_calendar.py). On a USD holiday the FX market is technically
    open but USD-denominated instruments (equities, commodities) are closed
    and have no live intraday data — this is treated as a partial closure.

    Returns (closed: bool, next_session: str) where next_session is used in
    the closed-market prompt to orient the AI toward the right time horizon.
    """
    now      = datetime.now(timezone.utc)
    # Python weekday: Mon=0…Fri=4, Sat=5, Sun=6
    # JS-style:       Sun=0…Fri=5, Sat=6  →  js_day = (py_weekday + 1) % 7
    js_day   = (now.weekday() + 1) % 7
    utc_hour = now.hour

    closed = (
        js_day == 6                          # Saturday all day
        or (js_day == 0 and utc_hour < 21)  # Sunday before 21:00 UTC
        or (js_day == 5 and utc_hour >= 21) # Friday at/after 21:00 UTC
    )

    if not closed:
        return False, ""

    # Next open is always Sydney/Tokyo Sunday 21:00 UTC.
    # Find days until next Sunday (Python weekday 6).
    py_day = now.weekday()
    days_until_sunday = (6 - py_day) % 7
    if days_until_sunday == 0 and utc_hour >= 21:
        days_until_sunday = 7  # already past Sunday 21:00 — next Sunday
    next_open = (now + timedelta(days=days_until_sunday)).replace(
        hour=21, minute=0, second=0, microsecond=0
    )
    next_session = f"Sydney/Tokyo open (Sunday {next_open.strftime('%d %b')} 21:00 UTC)"
    return True, next_session


def get_today_holidays() -> list[dict]:
    """Return today's G8 bank holidays from ff_calendar.json.

    Returns a list of {title, currency, dateISO} dicts for holidays today (UTC).
    Empty list if the file doesn't exist, can't be parsed, or no holidays today.
    Written by fetch_ff_calendar.py at each calendar workflow run.
    """
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        ff = load_json(SITE_DIR / "calendar-data" / "ff_calendar.json")
        if not ff:
            return []
        return [h for h in (ff.get("holidays") or []) if h.get("dateISO") == today_iso]
    except Exception:
        return []


def get_closed_symbols() -> set[str]:
    """Return set of symbol keys (e.g. 'gold', 'spx', 'dxy') with no live intraday data today.

    Reads closed_symbols from ai-analysis/context_snapshot.json — written by
    fetch_intraday_quotes.py PASO 6c. Falls back to deriving from ff_calendar
    holidays if context_snapshot is missing or has no closed_symbols field.
    """
    # Primary: read from context_snapshot (most accurate — written each quotes run)
    snapshot = load_json(SITE_DIR / "ai-analysis" / "context_snapshot.json")
    if snapshot and isinstance(snapshot.get("closed_symbols"), list):
        return set(snapshot["closed_symbols"])

    # Fallback: derive from today's holidays in ff_calendar.json
    USD_SYMS = {"spx", "nasdaq", "vix", "gold", "wti", "dxy", "move", "us2y", "us10y", "us3m", "us5y", "us30y", "btc"}
    EUR_SYMS = {"stoxx"}
    GBP_SYMS = {"ftse"}
    JPY_SYMS = {"nikkei"}
    CCY_TO_SYMS = {"USD": USD_SYMS, "EUR": EUR_SYMS, "GBP": GBP_SYMS, "JPY": JPY_SYMS}

    closed: set[str] = set()
    for h in get_today_holidays():
        ccy = (h.get("currency") or "").upper()
        closed |= CCY_TO_SYMS.get(ccy, set())
    return closed



def load_groq_keys() -> list:
    """Load all available Groq API keys from environment variables."""
    keys = []
    for var in ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4", "GROQ_API_KEY_5", "GROQ_API_KEY_6"]:
        val = os.environ.get(var, "").strip()
        if val:
            keys.append(val)
    return keys


def load_gemini_keys() -> list:
    """Load Gemini API keys from environment variables (GEMINI_API_KEY … GEMINI_API_KEY_5)."""
    keys = []
    for var in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
                "GEMINI_API_KEY_4", "GEMINI_API_KEY_5"]:
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
    market_closed, next_session = is_market_closed()
    lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"Current time: {now}\n")
    if market_closed:
        lines.append(f"MARKET STATUS: CLOSED — FX markets are closed. All price data below are FRIDAY CLOSING LEVELS, not live quotes.")
        lines.append(f"Next session: {next_session}")
        lines.append(f"Analysis orientation: project implications of weekend news for the next session open.\n")
    else:
        lines.append(f"MARKET STATUS: OPEN\n")

    # 0. Live FX spot rates (from intraday-data/quotes.json — yfinance, ~5 min refresh)
    # Previously sourced from fx-performance/*.json (ECB/Frankfurter, daily).
    # Migrated in v7.23.0: quotes.json has close (spot), pct1w (1W) and pct1m (1M)
    # for all 7 majors — fresher data (5-min vs daily), single source of truth.
    PAIR_KEYS = [
        ("eurusd", "EUR/USD", 4),
        ("gbpusd", "GBP/USD", 4),
        ("usdjpy", "USD/JPY", 2),
        ("audusd", "AUD/USD", 4),
        ("usdcad", "USD/CAD", 4),
        ("usdchf", "USD/CHF", 4),
        ("nzdusd", "NZD/USD", 4),
    ]

    intraday_raw = load_json(SITE_DIR / "intraday-data" / "quotes.json")
    iq = intraday_raw.get("quotes", {}) if intraday_raw else {}
    iq_updated = intraday_raw.get("updated", "") if intraday_raw else ""

    fx_lines = []
    for key, pair, dec in PAIR_KEYS:
        d = iq.get(key)
        if not d or not d.get("close"):
            continue
        spot = round(float(d["close"]), dec)
        chg_str = ""
        p1w = d.get("pct1w")
        p1m = d.get("pct1m")
        if p1w is not None:
            sign = "+" if p1w >= 0 else ""
            chg_str += f"  1W:{sign}{p1w:.2f}%"
        if p1m is not None:
            sign = "+" if p1m >= 0 else ""
            chg_str += f"  1M:{sign}{p1m:.2f}%"
        fx_lines.append(f"  {pair}: {spot} (updated:{iq_updated[:10]}){chg_str}")
    if fx_lines:
        rate_label = "=== Friday Closing FX Rates (reference levels — market closed) ===" if market_closed \
                     else "=== Live FX Spot Rates (yfinance · ~5 min delay) ==="
        note_label = "  NOTE: These are FRIDAY CLOSING LEVELS — reference only. Do not present as live prices." if market_closed \
                     else "  NOTE: Use these exact rates when referencing pair levels in the narrative."
        lines.append(rate_label)
        lines.extend(fx_lines)
        lines.append(note_label)
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

    # 1. Central bank rates with trend + forward bias from meetings.json
    # Pause detection: if the most recent rate change is ≥90 days ago, trend = ON HOLD
    # regardless of rate trajectory — mirrors Bloomberg behaviour.
    # Forward bias (meetings.json `bias` field) reflects OIS/futures market consensus
    # for the NEXT meeting — distinct from the historical trend arrow.
    meetings_data = load_json(SITE_DIR / "meetings-data" / "meetings.json", default={})
    meetings_map = meetings_data.get("meetings", {})

    lines.append("=== Central Bank Policy Rates ===")
    for ccy in CURRENCIES:
        d = load_json(SITE_DIR / "rates" / f"{ccy}.json")
        if d and d.get("observations"):
            obs = d["observations"]
            rate = obs[0].get("value", "N/A")
            last_date_str = obs[0].get("date", "?")
            trend = ""
            if len(obs) >= 2:
                try:
                    from datetime import date as _date
                    # Pause detection: find the date of the last actual rate change
                    last_change_date = None
                    r_prev = float(obs[0]["value"])
                    for o in obs[1:]:
                        r_this = float(o["value"])
                        if r_this != r_prev:
                            last_change_date = _date.fromisoformat(o["date"])
                            break
                    today = _date.today()
                    if last_change_date is None:
                        # All observations same rate — no change in full history
                        trend = " [ON HOLD]"
                    else:
                        days_since_change = (today - last_change_date).days
                        r0 = float(obs[0]["value"])
                        r_at_change = float(next(
                            o["value"] for o in obs[1:]
                            if _date.fromisoformat(o["date"]) == last_change_date
                        ))
                        if days_since_change >= 90:
                            trend = " [ON HOLD]"
                        elif r0 < r_at_change:
                            trend = " [CUTTING]"
                        elif r0 > r_at_change:
                            trend = " [HIKING]"
                        else:
                            trend = " [ON HOLD]"
                except (ValueError, KeyError, StopIteration):
                    pass

            # Forward bias from meetings.json (OIS/futures market consensus)
            mtg = meetings_map.get(ccy, {})
            bias_raw = mtg.get("bias", "")
            next_mtg = mtg.get("nextMeeting", "")
            bias_str = ""
            if bias_raw:
                bias_label = {"cut": "↓ Cut", "hold": "→ Hold", "hike": "↑ Hike"}.get(bias_raw, bias_raw)
                bias_str = f" | Next meeting: {next_mtg} | Market bias: {bias_label}"

            lines.append(f"  {CB_LABELS.get(ccy,ccy)} ({ccy}): {rate}%  (as of {last_date_str}){trend}{bias_str}")
    lines.append("")

    # 1b. OIS overnight rates — CIP-consistent carry benchmarks
    # Source: ois-rates/rates.json (fetch_ois_rates.py, daily at 23:15 UTC)
    # These are the overnight benchmark rates used for CIP forward pricing and
    # institutional carry analysis — NOT the CB policy rates above.
    # Policy rate vs OIS overnight can diverge by up to 50bp during transitions.
    # Industry convention (BIS, Goldman, JPM FX carry research):
    #   Use overnight benchmarks (SOFR, €STR, SONIA, TONA, CORRA, SARON) for carry
    #   calculations, not policy rates. This is the standard for CIP-covered carry.
    # OIS CARRY DIFFERENTIAL = the precise spread traders use for forward pricing.
    # When OIS differential ≠ CB rate differential, the OIS figure is authoritative
    # for carry framing — e.g. SOFR vs €STR is more precise than Fed Funds vs DFR.
    _ois_raw = load_json(SITE_DIR / "ois-rates" / "rates.json")
    if _ois_raw and _ois_raw.get("rates"):
        _ois_rates  = _ois_raw["rates"]
        _ois_src    = _ois_raw.get("sources", {})
        _ois_dates  = _ois_raw.get("dates", {})
        _ois_upd    = _ois_raw.get("updated", "")
        lines.append(f"=== OIS Overnight Benchmark Rates (CIP carry basis · updated: {_ois_upd}) ===")
        lines.append("  (Use these — not CB policy rates — for precise carry differentials and CIP forward framing)")
        _OIS_CCY_ORDER = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        _OIS_NAMES = {
            "USD": "SOFR", "EUR": "€STR", "GBP": "SONIA",
            "JPY": "TONA", "AUD": "AONIA", "CAD": "CORRA",
            "CHF": "SARON", "NZD": "OCR",
        }
        for _oc in _OIS_CCY_ORDER:
            _or = _ois_rates.get(_oc)
            if _or is None:
                continue
            _oname  = _ois_src.get(_oc) or _OIS_NAMES.get(_oc, _oc)
            _odate  = _ois_dates.get(_oc, "")
            _ocb    = CB_LABELS.get(_oc, _oc)
            lines.append(f"  {_ocb} ({_oc}): {_oname} = {_or:.4f}%  (as of {_odate})")
        # Pre-compute key OIS differentials for the most-traded pairs
        _KEY_OIS_DIFFS = [
            ("USD","EUR","EUR/USD"), ("USD","GBP","GBP/USD"), ("USD","JPY","USD/JPY"),
            ("USD","AUD","AUD/USD"), ("USD","CAD","USD/CAD"), ("GBP","JPY","GBP/JPY"),
            ("EUR","JPY","EUR/JPY"), ("AUD","JPY","AUD/JPY"),
        ]
        lines.append("")
        lines.append("  Key OIS carry differentials (base vs quote, bp):")
        for _ob, _oq, _opair in _KEY_OIS_DIFFS:
            _rb = _ois_rates.get(_ob)
            _rq = _ois_rates.get(_oq)
            if _rb is None or _rq is None:
                continue
            _diff_bp = round((_rb - _rq) * 100)
            _carry   = _ob if _diff_bp > 0 else (_oq if _diff_bp < 0 else "neutral")
            _ds = f"+{_diff_bp}" if _diff_bp >= 0 else str(_diff_bp)
            lines.append(f"    {_opair}: {_OIS_NAMES.get(_ob,_ob)} {_rb:.2f}% vs {_OIS_NAMES.get(_oq,_oq)} {_rq:.2f}% → {_ds}bp (carry favors {_carry})")
        lines.append("  RULE: When framing carry, cite these OIS bp values — not the CB policy bp from Section 1.")
        lines.append("  RULE: If OIS differential and CB rate differential point in opposite directions, OIS is authoritative.")
        lines.append("")

    # 2. CFTC COT — Disaggregated TFF: Leveraged Funds (primary) + Asset Manager + Dealer
    # Source: CFTC financial_lof.htm (Options+Futures Combined where available,
    # Futures Only as fallback). Leveraged Funds = hedge funds / CTAs — best
    # short-term positioning signal. Asset Manager = pension/mutual funds — slower
    # structural signal. Dealer = market-makers hedging client flow — contrarian
    # at extremes (very net-short dealer = crowded client longs = squeeze risk).
    lines.append("=== CFTC Speculative Positioning (COT — Disaggregated TFF) ===")
    for ccy in CURRENCIES:
        d = load_json(SITE_DIR / "cot-data" / f"{ccy}.json")
        if d:
            # Leveraged Funds (primary speculative signal)
            net       = d.get("netPosition", None)
            long_pos  = d.get("longPositions", 0) or 0
            short_pos = d.get("shortPositions", 0) or 0
            total     = long_pos + short_pos
            long_pct  = f"{round(long_pos/total*100,1)}%" if total > 0 else "N/A"
            week_end  = d.get("weekEnding") or d.get("reportDate", "?")
            src_type  = d.get("sourceType", "unknown")
            src_label = "O+F" if src_type == "options_futures_combined" else "F"
            if net is not None:
                bias    = "NET LONG" if net > 0 else "NET SHORT" if net < 0 else "FLAT"
                net_str = f"{net:+,}"
            else:
                bias, net_str = "N/A", "N/A"

            # Asset Manager (institutional trend signal)
            am_net  = d.get("assetManagerNet")
            am_str  = f", AM={am_net:+,}" if am_net is not None else ""

            # Dealer (contrarian signal at extremes)
            dd_net  = d.get("dealerNet")
            dd_str  = f", DD={dd_net:+,}" if dd_net is not None else ""

            lines.append(
                f"  {ccy}: {bias}, lev_net={net_str}, long%={long_pct}"
                f"{am_str}{dd_str}  [{src_label}] (week ending {week_end})"
            )
    lines.append("")

    # 3. (economic-data/ block removed in v7.24.1)
    # economic-data/{CCY}.json contained stale macro fields (GDP nominal 671d old,
    # termsOfTrade 1037d old, EUR unemployment sourced from Spain not Eurozone).
    # The AI already receives all market-relevant data from intraday-data/quotes.json
    # (VIX, SPX, yields, FX pairs, MOVE, Gold, BTC, equities — updated every 15min),
    # rates/*.json (CB policy rates), cot-data/ (CFTC weekly), meetings-data/ (OIS bias),
    # and news-data/news.json (headlines). economic-data/ added no incremental signal value
    # and introduced incorrect/outdated context. Update-economic-data.yml workflow disabled.

    # 4. News headlines — correct field name: 'cur' (not 'currency' or 'tag')
    news_raw = load_json(SITE_DIR / "news-data" / "news.json", default={})
    articles = news_raw.get("articles", news_raw.get("items", [])) \
               if isinstance(news_raw, dict) else news_raw

    # Two-layer EM filter — prevents non-G8 CB decisions from entering the narrative:
    #
    # Layer 1 — cur tag: reject articles whose currency tag is outside G8.
    #   Catches correctly-tagged EM articles (Banxico→MXN, BCB→BRL, PBOC→CNY, etc.)
    #
    # Layer 2 — title keywords: reject articles whose title names a non-G8 CB.
    #   Catches publisher mis-tags (e.g. Banxico tagged cur=NZD — confirmed production
    #   failure 2026-05-07 that caused model to cite Banxico in a G8 narrative).
    _G8_CURS = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
    _EM_CB_KW = {
        "banxico", "banco de mexico", "bcb", "banco do brasil",
        "pboc", "people's bank of china", "rbi", "reserve bank of india",
        "sarb", "south african reserve", "norges bank", "riksbank",
        "nbp", "national bank of poland", "cbrt", "turkey rate",
        "banco central de chile", "banco central de colombia",
    }
    def _is_em_cb(article: dict) -> bool:
        title_lower = article.get("title", "").lower()
        return any(kw in title_lower for kw in _EM_CB_KW)

    high_impact = [
        a for a in articles
        if a.get("impact") == "high"
        and a.get("lang", "en") == "en"
        and a.get("cur", "USD") in _G8_CURS
        and not _is_em_cb(a)
    ][:8]

    if high_impact:
        # ── Python TIER-1 tagger ─────────────────────────────────────────────
        # Two complementary methods — union of both determines TIER-1 status.
        #
        # METHOD 1 — Keyword matching (single-article events)
        # Covers events that produce ONE high-impact article: BoJ intervention
        # warning, NFP surprise, emergency FOMC. These are irreducibly specific
        # and cannot be detected by frequency alone.
        #
        # Categories (aligned with NARRATIVE_SYSTEM Rule 1):
        #   A) Geopolitical / military: known conflict actors + generic violence verbs.
        #      Uses \w{0,3} suffix on root words to match inflected forms ('attacked',
        #      'strikes', 'escalating', 'invasions', etc.) — bare \b produces false
        #      negatives on past-tense/plural forms.
        #   B) FX intervention / MoF verbal warning.
        #   C) Commodity supply shock: structural disruption patterns (OPEC cut,
        #      pipeline/refinery attack, oil embargo, LNG disruption). Region-specific
        #      location names intentionally excluded — those are handled by Method 2
        #      (clustering) so the list stays maintenance-free.
        #   D) Macro Tier-1 surprise: NFP, CPI shock, emergency FOMC, GDP miss,
        #      bank failure, systemic contagion.
        #
        # Inspection scope: title + expand[:150]. Geopolitical/commodity context is
        # often in the expand body when the title is neutral/generic.
        #
        # METHOD 2 — Topic clustering (multi-article events, zero maintenance)
        # When ≥ 3 of the 8 high-impact articles share a dominant content token,
        # ALL articles in that cluster are tagged TIER-1. This detects any emerging
        # shock — geopolitical, commodity, macro — purely from convergent coverage,
        # without needing to enumerate locations, event types, or terminology in advance.
        # A new conflict in any region, with any name, will self-elevate if it drives
        # ≥ 3 headlines. Complements Method 1: where keywords cover single-article
        # high-confidence events, clustering covers novel multi-article shocks.
        import re as _re_tier
        from collections import defaultdict as _dd_tier

        _TIER1_PATTERNS = [
            # A — geopolitical / military (known actors + generic violence verbs)
            r'\b(escalat\w{0,3}|military|strikes?\w{0,1}|attack\w{0,3}|invasion\w{0,2}|'
            r'conflict\w{0,1}|missile\w{0,1}|troops|offensive|'
            r'middle\s+east|gaza|iran\w{0,3}|ukraine|russia\w{0,1}|'
            r'north\s+korea|geopolit\w{0,4})\b',
            # B — FX intervention / MoF verbal warning
            r'\b(intervention|finmin|finance\s+minister|decisive\s+action|speculation\s+warning|'
            r'verbal\s+warning|currency\s+manipulation)\b',
            r'\bwarn(s|ing|ed)?\b.{0,40}\b(jpy|yen|fx|forex|currency|aud|gbp|usd)\b',
            r'\b(jpy|yen|fx|forex|currency|aud|gbp|usd)\b.{0,40}\bwarn(s|ing|ed)?\b',
            # C — commodity supply shock (structural patterns only — no region names)
            r'\b(supply\s+disruption|output\s+cut|opec|pipeline.{0,10}attack\w{0,3}|'
            r'refinery.{0,10}attack\w{0,3}|oil\s+embargo|lng\s+disruption|crude\s+supply)\b',
            # D — macro Tier-1 surprise
            r'\b(non.?farm|nfp|cpi.{0,20}(surprise|shock|beat|miss)|fomc.{0,20}(surprise|emergency|'
            r'unexpected)|gdp.{0,20}(miss|shock)|emergency\s+rate|systemic\s+risk|financial\s+crisis|'
            r'bank\s+failure|contagion)\b',
        ]

        def _is_tier1_keyword(title: str, expand: str = "") -> bool:
            # Inspect title + first 150 chars of expand so context buried in the
            # body (e.g. "Iran war" in an expand of a generic-titled article) is captured.
            combined = (title + " " + expand[:150]).lower()
            return any(_re_tier.search(p, combined) for p in _TIER1_PATTERNS)

        # ── Method 2: topic clustering ────────────────────────────────────────
        # Tokenise each article (title + expand[:150]) into meaningful words (≥4 chars,
        # not a stopword). Build an inverted index: token → set of article indices.
        # Tokens present in ≥ CLUSTER_THRESHOLD distinct articles are "dominant".
        # Any article containing a dominant token is elevated to TIER-1.
        #
        # Window: the same high_impact[:8] pool sent to the LLM — clustering over
        # this window means threshold=2 (≥25% convergence) reliably detects a dominant
        # topic without elevating generic weekly terms (policy, rate, outlook, etc.),
        # which are filtered by the enriched stopword list below.
        CLUSTER_THRESHOLD = 2  # ≥2 of the 8 LLM articles = convergent coverage
        _CLUSTER_STOPWORDS = {
            # Function words
            'this','that','with','from','have','will','been','they','them','were',
            'said','says','show','more','most','also','when','than','then','into',
            'their','there','about','would','could','should','after','before',
            # Generic finance/calendar terms present in virtually any news week
            'market','markets','week','weekly','focus','data','news','report',
            'rate','rates','central','bank','price','prices','trade','year',
            'global','world','economic','economy','dollar','euro','pound','yen',
            'ahead','next','look','outlook','preview','recap','since','below',
            'index','interest','policy','growth','lower','higher','holds','hold',
            'expected','forecasts','record','quarter','month','april','march',
            'business','consumer','confidence','sentiment','core','inflation',
            'banks','decisions','talks','hike','japan','china','risk','risks',
            'still','first','last','back','amid','time','over','some','even',
        }

        def _tokenize_article(article: dict) -> set:
            text = (
                article.get("title", "") + " " +
                (article.get("expand", "") or "")[:150]
            ).lower()
            return {
                t for t in _re_tier.findall(r'\b[a-z]{4,}\b', text)
                if t not in _CLUSTER_STOPWORDS
            }

        _token_to_indices: dict = _dd_tier(set)
        _article_tokens: list  = []
        for _ci, _art in enumerate(high_impact):
            _toks = _tokenize_article(_art)
            _article_tokens.append(_toks)
            for _tok in _toks:
                _token_to_indices[_tok].add(_ci)

        _dominant_tokens = {
            tok for tok, idxs in _token_to_indices.items()
            if len(idxs) >= CLUSTER_THRESHOLD
        }
        _cluster_tier1_indices = {
            i for i, toks in enumerate(_article_tokens)
            if toks & _dominant_tokens
        }
        if _dominant_tokens:
            print(f"  INFO Cluster TIER-1: dominant tokens = "
                  f"{sorted(_dominant_tokens)[:12]} "
                  f"({len(_cluster_tier1_indices)} articles elevated)")

        # ── Union of both methods ─────────────────────────────────────────────
        tier1_titles: set = set()
        for _i, _h in enumerate(high_impact):
            _t  = _h.get("title", "")
            _ex = _h.get("expand", "") or ""
            if _is_tier1_keyword(_t, _ex) or _i in _cluster_tier1_indices:
                tier1_titles.add(_t)

        tier1_headlines = [h for h in high_impact if h.get("title", "") in tier1_titles]

        lines.append("=== Latest FX Headlines (high-impact) ===")
        if tier1_headlines:
            lines.append(
                f"  ⚠️  TIER-1 EVENT DETECTED ({len(tier1_headlines)} headline(s)) — "
                f"NARRATIVE MUST OPEN WITH THIS. See NARRATIVE_SYSTEM Rule 1 (ABSOLUTE PRIORITY)."
            )
        for item in high_impact:
            title  = (item.get("title") or "").strip()
            ccy    = item.get("cur") or item.get("currency") or ""   # 'cur' is the correct field
            time   = item.get("time", "")
            expand = (item.get("expand") or "").replace("\n", " ").strip()[:100]
            if title:
                tag      = f"[{ccy}] " if ccy else ""
                time_str = f"{time} " if time else ""
                tier_tag = "  [⚠️ TIER-1 — MUST LEAD NARRATIVE] " if title in tier1_titles else "  "
                lines.append(f"{tier_tag}• {time_str}{tag}{title}")
                if expand and len(expand) > 30:
                    lines.append(f"    → {expand}{'…' if len(expand)==100 else ''}")
        lines.append("")

    # 5. Intraday quotes — VIX, SPX, Gold, WTI, DXY, yields, BTC, FX pairs (yfinance via GitHub Action)
    intraday = load_json(SITE_DIR / "intraday-data" / "quotes.json")
    if intraday and intraday.get("quotes"):
        q = intraday["quotes"]
        updated = intraday.get("updated", "")
        lines.append(f"=== Intraday Market Quotes (yfinance, updated: {updated}) ===")

        # Load closed symbols — instruments with no live intraday data today (bank holidays).
        # For closed symbols the pct is 0.0 (stale close) — the LLM must NOT report it as
        # a real intraday move. We replace "1D:+0.00%" with "market closed — no intraday data".
        _closed_syms = get_closed_symbols()
        if _closed_syms:
            lines.append(f"  NOTE: The following symbols have NO live intraday data today (bank holiday / market closed): "
                         f"{', '.join(sorted(_closed_syms))}. "
                         f"Do NOT report a % change for these — omit or note 'market closed'.")
            lines.append("")

        # Cross-asset
        cross_asset = []
        # vix_label is pre-computed by extract_tokens_from_intraday() and injected via context
        # so the LLM uses the rule-based label without needing to classify VIX bands itself.
        _tokens_for_ctx = extract_tokens_from_intraday()
        _vix_label_ctx  = _tokens_for_ctx.get("vix_label", "")

        for key, label, fmt_fn in [
            ("vix",   "VIX",          lambda v: f"{v:.1f} — {_vix_label_ctx}" if _vix_label_ctx else f"{v:.1f}"),
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
                if key in _closed_syms:
                    # Market closed — show last known price but no pct change
                    cross_asset.append(f"  {label}: {fmt_fn(d['close'])}  [market closed — last known price, no intraday move]")
                else:
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
                if key in _closed_syms:
                    yield_live.append(f"  {label}: {d['close']:.3f}%  [market closed — last known level, no intraday move]")
                else:
                    pct = d.get("pct", 0)
                    sign = "+" if pct >= 0 else ""
                    yield_live.append(f"  {label}: {d['close']:.3f}%  1D:{sign}{pct:.2f}%")
        if yield_live:
            lines.append("  -- Intraday Yields (yfinance, more current than extended-data) --")
            lines.extend(yield_live)
            lines.append("")

        # Yield curve regime classification -- Bloomberg standard
        # Computes 2s10s and 3m10s spreads and classifies curve regime.
        # Four canonical regimes per fixed-income convention:
        #   Bull steepener  : both yields falling, long-end faster -> recession pricing, risk-off
        #   Bull flattener  : both yields falling, short-end faster -> late-cycle, rate-cut pricing
        #   Bear steepener  : both yields rising, long-end faster  -> inflation/supply concern
        #   Bear flattener  : both yields rising, short-end faster -> Fed tightening, hiking cycle
        # Inversion (2s10s < 0 or 3m10s < 0) is flagged explicitly -- Bloomberg highlights it.
        us2y_d  = q.get("us2y");  us2y_cl  = us2y_d.get("close")  if us2y_d  else None
        us3m_d  = q.get("us3m");  us3m_cl  = us3m_d.get("close")  if us3m_d  else None
        us10y_d = q.get("us10y"); us10y_cl = us10y_d.get("close") if us10y_d else None
        us2y_p  = us2y_d.get("pct")  if us2y_d  else None
        us3m_p  = us3m_d.get("pct")  if us3m_d  else None
        us10y_p = us10y_d.get("pct") if us10y_d else None

        if us2y_cl is not None and us10y_cl is not None:
            spread_2s10s = us10y_cl - us2y_cl
            inv_2s10s    = spread_2s10s < 0
            spread_sign  = "+" if spread_2s10s >= 0 else ""

            # Regime classification from intraday direction (pct proxies daily move direction)
            curve_regime_2s10 = "unknown"
            if us2y_p is not None and us10y_p is not None:
                if us10y_p < 0 and us2y_p < 0:
                    curve_regime_2s10 = "bull steepener" if us10y_p > us2y_p else "bull flattener"
                elif us10y_p > 0 and us2y_p > 0:
                    curve_regime_2s10 = "bear steepener" if us10y_p > us2y_p else "bear flattener"
                elif us10y_p > 0 and us2y_p <= 0:
                    curve_regime_2s10 = "bear steepener"  # long-end rising, short-end flat/falling
                elif us10y_p < 0 and us2y_p >= 0:
                    curve_regime_2s10 = "bull flattener"  # long-end falling, short-end flat/rising

            inv_note_2s10 = " [INVERTED]" if inv_2s10s else ""
            lines.append("  -- Yield Curve Regime (Bloomberg standard) --")
            lines.append(
                f"  2s10s spread: {spread_sign}{spread_2s10s*100:.1f}bp{inv_note_2s10}"
                f"  | Regime: {curve_regime_2s10.upper()}"
                + (f"  | INVERSION: recession-risk signal active" if inv_2s10s else "")
            )

        if us3m_cl is not None and us10y_cl is not None:
            spread_3m10 = us10y_cl - us3m_cl
            inv_3m10    = spread_3m10 < 0
            spread_sign = "+" if spread_3m10 >= 0 else ""
            inv_note_3m = " [INVERTED]" if inv_3m10 else ""
            lines.append(
                f"  3m10s spread: {spread_sign}{spread_3m10*100:.1f}bp{inv_note_3m}"
                + (f"  | INVERSION: historically reliable US recession predictor" if inv_3m10 else "")
            )
        lines.append("")
        lines.append(
            "  NOTE: Use yield curve regime to frame CB divergence signals and cross-asset"
            " risk narrative. Bull steepener = recession pricing / risk-off. Bear flattener"
            " = tightening cycle dominant. Inversion = elevated recession risk."
        )
        lines.append("")

        # FX pairs intraday (if available — requires workflow v2.2+)
        # G8 majors + key crosses: GBP/JPY, EUR/JPY, AUD/JPY carry the largest rate
        # differentials in the cross space and are the most institutionally traded crosses.
        # AUD/CHF, NZD/CHF, CHF/JPY, CAD/JPY included as secondary carry/safe-haven signals.
        fx_intraday = []
        FX_PAIRS_DISPLAY = [
            ("eurusd","EUR/USD",5), ("gbpusd","GBP/USD",5), ("usdjpy","USD/JPY",3),
            ("audusd","AUD/USD",5), ("usdchf","USD/CHF",5), ("usdcad","USD/CAD",5),
            ("nzdusd","NZD/USD",5),
            # JPY crosses — critical for carry trade / risk-off analysis
            ("gbpjpy","GBP/JPY",3), ("eurjpy","EUR/JPY",3), ("audjpy","AUD/JPY",3),
            ("cadjpy","CAD/JPY",3), ("chfjpy","CHF/JPY",3),
            # CHF crosses — safe-haven demand proxy
            ("eurchf","EUR/CHF",5), ("gbpchf","GBP/CHF",5), ("audchf","AUD/CHF",5),
            ("nzdchf","NZD/CHF",5),
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

        # CB rate differentials for key crosses — institutional carry context
        # Computed from rates/*.json (same source as CB Rates table).
        # Format mirrors Bloomberg: BASE rate vs QUOTE rate -> spread in bp -> carry direction.
        # This block gives the LLM explicit carry math so it does not need to infer from
        # individual CB rates -- reduces hallucination of inverted carry logic (Rule 4 SIGNALS_SYSTEM).
        cross_diffs = []
        CROSS_CARRY = [
            # USD majors — listed first so the LLM sees the exact bp for EUR/USD, GBP/USD etc.
            # and never has to infer or recall from training memory (root cause of "175bp" errors).
            ("USD", "EUR", "USD/EUR (EUR/USD carry)"),
            ("USD", "GBP", "USD/GBP (GBP/USD carry)"),
            ("USD", "AUD", "USD/AUD (AUD/USD carry)"),
            ("USD", "NZD", "USD/NZD (NZD/USD carry)"),
            ("USD", "CAD", "USD/CAD carry"),
            ("USD", "CHF", "USD/CHF carry"),
            ("USD", "JPY", "USD/JPY carry"),
            # JPY crosses
            ("GBP", "JPY", "GBP/JPY"),
            ("EUR", "JPY", "EUR/JPY"),
            ("AUD", "JPY", "AUD/JPY"),
            ("CAD", "JPY", "CAD/JPY"),
            ("CHF", "JPY", "CHF/JPY"),
            # CHF crosses
            ("EUR", "CHF", "EUR/CHF"),
            ("GBP", "CHF", "GBP/CHF"),
            ("AUD", "CHF", "AUD/CHF"),
            ("NZD", "CHF", "NZD/CHF"),
        ]
        # Build rate lookup from rates/*.json (already loaded per-ccy above in section 1)
        _ccy_rates: dict = {}
        for _ccy in CURRENCIES:
            _rd = load_json(SITE_DIR / "rates" / f"{_ccy}.json")
            if _rd and _rd.get("observations"):
                try:
                    _ccy_rates[_ccy] = float(_rd["observations"][0]["value"])
                except (ValueError, KeyError):
                    pass

        for base, quote, pair_label in CROSS_CARRY:
            rb = _ccy_rates.get(base)
            rq = _ccy_rates.get(quote)
            if rb is None or rq is None:
                continue
            diff_bp = round((rb - rq) * 100)
            carry_side = base if diff_bp > 0 else (quote if diff_bp < 0 else "neutral")
            carry_note = f"carry favors {carry_side}" if diff_bp != 0 else "carry neutral"
            cross_diffs.append(
                f"  {pair_label}: {CB_LABELS.get(base,base)} {rb}% vs"
                f" {CB_LABELS.get(quote,quote)} {rq}% -> {diff_bp:+d}bp ({carry_note})"
            )
        if cross_diffs:
            lines.append("  -- Cross CB Rate Differentials (carry context) --")
            lines.extend(cross_diffs)
            lines.append("  NOTE: USD major pairs listed first — use these exact bp values for EUR/USD, GBP/USD, AUD/USD, NZD/USD carry claims."
                         " Never use training-memory rates to compute carry spreads — always use values above."
                         " Positive bp = base currency carry advantage. Risk-off unwinds compress JPY/CHF cross spreads rapidly.")
            lines.append("")

        # FX Options Volatility — Implied Vol (IV) vs Historical Vol (HV30) + Vol Risk Premium
        # Sources:
        #   fx_etf_iv → CBOE FX VIX indices: ^EUVIX (EUR/USD), ^BPVIX (GBP/USD), ^JYVIX (USD/JPY),
        #               ^AUDVIX (AUD/USD), ^USDVIX (USD/CHF) — from intraday-data/quotes.json
        #   hv30      → 30-day realized volatility (annualised) computed from OHLC close history
        #               by fetch_intraday_quotes.py — same quotes.json
        # Vol Risk Premium (VRP) = IV − HV30:
        #   VRP > 0  → options priced above realized vol → market paying premium for protection
        #              (consistent with elevated hedging demand, potential for vol mean-reversion)
        #   VRP ≈ 0  → fair value — options priced at realized vol
        #   VRP < 0  → options below realized vol → unusual; can indicate complacency or thin supply
        # Industry convention (Goldman Sachs FX Options Desk, BAML FX Vol Monitor):
        #   Cite IV alongside HV30 and VRP when characterizing volatility regime or option-implied
        #   directional bias. Never state IV without the HV30 context.
        _hv30_d   = intraday.get("hv30", {})   if intraday else {}
        _etf_iv_d = intraday.get("fx_etf_iv", {}) if intraday else {}
        _VOL_PAIRS = [
            ("eurusd", "EUR/USD"), ("gbpusd", "GBP/USD"), ("usdjpy", "USD/JPY"),
            ("audusd", "AUD/USD"), ("usdchf", "USD/CHF"),
        ]
        _vol_lines = []
        _vrp_extremes = []
        for _vk, _vlabel in _VOL_PAIRS:
            _hv   = _hv30_d.get(_vk)
            _iv_d = _etf_iv_d.get(_vk, {})
            _iv   = _iv_d.get("iv") if isinstance(_iv_d, dict) else None
            _src  = _iv_d.get("source", "CBOE") if isinstance(_iv_d, dict) else "CBOE"
            if _hv is None and _iv is None:
                continue
            parts = []
            if _iv is not None:
                parts.append(f"IV={_iv:.1f}% ({_src})")
            if _hv is not None:
                parts.append(f"HV30={_hv:.1f}%")
            if _iv is not None and _hv is not None:
                _vrp = round(_iv - _hv, 2)
                _vrp_s = f"+{_vrp:.1f}" if _vrp >= 0 else f"{_vrp:.1f}"
                _vrp_note = ""
                if _vrp >= 2.0:
                    _vrp_note = " — ELEVATED premium: strong hedging demand over realized vol"
                    _vrp_extremes.append((_vlabel, _iv, _hv, _vrp))
                elif _vrp >= 0.8:
                    _vrp_note = " — moderate premium over realized"
                elif _vrp <= -0.5:
                    _vrp_note = " — IV below realized: unusual complacency / thin options supply"
                    _vrp_extremes.append((_vlabel, _iv, _hv, _vrp))
                parts.append(f"VRP={_vrp_s}%{_vrp_note}")
            _vol_lines.append(f"  {_vlabel}: {' | '.join(parts)}")
        if _vol_lines:
            lines.append("  -- FX Options Volatility: IV vs HV30 + Vol Risk Premium --")
            lines.extend(_vol_lines)
            if _vrp_extremes:
                lines.append("")
                lines.append("  VRP EXTREMES (|VRP| notable — cite in volatility regime framing):")
                for _ve_l, _ve_iv, _ve_hv, _ve_vrp in sorted(_vrp_extremes, key=lambda x: -abs(x[3])):
                    _ve_s = f"+{_ve_vrp:.1f}" if _ve_vrp >= 0 else f"{_ve_vrp:.1f}"
                    lines.append(f"    {_ve_l}: IV {_ve_iv:.1f}% vs HV30 {_ve_hv:.1f}% → VRP {_ve_s}%")
            lines.append("  NOTE: IV reflects forward-looking market fear/demand. HV30 is realized vol from last 30 trading days.")
            lines.append("  RULE: Always cite IV and HV30 together when discussing vol regime — never cite IV alone. VRP > 2% = notable.")
            lines.append("")

    # 6. Myfxbook retail sentiment (community outlook, updated every 15min)
    # Coverage: 24 pairs — all G8 majors + key crosses (GBP/JPY, EUR/JPY, AUD/JPY, EUR/CHF,
    # GBP/CHF, EUR/GBP, AUD/JPY, AUD/CHF, etc.). NZD/CHF is NOT published by Myfxbook
    # Community Outlook and will always be absent. This is a source limitation, not a fetch
    # error — do not attempt to generate a retail signal for NZD/CHF.
    mfx = load_json(SITE_DIR / "sentiment-data" / "myfxbook.json")
    if mfx and mfx.get("pairs"):
        updated = mfx.get("updated", "")
        pairs_covered = [p["sym"] for p in mfx["pairs"]]
        # Identify which crosses are available — log for LLM so it knows which crosses
        # have retail data available for contrarian cross-checks
        majors = {"EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","USD/CHF","NZD/USD"}
        crosses_in_retail = [s for s in pairs_covered if s not in majors]
        lines.append(f"=== Retail FX Sentiment — Myfxbook Community Outlook (updated: {updated}) ===")
        lines.append("  (% of retail traders long vs short — contrarian signal: extremes often precede reversals)")
        if crosses_in_retail:
            lines.append(f"  Cross pairs with retail data: {', '.join(crosses_in_retail)}")
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
            if dom >= 85:
                # Threshold aligned with SIGNALS_SYSTEM rule: only >=85% qualifies as actionable extreme
                note = f"  EXTREME {side} — contrarian reversal risk (retail positioning extreme)"
                extremes.append((sym, side, dom, lng, sht))
            elif dom >= 70:
                note = f"  Strong {side} bias"
            lines.append(f"  {sym}: {bias}{note}")

        if extremes:
            lines.append("")
            lines.append("  RETAIL EXTREMES (contrarian signal — only use when diverging from COT direction):")
            for sym, side, dom, lng, sht in sorted(extremes, key=lambda x: -x[2]):
                lines.append(f"    {sym}: {dom}% {side} — retail positioning extreme; cross-check COT before signaling")
        lines.append("")

    # 6b. FX Options — 25-delta Risk Reversals (Saxo Bank · 1M tenor · indicative mid)
    # Source: rr-data/rr.json (primary) with rr2.json as fallback.
    # 25d RR = implied vol of 25-delta call − implied vol of 25-delta put (same tenor).
    # Interpretation:
    #   Negative RR → put vol > call vol → market paying MORE for downside protection on the BASE.
    #     e.g. EURUSD RR −0.50 → bias for EUR weakness / USD strength
    #   Positive RR → call vol > put vol → market paying MORE for upside on the BASE.
    #     e.g. USDJPY RR +1.20 → bias for USD strength / JPY weakness
    # This is the options market's directional bias, distinct from COT (futures/positioning).
    # A COT-RR divergence is a high-quality signal: specs positioned one way, options market another.
    # Industry convention (Morgan Stanley, JPM FX research): RR cited alongside COT to confirm or
    # contradict the directional bias. Always compare against the COT net for the same pair.
    _rr_raw  = load_json(SITE_DIR / "rr-data" / "rr.json")
    _rr2_raw = load_json(SITE_DIR / "rr-data" / "rr2.json")
    _rr_data = (_rr_raw if _rr_raw and _rr_raw.get("pairs") else None) or \
               (_rr2_raw if _rr2_raw and _rr2_raw.get("pairs") else None)
    if _rr_data and _rr_data.get("pairs"):
        _rr_fetched = _rr_data.get("fetched", "")
        _rr_source  = _rr_data.get("source", "Saxo Bank · FX Options · 25d RR · 1M tenor · indicative mid")
        lines.append(f"=== FX Options — 25-Delta Risk Reversals ({_rr_fetched}) ===")
        lines.append(f"  Source: {_rr_source}")
        lines.append("  (Negative = put vol > call vol = options market biased for BASE weakness)")
        lines.append("  (Positive = call vol > put vol = options market biased for BASE strength)")
        lines.append("")
        _rr_extremes = []
        for _pair, _vals in _rr_data["pairs"].items():
            _rr25 = _vals.get("rr25d")
            if _rr25 is None:
                continue
            _sign = "+" if _rr25 >= 0 else ""
            # Classify magnitude
            _abs = abs(_rr25)
            if _abs >= 1.5:
                _label = "STRONG put skew" if _rr25 < 0 else "STRONG call skew"
                _note  = f"  *** {_label} — elevated hedging demand for {'downside' if _rr25 < 0 else 'upside'} protection"
                _rr_extremes.append((_pair, _rr25, _label))
            elif _abs >= 0.75:
                _label = "Moderate put skew" if _rr25 < 0 else "Moderate call skew"
                _note  = f"  ** {_label}"
            elif _abs >= 0.30:
                _label = "Mild put skew" if _rr25 < 0 else "Mild call skew"
                _note  = f"  * {_label}"
            else:
                _note = "  (near neutral — no clear directional options bias)"
            lines.append(f"  {_pair}: {_sign}{_rr25:.2f}  {_note}")
        if _rr_extremes:
            lines.append("")
            lines.append("  RR EXTREMES (|RR| ≥ 1.5 — strong options skew, cross-check against COT net):")
            for _ep, _ev, _el in sorted(_rr_extremes, key=lambda x: -abs(x[1])):
                _es = "+" if _ev >= 0 else ""
                lines.append(f"    {_ep}: RR {_es}{_ev:.2f} — {_el}")
        lines.append("")

    # 7. Cross-asset correlations — 60d Pearson vs 252d historical norm (z_score)
    # Source: intraday-data/quotes.json → correlations[] computed by fetch_intraday_quotes.py
    # Only include pairs with |z_score| > 1.0 (meaningful deviation from historical norm).
    # These are regime-informative signals: a broken correlation often precedes trend shifts.
    if intraday and intraday.get("correlations"):
        notable = [
            c for c in intraday["correlations"]
            if c.get("z_score") is not None and abs(c["z_score"]) > 1.0
        ]
        if notable:
            lines.append("=== Cross-Asset Correlation Breaks (z > 1σ from 252d norm) ===")
            lines.append("  (Correlations deviating from historical norm — potential regime signals)")
            for c in sorted(notable, key=lambda x: -abs(x.get("z_score", 0))):
                a      = c.get("a", "?")
                b      = c.get("b", "?")
                corr   = c.get("corr")
                norm   = c.get("norm")
                z      = c.get("z_score")
                corr_s = f"{corr:+.3f}" if corr is not None else "N/A"
                norm_s = f"{norm:+.3f}" if norm is not None else "N/A"
                z_s    = f"{z:+.2f}σ"  if z    is not None else "N/A"
                # Contextual annotation based on pair + direction
                annotation = ""
                if a == "EUR/USD" and b == "DXY":
                    annotation = " [normal: strong negative; break = EUR/DXY decoupling]"
                elif a == "DXY" and b == "SPX":
                    annotation = " [normal: negative; positive = USD funding stress / risk-off USD bid]"
                elif a == "Gold" and b == "DXY":
                    annotation = " [normal: negative; positive = safe-haven model broken or real inflation bid]"
                elif a == "USD/JPY" and b == "VIX":
                    annotation = " [normal: negative; positive = JPY no longer acting as safe haven]"
                lines.append(
                    f"  {a} vs {b}: corr={corr_s}  norm={norm_s}  z={z_s}{annotation}"
                )
            lines.append("")
        else:
            # All correlations within norm — worth noting for the LLM
            lines.append("=== Cross-Asset Correlations ===")
            lines.append("  All major pair correlations within 1σ of 252d historical norm — no regime breaks detected.")
            lines.append("")

    # 8. G8 Currency Strength Scorecard (1W) — computed from pct1w across all available pairs
    # Method: for each pair X/Y, pct1w > 0 means X gained vs Y this week.
    #   → base currency gets +pct1w, quote currency gets -pct1w for that pair.
    # Each currency's score = average of all its directional contributions across G8 pairs.
    # This gives the AI an explicit, pre-ranked view of which currencies are leading/lagging
    # the week — the same "G8 currency performance" context that Bloomberg FXIP provides.
    # Without this block, the AI sees pct1w only for USD-quoted majors and cannot infer
    # which currency is broadly strongest/weakest across the full G8 cross matrix.
    SCORECARD_PAIRS = [
        # (quotes.json key, base_ccy, quote_ccy)
        ("eurusd",  "EUR", "USD"), ("gbpusd",  "GBP", "USD"), ("usdjpy",  "USD", "JPY"),
        ("audusd",  "AUD", "USD"), ("usdcad",  "USD", "CAD"), ("usdchf",  "USD", "CHF"),
        ("nzdusd",  "NZD", "USD"),
        ("eurjpy",  "EUR", "JPY"), ("eurchf",  "EUR", "CHF"), ("eurgbp",  "EUR", "GBP"),
        ("euraud",  "EUR", "AUD"), ("eurcad",  "EUR", "CAD"), ("eurnzd",  "EUR", "NZD"),
        ("gbpjpy",  "GBP", "JPY"), ("gbpchf",  "GBP", "CHF"),
        ("audjpy",  "AUD", "JPY"), ("audchf",  "AUD", "CHF"),
        ("cadjpy",  "CAD", "JPY"), ("chfjpy",  "CHF", "JPY"),
        ("nzdjpy",  "NZD", "JPY"), ("nzdchf",  "NZD", "CHF"),
    ]
    from collections import defaultdict as _dd
    _scores: dict = _dd(list)
    _iq_sc = (intraday_raw or {}).get("quotes", {})
    for _key, _base, _quote in SCORECARD_PAIRS:
        _d = _iq_sc.get(_key)
        if _d and _d.get("pct1w") is not None:
            _p = float(_d["pct1w"])
            _scores[_base].append(_p)
            _scores[_quote].append(-_p)

    if _scores:
        _ranked = sorted(
            [(_ccy, sum(_v) / len(_v), len(_v)) for _ccy, _v in _scores.items()],
            key=lambda x: -x[1]
        )
        lines.append("=== G8 Currency Strength Scorecard — 1W (avg % vs G8 peers) ===")
        lines.append("  (Computed from pct1w across all available G8 cross pairs. Positive = outperforming peers this week.)")
        lines.append("  Rank · CCY  · 1W avg%  · pairs")
        for _rank, (_ccy, _avg, _n) in enumerate(_ranked, 1):
            _sign  = "+" if _avg >= 0 else ""
            _arrow = "▲" if _avg >  0.10 else ("▼" if _avg < -0.10 else "→")
            lines.append(f"  {_rank}. {_ccy}  {_arrow} {_sign}{_avg:.3f}%  (n={_n} pairs)")

        # Identify leader and laggard for explicit signal framing
        _leader  = _ranked[0]
        _laggard = _ranked[-1]
        _spread  = round(_leader[1] - _laggard[1], 3)

        # Expose G8 thresholds for Python-side signal suppression in generate_signals()
        _G8_SCORECARD.update({
            "leader_ccy":  _leader[0],
            "leader_avg":  _leader[1],
            "leader_n":    _leader[2],
            "laggard_ccy": _laggard[0],
            "laggard_avg": _laggard[1],
            "laggard_n":   _laggard[2],
            "spread":      _spread,
        })
        lines.append("")
        lines.append(
            f"  LEADER:  {_leader[0]} {'+' if _leader[1]>=0 else ''}{_leader[1]:.3f}% (n={_leader[2]} pairs) — strongest G8 currency this week"
        )
        lines.append(
            f"  LAGGARD: {_laggard[0]} {'+' if _laggard[1]>=0 else ''}{_laggard[1]:.3f}% (n={_laggard[2]} pairs) — weakest G8 currency this week"
        )
        lines.append(
            f"  SPREAD:  {_spread:.3f}% leader-to-laggard gap  ({_leader[0]} vs {_laggard[0]})"
        )
        lines.append("")

        # ── Pre-compute trigger evaluation so the LLM uses the authoritative
        # leader/laggard pair and spread instead of re-deriving it from the ranked
        # table (which caused the EUR-vs-GBP false positive: the model compared two
        # non-extreme currencies instead of the actual leader vs laggard).
        # ─────────────────────────────────────────────────────────────────────────
        _leader_ok   = _leader[1] >= 0.50
        _laggard_ok  = _laggard[1] <= -0.50
        _bilateral_ok = _spread >= 0.80
        _critical_ok  = _spread >= 1.20 or _leader[1] >= 0.80 or _laggard[1] <= -0.80
        _sign_l = "+" if _leader[1] >= 0 else ""
        _sign_g = "+" if _laggard[1] >= 0 else ""

        if _critical_ok:
            _trigger_label = "CRITICAL — strong theme (spread ≥ 1.20% or avg ≥ ±0.80%)"
        elif _bilateral_ok:
            _trigger_label = "WARNING — bilateral G8 theme (spread ≥ 0.80%)"
        elif _leader_ok or _laggard_ok:
            _ccy_trig = _leader[0] if _leader_ok else _laggard[0]
            _trigger_label = f"WARNING — individual outperformance ({_ccy_trig} avg {'≥ +0.50%' if _leader_ok else '≤ −0.50%'})"
        else:
            _trigger_label = "SUPPRESS — all thresholds below trigger levels"

        lines.append(f"  TRIGGER EVALUATION (pre-computed — use these values, do NOT re-derive from the table above):")
        lines.append(f"    Leader avg {_sign_l}{_leader[1]:.3f}% ≥ +0.50%? {'YES' if _leader_ok else 'NO'}")
        lines.append(f"    Laggard avg {_sign_g}{_laggard[1]:.3f}% ≤ −0.50%? {'YES' if _laggard_ok else 'NO'}")
        lines.append(f"    Spread {_spread:.3f}% ≥ 0.80%? {'YES' if _bilateral_ok else 'NO'}")
        lines.append(f"    → VERDICT: {_trigger_label}")
        lines.append("")

        if _bilateral_ok or _leader_ok or _laggard_ok:
            # Explicit framing instruction for the signal title and evidence chips
            if _bilateral_ok:
                _title_fmt = f"{_leader[0]}/{_laggard[0]} — G8 Bilateral Theme + [driver]"
                _ev_fmt = (
                    f"Evidence chips MUST be: "
                    f"'{_leader[0]} 1W avg: {_sign_l}{_leader[1]:.3f}% (n={_leader[2]})', "
                    f"'{_laggard[0]} 1W avg: {_sign_g}{_laggard[1]:.3f}% (n={_laggard[2]})', "
                    f"'G8 spread: {_spread:.3f}%'"
                )
            else:
                _ccy_ind = _leader[0] if _leader_ok else _laggard[0]
                _avg_ind = _leader[1] if _leader_ok else _laggard[1]
                _n_ind   = _leader[2] if _leader_ok else _laggard[2]
                _sign_i  = "+" if _avg_ind >= 0 else ""
                _title_fmt = f"{_ccy_ind} — G8 Outperformance + [driver]"
                _ev_fmt = f"First evidence chip MUST be: '{_ccy_ind} 1W avg: {_sign_i}{_avg_ind:.3f}% (n={_n_ind})'"
            lines.append(f"  SIGNAL FORMAT — use exactly:")
            lines.append(f"    Title: \"{_title_fmt}\"")
            lines.append(f"    {_ev_fmt}")
            lines.append(f"  CRITICAL: Do NOT compare other pairs from the ranked table above. The signal")
            lines.append(f"  MUST use {_leader[0]} (LEADER) vs {_laggard[0]} (LAGGARD) as the canonical pair.")
        else:
            lines.append("  NO G8 SIGNAL — all three thresholds below trigger levels. Do NOT generate a G8 signal.")
            lines.append("  Scorecard data may appear as supporting context inside CB or carry signals only.")
        lines.append("")

        lines.append(
            "  SIGNAL RULES for this block (see SIGNALS_SYSTEM Rule 13 for full detail):"
        )
        lines.append(
            "  • LEADER avg ≥ +0.50% OR LAGGARD avg ≤ −0.50%: individual outperformance signal (WARNING)."
        )
        lines.append(
            f"  • SPREAD ≥ 0.80%: bilateral G8 theme signal (WARNING) — even if neither end crosses ±0.50% alone."
        )
        lines.append(
            f"  • SPREAD ≥ 1.20% or avg ≥ +0.80%: strong theme (CRITICAL) — anchors the narrative lead."
        )
        lines.append(
            "  • If LEADER has carry disadvantage (lower rate than currencies it beats): flag divergence with bp."
        )
        lines.append(
            "  • If LEADER or LAGGARD aligns with a TIER-1 headline: fuse into one signal — do NOT split."
        )
        lines.append(
            "  • All scorecard signals must cite: avg %, number of pairs (n=X), and SPREAD if bilateral trigger."
        )
        lines.append("")

    # 9. Economic Surprise Index — CESI-style rolling beat/miss by G8 currency
    # Source: calendar-data/calendar.json (high+medium impact events with actual+forecast).
    # Method: for each event where actual ≠ forecast, tally beat (+1) vs miss (-1) per currency.
    # Net score = beats - misses over the rolling window. Positive = data consistently
    # surprising to the upside (Citi CESI equivalent: positive CESI = currency tailwind).
    # Also surfaces per-series z-scores (gap 2): surpriseStats has mean+std for 80 series.
    # When n≥3, compute z = (latest_actual - mean) / std for the most recent release.
    # Bloomberg BEEI / Citi CESI conventions: positive score = outperformance vs expectations.
    _cal_raw = load_json(SITE_DIR / "calendar-data" / "calendar.json")
    if _cal_raw and _cal_raw.get("events"):
        from collections import defaultdict as _dd_cesi
        _G8 = set(CURRENCIES)
        _HI = {"high", "medium"}
        _beats: dict = _dd_cesi(int)
        _misses: dict = _dd_cesi(int)
        _total: dict = _dd_cesi(int)
        _last_actual: dict = {}  # ccy -> list of (event_title, actual_str, forecast_str, beat)

        def _parse_num(s: str) -> float | None:
            """Parse economic data strings like '3.4%', '120K', '-0.5'."""
            if not s:
                return None
            try:
                return float(str(s).replace("%", "").replace("K", "e3")
                             .replace("M", "e6").replace("B", "e9")
                             .replace(",", "").strip())
            except (ValueError, TypeError):
                return None

        for _ev in _cal_raw["events"]:
            _ccy = _ev.get("currency", "")
            if _ccy not in _G8:
                continue
            if _ev.get("impact") not in _HI:
                continue
            _a = _parse_num(_ev.get("actual"))
            _f = _parse_num(_ev.get("forecast"))
            if _a is None or _f is None:
                continue
            _total[_ccy] += 1
            if _a > _f:
                _beats[_ccy] += 1
                _last_actual.setdefault(_ccy, []).append(
                    (_ev.get("event", ""), _ev.get("actual", ""), _ev.get("forecast", ""), True)
                )
            elif _a < _f:
                _misses[_ccy] += 1
                _last_actual.setdefault(_ccy, []).append(
                    (_ev.get("event", ""), _ev.get("actual", ""), _ev.get("forecast", ""), False)
                )

        if any(_total.values()):
            lines.append("=== Economic Surprise Index — CESI-style rolling (calendar-data/calendar.json) ===")
            lines.append("  (beats − misses vs forecast for high/medium-impact G8 events in the rolling window)")
            lines.append("  (Positive net score = data consistently outperforming expectations → currency tailwind)")
            lines.append("  CCY  | Beats | Misses | Total | Net  | Signal")
            _sorted_cesi = sorted(
                [(c, _beats[c], _misses[c], _total[c], _beats[c] - _misses[c])
                 for c in CURRENCIES if _total[c] > 0],
                key=lambda x: -x[4]
            )
            _surprise_leaders = []
            _surprise_laggards = []
            for _cc, _b, _m, _t, _net in _sorted_cesi:
                _sig = ""
                if _net >= 10:
                    _sig = "STRONG POSITIVE — sustained data beat vs expectations"
                    _surprise_leaders.append((_cc, _net))
                elif _net >= 4:
                    _sig = "Moderate positive bias"
                    _surprise_leaders.append((_cc, _net))
                elif _net <= -10:
                    _sig = "STRONG NEGATIVE — persistent data misses"
                    _surprise_laggards.append((_cc, _net))
                elif _net <= -4:
                    _sig = "Moderate negative bias"
                    _surprise_laggards.append((_cc, _net))
                else:
                    _sig = "Neutral"
                _lines_entry = f"  {_cc:<4} |  {_b:>4} |   {_m:>4} | {_t:>5} | {_net:>+4} | {_sig}"
                lines.append(_lines_entry)
            lines.append("")
            if _surprise_leaders or _surprise_laggards:
                lines.append("  CESI EXTREMES (use to support or contradict COT/RR directional bias):")
                for _cc, _n in _surprise_leaders:
                    lines.append(f"    {_cc}: net={_n:+d} beats → positive CESI → data outperforming, supports {_cc} strength bias")
                for _cc, _n in _surprise_laggards:
                    lines.append(f"    {_cc}: net={_n:+d} misses → negative CESI → data disappointing, supports {_cc} weakness bias")
                lines.append("  NOTE: CESI diverging from COT direction = fade risk. CESI + COT aligned = high-conviction theme.")
            lines.append("")

        # 9b. surpriseStats z-scores — per-series normalized surprises (gap 2)
        # calendar.json surpriseStats has {mean, std, n} for up to 80 series.
        # IMPORTANT: mean and std are of the SURPRISE DELTA (actual − forecast), not of raw actuals.
        # z = (latest_surprise - mean_surprise) / std_surprise.
        # |z| >= 2.0 = 2σ shock (Bloomberg ESURPRISE tier). Only emit n >= 3, |z| >= 1.5.
        _ss = _cal_raw.get("surpriseStats", {})
        _z_extremes = []
        if _ss:
            # Build lookup: latest (actual, forecast) per (ccy/event_title) key
            _title_to_delta: dict = {}
            for _ev2 in reversed(_cal_raw["events"]):
                _ccy2 = _ev2.get("currency", "")
                if _ccy2 not in _G8:
                    continue
                _ttl = _ev2.get("event", "")
                _key2 = f"{_ccy2}/{_ttl}"
                if _key2 not in _title_to_delta:
                    _a2 = _parse_num(_ev2.get("actual"))
                    _f2 = _parse_num(_ev2.get("forecast"))
                    if _a2 is not None and _f2 is not None:
                        _title_to_delta[_key2] = (_a2 - _f2, _a2, _f2)

            for _series_key, _stats in _ss.items():
                _n2 = _stats.get("n", 0)
                _std2 = _stats.get("std", 0.0)
                _mean2 = _stats.get("mean")
                if _n2 < 3 or _std2 < 0.0001 or _mean2 is None:
                    continue
                _entry = _title_to_delta.get(_series_key)
                if _entry is None:
                    continue
                _delta2, _a2_raw, _f2_raw = _entry
                _z2 = (_delta2 - _mean2) / _std2
                if abs(_z2) >= 1.5:
                    _z_extremes.append((_series_key, _delta2, _a2_raw, _f2_raw, _mean2, _std2, _n2, _z2))

        if _z_extremes:
            _z_extremes.sort(key=lambda x: -abs(x[7]))
            lines.append("  -- surpriseStats: per-series z-scores (|z| ≥ 1.5, n ≥ 3) --")
            lines.append("  (z = (latest_surprise − mean_surprise) / std. |z| ≥ 2.0 = 2σ shock per Bloomberg ESURPRISE tier)")
            for _sk, _dlt, _araw, _fraw, _mn, _sd, _nn, _zv in _z_extremes[:8]:
                _zs = f"{_zv:+.2f}σ"
                _tier = "2σ SHOCK" if abs(_zv) >= 2.0 else "1.5σ notable"
                _dir = "BEAT" if _dlt > 0 else "MISS"
                lines.append(
                    f"  {_sk}: actual={_araw:.4g} vs forecast={_fraw:.4g}"
                    f" (surprise={_dlt:+.3g}, hist_mean={_mn:+.3g}, std={_sd:.3g}, n={_nn})"
                    f" → z={_zs}  {_dir}  {_tier}"
                )
            lines.append("")

    # 10. Inflation expectations vs policy rate (gap 4)
    # Source: extended-data/{CCY}.json → inflationExpectations field.
    # Institutional convention (Bloomberg): cite BEI (breakeven inflation) vs policy rate
    # to frame real rate differential. Real rate = policy rate - inflation expectations.
    # Positive real rate = restrictive CB stance. Negative = accommodative.
    # This block is in the drivers workflow but NOT in the main build_context() — blind spot.
    _infl_lines = []
    for _ic in CURRENCIES:
        _id = load_json(SITE_DIR / "extended-data" / f"{_ic}.json")
        if not _id or "data" not in _id:
            continue
        _inf_exp = _id["data"].get("inflationExpectations")
        if _inf_exp is None:
            continue
        _cb_rate = _ccy_rates.get(_ic)  # from CB rates lookup built in section 5
        _real_rate = None
        if _cb_rate is not None:
            _real_rate = round(_cb_rate - _inf_exp, 2)
        _rr_str = f" | real rate: {_real_rate:+.2f}%" if _real_rate is not None else ""
        _rr_label = ""
        if _real_rate is not None:
            if _real_rate >= 1.5:
                _rr_label = " — restrictive (positive real rate)"
            elif _real_rate >= 0.0:
                _rr_label = " — mildly restrictive"
            elif _real_rate >= -1.0:
                _rr_label = " — mildly accommodative"
            else:
                _rr_label = " — accommodative (negative real rate)"
        _infl_lines.append(
            f"  {_ic}: infl_exp={_inf_exp:.2f}%  cb_rate={_cb_rate if _cb_rate is not None else 'N/A'}%"
            f"{_rr_str}{_rr_label}"
        )

    if _infl_lines:
        lines.append("=== Inflation Expectations vs Real Policy Rate (extended-data) ===")
        lines.append("  (real rate = policy rate − inflation expectations. Source: extended-data/{CCY}.json)")
        lines.append("  (Positive real rate = restrictive. Negative = accommodative. Key for CB divergence framing.)")
        lines.extend(_infl_lines)
        lines.append("")
        lines.append(
            "  RULE: When framing CB divergence, cite the real rate differential — not just the nominal spread."
            " A CB with low nominal rate but negative inflation expectations (CHF, JPY) may have a less"
            " accommodative real stance than the nominal rate implies."
        )
        lines.append("")

    # 11. Finnhub source label clarification (gap 5)
    # fetch_news.py v5.12 now pulls Finnhub /news?category=forex articles in addition to RSS.
    # The AI prompt uses the unified news.json but does not distinguish source.
    # Add a note so the model can correctly attribute Finnhub-sourced articles.
    _news_raw2 = load_json(SITE_DIR / "news-data" / "news.json", default={})
    if isinstance(_news_raw2, dict):
        _finnhub_count = sum(
            1 for _a in _news_raw2.get("articles", [])
            if _a.get("source_type") == "finnhub" or _a.get("provider") == "Finnhub"
        )
        if _finnhub_count > 0:
            lines.append(f"  NOTE: {_finnhub_count} of the news articles above are sourced from Finnhub"
                         f" /news?category=forex (structured FX market news). The remainder are from RSS"
                         f" aggregators (NewsData, GNews). All have been deduplicated and TIER-1 tagged.")
            lines.append("")

    return "\n".join(lines)


def call_groq(api_key: str, system: str, user: str, max_tokens: int = 800, model: str = MODEL) -> str:
    """Call Groq API. Raises RuntimeError with DAILY_LIMIT or RATE_LIMIT tag on 429.

    Returns the response text. Token usage is tracked globally via _signals_tokens_used
    when called from within the signals path (tracked externally by the caller).
    """
    import time
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.25,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Two retries on transient 429 (rate limit within key) — 3 total attempts
    for attempt in range(3):
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        if r.status_code == 429:
            try:
                body = r.json().get("error", {}).get("message", "").lower()
            except Exception:
                body = r.text.lower()
            # Distinguish daily quota exhaustion from transient RPM rate limit:
            # daily limit → rotate immediately (no sleep), key is dead for today
            # rate limit  → respect retry-after header or fall back to 20s backoff
            if "daily" in body or "quota" in body or "per day" in body or "rate_limit_exceeded" in body.replace(" ","_") and "minute" not in body:
                raise RuntimeError("DAILY_LIMIT")
            # Transient RPM rate limit — honour Retry-After if present
            retry_after = int(r.headers.get("retry-after", 0))
            wait = retry_after if retry_after > 0 else KEY_SWITCH_PAUSE_RATE
            print(f"  Groq rate limit — waiting {wait}s (attempt {attempt+1}/3)...")
            time.sleep(wait)
            continue
        if r.status_code == 401:
            raise RuntimeError("INVALID_KEY")
        r.raise_for_status()
        data = r.json()
        # Expose token usage via a side-channel attribute so callers can track budget
        # without changing the return type (str). Attach to the function object itself.
        _usage = data.get("usage", {})
        call_groq._last_tokens = _usage.get("total_tokens", 0)
        return data["choices"][0]["message"]["content"].strip()
    raise RuntimeError("RATE_LIMIT")


# ── Gemini call + audit layer ─────────────────────────────────────────────────

def call_gemini(api_key: str, system: str, user: str, max_tokens: int = 4000,
                model: str = GEMINI_MODEL) -> str:
    """Call Gemini generateContent endpoint.

    Mirrors call_groq's error-tag convention (DAILY_LIMIT / RATE_LIMIT / INVALID_KEY)
    so the caller can handle both backends uniformly.

    max_tokens default is 4000 to ensure full JSON arrays are never truncated.

    Thinking config adapts to model family:
    - Gemini 3.x (gemini-3-*): uses thinkingLevel="minimal" — Gemini 3 Flash requires
      thought signatures even at minimal; thinkingBudget is not supported.
    - Gemini 2.5 (gemini-2.5-*): uses thinkingBudget=512 — low fixed budget for
      auditor role; enough for coherence checks, avoids token bloat.
    """
    import time as _t
    url = GEMINI_URL.format(model=model)

    # Build thinking config based on model family
    _model_lower = model.lower()
    if "gemini-3" in _model_lower:
        # Gemini 3 series: thinkingLevel replaces thinkingBudget
        # "minimal" = lowest reasoning overhead, fastest latency for structured output tasks
        _thinking_cfg = {"thinkingLevel": "minimal"}
    else:
        # Gemini 2.x series: thinkingBudget in tokens
        _thinking_cfg = {"thinkingBudget": 512}

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.15,   # Conservative — auditor/fallback should not drift
            "candidateCount": 1,
            "thinkingConfig": _thinking_cfg,
        },
    }
    # Dynamic timeout: scales with output budget so signals (max_tokens=3000) get
    # more time than narrative (max_tokens=600) on slow Gemini preview endpoints.
    # Formula: 90s base + 20s per additional 1k tokens above 600.
    # narrative (600)  → 90s   (unchanged)
    # signals   (3000) → 90 + 48 = 138s  → rounded to 150s for safety margin
    # session_ctx(2000)→ 90 + 28 = 118s
    _dynamic_timeout = min(180, 90 + max(0, round((max_tokens - 600) / 1000) * 20))

    for attempt in range(3):
        r = requests.post(url, json=payload, params={"key": api_key}, timeout=_dynamic_timeout)
        if r.status_code == 429:
            try:
                body = r.json()
            except Exception:
                body = {}
            msg = str(body).lower()
            if "quota" in msg or "daily" in msg or "per day" in msg or "resource_exhausted" in msg:
                raise RuntimeError("DAILY_LIMIT")
            retry_after = int(r.headers.get("retry-after", 0))
            wait = retry_after if retry_after > 0 else 30
            print(f"    Gemini 429 — waiting {wait}s (attempt {attempt + 1}/3)...")
            _t.sleep(wait)
            continue
        if r.status_code == 400:
            raise RuntimeError(f"GEMINI_BAD_REQUEST: {r.text[:300]}")
        if r.status_code in (401, 403):
            raise RuntimeError("INVALID_KEY")
        r.raise_for_status()
        data = r.json()
        try:
            candidate = data["candidates"][0]
            # Detect truncation before returning — MAX_TOKENS finish reason means the
            # JSON array was cut short and will fail parsing downstream.
            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "MAX_TOKENS":
                raise RuntimeError(f"GEMINI_TRUNCATED: response cut at max_tokens={max_tokens}. "
                                   f"Increase max_tokens or shorten input.")
            return candidate["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"GEMINI_PARSE_ERROR: {exc} — {str(data)[:300]}")
    raise RuntimeError("RATE_LIMIT")


def generate_narrative_via_gemini(gemini_keys: list, context: str, system_prompt=None) -> dict | None:
    """Generate narrative using Gemini as primary draft author (v7.54.0).

    Uses the same system prompt as generate_narrative so output is structurally identical.
    Returns a narrative dict or None if all Gemini keys fail (triggering Groq fallback).

    Gemini 3 Flash has superior instruction-following for structured JSON output,
    better reasoning for regime classification, and larger context window vs Groq.
    """
    import json as _jg
    import time as _tg

    _sys = system_prompt or NARRATIVE_SYSTEM
    _user = f"Context:\n{context}"

    for _gk in gemini_keys:
        try:
            print(f"  Gemini narrative — key {_gk[:8]}...")
            raw = call_gemini(_gk, _sys, _user, max_tokens=600, model=GEMINI_MODEL)
            raw = raw.strip()
            for fence in ("```json", "```"):
                if raw.startswith(fence):
                    raw = raw[len(fence):]
                if raw.endswith("```"):
                    raw = raw[:-3]
            raw = raw.strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start < 0 or end <= start:
                print(f"  WARNING Gemini narrative: no JSON object in response — {raw[:120]}")
                continue
            result = _jg.loads(raw[start:end])
            if not isinstance(result, dict) or "narrative" not in result:
                print(f"  WARNING Gemini narrative: missing 'narrative' key — {raw[:120]}")
                continue
            result.setdefault("regime", "NEUTRAL")
            result.setdefault("generated_at", "")
            result["_source"] = "gemini"
            print(f"  ✅ Gemini narrative succeeded (regime={result.get('regime','?')})")
            return result
        except RuntimeError as _ge:
            err = str(_ge)
            if "DAILY_LIMIT" in err:
                print(f"  Gemini narrative: key {_gk[:8]} daily limit — trying next")
                _tg.sleep(1)
            elif "RATE_LIMIT" in err:
                print(f"  Gemini narrative: rate limit — waiting 30s")
                _tg.sleep(30)
            else:
                print(f"  Gemini narrative: error {err[:80]} — trying next key")
        except Exception as _ge2:
            print(f"  Gemini narrative: unexpected error {_ge2} — trying next key")
    print("  WARNING Gemini narrative: all keys exhausted — triggering Groq fallback")
    return None


def generate_signals_via_gemini(gemini_keys: list, context: str, system_prompt=None) -> list:
    """Generate signals using Gemini as primary draft author (v7.54.0).

    Uses the same SIGNALS_SYSTEM prompt so output is structurally identical.
    Returns a list of signal dicts (possibly empty) — never raises.

    Gemini is the primary signals editor. Superior instruction-following reduces
    COT-OIS divergence retries and JSON malformation vs Groq. Python guards still
    run on all output. Groq is fallback only.
    """
    import json as _js
    import time as _ts

    _sys = system_prompt or SIGNALS_SYSTEM
    _user = f"Context:\n{context}"

    # Model cascade for signals: gemini-3-flash-preview (primary) → gemini-2.5-flash (stable fallback).
    # gemini-3-flash-preview is a preview endpoint with degraded SLA (503s, timeouts expected).
    # When all keys fail on the primary model, retry the same key pool on the stable model
    # before handing off to Groq. This avoids wasting Groq quota on Gemini preview outages.
    _SIGNALS_MODEL_CASCADE = [GEMINI_MODEL, "gemini-2.5-flash"]

    for _sig_model in _SIGNALS_MODEL_CASCADE:
        _model_label = _sig_model.replace("gemini-", "Gemini ")
        _model_had_daily_limit = False  # if all keys are daily-limited, skip cascade

        for _gk in gemini_keys:
            try:
                print(f"  Gemini signals ({_sig_model}) — key {_gk[:8]}...")
                # max_tokens=3000 — signals array is larger than narrative; give Gemini room
                raw = call_gemini(_gk, _sys, _user, max_tokens=3000, model=_sig_model)
                raw = raw.strip()
                for fence in ("```json", "```"):
                    if raw.startswith(fence):
                        raw = raw[len(fence):]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                raw = raw.strip()
                start = raw.find("[")
                end   = raw.rfind("]") + 1
                if start < 0 or end <= start:
                    print(f"  WARNING Gemini signals: no JSON array — {raw[:120]}")
                    continue
                result = _js.loads(raw[start:end])
                if not isinstance(result, list):
                    print(f"  WARNING Gemini signals: result is not a list")
                    continue
                if len(result) == 0:
                    print(f"  WARNING Gemini signals: returned empty array — trying next key")
                    continue
                # Tag each signal so logs show source model
                for _sig in result:
                    if isinstance(_sig, dict):
                        _sig["_source"] = f"gemini:{_sig_model}"
                print(f"  ✅ Gemini signals ({_sig_model}): {len(result)} signal(s) generated")
                return result
            except RuntimeError as _ge:
                err = str(_ge)
                if "DAILY_LIMIT" in err:
                    print(f"  Gemini signals ({_sig_model}): key {_gk[:8]} daily limit — trying next")
                    _model_had_daily_limit = True
                    _ts.sleep(1)
                elif "RATE_LIMIT" in err:
                    print(f"  Gemini signals ({_sig_model}): rate limit — waiting 30s")
                    _ts.sleep(30)
                else:
                    print(f"  Gemini signals ({_sig_model}): error {err[:80]} — trying next key")
            except Exception as _ge2:
                print(f"  Gemini signals ({_sig_model}): unexpected error {_ge2} — trying next key")

        if _model_had_daily_limit and _sig_model == GEMINI_MODEL:
            # All keys are daily-limited on primary model — cascade won't help, skip to Groq
            print(f"  WARNING Gemini signals: all keys daily-limited on {GEMINI_MODEL} — skipping cascade to Groq")
            break
        if _sig_model != _SIGNALS_MODEL_CASCADE[-1]:
            print(f"  WARNING Gemini signals: all keys failed on {_sig_model} — trying {_SIGNALS_MODEL_CASCADE[_SIGNALS_MODEL_CASCADE.index(_sig_model)+1]}")

    print("  WARNING Gemini signals: all Gemini keys/models exhausted — returning []")
    return []


def _apply_narrative_guards(narrative: dict, tokens: dict | None = None) -> dict:
    """Run essential post-processing guards on Gemini-primary narrative output.

    Mirrors guards inside generate_narrative() (Groq path) that were bypassed
    when Gemini became the primary author (v7.54.0).

    Guards applied:
      1. Safe-haven framing stripped (MIXED/RISK-ON) → risk-premium framing
      2. Modal verb sanitizer: may/could/might [verb] → is likely to [verb]
      3. Plural subject-verb agreement: tensions is → tensions are
      4. Token resolution: {{gold_pct}}, {{spx_pct}}, {{dxy_pct}}, {{vix}} → live values
         (mirrors _parse_and_process in the Groq path — fixes placeholders reaching frontend)
    """
    import re as _rn

    narr   = narrative.get("narrative", "")
    regime = narrative.get("regime", "NEUTRAL").upper()

    if not narr:
        return narrative

    # Guard 1: safe-haven framing — only correct in RISK-OFF
    if regime not in ("RISK-OFF",):
        _sh_subs = [
            (r'\bsafe[\s\-]haven(?:\s+\w+){0,2}\s+bid\b', 'geopolitical risk premium'),
            (r'\bsafe[\s\-]haven\s+demand\b',                'risk-premium demand'),
            (r'\bsafe[\s\-]haven\s+flows?\b',                'risk-premium flows'),
        ]
        _orig = narr
        for _p, _r in _sh_subs:
            narr = _rn.sub(_p, _r, narr, flags=_rn.IGNORECASE)
        if narr != _orig:
            print(f"  INFO _apply_narrative_guards: safe-haven → risk-premium (regime={regime})")

    # Guard 2: modal verbs
    # _VB uses a capturing group so \1 backreference works in replacements.
    # For ADV+VB patterns the verb is the last group — handled via lambda.
    _ADV = r'(?:further|also|still|now|soon|yet|already|again|ultimately|additionally|potentially)'
    _VB  = (r'(weigh|support|cap|limit|pressure|drive|push|pull|rise|fall|extend|rally|reverse|'
            r'hold|break|reach|test|trigger|sustain|add|signal|indicate|suggest|impact|determine|'
            r'affect|shape|guide|keep|introduce|reduce|increase|narrow|widen|compress|accelerate|'
            r'delay|reinforce|amplify|close|stabilize|prevent)')
    _adv_vb_repl = lambda m: f'is likely to {m.group(m.lastindex)}'
    _modal_subs = [
        (rf'(?<!\d\s)\bmay\s+{_ADV}\s+{_VB}\b',  _adv_vb_repl),
        (rf'\bcould\s+{_ADV}\s+{_VB}\b',            _adv_vb_repl),
        (rf'(?<!\d\s)\bmay\s+{_VB}\b',             r'is likely to \1'),
        (rf'\bcould\s+{_VB}\b',                       r'is likely to \1'),
        (r'(?<!\d\s)\bmay\s+be\b',                  r'is likely to be'),
        (r'\bcould\s+be\b',                            r'is likely to be'),
    ]
    for _p, _r in _modal_subs:
        narr = _rn.sub(_p, _r, narr, flags=_rn.IGNORECASE)

    # Guard 3: plural subject-verb agreement
    _PLUR = (r'(?:tensions?|pressures?|flows?|movements?|developments?|'
             r'risks?|events?|concerns?|signals?|conditions?)')
    narr = _rn.sub(rf'\b({_PLUR})\s+is\s+likely\s+to\b', r'\1 are likely to',
                   narr, flags=_rn.IGNORECASE)

    # Guard 4: carry bp correction — replaces hardcoded stale bp values near CB pair references
    # Root cause: if rates/*.json are present but Gemini ignores the context block and falls
    # back to training-memory values (e.g. 175bp for Fed-ECB), this guard corrects them
    # deterministically using the same rates/*.json source as CROSS_CARRY.
    try:
        import re as _rc
        _CB_PAIRS_NAR = [
            # (base_ccy, quote_ccy, regex pattern to match hardcoded bp near pair mention)
            ("USD", "EUR", r'(?:Fed[-\s]ECB|ECB[-\s]Fed)'),
            ("USD", "GBP", r'(?:Fed[-\s]BoE|BoE[-\s]Fed)'),
            ("USD", "JPY", r'(?:Fed[-\s]BoJ|BoJ[-\s]Fed)'),
            ("USD", "AUD", r'(?:Fed[-\s]RBA|RBA[-\s]Fed)'),
            ("USD", "CAD", r'(?:Fed[-\s]BoC|BoC[-\s]Fed)'),
            ("USD", "CHF", r'(?:Fed[-\s]SNB|SNB[-\s]Fed)'),
            ("USD", "NZD", r'(?:Fed[-\s]RBNZ|RBNZ[-\s]Fed)'),
        ]
        _ccy_rates_nar: dict = {}
        for _ccy in ("USD","EUR","GBP","JPY","AUD","CAD","CHF","NZD"):
            _rd = load_json(SITE_DIR / "rates" / f"{_ccy}.json")
            if _rd and _rd.get("observations"):
                try:
                    _ccy_rates_nar[_ccy] = float(_rd["observations"][0]["value"])
                except (ValueError, KeyError):
                    pass
        if _ccy_rates_nar:
            # Match: "[+]NNNbp" within 120 chars of a CB pair mention
            _bp_re = _rc.compile(r'\+?\d{2,3}bp', _rc.IGNORECASE)
            for _base, _quote, _cb_pat in _CB_PAIRS_NAR:
                _rb = _ccy_rates_nar.get(_base)
                _rq = _ccy_rates_nar.get(_quote)
                if _rb is None or _rq is None:
                    continue
                _actual_bp = round((_rb - _rq) * 100)
                _sign = "+" if _actual_bp >= 0 else ""
                _actual_str = f"{_sign}{_actual_bp}bp"
                # Find CB pair mention, then scan ±120 chars for a hardcoded bp value
                for _pair_m in _rc.finditer(_cb_pat, narr, _rc.IGNORECASE):
                    _win_start = max(0, _pair_m.start() - 120)
                    _win_end   = min(len(narr), _pair_m.end() + 120)
                    _window    = narr[_win_start:_win_end]
                    for _bp_m in _bp_re.finditer(_window):
                        _written_str = _bp_m.group(0)
                        _written_val = int(_rc.sub(r'[^\d]', '', _written_str))
                        if abs(_written_val - abs(_actual_bp)) >= 20:  # only fix if clearly wrong
                            _abs_start = _win_start + _bp_m.start()
                            _abs_end   = _win_start + _bp_m.end()
                            narr = narr[:_abs_start] + _actual_str + narr[_abs_end:]
                            print(f"  INFO _apply_narrative_guards: carry bp {_written_str} → {_actual_str} ({_base}/{_quote})")
                            break  # one fix per CB pair
    except Exception as _e:
        print(f"  WARN _apply_narrative_guards carry-bp guard failed: {_e}")

    # Guard 5: TIER-1/2 internal label strip
    # Gemini occasionally echoes the context-block directive [⚠️ TIER-1 — MUST LEAD NARRATIVE]
    # into narrative output as "primary TIER-1 focus", "Tier-1 policy catalyst", etc.
    # This is context bleed — the label is a pipeline directive, not client-facing language.
    # Four ordered passes strip all forms deterministically.
    import re as _t1re
    _t1_subs = [
        # Pass A: specific compound phrase
        (_t1re.compile(r'primary\s+TIER[-\s]?[12]\s+focus', _t1re.IGNORECASE), 'primary event risk'),
        # Pass B: "the/a TIER-1 [noun]" → "the key [noun]"
        (_t1re.compile(r'\b(the|a)\s+TIER[-\s]?[12]\s+(\w+)', _t1re.IGNORECASE), r'the key \2'),
        # Pass C: "next/major/upcoming TIER-1 [noun]" → "next key [noun]"
        (_t1re.compile(r'\b(next|major|upcoming)\s+TIER[-\s]?[12]\s+(\w+)', _t1re.IGNORECASE), r'\1 key \2'),
        # Pass D: any remaining bare label
        (_t1re.compile(r'\bTIER[-\s]?[12]\b', _t1re.IGNORECASE), 'high-impact'),
    ]
    _narr_before = narr
    for _t1pat, _t1rep in _t1_subs:
        narr = _t1pat.sub(_t1rep, narr)
    if narr != _narr_before:
        print("  INFO _apply_narrative_guards: TIER-1/2 label stripped from narrative output")

    # Guard 4 (v7.58.2): token resolution — mirrors _parse_and_process in the Groq path.
    # Gemini is instructed to emit {{gold_pct}}, {{spx_pct}}, {{dxy_pct}}, {{vix}} and
    # generate_narrative_via_gemini() returns the dict as-is without resolving them.
    # Without this step those placeholders reach index.json and are shown verbatim in the
    # frontend narrative panel. resolve_tokens() + fix_direction_mismatch() replace the
    # tokens with live intraday values (or strip them when data is unavailable).
    _resolve_tokens = tokens if tokens is not None else extract_tokens_from_intraday()
    if _resolve_tokens:
        _narr_pre_tok = narr
        narr = resolve_tokens(narr, _resolve_tokens)
        narr = fix_direction_mismatch(narr, _resolve_tokens)
        if narr != _narr_pre_tok:
            print("  INFO _apply_narrative_guards: placeholder tokens resolved in narrative.")

    return {**narrative, "narrative": narr}


def _apply_signal_guards(signals: list) -> list:
    """Run essential post-processing guards on Gemini-primary signals.

    In the old architecture (Groq draft → Gemini audit) guards ran inside
    generate_signals() and audit_signals_with_gemini(). In the Gemini-primary
    architecture, generate_signals_via_gemini() returns raw signals that bypass
    both those paths. This function applies the critical guards so Gemini-primary
    output receives identical deterministic cleaning as Groq output did.

    Guards applied (in order):
      1. Stale past meeting date         → strips past dates from CB meeting refs
      2. Modal verb sanitizer            → may/could/might → is likely to
      3. ADV_SUBS filler strip           → removes editorial/vague filler phrases
      4. CB rate stale-training guard    → replaces memorised rates with data values
      5. OIS probability injection       → injects mkt% probability into CB meeting refs
      6. Squeeze direction disambiguation → clarifies crowded-short squeeze direction
    """
    import re as _rg
    from datetime import datetime as _gdt, timezone as _gtz, date as _gdate

    _today = _gdt.now(_gtz.utc).date()
    _MONTHS = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
               'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    _CBS_PAT = r'(?:BoJ|BoE|RBA|RBNZ|SNB|ECB|Fed|FOMC|BoC)'
    _MON_PAT = (r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?'
                r'|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)')

    # Guard 1: stale past meeting date
    _spd_re = _rg.compile(
        rf'\b(?:next\s+)?({_CBS_PAT})?\s*meeting(?:\s+is)?\s+(?:on\s+)?'
        rf'(?:(\d{{1,2}})\s+({_MON_PAT})|({_MON_PAT})\s+(\d{{1,2}}))\b',
        _rg.IGNORECASE
    )
    def _spd_fix(m):
        cb = m.group(1) or "CB"
        if m.group(2) and m.group(3):   day_s, mon_s = m.group(2), m.group(3)
        elif m.group(4) and m.group(5): mon_s, day_s = m.group(4), m.group(5)
        else: return m.group(0)
        try:
            mn = _MONTHS.get(mon_s.lower()[:3])
            if mn is None: return m.group(0)
            d = _gdate(_today.year, mn, int(day_s))
            if d < _today: return f"next {cb} meeting"
        except Exception:
            pass
        return m.group(0)

    # Guard 2: modal verb substitution
    _ADV2 = r'(?:further|also|still|now|soon|yet|already|again|ultimately|additionally|potentially)'
    _VB2  = (r'(compress|sustain|retest|move|extend|target|reach|test|push|pull|drift|slide|rally|'
             r'break|hold|continue|reverse|unwind|trigger|create|offset|influence|propel|drive|'
             r'pressure|support|weigh|cap|limit|add|signal|indicate|suggest|impact|resolve|'
             r'determine|affect|shape|dominate|dictate|guide|keep|introduce|stabilize|prevent|'
             r'accelerate|delay|reinforce|amplify|reduce|increase|narrow|widen|compress|close|set\s+up)')
    _modal_subs2 = [
        (rf'(?<!\d\s)\bmay\s+{_ADV2}\s+{_VB2}\b',  r'is likely to \1'),
        (rf'\bcould\s+{_ADV2}\s+{_VB2}\b',            r'is likely to \1'),
        (rf'\bmight\s+{_ADV2}\s+{_VB2}\b',            r'is likely to \1'),
        (rf'(?<!\d\s)\bmay\s+{_VB2}\b',              r'is likely to \1'),
        (rf'\bcould\s+{_VB2}\b',                        r'is likely to \1'),
        (rf'\bmight\s+{_VB2}\b',                        r'is likely to \1'),
        (r'(?<!\d\s)\bmay\s+be\b',                    r'is likely to be'),
        (r'\bcould\s+be\b',                              r'is likely to be'),
        (r'\bmight\s+be\b',                              r'is likely to be'),
    ]

    # Guard 3: ADV_SUBS filler
    _adv_subs2 = [
        (r'\bpotential\s+(?:unwind|squeeze|reversal|rally|decline|move|shift|catalyst|'
         r'divergence|carry\s+extension|decrease|increase|impact|drop|rise|compression|'
         r'expansion|correction)\b',
         lambda m: m.group(0).split()[-1]),
        (r'\s*\bdrives?\s+the\s+(?:move|pair)\b\.?', '.'),
        (r'\b(?:especially\s+considering|considering\s+the)\b', 'given'),
        (r'\bwill\s+be\s+(?:a|the)\s+key\s+catalyst\b',         'is the catalyst'),
        (r'\bis\s+likely\s+to\s+(?:be\s+)?(?:a|the)\s+key\s+catalyst\b', 'is the catalyst'),
        (r'\bwill\s+be\s+important\s+for\s+resolving\b',         'determines'),
        (r'\bThe\s+mechanism\s+driving\s+the\s+move\s+is\s+(?:the|a)\s*', ''),
        (r'\b(?:BoJ|BoE|RBA|RBNZ|SNB|ECB|Fed|FOMC|BoC)\s+closed\s+at\b',
         lambda m: m.group(0).split(' closed')[0] + ' at'),
        (r'\bcreating\s+an?\s+asymmetric\s+(squeeze|unwind)\s+risk\b', r'creating \1 risk'),
        (r'  +', ' '),
    ]

    # Guard 4: CB rate stale-training
    _CB_RATE_FILES = {
        "Fed": "USD", "ECB": "EUR", "BoE": "GBP", "BoJ": "JPY",
        "RBA": "AUD", "BoC": "CAD", "SNB": "CHF", "RBNZ": "NZD",
    }

    # Guard 5: OIS probability injection
    try:
        _meetings_raw = load_json(SITE_DIR / "meetings-data" / "meetings.json") or {}
        _meetings_map = _meetings_raw.get("meetings", {})
    except Exception:
        _meetings_map = {}
    _CB_CCY_MAP = {
        "Fed": "USD", "FOMC": "USD", "ECB": "EUR", "BoE": "GBP",
        "BoJ": "JPY", "RBA": "AUD", "BoC": "CAD", "SNB": "CHF", "RBNZ": "NZD",
    }
    _oi_prob_re  = _rg.compile(r'\bnext\s+(' + _CBS_PAT + r')\s+meeting\b', _rg.IGNORECASE)
    _oi_already  = _rg.compile(r'(?:mkt[:\s]|\d+%\s+(?:hold|hike|cut)\s+priced|priced)',
                               _rg.IGNORECASE)

    # Guard 6: squeeze direction (JPY signals only)
    # Expanded _sq_crowd_re: catches "crowded short", "net short of -NNN,NNN", "massive short"
    # (Gemini often writes "COT net short of -88,543" without the word "crowded")
    # Fixed _sq_dir_re: no longer self-matches "short-squeeze risk" (the trigger phrase).
    # Only fires on explicit direction statements: "USD/JPY downside", "downside if shorts cover", etc.
    _sq_title_re   = _rg.compile(r'\bJPY\b', _rg.IGNORECASE)
    _sq_crowd_re   = _rg.compile(
        r'\b(?:crowded\s+(?:net\s+)?short|net\s+short\s+of\s+-[\d,]+|massive\s+short)\b',
        _rg.IGNORECASE
    )
    _sq_squeeze_re = _rg.compile(r'\bsqueeze\s+risk\b', _rg.IGNORECASE)
    _sq_dir_re     = _rg.compile(
        r'\bUSD/JPY\s+(?:downside|upside)\b|'
        r'\b(?:downside|upside)\s+if\s+shorts?\s+cover\b|'
        r'\bshorts?\s+cover\b',
        _rg.IGNORECASE
    )
    _sq_replace_re = _rg.compile(
        r'\b(?:a\s+)?(?:massive\s+|significant\s+)?short.squeeze\s+risk\b',
        _rg.IGNORECASE
    )

    result = []
    for sig in signals:
        if not isinstance(sig, dict):
            result.append(sig)
            continue

        text  = sig.get("text", "")
        title = sig.get("title", "")

        # Guard 1: stale dates
        before = text
        text = _spd_re.sub(_spd_fix, text)
        if text != before:
            print(f"  INFO _apply_signal_guards stale-date: removed past date in '{title[:40]}'")

        # Guard 2: modal verbs
        for _p, _r in _modal_subs2:
            text = _rg.sub(_p, _r, text, flags=_rg.IGNORECASE)

        # Guard 3: ADV_SUBS filler
        for _p, _r in _adv_subs2:
            if callable(_r):
                text = _rg.sub(_p, _r, text, flags=_rg.IGNORECASE)
            else:
                text = _rg.sub(_p, _r, text, flags=_rg.IGNORECASE)
        text = text.strip()

        # Guard 4: CB rate stale-training
        for _cb_label, _ccy_code in _CB_RATE_FILES.items():
            _rate_re = _rg.compile(rf'\b{_rg.escape(_cb_label)}\s+([\d.]+)%', _rg.IGNORECASE)
            for _m in _rate_re.finditer(text):
                _written = float(_m.group(1))
                try:
                    _rdata   = load_json(SITE_DIR / "rates" / f"{_ccy_code}.json")
                    _obs     = (_rdata or {}).get("observations", [])
                    _actual  = float(_obs[0]["value"]) if _obs else None
                except Exception:
                    _actual = None
                if _actual is not None and abs(_written - _actual) >= 0.10:
                    text = text.replace(f"{_cb_label} {_m.group(1)}%",
                                        f"{_cb_label} {_actual:.2f}%", 1)
                    print(f"  INFO _apply_signal_guards rate-fix: {_cb_label} {_written}% → {_actual:.2f}% in '{title[:30]}'")
                    break

        # Guard 5: OIS probability injection
        for _oi_m in _oi_prob_re.finditer(text):
            _oi_cb  = _oi_m.group(1)
            _start  = max(0, _oi_m.start() - 30)
            _end    = min(len(text), _oi_m.end() + 60)
            if _oi_already.search(text[_start:_end]):
                continue
            _oi_ccy = _CB_CCY_MAP.get(_oi_cb)
            if not _oi_ccy:
                continue
            _meet      = _meetings_map.get(_oi_ccy, {})
            _hike_prob = _meet.get("hikeProb") or 0
            _cut_prob  = _meet.get("cutProb")  or 0
            _hold_prob = 100 - _hike_prob - _cut_prob
            _bias      = _meet.get("bias", "hold")
            # Guard: bias label is only used when the corresponding probability is non-zero.
            # meetings.json can have bias="hike" with hikeProb=0 (stale bias tag after
            # market repricing) — showing "0% hike priced" is misleading. Fall through to hold.
            if _bias == "hike" and _hike_prob > 0:   _prob_str = f"mkt: {_hike_prob}% hike priced"
            elif _bias == "cut" and _cut_prob > 0:   _prob_str = f"mkt: {_cut_prob}% cut priced"
            else:                                      _prob_str = f"mkt: {_hold_prob}% hold priced"
            _inject = f"next {_oi_cb} meeting ({_prob_str})"
            text = text[:_oi_m.start()] + _inject + text[_oi_m.end():]
            print(f"  INFO _apply_signal_guards OIS: injected '{_prob_str}' for {_oi_cb} in '{title[:30]}'")
            break  # one injection per signal

        # Guard 6: squeeze direction disambiguation (JPY signals only)
        if (_sq_title_re.search(title) and
                _sq_crowd_re.search(text) and
                _sq_squeeze_re.search(text) and
                not _sq_dir_re.search(text)):
            # _sq_replace_re is already compiled with IGNORECASE — no flags kwarg needed.
            # Replacement omits "sets up" since the new pattern matches the whole phrase.
            text = _sq_replace_re.sub(
                "short-squeeze risk (USD/JPY downside if shorts cover)",
                text
            )
            print(f"  INFO _apply_signal_guards squeeze: disambiguated in '{title[:30]}'")

        # Guard 7: carry cross-pair mismatch
        # Detects when a signal cites a carry spread that belongs to a DIFFERENT pair
        # than the one in the title. Root cause: the context block lists carry differentials
        # for all G8 pairs; the model anchors on the most prominent bp value and applies
        # it to the wrong bilateral (e.g. NZD-CHF +225bp cited in a NZD/CAD signal).
        #
        # Logic:
        #   1. Extract the two currencies from the signal title (e.g. "NZD/CAD" → NZD, CAD).
        #   2. Load actual CB rates from rates/*.json.
        #   3. Compute the CORRECT carry spread for that pair (in bp).
        #   4. Scan text for "Xbp carry" or "carry advantage of Xbp" patterns.
        #   5. If the cited bp value matches a DIFFERENT pair's spread (tolerance ±15bp)
        #      AND does NOT match the signal pair's spread, flag and remove the carry claim
        #      and any erroneous CB label (e.g. "over the SNB" in a NZD/CAD signal).
        try:
            _g7_title_re  = _rg.compile(
                r'\b(EUR|GBP|USD|JPY|AUD|CAD|CHF|NZD)[/](EUR|GBP|USD|JPY|AUD|CAD|CHF|NZD)\b'
            )
            _g7_bp_re     = _rg.compile(r'\b(\d{2,4})bp\b')
            _g7_cb_ref_re = _rg.compile(
                r'\bover\s+the\s+(?:SNB|BoJ|BoE|RBA|RBNZ|BoC|ECB|Fed)\b',
                _rg.IGNORECASE
            )
            _g7_title_m = _g7_title_re.search(title)
            if _g7_title_m:
                _g7_base, _g7_quote = _g7_title_m.group(1), _g7_title_m.group(2)
                # Load actual rates for base and quote
                _g7_rates: dict = {}
                for _g7_ccy in (_g7_base, _g7_quote):
                    try:
                        _g7_rd = load_json(SITE_DIR / "rates" / f"{_g7_ccy}.json")
                        _g7_obs = (_g7_rd or {}).get("observations", [])
                        if _g7_obs:
                            _g7_rates[_g7_ccy] = float(_g7_obs[0]["value"])
                    except Exception:
                        pass
                if len(_g7_rates) == 2:
                    _g7_correct_bp = abs(
                        round((_g7_rates[_g7_base] - _g7_rates[_g7_quote]) * 100)
                    )
                    # Build a map of ALL G8 pair bp values to detect cross-pair anchoring
                    _G7_ALL_CCYS = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
                    _g7_all_rates: dict = {}
                    for _g7_c in _G7_ALL_CCYS:
                        try:
                            _g7_rd2 = load_json(SITE_DIR / "rates" / f"{_g7_c}.json")
                            _g7_obs2 = (_g7_rd2 or {}).get("observations", [])
                            if _g7_obs2:
                                _g7_all_rates[_g7_c] = float(_g7_obs2[0]["value"])
                        except Exception:
                            pass
                    # Compute all pairwise spreads to identify cross-pair contamination
                    _g7_foreign_spreads: set = set()
                    for _ga in _G7_ALL_CCYS:
                        for _gb in _G7_ALL_CCYS:
                            if _ga == _gb or {_ga, _gb} == {_g7_base, _g7_quote}:
                                continue
                            if _ga in _g7_all_rates and _gb in _g7_all_rates:
                                _g7_foreign_spreads.add(
                                    abs(round((_g7_all_rates[_ga] - _g7_all_rates[_gb]) * 100))
                                )
                    # Check each cited bp value in the signal text
                    for _g7_bp_m in _g7_bp_re.finditer(text):
                        _cited_bp = int(_g7_bp_m.group(1))
                        _correct_match  = abs(_cited_bp - _g7_correct_bp) <= 15
                        _foreign_match  = any(abs(_cited_bp - _f) <= 15
                                              for _f in _g7_foreign_spreads)
                        if not _correct_match and _foreign_match:
                            # Strip the erroneous bp carry claim and any wrong CB reference
                            # Pattern: "a +225bp carry advantage over the SNB" or
                            # "225bp carry advantage"
                            _carry_phrase_re = _rg.compile(
                                rf'(?:a\s+)?[+]?{_cited_bp}bp\s+carry\s+(?:advantage|premium|deficit)'
                                rf'(?:\s+over\s+the\s+\w+)?',
                                _rg.IGNORECASE
                            )
                            _before = text
                            text = _carry_phrase_re.sub("", text)
                            text = _g7_cb_ref_re.sub("", text)
                            text = _rg.sub(r'  +', ' ', text).strip()
                            if text != _before:
                                print(
                                    f"  INFO _apply_signal_guards carry-xpair: removed "
                                    f"{_cited_bp}bp claim (signal: {_g7_base}/{_g7_quote}, "
                                    f"correct spread: {_g7_correct_bp}bp) in '{title[:40]}'"
                                )
                            break  # one correction per signal is sufficient
        except Exception as _g7_e:
            print(f"  WARN _apply_signal_guards carry-xpair guard failed: {_g7_e}")

        result.append({**sig, "text": text.strip()})

    return result


# Gemini receives already-generated text and only needs to correct specific
# error classes — not discover what to write.
_GEMINI_AUDIT_SYSTEM = """\
You are a senior FX data auditor at a tier-1 bank. You receive a JSON array of FX setup
signals plus an authoritative fact-check data block. Your ONLY job is to fix factual
data errors. You are NOT an editor, NOT a stylist — you are a data verifier.

═══ CARDINAL RULE — READ FIRST ═══
You are a DATA CORRECTOR only. Fix specific factual errors and nothing else.
If a signal has no factual error, return it CHARACTER-FOR-CHARACTER unchanged.

STRICT DATA QUARANTINE — your ONLY data source is the AUTHORITATIVE FACT-CHECK DATA block.
  ✗ NEVER use training memory for any number (yields, correlations, rates, probabilities, prices)
  ✗ NEVER add a value not explicitly listed in the fact-check block
  ✗ NEVER invent yields (e.g. if JGB yield is absent, do not write any JGB percentage)
  ✗ NEVER invent correlations (e.g. "GBP/FTSE +0.327" unless in the block)
  ✗ NEVER add new sentences — corrections must stay within the word count of the original
  ✗ NEVER change wording that has no factual error — do not "improve" style or language
  ✗ NEVER introduce "may", "could", "might", "potential", "would" — keep existing language
  ✗ For CB meeting dates: the block shows "next meeting: YYYY-MM-DD". If that date is
    BEFORE the CURRENT UTC DATE listed in the block, write "next [CB] meeting" with no date.
    Do NOT guess or calculate a future date — omit the specific date entirely.

If you cannot correct an error using ONLY data from the fact-check block, leave text unchanged.

═══ WHAT TO CORRECT — 4 CLASSES OF FACTUAL ERROR ONLY ═══

CLASS 1 — COT DIRECTION INVERTED (highest priority):
  The block shows "CCY: SHORT/LONG [net]" — verify direction claims against this.
  JPY net NEGATIVE (e.g. −88,543) = specs SHORT JPY = USD/JPY UPSIDE thesis.
  A signal claiming "JPY strength" when JPY net is deeply negative is factually wrong.
  AUD net POSITIVE = specs LONG AUD = AUD/USD UPSIDE bias.
  Fix the direction. Keep all other wording unchanged.

CLASS 2 — YIELD ADVANTAGE WRONG:
  Verify from the bond yields block which currency has higher yield.
  Use ONLY yield values in the block. If JP 10Y is absent, do NOT write any JGB percentage.
  Fix inversion only. Keep all other wording unchanged.

CLASS 3 — CB RATE / PROBABILITY CONFUSED:
  "RBA bias at 4.10%" mixes the policy rate with bias probability — factual error.
  Correct form: "RBA at 4.10% with hike bias (N% priced)" using block values.
  Carry spread: AUD/USD carry = RBA rate minus Fed rate (not a third CCY).
  Fix the rate/probability confusion only. Keep all other wording unchanged.
  ⚠️  SPECIAL CASE — if the block shows bias=Hike but hikeProb=0%, the correct label
  is Hold (0% priced for a hike means no hike is expected). Fix the bias label to Hold.

CLASS 4 — DIRECTIONAL INCOHERENCE:
  A signal asserting two opposite directions for the same pair is factually wrong.
    WRONG: "yield gap limits JPY upside ... sustained JPY bid likely"  ← contradictory
    WRONG: "downside toward 1.16 ... reinforces upside bias"           ← contradictory
  Identify the DOMINANT driver (the one with most data support), keep it, delete the
  contradicting clause. Do not add new content. Keep all other wording unchanged.

═══ DO NOT CHANGE (these are handled by deterministic Python guards) ═══
• Modal language ("may", "could") — Python converts these; do not re-introduce them
• Filler phrases — Python strips these; do not re-introduce them
• Signal structure: time, priority, title, evidence — preserve exactly
• Directionally correct theses — do not alter substance if factually sound
• Pair selection and priority levels (unless a signal is factually inverted)
• Signal count — return exactly the same number of signals you receive
• Any signal with no factual error — copy it verbatim, character-for-character

═══ OUTPUT FORMAT ═══
Respond ONLY with a valid JSON array in the same format as the input.
No markdown fences, no preamble, no commentary — pure JSON only.
"""


def _build_gemini_fact_block() -> str:
    """Build a compact, structured fact-check block for Gemini.

    Contains ONLY the exact values Gemini needs to audit signals:
    CB rates, bond yields, COT net positions, FX spot levels, and CB meeting dates.
    Replaces the full 11k-char context to prevent Gemini from pulling in noise
    or falling back to training memory for specific numbers.
    """
    import json as _json2
    lines = []

    # Current time (so Gemini can flag past meeting dates)
    from datetime import datetime as _dt2, timezone as _tz2
    lines.append(f"CURRENT UTC DATE: {_dt2.now(_tz2.utc).strftime('%Y-%m-%d')}")
    lines.append("")

    # CB policy rates + next meeting + market bias
    meetings_raw = load_json(SITE_DIR / "meetings-data" / "meetings.json", default={})
    mtg_map = meetings_raw.get("meetings", {})
    CB_MAP = {"USD": "Fed", "EUR": "ECB", "GBP": "BoE", "JPY": "BoJ",
              "AUD": "RBA", "CAD": "BoC", "CHF": "SNB", "NZD": "RBNZ"}
    lines.append("=== CB RATES (authoritative — use ONLY these) ===")
    for ccy, label in CB_MAP.items():
        rd = load_json(SITE_DIR / "rates" / f"{ccy}.json")
        rate_str = "N/A"
        if rd and rd.get("observations"):
            try:
                rate_str = f"{float(rd['observations'][0]['value']):.2f}%"
            except (ValueError, KeyError):
                pass
        mtg = mtg_map.get(ccy, {})
        next_mtg   = mtg.get("nextMeeting", "unknown")
        bias_raw   = mtg.get("bias", "hold")
        hike_p     = mtg.get("hikeProb", 0) or 0
        cut_p      = mtg.get("cutProb",  0) or 0
        # Coherence fix: bias=hike + hikeProb=0 is contradictory → override to hold
        if bias_raw == "hike" and hike_p == 0:
            bias_raw = "hold"
        if bias_raw == "cut" and cut_p == 0:
            bias_raw = "hold"
        bias_label = {"cut": "Cut", "hold": "Hold", "hike": "Hike"}.get(bias_raw, bias_raw)
        prob_str   = f", hike={hike_p}%, cut={cut_p}%"
        lines.append(f"  {label} ({ccy}): {rate_str} | next meeting: {next_mtg} | bias: {bias_label}{prob_str}")
    lines.append("")

    # Bond yields (10Y only — what matters for carry signals)
    yield_lines = []
    YIELD_MAP = {"USD": "US 10Y", "EUR": "DE 10Y (Bund)", "GBP": "UK 10Y", "JPY": "JP 10Y (JGB)"}
    for ccy, label in YIELD_MAP.items():
        d = load_json(SITE_DIR / "extended-data" / f"{ccy}.json")
        if d and "data" in d:
            y10 = d["data"].get("bond10y")
            if y10 is not None:
                yield_lines.append(f"  {label}: {y10:.2f}%")
    # Fallback: intraday cross data for US yields
    intra = load_json(SITE_DIR / "intraday-data" / "quotes.json")
    if intra:
        cross = intra.get("cross", {})
        if not yield_lines:
            us10 = cross.get("us10y")
            us2  = cross.get("us2y")
            if us10: yield_lines.append(f"  US 10Y: {us10:.2f}%")
            if us2:  yield_lines.append(f"  US 2Y:  {us2:.2f}%")
    if yield_lines:
        lines.append("=== BOND YIELDS (authoritative) ===")
        lines.extend(yield_lines)
        lines.append("  NOTE: Do NOT invent yields not listed here (e.g. JGB rate must come from 'JP 10Y' above).")
        lines.append("")

    # COT net positions
    lines.append("=== CFTC COT NET POSITIONS (Leveraged Funds) ===")
    for ccy in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]:
        cd = load_json(SITE_DIR / "cot-data" / f"{ccy}.json")
        if cd:
            net = cd.get("netPosition")
            lp  = cd.get("longPositions", 0) or 0
            sp  = cd.get("shortPositions", 0) or 0
            tot = lp + sp
            lp_pct = f"{round(lp/tot*100,1)}%" if tot > 0 else "N/A"
            we  = cd.get("weekEnding", cd.get("reportDate", "?"))
            if net is not None:
                dir_str = "LONG" if net > 0 else "SHORT"
                lines.append(f"  {ccy}: {dir_str} {net:+,} (long% {lp_pct}) — week ending {we}")
    lines.append("  KEY: NEGATIVE net = specs SHORT that currency. POSITIVE net = specs LONG.")
    lines.append("  JPY SHORT net = bearish JPY positioning = USD/JPY UPSIDE bias.")
    lines.append("")

    # FX spot levels
    lines.append("=== FX SPOT LEVELS (Friday closing — reference only) ===")
    PAIR_KEYS = [("eurusd","EUR/USD",4),("gbpusd","GBP/USD",4),("usdjpy","USD/JPY",2),
                 ("audusd","AUD/USD",4),("usdcad","USD/CAD",4),("usdchf","USD/CHF",4),("nzdusd","NZD/USD",4)]
    iq = (intra or {}).get("quotes", {}) if intra else {}
    for key, pair, dec in PAIR_KEYS:
        d = iq.get(key)
        if d and d.get("close"):
            lines.append(f"  {pair}: {round(float(d['close']), dec)}")
    lines.append("")

    return "\n".join(lines)


def audit_signals_with_gemini(signals: list, context: str, gemini_keys: list) -> list:
    """Run Groq-generated signals through Gemini as an auditor/corrector.

    Returns corrected signals. Falls back to the original signals on any error —
    the audit layer is never a blocker for publication.

    Key pool: tries each key in order, rotates on DAILY_LIMIT or RATE_LIMIT.
    If all keys exhausted, logs a warning and returns original signals unchanged.
    """
    import json as _json
    import time as _time

    if not gemini_keys:
        print("  INFO Gemini audit skipped — no GEMINI_API_KEY configured.")
        return signals

    # Build a compact, structured fact-check block instead of the full 11k context.
    # This prevents Gemini from (a) getting overwhelmed by noise and falling back to
    # training memory for specific numbers, and (b) "finding" data in the context to
    # introduce into signals (e.g. a correlation value from the headlines block).
    fact_block = _build_gemini_fact_block()
    signals_json = _json.dumps(signals, ensure_ascii=False, indent=2)
    user_msg = (
        f"AUTHORITATIVE FACT-CHECK DATA (use ONLY these values — ignore training memory):\n"
        f"{fact_block}\n"
        f"SIGNALS TO AUDIT AND CORRECT:\n{signals_json}"
    )

    last_err = None
    for key in gemini_keys:
        label = f"{key[:8]}..."
        try:
            print(f"  Gemini audit — key {label}")
            raw = call_gemini(key, _GEMINI_AUDIT_SYSTEM, user_msg, max_tokens=4000)
            # Strip accidental markdown fences
            raw = raw.strip()
            for fence in ("```json", "```"):
                if raw.startswith(fence):
                    raw = raw[len(fence):]
                if raw.endswith("```"):
                    raw = raw[:-3]
            raw = raw.strip()
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start < 0 or end <= start:
                raise ValueError(f"No JSON array in Gemini response: {raw[:200]}")
            corrected = _json.loads(raw[start:end])
            if not isinstance(corrected, list) or len(corrected) == 0:
                raise ValueError("Gemini returned empty or non-list")
            if len(corrected) > len(signals):
                print(f"  WARNING Gemini returned {len(corrected)} signals vs {len(signals)} input — truncating")
                corrected = corrected[: len(signals)]
            # Merge: preserve all original metadata; accept only text/title/evidence/priority from Gemini
            VALID_PRIORITIES = {"critical", "warning", "info"}
            merged = []
            for i, orig in enumerate(signals):
                gem = corrected[i] if i < len(corrected) else orig
                merged.append({
                    **orig,
                    "text":     gem.get("text",     orig.get("text", "")),
                    "title":    gem.get("title",    orig.get("title", "")),
                    "evidence": gem.get("evidence", orig.get("evidence", [])),
                    "priority": (gem.get("priority", orig.get("priority", "info"))
                                 if gem.get("priority", "") in VALID_PRIORITIES
                                 else orig.get("priority", "info")),
                })
            changes = sum(1 for o, m in zip(signals, merged) if o.get("text") != m.get("text"))

            # ── Post-audit hallucination guard ─────────────────────────────────────
            # Gemini sometimes introduces data points not in the fact-check block
            # (e.g. "JGB 2.42%", "FTSE correlation +0.327") despite the data quarantine
            # instruction. This guard detects X.XX% patterns in audited text that don't
            # match any known authoritative value, and reverts that signal to the original.
            #
            # Known-good values: CB rates + bond yields from the fact-check block.
            import re as _re_hg
            _known_pcts: set[str] = set()
            for _hg_ccy in ["USD","EUR","GBP","JPY","AUD","CAD","CHF","NZD"]:
                _hg_rd = load_json(SITE_DIR / "rates" / f"{_hg_ccy}.json")
                if _hg_rd and _hg_rd.get("observations"):
                    try:
                        _hg_r = float(_hg_rd["observations"][0]["value"])
                        _known_pcts.add(f"{_hg_r:.2f}")
                        _known_pcts.add(f"{_hg_r:.1f}")
                        _known_pcts.add(str(int(_hg_r)) if _hg_r == int(_hg_r) else "")
                    except Exception: pass
                _hg_yd = load_json(SITE_DIR / "extended-data" / f"{_hg_ccy}.json")
                if _hg_yd and "data" in _hg_yd:
                    for _yk in ("bond10y","bond2y","bond5y"):
                        _yv = _hg_yd["data"].get(_yk)
                        if _yv is not None:
                            _known_pcts.add(f"{float(_yv):.2f}")
                            _known_pcts.add(f"{float(_yv):.1f}")
            # Also allow hike/cut probabilities from meetings.json
            _hg_mtgs = (load_json(SITE_DIR / "meetings-data" / "meetings.json") or {}).get("meetings", {})
            for _hg_mc in _hg_mtgs.values():
                for _hg_pk in ("hikeProb","cutProb"):
                    _hg_pv = _hg_mc.get(_hg_pk)
                    if _hg_pv is not None:
                        _known_pcts.add(str(int(_hg_pv)))
            # Common non-suspicious values: 0, 100, and any value that appears in the original signal
            _known_pcts.discard("")

            reverted = 0
            for i, (orig, gem_merged) in enumerate(zip(signals, merged)):
                if orig.get("text") == gem_merged.get("text"):
                    continue  # unchanged — no need to check
                # Find all X.XX% values in the Gemini-revised text
                new_text = gem_merged.get("text", "")
                orig_text = orig.get("text", "")
                _new_nums = set(_re_hg.findall(r'(\d+\.\d+)%', new_text))
                # Extract all numbers from original text (allowed reference set)
                _orig_nums = set(_re_hg.findall(r'(\d+\.\d+)%', orig_text))
                # Any number in the new text that is not in known_pcts AND not in orig_text
                # is a candidate hallucination
                _suspicious = _new_nums - _known_pcts - _orig_nums
                if _suspicious:
                    print(f"  WARNING Gemini hallucination guard: signal {i} ('{orig.get('title','')[:30]}') "
                          f"introduced unknown value(s) {_suspicious} — reverting to original")
                    merged[i] = orig
                    reverted += 1

            if reverted:
                changes = sum(1 for o, m in zip(signals, merged) if o.get("text") != m.get("text"))
                print(f"  INFO Post-audit guard reverted {reverted} signal(s). Net changes: {changes}/{len(signals)}")

            # ── Post-Gemini re-sanitization pass ───────────────────────────────
            # Gemini rewrites signal text AFTER the Python guards have already run
            # on the Groq output. This means Gemini can re-introduce errors the
            # Python guards fixed (e.g. hike-prob contradiction, squeeze-direction
            # confusion). This pass re-applies the critical guards to merged output.
            import re as _re_pg

            # Load authoritative data once for the pass
            _pg_meetings = (load_json(SITE_DIR / "meetings-data" / "meetings.json") or {}).get("meetings", {})
            _pg_cot = {}
            for _pgc in ["USD","EUR","GBP","JPY","AUD","CAD","CHF","NZD"]:
                _pgd = load_json(SITE_DIR / "cot-data" / f"{_pgc}.json")
                if _pgd and _pgd.get("netPosition") is not None:
                    _pg_cot[_pgc] = _pgd["netPosition"]

            _CB_CCY_PG = {"RBNZ":"NZD","RBA":"AUD","BoC":"CAD","BoE":"GBP",
                          "ECB":"EUR","BoJ":"JPY","Fed":"USD","FOMC":"USD","SNB":"CHF"}
            _pg_fixes = 0

            for _pgi, _pgm in enumerate(merged):
                _pgt = _pgm.get("text", "")
                _pg_orig = _pgt  # track changes

                # Guard A: hike/cut probability — fix Gemini re-introductions
                for _pgcb, _pgccy in _CB_CCY_PG.items():
                    _pgccy_d  = _pg_meetings.get(_pgccy, {})
                    _pg_hike  = _pgccy_d.get("hikeProb")
                    _pg_cut   = _pgccy_d.get("cutProb")
                    if _pg_hike is None:
                        continue
                    _pg_prob_re = _re_pg.compile(
                        rf"(?:{_re_pg.escape(_pgcb)}\s+(?:hike|cut)\s+(\d{{1,3}})%"
                        rf"|{_re_pg.escape(_pgcb)}\s+.*?(\d{{1,3}})%\s+(?:priced|probability)"
                        rf"|(\d{{1,3}})%\s+(?:priced|probability)(?:\s+for)?(?:\s+a)?\s+{_re_pg.escape(_pgcb)})",
                        _re_pg.IGNORECASE | _re_pg.DOTALL
                    )
                    for _pgm2 in _pg_prob_re.finditer(_pgt):
                        _pgw = int(next(g for g in _pgm2.groups() if g is not None))
                        if abs(_pgw - _pg_hike) >= 15:
                            _pgt = _pgt.replace(f"{_pgw}%", f"{_pg_hike}%", 1)
                            if _pgt != _pg_orig:
                                print(f"  INFO Post-Gemini hike-prob fix: {_pgcb} {_pgw}% → {_pg_hike}% in signal {_pgi}")
                        break

                # Guard B: COT direction confusion for inverted pairs
                # "squeeze risk for JPY" when JPY net is deeply negative = specs SHORT JPY
                # → short-squeeze WOULD rally JPY = USD/JPY downside. But the thesis on
                # a CRITICAL JPY signal is USD/JPY upside from carry/yield advantage.
                # The correct framing is "crowded-short unwind risk" (= JPY rally risk),
                # NOT "squeeze risk for JPY" in the context of USD/JPY upside.
                # Simplest deterministic fix: replace "squeeze risk for [CCY]" with
                # the institutionally correct "crowded-short unwind risk" when that CCY
                # has a deeply negative COT net (specs are short that CCY).
                for _pgccy, _pgnet in _pg_cot.items():
                    if _pgnet < -10_000:
                        # Specs SHORT this CCY → "squeeze risk for [CCY]" is ambiguous/wrong
                        _squeeze_pat = _re_pg.compile(
                            rf"\bsqueeze\s+risk\s+for\s+{_pgccy}\b",
                            _re_pg.IGNORECASE
                        )
                        _pgt = _squeeze_pat.sub(
                            f"crowded-short unwind risk in {_pgccy}",
                            _pgt
                        )

                # Guard C: "WTI crude oil price drop" — verify against actual WTI 1D%
                # If WTI is not down on the day, replace "WTI crude oil price drop" with
                # "WTI crude oil move" to avoid claiming a drop that isn't happening.
                if "wti crude oil price drop" in _pgt.lower():
                    _pg_intra = load_json(SITE_DIR / "intraday-data" / "quotes.json")
                    _pg_wti_pct = None
                    if _pg_intra:
                        _pg_wti_d = (_pg_intra.get("quotes") or {}).get("wti") or \
                                    (_pg_intra.get("cross") or {}).get("wti")
                        if isinstance(_pg_wti_d, dict):
                            _pg_wti_pct = _pg_wti_d.get("pct")
                        elif isinstance(_pg_wti_d, (int, float)):
                            pass  # raw value, no pct available
                    if _pg_wti_pct is None or _pg_wti_pct >= 0:
                        # WTI not down (or data unavailable) — neutralize the claim
                        _pgt = _re_pg.sub(
                            r"WTI crude oil price drop",
                            "WTI crude oil price move",
                            _pgt, flags=_re_pg.IGNORECASE
                        )
                        if _pgt != _pg_orig:
                            print(f"  INFO Post-Gemini WTI guard: 'price drop' → 'price move' (WTI 1D not negative) in signal {_pgi}")

                # Guard D: Stale past meeting dates
                # Groq/Gemini sometimes writes "next BoJ meeting on 28 Apr" when Apr 28
                # is already in the past. Detect and replace with "next [CB] meeting".
                # Only removes the date — keeps the CB name and surrounding context.
                # Uses the UTC date at generation time so the guard is timezone-correct.
                import re as _re_pgd
                from datetime import datetime as _pgdt, timezone as _pgtz, date as _pgdate
                _pg_today = _pgdt.now(_pgtz.utc).date()
                _PGD_MONTHS_S = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                                 'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
                _PGD_CBS     = r'(?:BoJ|BoE|RBA|RBNZ|SNB|ECB|Fed|FOMC|BoC)'
                _PGD_MON     = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                _pgd_pat = _re_pgd.compile(
                    rf'\b(?:next\s+)?({_PGD_CBS})?\s*meeting(?:\s+is)?\s+(?:on\s+)?(?:(\d{{1,2}})\s+({_PGD_MON})|({_PGD_MON})\s+(\d{{1,2}}))\b',
                    _re_pgd.IGNORECASE
                )
                def _pgd_replace(m):
                    cb = m.group(1) or "CB"  # fallback when no CB name in the match
                    if m.group(2) and m.group(3):     # "28 Apr"
                        day_s, mon_s = m.group(2), m.group(3)
                    elif m.group(4) and m.group(5):   # "Apr 28"
                        mon_s, day_s = m.group(4), m.group(5)
                    else:
                        return m.group(0)
                    try:
                        mn = _PGD_MONTHS_S.get(mon_s.lower()[:3])
                        if mn is None:
                            return m.group(0)
                        d = _pgdate(_pg_today.year, mn, int(day_s))
                        if d < _pg_today:  # date is in the past
                            return f"next {cb} meeting"
                    except Exception:
                        pass
                    return m.group(0)
                _pgt_before_d = _pgt
                _pgt = _pgd_pat.sub(_pgd_replace, _pgt)
                if _pgt != _pgt_before_d:
                    print(f"  INFO Post-Gemini stale-date guard: removed past meeting date in signal {_pgi}")

                # Guard E: Wrong carry counterpart for USD counter-pairs
                # Gemini occasionally compares GBP/USD carry to BoJ instead of Fed,
                # e.g. "+133bp carry advantage over the BoJ (BoE 3.75% vs BoJ 0.75%)"
                # in a GBP/USD signal. For USD counter-pairs the correct counterpart
                # is always Fed (USD). This guard detects and removes the incorrect
                # BoJ reference from GBP/USD, AUD/USD, NZD/USD, EUR/USD signals
                # by checking the signal title for the pair and the text for "vs BoJ"
                # in a non-JPY context.
                import re as _re_pge
                _pge_title = _pgm.get("title", "")
                _USD_COUNTER_PAIRS = ("EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD")
                _is_usd_counter = any(p in _pge_title for p in _USD_COUNTER_PAIRS)
                if _is_usd_counter:
                    # Remove spurious "vs BoJ" carry comparisons from non-JPY pairs
                    # e.g. "BoE 3.75% vs BoJ 0.75%" → "BoE 3.75% vs Fed 3.75%"
                    # We cannot know Fed rate here without loading rates; safer to strip
                    # the entire carry clause and let the correct one (already in text) stand.
                    _pge_boj_pat = _re_pge.compile(
                        r'\(\s*(?:BoE|RBA|RBNZ|ECB)\s+[\d.]+%\s+vs\s+BoJ\s+[\d.]+%\s*\)',
                        _re_pge.IGNORECASE
                    )
                    _pgt_before_e = _pgt
                    _pgt = _pge_boj_pat.sub('', _pgt).strip()
                    # Also catch "carry advantage over the BoJ" in non-JPY pairs
                    _pgt = _re_pge.sub(
                        r'\b(carry\s+(?:advantage|differential|spread))\s+over\s+the\s+BoJ\b',
                        r'\1 over the Fed',
                        _pgt, flags=_re_pge.IGNORECASE
                    )
                    if _pgt != _pgt_before_e:
                        print(f"  INFO Post-Gemini carry-counterpart guard: removed BoJ reference from {_pge_title} in signal {_pgi}")

                # Guard F: Subject-verb agreement in signal text
                # The modal sanitizer converts "movements may influence" → "movements is likely to"
                # Fix: plural FX nouns + "is likely to" → "are likely to"
                import re as _re_pgf
                _PGF_PLURAL = (
                    r'(?:tensions|forces|pressures|flows|movements|moves|developments|'
                    r'risks|concerns|factors|headwinds|tailwinds|conditions|dynamics)'
                )
                _pgt = _re_pgf.sub(
                    rf'\b({_PGF_PLURAL})\b([^.;]{{0,60}}?)\bis\s+likely\s+to\b',
                    r'\1\2are likely to',
                    _pgt, flags=_re_pgf.IGNORECASE
                )

                # Guard G: Post-Gemini modal pass
                # Gemini can re-introduce "may impact", "may resolve", "may determine" after
                # the pre-Gemini modal sanitizer has already cleaned the Groq output.
                # Apply the same modal → "is likely to" substitution on Gemini's output.
                # This uses the same expanded verb list as the pre-Gemini signal guard.
                import re as _re_pgg
                _PGG_VERBS = (
                    r'compress|sustain|retest|move|extend|target|reach|test|push|pull|'
                    r'drift|slide|rally|break|hold|continue|reverse|unwind|trigger|create|set\s+up|'
                    r'offset|influence|propel|drive|pressure|support|weigh|cap|limit|add|signal|'
                    r'indicate|suggest|impact|resolve|determine|affect|shape|dominate|dictate|guide|'
                    r'keep|introduce|stabilize|prevent|accelerate|delay|reinforce|amplify|reduce|'
                    r'increase|narrow|widen|close'
                )
                _pgt = _re_pgg.sub(
                    rf'(?<!\d\s)\bmay\s+({_PGG_VERBS})\b',
                    r'is likely to \1', _pgt, flags=_re_pgg.IGNORECASE
                )
                # Guard G adverb-bridge: "may further/also/still [verb]" → "is likely to [verb]"
                _PGG_ADV = r'(?:further|also|still|now|soon|yet|already|again|ultimately|additionally|potentially)'
                _pgt = _re_pgg.sub(
                    rf'(?<!\d\s)\bmay\s+{_PGG_ADV}\s+({_PGG_VERBS})\b',
                    r'is likely to \1', _pgt, flags=_re_pgg.IGNORECASE
                )
                _pgt = _re_pgg.sub(
                    rf'\bcould\s+{_PGG_ADV}\s+({_PGG_VERBS})\b',
                    r'is likely to \1', _pgt, flags=_re_pgg.IGNORECASE
                )
                _pgt = _re_pgg.sub(
                    rf'\bmight\s+{_PGG_ADV}\s+({_PGG_VERBS})\b',
                    r'is likely to \1', _pgt, flags=_re_pgg.IGNORECASE
                )
                _pgt = _re_pgg.sub(
                    rf'\bcould\s+({_PGG_VERBS})\b',
                    r'is likely to \1', _pgt, flags=_re_pgg.IGNORECASE
                )
                _pgt = _re_pgg.sub(
                    rf'\bmight\s+({_PGG_VERBS})\b',
                    r'is likely to \1', _pgt, flags=_re_pgg.IGNORECASE
                )
                _pgt = _re_pgg.sub(
                    r'(?<!\d\s)\bmay\s+be\b', 'is likely to be', _pgt, flags=_re_pgg.IGNORECASE
                )

                # Guard H: Named-entity correlation claims
                # Gemini invents cross-asset correlations not in the data block
                # (e.g. "correlation break with FTSE 100", "GBP/FTSE correlation +0.327").
                # These are unfalsifiable by the numeric hallucination guard (no X.XX% pattern).
                # Strip any mention of FTSE, DAX, Nikkei, S&P-as-correlation-proxy in a signal
                # that wasn't present in the original Groq text.
                import re as _re_pgh
                _PGH_PAIR  = _pgm.get("title", "")
                _PGH_ORIG  = next((s.get("text","") for s in signals if s.get("title","") == _PGH_PAIR), "")
                _CORR_ENTITIES = r'(?:FTSE(?:\s+100)?|DAX(?:\s+40)?|Nikkei(?:\s+225)?|S&P\s*500|CAC\s*40|ASX\s*200)'
                _corr_pat = _re_pgh.compile(
                    rf'(?:correlation|corr)(?:\s+\w+){{0,3}}\s+(?:with|to|between)\s+(?:the\s+)?{_CORR_ENTITIES}|'
                    rf'{_CORR_ENTITIES}\s+(?:correlation|corr)',
                    _re_pgh.IGNORECASE
                )
                for _cm in _corr_pat.finditer(_pgt):
                    if _cm.group() not in _PGH_ORIG:
                        # Gemini introduced this correlation claim — strip the clause
                        # Replace the full clause containing the match (up to sentence boundary)
                        _pgt = _re_pgh.sub(
                            rf'[^.;]*?(?:correlation|corr)(?:\s+\w+){{0,3}}\s+(?:with|to|between)\s+(?:the\s+)?{_CORR_ENTITIES}[^.;]*[.;]?',
                            '', _pgt, flags=_re_pgh.IGNORECASE
                        ).strip()
                        print(f"  INFO Post-Gemini correlation guard: removed hallucinated FTSE/DAX/Nikkei correlation in signal {_pgi}")
                        break

                # Guard I: post-Gemini filler patterns — mirrors ADV_SUBS for post-Gemini text
                import re as _re_pgi_
                # "potential [X]" — full list matching ADV_SUBS
                _pgt = _re_pgi_.sub(
                    r'\bpotential\s+(carry\s+extension|shift|catalyst|upside|downside|breakout|move|divergence|decrease|increase|impact|drop|rise|compression|expansion|correction|unwind|squeeze|reversal)\b',
                    r'\1', _pgt, flags=_re_pgi_.IGNORECASE
                )
                # "The mechanism driving the move is [the/a]" → strip
                _pgt = _re_pgi_.sub(
                    r'\bThe\s+mechanism\s+driving\s+the\s+move\s+is\s+(?:the\s+|a\s+)?',
                    r'', _pgt, flags=_re_pgi_.IGNORECASE
                ).strip()
                # "[X] drives the move/pair" → strip filler tail
                _pgt = _re_pgi_.sub(
                    r'\s+drives\s+the\s+(?:move|pair)\b\.?',
                    r'.', _pgt, flags=_re_pgi_.IGNORECASE
                ).strip()
                # "will be a/the key catalyst" → "is the catalyst"
                _pgt = _re_pgi_.sub(
                    r'\bwill\s+be\s+(?:a|the)\s+key\s+catalyst\b',
                    r'is the catalyst', _pgt, flags=_re_pgi_.IGNORECASE
                )
                # "is a/the key catalyst as the key catalyst" dedup
                _pgt = _re_pgi_.sub(
                    r'\bis\s+(?:a|the)\s+key\s+catalyst\s+as\s+the\s+key\s+catalyst\b',
                    r'is the catalyst', _pgt, flags=_re_pgi_.IGNORECASE
                )
                # "is likely to impact direction/outcome" → "determines direction"
                _pgt = _re_pgi_.sub(
                    r'\bis\s+likely\s+to\s+impact\s+(?:the\s+)?(?:pair\'s\s+)?(?:direction|outcome|trend|bias)\b',
                    r'determines direction', _pgt, flags=_re_pgi_.IGNORECASE
                )
                # "[CB] closed at [rate]%" artefact → "[CB] at [rate]%"
                _pgt = _re_pgi_.sub(
                    r'\b(BoE|BoC|BoJ|RBA|RBNZ|SNB|ECB|Fed|FOMC)\s+closed\s+at\b',
                    r'\1 at', _pgt, flags=_re_pgi_.IGNORECASE
                )
                # "creating an asymmetric risk" → "creating unwind risk"
                _pgt = _re_pgi_.sub(
                    r'\bcreating\s+an\s+asymmetric\s+risk\b',
                    r'creating unwind risk', _pgt, flags=_re_pgi_.IGNORECASE
                )
                # "affecting [X]'s value" / "affecting [pair]'s direction" → strip tail
                _pgt = _re_pgi_.sub(
                    r',?\s*affecting\s+(?:the\s+)?(?:\w+\'s\s+)?(?:value|direction|movement)\.?\s*$',
                    r'', _pgt, flags=_re_pgi_.IGNORECASE
                ).strip()
                # "pointing to a [X]" → strip editorial wrapper
                _pgt = _re_pgi_.sub(
                    r',?\s*pointing\s+to\s+a\s+(?:potential\s+)?(\w+(?:\s+\w+){0,3})',
                    r'', _pgt, flags=_re_pgi_.IGNORECASE
                ).strip()

                if _pgt != _pg_orig:
                    merged[_pgi] = {**_pgm, "text": _pgt}
                    _pg_fixes += 1

            if _pg_fixes:
                print(f"  INFO Post-Gemini re-sanitization: {_pg_fixes} fix(es) applied.")

            print(f"  ✅ Gemini audit complete — {changes}/{len(signals)} signal(s) revised")
            return merged

        except RuntimeError as exc:
            err = str(exc)
            if "DAILY_LIMIT" in err or "RATE_LIMIT" in err:
                print(f"  Gemini key {label} exhausted ({err}) — trying next key")
                last_err = err
                _time.sleep(2)
                continue
            # Non-retryable
            print(f"  WARNING Gemini audit failed ({err}) — using Groq signals unchanged")
            return signals
        except Exception as exc:
            print(f"  WARNING Gemini audit error ({exc}) — using Groq signals unchanged")
            return signals

    print(f"  WARNING All Gemini keys exhausted ({last_err}) — using Groq signals unchanged")
    return signals


# ── Prompts ───────────────────────────────────────────────────────────────────

NARRATIVE_SYSTEM = """\
You are a senior FX strategist writing a live market snapshot for a professional trading terminal.
Write like a Bloomberg terminal desk note: declarative, data-driven, present-tense, 3–4 sentences, 380–480 characters.

=== MANDATORY RULES (violations cause rejection) ===

1. TIER-1 EVENT LEADS — when a high-impact headline contains any of these, it MUST open the narrative.
   This rule has ABSOLUTE PRIORITY — it overrides Rule 6 (G8 theme day), Rule 11 (CB differential),
   and every other lead-selection rule. Even when a G8 scorecard theme is present, if a TIER-1
   event is in the headlines, the narrative opens with the TIER-1 event first.

   TIER-1 CATEGORIES — any headline tagged [⚠️ TIER-1 — MUST LEAD NARRATIVE] in the context, OR
   any headline you recognise as belonging to these categories, requires you to lead:
   A) Geopolitical / military: escalation, military strike, invasion, conflict, missile, troops,
      Middle East, Gaza, Iran, Ukraine, Russia, North Korea — any active geopolitical shock or
      maritime corridor disruption in any region.
   B) Verbal FX intervention / MoF/FinMin warning: Finance Minister or FinMin warning on FX
      speculation, "decisive action" on currency, central bank emergency FX statement, MoF verbal
      intervention. THIS IS TIER-1 IN FX — the Japan FinMin warning on a specific exchange rate
      is THE single most market-moving JPY catalyst outside actual intervention. Lead with it.
      CORRECT: "Japan FinMin Katayama warns of decisive action on yen speculation — USD/JPY at
                159.78, testing 160.00 intervention threshold; 300bp carry vs intervention risk
                is the dominant trade-off. Watch next BoJ meeting."
   C) Commodity supply shock: OPEC output cut, pipeline attack, refinery strike, oil embargo,
      LNG disruption, maritime corridor blockade — any event repricing the energy complex by >2%
      or disrupting a major oil/gas transit route.
   D) Macro Tier-1 surprise: NFP surprise, CPI shock, emergency FOMC, GDP miss vs consensus,
      bank failure, systemic contagion risk.

   NOTE: The context pre-tags TIER-1 headlines with [⚠️ TIER-1 — MUST LEAD NARRATIVE].
   When you see this tag, open with that event. No exceptions.

   CORRECT (TIER-1 MoF warning):
     "Japan FinMin warns of decisive FX action — USD/JPY at 159.78, 160.00 intervention threshold
      in focus; 300bp carry sustains the bid but verbal risk is rising. Watch next BoJ meeting."
   CORRECT (TIER-1 geopolitical, G8 theme as secondary ≤1 clause):
     "Iran escalation drives crude bid — WTI higher; USD leads G8 (+0.42%) on carry;
      USD/JPY at 159.76. Watch UK Retail Sales at 08:00 UTC."
   WRONG:  "USD leads G8 this week (+0.42% avg) on Fed-BoJ rate divergence..." ← G8 lead when TIER-1 is tagged
   WRONG:  "Geopolitical tensions simmer as Trump comments on Iran..." ← literary verb; buries the event

2. DECLARATIVE LANGUAGE — use direct, action verbs. Literary or hedge language is forbidden:
   FORBIDDEN WORDS/PHRASES: "simmer", "creeping in", "lingers", "looms", "weighs on", "on the back of",
                             "amid", "as investors digest", "market participants", "cautious tone"
   CORRECT verbs: retreats · firms · bids · sells off · holds · breaks · compresses · widens · unwinds
   CORRECT:   "Middle East escalation drives JPY safe-haven bid — USD/JPY retreats to 158.82; gold higher {{gold_pct}}."
   WRONG:     "Risk-off creeping in as geopolitical tensions simmer — USD/JPY retreats..." ← two forbidden phrases
   FORBIDDEN INTERNAL LABELS: "TIER-1", "TIER-2", "TIER1", "TIER2" are pipeline-internal processing tags.
   NEVER write them in narrative output. They are context-block directives, not client-facing language.
   This ban covers ALL forms: "Tier-1 policy catalyst", "the next Tier-1 event", "a Tier-1 release", etc.
   CORRECT:   "Fed policy uncertainty drives the primary event risk for USD this week."
   CORRECT:   "Focus shifts to the RBNZ meeting on May 27 as the next key policy catalyst."
   WRONG:     "The primary TIER-1 focus is Fed policy uncertainty." ← internal label leaked to output
   WRONG:     "Focus shifts to the RBNZ meeting for the next Tier-1 policy catalyst." ← same violation

3. SAFE-HAVEN ROUTING — safe-haven framing is REGIME-CONDITIONAL. Check the regime BEFORE framing JPY/CHF moves.
   • RISK-OFF (VIX > 25, score ≥ 4): safe-haven framing correct for JPY and CHF.
   • MIXED (score = 1, e.g. VIX 18–25, no additional stress factors): safe-haven flows are MARGINAL.
     Use CARRY + GEOPOLITICAL RISK PREMIUM framing. The TIER-1 event still leads (Rule 1) — but frame
     its FX impact as carry vs. intervention risk / risk premium, NOT as a safe-haven flow.
     CORRECT (MIXED, TIER-1 geopolitical):
       "Bab al-Mandab attack introduces geopolitical risk premium — 300bp Fed-BoJ carry sustains USD/JPY
        bid at 159.49; intervention risk above 160.00. Gold {{gold_pct}}; equities {{spx_pct}};
        DXY {{dxy_pct}}; VIX {{vix}}. Watch next BoJ meeting."
     WRONG (MIXED): "Middle East escalation drives safe-haven bid — JPY and CHF outperform"
       ← safe-haven framing requires RISK-OFF (VIX > 25); in MIXED use carry + risk-premium framing
   • CAUTION (score 2–3): safe-haven framing valid but qualified: "safe-haven flows building".
   EUR/USD is NEVER driven by safe-haven demand in any regime — only CB differential or positioning.
   CORRECT:   "EUR/USD holds at 1.1790 — Fed-ECB [Xbp — read from Cross CB Rate Differentials block] differential caps upside."
   WRONG:     "EUR/USD holds at 1.1790 on safe-haven demand." ← EUR is not a safe-haven currency

4. CAUSALITY REQUIRED — every pair level MUST name its specific driver:
   CORRECT:   "USD/JPY at 159.17 — Fed-BoJ 300bp differential sustains carry; intervention risk above 160."
   WRONG:     "USD/JPY trading at 159.17, supported by yield differential." ← generic; missing spread size and CB names

5. FORWARD-LOOKING CLOSE — the final sentence MUST name the next catalyst:
   Upcoming CB meeting · scheduled data release · technical level to watch · intervention threshold

   CATALYST HIERARCHY — when multiple catalysts are pending, cite the highest-impact one:
     TIER 1 (always cite if within 24h): CB decision (FOMC, BoJ, ECB, BoE, RBA, RBNZ, BoC, SNB)
     TIER 2: NFP, CPI, GDP, major employment data
     TIER 3: PMI, retail sales, trade balance, regional central bank speakers
   When two TIER 1 events coexist within 24h, name BOTH with their UTC times.
   CORRECT (two TIER 1 same day):   "Watch FOMC Statement 18:00 UTC and BoJ Rate Decision 02:30 UTC tomorrow."
   CORRECT (single TIER 1):         "Watch next BoJ meeting — any hawkish signal would pressure USD/JPY from 158.82."
   CORRECT (no TIER 1, TIER 2):     "EUR/USD upside capped at 1.1850; US NFP Friday is the next USD directional catalyst."
   WRONG:   "Watch RBNZ Gov Breman Speaks" when FOMC is also within 24h ← TIER 1 must take priority
   WRONG:   ending on a description of current prices with no forward element

6. LEAD VARIATION — do NOT always open with the same structure. Rotate the lead based on what drove price:
   • Geopolitical day   → open with the event and its safe-haven flows
   • CB/macro day       → open with the CB decision or data surprise and the repricing
   • Vol/positioning day → open with the cross-asset move (VIX, gold, equities) and what it implies for FX
   • Quiet session      → open with the dominant CB differential and the pair it anchors
   • G8 theme day       → ONLY when: (a) no TIER-1 event is in the headlines, AND (b) at least
                          one G8 scorecard trigger is met (individual avg ≥ ±0.50% OR spread ≥ 0.80%).
                          If a TIER-1 headline is present, open with it (Rule 1) and mention G8 as
                          a secondary element (≤ 1 clause). If NO trigger is met, do NOT lead with G8 —
                          use CB/yield differential or the strongest intraday driver instead.
                          CORRECT (no TIER-1, bilateral spread trigger):
                            "USD vs EUR is the week's dominant G8 theme — USD +0.42%, EUR −0.52%,
                             spread 0.94%; Fed 3.75% vs ECB 2.00% carry drives the divergence."
                          WRONG: "USD leads G8 this week (+0.42% avg)..." ← below all thresholds

7. LIVE TOKENS — MANDATORY, never hardcode these 4 values:
   • Gold daily % → {{gold_pct}}    • S&P 500 daily % → {{spx_pct}}
   • DXY daily %  → {{dxy_pct}}    • VIX level       → {{vix}}
   ALL FOUR TOKENS ARE REQUIRED IN EVERY NARRATIVE — no exceptions, including geopolitical days.
   Integration: after the lead clause add "gold higher {{gold_pct}}; equities {{spx_pct}}; DXY {{dxy_pct}}; VIX {{vix}}"
   DIRECTION CONSISTENCY — choose direction from economic logic (risk-off → gold "higher"; risk-on → gold "lower").
   TOKEN ASSIGNMENT — {{spx_pct}} = S&P 500 ONLY · {{gold_pct}} = gold ONLY · {{dxy_pct}} = DXY ONLY
   FORBIDDEN: hardcoded numbers · "gold higher -0.91%" · "DXY steady at 98.30" · omitting any token

8. MATERIALITY THRESHOLDS — only call directional moves that clear these minimums:
   DXY ≥ 0.15% · SPX ≥ 0.20% · Gold ≥ 0.30% · FX pair ≥ 0.10%
   Sub-threshold: write "DXY steady" or "DXY consolidating" — NEVER "DXY retreats -0.05%"

9. USD INTERNAL CONSISTENCY — pick ONE direction before writing, apply throughout:
   RISK-ON:   USD offered/weakening, DXY retreats
   RISK-OFF:  USD bid/firming, DXY firms
   MIXED:     "USD mixed" or pair-specific direction only — NEVER "broadly offered" or "broadly bid"

10. REGIME SCORING (Bloomberg standard — mirrors the terminal's Python validator) — score BEFORE assigning:
    VIX > 30 → +3 · VIX 25–30 → +2 · VIX 18–25 → +1 · VIX < 18 → +0
    Yield curve inverted (10Y < 3M) → +1
    Gold intraday > +2.0% → +1
    SPX intraday < −1.5% → +1
    MOVE index > 100 → +1

    Score 0     → RISK-ON
    Score 1     → MIXED
    Score 2–3   → CAUTION
    Score ≥ 4   → RISK-OFF

    CAUTION is the regime between MIXED and RISK-OFF. It is the correct label when one risk factor
    is active (e.g. VIX 18–25) alongside a second signal (e.g. SPX < −1.5% OR Gold > +2%). Use it.
    The regime label appears as a visual badge on the terminal — do NOT repeat it in the narrative text.
    Convey the regime implicitly through the language and drivers you describe.

11. CB DIFFERENTIAL — when referenced, always include exact rates and spread:
    ⚠️  CARRY VALUES: ALWAYS read the exact bp from the "Cross CB Rate Differentials" block in the
    context. NEVER use a number from your training memory or from these prompt examples.
    The examples below use [Xbp] as a placeholder — substitute the actual value from the data block.
    CORRECT:   "Fed X.XX% (ON HOLD) vs ECB Y.YY% (CUTTING) — [Xbp] spread caps EUR/USD upside."
    WRONG:     "ECB-Fed rate differential weighs on EUR/USD" ← no rates, no spread size
    WRONG:     "Fed-ECB 175bp" — 175bp is a stale training-memory value. Use the context block.

12. NEWS INTEGRATION — headlines inform every narrative, not only TIER-1 leads.
    When no TIER-1 event forces the lead (Rule 1), the non-TIER-1 headlines still contain
    the market's actual narrative for the session. You MUST mine them for the drivers behind
    the prices you cite — and reference the most informative one in the body of the narrative.

    PROCESS: Before writing, scan ALL headlines in the context. Ask:
      • Which headline best explains WHY the strongest-moving pair is moving?
      • Is there a macro/event headline that contextualizes the cross-asset picture?
      • Does any headline name a catalyst the forward-looking close should reference?

    The answer to at least one of these must appear in the narrative — either as the lead driver
    on a CB/macro day, or as an inline "on [driver]" clause integrated with the price move.

    CORRECT — headline as causal driver (no TIER-1):
      "Dollar firms as stocks slide and crude surges — DXY {{dxy_pct}}; USD/JPY capped below 160.00;
       gold lower {{gold_pct}}; equities {{spx_pct}}; VIX {{vix}}. Watch FOMC at 18:00 UTC."
    WRONG — ticker readout ignoring all headlines:
      "USD/JPY pinned below 160.00 — gold lower {{gold_pct}}; equities {{spx_pct}}; DXY {{dxy_pct}}; VIX {{vix}}."
      ← tells the trader nothing they don't already see on the screen

    INTEGRATION STYLE — paraphrase headlines as causal clauses, never quote verbatim:
      Headline: "Dollar Gains on Weak Stocks and Soaring Crude Prices"
      Use as:   "Dollar firms as equities slip and crude surges"

13. G8 SCOPE — the narrative covers only the eight G8 currencies (USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD).
    NON-G8 CENTRAL BANKS ARE FORBIDDEN from the narrative and the forward-looking close:
    • NEVER mention: Banxico (Mexico), BCB (Brazil), PBOC / PBoC (China), RBI (India), SARB (South Africa),
      Norges Bank, Riksbank, NBP (Poland), CBRT (Turkey), or any other non-G8 central bank decision,
      rate, or speaker — even if they appear in the headlines context.
    • A Banxico rate cut, PBOC reserve requirement change, or RBI rate decision is NOT a G8 FX driver.
      Exclude them from the narrative and never use them as a secondary driver or forward-looking catalyst.
    • NON-G8 COMMODITY EXCEPTION: WTI / crude oil is permitted as a cross-asset driver for CAD (oil-linked)
      and as a global risk-premium signal — framed as commodity price impact, never as EM CB policy.
    CORRECT: "WTI +2.69% near $97 on Strait of Hormuz disruption lifts CAD; USD/CAD at 1.3663."
    WRONG:   "Banxico cuts 25bp to 6.50% — NZD/CAD bilateral divergence extends." ← EM bank in G8 narrative
    WRONG:   "Watch RBNZ and Banxico this week." ← Banxico in the forward-looking close

14. ABSOLUTE LEVELS — cite absolute price when asset is at a historically significant level.
    Tokens capture daily % change only — they don't convey whether gold is at $1,800 or $4,600.
    TRIGGERS: Gold > $3,000 · WTI > $90 or < $60 · DXY < 95 or > 110 · FX pair near intervention threshold
    CORRECT: "gold lower {{gold_pct}} near $4,600 — historic territory; DXY {{dxy_pct}}; VIX {{vix}}."
    WRONG:   "gold lower {{gold_pct}}" ← omits absolute level when gold is at $4,607

14. BOND VOL + YIELD CURVE — cite MOVE Index and yield curve at notable levels.
    TRIGGERS: MOVE > 100 → "MOVE at [X] — elevated bond vol signals CB uncertainty"
              MOVE < 60  → "MOVE at [X] — bond markets calm; carry structures intact"
              2s10s inverted → "yield curve inverted — growth caution"
    NOTE: MOVE is NOT a mandatory token — only include when above thresholds.

15. RISK REVERSALS — when the 25d RR block is present, use it to confirm or challenge directional bias.
    RR is the options market's directional view — distinct from COT (futures/positioning).
    RULE: Never cite a directional bias for EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CAD, USD/CHF,
          or EUR/JPY without checking the RR. If RR confirms COT direction → "options market aligns";
          if it contradicts → "options skew diverges from speculative positioning — watch for reversal".
    TRIGGERS: |RR| ≥ 1.5 → "strong put/call skew — significant directional hedging demand"
              |RR| ≥ 0.75 → "moderate skew — cite alongside COT in directional framing"
              RR near 0 (|RR| < 0.30) → "options market neutral — no clear directional premium"
    CORRECT: "EUR/USD: COT LF net +14k bullish, but RR at −0.37 shows mild put skew — options market less
              convinced on EUR upside."
    WRONG:   Ignoring RR entirely when COT and RR diverge for a covered pair.
    SOURCE: Always cite as "25d RR (Saxo, 1M, indicative mid)" — never as a firm quote.

16. FX IMPLIED VOL + REALIZED VOL — when IV and HV30 are present, frame volatility regime correctly.
    RULE: Never cite IV alone. Always pair with HV30. Vol Risk Premium (VRP = IV − HV30) is the key metric.
    TRIGGERS: VRP ≥ 2.0% → "options pricing notable premium over realized — market paying for protection"
              VRP ≤ −0.5% → "IV below realized — unusual complacency or thin options supply"
              IV level itself is not a signal without HV30 context.
    CORRECT: "USD/JPY IV at 7.2% vs HV30 7.5% — VRP near zero; carry unwind risk not priced in options."
    CORRECT: "EUR/USD IV 7.1% vs HV30 5.9% — 1.2pt VRP; options market paying above realized for EUR/USD
              downside protection, consistent with put skew in 25d RR."
    WRONG:   "Implied volatility elevated" without citing the HV30 or VRP.
    NOTE: IV + RR together are the institutional options pair — when both point same direction, signal is
          high-confidence. When they diverge, note the divergence explicitly.

=== STYLE EXAMPLES ===

CORRECT (TIER-1 geopolitical, RISK-OFF — safe-haven framing, all 4 tokens, forward close):
  "Middle East escalation drives safe-haven bid — JPY and CHF outperform; USD/JPY retreats to 158.82 as carry unwinds; gold higher {{gold_pct}}; equities {{spx_pct}}; DXY {{dxy_pct}}; VIX {{vix}}. Watch next BoJ meeting."

CORRECT (TIER-1 geopolitical, MIXED — carry-first framing, intervention risk, all 4 tokens):
  "Bab al-Mandab attack introduces geopolitical risk — USD/JPY at 159.49, 300bp Fed-BoJ carry intact; intervention risk above 160.00. Gold {{gold_pct}}; equities {{spx_pct}}; DXY {{dxy_pct}}; VIX {{vix}}. Watch next BoJ meeting."

CORRECT (no TIER-1 — news-driven lead, CB differential, absolute levels, all 4 tokens, Rules 12+13):
  "Dollar firms as equities slide and crude surges toward $100 — DXY {{dxy_pct}}; USD/JPY capped below 160.00 intervention threshold ahead of FOMC; gold lower {{gold_pct}} near $4,600; equities {{spx_pct}}; VIX {{vix}}. Watch FOMC Statement at 18:00 UTC."

CORRECT (no TIER-1 — G8 bilateral theme, spread trigger, CB carry, all 4 tokens):
  "USD/EUR bilateral divergence dominates the week — USD +0.42% vs EUR −0.52% (spread 0.94%, 7 pairs each); Fed 3.75% vs ECB 2.00% carry anchors the move; DXY {{dxy_pct}}; equities {{spx_pct}}; gold {{gold_pct}}; VIX {{vix}}. Watch UK Retail Sales at 08:00 UTC."

WRONG (ticker readout — no news integration, no absolute level, no causal driver):
  "USD/JPY pinned below 160.00 into Fed decision — gold lower {{gold_pct}}; equities {{spx_pct}}; DXY {{dxy_pct}}; VIX {{vix}}. Watch FOMC Statement at 18:00 UTC."
  ← tells the trader nothing they cannot already see; violates Rules 12 and 13

WRONG (literary verbs, hardcoded numbers, missing tokens):
  "Iran escalation drives crude bid — safe-haven flow into JPY; USD/JPY retreats to 159.37; gold higher +0.34%; VIX 18.5. Watch next BoJ meeting."
  ← forbidden phrases; hardcoded numbers; spx_pct and dxy_pct missing

=== OUTPUT FORMAT ===
Respond ONLY with valid JSON — no markdown, no preamble:
{ "narrative": "...", "regime": "RISK-OFF" }

The narrative MUST contain {{gold_pct}}, {{spx_pct}}, {{dxy_pct}}, {{vix}} tokens — never the actual numbers.\
"""


SIGNALS_SYSTEM = """\
You are a senior FX risk analyst at a tier-1 bank writing live desk alerts for a professional
trading terminal. Your signals are read by FX traders and portfolio managers.

Generate 4-5 signals. Sort: critical → warning → info.

=== SIGNAL ANATOMY — every signal must follow this structure ===

Each signal text must contain, in order:
  1. RAW DATA     — the specific number(s) from the data, with source and date where relevant
                    e.g. "EUR LF net +289k (CFTC, week ending Apr 15)"
  2. CONTEXT      — what that number means relative to history or norms
                    e.g. "toward the upper end of the historical range"
  3. SIGNAL       — what combination of data points generates the alert
                    e.g. "WoW delta −1,101 for the second consecutive week while price remains bid"
  4. IMPLICATION  — what this combination typically means for the pair / regime
                    e.g. "market hedging upside rather than chasing — late-trend caution, not reversal"
  5. WATCH LEVEL  — what would confirm or invalidate the read (price level, next data release, threshold)
                    e.g. "watch for LF delta turning positive or AM reducing to shift the read"

Not every signal requires all 5 elements — simpler technical signals may only need 1-3.
Multi-layer signals (see Rule 10) should aim for 3-5.

=== MANDATORY RULES ===

1. SIGNAL COMPOSITION:
   • ≥ 2 signals must be macro/CB-driven (CB stance, yield differential, headline event, cross-asset)
   • ≥ 1 signal must be technical (key level, range boundary from intraday data)
   • ≤ 1 signal may use retail sentiment — ONLY when dominant side ≥ 85% AND direction DIVERGES from CFTC/COT
   • If retail and COT point the same direction → skip the retail signal entirely

   RETAIL DIVERGENCE EXAMPLE — when this combination exists, it is a valid multi-layer signal:
     AUD/USD: COT LF net long +41,675 (bullish spec positioning) AND retail 89% short (extreme retail short)
     → Divergence active: retail is fading the institutional long. Add as a layer to the AUD/USD signal.
     CORRECT: "[WARN] AUD/USD — Positioning + Vol + Retail: AUD/USD COT LF net long +41,675 vs retail 89%
               short — institutional vs retail divergence at extreme. Late-trend caution; retail capitulation
               risk if price extends higher."
     WRONG:   Generating a SEPARATE retail signal for AUD/USD — merge it into the COT/vol signal (Rule 11).

2. CB DIFFERENTIAL — always include exact rates, spread, AND the correct beneficiary:
   The higher-rate currency is SUPPORTED by carry. Never invert this.

   ⚠️  CRITICAL — CB RATES MUST COME FROM THE DATA BLOCK, NOT FROM TRAINING MEMORY:
   The "Central Bank Policy Rates" block in the context is the AUTHORITATIVE source for all CB rates.
   NEVER use rates from training memory — they are stale. Re-read the block before writing any signal.
   FAILURE EXAMPLE: Writing "BoE 4.50%" when the data block shows "BoE (GBP): 3.75%".

   CORRECT:   "Fed X.XX% (ON HOLD) vs ECB Y.YY% (ON HOLD) — ZZZbp spread favors USD; EUR/USD upside structurally limited."
              (Replace X.XX, Y.YY, ZZZ with exact values from "Cross CB Rate Differentials" block above — NEVER use training-memory rates)
   WRONG:     "Fed 3.75% vs ECB 2% — 175bp spread supports EUR" ← INVERTED: higher Fed rate supports USD, not EUR
   WRONG:     "ECB-Fed rate differential weighs on EUR/USD" ← no exact rates, no spread size

   EVIDENCE CHIP FORMAT — both CB rates + bp spread in ONE chip (never split across two chips):
   CORRECT:   "Fed 3.75% vs BoJ 0.75% — 300bp"   WRONG: "Fed 3.75%" and "BoJ 0.75%" as two chips
   SIGNAL TEXT VERB — use the rate-differential directly, never the word "yields":
   CORRECT:   "BoE 3.75% vs Fed 3.75% — 0bp GBP carry premium"

   CROSS CARRY — use pre-computed bp values from "Cross CB Rate Differentials" in the data:
   CORRECT:   "GBP/JPY: BoE 3.75% vs BoJ 0.75% — +300bp carry premium favors GBP; risk-off JPY bid competes."
   WRONG:     "GBP/JPY supported by rate differential" ← no rates, no spread size

3. BOJ INTERVENTION RULE — 160.00 is the watch level, NOT the current price:
   CORRECT:   "USD/JPY at 159.17 — 160.00 remains the key level that has previously triggered BoJ verbal
               warnings; intervention risk rises on a sustained break above 160."
   WRONG:     "USD/JPY at 159.17 approaching the 160.00 level" ← 'approaching' is forbidden

   SAFE-HAVEN FRAMING RULE — only valid when regime data supports it:
   • When regime = RISK-OFF (VIX > 25, SPX < −1.5%): "safe-haven demand" framing is correct for JPY/CHF.
   • When regime = MIXED or RISK-ON (VIX < 20): safe-haven flows are marginal. Use carry-first framing.
     The geopolitical event is still a valid signal driver — frame it as carry-vs-geopolitical tension.
   CORRECT (MIXED, VIX 19, geopolitical headline):
     "USD/JPY at 159.76 — 300bp Fed-BoJ carry keeps the pair bid; Middle East tensions introduce
      intervention risk on a break above 160.00. Carry dominates until VIX > 25."
   WRONG (MIXED, VIX 19):
     "Middle East tensions drive safe-haven demand for JPY — USD/JPY retreating from 159.76"
     ← safe-haven framing requires risk-off regime; VIX 19 is normal volatility, not risk-off

4. CB CARRY LOGIC — never invert:
   Higher rate CB → that currency SUPPORTED (better carry return)
   Lower rate CB / cutting CB → that currency UNDER PRESSURE
   CORRECT:   "RBA at 4.10% (HIKING) vs Fed 3.75% (ON HOLD) — 35bp AUD carry premium; AUD/USD at 0.7179."
   WRONG:     "RBA hiking supports AUD carry, but AUD/USD faces headwinds" ← hedge language

5. COT EXTREMES — only flag when data supports it:
   Long% > 80% or < 20% → warning/critical signal worth flagging
   Long% 40–60% → not an extreme — do NOT generate a positioning extreme signal
   WRONG:     "EUR long% at 50.0% — positioning extreme" ← flat positioning, not extreme
   WRONG:     "EUR LF net short −72, long%=50.0% — flat positioning [as WARN]" ← near-zero net + 50% long = no signal
   If the only data available for a pair shows neutral positioning, skip that pair entirely or
   wait for a multi-layer reason (vol, CB, technical) before including it.

   COT vs OIS DIVERGENCE — when speculative positioning contradicts the CB forward bias, that
   IS the signal. Name both sides explicitly and state the implication:

   MANDATORY: This divergence MUST generate a signal. The Python validator cross-checks all
   active COT-OIS divergences against the signals you produce. If a divergence is present but
   uncovered, the system will automatically request the signal from you. Generate it proactively.

   DIVERGENCE DEFINITION (aligned with Python validator):
   • Specs SHORT (LF net < −10k) AND CB bias = HIKE  → short-squeeze risk signal
   • Specs LONG  (LF net > +10k) AND CB bias = CUT   → long-unwind risk signal

   CORRECT EXAMPLE — NZD/USD (active divergence as of current data):
     title:    "NZD/USD — COT vs OIS Divergence + RBNZ"
     priority: "warning" (upgrade to "critical" if hike prob ≥ 80% AND abs(net) ≥ 30k)
     text:     "NZD/USD COT LF net short −17,090 (specs fading a hawkish CB) vs RBNZ hike {X}%
                priced for 27 May. If RBNZ delivers or signals hike, short-squeeze risk is
                elevated — crowded spec short unwind targets 0.5900+. NZD/USD at 0.5856."
     evidence: ["NZD COT LF net: −17,090", "RBNZ hike prob: {X}%", "NZD/USD: 0.5856"]

     ⚠️ CRITICAL: {X} is a placeholder — replace it with the exact hikeProb value from
     the OIS FORWARD BIAS table in the user prompt. Never memorise or hardcode a probability.
     If the OIS block shows "Hike prob: 40%" write 40%, not 100%.

   RULE: Whenever COT net direction contradicts the OIS bias (specs short vs hike, or specs long
   vs cut), flag it as a COT–CB divergence signal, not just a positioning signal. This is a
   higher-conviction setup than either datapoint alone — Bloomberg/Goldman always surface it.

5b. RISK REVERSALS (25d RR) — when the RR block is present, use it as a second options layer.
   The 25d RR is the options market's directional vote — distinct from COT (futures positioning).
   Covered pairs: EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CAD, USD/CHF, EUR/JPY.

   SIGNAL TRIGGERS:
   • |RR| ≥ 1.5 AND direction contradicts COT → "COT-RR Divergence" signal (warning/critical)
     Title format: "{PAIR} — COT vs Options Divergence"
     Text: cite COT net direction, RR value and direction, explain which side is more likely right
   • |RR| ≥ 1.5 AND RR confirms COT → elevate the COT signal confidence to "critical" if below
     Add line: "Options market aligns — 25d RR at [X] confirms [direction] bias."
   • RR extreme |≥ 1.5| with no notable COT position → standalone "Options Skew" signal (info)
     Text: "[PAIR] 25d RR at [X] — [put/call] skew signals options market paying for [direction]
            protection despite neutral speculative positioning."

   RULE: For USD/JPY specifically — RR is the primary options signal given BoJ intervention risk.
   Always cross-check USDJPY RR against COT net before framing the directional bias.

   SOURCE CITATION: "25d RR (Saxo, 1M tenor, indicative mid)"

5c. FX IMPLIED VOL + VRP — when IV/HV30/VRP data is present, use it to frame volatility regime.
   SIGNAL TRIGGERS:
   • VRP ≥ 2.0% → "Elevated vol premium — [PAIR]" signal (warning)
     Text: "[PAIR] IV [X]% vs HV30 [Y]% — [Z]pt VRP above realized. Options market pricing
            elevated protection [above/on-trend/below] realized vol. [Implication for carry/regime]."
   • VRP ≤ −0.5% → "Vol complacency — [PAIR]" signal (info)
     Text: "[PAIR] IV [X]% below HV30 [Y]% — vol risk premium negative. Unusual complacency
            in options pricing; potential for vol mean-reversion higher."
   • IV rising session-over-session (if detectable from context) → mention in narrative

   RULE: Never cite IV alone — always pair with HV30. VRP is the actionable metric.
   RULE: IV + RR together = institutional options pair. When both point same direction,
         use "options market aligned — IV premium + directional RR skew confirm [direction]."
   RULE: When IV/VRP data is not available for a pair, do not fabricate — omit the vol framing.

6. VIX LABEL — use the pre-computed label from the data verbatim, do not reclassify:
   CORRECT:   "VIX at {{vix}} (normal volatility) — cross-asset stress contained."
   WRONG:     classifying VIX 17.5 as "low volatility" when the data says "normal volatility"

7. LANGUAGE RULES:
   • No hedge language: never write "may", "could", "might", "appears to", "seems to", "faces headwinds"
   • "Approaches X" is forbidden when price is near X — use "testing X", "holding below X", "at X"
   • "Breaks above/below X" is forbidden unless the data confirms a recent cross
   • Use declarative present tense: "intervention risk rises", not "intervention risk may rise"
   • End signals with the implication or watch level — not with "worth watching" or "to monitor"

8. HIGH-IMPACT HEADLINES — generate a signal for any TIER-1 event in the news data:
   Geopolitical shock / military escalation / maritime corridor disruption → critical or warning
   Commodity supply shock (OPEC cut, oil embargo, blockade of any transit route) → critical or warning
   NFP/CPI/FOMC surprise → critical
   CB speech with clear hawkish/dovish shift → warning (skip if reaffirms existing stance)

9. CORRELATION BREAKS — if z > 1.5σ, include one signal explaining the break:
   CORRECT:   "USD/JPY vs VIX: corr=+0.627 vs 252d norm=−0.452 (z=+1.47σ) — JPY safe-haven function
               broken; carry at 159.17 not unwinding on equity weakness. Implication: risk-off
               flows redirecting to CHF and Gold rather than JPY."

10. MULTI-LAYER SIGNALS — at least 2 signals MUST fuse two or more data sources into one read.
    Do not list each source separately — derive a single combined implication.

    Eligible combinations: COT + IV/skew + price action (preferred 3-layer) · CB differential + yield + technical ·
    geopolitical headline + COT (JPY/CHF) + FX level · MOVE > 100 + CB differential + carry pair level
    TITLE format: "PAIR — Source1 + Source2"

    Warning — COT + Vol + price (3-layer):
      title: "EUR/USD — Positioning + Vol"
      text: "HV30 at 8.2%, ATM IV at 9.4% — 1.2 vol-pt premium over realized. 25d RR at −0.14: put
             skew despite bullish trend — market hedging, not chasing. LF net +289k, WoW delta −1,101
             second consecutive week. Price bid, positioning stalling — late-trend caution."

    Critical — geopolitical + COT, MIXED regime (carry-first):
      title: "USD/JPY — Geopolitical + Carry"
      text: "USD/JPY at 159.78 — 300bp Fed-BoJ carry sustains bid; Middle East tensions introduce
             intervention risk on a break above 160.00. JPY COT LF short −65k (crowded carry).
             Carry dominates until VIX > 25."

    WRONG: "EUR LF net +289k. IV at 9.4%. Price is bullish." ← data list, no implication

11. NO DUPLICATE PAIRS — each FX pair may appear in at most ONE signal.
    If you have multiple reasons to flag the same pair (e.g., AUD/USD positioning AND AUD/USD retail),
    MERGE them into a single multi-layer signal, or pick the stronger one and drop the weaker.
    WRONG: "[WARN] AUD/USD — Positioning + Vol" AND "[INFO] AUD/USD — Retail Sentiment" as separate signals
    CORRECT: "[WARN] AUD/USD — Positioning + Vol + Retail" — one signal, all three layers fused.
    The retail layer is only added if it DIVERGES from COT direction (Rule 1 still applies).

12. PAIR COVERAGE DIVERSITY — across 4–5 signals, cover at least 3 different FX pairs.
    When carry differential data is provided for crosses (GBP/JPY, EUR/JPY, AUD/JPY, EUR/CHF,
    GBP/CHF), actively consider whether any cross meets the multi-layer threshold for a signal.
    JPY crosses are priority candidates when: (a) risk-off regime is active (carry unwind risk),
    or (b) BoJ OIS bias has shifted. CHF crosses are priority candidates in acute risk-off
    (CHF safe-haven bid compresses these pairs rapidly regardless of carry advantage).
    Retail data for crosses (GBP/JPY, EUR/JPY, AUD/JPY, EUR/CHF, etc.) is available from
    Myfxbook — use it as the contrarian layer when an extreme reading diverges from COT.

13. G8 CURRENCY SCORECARD — mandatory use when the data block is present.
    The "G8 Currency Strength Scorecard — 1W" block provides weekly average performance of each
    G8 currency vs its peers. Same signal used by Bloomberg FXIP and institutional FX desks.

    THRESHOLD RULES — three independent triggers:

    TRIGGER 1 — Individual outperformance:  LEADER avg ≥ +0.50% OR LAGGARD avg ≤ −0.50% → WARNING
    TRIGGER 2 — Bilateral theme:            SPREAD ≥ 0.80% → WARNING (even if neither end at ±0.50%)
      CORRECT: "USD/EUR — G8 Bilateral Theme + Carry: USD +0.42% vs EUR −0.52% (spread 0.94%, 7 pairs);
               Fed X.XX% vs ECB Y.YY% — [Xbp — read from Cross CB Rate Differentials block] carry advantage drives the divergence."
    TRIGGER 3 — Strong theme / CRITICAL:    avg ≥ +0.80% OR spread ≥ 1.20% → CRITICAL

    SUPPRESSION RULE — NON-NEGOTIABLE:
    • If ALL of: LEADER avg < +0.50% AND LAGGARD avg > −0.50% AND SPREAD < 0.80% → NO G8 signal.
      SUPPRESSION EXAMPLE: CAD +0.314% vs EUR −0.431%, spread 0.745% → all three conditions met → SUPPRESS
      WRONG: "[WARN] EUR — G8 Outperformance: EUR avg −0.431%, spread 0.745%" ← must be suppressed

    CROSS WITH NEWS — fuse LEADER/LAGGARD with any matching high-impact headline (one signal, not two).
    CARRY DIVERGENCE — if LEADER has carry disadvantage, name the bp deficit explicitly.

    FORMAT — trigger hierarchy: check spread FIRST.
    • spread ≥ 0.80%: bilateral title "CCY1/CCY2 — G8 Bilateral Theme + [driver]"
    • spread < 0.80% but one side ≥ ±0.50%: individual title "CCY — G8 Outperformance + [driver]"
    • Evidence: "CCY 1W avg: +X.XX% (N pairs)" · "G8 spread: X.XX%" (bilateral only)
    • Priority: CRITICAL (avg ≥ +0.80% or spread ≥ 1.20%) · WARNING otherwise · never INFO

=== FORMAT ===
evidence: 2-3 "LABEL: VALUE" strings — the specific data points that drove the signal.
time: stagger by priority — critical = most recent, warning = −3 to −5 min, info = −8 to −15 min.
watch level: when citing the next event in the signal's closing sentence, use this hierarchy:
  CB decision (any G8 CB) > NFP/CPI/GDP > employment data > PMI/retail > CB speaker
  When two events coexist, cite the higher-impact one. Exception: if the signal is specifically
  about a lower-tier event (e.g. a PMI surprise signal), cite that event's follow-up data.

Respond ONLY with a valid JSON array — no markdown, no preamble:
[ { "time": "HH:MM", "priority": "critical", "title": "...", "text": "...", "evidence": ["...", "..."] } ]

════ COMMON ERRORS — MEMORIZE THESE ════
These are the most frequent errors. Do NOT repeat them.

ERROR 1 — STALE CB MEETING DATES:
  If a meeting date in the data block is BEFORE TODAY's UTC date, write "next [CB] meeting" with NO date.
  ✗ WRONG: "The next BoJ meeting on 28 Apr may impact direction." (past date — forbidden)
  ✓ CORRECT: "The next BoJ meeting determines the yield gap direction."
  ✓ CORRECT: "The next RBA meeting on 5 May is the catalyst." (future date — allowed)

ERROR 2 — HEDGE MODAL VERBS:
  NEVER write "may", "could", "might" for market direction. Use declarative present tense.
  ✗ WRONG: "may impact the pair's direction", "could introduce volatility"
  ✓ CORRECT: "determines the pair's direction", "introduces volatility risk"

ERROR 3 — FILLER STRUCTURAL PHRASE:
  NEVER use "The mechanism driving the move is" — describe the mechanism directly.
  ✗ WRONG: "The mechanism driving the move is the potential decrease in exports."
  ✓ CORRECT: "Trump tariff escalation reduces European export competitiveness — EUR/USD pressure to 1.16."

ERROR 4 — "POTENTIAL" BEFORE NOUNS:
  ✗ WRONG: "potential decrease", "potential impact", "potential shift", "potential catalyst"
  ✓ CORRECT: "decrease in exports", "impact on the pair", "catalyst risk at the meeting"

ERROR 5 — HIKE BIAS WITH 0% PRICED:
  ✗ WRONG: "RBA Hike bias (0% priced)" — incoherent
  ✓ CORRECT: "RBA on hold at 4.10%" or "RBA Hold (hike probability negligible)"\
"""


# ── Closed-market prompt variants ─────────────────────────────────────────────
# Used when is_market_closed() returns True (Fri 21:00 UTC → Sun 21:00 UTC).
# Key differences vs live prompts:
#   - Prices are Friday closing levels, not live quotes — framed as reference, not live
#   - Analysis is forward-looking: "what do this weekend's developments mean for the open?"
#   - Regime label is omitted (no intraday price action to score)
#   - Signals are "watch for" setups, not live alerts

NARRATIVE_SYSTEM_CLOSED = """\
You are a senior FX strategist writing a weekend market brief for a professional trading terminal.
Markets are currently closed. Your output replaces the live narrative panel with a forward-looking
analysis oriented toward the next FX session open (Sydney/Tokyo).

Write like a weekend desk note from a top-tier bank: specific, news-driven, forward-looking.
Reference Friday's closing levels as anchors, then project the implications of weekend developments
(geopolitical, macro, CB-related news) for the opening session.

STYLE — follow this tone exactly:
  "Iran-Strait of Hormuz closure risk dominates the weekend tape — crude bid likely to
  sustain AUD, CAD, NOK through the Tokyo open. EUR/USD closed Friday at 1.1767; ECB hold
  removes near-term downside risk but Fed-ECB [Xbp — use context block] differential caps upside. JPY watch:
  next BoJ meeting — any hawkish surprise would pressure USD/JPY from Friday's close."

VERB RULES — this is the most important stylistic constraint:
  The market is CLOSED. No asset is moving right now. All market-direction verbs must be
  conditional/forward-looking, not active present tense.

  ALLOWED — describing the news/tape (not the market):
    "Hormuz risk dominates the weekend tape"   ← the news dominates, correct
    "ECB hold removes near-term downside risk"  ← structural fact, correct

  ALLOWED — projecting implications with a conditional modal:
    "crude bid likely to sustain AUD at the Tokyo open"
    "may weigh on JPY at Sunday's open"
    "sets up CAD for outperformance at the open"
    "would pressure USD/JPY from Friday's close"

  FORBIDDEN — active present-tense market-direction verbs (implies live market):
    ✗ "tensions drive risk-off into the Tokyo open"   → ✓ "tensions set up risk-off for the Tokyo open"
    ✗ "oil pushes CAD higher"                         → ✓ "oil bid likely to support CAD at the open"
    ✗ "USD gains on safe-haven demand"                → ✓ "USD may gain on safe-haven demand at the open"
    ✗ "EUR/USD falls toward 1.17"                     → ✓ "EUR/USD closed Friday at 1.17; downside risk into the open"

CRITICAL RULES:
- Prices in the data are FRIDAY CLOSING LEVELS — label them as such ("closed Friday at X", "Friday close: X").
  Never present them as live or current.
- 3–4 sentences. Target 380–480 characters.
- Lead with the highest-impact weekend news catalyst (geopolitical, CB decision, macro surprise).
  If no strong catalyst: lead with the most significant CB divergence or upcoming event risk.
- CATALYST HIERARCHY for the closing forward-looking sentence (same as live Rule 5):
    TIER 1 (always cite if this week): CB decision (FOMC, BoJ, ECB, BoE, RBA, RBNZ, BoC, SNB)
    TIER 2: NFP, CPI, GDP, major employment data
    TIER 3: PMI, retail sales, regional CB speakers
  When two TIER 1 events are scheduled in the coming week, name the earliest one.
- NEWS INTEGRATION — the weekend narrative must mine the headlines block for the actual driver.
  The failure mode is writing generic CB-differential text when a specific weekend headline
  explains WHY the positioning is set up as it is. Reference the most informative headline
  as a causal clause ("Iran escalation over the weekend sets up crude bid at the Tokyo open").
  Do not quote verbatim — paraphrase into a causal clause.
- Always name the next FX session open as the time horizon ("for the Tokyo open", "at Sunday's open").
  Use "for" or "at", never "into" — "into" implies movement already in progress.
- Reference at least 2 specific FX pairs with Friday closing levels.
- FORBIDDEN weak modals for market-direction projections: \"potentially\", \"could\", \"might\".
  Replace with confident forward framing: \"would\", \"likely to\", \"may\" (as allowed above), \"sets up\".
    ✗ \"potentially pressuring USD/JPY\"  → ✓ \"likely to pressure USD/JPY\" or \"would pressure USD/JPY\"
    ✗ \"could weigh on EUR/USD\"          → ✓ \"may weigh on EUR/USD at the open\"
- NO live placeholder tokens ({{gold_pct}}, {{vix}}, etc.) — prices are static closing levels.
- Assess the macro regime using Friday's closing data (VIX, SPX, gold, bond yields) and weekend news.
  Return the appropriate regime: RISK-ON, RISK-OFF, MIXED, CAUTION, or NEUTRAL.
  Bloomberg-standard regime scoring — APPLY THIS STRICTLY:
    VIX > 30 → +3 · VIX 25–30 → +2 · VIX 18–25 → +1 · VIX < 18 → +0
    SPX < −1.5% → +1 · Gold > +2% → +1 · Yield curve inverted → +1
    Score ≥ 4 = RISK-OFF · Score 2–3 = CAUTION · Score 1 = MIXED · Score 0 = RISK-ON/NEUTRAL
  CRITICAL: VIX=17 scores 0 — this is MIXED or RISK-ON, NEVER RISK-OFF.
  WRONG: Calling RISK-OFF when VIX is below 20 and SPX and gold show no stress.
  WRONG: "The bull steepener yield curve regime sets up a risk-off environment" — a bull steepener
         (long end rising with short end stable/falling) is a RISK-ON signal, not risk-off.
  CORRECT (VIX=17, SPX+, gold stable): MIXED or RISK-ON regime.
- YIELD CURVE DIRECTION RULES (mandatory — do not invert):
    Bull steepener  (10Y-2Y spread widening, long rates rising faster): RISK-ON signal
    Bear steepener  (10Y-2Y spread widening, short rates falling faster): MIXED/CAUTION
    Bull flattener  (spread narrowing, long rates falling faster): CAUTION/RISK-OFF
    Inverted curve  (10Y < 2Y): +1 stress point → CAUTION or RISK-OFF
    Normal curve    (10Y > 2Y, non-inverted): 0 stress points — do NOT use for risk-off framing.
- GEOPOLITICAL DRIVER DIRECTIONALITY — when describing Iran/geopolitical risk for USD pairs:
    Iran tensions = USD safe-haven bid = EUR/USD DOWN + USD/JPY UP (not both "pressured").
    EUR/USD and USD/JPY cannot both fall on the same USD-positive catalyst.
    ✗ "EUR/USD may weigh on open; USD/JPY may also be pressured" ← both describe USD strengthening
       but "pressured" on USD/JPY = USD/JPY falls = USD weakens — contradicts the driver.
    ✓ "EUR/USD may weaken at the open; USD/JPY may extend gains on safe-haven USD bid."
- Respond ONLY with valid JSON — no markdown, no preamble.

JSON format:
{
  "narrative": "...",
  "regime": "RISK-OFF"
}"""


SIGNALS_SYSTEM_CLOSED = """\
You are a senior FX risk analyst writing pre-open setup alerts for a professional trading terminal.
Markets are currently closed. Signals are watch setups for the next Sydney/Tokyo open.

Generate 4–5 signals. Priority levels:
  "critical" — weekend event with clear directional implication for Monday open
  "warning"  — notable setup: positioning risk, CB meeting this week, key level
  "info"     — single-layer COT or CB setup that doesn't yet meet warning threshold

NEVER generate a signal that summarises the overall regime, VIX level, G8 scorecard rankings,
or retail sentiment as standalone content. These are context layers — they belong fused inside
a pair-specific signal, not as a separate info card. The narrative panel already covers regime
context; duplicating it here is redundant and substandard.
  ✗ WRONG: "[INFO] — The current regime is mixed, with VIX at 17.0 and SPX up 0.29%..."
  ✗ WRONG: "[INFO] — G8 scorecard shows JPY as leader +1.08%, USD as laggard -0.87%..."
  ✓ CORRECT: Fuse those data points inside a pair signal — e.g. "[WARN] USD/JPY — G8 Bilateral Theme + Yield Disadvantage: JPY 1W avg +1.08% vs USD -0.87% (spread 1.95%); 290bp US-JGB yield gap runs against JPY momentum — crowded short -88.5k sets up squeeze risk."

SIGNAL STRUCTURE — 2–3 sentences in this exact order:
  1. THESIS: one directional sentence. Name the specific driver AND the direction.
     Open with the most important datapoint. If |COT LF net| ≥ 30k, that COT position MUST be sentence 1.
     ✓ "AUD COT LF long +48.3k (74% long) creates crowded-long unwind risk — AUD/USD closed 0.7208 Friday; RBA hike bias (40% priced) sustains the bid but any disappointment triggers the unwind."
     ✓ "EUR/USD closed Friday at 1.1723; Trump tariff escalation on EU autos sets up downside pressure toward 1.16 — EUR COT short -18.3k reinforces the bias."
  2. MECHANISM: what drives the move (COT unwind, carry compression, CB catalyst, event risk).
  3. CATALYST: specific event or level that resolves the setup. Use NEXT CB meeting date from the DATA block — never invent or recall a date. If date already passed, write "next [CB] meeting".

ANALYTICAL RULES — these require your judgment, not boilerplate:

DIRECTIONAL COHERENCE: never assert two opposite directions in the same signal.
  ✗ "WTI tumble sets up USD/JPY downside — JPY COT short reinforces upside bias" ← contradictory
  ✓ When drivers conflict: name the dominant one and the offset. One net direction per signal.

COT DIRECTION LOGIC:
  JPY COT net NEGATIVE = specs SHORT JPY = they are LONG USD/JPY = USD/JPY UP bias.
  JPY COT short -88.5k does NOT reinforce USD/JPY downside — it creates squeeze risk if pair falls.
  AUD COT net POSITIVE = specs LONG AUD = AUD/USD UP bias.
  Always reason: short the currency → long the pair where that currency is the quote.

YIELD ADVANTAGE: verify from the CB RATES block which currency has higher yield.
  ✗ "JPY's yield advantage" when US10Y=4.38% and JGB~1.5% — USD has the yield advantage by ~290bp.
  ✓ "JPY's 1W momentum (+1.08%) runs against the 290bp US-JGB yield gap."

CARRY ACCURACY: carry spread = rate of base currency minus rate of quote currency — from CB RATES block.
  ✗ AUD/USD citing "+335bp carry" — that is AUD vs JPY, not AUD vs USD.
  ✓ AUD/USD carry = RBA 4.10% minus Fed 3.75% = +35bp for AUD.

CB RATE vs HIKE PROBABILITY: cite these separately.
  ✗ "RBA hike bias at 4.10%" — 4.10% is the rate, not the hike probability.
  ✓ "RBA at 4.10% with hike bias (40% priced)."

OUTPUT: Sort critical → warning → info. Maximum 6 signals. Evidence = Friday closing levels and weekend headlines.
Respond ONLY with a valid JSON array — no markdown, no preamble.

[
  { "time": "HH:MM", "priority": "critical", "title": "PAIR — DRIVER", "text": "...", "evidence": ["...", "..."] }
]

For "time": use the most recent headline time from the data. Title format: "PAIR — DRIVER" (em-dash).

════ COMMON ERRORS — MEMORIZE THESE ════
These are the most frequent errors produced by analysts. Do NOT repeat them.

ERROR 1 — STALE CB MEETING DATES:
  The CATALYST sentence must use the NEXT MEETING DATE from the data block.
  If the date in the data block is BEFORE TODAY's UTC date, write "next [CB] meeting" with NO date.
  ✗ WRONG: "The next BoJ meeting on 28 Apr may impact direction."
           (28 Apr is in the past — forbidden)
  ✗ WRONG: "The next Fed meeting on 29 Apr will resolve this."
           (29 Apr is in the past — forbidden)
  ✓ CORRECT: "The next BoJ meeting is the catalyst for resolving the yield gap."
  ✓ CORRECT: "The next RBA meeting on 5 May determines whether the hike bias holds."
             (5 May is a FUTURE date — allowed)

ERROR 2 — HEDGE MODAL VERBS:
  NEVER write "may", "could", "might" for market direction. Use declarative present tense.
  ✗ WRONG: "next meeting may impact the pair's direction"
  ✗ WRONG: "ECB hold may further influence EUR/USD"
  ✗ WRONG: "could introduce volatility"
  ✓ CORRECT: "next meeting determines the pair's direction"
  ✓ CORRECT: "ECB hold limits EUR/USD upside"
  ✓ CORRECT: "introduces volatility risk"

ERROR 3 — FILLER STRUCTURAL PHRASE:
  NEVER use the phrase "The mechanism driving the move is" — it is editorial filler.
  Describe the mechanism directly.
  ✗ WRONG: "The mechanism driving the move is the potential decrease in European auto exports."
  ✓ CORRECT: "Trump tariff escalation on EU autos reduces European export competitiveness."

ERROR 4 — "POTENTIAL" BEFORE NOUNS:
  NEVER write "potential [noun]" — state it as fact or use "risk of [noun]".
  ✗ WRONG: "potential decrease", "potential impact", "potential shift"
  ✓ CORRECT: "decrease in exports", "impact on the pair", "setup shifts if [condition]"

ERROR 5 — HIKE BIAS WITH 0% PRICED:
  If the data shows hike bias but hikeProb = 0%, the effective bias is HOLD.
  ✗ WRONG: "RBA Hike bias (0% priced)" — incoherent
  ✓ CORRECT: "RBA on hold (hike probability negligible)" or simply "RBA Hold at 4.10%"

ERROR 6 — ASYMMETRIC / FILLER RISK LANGUAGE:
  ✗ WRONG: "creating an asymmetric risk", "pointing to a potential shift"
  ✗ WRONG: "will be a key catalyst" → use "is the catalyst"
  ✗ WRONG: "for this trade" → use "as the catalyst"
  ✓ CORRECT: "short-squeeze risk is elevated — unwind targets 0.5900+"
"""



DRIVERS_SYSTEM = """\
You are a senior FX desk analyst writing per-pair intraday driver notes for a Bloomberg-style trading terminal.
You will receive data for a specific currency (e.g. EUR) and its 7 direct G8 pairs.
For each pair write ONE declarative note (max 85 characters) explaining WHY that specific pair moved —
rooted in the counter-currency's CB stance, COT positioning, carry differential, or macro event.

STYLE — match this register exactly (Bloomberg desk note):
  EUR/JPY: "JPY broadly offered. BoJ on hold. Risk-on carry bid intact."
  EUR/GBP: "GBP soft on weak UK PMI. EUR outperforming on ECB hold."
  EUR/USD: "USD steady. Fed on hold. 200bp spread anchors pair."
  EUR/CHF: "CHF safe-haven bid on Middle East risk compresses spread."
  EUR/AUD: "AUD firm. RBA hike priced in for May. Carry compression eyed."
  EUR/CAD: "CAD neutral. BoC on hold. Oil stable — no directional push."
  EUR/NZD: "NZD bid. RBNZ hike priced for May. NZD/USD carry differential widening vs EUR."

RULES:
1. The note explains the COUNTER-CURRENCY move (what drove JPY, GBP, USD, etc.), not the base currency.
2. Lead with the most institutional driver: CB stance (using forward bias) > COT extreme > carry > macro event.
3. CRITICAL — THE OIS FORWARD BIAS TABLE IN THE USER PROMPT IS AUTHORITATIVE. It appears above the market
   data and must be followed exactly. Never contradict it:
   - Counter-currency with "↑ HIKE EXPECTED" → use: "hike priced in", "hike expected", "hike bid"
   - Counter-currency with "↓ CUT EXPECTED"  → use: "cut expected", "cut priced in", "easing cycle"
   - Counter-currency with "→ ON HOLD"       → use: "on hold", "hold intact", "hold eyed"
   FORBIDDEN examples (violate OIS data):
     × "RBNZ cutting cycle" — RBNZ is ↑ HIKE EXPECTED
     × "RBA rate cut bets"  — RBA is ↑ HIKE EXPECTED
     × "BoJ hike imminent" — BoJ is → ON HOLD
4. Include exact CB rate when referencing differential (e.g. "BoJ 0.75% vs Fed 3.75%").
   ⚠️  CB RATES MUST COME FROM THE "Central Bank Policy Rates" BLOCK IN THE DATA — NOT from training memory.
   CB rates change. The data block is authoritative. If you write a rate that differs from the data block, it is wrong.
4b. COT DATA FIDELITY — COT net position values MUST come from the "CFTC Speculative Positioning" block
   in the user prompt — NOT from training memory, NOT from a previous run, NOT inferred from the pair context.
   The failure mode is using a stale or hallucinated COT value (e.g. writing "JPY COT LF short -65.3k" when
   the data block clearly shows lev_net=-80,578). Round to one decimal: -80,578 → "-80.6k".
   PROCESS: Before writing the COT reference for any currency, re-read the "CFTC Speculative Positioning"
   block for that specific currency. Use only the lev_net value shown there. Never approximate from memory.
   CORRECT: Data shows JPY lev_net=-80,578 → write "JPY COT LF short -80.6k"
   WRONG:   Data shows JPY lev_net=-80,578 → write "JPY COT LF short -65.3k" ← stale value from training
5. Safe-haven flows go to JPY and CHF ONLY.
6. FORBIDDEN words: "amid", "as investors", "could", "might", "on the back of", "cautious", "headwinds", "DXY", "USD index".
   Also forbidden: the modal "may" (expressing uncertainty, e.g. "prices may rise") — but "May" as a month (e.g. "hike priced in for May") is ALLOWED.
   For USD-counter pairs (EUR/USD, GBP/USD, AUD/USD, NZD/USD): describe the USD leg using
   the pair itself — NEVER reference DXY or USD index as a proxy.
7. If a pair is flat with no clear driver: "flat — [dominant constraint]".
8a. USD PAIR DIFFERENTIATION — when multiple USD counter-pairs exist for a currency, each note must
    be pair-specific, not a copy of the USD macro story. The USD macro backdrop is ONE sentence;
    the rest must explain the non-USD currency's specific driver for that pair.
    WRONG (formulaic copy-paste): EUR/USD, GBP/USD, AUD/USD all saying "USD bounces off 50% Fibo"
    CORRECT: each note leads with the counter-currency's specific COT/CB/carry story:
      EUR/USD: "EUR COT LF short -18.3k. ECB hold limits EUR upside. Fed-ECB [Xbp] carry cap." ← use Cross CB Rate Differentials block for bp value
      GBP/USD: "GBP COT LF long +32.6k. BoE hold. USD bid on geopolitical risk — GBP/USD soft."
      AUD/USD: "AUD COT LF long +48.3k. RBA hike priced — USD bid compresses carry advantage."
      NZD/USD: "NZD COT LF short -16.8k. RBNZ hike priced at 35%. USD bid limits NZD recovery."
8b. CROSS-PAIR CONSISTENCY — when the same pair appears under two different currency sections
    (e.g. CHF/JPY under CHF AND under JPY), the framing must be directionally consistent:
    The pair is the same instrument — it cannot have opposite carry conclusions in two sections.
    ✗ CHF section: "CHF/JPY: carry drag limits demand"  AND
       JPY section: "CHF/JPY: carry favors JPY -75bp"    ← contradictory carry direction
    ✓ Both sections must agree on which side has the carry advantage and the current spread.
    Rule: compute carry direction once (CHF rate vs JPY rate from the CB RATES block) and
    use the same framing in both currency sections. Do not re-derive independently.
8. MATERIALITY THRESHOLD: If a pair's intraday move is < 0.20%, the note MUST use range-bound framing
   ("range-bound", "flat", "steady", "consolidating"). Do NOT attribute a micro-move to a fundamental
   driver (e.g. do NOT write "RBA hike priced in" for a 0.03% move — write "flat — carry differential
   anchors pair"). Reserve fundamental attribution for moves ≥ 0.20% only.
9. COT EXTREME OVERRIDE — when the counter-currency has a Leveraged Fund net position with absolute
   value ≥ 30,000 contracts, the COT reading MUST be mentioned in the driver note. This overrides the
   normal CB-first hierarchy for that pair. The crowded positioning IS the primary risk driver.
   Examples of mandatory COT mentions (these values trigger the override):
     JPY net short −65,309: USD/JPY note MUST say "JPY COT LF short −65.3k"
     CAD net short −63,570: USD/CAD note MUST say "CAD COT LF short −63.6k"
     AUD net long +41,675:  AUD/USD note MUST say "AUD COT LF long +41.7k"
   Format: "COT LF [long/short] [±NNk]. [CB note]." — COT first, CB second.
   WRONG: "CAD soft on weak oil. BoC on hold." ← omits the extreme COT positioning
   CORRECT: "CAD COT LF short −63.6k. BoC on hold. WTI rise limits downside."
10. Max 85 characters per note. No trailing punctuation.
11. Return ONLY valid JSON keyed by the pair symbol shown in the input — no markdown, no preamble:

{
  "CCY/OPP1": "...",
  "CCY/OPP2": "...",
  ...
}\
"""


SESSION_CONTEXT_SYSTEM = """\
You are a senior FX market analyst writing per-session context notes for a Bloomberg-style trading terminal.
You will receive intraday FX data, the current UTC time, and a SESSION STATUS block that classifies
each session as CLOSED, ACTIVE, or UPCOMING. Write ONE short note per session per G8 currency.
Notes must be institutional, factual, and data-grounded. Every catalyst you name must exist in the
data provided to you. You are NOT permitted to invent events.

SESSION STATUS RULES — apply strictly based on the SESSION STATUS block in the user prompt:
  CLOSED sessions   → Factual recap of what happened. Reference the pair and % move.
                      Style: "JPY weakness extended. EUR/JPY +0.18% during session. BoJ passivity."
  ACTIVE session    → What is happening now, present tense. Most specific note of the four.
                      Style: "Core bid on EUR/USD. ECB's Knot: June cut not ruled out. Main driver."
  UPCOMING sessions → Forward-looking watch note using ONLY catalysts from the CALENDAR block.
                      Style: "Watch for GBP on Flash PMI at 08:30 UTC. BoE hold intact — carry eyed."

CATALYST DISCIPLINE — MANDATORY:
  The "Economic Calendar — Key Events" block is the ONLY permitted source for UPCOMING catalysts.
  You MUST cite events from the CALENDAR block by their exact title.
  If the calendar has no high-impact event for a session + currency: use carry differential or
  CB hold as the note. Do NOT invent a rate decision, speaker, or data release.

  ✗ PROHIBITED INVENTIONS — never write these unless they appear in the CALENDAR block:
    "Watch for EUR/USD on ECB rate decision"   ← forbidden if no ECB decision in calendar
    "Watch for GBP/USD on BoE rate decision"   ← forbidden if no BoE decision in calendar
    "Watch for AUD/USD on RBA rate decision"   ← forbidden if no RBA decision in calendar
    "Watch for NZD/USD on RBNZ rate decision"  ← forbidden if no RBNZ decision in calendar
    "Watch for USD/CAD on BoC rate decision"   ← forbidden if no BoC decision in calendar
    "Watch for USD/JPY on BoJ rate decision"   ← forbidden if no BoJ decision in calendar
    "Watch for USD/CHF on SNB rate decision"   ← forbidden if no SNB decision in calendar
    "Fed's Powell speaks"                      ← forbidden in ANY session if Powell not in calendar
    "BoJ's Ueda speaks"                        ← forbidden in ANY session if Ueda not in calendar
    "ECB's Lagarde speaks"                     ← forbidden in ANY session if not in calendar
    "BoE's Pill speaks"                        ← forbidden in ANY session if Pill not in calendar
    "BoE's Bailey speaks"                      ← forbidden in ANY session if Bailey not in calendar
    "SNB's Schlegel speaks"                    ← forbidden in ANY session if Schlegel not in calendar

  CRITICAL: The CB speaker prohibition applies to CLOSED and ACTIVE session notes too —
  not just UPCOMING. Do NOT write that a CB official "speaks" or "spoke" unless their
  name appears in the CALENDAR block. If no speaker is in the calendar, use carry
  differential or CB stance language instead. A post-processing guard will strip any
  hallucinated speaker name and replace the note with a carry fallback.

  ✓ CORRECT UPCOMING note when no calendar event exists for that currency/session:
    "Carry differential holds. EUR/USD range-bound — no catalyst in session window."
    "JPY softness persists. USD/JPY bid on 300bp carry — no scheduled data."
    "AUD steady. RBA hike priced; no intraday catalyst scheduled."
    "NZD/USD carry intact. No high-impact events scheduled in session window."

  ✗ WRONG UPCOMING note when no calendar event exists:
    "Watch for AUD on no high-impact events. RBA hike priced."   ← NEVER use "Watch for X on no events"
    "Watch for NZD on no high-impact events. RBNZ hike priced."  ← same error
    The "Watch for X on Y" template REQUIRES a real calendar event as Y.
    When no event exists, drop the "Watch for" format entirely and use the carry/CB fallback.

STYLE EXAMPLES (Bloomberg session register):
  Sydney CLOSED:      "Low liquidity. EUR/JPY led with JPY softness. Thin range."
  Tokyo CLOSED:       "JPY weakness extended. EUR/JPY +0.18% during session. BoJ passivity."
  London ACTIVE:      "Core bid on EUR/USD. ECB speakers broadly constructive. Main driver."
  London UPCOMING:    "Watch for EUR on German Flash Manufacturing PMI at 07:30 UTC. Consensus 51.4."
  New York UPCOMING:  "Watch for USD on Unemployment Claims 12:30 UTC + Flash PMIs 13:45 UTC."

RULES:
1. Max 95 characters per note. No trailing punctuation.
2. CLOSED/ACTIVE notes: reference the intraday % of the most significant pair in that session window.
3. CLOSED/ACTIVE catalyst — when a headline is listed for that session in the "High-Impact Headlines
   by Session" block, name it explicitly:
   - CB speaker → cite the bank and key message: "ECB's Knot: June cut not ruled out."
   - Data release → cite the release and direction: "US ISM beat. USD/JPY bid on yield spike."
   - Geopolitical → cite the event and frame by regime:
       RISK-OFF (VIX > 25): "Middle East escalation. JPY safe-haven bid."
       MIXED/RISK-ON (VIX < 20): "Middle East risk premium. USD/JPY carry bid vs intervention risk at 160.00."
       Never use "safe-haven demand" in MIXED or RISK-ON sessions — use geopolitical risk premium language instead.
   If no headline is listed, use carry differential or CB stance as fallback.
4. UPCOMING notes: cite ONLY events from the CALENDAR block. If multiple high-impact events exist
   for one session, cite the most market-moving one (CB decision > employment > inflation > PMI).
5. London is highest volume — always the most specific note of the day when ACTIVE or CLOSED.
6. Sydney is lowest volume — if nothing notable: "Thin range. [pair] drifted [direction]."
7. USD notes MUST reference direct pairs (EUR/USD, USD/JPY, GBP/USD, etc.) — NEVER use "DXY" or "USD index".
   NEGATIVE EXAMPLES — these are WRONG for USD session notes:
     ✗ "DXY +0.22%. WTI surge supports USD."    ← DXY is forbidden; name the pair instead
     ✗ "DXY firmed +0.15% during session."       ← forbidden; write "USD/JPY +0.23% during session."
   CORRECT USD note templates (use these patterns):
     Sydney CLOSED:  "USD/JPY +0.11% during session. Dollar bid on crude surge — Fed 3.75% on hold."
     Tokyo CLOSED:   "USD/JPY held 159.37. Fed-BoJ 300bp carry intact — BoJ passivity confirmed."
     London ACTIVE:  "EUR/USD soft at 1.1721. Dollar firms on equity weakness and crude bid."
     New York UP:    "Watch USD/JPY on Federal Funds Rate 18:00 UTC. 160.00 intervention threshold in focus."
   CORRECT: "USD/JPY +0.23% during session. WTI surge supports USD leg."
   CORRECT: "EUR/USD holds 1.1686. Middle East tensions support USD safe-haven bid."
8. Move threshold: if a pair moved less than 0.15% intraday, describe it as "range-bound" or "flat"
   and give the structural constraint — do NOT attribute a micro-move to a fundamental driver.
9. FORBIDDEN phrases: "US data neutral", "DXY", "USD index", "amid", "as investors", "could",
   "might", "potentially", "cautious", "on the back of".
   Also forbidden: the modal "may" (expressing uncertainty) — "May" as a month is ALLOWED (e.g. "Watch RBNZ meeting in May").
10. Each currency's ACTIVE note must differ from all other currencies' ACTIVE notes — vary framing.
11. SESSION DIFFERENTIATION — each of the 4 session notes for a single currency MUST read differently.
    The failure mode is cloned notes: Sydney, Tokyo, London and New York all reading "X range-bound. CB on hold."
    This is a formatting error — it means no real analysis was done.

    REQUIRED DIFFERENTIATION per session slot:
      Sydney   → liquidity angle or overnight dynamic: "Thin range. AUD/USD drifted lower on light vol."
      Tokyo    → Asian session driver: JPY, AUD, NZD flow-specific note, or Asia macro context
      London   → most specific note; if ACTIVE, name the specific driver; if CLOSED, name the % and catalyst
      New York → forward-looking or US session-specific driver; cite calendar event if available

    WRONG — cloned notes (never produce this):
      EUR Sydney:   "EUR/USD range-bound. ECB on hold."
      EUR Tokyo:    "EUR/JPY +0.05%. BoJ passivity."
      EUR London:   "EUR/GBP -0.12%. ECB speakers broadly constructive."
      EUR New York: "Carry differential holds. EUR/USD range-bound — no catalyst in session window."
      ← every note is a generic template; London and New York are interchangeable

    CORRECT — differentiated notes:
      EUR Sydney:   "Thin range. EUR/USD 1.1721, anchored by [Xbp] Fed-ECB carry spread." ← use context block
      EUR Tokyo:    "EUR/JPY +0.05%. BoJ passivity keeps cross bid — carry intact at 200bp."
      EUR London:   "EUR/USD +0.06% session. Dollar slip on equity weakness drives the move."
      EUR New York: "Watch EUR/USD on FOMC 18:00 UTC — [Xbp] carry cap at 1.1850 in focus." ← use context block

12. CLOSED SESSION NEWS INTEGRATION — each CLOSED session note must reflect what actually drove price
    in that session, not just the structural CB stance. If a headline is listed for that session in
    the "High-Impact Headlines by Session" block, the CLOSED note MUST incorporate its driver.
    The failure mode is writing "USD/JPY +0.11%. Fed on hold." when the data shows the move was
    driven by "Dollar Gains on Weak Stocks and Soaring Crude Prices" — the headline IS the driver.

    CORRECT — closed note with news driver:
      Sydney: "Dollar bid as stocks slip and crude surges — USD/JPY +0.11%. Fed 3.75% on hold."
    WRONG — closed note ignores the headline cause:
      Sydney: "Dollar gains on weak stocks and soaring crude prices."  ← parrots the headline verbatim
      Sydney: "USD/JPY +0.11%. Fed on hold."  ← no cause, just the move

    NOTE: Do not quote headlines verbatim. Paraphrase into a factual causal clause (Rule 3 of CATALYST DISCIPLINE).

13. Return ONLY valid JSON — no markdown, no preamble:
  "EUR": { "Sydney": "...", "Tokyo": "...", "London": "...", "New York": "..." },
  "GBP": { "Sydney": "...", "Tokyo": "...", "London": "...", "New York": "..." },
  ... (all 8 G8 currencies: EUR, GBP, JPY, AUD, CAD, CHF, NZD, USD)
}\
"""


# Weekend session context system — used when is_market_closed() returns True.
# FX market is closed (Fri 21:00 – Sun 21:00 UTC). No intraday data is live.
# Each session note plays a specific role in a weekend market brief:
#   Sydney    → Friday close recap: final session of the week, closing-level anchor
#   Tokyo     → Weekly range recap: week's key move and structural driver
#   London    → Main weekly catalyst: the single highest-impact event of the week
#   New York  → Monday open outlook: what to watch, key level or event ahead
# Style mirrors a weekend desk note from a top-tier bank (Goldman, JPM, Barclays).
SESSION_CONTEXT_SYSTEM_CLOSED = """\
You are a senior FX strategist writing weekend session notes for a Bloomberg-style trading terminal.
Markets are CLOSED (Friday 21:00 UTC → Sunday 21:00 UTC). All prices are Friday closing levels.
You will receive Friday's closing FX rates, the week's news headlines, and OIS-implied CB biases.

Each currency gets four notes — one per trading session — framing the WEEKEND REVIEW + MONDAY OUTLOOK:
  Sydney    → Friday close recap. One key pair, its Friday closing level, final-session dynamic.
              Style: "AUD/USD closed 0.6432. RBA hike priced — no fresh catalyst into the close."
  Tokyo     → Weekly range recap. Week's dominant move and structural driver for that currency.
              Style: "JPY weakest G8 ccy this week. USD/JPY range 151.20–153.40 on BoJ passivity."
  London    → Main catalyst of the week. Single highest-impact event — CB decision, macro surprise,
              geopolitical. Tie directly to the pair and direction.
              Style: "ECB held at 2.0%. EUR/USD +0.34% wk — energy risk premium eased post-ceasefire."
  New York  → Monday open outlook. One specific catalyst or level to watch at the open.
              Use ONLY events from the ECONOMIC CALENDAR block. If no event, use CB stance or
              carry as the watchpoint. Do NOT invent a rate decision or speaker.
              Style: "Watch USD/JPY at Friday close. Next BoJ meeting — check CB RATES block for probability."
              Style: "AUD/USD 0.6430 support into RBA May 5. Q1 CPI Apr 29 key swing factor."

RULES:
1. Max 95 characters per note. No trailing punctuation.
2. All prices are FRIDAY CLOSING LEVELS — never describe them as "live" or "current".
3. Sydney and Tokyo notes: always name the pair with its Friday closing price or weekly range.
4. London note: name the weekly catalyst explicitly — CB decision > macro data > geopolitical.
5. New York note: must reference Monday or the coming week. Use calendar events if available.
   EVENT IMPACT DIFFERENTIATION — when a data release affects multiple currencies,
   each currency's New York note must name the SAME event but explain its SPECIFIC impact
   on that pair, not copy "Watch [pair] on [event]" identically for all 8 currencies.
   ✗ WRONG (copy-paste for all 8):
      EUR New York: "Watch EUR/USD on ISM Manufacturing PMI at 14:00 UTC. Forecast 53.1."
      GBP New York: "Watch GBP/USD on ISM Manufacturing PMI at 14:00 UTC. Forecast 53.1."
      JPY New York: "Watch JPY on ISM Manufacturing PMI at 14:00 UTC. Forecast 53.1."
   ✓ CORRECT — differentiate by impact type:
      EUR New York: "ISM PMI at 14:00 UTC — USD-direct data. Fed-ECB [Xbp] spread in focus." ← use context block
      GBP New York: "ISM PMI at 14:00 UTC — USD driver. GBP/USD range-bound; no UK event."
      JPY New York: "ISM PMI at 14:00 UTC — risk appetite gauge. USD/JPY 157 level in focus."
      AUD New York: "ISM PMI at 14:00 UTC — risk proxy for AUD. RBA hike still the base case."
      CHF New York: "Carry note: CHF/JPY carry minimal at current rates. No CH event Monday."
      NZD New York: "RBNZ hike priced — NZD/USD Monday open at 0.5898 carry watch."
   The rule: USD-direct data (Fed, ISM, NFP, CPI) → explain the cross-currency implication;
   do not repeat the same "Watch X on Y" template for currencies with no direct exposure.
6. USD notes: use direct pair names (EUR/USD, USD/JPY, etc.) — NEVER "DXY" or "USD index".
7. If a pair moved less than 0.10% on the week: describe as "range-bound" — no driver attribution.
8. FORBIDDEN phrases: "amid", "as investors", "could", "might", "potentially", "DXY", "USD index",
   "on the back of", "cautious", "see OIS table", "see the OIS", "check the OIS", "refer to OIS".
   Modal "may" expressing uncertainty is forbidden.
   ALSO FORBIDDEN — investment advice language: "for this trade", "go long", "go short",
   "reduce exposure", "entry point", "take profit", "stop loss". Notes are descriptive only.
9. SAFE-HAVEN FRAMING IS REGIME-CONDITIONAL — check the data block for VIX before framing JPY/CHF:
   VIX > 25 (risk-off): "safe-haven demand" framing correct for JPY and CHF.
   VIX < 20 (mixed/normal): DO NOT use "safe-haven demand" or "safe-haven bid". Use carry-first
   framing: "carry differential" or "geopolitical risk premium" instead.
   WRONG (VIX=17): "JPY bid on safe-haven appeal. Iran-US tensions boost USD/JPY."
   CORRECT (VIX=17): "USD/JPY closed 157.03. 300bp carry intact; geopolitical risk premium on Iran."
   EUR/USD is NEVER driven by safe-haven demand — only CB differential or positioning.
10. Return ONLY valid JSON — no markdown, no preamble:

{
  "EUR": { "Sydney": "...", "Tokyo": "...", "London": "...", "New York": "..." },
  "GBP": { "Sydney": "...", "Tokyo": "...", "London": "...", "New York": "..." },
  ... (all 8 G8 currencies: EUR, GBP, JPY, AUD, CAD, CHF, NZD, USD)
}\
"""


def _build_session_status_block() -> str:
    """Classify each FX trading session as CLOSED, ACTIVE, or UPCOMING based on current UTC time.

    Session windows (UTC):
      Sydney:   22:00 – 07:00 (crosses midnight)
      Tokyo:    00:00 – 09:00
      London:   07:00 – 16:00
      New York: 12:00 – 21:00

    Returns a compact block injected into the session context user_prompt so Groq
    knows exactly which sessions to recap (CLOSED), describe live (ACTIVE), and
    watch forward (UPCOMING). Prevents Groq from inventing recaps for future sessions.
    """
    now_utc = datetime.now(timezone.utc)
    h = now_utc.hour

    def sess_status(open_h: int, close_h: int, crosses_midnight: bool = False) -> str:
        if crosses_midnight:
            is_open = h >= open_h or h < close_h
        else:
            is_open = open_h <= h < close_h
        if is_open:
            return "ACTIVE"
        # Determine if it already closed today or hasn't opened yet
        if crosses_midnight:
            closed_today = h >= close_h and h < open_h
        else:
            closed_today = h >= close_h
        return "CLOSED" if closed_today else "UPCOMING"

    sydney_status  = sess_status(22, 7,  crosses_midnight=True)
    tokyo_status   = sess_status(0,  9)
    london_status  = sess_status(7,  16)
    newyork_status = sess_status(12, 21)

    lines = [
        "=== SESSION STATUS (use this to calibrate note tense and content) ===",
        f"Current UTC time: {now_utc.strftime('%H:%M')}",
        f"  Sydney   (22:00–07:00 UTC): {sydney_status}",
        f"  Tokyo    (00:00–09:00 UTC): {tokyo_status}",
        f"  London   (07:00–16:00 UTC): {london_status}",
        f"  New York (12:00–21:00 UTC): {newyork_status}",
        "",
        "CLOSED   → factual recap, past tense, specific pair + % move",
        "ACTIVE   → live description, present tense, most specific note of the day",
        "UPCOMING → forward-looking watch note, 'Watch for...' language, NO past tense",
        "",
    ]
    return "\n".join(lines)


def generate_session_context(api_key: str, context: str, _all_keys: list = None) -> dict:
    """Generate per-session context notes for all G8 currencies.

    Primary: Gemini (larger context window, better structured JSON, single call).
    Fallback: Groq key pool (when all Gemini keys are exhausted or unavailable).

    Makes ONE call covering all 8 currencies × 4 sessions in a single JSON response.
    Written to ai-analysis/session-context.json.
    Read by heatmap-modal.js to populate the SESSION CONTEXT block in the Session tab.

    Token estimate: ~600-800 input + 2000 output tokens per call (1 call/run).
    """
    slim_ctx = build_drivers_context(context)
    ois_block = build_ois_bias_block()
    # Build the key pool for intra-loop rotation.
    # _all_keys is passed from call_with_key_rotation so we can rotate
    # to Key 2 / Key 3 when Key 1 hits DAILY_LIMIT mid-loop (e.g. on NZD/USD).
    # Falls back to [api_key] if not provided (backward-compat).
    _key_pool: list = list(_all_keys) if _all_keys else [api_key]
    if api_key not in _key_pool:
        _key_pool.insert(0, api_key)
    _cur_key_idx: int = _key_pool.index(api_key)

    def _next_key() -> str | None:
        """Advance to the next available key. Returns None when exhausted."""
        nonlocal _cur_key_idx
        _cur_key_idx += 1
        if _cur_key_idx < len(_key_pool):
            new_key = _key_pool[_cur_key_idx]
            print(f"  🔄 Drivers key rotation → Key {_cur_key_idx+1} ({mask_key(new_key)})")
            return new_key
        print("  ⛔ Drivers: all keys exhausted — remaining currencies skipped.")
        return None

    session_status_block = _build_session_status_block()
    news_block = build_session_news_block()
    calendar_block = build_calendar_block()

    user_prompt = (
        session_status_block
        + ois_block
        + calendar_block
        + news_block
        + "Generate session context notes for ALL 8 G8 currencies: EUR, GBP, JPY, AUD, CAD, CHF, NZD, USD.\n"
        "For each currency, write one note per session (Sydney, Tokyo, London, New York).\n"
        "CRITICAL: Use the SESSION STATUS block above to determine tense and content:\n"
        "  - CLOSED sessions → factual recap with pair name and % move from intraday data\n"
        "  - ACTIVE session  → live present-tense note, most specific of the four\n"
        "  - UPCOMING sessions → 'Watch for...' forward-looking, no past tense\n"
        "\n"
        "MANDATORY CALENDAR RULE — NO EXCEPTIONS:\n"
        "  UPCOMING notes MUST be grounded in the 'Economic Calendar — Key Events' block above.\n"
        "  You MUST read the UPCOMING section of the calendar before writing any UPCOMING note.\n"
        "  If the calendar lists 'German Flash Manufacturing PMI' for London at 07:30 UTC [EUR],\n"
        "  write: \"Watch for EUR on German Flash Manufacturing PMI at 07:30 UTC. Consensus 51.4.\"\n"
        "  If the calendar lists 'Flash Manufacturing PMI' + 'Flash Services PMI' for London [GBP],\n"
        "  write: \"Watch for GBP Flash PMI at 08:30 UTC. Manufacturing 50.3, Services 50.0 forecast.\"\n"
        "  If NO calendar event exists for a currency in a session window: write the carry/CB note.\n"
        "  DO NOT invent rate decisions, CB speakers, or data releases not present in the calendar.\n"
        "\n"
        "USD notes must use direct pair names (EUR/USD, USD/JPY, etc.) — never 'DXY' or 'USD index'.\n"
        "If a pair moved less than 0.15% intraday, call it range-bound — no fundamental attribution.\n"
        "Return ONLY valid JSON with all 8 currencies and all 4 sessions. No markdown.\n\n"
        + slim_ctx
    )

    # ── Gemini primary → Groq fallback ───────────────────────────────────────
    # Session context is a single structured JSON call — Gemini handles it cleanly
    # in one shot with its larger context window. Groq (key pool) is the fallback
    # when all Gemini keys are exhausted or unavailable.
    raw = None
    _sc_key_used = "gemini"
    _gemini_sc_keys = load_gemini_keys()
    for _gsc_idx, _gsc_k in enumerate(_gemini_sc_keys):
        try:
            if _gsc_idx > 0:
                import time as _time_gsc; _time_gsc.sleep(KEY_SWITCH_PAUSE_DAILY)
                print(f"  🔄 Session context Gemini key rotation → Key {_gsc_idx+1} ({mask_key(_gsc_k)})")
            raw = call_gemini(_gsc_k, SESSION_CONTEXT_SYSTEM, user_prompt, max_tokens=2000,
                              model=GEMINI_MODEL)
            print(f"  ✅ Session context via Gemini key {_gsc_idx+1}")
            break
        except RuntimeError as _gsc_e:
            _gsc_e_str = str(_gsc_e)
            if "DAILY_LIMIT" in _gsc_e_str and _gsc_idx < len(_gemini_sc_keys) - 1:
                print(f"  ⛔ Session context Gemini Key {_gsc_idx+1} daily limit — rotating.")
                continue
            elif "RATE_LIMIT" in _gsc_e_str and _gsc_idx < len(_gemini_sc_keys) - 1:
                import time as _time_gsc_rl; _time_gsc_rl.sleep(KEY_SWITCH_PAUSE_RATE)
                print(f"  ⏳ Session context Gemini Key {_gsc_idx+1} rate limit — rotating after {KEY_SWITCH_PAUSE_RATE}s.")
                continue
            print(f"  WARNING session context Gemini error ({_gsc_e}) — falling back to Groq.")
            break

    # Groq fallback — only reached when all Gemini keys fail or are unavailable
    if raw is None:
        print(f"  INFO Session context: Gemini unavailable — trying Groq fallback pool.")
        _sc_key_used = api_key
        for _sc_k_idx, _sc_k in enumerate(_key_pool):
            try:
                if _sc_k_idx > 0:
                    import time as _time_sc; _time_sc.sleep(KEY_SWITCH_PAUSE_DAILY)
                    print(f"  🔄 Session context Groq key rotation → Key {_sc_k_idx+1} ({mask_key(_sc_k)})")
                raw = call_groq(_sc_k, SESSION_CONTEXT_SYSTEM, user_prompt, max_tokens=1200)
                _sc_key_used = _sc_k
                break
            except RuntimeError as _sc_e:
                _sc_e_str = str(_sc_e)
                if "DAILY_LIMIT" in _sc_e_str and _sc_k_idx < len(_key_pool) - 1:
                    print(f"  ⛔ Session context Groq Key {_sc_k_idx+1} daily limit reached — rotating.")
                    continue
                elif "RATE_LIMIT" in _sc_e_str and _sc_k_idx < len(_key_pool) - 1:
                    import time as _time_sc_rl; _time_sc_rl.sleep(KEY_SWITCH_PAUSE_RATE)
                    print(f"  ⏳ Session context Groq Key {_sc_k_idx+1} rate limit — rotating after {KEY_SWITCH_PAUSE_RATE}s.")
                    continue
                print(f"  WARNING session context Groq error ({_sc_e}) — skipping.")
                return {}
    if raw is None:
        print(f"  WARNING session context: all Gemini and Groq keys exhausted — skipping.")
        return {}

    try:
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        parsed = json.loads(raw)

        VALID_CCYS = {"EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "USD"}
        VALID_SESS = {"Sydney", "Tokyo", "London", "New York"}
        clean: dict[str, dict[str, str]] = {}
        for ccy, sessions in parsed.items():
            if ccy not in VALID_CCYS or not isinstance(sessions, dict):
                continue
            clean[ccy] = {
                s: str(note)[:110]
                for s, note in sessions.items()
                if s in VALID_SESS
            }
        total_notes = sum(len(v) for v in clean.values())

        # ── Post-validation: detect hallucinated UPCOMING catalysts ──────────
        # Groq ignores the CATALYST DISCIPLINE block and invents rate decisions
        # or data releases not present in ff_calendar.json. This guard builds a
        # set of calendar event title words and strips invented UPCOMING notes,
        # replacing them with the approved carry/CB fallback.
        # Only fires on UPCOMING sessions — CLOSED/ACTIVE notes use intraday data.
        cal_raw = load_json(SITE_DIR / "calendar-data" / "ff_calendar.json")
        cal_events = (cal_raw or {}).get("events", [])
        # Build a set of lowercased title words from upcoming events in the calendar
        import re as _re_cal
        cal_title_words: set[str] = set()
        from datetime import datetime as _dt_cal, timezone as _tz_cal, timedelta as _td_cal
        now_utc_sc = _dt_cal.now(_tz_cal.utc)
        cutoff_sc  = now_utc_sc + _td_cal(hours=24)
        for ev in cal_events:
            if ev.get("impact") not in ("high", "medium"):
                continue
            if ev.get("released"):
                continue
            try:
                ev_dt = _dt_cal.strptime(
                    f"{ev['dateISO']} {ev['timeUTC']}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=_tz_cal.utc)
                if ev_dt > now_utc_sc and ev_dt <= cutoff_sc:
                    for word in (ev.get("title") or "").lower().split():
                        if len(word) >= 4:
                            cal_title_words.add(word)
            except Exception:
                continue

        # Helper: determine if a session is UPCOMING relative to current UTC time
        h_now = now_utc_sc.hour
        def _is_upcoming(sess_name: str) -> bool:
            windows = {
                "Sydney":   (22, 7,  True),
                "Tokyo":    (0,  9,  False),
                "London":   (7,  16, False),
                "New York": (12, 21, False),
            }
            w = windows.get(sess_name)
            if not w:
                return False
            open_h, close_h, crosses = w
            if crosses:
                is_open = h_now >= open_h or h_now < close_h
            else:
                is_open = open_h <= h_now < close_h
            if is_open:
                return False  # ACTIVE, not UPCOMING
            if crosses:
                return h_now >= close_h and h_now < open_h
            return h_now < open_h  # hasn't opened yet

        # Forbidden catalyst patterns — two tiers:
        #
        # TIER A — rate decisions: apply to UPCOMING sessions only.
        # A rate decision in a CLOSED/ACTIVE note is a factual recap (it happened)
        # and is legitimate. Only UPCOMING notes can hallucinate a future decision.
        #
        # TIER B — CB speaker names: apply to ALL sessions (CLOSED, ACTIVE, UPCOMING).
        # The model invents specific CB official names (Ueda, Pill, Lagarde, Powell,
        # Schlegel, etc.) regardless of session status. If the name is not in the
        # calendar, it is a hallucination whether the session is past or future.
        # This covers the observed failure mode: JPY/London "BOJ's Ueda speaks" and
        # GBP/New York "BoE's Pill speaks" were CLOSED/ACTIVE notes — outside the
        # prior UPCOMING-only scope.

        HALLUCINATION_PATTERNS_UPCOMING = [
            # Rate decisions — only checked for UPCOMING sessions
            r"ecb\s+rate\s+decision",
            r"boe\s+rate\s+decision", r"bank\s+of\s+england\s+rate",
            r"rba\s+rate\s+decision",
            r"rbnz\s+rate\s+decision",
            r"boc\s+rate\s+decision", r"bank\s+of\s+canada\s+rate",
            r"boj\s+rate\s+decision", r"bank\s+of\s+japan\s+rate",
            r"snb\s+rate\s+decision",
            r"fed\s+rate\s+decision",
        ]

        # CB speaker name patterns — checked for ALL sessions.
        # Format: possessive or standalone ("BOJ's Ueda", "Ueda speaks", "BoE's Pill").
        # Covers the most commonly hallucinated names across G8 CBs.
        HALLUCINATION_PATTERNS_ALL_SESSIONS = [
            # Fed
            r"\bpowell\s+speaks?\b", r"\bfed['']?s\s+powell\b",
            r"\bwaller\s+speaks?\b", r"\bjefferson\s+speaks?\b",
            r"\bkugler\s+speaks?\b", r"\bwarsh\s+speaks?\b",
            # ECB
            r"\blagarde\s+speaks?\b", r"\becb['']?s\s+lagarde\b",
            r"\bknot\s+speaks?\b",    r"\blane\s+speaks?\b",
            r"\bschnabel\s+speaks?\b",r"\bnagel\s+speaks?\b",
            # BoJ
            r"\bueda\s+speaks?\b",    r"\bboj['']?s\s+ueda\b",
            r"\bhimino\s+speaks?\b",  r"\bnakamura\s+speaks?\b",
            # BoE
            r"\bbailey\s+speaks?\b",  r"\bboe['']?s\s+bailey\b",
            r"\bpill\s+speaks?\b",    r"\bboe['']?s\s+pill\b",
            r"\bbreeden\s+speaks?\b", r"\bhaskel\s+speaks?\b",
            r"\bramsden\s+speaks?\b",
            # RBA
            r"\bbullock\s+speaks?\b", r"\brba['']?s\s+bullock\b",
            # BoC
            r"\bmacklem\s+speaks?\b", r"\bboc['']?s\s+macklem\b",
            # SNB
            r"\bschlegel\s+speaks?\b",r"\bsnb['']?s\s+schlegel\b",
            # RBNZ
            r"\borr\s+speaks?\b",     r"\bbreman\s+speaks?\b",
            r"\brbnz['']?s\s+orr\b",  r"\brbnz['']?s\s+breman\b",
        ]

        def _speaker_is_in_calendar(pat: str, cal_events: list) -> bool:
            """Return True if a matching speaker event exists in the calendar."""
            return any(
                _re_cal.search(pat, (ev.get("title") or "").lower())
                for ev in cal_events
                if not ev.get("released")
            )

        hallucinations_found = 0
        for ccy, sessions in clean.items():
            for sess, note in list(sessions.items()):
                note_lower = note.lower()
                replaced = False

                # --- TIER A: rate decisions (UPCOMING only) ---
                if _is_upcoming(sess):
                    for pat in HALLUCINATION_PATTERNS_UPCOMING:
                        if _re_cal.search(pat, note_lower):
                            is_real = any(
                                _re_cal.search(pat, (ev.get("title") or "").lower())
                                for ev in cal_events
                                if not ev.get("released")
                            )
                            if not is_real:
                                print(f"  WARNING hallucinated rate-decision stripped "
                                      f"({sess}): {ccy}: '{note[:80]}'")
                                clean[ccy][sess] = (
                                    f"{ccy}/USD carry differential holds. "
                                    f"No scheduled high-impact event in session window."
                                )[:110]
                                hallucinations_found += 1
                                replaced = True
                            break

                if replaced:
                    continue

                # --- TIER B: CB speaker names (ALL sessions) ---
                for pat in HALLUCINATION_PATTERNS_ALL_SESSIONS:
                    if _re_cal.search(pat, note_lower):
                        if not _speaker_is_in_calendar(pat, cal_events):
                            print(f"  WARNING hallucinated CB speaker stripped "
                                  f"({sess}): {ccy}: '{note[:80]}'")
                            # Build a carry/CB-stance fallback using live data
                            _cb_n   = _CB_NAMES.get(ccy, ccy)
                            _rt     = _rate_str(ccy)
                            _mt_h   = _meetings_map.get(ccy, {})
                            _bias_h = _mt_h.get("bias", "hold")
                            _bl_h   = {"hike": "hike priced", "cut": "cut priced",
                                       "hold": "on hold"}.get(_bias_h, "on hold")
                            _pl_h   = _PAIR_FOR_SESSION.get((ccy, sess), f"{ccy}/USD")
                            clean[ccy][sess] = (
                                f"{_pl_h} range-bound. {_cb_n} {_rt} — {_bl_h}."
                            )[:110]
                            hallucinations_found += 1
                        break  # one pattern match per note is sufficient

        if hallucinations_found:
            print(f"  INFO {hallucinations_found} hallucinated catalyst(s) stripped "
                  f"(rate-decisions + CB speakers).")

        # ── DXY guard — strip forbidden DXY references from USD session notes ──
        # Rule 7 in SESSION_CONTEXT_SYSTEM forbids "DXY" and "USD index" in USD notes.
        # Groq persistently violates this (e.g. "DXY +0.22%. WTI surge supports USD.")
        # despite the system prompt. This post-processing guard catches all violations
        # and replaces the offending note with a pair-based fallback.
        import re as _re_dxy
        DXY_PATTERN = _re_dxy.compile(r'\bdxy\b|usd\s+index', _re_dxy.IGNORECASE)
        dxy_violations = 0
        usd_notes = clean.get("USD", {})
        for sess, note in list(usd_notes.items()):
            if DXY_PATTERN.search(note):
                # Build a pair-specific replacement from intraday data
                intra_q = load_json(SITE_DIR / "intraday-data" / "quotes.json")
                q_data  = (intra_q or {}).get("quotes", {})
                # Pick the most significant USD pair for this session
                SESSION_USD_PAIR = {
                    "Sydney":   ("eurusd", "EUR/USD"),
                    "Tokyo":    ("usdjpy", "USD/JPY"),
                    "London":   ("eurusd", "EUR/USD"),
                    "New York": ("usdjpy", "USD/JPY"),
                }
                pair_key, pair_label = SESSION_USD_PAIR.get(sess, ("eurusd", "EUR/USD"))
                pair_data = q_data.get(pair_key, {})
                pair_close = pair_data.get("close")
                pair_pct   = pair_data.get("pct")
                if pair_close and pair_pct is not None:
                    sign = "+" if pair_pct >= 0 else ""
                    fallback = f"{pair_label} {pair_close:.4f} ({sign}{pair_pct:.2f}%). USD steady — Fed on hold."
                else:
                    fallback = f"{pair_label} range-bound. USD steady — Fed on hold."
                print(f"  WARNING DXY in USD/{sess} note stripped: '{note[:80]}' → fallback applied.")
                clean["USD"][sess] = fallback[:110]
                dxy_violations += 1
        if dxy_violations:
            print(f"  INFO {dxy_violations} DXY violation(s) in USD session notes corrected.")

        # ── Off-topic note guard ──────────────────────────────────────────────
        # Detects notes for non-USD currencies that contain no currency-specific
        # content — e.g. NZD/NY = "Treas Sec. Bessent: Growth-first strategy
        # drives U.S. economic push." (pure USD macro, no NZD pair or RBNZ ref).
        # Root cause: LLM runs out of relevant NZD/AUD/CAD data for quiet sessions
        # and copies a USD headline instead.
        # Fix: if a note for currency CCY contains none of the expected signals
        # (pair symbols, CB name, or currency ticker), replace it with a
        # carry-differential / CB-stance fallback derived from live data.
        # Applies to ALL sessions (CLOSED, ACTIVE, UPCOMING).
        _CCY_KEYWORDS = {
            "EUR": ["eur", "ecb", "eurozone", "euro"],
            "GBP": ["gbp", "boe", "bank of england", "sterling", "pound"],
            "JPY": ["jpy", "boj", "bank of japan", "yen"],
            "AUD": ["aud", "rba", "reserve bank of australia", "aussie"],
            "CAD": ["cad", "boc", "bank of canada", "loonie"],
            "CHF": ["chf", "snb", "swiss", "franc"],
            "NZD": ["nzd", "rbnz", "reserve bank of new zealand", "kiwi"],
        }
        import re as _re_ot
        # Load rates once for fallback construction
        _rates_dir = SITE_DIR / "rates"
        def _rate_str(ccy: str) -> str:
            try:
                rd = load_json(_rates_dir / f"{ccy}.json")
                obs = (rd or {}).get("observations", [{}])
                v = float(obs[0].get("value", 0)) if obs else 0
                return f"{v:.2f}%"
            except Exception:
                return "N/A"

        _CB_NAMES = {
            "EUR": "ECB", "GBP": "BoE", "JPY": "BoJ",
            "AUD": "RBA", "CAD": "BoC", "CHF": "SNB", "NZD": "RBNZ",
        }
        _PAIR_FOR_SESSION = {
            # (ccy, session) → canonical pair label
            ("EUR", "Sydney"): "EUR/USD", ("EUR", "Tokyo"): "EUR/JPY",
            ("EUR", "London"): "EUR/USD", ("EUR", "New York"): "EUR/USD",
            ("GBP", "Sydney"): "GBP/USD", ("GBP", "Tokyo"): "GBP/JPY",
            ("GBP", "London"): "GBP/USD", ("GBP", "New York"): "GBP/USD",
            ("JPY", "Sydney"): "AUD/JPY", ("JPY", "Tokyo"): "USD/JPY",
            ("JPY", "London"): "USD/JPY", ("JPY", "New York"): "USD/JPY",
            ("AUD", "Sydney"): "AUD/USD", ("AUD", "Tokyo"): "AUD/JPY",
            ("AUD", "London"): "AUD/USD", ("AUD", "New York"): "AUD/USD",
            ("CAD", "Sydney"): "USD/CAD", ("CAD", "Tokyo"): "CAD/JPY",
            ("CAD", "London"): "USD/CAD", ("CAD", "New York"): "USD/CAD",
            ("CHF", "Sydney"): "USD/CHF", ("CHF", "Tokyo"): "CHF/JPY",
            ("CHF", "London"): "EUR/CHF", ("CHF", "New York"): "USD/CHF",
            ("NZD", "Sydney"): "NZD/USD", ("NZD", "Tokyo"): "NZD/JPY",
            ("NZD", "London"): "NZD/USD", ("NZD", "New York"): "NZD/USD",
        }
        # Bias labels from meetings for CB name in fallback
        _meetings_raw = load_json(SITE_DIR / "meetings-data" / "meetings.json")
        _meetings_map = (_meetings_raw or {}).get("meetings", {})

        off_topic_replaced = 0
        for ccy, keywords in _CCY_KEYWORDS.items():
            if ccy not in clean:
                continue
            for sess, note in list(clean[ccy].items()):
                note_lower = note.lower()
                if any(kw in note_lower for kw in keywords):
                    continue  # note is on-topic ✓
                # Note has no CCY-specific signal — build carry/CB fallback
                pair_label = _PAIR_FOR_SESSION.get((ccy, sess), f"{ccy}/USD")
                cb = _CB_NAMES.get(ccy, ccy)
                rate = _rate_str(ccy)
                mt   = _meetings_map.get(ccy, {})
                bias = mt.get("bias", "hold")
                bias_label = {"hike": "hike priced", "cut": "cut priced", "hold": "on hold"}.get(bias, "on hold")
                fallback = f"{pair_label} range-bound. {cb} {rate} — {bias_label}."
                print(
                    f"  WARNING off-topic {ccy}/{sess} note replaced: "
                    f"'{note[:70]}' → carry fallback"
                )
                clean[ccy][sess] = fallback[:110]
                off_topic_replaced += 1
        if off_topic_replaced:
            print(f"  INFO {off_topic_replaced} off-topic session note(s) replaced with carry fallback.")

        print(f"    Session context: {len(clean)} currencies, {total_notes} notes")
        return clean
    except (json.JSONDecodeError, Exception) as e:
        print(f"  WARNING session context parse error ({e}) — skipping.")
        return {}


def generate_session_context_closed(api_key: str, context: str, _all_keys: list = None) -> dict:
    """Generate weekend session recap notes for all G8 currencies.

    Primary: Gemini. Fallback: Groq key pool.
    Called when is_market_closed() returns True. Uses SESSION_CONTEXT_SYSTEM_CLOSED
    which frames each note as: Friday close recap / weekly range / main catalyst /
    Monday open outlook — matching Bloomberg/Reuters weekend desk note style.

    Same JSON schema as generate_session_context(), so heatmap-modal.js renders
    without any changes to the reading side.
    Written to ai-analysis/session-context.json (same file, different content).
    """
    slim_ctx = build_drivers_context(context)
    ois_block = build_ois_bias_block()
    calendar_block = build_calendar_block()      # upcoming events for Monday outlook
    news_block = build_session_news_block()      # weekend headlines as the primary catalyst source

    _key_pool: list = list(_all_keys) if _all_keys else [api_key]
    if api_key not in _key_pool:
        _key_pool.insert(0, api_key)

    user_prompt = (
        "=== WEEKEND MARKET BRIEF — FX market closed (Fri 21:00 – Sun 21:00 UTC) ===\n"
        "All prices below are FRIDAY CLOSING LEVELS. Do not describe them as live.\n\n"
        + ois_block
        + calendar_block        # used for Monday outlook (New York note)
        + news_block            # weekend headlines — primary source for London note
        + "Generate the weekend session recap for ALL 8 G8 currencies: EUR, GBP, JPY, AUD, CAD, CHF, NZD, USD.\n"
        "Sydney  → Friday close recap (closing level + final-session dynamic)\n"
        "Tokyo   → Weekly range recap (week's dominant move + structural driver)\n"
        "London  → Main catalyst of the week (CB decision, macro surprise, geopolitical)\n"
        "New York → Monday open outlook (specific level or calendar event to watch at open)\n"
        "USD notes: use pair names (EUR/USD, USD/JPY) — never 'DXY' or 'USD index'.\n"
        "Return ONLY valid JSON, no markdown.\n\n"
        + slim_ctx
    )

    # ── Gemini primary → Groq fallback ───────────────────────────────────────
    raw = None
    _gemini_wsc_keys = load_gemini_keys()
    for _k_idx, _k in enumerate(_gemini_wsc_keys):
        try:
            if _k_idx > 0:
                import time as _t_gsc; _t_gsc.sleep(KEY_SWITCH_PAUSE_DAILY)
                print(f"  🔄 Weekend session context Gemini key rotation → Key {_k_idx+1} ({mask_key(_k)})")
            raw = call_gemini(_k, SESSION_CONTEXT_SYSTEM_CLOSED, user_prompt, max_tokens=2000,
                              model=GEMINI_MODEL)
            print(f"  ✅ Weekend session context via Gemini key {_k_idx+1}")
            break
        except RuntimeError as _e:
            _e_str = str(_e)
            if "DAILY_LIMIT" in _e_str and _k_idx < len(_gemini_wsc_keys) - 1:
                print(f"  ⛔ Weekend session context Gemini Key {_k_idx+1} daily limit — rotating.")
                continue
            elif "RATE_LIMIT" in _e_str and _k_idx < len(_gemini_wsc_keys) - 1:
                import time as _t_gsc_rl; _t_gsc_rl.sleep(KEY_SWITCH_PAUSE_RATE)
                print(f"  ⏳ Weekend session context Gemini Key {_k_idx+1} rate limit — rotating after {KEY_SWITCH_PAUSE_RATE}s.")
                continue
            print(f"  WARNING weekend session context Gemini error ({_e}) — falling back to Groq.")
            break

    # Groq fallback
    if raw is None:
        print(f"  INFO Weekend session context: Gemini unavailable — trying Groq fallback pool.")
        for _k_idx, _k in enumerate(_key_pool):
            try:
                if _k_idx > 0:
                    import time as _t_sc; _t_sc.sleep(KEY_SWITCH_PAUSE_DAILY)
                    print(f"  🔄 Weekend session context Groq key rotation → Key {_k_idx+1} ({mask_key(_k)})")
                raw = call_groq(_k, SESSION_CONTEXT_SYSTEM_CLOSED, user_prompt, max_tokens=1200)
                break
            except RuntimeError as _e:
                _e_str = str(_e)
                if "DAILY_LIMIT" in _e_str and _k_idx < len(_key_pool) - 1:
                    print(f"  ⛔ Weekend session context Groq Key {_k_idx+1} daily limit — rotating.")
                    continue
                elif "RATE_LIMIT" in _e_str and _k_idx < len(_key_pool) - 1:
                    import time as _t_sc_rl; _t_sc_rl.sleep(KEY_SWITCH_PAUSE_RATE)
                    print(f"  ⏳ Weekend session context Groq Key {_k_idx+1} rate limit — rotating after {KEY_SWITCH_PAUSE_RATE}s.")
                    continue
                print(f"  WARNING weekend session context Groq error ({_e}) — skipping.")
                return {}
    if raw is None:
        print("  WARNING weekend session context: all Gemini and Groq keys exhausted — skipping.")
        return {}

    try:
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        parsed = json.loads(raw)

        VALID_CCYS = {"EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "USD"}
        VALID_SESS = {"Sydney", "Tokyo", "London", "New York"}
        clean: dict[str, dict[str, str]] = {}
        for ccy, sessions in parsed.items():
            if ccy not in VALID_CCYS or not isinstance(sessions, dict):
                continue
            clean[ccy] = {
                s: str(note)[:110]
                for s, note in sessions.items()
                if s in VALID_SESS
            }
        # DXY guard (same as weekday version — Groq ignores the rule consistently)
        import re as _re_dxy_wk
        DXY_PAT = _re_dxy_wk.compile(r'\bdxy\b|usd\s+index', _re_dxy_wk.IGNORECASE)
        for sess, note in list(clean.get("USD", {}).items()):
            if DXY_PAT.search(note):
                clean["USD"][sess] = "EUR/USD Friday close. USD steady — Fed on hold."
                print(f"  INFO DXY stripped from USD/{sess} weekend note.")

        total_notes = sum(len(v) for v in clean.values())
        print(f"    Weekend session context: {len(clean)} currencies, {total_notes} notes")
        return clean
    except (json.JSONDecodeError, Exception) as e:
        print(f"  WARNING weekend session context parse error ({e}) — skipping.")
        return {}


def build_drivers_context(context: str) -> str:
    """Extract the FX-relevant slice of context for the drivers call.

    Keeps the FX intraday prices, CB rates, COT, and yield curve sections —
    drops the verbose correlation, retail sentiment, and news blocks that are
    not needed for a per-currency driver note. Reduces token cost by ~40%.
    """
    lines = context.split("\n")
    keep = []
    in_section = False
    KEEP_HEADERS = {
        "=== Live FX Spot Rates",
        "=== Friday Closing FX Rates",
        "=== Central Bank Policy Rates",
        "=== CFTC Speculative Positioning",
        "=== Intraday Market Quotes",
        "  -- FX Intraday Prices",
        "  -- Cross CB Rate Differentials",
        "  -- Yield Curve Regime",
        "  -- Intraday Yields",
        "Current time:",
        "MARKET STATUS:",
    }
    SKIP_HEADERS = {
        "=== Retail FX Sentiment",
        "=== Cross-Asset Correlation",
        "=== Latest FX Headlines",
    }
    skip_until_next = False
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(h) for h in SKIP_HEADERS):
            skip_until_next = True
            continue
        if skip_until_next:
            if stripped.startswith("===") or stripped.startswith("--"):
                skip_until_next = False
            else:
                continue
        keep.append(line)

    return "\n".join(keep)


def build_drivers_context_for_ccy(context: str, ccy: str, pairs: list) -> str:
    """Return a context slice scoped to the currencies relevant for a single driver call.

    Instead of sending the full slim_ctx (~2,800 tokens) to every one of the 8
    per-currency calls, this function strips CB rate and COT lines down to only
    the currencies that appear in the pair list for ccy (plus ccy itself).
    Global sections (intraday FX prices, yields, current time) are kept in full
    because they are short and provide the intraday anchor the model needs.

    Token reduction: ~2,800 tokens → ~500-700 tokens per call (~75% reduction).
    Accuracy: no data is lost — every pair note only references the two currencies
    in that pair, so lines for unrelated currencies are genuinely irrelevant.
    """
    # Collect all currencies touched by this call (ccy + both legs of every pair)
    relevant = {ccy}
    for p in pairs:
        parts = p.split("/")
        if len(parts) == 2:
            relevant.update(parts)

    lines = context.split("\n")
    keep = []
    in_cb_section  = False
    in_cot_section = False
    in_skip_section = False

    SKIP_HEADERS = {
        "=== Retail FX Sentiment",
        "=== Cross-Asset Correlation",
        "=== Latest FX Headlines",
    }

    for line in lines:
        stripped = line.strip()

        # Track section transitions
        if stripped.startswith("=== Central Bank Policy Rates"):
            in_cb_section, in_cot_section, in_skip_section = True, False, False
            keep.append(line)
            continue
        if stripped.startswith("=== CFTC Speculative Positioning"):
            in_cb_section, in_cot_section, in_skip_section = False, True, False
            keep.append(line)
            continue
        if any(stripped.startswith(h) for h in SKIP_HEADERS):
            in_cb_section, in_cot_section, in_skip_section = False, False, True
            continue
        if stripped.startswith("==="):
            in_cb_section, in_cot_section, in_skip_section = False, False, False

        # Drop unwanted sections entirely
        if in_skip_section:
            continue

        # CB rates section — keep only lines for currencies in this pair set
        if in_cb_section:
            if stripped == "":
                keep.append(line)
            elif any(f"({c})" in stripped for c in relevant):
                keep.append(line)
            continue

        # COT section — keep only lines for currencies in this pair set
        # COT lines formatted as "  {CCY}: NET ..."
        if in_cot_section:
            if stripped == "":
                keep.append(line)
                continue
            line_ccy = stripped.split(":")[0].strip() if ":" in stripped else ""
            if line_ccy in relevant:
                keep.append(line)
            continue

        # All other lines — keep as-is
        keep.append(line)

    return "\n".join(keep)


def build_session_news_block() -> str:
    """Build a session-bucketed news block for the session context prompt.

    Institutional rationale: session context notes cite the catalysts that actually
    moved the market during each session window. Bloomberg session recaps always name
    the CB speaker or data release — not a generic carry note. This function:

      1. Loads news-data/news.json (same source as build_context()).
      2. Filters to high-impact English headlines only (same filter as narrative).
      3. Buckets each headline into the session whose UTC window contains its publish
         time (HH:MM). Headlines without a parseable time go into a catch-all block.
      4. Returns a compact, session-labelled block injected into the session context
         user_prompt so Groq can cite the correct catalyst per session.

    Session windows (UTC):
      Sydney:   22:00–07:00 (crosses midnight)
      Tokyo:    00:00–09:00
      London:   07:00–16:00
      New York: 12:00–21:00

    Token cost: ~80-150 tokens (5 headlines max, one sentence each).
    """
    SESSION_WINDOWS = [
        # (name, open_h, close_h, crosses_midnight)
        ("Sydney",   22, 7,  True),
        ("Tokyo",    0,  9,  False),
        ("London",   7,  16, False),
        ("New York", 12, 21, False),
    ]

    # Priority order for overlap resolution: highest volume wins
    # London/Tokyo overlap (07-09) → London; London/NY overlap (12-16) → New York
    SESSION_PRIORITY = [
        ("New York", 12, 21, False),
        ("London",   7,  16, False),
        ("Tokyo",    0,  9,  False),
        ("Sydney",   22, 7,  True),
    ]

    def _session_for_time(hhmm: str) -> str | None:
        """Return the highest-volume active session for a HH:MM string."""
        try:
            h = int(hhmm[:2])
        except (ValueError, IndexError):
            return None
        for name, open_h, close_h, crosses in SESSION_PRIORITY:
            if crosses:
                if h >= open_h or h < close_h:
                    return name
            else:
                if open_h <= h < close_h:
                    return name
        return None  # 21:00–22:00 gap — no major session open

    news_raw = load_json(SITE_DIR / "news-data" / "news.json", default={})
    articles = news_raw.get("articles", news_raw.get("items", [])) \
               if isinstance(news_raw, dict) else news_raw

    # Same two-layer EM filter as build_context() — cur tag + title keyword check.
    _G8_CURS_SESS = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
    _EM_CB_KW_SESS = {
        "banxico", "banco de mexico", "bcb", "banco do brasil",
        "pboc", "people's bank of china", "rbi", "reserve bank of india",
        "sarb", "south african reserve", "norges bank", "riksbank",
        "nbp", "national bank of poland", "cbrt", "turkey rate",
        "banco central de chile", "banco central de colombia",
    }
    def _is_em_cb_sess(article: dict) -> bool:
        title_lower = article.get("title", "").lower()
        return any(kw in title_lower for kw in _EM_CB_KW_SESS)

    high_impact = [
        a for a in articles
        if a.get("impact") == "high"
        and a.get("lang", "en") == "en"
        and a.get("cur", "USD") in _G8_CURS_SESS
        and not _is_em_cb_sess(a)
    ][:12]  # cap at 12 before bucketing — ~5 per session at most

    if not high_impact:
        return ""

    # Bucket by session
    buckets: dict[str, list[str]] = {s[0]: [] for s in SESSION_WINDOWS}
    unmatched: list[str] = []

    for item in high_impact:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        ccy   = item.get("cur") or item.get("currency") or ""
        time  = item.get("time", "")
        tag   = f"[{ccy}] " if ccy else ""
        label = f"{time} {tag}{title}" if time else f"{tag}{title}"
        sess  = _session_for_time(time)
        if sess:
            buckets[sess].append(label)
        else:
            unmatched.append(label)

    # Build block — only emit sessions that have headlines
    lines = ["=== High-Impact Headlines by Session ===",
             "(Use these to name the specific catalyst in each session note — CB speakers, data releases, geopolitical events.)",
             ""]
    any_content = False
    for name, *_ in SESSION_WINDOWS:
        items = buckets[name]
        if items:
            lines.append(f"  {name}:")
            for h in items[:4]:  # max 4 per session
                lines.append(f"    • {h}")
            any_content = True

    if unmatched:
        lines.append("  Other (time unknown):")
        for h in unmatched[:3]:
            lines.append(f"    • {h}")
        any_content = True

    if not any_content:
        return ""

    lines.append("")
    return "\n".join(lines) + "\n"


def build_calendar_block(window_hours_back: int = 8, window_hours_ahead: int = 24) -> str:
    """Build a session-aware economic calendar block from calendar-data/ff_calendar.json.

    Institutional rationale: Bloomberg session recaps always cite the specific CB speaker
    or data release that drove the session — not generic carry notes. This function
    provides Groq with the ground-truth event list so it can name real catalysts instead
    of hallucinating upcoming rate decisions that don't exist.

    CLOSED / ACTIVE session notes → events that already released (actual present) in
    the past `window_hours_back` hours, grouped by session.

    UPCOMING session notes → high-impact events with no actual yet, scheduled in the
    next `window_hours_ahead` hours, grouped by session.

    The block is structured to mirror the SESSION STATUS block format so Groq can
    directly cross-reference which events belong to which session note.

    Token cost: ~80–150 tokens/run (high-impact only, two windows).
    Falls back silently to empty string if ff_calendar.json is unavailable —
    generate_session_context() will still work via the news headlines fallback.
    """
    FF_CALENDAR_PATH = SITE_DIR / "calendar-data" / "ff_calendar.json"

    # Session windows (UTC) — same as _build_session_status_block()
    SESSION_PRIORITY = [
        ("New York", 12, 21, False),
        ("London",   7,  16, False),
        ("Tokyo",    0,  9,  False),
        ("Sydney",   22, 7,  True),
    ]
    SESSION_ORDER = ["Sydney", "Tokyo", "London", "New York"]

    def _session_for_hour(h: int) -> str | None:
        for name, open_h, close_h, crosses in SESSION_PRIORITY:
            if crosses:
                if h >= open_h or h < close_h:
                    return name
            else:
                if open_h <= h < close_h:
                    return name
        return None

    try:
        cal = load_json(FF_CALENDAR_PATH)
        if not cal or not cal.get("events"):
            return ""

        now_utc   = datetime.now(timezone.utc)
        cutoff_past   = now_utc - timedelta(hours=window_hours_back)
        cutoff_future = now_utc + timedelta(hours=window_hours_ahead)

        released_events: list[dict]  = []
        upcoming_events: list[dict]  = []

        for ev in cal["events"]:
            if ev.get("impact") not in ("high", "medium"):
                continue  # skip low-impact only

            try:
                dt_utc = datetime.strptime(
                    f"{ev['dateISO']} {ev['timeUTC']}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=timezone.utc)
            except (ValueError, KeyError):
                continue

            if ev.get("released") and dt_utc >= cutoff_past and dt_utc <= now_utc:
                released_events.append((dt_utc, ev))
            elif not ev.get("released") and dt_utc > now_utc and dt_utc <= cutoff_future:
                upcoming_events.append((dt_utc, ev))

        if not released_events and not upcoming_events:
            return ""

        lines = [
            "=== Economic Calendar — Key Events ===",
            "(Cite these catalysts by name in session notes: speaker name, release title, actual vs forecast.)",
            "",
        ]

        def _fmt_event(dt_utc: datetime, ev: dict, show_actual: bool) -> str:
            time_str = dt_utc.strftime("%H:%M UTC")
            ccy      = ev.get("currency", "")
            title    = ev.get("title", "")
            impact   = "●" if ev.get("impact") == "high" else "○"
            parts    = [f"  {impact} {time_str} [{ccy}] {title}"]
            if show_actual and ev.get("actual"):
                fc = ev.get("forecast", "—") or "—"
                parts[0] += f" → actual {ev['actual']} (forecast {fc})"
            elif ev.get("forecast"):
                parts[0] += f" | forecast {ev['forecast']}"
            return parts[0]

        # Released events — bucket by session for CLOSED/ACTIVE notes
        if released_events:
            lines.append("RELEASED (use in CLOSED/ACTIVE session notes):")
            # Group by session
            by_session: dict[str, list] = {s: [] for s in SESSION_ORDER}
            ungrouped: list = []
            for dt_utc, ev in sorted(released_events, key=lambda x: x[0]):
                sess = _session_for_hour(dt_utc.hour)
                if sess:
                    by_session[sess].append((dt_utc, ev))
                else:
                    ungrouped.append((dt_utc, ev))
            for sess in SESSION_ORDER:
                if by_session[sess]:
                    lines.append(f"  {sess}:")
                    for dt_utc, ev in by_session[sess]:
                        lines.append(_fmt_event(dt_utc, ev, show_actual=True))
            if ungrouped:
                for dt_utc, ev in ungrouped:
                    lines.append(_fmt_event(dt_utc, ev, show_actual=True))
            lines.append("")

        # Upcoming events — for UPCOMING session notes
        if upcoming_events:
            lines.append("UPCOMING (use in UPCOMING session notes — name the catalyst to watch):")
            by_session_up: dict[str, list] = {s: [] for s in SESSION_ORDER}
            ungrouped_up: list = []
            for dt_utc, ev in sorted(upcoming_events, key=lambda x: x[0]):
                sess = _session_for_hour(dt_utc.hour)
                if sess:
                    by_session_up[sess].append((dt_utc, ev))
                else:
                    ungrouped_up.append((dt_utc, ev))
            for sess in SESSION_ORDER:
                if by_session_up[sess]:
                    lines.append(f"  {sess}:")
                    for dt_utc, ev in by_session_up[sess]:
                        lines.append(_fmt_event(dt_utc, ev, show_actual=False))
            if ungrouped_up:
                for dt_utc, ev in ungrouped_up:
                    lines.append(_fmt_event(dt_utc, ev, show_actual=False))
            lines.append("")

        n_released = len(released_events)
        n_upcoming = len(upcoming_events)
        print(f"    Calendar block: {n_released} released, {n_upcoming} upcoming events")
        return "\n".join(lines) + "\n"

    except Exception as e:
        print(f"  WARNING: Could not build calendar block ({e})")
        return ""


def build_ois_bias_block() -> str:
    """Build a pre-processed OIS/forward bias table from meetings.json.

    Returns a compact, clearly labelled block injected at the TOP of every
    driver user_prompt so Groq cannot overlook or contradict the forward bias.
    This resolves the root cause of notes like "RBNZ cutting cycle" when the
    OIS data shows hike=100% — the model was seeing the bias buried in a large
    context block and defaulting to the historical cutting trend instead.
    """
    CB_LABELS_LOCAL = {
        "USD": "Fed", "EUR": "ECB", "GBP": "BoE", "JPY": "BoJ",
        "AUD": "RBA", "CAD": "BoC", "CHF": "SNB", "NZD": "RBNZ",
    }
    BIAS_LABEL = {"cut": "↓ CUT EXPECTED", "hold": "→ ON HOLD", "hike": "↑ HIKE EXPECTED"}
    try:
        meetings_data = load_json(SITE_DIR / "meetings-data" / "meetings.json")
        if not meetings_data:
            return ""
        meetings = meetings_data.get("meetings", {})
        lines = ["=== OIS FORWARD BIAS (AUTHORITATIVE — OVERRIDES ALL HISTORICAL TRENDS) ==="]
        lines.append("Use ONLY the forward bias below. Never use historical rate trends to infer direction.")
        lines.append("")
        for ccy in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]:
            m = meetings.get(ccy, {})
            bias_raw = m.get("bias", "hold")
            hike_p = m.get("hikeProb") or 0
            cut_p  = m.get("cutProb")  or 0
            # Coherence fix: if bias=hike but hikeProb=0, the effective bias is Hold.
            # "Hike bias (0% priced)" is self-contradictory — override to hold.
            if bias_raw == "hike" and hike_p == 0:
                bias_raw = "hold"
            # Coherence fix: if bias=cut but cutProb=0, override to hold.
            if bias_raw == "cut" and cut_p == 0:
                bias_raw = "hold"
            bias_str = BIAS_LABEL.get(bias_raw, "→ ON HOLD")
            cb = CB_LABELS_LOCAL.get(ccy, ccy)
            next_mtg = m.get("nextMeeting", "N/A")
            prob_str = ""
            if hike_p and bias_raw == "hike":
                prob_str = f" | Hike prob: {hike_p}%"
            elif cut_p and bias_raw == "cut":
                prob_str = f" | Cut prob: {cut_p}%"
            lines.append(f"  {ccy} ({cb}): {bias_str}{prob_str} | Next meeting: {next_mtg}")
        lines.append("")
        lines.append("MANDATORY: If a CB is ↑ HIKE EXPECTED, write 'hike priced in', NOT 'cut cycle' or 'cut bets'.")
        lines.append("MANDATORY: If a CB is ↓ CUT EXPECTED, write 'cut expected', NOT 'on hold' or 'hike bets'.")
        lines.append("")
        return "\n".join(lines)
    except Exception as e:
        print(f"  WARNING: Could not build OIS bias block ({e})")
        return ""


def _build_cot_extreme_hint(ccy: str, site_dir) -> str:
    """Return a COT EXTREME note for the given currency if its LF net abs >= 30k.

    Injected into the drivers user_prompt so the model has the exact number
    and cannot omit it — Rule 9 (COT EXTREME OVERRIDE) in DRIVERS_SYSTEM.
    """
    COT_EXTREME_THRESHOLD = 30_000
    d = load_json(site_dir / "cot-data" / f"{ccy}.json")
    if not d:
        return ""
    net = d.get("netPosition")
    if net is None or abs(net) < COT_EXTREME_THRESHOLD:
        return ""
    direction = "long" if net > 0 else "short"
    net_k = f"{net/1000:+.1f}k"
    return (
        f"\nCOT EXTREME ALERT — {ccy} LF net {direction} {net_k} (abs ≥ 30k — RULE 9 APPLIES):\n"
        f"  All pairs involving {ccy} as counter-currency MUST include this COT extreme in the note.\n"
        f"  Format: \"{ccy} COT LF {direction} {net_k}. [CB stance].\"\n"
    )


def generate_currency_drivers(api_key: str, context: str, _all_keys: list = None) -> dict:
    """Generate per-pair driver notes for all 8 G8 currencies.

    Primary: Gemini — single call covering all 8 currencies in one shot.
      Returns nested JSON: { "EUR": { "EUR/USD": "...", ... }, "GBP": { ... }, ... }
      Eliminates the 8 round-trip overhead of the legacy Groq serial approach.
      max_tokens=3000 ensures all 56 pair notes fit without truncation.

    Fallback: Groq (llama-3.1-8b-instant) — 8 serial per-currency calls.
      Only reached when all Gemini keys are exhausted or return incomplete data.
      On parse failure for a given currency, that currency's dict is empty — frontend
      falls back gracefully to showing pairs without notes.

    Token estimate (Gemini path): ~2,000 input + 3,000 output = ~5,000 tokens/run.
    Token estimate (Groq fallback): ~220 tokens × 8 currencies = ~1,760 tokens/run.
    """
    # Build base slim context once (strips sentiment/correlation/headlines).
    # Per-currency calls use build_drivers_context_for_ccy() to further reduce
    # CB rates and COT sections to only the 2-3 currencies relevant per call.
    # Token reduction: ~2,800 → ~500-700 tokens per call (~75% per call).
    full_slim_ctx = build_drivers_context(context)
    ois_block = build_ois_bias_block()

    # ── Regime-aware safe-haven framing hint (built once, injected per currency) ──
    # Rule 3 of SIGNALS_SYSTEM: safe-haven framing (CHF/JPY "safe-haven bid on risk-off")
    # is only correct in RISK-OFF (VIX > 25, SPX < -1.5%). In MIXED/RISK-ON regimes
    # the model must use carry or neutral framing for CHF and JPY safe-haven pairs.
    _regime_hint = ""
    _tokens_drv = extract_tokens_from_intraday()
    _vix_drv    = None
    try:
        _vix_drv = float(_tokens_drv.get("vix", "") or 0) or None
    except (ValueError, TypeError):
        pass
    if _vix_drv is not None:
        if _vix_drv < 20:
            _sh_framing = "carry-neutral or range-bound"
            _sh_example_chf = f"CHF steady. SNB on hold. Carry drag limits demand."
            _sh_example_jpy = f"JPY COT LF short. BoJ on hold. Carry bid intact."
            _regime_hint = (
                f"\nREGIME: VIX {_vix_drv:.1f} — NOT risk-off (VIX < 20). "
                f"SAFE-HAVEN FRAMING RULE:\n"
                f"  Do NOT use 'safe-haven bid on risk-off' or 'safe-haven demand' for CHF or JPY.\n"
                f"  VIX < 20 = normal/low vol — CHF/JPY safe-haven flows are marginal.\n"
                f"  Use {_sh_framing} framing instead:\n"
                f"  CORRECT (CHF): \"{_sh_example_chf}\"\n"
                f"  CORRECT (JPY): \"{_sh_example_jpy}\"\n"
                f"  WRONG: \"CHF safe-haven bid on risk-off\" ← only valid when VIX > 25\n"
            )
        elif _vix_drv > 25:
            _regime_hint = (
                f"\nREGIME: VIX {_vix_drv:.1f} — RISK-OFF conditions active (VIX > 25). "
                f"Safe-haven framing for CHF and JPY is appropriate.\n"
            )

    # G8 direct pairs per currency — matches PAIR_DEFS in heatmap-modal.js
    CCY_PAIRS: dict[str, list[str]] = {
        "EUR": ["EUR/USD", "EUR/GBP", "EUR/JPY", "EUR/AUD", "EUR/CAD", "EUR/CHF", "EUR/NZD"],
        "GBP": ["GBP/USD", "GBP/JPY", "EUR/GBP", "GBP/AUD", "GBP/CAD", "GBP/CHF", "GBP/NZD"],
        "JPY": ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY", "CAD/JPY", "CHF/JPY", "NZD/JPY"],
        "AUD": ["AUD/USD", "EUR/AUD", "GBP/AUD", "AUD/JPY", "AUD/CAD", "AUD/CHF", "AUD/NZD"],
        "CAD": ["USD/CAD", "EUR/CAD", "GBP/CAD", "AUD/CAD", "CAD/JPY", "CAD/CHF", "NZD/CAD"],
        "CHF": ["USD/CHF", "EUR/CHF", "GBP/CHF", "AUD/CHF", "CAD/CHF", "CHF/JPY", "NZD/CHF"],
        "NZD": ["NZD/USD", "EUR/NZD", "GBP/NZD", "AUD/NZD", "NZD/JPY", "NZD/CAD", "NZD/CHF"],
        "USD": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"],
    }

    all_drivers: dict[str, dict[str, str]] = {}

    # Currencies whose pair list includes a USD-leg pair — these all get the intraday
    # USD catalyst hint so the counter-USD pairs break the "USD steady. Fed on hold."
    # boilerplate when a real headline exists. Built once, reused across calls.
    USD_LEG_CCYS = {"EUR", "GBP", "AUD", "CAD", "CHF", "NZD", "USD"}

    def _build_usd_headline_hint() -> str:
        """Return a compact catalyst note for the top high-impact USD headline today."""
        news_raw = load_json(SITE_DIR / "news-data" / "news.json", default={})
        articles = news_raw.get("articles", news_raw.get("items", [])) \
                   if isinstance(news_raw, dict) else news_raw
        usd_high = [
            a for a in articles
            if a.get("impact") == "high"
            and a.get("lang", "en") == "en"
            and a.get("cur") == "USD"
        ][:1]
        if not usd_high:
            any_high = [
                a for a in articles
                if a.get("impact") == "high" and a.get("lang", "en") == "en"
            ][:1]
            usd_high = any_high
        if usd_high:
            item = usd_high[0]
            t = (item.get("time", "") + " " if item.get("time") else "")
            return (
                f"\nKEY USD INTRADAY CATALYST — cite this in counter-USD pair notes where relevant:\n"
                f"  {t}{item.get('title', '')}\n"
                f"For pairs where USD is the counter (EUR/USD, GBP/USD, AUD/USD, NZD/USD, USD/CAD,\n"
                f"USD/CHF, USD/JPY): reference this catalyst instead of generic 'USD steady. Fed on hold.'\n"
                f"Only use CB stance boilerplate if the headline is not USD-relevant.\n"
            )
        return ""

    usd_hint = _build_usd_headline_hint()  # build once, reuse

    # ── Parse helper shared by Gemini and Groq paths ─────────────────────────
    def _parse_driver_response(raw: str, pairs: list) -> dict:
        """Strip markdown fences, extract first JSON object, return valid pair notes."""
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        start = raw.find("{")
        if start < 0:
            return {}
        raw = raw[start:]
        try:
            parsed, _ = json.JSONDecoder().raw_decode(raw)
        except json.JSONDecodeError:
            end = raw.rfind("}") + 1
            if end <= 0:
                return {}
            raw = raw[:end]
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {k: str(v)[:100] for k, v in parsed.items() if k in pairs}

    # ── Gemini primary: single call for all 8 currencies ─────────────────────
    # Gemini's large context window handles all 8 currency blocks in one request,
    # eliminating the 8-round-trip overhead of the Groq approach. Output is a
    # nested JSON: { "EUR": { "EUR/USD": "...", ... }, "GBP": { ... }, ... }
    _gemini_drv_keys = load_gemini_keys()
    _gemini_drv_succeeded = False

    if _gemini_drv_keys:
        # Build the consolidated prompt for all 8 currencies at once
        all_pairs_block = ""
        for _drv_ccy, _drv_pairs in CCY_PAIRS.items():
            all_pairs_block += (
                f"  {_drv_ccy}: {', '.join(_drv_pairs)}\n"
            )
        gemini_drv_prompt = (
            ois_block
            + _regime_hint
            + f"Write one-line driver notes for ALL 8 G8 currencies × their 7 direct pairs.\n"
            f"For each pair, explain why the COUNTER-currency (not the base listed) drove the move.\n"
            f"CRITICAL: Use the OIS FORWARD BIAS table above — not historical rate trends.\n"
            f"CRITICAL: Carry spread values must match the EXACT bilateral pair — never apply\n"
            f"  a spread from a different pair (e.g. NZD-CHF spread must not appear in NZD/CAD notes).\n"
            f"Currency → pairs:\n"
            + all_pairs_block
            + usd_hint
            + "Return ONLY valid JSON: {{\"EUR\": {{\"EUR/USD\": \"...\", ...}}, \"GBP\": {{...}}, ...}}\n"
            f"All 8 currencies must be present. No markdown.\n\n"
            + full_slim_ctx
        )

        for _gdrv_idx, _gdrv_k in enumerate(_gemini_drv_keys):
            try:
                if _gdrv_idx > 0:
                    import time as _tgd; _tgd.sleep(KEY_SWITCH_PAUSE_DAILY)
                    print(f"  🔄 Drivers Gemini key rotation → Key {_gdrv_idx+1} ({mask_key(_gdrv_k)})")
                raw_gemini_drv = call_gemini(
                    _gdrv_k, DRIVERS_SYSTEM, gemini_drv_prompt, max_tokens=3000,
                    model=GEMINI_MODEL
                )
                # Parse the nested JSON response
                _graw = raw_gemini_drv.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                _gs = _graw.find("{")
                if _gs >= 0:
                    _graw = _graw[_gs:]
                try:
                    _gparsed, _ = json.JSONDecoder().raw_decode(_graw)
                except json.JSONDecodeError:
                    _ge = _graw.rfind("}") + 1
                    _gparsed = json.loads(_graw[:_ge]) if _ge > 0 else {}

                _VALID_G8 = set(CCY_PAIRS.keys())
                for _gccy, _gpairs_data in _gparsed.items():
                    if _gccy not in _VALID_G8 or not isinstance(_gpairs_data, dict):
                        continue
                    _gvalid = {
                        k: str(v)[:100]
                        for k, v in _gpairs_data.items()
                        if k in CCY_PAIRS.get(_gccy, [])
                    }
                    all_drivers[_gccy] = _gvalid
                    print(f"    {_gccy} (Gemini): {len(_gvalid)}/{len(CCY_PAIRS[_gccy])} pair notes")

                # Accept if we got notes for at least 6 of 8 currencies
                _covered = sum(1 for c in CCY_PAIRS if all_drivers.get(c))
                if _covered >= 6:
                    print(f"  ✅ Drivers via Gemini key {_gdrv_idx+1} — {_covered}/8 currencies covered")
                    _gemini_drv_succeeded = True
                    break
                else:
                    print(f"  WARNING Drivers Gemini key {_gdrv_idx+1}: only {_covered}/8 currencies — "
                          f"keeping partial result, Groq will fill gaps.")
                    # FIX-21: Don't clear partial Gemini results. Keep what we got
                    # and let the Groq fallback fill only the missing currencies.
                    # Previously all_drivers.clear() discarded valid Gemini output
                    # and then Groq had to regenerate everything — wasting quota.
                    # With this fix, Groq only calls for currencies with no notes.
                    if _covered > 0:
                        _gemini_drv_succeeded = True  # partial success counts
                        print(f"  INFO Partial Gemini success — Groq will fill remaining {8 - _covered} currencies.")
                        break
                    # Only clear and retry next key if we got absolutely nothing
                    all_drivers.clear()
                    continue
            except RuntimeError as _gdrv_e:
                _gdrv_estr = str(_gdrv_e)
                if "DAILY_LIMIT" in _gdrv_estr and _gdrv_idx < len(_gemini_drv_keys) - 1:
                    print(f"  ⛔ Drivers Gemini Key {_gdrv_idx+1} daily limit — rotating.")
                    continue
                elif "RATE_LIMIT" in _gdrv_estr and _gdrv_idx < len(_gemini_drv_keys) - 1:
                    import time as _tgd_rl; _tgd_rl.sleep(KEY_SWITCH_PAUSE_RATE)
                    print(f"  ⏳ Drivers Gemini Key {_gdrv_idx+1} rate limit — rotating after {KEY_SWITCH_PAUSE_RATE}s.")
                    continue
                print(f"  WARNING Drivers Gemini key {_gdrv_idx+1} failed ({_gdrv_e}) — falling back to Groq.")
                break
            except (json.JSONDecodeError, Exception) as _gdrv_pe:
                print(f"  WARNING Drivers Gemini key {_gdrv_idx+1} parse error ({_gdrv_pe}) — falling back to Groq.")
                all_drivers.clear()
                break

    # ── Groq fallback: 8 serial per-currency calls (original architecture) ───
    # Only reached when all Gemini keys fail. Uses llama-3.1-8b-instant (DRIVERS_MODEL)
    # via Groq key pool for the per-currency scoped calls.
    if not _gemini_drv_succeeded:
        print(f"  INFO Drivers: Gemini unavailable or incomplete — falling back to Groq serial calls.")
        # Key pool for rotation across the 8 per-currency calls
        _key_pool: list = list(_all_keys) if _all_keys else [api_key]
        if api_key not in _key_pool:
            _key_pool.insert(0, api_key)
        _cur_key_idx: int = _key_pool.index(api_key)

        def _rotate_drivers_key() -> str | None:
            """Advance to the next available key. Returns the new key or None when exhausted."""
            nonlocal _cur_key_idx
            _cur_key_idx += 1
            if _cur_key_idx < len(_key_pool):
                new_key = _key_pool[_cur_key_idx]
                print(f"  🔄 Drivers Groq key rotation → Key {_cur_key_idx+1} ({mask_key(new_key)})")
                return new_key
            print("  ⛔ Drivers: all Groq keys exhausted — remaining currencies skipped.")
            return None

        for ccy, pairs in CCY_PAIRS.items():
            if all_drivers.get(ccy):
                continue  # already populated by Gemini partial success
            pairs_str = ", ".join(pairs)
            headline_hint = usd_hint if ccy in USD_LEG_CCYS else ""
            cot_hints = ""
            counter_ccys_in_pairs = set()
            for p in pairs:
                parts_p = p.split("/")
                if len(parts_p) == 2:
                    base_p, quote_p = parts_p
                    counter = quote_p if base_p == ccy else base_p
                    counter_ccys_in_pairs.add(counter)
            for counter in sorted(counter_ccys_in_pairs):
                cot_hints += _build_cot_extreme_hint(counter, SITE_DIR)

            scoped_ctx = build_drivers_context_for_ccy(full_slim_ctx, ccy, pairs)

            user_prompt = (
                ois_block
                + _regime_hint
                + f"Currency: {ccy}. Direct G8 pairs: {pairs_str}.\n"
                f"Write a one-line driver note for each of these 7 pairs explaining "
                f"why the counter-currency (not {ccy}) drove the move.\n"
                f"CRITICAL: Check the OIS FORWARD BIAS table above before writing any note. "
                f"Use the forward bias, not historical rate trends.\n"
                + cot_hints
                + headline_hint
                + f"Return JSON keyed exactly by the pair symbols listed above.\n\n"
                + scoped_ctx
            )
            if _cur_key_idx >= len(_key_pool):
                print(f"  ⛔ Drivers: all Groq keys exhausted — skipping {ccy} and remaining currencies.")
                all_drivers[ccy] = {}
                continue
            _active_key = _key_pool[_cur_key_idx]
            try:
                raw = call_groq(_active_key, DRIVERS_SYSTEM, user_prompt, max_tokens=500, model=DRIVERS_MODEL)
                valid = _parse_driver_response(raw, pairs)
                # Retry once if response is sparse
                min_expected = max(1, len(pairs) // 2)
                if len(valid) < min_expected:
                    missing_pairs = [p for p in pairs if p not in valid]
                    print(f"  WARNING drivers incomplete response for {ccy} "
                          f"({len(valid)}/{len(pairs)} pairs) — retrying for: {missing_pairs}")
                    import time as _time
                    _time.sleep(2)
                    retry_pairs_str = ", ".join(missing_pairs)
                    retry_prompt = (
                        ois_block
                        + f"Currency: {ccy}. Direct G8 pairs: {retry_pairs_str}.\n"
                        f"Write a one-line driver note for ONLY these missing pairs: {retry_pairs_str}.\n"
                        f"CRITICAL: Check the OIS FORWARD BIAS table above before writing any note. "
                        f"Use the forward bias, not historical rate trends.\n"
                        + cot_hints
                        + headline_hint
                        + f"Return JSON keyed exactly by the pair symbols listed above.\n\n"
                        + scoped_ctx
                    )
                    try:
                        raw2 = call_groq(_active_key, DRIVERS_SYSTEM, retry_prompt, max_tokens=500, model=DRIVERS_MODEL)
                        parsed2 = _parse_driver_response(raw2, missing_pairs)
                        valid.update(parsed2)
                        print(f"    {ccy} Groq retry: recovered {len(parsed2)} additional notes")
                    except Exception as retry_e:
                        print(f"  WARNING drivers Groq retry parse error for {ccy} ({retry_e}) — using partial result.")
                all_drivers[ccy] = valid
                print(f"    {ccy} (Groq): {len(valid)}/{len(pairs)} pair notes")
            except RuntimeError as e:
                err_str = str(e)
                if "DAILY_LIMIT" in err_str:
                    succeeded = False
                    while True:
                        rotated = _rotate_drivers_key()
                        if rotated is None:
                            all_drivers[ccy] = {}
                            break
                        import time as _time_dl; _time_dl.sleep(KEY_SWITCH_PAUSE_DAILY)
                        try:
                            raw_rot = call_groq(rotated, DRIVERS_SYSTEM, user_prompt, max_tokens=500, model=DRIVERS_MODEL)
                            valid_rot = _parse_driver_response(raw_rot, pairs)
                            all_drivers[ccy] = valid_rot
                            print(f"    {ccy} (Groq rotated key): {len(valid_rot)}/{len(pairs)} pair notes")
                            succeeded = True
                            break
                        except RuntimeError as rot_rt_e:
                            if "DAILY_LIMIT" in str(rot_rt_e):
                                print(f"  WARNING drivers rotated-key DAILY_LIMIT for {ccy} — trying next key.")
                                continue
                            print(f"  WARNING drivers rotated-key call failed for {ccy} ({rot_rt_e}) — skipping.")
                            all_drivers[ccy] = {}
                            succeeded = True
                            break
                        except Exception as rot_e:
                            print(f"  WARNING drivers rotated-key call failed for {ccy} ({rot_e}) — skipping.")
                            all_drivers[ccy] = {}
                            succeeded = True
                            break
                elif "RATE_LIMIT" in err_str:
                    import time as _time_rl; _time_rl.sleep(KEY_SWITCH_PAUSE_RATE)
                    try:
                        raw_rl = call_groq(_active_key, DRIVERS_SYSTEM, user_prompt, max_tokens=500, model=DRIVERS_MODEL)
                        valid_rl = _parse_driver_response(raw_rl, pairs)
                        all_drivers[ccy] = valid_rl
                        print(f"    {ccy} Groq rate-limit retry: {len(valid_rl)}/{len(pairs)} pair notes")
                    except Exception as rl_e:
                        print(f"  WARNING drivers rate-limit retry failed for {ccy} ({rl_e}) — skipping.")
                        all_drivers[ccy] = {}
                else:
                    print(f"  WARNING drivers unexpected error for {ccy} ({e}) — skipping.")
                    all_drivers[ccy] = {}
            except (json.JSONDecodeError, Exception) as e:
                print(f"  WARNING drivers parse error for {ccy} ({e}) — skipping.")
                all_drivers[ccy] = {}

    # FIX-21: always return all_drivers regardless of which path was taken.
    # Previously the Groq fallback loop ended without a return statement, causing
    # Python to return None implicitly. main() received None → has_content=False →
    # never wrote currency-drivers.json even when Groq produced valid notes.
    return all_drivers


def _sanitize_driver_notes(all_drivers: dict) -> dict:
    """Strip headline timestamp fragments leaked into driver notes.

    The model sometimes copies a news headline verbatim into a note, including
    its timestamp prefix (e.g. '17:02 USD: Inflation shock risk from Hormuz blockade.').
    These fragments are not pair-specific analysis and must be removed.

    Pattern matched: one or more occurrences of HH:MM WORD: ... at any position
    in the note string. The timestamp + source prefix is stripped; the remaining
    text is kept only if it is substantive (> 15 chars). Otherwise the note is
    dropped (set to empty string — frontend renders nothing).
    """
    import re as _re_drv
    _TS_PATTERN = _re_drv.compile(
        r'(?<![Aa][Tt] )'               # negative lookbehind: not preceded by "at " (calendar refs)
        r'(?:^|(?<=\. )|(?<=\.\s))'     # at start of string or after a sentence boundary
        r'\d{1,2}:\d{2}\s+'             # HH:MM + space
        r'(?:[A-Z][A-Za-z/]+:\s*)?',    # optional "WORD: " source prefix (e.g. "USD: " "EUR/GBP: ")
        _re_drv.MULTILINE,
        # Covers two leakage modes:
        #   Mode A — "17:02 USD: Inflation shock..." → full prefix stripped
        #   Mode B — "GBP bid. 17:02 USD inflation risk eyed" → timestamp stripped, text kept
        # Does NOT strip "Watch US PMI at 09:45 UTC" (lookbehind guards calendar time refs)
    )
    cleaned = 0
    for ccy, pairs in all_drivers.items():
        for pair, note in list(pairs.items()):
            stripped = _TS_PATTERN.sub('', note).strip()
            # Remove leading/trailing punctuation artefacts left after strip
            stripped = stripped.lstrip('.,;— ').rstrip('.,;— ').strip()
            if stripped != note:
                if len(stripped) > 15:
                    all_drivers[ccy][pair] = stripped
                else:
                    all_drivers[ccy][pair] = ''
                cleaned += 1
    if cleaned:
        print(f"  INFO Driver timestamp-fragment sanitization: {cleaned} note(s) corrected.")
    return all_drivers


def extract_tokens_from_intraday() -> dict:
    """Extract current values for live placeholder tokens from intraday quotes.json.

    Returns a dict with string-formatted values ready for frontend substitution:
      gold_pct  — e.g. "+3.20%"  (with sign)
      spx_pct   — e.g. "-0.85%"
      dxy_pct   — e.g. "+0.12%"
      vix       — e.g. "22.4"
    Values are None when data is unavailable OR when the symbol is in closed_symbols
    (bank holiday / market closed). None causes resolve_tokens() to strip the
    placeholder rather than inject a misleading "+0.00%" into the narrative.
    """
    intraday = load_json(SITE_DIR / "intraday-data" / "quotes.json")
    tokens: dict = {}
    if not intraday or not intraday.get("quotes"):
        return tokens
    q = intraday["quotes"]

    # FIX-31: Suppress pct tokens for symbols that are closed today (bank holidays).
    # On US holidays, gold/spx/dxy carry pct=0.0 (stale Friday close) in quotes.json.
    # Without this guard, resolve_tokens() injects "+0.00%" into the narrative output,
    # producing "gold higher +0.00%" — a misleading precision on a closed market day.
    _closed = get_closed_symbols()

    def fmt_pct(key: str) -> str | None:
        if key in _closed:
            return None  # resolve_tokens() strips the placeholder cleanly
        d = q.get(key)
        if d and d.get("pct") is not None:
            pct = d["pct"]
            sign = "+" if pct >= 0 else ""
            return f"{sign}{pct:.2f}%"
        return None

    tokens["gold_pct"] = fmt_pct("gold")
    tokens["spx_pct"]  = fmt_pct("spx")
    tokens["dxy_pct"]  = fmt_pct("dxy")

    vix_d = q.get("vix")
    if vix_d and vix_d.get("close") is not None:
        vix_val = vix_d["close"]
        tokens["vix"] = f"{vix_val:.1f}"
        if vix_val < 15:
            tokens["vix_label"] = "low volatility"
        elif vix_val < 20:
            tokens["vix_label"] = "normal volatility"
        elif vix_val <= 28:
            tokens["vix_label"] = "elevated volatility"
        else:
            tokens["vix_label"] = "high volatility / fear"

    return tokens


def _sanitize_cot_values(all_drivers: dict, site_dir) -> dict:
    """Post-process driver notes to correct stale or hallucinated COT net values.

    The 8b model sometimes uses a COT value from training memory (e.g. JPY -65.3k)
    instead of the authoritative CFTC data block (e.g. JPY -80.6k). This validator
    reads the actual COT file for each currency and regex-replaces any mismatched
    COT LF [long/short] [value]k pattern with the correct authoritative value.

    Only corrects values that DIFFER from the authoritative data — does not touch
    notes that already have the correct value or notes without COT references.
    """
    import re as _re_cot
    COT_EXTREME_THRESHOLD = 30_000
    corrections = 0

    for ccy in list(all_drivers.keys()):
        cot_data = load_json(site_dir / "cot-data" / f"{ccy}.json")
        if not cot_data:
            continue
        net = cot_data.get("netPosition")
        if net is None or abs(net) < COT_EXTREME_THRESHOLD:
            # Only validate COT values that are required (Rule 9 — abs >= 30k)
            continue
        direction = "long" if net > 0 else "short"
        authoritative_k = f"{net/1000:+.1f}k"  # e.g. "-80.6k"

        # Pattern: "CCY COT LF long/short ±NNk" — match any value
        _cot_pattern = _re_cot.compile(
            rf'\b{_re_cot.escape(ccy)}\s+COT\s+LF\s+(?:long|short)\s+([-+]?\d+\.?\d*k)',
            _re_cot.IGNORECASE
        )
        for base_ccy, pairs in all_drivers.items():
            for pair, note in list(pairs.items()):
                match = _cot_pattern.search(note)
                if match:
                    found_val = match.group(1)
                    # Normalize both for comparison (handle +/- and trailing k)
                    found_norm = found_val.lstrip('+')
                    auth_norm  = authoritative_k.lstrip('+')
                    if found_norm != auth_norm:
                        corrected = _cot_pattern.sub(
                            f'{ccy} COT LF {direction} {authoritative_k}',
                            note
                        )
                        all_drivers[base_ccy][pair] = corrected
                        corrections += 1
                        print(f"  INFO COT value corrected in {pair}: {found_val} → {authoritative_k} ({ccy})")

    if corrections:
        print(f"  INFO COT consistency fix: {corrections} note(s) corrected.")
    return all_drivers


    """Extract current values for live placeholder tokens from intraday quotes.json.

    Returns a dict with string-formatted values ready for frontend substitution:
      gold_pct  — e.g. "+3.20%"  (with sign)
      spx_pct   — e.g. "-0.85%"
      dxy_pct   — e.g. "+0.12%"
      vix       — e.g. "22.4"
    Values are None when data is unavailable.
    """
    intraday = load_json(SITE_DIR / "intraday-data" / "quotes.json")
    tokens: dict = {}
    if not intraday or not intraday.get("quotes"):
        return tokens
    q = intraday["quotes"]

    def fmt_pct(key: str) -> str | None:
        d = q.get(key)
        if d and d.get("pct") is not None:
            pct = d["pct"]
            sign = "+" if pct >= 0 else ""
            return f"{sign}{pct:.2f}%"
        return None

    tokens["gold_pct"] = fmt_pct("gold")
    tokens["spx_pct"]  = fmt_pct("spx")
    tokens["dxy_pct"]  = fmt_pct("dxy")

    vix_d = q.get("vix")
    if vix_d and vix_d.get("close") is not None:
        vix_val = vix_d["close"]
        tokens["vix"] = f"{vix_val:.1f}"
        # Pre-compute the deterministic VIX band label here in Python so the LLM
        # receives it as a data fact and does not need to infer it from the number.
        # This eliminates the _correct_vix_label() post-processing band-aid:
        # the LLM is instructed to use vix_label verbatim, making the label
        # generation rule-based (Python) rather than probabilistic (LLM).
        if vix_val < 15:
            tokens["vix_label"] = "low volatility"
        elif vix_val < 20:
            tokens["vix_label"] = "normal volatility"
        elif vix_val <= 28:
            tokens["vix_label"] = "elevated volatility"
        else:
            tokens["vix_label"] = "high volatility / fear"

    return tokens


def resolve_tokens(text: str, tokens: dict) -> str:
    """Replace {{token}} placeholders with live intraday values.

    Handles the sign-collision case: Groq may write "gold +{{gold_pct}}" and
    the token value already includes the sign ("+3.20%"), which would produce
    "gold ++3.20%". We strip a leading sign from the text before the placeholder
    when the replacement value starts with a sign.

    Falls back gracefully: if a token has no value, removes the placeholder
    and its preceding sign so "gold +{{gold_pct}}" becomes "gold".
    """
    import re
    for key, value in tokens.items():
        placeholder = "{{" + key + "}}"
        if placeholder not in text:
            continue
        if value is not None:
            # If value starts with sign AND text has a redundant sign right before
            # the placeholder (possibly with a space), strip that redundant sign.
            if value and value[0] in ('+', '-'):
                text = re.sub(r'([+\-])\s*' + re.escape(placeholder),
                              lambda m: value, text)
            text = text.replace(placeholder, value)
        else:
            text = re.sub(r'[+\-]?\s*' + re.escape(placeholder), '', text)
    # Strip any remaining unknown tokens
    text = re.sub(r'[+\-]?\s*\{\{[^}]+\}\}', '', text)
    text = re.sub(r'  +', ' ', text).strip()
    return text

def fix_direction_mismatch(text: str, tokens: dict) -> str:
    """Fix semantic contradictions after token resolution.

    The LLM sometimes writes directional words that contradict the resolved value:
      "gold higher -0.91%"  — 'higher' but value is negative → should be 'lower'
      "equities rise -0.23%" — 'rise' but value is negative → should be 'fall'
      "DXY retreats +0.18%" — 'retreats' but value is positive → should be 'firms'

    Runs AFTER resolve_tokens() so the actual sign is in the text.
    Only corrects the directional word — leaves the rest of the sentence intact.

    Pairs checked: gold, equities/stocks/spx, DXY.
    Pattern: (word) (±pct%) — if sign contradicts word, replace word.
    """
    import re

    pct_re = r'([+\-]\d+\.\d+%)'  # already-resolved signed pct

    CORRECTIONS = [
        # (asset_pattern, up_words, down_words)
        # up_words: should be replaced with down_word when value is negative
        # down_words: should be replaced with up_word when value is positive
        (r'gold',        ['higher', 'up', 'rises?', 'gains?', 'firms?'],   ['lower', 'down', 'falls?', 'drops?', 'weakens?']),
        (r'(?:equit\w*|stocks?|spx|s&p\s*500)', ['higher', 'up', 'rises?', 'gains?', 'bid'],  ['lower', 'down', 'falls?', 'drops?', 'slides?']),
        (r'dxy',         ['higher', 'up', 'firms?', 'rises?', 'strengthens?'], ['lower', 'down', 'retreats?', 'falls?', 'weakens?']),
    ]

    for asset_pat, up_words, down_words in CORRECTIONS:
        all_words = up_words + down_words
        word_re = r'(' + '|'.join(all_words) + r')'
        # Match: asset ... directional_word ... ±pct%
        pattern = r'(?i)(' + asset_pat + r'\s+(?:\w+\s+){0,3}?)' + word_re + r'(\s+)' + pct_re
        def _fix(m, _up=up_words, _down=down_words):
            prefix, direction, space, pct = m.group(1), m.group(2), m.group(3), m.group(4)
            is_negative = pct.startswith('-')
            direction_l = direction.lower()
            # Determine if current word implies UP or DOWN
            is_up_word = any(re.match('^' + w + '$', direction_l) for w in _up)
            if is_negative and is_up_word:
                # Replace with canonical down word
                return prefix + 'lower' + space + pct
            if not is_negative and not is_up_word:
                # Replace with canonical up word
                return prefix + 'higher' + space + pct
            return m.group(0)  # consistent — no change
        text = re.sub(pattern, _fix, text)

    return text


def sanitize_submateriality(text: str) -> str:
    """Replace sub-threshold directional moves with neutral language.

    Rule 8 thresholds: SPX >= 0.20%, DXY >= 0.15%, Gold >= 0.30%.
    The model frequently cites sub-threshold values as directional (e.g. "equities -0.04%")
    which violates Rule 8. This function replaces them with neutral language before
    sanitize_hardcoded_tokens runs, preventing a sub-threshold number from being tokenized
    and surfaced on the frontend as a directional move.

    Must run BEFORE sanitize_hardcoded_tokens.
    """
    import re as _re_mat

    pct_re = r'([+\-]?\s*\d+\.\d+%)'

    def _abs_val(s):
        m = _re_mat.search(r'[+\-]?\s*(\d+\.\d+)', s)
        return abs(float(m.group(0).replace(' ', ''))) if m else 0.0

    # Equities/SPX — threshold 0.20%: "equities -0.04%" → "equities flat"
    def _fix_equities(m):
        return "equities flat" if _abs_val(m.group(2)) < 0.20 else m.group(0)
    text = _re_mat.sub(
        r'((?:equit\w*|stocks?|spx)\s+)' + pct_re, _fix_equities, text, flags=_re_mat.IGNORECASE)

    # DXY — threshold 0.15%: "DXY +0.08%" → "DXY steady"
    def _fix_dxy(m):
        return "DXY steady" if _abs_val(m.group(2)) < 0.15 else m.group(0)
    text = _re_mat.sub(
        r'(\bDXY\s+)' + pct_re, _fix_dxy, text, flags=_re_mat.IGNORECASE)

    # Gold — threshold 0.30%: "; gold lower -0.15%" → "; gold steady"
    def _fix_gold(m):
        return m.group(1) + "gold steady" if _abs_val(m.group(3)) < 0.30 else m.group(0)
    text = _re_mat.sub(
        r'((?:;|\u2014)\s*)(gold\s+(?:higher|lower|up|down|rises?|falls?)?\s*)' + pct_re,
        _fix_gold, text, flags=_re_mat.IGNORECASE)

    return text


def sanitize_hardcoded_tokens(text: str, tokens: dict) -> str:
    """Fix Groq's most common token violation: hardcoding numeric values instead of placeholders.

    Groq frequently ignores the {{token}} instruction and writes numbers directly
    (e.g. "DXY retreats -0.23%" instead of "DXY retreats {{dxy_pct}}").

    ARCHITECTURE — must run BEFORE resolve_tokens(), not after.
    Replaces hardcoded numbers with {{placeholder}} tokens so that resolve_tokens()
    can substitute the live value normally. This also preserves {{token}} placeholders
    in index.json for JS live-refresh on the frontend.

    Running after resolve_tokens() silently fails when hardcoded value == token value
    (same execution cycle — the numbers match so the string is unchanged).

    Patterns: keyword-anchored regex — narrow enough to avoid touching FX pair levels.
    """
    import re

    pct_num = r'[+\-]?\s*\d+\.\d+%'  # signed or unsigned pct, e.g. -0.23% or +1.20%

    # DXY — "DXY retreats -0.23%" / "DXY firms +0.12%" / "DXY steady with -0.27%" (incoherent)
    text = re.sub(
        r'(DXY\s+(?:retreats?|rises?|firms?|falls?|drops?|higher|lower|up|down|strengthens?|weakens?|steady|consolidating)\s+(?:with\s+|at\s+)?)' + pct_num,
        lambda m: m.group(1) + "{{dxy_pct}}",
        text, flags=re.IGNORECASE
    )

    # Gold — "gold lower -0.56%" / "gold higher +1.2%"
    text = re.sub(
        r'(gold\s+(?:higher|lower|up|down|rises?|falls?|firms?|drops?)\s*)' + pct_num,
        lambda m: m.group(1) + "{{gold_pct}}",
        text, flags=re.IGNORECASE
    )

    # Equities — "equities rise +0.66%" / "equity gains +0.65%" / "stocks higher +0.44%"
    text = re.sub(
        r'((?:equit\w*|stocks?)\s+(?:gains?|rises?|falls?|slides?|higher|lower|up|down|bid|offered)\s*)' + pct_num,
        lambda m: m.group(1) + "{{spx_pct}}",
        text, flags=re.IGNORECASE
    )

    # VIX — "VIX at 19.4" / "VIX 19.36" — no % sign, avoid FX levels
    text = re.sub(
        r'(VIX\s+(?:at\s+)?)(\d+\.\d+)(?!%)',
        lambda m: m.group(1) + "{{vix}}",
        text, flags=re.IGNORECASE
    )

    # Bare chip format — model sometimes drops the directional word entirely, writing
    # "equities -0.04%; DXY +0.28%" instead of "equities lower {{spx_pct}}; DXY {{dxy_pct}}".
    # The directional-anchored patterns above don't match these — add a bare fallback.
    # Pattern: asset keyword immediately followed by a signed pct with no intervening verb.
    # Negative lookahead (?!\s*\w) prevents matching mid-sentence numeric contexts.
    text = re.sub(
        r'((?:equit\w*|stocks?|spx)\s+)([+\-]?\s*\d+\.\d+%)(?!\s*\w)',
        lambda m: m.group(1) + "{{spx_pct}}",
        text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'(\bDXY\s+)([+\-]?\s*\d+\.\d+%)(?!\s*\w)',
        lambda m: m.group(1) + "{{dxy_pct}}",
        text, flags=re.IGNORECASE
    )

    # Token misassignment guard: model sometimes writes "WTI oil rises {{spx_pct}}" by analogy
    # with the "gold lower {{gold_pct}} as equities firm {{spx_pct}}" example pattern.
    # {{spx_pct}} must ONLY appear adjacent to equities/stocks language — strip it from oil/WTI context.
    text = re.sub(
        r'(?i)(wti|crude|oil)\s+(?:oil\s+)?(?:rises?|falls?|higher|lower|up|down|firms?|drops?|surges?|dips?)\s*\{\{spx_pct\}\}',
        lambda m: m.group(0).replace("{{spx_pct}}", "").rstrip(),
        text
    )
    # Also catch the reverse: {{spx_pct}} immediately preceded by oil context with no directional word
    text = re.sub(
        r'(?i)(wti|crude|oil)[^.;]{0,40}\{\{spx_pct\}\}',
        lambda m: m.group(0).replace("{{spx_pct}}", "").rstrip(),
        text
    )

    return text


def validate_regime(ai_regime: str, tokens: dict) -> str:
    """Hard Python-side guardrail — overrides LLM regime when cross-asset evidence contradicts it.

    Implements the CANONICAL GUIDELINES stress scoring (§Stress scoring table):

      VIX > 30        → +3   (crisis / acute stress)
      VIX 25–30       → +2   (elevated stress — tail risk visible)
      VIX 18–25       → +1   (uncertainty zone)
      VIX < 18        → +0   (neutral to risk-on)
      Yield curve inverted (10Y < 3M)  → +1
      Gold intraday > +2.0%            → +1
      SPX intraday  < −1.5%            → +1
      MOVE index    > 100              → +1   ← threshold is 100, not 120

    Regime mapping:
      Score 0     → RISK-ON
      Score 1     → MIXED
      Score 2–3   → CAUTION   ← distinct tier between MIXED and RISK-OFF
      Score ≥ 4   → RISK-OFF
    """
    intraday = load_json(SITE_DIR / "intraday-data" / "quotes.json")
    if not intraday or not intraday.get("quotes"):
        return ai_regime  # no data to validate against — trust LLM

    q = intraday["quotes"]
    vix_close   = (q.get("vix")   or {}).get("close")
    spx_pct     = (q.get("spx")   or {}).get("pct")
    gold_pct    = (q.get("gold")  or {}).get("pct")
    move_close  = (q.get("move")  or {}).get("close")
    us3m_close  = (q.get("us3m")  or {}).get("close")
    us10y_close = (q.get("us10y") or {}).get("close")

    score = 0

    # VIX bands — tiered scoring per GUIDELINES §Stress scoring
    if vix_close is not None:
        if   vix_close > 30: score += 3
        elif vix_close > 25: score += 2
        elif vix_close >= 18: score += 1

    # Yield curve inversion (3m10s — Bloomberg-standard recession signal)
    if us3m_close is not None and us10y_close is not None:
        if us10y_close < us3m_close:
            score += 1

    # Gold safe-haven demand — threshold >2.0% filters routine noise
    if gold_pct is not None and gold_pct > 2.0:
        score += 1

    # Equity sell-off — −1.5% threshold per GUIDELINES (−0.5% was too sensitive)
    if spx_pct is not None and spx_pct < -1.5:
        score += 1

    # Bond market stress — MOVE > 100 is BofA/ICE "elevated" boundary
    if move_close is not None and move_close > 100:
        score += 1

    # Map score to canonical regime
    if score >= 4:
        correct_regime = "RISK-OFF"
    elif score >= 2:
        correct_regime = "CAUTION"
    elif score == 1:
        correct_regime = "MIXED"
    else:
        # Score 0: RISK-ON only when VIX clearly below uncertainty zone
        if vix_close is not None and vix_close < 18 and (spx_pct is None or spx_pct >= 0):
            correct_regime = "RISK-ON"
        else:
            correct_regime = "MIXED"

    regime = ai_regime.upper()

    # Clamp RISK-ON if evidence doesn't support it
    if regime == "RISK-ON" and correct_regime != "RISK-ON":
        return correct_regime

    # Clamp RISK-OFF if score doesn't reach crisis threshold
    if regime == "RISK-OFF" and score < 4:
        return correct_regime

    # Promote MIXED/NEUTRAL to CAUTION/RISK-OFF when score warrants it
    if regime in ("MIXED", "NEUTRAL") and score >= 2:
        return correct_regime

    # Promote MIXED/NEUTRAL → RISK-ON when score=0 and VIX clearly in risk-on zone
    # The LLM frequently defaults to MIXED even when all cross-asset data is benign.
    # A score of 0 (VIX < 18, no equity sell-off, no gold spike, no MOVE stress,
    # no yield curve inversion) is definitively RISK-ON — override the LLM.
    if regime in ("MIXED", "NEUTRAL") and correct_regime == "RISK-ON":
        return correct_regime

    # Clamp CAUTION downward when score doesn't support it
    if regime == "CAUTION" and score < 2:
        return correct_regime

    return regime  # LLM regime is consistent with evidence
def generate_narrative(api_key: str, context: str, _extra_keys=None, system_prompt: str = None) -> dict:
    """Generate narrative with one conditional retry on regime/tone mismatch.

    system_prompt: override NARRATIVE_SYSTEM (used for closed-market mode).
    When market_closed=True, NARRATIVE_SYSTEM_CLOSED is passed — it skips
    regime validation, token resolution, and tone retry (not applicable
    when markets are closed and prices are static Friday closing levels).

    First call: normal. If validate_regime() overrides the LLM regime AND the
    narrative tone contradicts the corrected regime, a single retry is issued
    with the correct regime pre-declared in the user message. This covers the
    persistent RISK-ON -> MIXED case where VIX 18-20 is risk-neutral but Groq
    reads SPX+/Gold- as unambiguously risk-on.

    Tone detection uses word-boundary regex, not substring match, to avoid false
    positives like "AUD/USD firms" triggering the "usd firms" risk-off signal.

    _extra_keys: remaining keys available for the retry if api_key hits daily limit
    during the retry call. Passed by call_with_key_rotation in main() so the retry
    can rotate without re-running the entire generate_narrative flow.
    """
    import re as _re
    import time as _time

    # Regex patterns — word-boundary aware to avoid pair-qualified false positives.
    # "AUD/USD firms" must NOT trigger usd firms; "USD firms" must.
    ON_PATTERNS  = [
        # Only flag explicitly GLOBAL USD weakness — pair-specific phrases like
        # "CAD leads on USD weakness" or "AUD/USD firms as USD weakens" describe
        # a single pair and are valid MIXED-regime language.
        r'\busd\s+broadly\s+offered\b',  # explicitly global
        r'\bbroadly\s+offered\b',        # explicitly global
        r'\brisk[\s\-]on\b',             # explicit regime label
        r'\bdxy\s+retreats?\b',          # explicit DXY directional move
        r'\busd\s+under\s+pressure\b',   # explicitly directional
        # Removed: r'\busd\s+offered\b', r'\busd\s+weakens?\b', r'\busd\s+weakness\b'
        # These are too broad — they fire on pair-specific driver language in MIXED narratives
        # (e.g. "CAD leads G8 on BoC hold and USD weakness") causing spurious tone-mismatch retries.
    ]
    OFF_PATTERNS = [
        r'\busd\s+broadly\s+bid\b', r'\busd\s+bid\b',
        r'(?<![/\w])usd\s+firms?\b',   # USD firms but NOT "AUD/USD firms"
        r'\bsafe[\s\-]haven\s+bid\b', r'\brisk[\s\-]off\b',
        r'\bdxy\s+firms?\b', r'\busd\s+strength\b', r'\bbroadly\s+bid\b',
    ]

    # Negation words for context-aware OFF_PATTERNS detection.
    # "safe-haven bid absent/muted" and "risk-off sentiment wanes/eases" are NEUTRAL/MIXED
    # language — not genuine risk-off signals. Both need negation-aware matching.
    _SHB_NEGATIONS   = r'(?:absent|muted|missing|limited|weak(?:ened)?|fading|reduced|suppressed|compromised)'
    _ROFF_NEGATIONS  = r'(?:wanes?|fades?|fading|eases?|easing|diminishes?|recedes?|subsides?|abates?|declining|reduced|limited)'

    def _tone_hits(text: str) -> tuple[list, list]:
        """Return lists of MATCHED STRINGS (not patterns) for readable log output.

        Context-aware for two OFF_PATTERNS that appear in MIXED narratives without being
        genuine risk-off signals:
          - "safe-haven bid absent/muted" — bid is NOT present
          - "risk-off sentiment wanes/eases/fades" — risk-off is RECEDING
        Both cases correctly describe a MIXED regime but were triggering spurious retries.
        """
        n = text.lower()
        def _matches(patterns):
            hits = []
            for p in patterns:
                if r'safe[\s\-]haven\s+bid' in p:
                    # Only fire if NOT followed/preceded by a negation (bid is absent/muted)
                    for m in _re.finditer(p, n):
                        after  = n[m.end():m.end()+35]
                        before = n[max(0, m.start()-15):m.start()]
                        if _re.search(_SHB_NEGATIONS, after) or _re.search(_SHB_NEGATIONS, before):
                            continue  # "safe-haven bid absent/muted" = neutral, not risk-off
                        hits.append(m.group(0).strip())
                elif r'risk[\s\-]off' in p:
                    # Only fire if NOT followed by a waning/easing negation
                    for m in _re.finditer(p, n):
                        after  = n[m.end():m.end()+30]
                        before = n[max(0, m.start()-15):m.start()]
                        if _re.search(_ROFF_NEGATIONS, after) or _re.search(_ROFF_NEGATIONS, before):
                            continue  # "risk-off wanes/eases/fades" = MIXED tone, not risk-off
                        hits.append(m.group(0).strip())
                else:
                    m = _re.search(p, n)
                    if m:
                        hits.append(m.group(0).strip())
            return hits
        return _matches(ON_PATTERNS), _matches(OFF_PATTERNS)

    def _parse_and_process(raw_text: str, tokens: dict) -> tuple[str, str, str]:
        """Parse Groq JSON, tokenize, resolve, validate. Returns (narrative, ai_regime, regime)."""
        raw = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        data = json.loads(raw)
        raw_narrative = data.get("narrative", "Market data unavailable.")
        materialized = sanitize_submateriality(raw_narrative)
        if materialized != raw_narrative:
            print(f"  INFO Sub-threshold move(s) neutralized (Rule 8 materiality).")
        tokenized  = sanitize_hardcoded_tokens(materialized, tokens)
        if tokenized != materialized:
            print(f"  INFO Hardcoded token values tokenized in narrative.")
        resolved   = resolve_tokens(tokenized, tokens)
        narrative  = fix_direction_mismatch(resolved, tokens)
        if narrative != resolved:
            print(f"  INFO Direction mismatch corrected in narrative.")
        ai_regime  = data.get("regime", "NEUTRAL").upper()
        regime     = validate_regime(ai_regime, tokens)
        return narrative, ai_regime, regime

    import re as _re

    # Forbidden-language patterns from NARRATIVE_SYSTEM Rule 2.
    # Used to detect violations post-generation and trigger a retry.
    NARRATIVE_FORBIDDEN = [
        r'\bsimmer\b', r'\bcreeping\s+in\b', r'\blingers?\b', r'\blooms?\b',
        r'\bon\s+the\s+back\s+of\b', r'\bamid\b', r'\bas\s+investors?\s+digest\b',
        r'\bmarket\s+participants?\b', r'\bcautious\s+tone\b',
        r'\binvestors?\s+cautious\b', r'\b(?:keep|keeps|kept)\s+\w+\s+cautious\b',
        r'\bcautious\b',  # standalone: "investors cautious", "cautious tone", "keeps cautious"
        r'\bTIER[-\s]?[12]\b',  # internal pipeline labels must never appear in narrative output
    ]

    def _forbidden_hits(text: str) -> list[str]:
        return [
            m.group(0) for pat in NARRATIVE_FORBIDDEN
            for m in [_re.search(pat, text, _re.IGNORECASE)] if m
        ]

    tokens = extract_tokens_from_intraday()

    # ── Closed-market shortcut ────────────────────────────────────────────────
    # When system_prompt=NARRATIVE_SYSTEM_CLOSED, markets are closed and prices
    # are static Friday closing levels. Skip token resolution, regime validation,
    # tone retry, and MIXED checks — none are applicable to a forward-looking
    # weekend brief. Return directly after the first Groq call.
    active_system = system_prompt if system_prompt is not None else NARRATIVE_SYSTEM
    if system_prompt is NARRATIVE_SYSTEM_CLOSED:
        raw = call_groq(api_key, active_system, f"Market data:\n\n{context}")
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        data = json.loads(raw)
        narrative_out = data.get("narrative", "Markets closed — next session preview unavailable.")

        # ── Weekend narrative verb sanitizer ─────────────────────────────────
        # Replaces active present-tense market-direction verbs that imply a live
        # market. The verb rules in NARRATIVE_SYSTEM_CLOSED allow present tense
        # only for news/tape descriptions ("dominates the weekend tape"). Market
        # direction verbs must always be conditional ("likely to", "may", "sets up").
        # This sanitizer catches LLM non-compliance as a deterministic fallback.
        #
        # Pattern: [subject] [active verb] [direction/object] [into/toward] [session]
        # Replace: "drives risk-off into" → "sets up risk-off for"
        #          "pushes X higher"      → "may push X higher"
        _NARR_VERB_SUBS = [
            # "X drives/pushes Y into the open" → "X sets up Y for the open"
            (r'\b(drives?|pushes?|sends?|pulls?|lifts?|weighs?)\s+(\w[\w\s\-/]{0,30}?)\s+into\s+(the\s+(?:Tokyo|Sydney|Monday|Sunday)\s+open)',
             r'sets up \2 for \3'),
            # "X drives risk-off/risk-on into" → "X sets up risk-off for"
            (r'\b(drives?|pushes?|sends?)\s+(risk-(?:on|off))\s+into\s+(the\s+\w+\s+open)',
             r'sets up \2 for \3'),
            # standalone: "X drives Y" without explicit session → add conditional
            (r'\b(drives?)\s+(risk-(?:on|off))\b',
             r'sets up \2'),
            # "from current levels" / "at current levels" / "from current price"
            # → "from Friday's close" (implies live market)
            (r'\b(from|at)\s+current\s+(?:levels?|prices?|rates?)\b',
             r'\1 Friday\'s close'),
        ]
        # "potentially Xing" → "likely to X" (weak modal → confident forward framing)
        # Uses a lookup dict for FX-relevant gerunds to produce the correct infinitive.
        # e.g. "potentially pressuring USD/JPY" → "likely to pressure USD/JPY"
        _FX_GERUND_INF = {
            'pressuring': 'pressure', 'weighing': 'weigh', 'sustaining': 'sustain',
            'supporting': 'support', 'pushing': 'push', 'pulling': 'pull',
            'lifting': 'lift', 'dragging': 'drag', 'driving': 'drive',
            'falling': 'fall', 'rising': 'rise', 'climbing': 'climb',
            'dropping': 'drop', 'extending': 'extend', 'reversing': 'reverse',
            'moving': 'move', 'heading': 'head', 'testing': 'test',
            'breaking': 'break', 'holding': 'hold', 'capping': 'cap',
        }
        def _fix_potentially(m):
            gerund = m.group(1).lower()
            return 'likely to ' + _FX_GERUND_INF.get(gerund, gerund[:-3] if gerund.endswith('ing') else gerund)
        narrative_out = _re.sub(r'\bpotentially\s+(\w+ing)\b', _fix_potentially, narrative_out, flags=_re.IGNORECASE)
        _narr_orig = narrative_out
        for _npat, _nrep in _NARR_VERB_SUBS:
            narrative_out = _re.sub(_npat, _nrep, narrative_out, flags=_re.IGNORECASE)
        if narrative_out != _narr_orig:
            print(f"  INFO Weekend narrative verb sanitizer fired.")

        # ── Narrative modal ("may/could/might") sanitizer ────────────────────
        # The NARRATIVE_SYSTEM_CLOSED prompt uses forward-looking language but
        # "may", "could", "might" produce weak, non-institutional output.
        # Replaces uncertainty modals with decisive "is likely to" framing.
        # Excludes "May" as a month name (preceded by digit or capitalized in date).
        # Excludes "may not" → keep as "is unlikely to" (handle separately).
        _narr_modal_orig = narrative_out
        _NARR_MODAL_FX_VERBS = (
            r'(?:compress|sustain|retest|move|extend|target|reach|test|push|pull|'
            r'drift|slide|rally|break|hold|continue|reverse|unwind|trigger|create|'
            r'set\s+up|offset|influence|propel|drive|pressure|support|weigh|cap|'
            r'limit|add|signal|indicate|suggest|find|face|see|remain|rise|fall|'
            r'climb|drop|gain|lose|stabilize|consolidate|attract|reflect|'
            r'dominate|impact|resolve|determine|affect|shape|dictate|guide|'
            r'keep|introduce|prevent|accelerate|delay|reinforce|amplify|reduce|'
            r'increase|narrow|widen|close|'
            r'weaken|strengthen|appreciate|depreciate|recover|retreat|advance|'
            r'outperform|underperform|firm|soften|ease|tighten)'
        )
        # "may [verb]" → "is likely to [verb]" (exclude "May" as month)
        narrative_out = _re.sub(
            rf'(?<!\d\s)(?<!\d)\bmay\s+({_NARR_MODAL_FX_VERBS})\b',
            r'is likely to \1',
            narrative_out, flags=_re.IGNORECASE
        )
        # "could [verb]" → "is likely to [verb]"
        narrative_out = _re.sub(
            rf'\bcould\s+({_NARR_MODAL_FX_VERBS})\b',
            r'is likely to \1',
            narrative_out, flags=_re.IGNORECASE
        )
        # "might [verb]" → "is likely to [verb]"
        narrative_out = _re.sub(
            rf'\bmight\s+({_NARR_MODAL_FX_VERBS})\b',
            r'is likely to \1',
            narrative_out, flags=_re.IGNORECASE
        )
        # "may be [verb/adj]" → "is likely to be [verb/adj]" (generic fallback)
        # Catches "may be supported", "may be influenced", "may be under pressure" etc.
        # Excludes "27 May be" (month name as subject via lookbehind on digit+space).
        narrative_out = _re.sub(
            r'(?<!\d\s)(?<!\d)\bmay\s+be\b',
            'is likely to be',
            narrative_out, flags=_re.IGNORECASE
        )
        if narrative_out != _narr_modal_orig:
            print(f"  INFO Narrative modal sanitizer: replaced uncertainty verb(s) in narrative.")

        # ── Subject-verb agreement: plural nouns + "is likely to" ────────────
        # The modal sanitizer replaces "tensions may weigh" → "tensions is likely to weigh"
        # which is grammatically incorrect (plural subject needs "are"). This pass
        # corrects "is likely to" → "are likely to" when a plural FX noun leads the clause.
        # Scope: same sentence only (bounded by period or semicolon).
        _PLURAL_NARR_SUBJECTS = (
            r'(?:tensions|forces|pressures|flows|movements|moves|developments|'
            r'risks|concerns|factors|headwinds|tailwinds|conditions|dynamics)'
        )
        # Matches "[plural noun] [0-6 words] is likely to" and replaces "is" with "are"
        narrative_out = _re.sub(
            rf'\b({_PLURAL_NARR_SUBJECTS})\b([^.;]{{0,60}}?)\bis\s+likely\s+to\b',
            r'\1\2are likely to',
            narrative_out, flags=_re.IGNORECASE
        )

        # ── Closed-market safe-haven guard ──────────────────────────────────
        # The live-path safe-haven guard (line ~4928) runs AFTER the closed-market
        # path returns — so closed-market narratives bypass it entirely.
        # This guard mirrors the live-path logic for the MIXED / RISK-ON cases.
        # RISK-OFF is excluded: safe-haven framing IS correct there.
        _cm_regime_peek = data.get("regime", "NEUTRAL").upper()
        if _cm_regime_peek not in ("RISK-OFF",):
            import re as _re_cm_sh
            _CM_SH_REPLACEMENTS = [
                # "safe-haven [word(s)] bid" → "geopolitical risk premium"
                (r'\bsafe[\s\-]haven(?:\s+\w+){0,2}\s+bid\b', 'geopolitical risk premium'),
                # "safe-haven demand" → "risk-premium demand"
                (r'\bsafe[\s\-]haven\s+demand\b', 'risk-premium demand'),
                # "safe-haven flows?" → "risk-premium flows"
                (r'\bsafe[\s\-]haven\s+flows?\b', 'risk-premium flows'),
            ]
            _cm_sh_orig = narrative_out
            for _sh_p, _sh_r in _CM_SH_REPLACEMENTS:
                narrative_out = _re_cm_sh.sub(_sh_p, _sh_r, narrative_out, flags=_re_cm_sh.IGNORECASE)
            if narrative_out != _cm_sh_orig:
                print(f"  INFO Closed-market safe-haven guard: replaced safe-haven framing with risk-premium language.")

        raw_regime = data.get("regime", "NEUTRAL").upper()
        valid_regimes = {"RISK-ON", "RISK-OFF", "MIXED", "CAUTION", "NEUTRAL"}
        raw_regime_valid = raw_regime if raw_regime in valid_regimes else "NEUTRAL"
        # Apply Python-side regime validator to closed-market regime too.
        # Prevents RISK-OFF from being returned when VIX data shows VIX < 18.
        # validate_regime() reads quotes.json independently — if unavailable it
        # falls through and trusts the LLM (same as the live path).
        regime_out = validate_regime(raw_regime_valid, tokens)
        if regime_out != raw_regime_valid:
            print(f"  INFO Closed-market regime corrected: {raw_regime_valid} → {regime_out} (validate_regime).")

        # ── Regime-narrative consistency guard ───────────────────────────────
        # Groq sometimes writes "setting up a risk-off environment" even when
        # validate_regime() has determined regime=RISK-ON (VIX<18, SPX+, gold-).
        # This is a direct factual contradiction: the narrative label and the
        # validated regime badge must agree.
        # Only active when regime_out is RISK-ON or RISK-OFF — not MIXED/CAUTION.
        import re as _re_rnc
        if regime_out == "RISK-ON":
            narrative_out = _re_rnc.sub(
                r'\bsetting\s+up\s+a\s+risk[\s\-]off\s+environment\b',
                'setting up a risk-on environment',
                narrative_out, flags=_re_rnc.IGNORECASE
            )
            _narr_roff_before = narrative_out
            narrative_out = _re_rnc.sub(
                r'\brisk[\s\-]off\s+environment\s+for\s+the\b',
                'risk-on environment for the',
                narrative_out, flags=_re_rnc.IGNORECASE
            )
            # Also catch standalone "risk-off" regime claims (not safe-haven, handled above)
            narrative_out = _re_rnc.sub(
                r'\ba\s+risk[\s\-]off\s+(?:open|session|bid|tone|backdrop)\b',
                'a risk-on open',
                narrative_out, flags=_re_rnc.IGNORECASE
            )
            if narrative_out != _narr_roff_before:
                print(f"  INFO Regime-narrative guard: corrected risk-off → risk-on framing in narrative.")
        elif regime_out == "RISK-OFF":
            narrative_out = _re_rnc.sub(
                r'\bsetting\s+up\s+a\s+risk[\s\-]on\s+environment\b',
                'setting up a risk-off environment',
                narrative_out, flags=_re_rnc.IGNORECASE
            )

        return {
            "narrative":    narrative_out,
            "regime":       regime_out,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tokens":       tokens,
        }

    # --- First attempt ---
    # Build upcoming calendar events block (next 48h) so Rule 5 forward-close
    # is grounded in real scheduled events, not the model's training prior.
    # build_calendar_block() returns "" silently if ff_calendar.json is unavailable.
    _upcoming_cal = build_calendar_block(window_hours_back=0, window_hours_ahead=48)

    # ── G8 narrative lead hint ────────────────────────────────────────────────
    # If the G8 scorecard has a triggered bilateral spread, inject the correct
    # lead format so the narrative uses LEADER/LAGGARD framing, not "CCY leads G8".
    # This eliminates the case where individual avg < 0.50% but spread >= 0.80%
    # triggers a bilateral signal that the LLM would otherwise frame as individual.
    _g8_narrative_hint = ""
    if _G8_SCORECARD:
        _sc = _G8_SCORECARD
        _l_ccy  = _sc.get("leader_ccy", "")
        _g_ccy  = _sc.get("laggard_ccy", "")
        _l_avg  = _sc.get("leader_avg", 0)
        _g_avg  = _sc.get("laggard_avg", 0)
        _sp     = _sc.get("spread", 0)
        _l_n    = _sc.get("leader_n", 0)
        _l_sign = "+" if _l_avg >= 0 else ""
        _g_sign = "+" if _g_avg >= 0 else ""
        _bilateral_triggered = _sp >= 0.80 and not (_l_avg >= 0.50 or _g_avg <= -0.50)
        # bilateral-only trigger (spread met, but neither individual end at ±0.50%)
        if _bilateral_triggered and not any(
            kw in (active_system or "") for kw in ["NARRATIVE_SYSTEM_CLOSED"]
        ):
            _g8_narrative_hint = (
                f"\nG8 NARRATIVE LEAD FORMAT (pre-computed — Rule 6 bilateral trigger active):\n"
                f"  Leader: {_l_ccy} {_l_sign}{_l_avg:.3f}% · Laggard: {_g_ccy} {_g_sign}{_g_avg:.3f}% · Spread: {_sp:.3f}%\n"
                f"  Individual trigger NOT met ({_l_ccy} avg {_l_sign}{_l_avg:.3f}% < +0.50%; "
                f"{_g_ccy} avg {_g_sign}{_g_avg:.3f}% > −0.50%).\n"
                f"  Bilateral spread {_sp:.3f}% ≥ 0.80% → BILATERAL format required.\n"
                f"  CORRECT lead: \"{_l_ccy}/{_g_ccy} bilateral divergence dominates — "
                f"{_l_ccy} {_l_sign}{_l_avg:.3f}% vs {_g_ccy} {_g_sign}{_g_avg:.3f}% (spread {_sp:.3f}%);\"\n"
                f"  WRONG lead:   \"{_l_ccy} leads G8 this week ({_l_sign}{_l_avg:.3f}%, n={_l_n} pairs)\" "
                f"← individual format when bilateral trigger is active\n"
            )

    # ── CB rates reminder block ───────────────────────────────────────────────
    # Rule 11 requires exact rates + spread whenever a CB differential is referenced.
    # Inject the current rates as a pre-computed reminder directly in the user_prompt
    # so the model has them immediately before the market data, not buried in context.
    _cb_rates_lines = []
    _cb_labels_map = {"USD": "Fed", "EUR": "ECB", "GBP": "BoE", "JPY": "BoJ",
                      "AUD": "RBA", "NZD": "RBNZ", "CAD": "BoC", "CHF": "SNB"}
    from datetime import date as _cbdate
    for _cb_ccy in ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]:
        _cb_data = load_json(SITE_DIR / "rates" / f"{_cb_ccy}.json")
        if _cb_data and _cb_data.get("observations"):
            _cb_obs = _cb_data["observations"]
            _cb_rate = _cb_obs[0].get("value", "")
            _cb_trend = "ON HOLD"
            if len(_cb_obs) >= 2:
                try:
                    _r0 = float(_cb_obs[0]["value"])
                    _last_change = None
                    for _o in _cb_obs[1:]:
                        if float(_o["value"]) != _r0:
                            _last_change = _cbdate.fromisoformat(_o["date"])
                            _r_prev = float(_o["value"])
                            break
                    if _last_change is None:
                        _cb_trend = "ON HOLD"
                    elif (_cbdate.today() - _last_change).days >= 90:
                        _cb_trend = "ON HOLD"
                    elif _r0 < _r_prev:
                        _cb_trend = "CUTTING"
                    elif _r0 > _r_prev:
                        _cb_trend = "HIKING"
                except (ValueError, KeyError):
                    pass
            if _cb_rate:
                _lbl = _cb_labels_map.get(_cb_ccy, _cb_ccy)
                _cb_rates_lines.append(f"  {_lbl} ({_cb_ccy}): {_cb_rate}% ({_cb_trend})")
    _cb_rates_block = ""
    if _cb_rates_lines:
        _cb_rates_block = (
            "RULE 11 — CB DIFFERENTIAL RATES (use these exact values when referencing differentials):\n"
            + "\n".join(_cb_rates_lines) + "\n"
            "When you mention a CB differential, ALWAYS include: Bank Name Rate% (STANCE) vs Bank Name Rate% (STANCE) — Xbp.\n"
            "CORRECT: 'Fed 4.50% (ON HOLD) vs BoJ 0.50% (ON HOLD) — 400bp carry'\n"
            "WRONG:   'Fed-BoJ rate differential sustains USD/JPY bid' ← no rates, no spread\n\n"
        )

    _narrative_user_prompt = (
        (_g8_narrative_hint + "\n" if _g8_narrative_hint else "")
        + (_upcoming_cal + "\n" if _upcoming_cal else "")
        + _cb_rates_block
        + "RULE 5 — FORWARD-LOOKING CLOSE: The final sentence MUST name a specific catalyst "
        "from the 'Economic Calendar — Key Events > UPCOMING' block above. "
        "Quote the event title VERBATIM from the UPCOMING block — do NOT paraphrase or generalise. "
        "CORRECT: 'Watch SNB Chairman Schlegel speech at 08:00 UTC.' "
        "CORRECT: 'Watch Core Retail Sales (USD) at 12:30 UTC.' "
        "WRONG:   'Watch BoE and ECB meetings next week for potential policy updates.' ← paraphrase + hedge language "
        "WRONG:   'Watch US GDP Friday' ← invented event not in calendar block "
        "WRONG:   'Watch RBNZ meeting May 27 02:00 UTC' ← event already occurred; NEVER cite a CB meeting or "
        "data release that has already passed (i.e. one that appears in the RELEASED section, or whose UTC "
        "datetime is before now). If the RBNZ/RBA/BoC/Fed/BoE/ECB/BoJ/SNB meeting already happened today "
        "or earlier, do NOT reference it as a forward catalyst — cite the NEXT future event instead. "
        "If no UPCOMING event exists in the calendar block, close with the next FUTURE technical level "
        "or the NEXT scheduled CB meeting (not one that already released today). "
        "NEVER use the phrase 'potential policy updates' or any hedge language in the close.\n\n"
    )

    # ── First-call robustness injections ─────────────────────────────────────
    # These blocks inject the most violation-prone constraints immediately before
    # the market data, where the model reads them last before generating.
    # Goal: eliminate the 4 most common retry triggers without changing any rule.
    #
    # Injection 1 — Regime + VIX (eliminates MIXED-style and safe-haven violations)
    # Pre-compute regime so the model cannot mis-score it from first principles.
    _pre_regime = validate_regime("MIXED", tokens)  # validate_regime ignores the hint arg, scores from data
    _pre_vix    = tokens.get("vix", "N/A")
    _pre_spx    = tokens.get("spx_pct", "N/A")
    _regime_injection = (
        f"PRE-COMPUTED REGIME (authoritative — do NOT re-score): {_pre_regime}\n"
        f"VIX: {_pre_vix} | SPX: {_pre_spx}\n"
    )
    if _pre_regime == "MIXED":
        _regime_injection += (
            "MIXED REGIME RULES — apply before writing:\n"
            "  • USD is neither broadly bid nor broadly offered — use pair-specific direction only\n"
            "  • Safe-haven framing (JPY/CHF) requires VIX > 25 — NOT applicable here\n"
            "  • Frame JPY/CHF moves as carry differential + geopolitical risk premium\n"
            "  • FORBIDDEN in MIXED: 'broadly bid', 'broadly offered', 'USD broadly ...', "
            "'safe-haven bid', 'safe-haven demand', 'safe-haven flows'\n"
        )
    elif _pre_regime in ("RISK-ON",):
        _regime_injection += (
            "RISK-ON REGIME: safe-haven framing for JPY/CHF not warranted. "
            "Do NOT use 'safe-haven bid' or 'safe-haven demand'.\n"
        )

    # Injection 2 — Forbidden language negative examples (eliminates forbidden-language retry)
    # Placed immediately before market data so the constraint is the last thing read.
    _forbidden_injection = (
        "FORBIDDEN WORDS — do NOT use any of these in the narrative:\n"
        "  simmer · creeping in · lingers · looms · weighs on · on the back of · amid\n"
        "  as investors digest · market participants · cautious tone · cautious\n"
        "Use ONLY direct action verbs: retreats · firms · bids · sells off · holds · "
        "breaks · compresses · widens · unwinds · surges · slips · tests\n"
    )

    # Injection 3 — Token values pre-resolved (eliminates missing-token retry)
    # Provide exact placement instructions so the model cannot omit a token.
    _tok_inj_parts = []
    if tokens.get("gold_pct"):
        _tok_inj_parts.append(f'{{{{gold_pct}}}} = {tokens["gold_pct"]} — use as "gold higher/lower {{{{gold_pct}}}}"')
    if tokens.get("spx_pct"):
        _tok_inj_parts.append(f'{{{{spx_pct}}}} = {tokens["spx_pct"]} — use as "equities {{{{spx_pct}}}}"')
    if tokens.get("dxy_pct"):
        _tok_inj_parts.append(f'{{{{dxy_pct}}}} = {tokens["dxy_pct"]} — use as "DXY {{{{dxy_pct}}}}"')
    if tokens.get("vix"):
        _tok_inj_parts.append(f'{{{{vix}}}} = {tokens["vix"]} — use as "VIX {{{{vix}}}}"')
    if _tok_inj_parts:
        _token_injection = (
            "MANDATORY TOKENS — all four MUST appear in your narrative using the {{token}} syntax:\n"
            + "\n".join(f"  {t}" for t in _tok_inj_parts)
            + "\nNEVER hardcode the numeric values — always use the {{token}} placeholder.\n"
        )
    else:
        _token_injection = ""

    # Injection 4 — Top headline driver (eliminates news-integration retry)
    # Extract the highest-impact headline and frame it as the lead driver instruction.
    _news_inj = load_json(SITE_DIR / "news-data" / "news.json", default={})
    _news_arts = _news_inj.get("articles", _news_inj.get("items", [])) if isinstance(_news_inj, dict) else _news_inj
    _top_headlines = [
        a for a in _news_arts
        if a.get("impact") == "high" and a.get("lang", "en") == "en"
    ][:3]
    _headline_injection = ""
    if _top_headlines:
        _hl_lines = [f"  • {h.get('time','')} [{h.get('cur','')}] {h.get('title','')}" for h in _top_headlines if h.get("title")]
        if _hl_lines:
            _headline_injection = (
                "TOP HEADLINES (RULE 12 — integrate at least one as the causal driver, not as a ticker readout):\n"
                + "\n".join(_hl_lines[:3])
                + "\nUse as causal clause: 'Dollar firms as stocks slide and crude surges' — "
                "NOT 'USD/JPY at 159.00' without naming why.\n"
            )

    _narrative_user_prompt += (
        _regime_injection + "\n"
        + _forbidden_injection + "\n"
        + (_token_injection + "\n" if _token_injection else "")
        + (_headline_injection + "\n" if _headline_injection else "")
        + f"Market data:\n\n{context}"
    )
    raw = call_groq(api_key, active_system, _narrative_user_prompt)
    narrative, ai_regime, regime = _parse_and_process(raw, tokens)
    print(f"  Tokens resolved: {tokens}")

    # FORBIDDEN LANGUAGE CHECK — fires when the LLM uses literary/hedge words
    # (e.g. "simmer", "creeping in", "amid") that are explicitly banned in Rule 2.
    # Retry once with the exact violations listed so the model can correct them.
    lang_violations = _forbidden_hits(narrative)
    if lang_violations:
        print(f"  WARNING Forbidden language detected: {lang_violations} — retrying...")
        lang_retry_prefix = (
            f"Your previous narrative contained forbidden language: {lang_violations}. "
            f"These words/phrases are explicitly prohibited (Rule 2 — DECLARATIVE LANGUAGE). "
            f"Replace them with direct action verbs: retreats, firms, bids, sells off, holds, breaks, "
            f"compresses, widens, unwinds. Do not use literary or hedge language. "
            f"Keep all other rules — forward-looking close, correct safe-haven routing (JPY/CHF/Gold only), "
            f"exact CB rates and spreads, live tokens. "
            f"Additionally: TIER-1 and TIER-2 are internal pipeline labels — never write them in narrative output. "
            f"Replace 'primary TIER-1 focus' -> 'primary event risk'; 'the TIER-1 [noun]' -> 'the key [noun]'; "
            f"any bare TIER-1/2 -> 'high-impact'.\n\n"
        )
        retry_keys_lang = [api_key] + (list(_extra_keys) if _extra_keys else [])
        raw_lang = None
        for r_idx, r_key in enumerate(retry_keys_lang):
            try:
                if r_idx > 0:
                    import time as _time2; _time2.sleep(KEY_SWITCH_PAUSE_DAILY)
                raw_lang = call_groq(r_key, active_system, lang_retry_prefix + f"Market data:\n\n{context}")
                break
            except RuntimeError as e:
                if "DAILY_LIMIT" in str(e) and r_idx < len(retry_keys_lang) - 1:
                    continue
                print(f"  ⚠️  Language retry unavailable ({e}). Using original narrative.")
                break
        if raw_lang is not None:
            try:
                narrative_lang, ai_regime_lang, regime_lang = _parse_and_process(raw_lang, tokens)
                remaining = _forbidden_hits(narrative_lang)
                if remaining:
                    print(f"  WARNING Language retry still has violations: {remaining}. Using retry anyway.")
                else:
                    print(f"  INFO Language retry resolved forbidden-language violations.")
                narrative, ai_regime, regime = narrative_lang, ai_regime_lang, regime_lang
            except Exception as e:
                print(f"  WARNING Language retry parse error ({e}). Using original narrative.")

    # TOKEN PRESENCE CHECK — fires when the LLM silently omits one or more of the 4 mandatory
    # NEWS INTEGRATION CHECK (Rule 12) — fires when the narrative is a pure ticker readout
    # with no reference to any headline driver. This is the "what" without the "why" failure:
    # the LLM correctly cites pair levels but ignores the news context that explains them.
    # Strategy: extract 2–3 key content words from each headline, check if ANY appear in the
    # narrative. If none do, fire one retry with an explicit quote of the top 2 headlines and
    # an instruction to integrate at least one driver into the narrative body.
    # This check only runs when headlines were present in the context.
    _news_check_headlines = []
    for _hl in high_impact[:4] if 'high_impact' in dir() else []:
        _hl_title = (_hl.get("title") or "").strip()
        if _hl_title:
            _news_check_headlines.append(_hl_title)
    if _news_check_headlines:
        # Extract content words (≥5 chars, not generic finance terms) from top headlines
        import re as _re_news
        _NEWS_STOPWORDS = {
            'ahead','after','before','below','above','their','there','about',
            'could','would','should','while','since','which','where','dollar',
            'euros','pound','market','session','trade','today','daily','weekly',
        }
        _headline_keywords: set = set()
        for _hl_t in _news_check_headlines[:3]:
            for _w in _re_news.findall(r'\b[a-z]{5,}\b', _hl_t.lower()):
                if _w not in _NEWS_STOPWORDS:
                    _headline_keywords.add(_w)
        _narrative_lower_news = narrative.lower()
        _news_integrated = any(_kw in _narrative_lower_news for _kw in _headline_keywords)
        if not _news_integrated and _headline_keywords:
            print(f"  WARNING News integration missing (Rule 12) — narrative contains no reference "
                  f"to headline keywords {sorted(_headline_keywords)[:8]}. Retrying...")
            _news_retry_prefix = (
                "Your previous narrative was a ticker readout — it cited price levels but "
                "did not integrate any of the session's news drivers (Rule 12). "
                "A desk note always explains WHY prices are moving, not just WHERE they are.\n"
                f"Top headlines from the context:\n"
                + "\n".join(f"  • {_h}" for _h in _news_check_headlines[:3])
                + "\n\nRewrite the narrative integrating at least one of these drivers as a causal "
                "clause (e.g. 'Dollar firms as stocks slide and crude surges — DXY {{dxy_pct}}...'). "
                "Keep ALL other rules: 4 tokens, forward-looking close, correct CB rates, no forbidden language.\n\n"
            )
            _news_retry_keys = [api_key] + (list(_extra_keys) if _extra_keys else [])
            for _nr_idx, _nr_key in enumerate(_news_retry_keys):
                try:
                    if _nr_idx > 0:
                        import time as _tnews; _tnews.sleep(KEY_SWITCH_PAUSE_DAILY)
                    _raw_news = call_groq(_nr_key, active_system,
                                         _news_retry_prefix + f"Market data:\n\n{context}")
                    _narr_news, _aireg_news, _reg_news = _parse_and_process(_raw_news, tokens)
                    # Accept the retry only if it actually integrated a headline keyword
                    _narr_news_lower = _narr_news.lower()
                    if any(_kw in _narr_news_lower for _kw in _headline_keywords):
                        narrative, ai_regime, regime = _narr_news, _aireg_news, _reg_news
                        print(f"  INFO News integration retry succeeded.")
                    else:
                        print(f"  WARNING News integration retry also skipped headlines — keeping retry anyway.")
                        narrative, ai_regime, regime = _narr_news, _aireg_news, _reg_news
                    break
                except RuntimeError as _enews:
                    if "DAILY_LIMIT" in str(_enews) and _nr_idx < len(_news_retry_keys) - 1:
                        continue
                    print(f"  ⚠️  News integration retry unavailable ({_enews}). Using original narrative.")
                    break

    # live tokens ({{gold_pct}}, {{spx_pct}}, {{dxy_pct}}, {{vix}}) from the narrative.
    # Root cause: model writes a G8/CB-focused narrative that doesn't mention cross-asset data,
    # so the tokens never appear. resolve_tokens() can only replace what's there; it can't inject
    # missing tokens. This check forces a retry with an explicit list of the missing tokens.
    # The retry prompt instructs the model to weave the missing cross-asset data into the narrative.
    MANDATORY_TOKENS = ["gold_pct", "spx_pct", "dxy_pct", "vix"]
    _missing_tokens = [t for t in MANDATORY_TOKENS if tokens.get(t) and tokens[t] not in narrative]
    # Also catch raw {{token}} still present (resolve failed or token value was None)
    _raw_tokens = [t for t in MANDATORY_TOKENS if "{{" + t + "}}" in narrative]
    # Also catch misattribution: token value present in narrative but near a DIFFERENT asset.
    # E.g. gold_pct="+0.07%" appears in "WTI ... up +0.07%" — value is correct but context is wrong.
    # Each token must appear within 80 chars of its expected asset keyword(s).
    _TOKEN_ASSET_KEYWORDS = {
        "gold_pct": ("gold",),
        "spx_pct":  ("equities", "spx", "s&p", "stocks"),
        "dxy_pct":  ("dxy", "dollar index", "usd index"),
        "vix":      ("vix",),
    }
    _misattributed = []
    _narrative_lower = narrative.lower()
    for _ta_tok in ["gold_pct", "spx_pct", "dxy_pct"]:  # vix check is straightforward
        _ta_val = tokens.get(_ta_tok)
        if not _ta_val or _ta_val not in narrative:
            continue  # already caught by _missing_tokens above
        _keywords = _TOKEN_ASSET_KEYWORDS.get(_ta_tok, ())
        _val_pos = narrative.find(_ta_val)
        if _val_pos == -1:
            continue
        _window = _narrative_lower[max(0, _val_pos - 80): _val_pos + 80]
        if not any(_kw in _window for _kw in _keywords):
            _misattributed.append(_ta_tok)
    _token_issues = list(dict.fromkeys(_missing_tokens + _misattributed + _raw_tokens))
    if _token_issues:
        print(f"  WARNING Narrative missing/unresolved tokens: {_token_issues} — retrying...")
        _tok_vals = {t: tokens.get(t, "N/A") for t in MANDATORY_TOKENS}
        # Build per-token insertion hints so the model knows exactly where and
        # how to splice each missing value into the narrative it already wrote.
        # This "surgical rewrite" approach is more reliable than a full regeneration
        # because the model only needs to add the missing tokens, not re-architect
        # the entire narrative — which risks dropping currently-present tokens.
        _missing_hints = []
        for _mt in _token_issues:
            _mv = _tok_vals.get(_mt, "N/A")
            if _mt == "gold_pct":
                _missing_hints.append(
                    f'  {{{{gold_pct}}}} = {_mv} — add after gold mention, e.g. "gold higher {{{{gold_pct}}}}"'
                    f' or append "; gold {{{{gold_pct}}}}" before the forward-looking close.'
                )
            elif _mt == "spx_pct":
                _missing_hints.append(
                    f'  {{{{spx_pct}}}} = {_mv} — add e.g. "; equities {{{{spx_pct}}}}"'
                    f' or "S&P {{{{spx_pct}}}}" before the forward-looking close.'
                )
            elif _mt == "dxy_pct":
                _missing_hints.append(
                    f'  {{{{dxy_pct}}}} = {_mv} — add e.g. "; DXY {{{{dxy_pct}}}}"'
                    f' before the forward-looking close.'
                )
            elif _mt == "vix":
                _missing_hints.append(
                    f'  {{{{vix}}}} = {_mv} — add e.g. "; VIX {{{{vix}}}}"'
                    f' (always write the token, never the raw number).'
                )
        _token_retry_prefix = (
            f"Your previous narrative was missing these mandatory live data tokens: {_token_issues}.\n"
            f"SURGICAL REWRITE: Keep your existing narrative structure and ONLY add the missing tokens.\n"
            f"Do NOT restructure, shorten, or change the lead. Just splice in the missing values:\n"
            + "\n".join(_missing_hints) + "\n"
            f"All four tokens are required: {{gold_pct}} {{spx_pct}} {{dxy_pct}} {{vix}}\n"
            f"Current values: gold_pct={_tok_vals['gold_pct']}  spx_pct={_tok_vals['spx_pct']}  "
            f"dxy_pct={_tok_vals['dxy_pct']}  vix={_tok_vals['vix']}\n"
            f"Keep all other rules — TIER-1 geopolitical lead (Rule 1), declarative language, forward-looking close.\n\n"
            f"Your previous narrative to amend (add missing tokens, change nothing else):\n{narrative}\n\n"
        )
        retry_keys_tok = [api_key] + (list(_extra_keys) if _extra_keys else [])
        raw_tok = None
        for r_idx, r_key in enumerate(retry_keys_tok):
            try:
                if r_idx > 0:
                    _time.sleep(KEY_SWITCH_PAUSE_DAILY)
                raw_tok = call_groq(r_key, active_system,
                                    _token_retry_prefix + f"Market data:\n\n{context}")
                break
            except RuntimeError as e:
                if "DAILY_LIMIT" in str(e) and r_idx < len(retry_keys_tok) - 1:
                    continue
                print(f"  ⚠️  Token retry unavailable ({e}). Using narrative with missing tokens.")
                break
        if raw_tok is not None:
            try:
                narrative_tok, ai_regime_tok, regime_tok = _parse_and_process(raw_tok, tokens)
                # Check if retry resolved the missing tokens
                still_missing = [t for t in MANDATORY_TOKENS if tokens.get(t) and tokens[t] not in narrative_tok]
                if still_missing:
                    print(f"  WARNING Token retry still missing: {still_missing} — attempting second retry...")
                    # Second token retry: direct append instruction using the narrative_tok text
                    # that was produced by the first retry. The model receives its own output
                    # and is told to append exactly the missing tokens — minimal change, maximum reliability.
                    _tok_vals2 = {t: tokens.get(t, "N/A") for t in still_missing}
                    _append_parts = []
                    for _st in still_missing:
                        _sv = _tok_vals2[_st]
                        if _st == "gold_pct":
                            _append_parts.append(f'gold {{{{gold_pct}}}}')
                        elif _st == "spx_pct":
                            _append_parts.append(f'equities {{{{spx_pct}}}}')
                        elif _st == "dxy_pct":
                            _append_parts.append(f'DXY {{{{dxy_pct}}}}')
                        elif _st == "vix":
                            _append_parts.append(f'VIX {{{{vix}}}}')
                    _append_clause = "; ".join(_append_parts)
                    _retry2_prefix = (
                        f"Your narrative is STILL missing tokens: {still_missing}.\n"
                        f"Take the narrative below and INSERT this clause before the final sentence:\n"
                        f'  "; {_append_clause}"\n'
                        f"Token values: " + ", ".join(f"{t}={v}" for t, v in _tok_vals2.items()) + "\n"
                        f"Use the token placeholders ({{{{gold_pct}}}}, {{{{spx_pct}}}}, etc.) — never the raw numbers.\n"
                        f"Change NOTHING else. Return the complete narrative with the clause inserted.\n\n"
                        f"Previous narrative to amend:\n{narrative_tok}\n\n"
                    )
                    raw_tok2 = None
                    for r2_idx, r2_key in enumerate([api_key] + (list(_extra_keys) if _extra_keys else [])):
                        try:
                            if r2_idx > 0:
                                _time.sleep(KEY_SWITCH_PAUSE_DAILY)
                            raw_tok2 = call_groq(r2_key, active_system,
                                                 _retry2_prefix + f"Market data:\n\n{context}")
                            break
                        except RuntimeError as e2:
                            if "DAILY_LIMIT" in str(e2) and r2_idx < 2:
                                continue
                            break
                    if raw_tok2 is not None:
                        try:
                            narrative_tok2, ai_regime_tok2, regime_tok2 = _parse_and_process(raw_tok2, tokens)
                            still_missing2 = [t for t in MANDATORY_TOKENS
                                              if tokens.get(t) and tokens[t] not in narrative_tok2]
                            if still_missing2:
                                print(f"  WARNING Second retry still missing: {still_missing2}. Using second retry anyway.")
                            else:
                                print(f"  INFO Second token retry resolved all missing tokens.")
                            narrative_tok, ai_regime_tok, regime_tok = narrative_tok2, ai_regime_tok2, regime_tok2
                        except Exception as e2:
                            print(f"  WARNING Second token retry parse error ({e2}). Using first retry.")
                    else:
                        print(f"  WARNING Second token retry unavailable. Using first retry anyway.")
                else:
                    print(f"  INFO Token retry resolved all missing tokens.")
                narrative, ai_regime, regime = narrative_tok, ai_regime_tok, regime_tok
            except Exception as e:
                print(f"  WARNING Token retry parse error ({e}). Using original narrative.")

    # SECOND FORBIDDEN LANGUAGE PASS — token retry may introduce new violations.
    # The first forbidden check ran before the token retry; this second pass catches
    # any forbidden language that the token-retry narrative re-introduced.
    lang_violations_2 = _forbidden_hits(narrative)
    if lang_violations_2:
        print(f"  WARNING Forbidden language in token-retry narrative: {lang_violations_2} — retrying...")
        lang_retry_prefix_2 = (
            f"Your previous narrative contained forbidden language: {lang_violations_2}. "
            f"These words/phrases are explicitly prohibited (Rule 2 — DECLARATIVE LANGUAGE). "
            f"Replace them with direct action verbs: retreats, firms, bids, sells off, holds, breaks, "
            f"compresses, widens, unwinds. Do not use 'cautious', 'on the back of', 'amid', or similar. "
            f"Keep all tokens ({{gold_pct}}, {{spx_pct}}, {{dxy_pct}}, {{vix}}) and all other rules.\n\n"
        )
        retry_keys_lang2 = [api_key] + (list(_extra_keys) if _extra_keys else [])
        for r_idx2, r_key2 in enumerate(retry_keys_lang2):
            try:
                if r_idx2 > 0:
                    import time as _time3; _time3.sleep(KEY_SWITCH_PAUSE_DAILY)
                raw_lang2 = call_groq(r_key2, active_system, lang_retry_prefix_2 + f"Market data:\n\n{context}")
                try:
                    narrative_lang2, ai_regime_lang2, regime_lang2 = _parse_and_process(raw_lang2, tokens)
                    remaining2 = _forbidden_hits(narrative_lang2)
                    if remaining2:
                        print(f"  WARNING Second language retry still has violations: {remaining2}. Using retry anyway.")
                    else:
                        print(f"  INFO Second language retry resolved forbidden-language violations.")
                    # Guard: ensure language retry didn't drop mandatory tokens
                    _lang2_missing = [t for t in MANDATORY_TOKENS if tokens.get(t) and tokens[t] not in narrative_lang2]
                    if _lang2_missing:
                        print(f"  WARNING Second language retry dropped tokens {_lang2_missing} — keeping pre-language narrative.")
                        # Do NOT update narrative — keep the token-correct version
                    else:
                        narrative, ai_regime, regime = narrative_lang2, ai_regime_lang2, regime_lang2
                except Exception as e2:
                    print(f"  WARNING Second language retry parse error ({e2}). Keeping current narrative.")
                break
            except RuntimeError as e2:
                if "DAILY_LIMIT" in str(e2) and r_idx2 < len(retry_keys_lang2) - 1:
                    continue
                print(f"  ⚠️  Second language retry unavailable ({e2}). Keeping current narrative.")
                break

    # MIXED STYLE COHERENCE CHECK — independent of regime match.
    # Fires when: LLM says regime=MIXED but narrative contains strong directional language
    # (e.g. "USD broadly offered", "broadly bid") that contradicts a MIXED tone.
    # These phrases are institutionally incorrect for MIXED — they imply full RISK-ON/RISK-OFF.
    MIXED_FORBIDDEN = [
        r'\busd\s+broadly\s+offered\b', r'\busd\s+broadly\s+bid\b',
        r'\bbroadly\s+offered\b', r'\bbroadly\s+bid\b',
        r'\busd\s+bid\s+across\b', r'\busd\s+offered\s+across\b',
    ]
    if regime == "MIXED":
        mixed_violations = [
            m.group(0) for pat in MIXED_FORBIDDEN
            for m in [_re.search(pat, narrative, _re.IGNORECASE)] if m
        ]
        if mixed_violations:
            print(f"  WARNING MIXED style violation: strong directional language in MIXED narrative "
                  f"({mixed_violations}). Retrying...")
            mixed_retry_prefix = (
                f"REGIME PRE-DECLARED: MIXED.\n"
                f"Your previous narrative contained strong directional language ({mixed_violations}) "
                f"which is inconsistent with a MIXED regime. "
                f"In a MIXED regime, USD is neither broadly bid nor broadly offered. "
                f"Use 'USD mixed', 'USD consolidating', or pair-specific language without broad USD directionality. "
                f"CRITICAL SAFE-HAVEN RULE for MIXED: do NOT use 'safe-haven bid', 'safe-haven demand', "
                f"or 'safe-haven flows' for JPY or CHF. VIX must be > 25 for safe-haven framing. "
                f"Instead, frame JPY/CHF as: carry differential + geopolitical risk premium. "
                f"CORRECT: '[Event] introduces geopolitical risk premium — [carry bp] carry sustains [pair] bid; "
                f"intervention risk above [level].' "
                f"MANDATORY: open with the highest-tier catalyst in the news data — "
                f"TIER 1 (leads narrative): geopolitical shock, commodity disruption, "
                f"macro Tier-1 surprise (NFP, CPI, FOMC, GDP), systemic risk. "
                f"TIER 2 (supporting only): PMI, retail sales, routine CB speech. "
                f"Use CB/yield differential ONLY when no TIER 1 event is in the headlines. "
                f"Do not list price levels without naming the driver.\n\n"
            )
            mixed_retry_prompt = f"{mixed_retry_prefix}Market data:\n\n{context}"
            retry_keys_mixed = [api_key] + (list(_extra_keys) if _extra_keys else [])
            raw_m = None
            for r_idx, r_key in enumerate(retry_keys_mixed):
                try:
                    if r_idx > 0:
                        print(f"  🔄 MIXED retry switching to Key {r_idx+1} — pausing {KEY_SWITCH_PAUSE_DAILY}s...")
                        _time.sleep(KEY_SWITCH_PAUSE_DAILY)
                    raw_m = call_groq(r_key, NARRATIVE_SYSTEM, mixed_retry_prompt)
                    break
                except RuntimeError as e:
                    if "DAILY_LIMIT" in str(e) and r_idx < len(retry_keys_mixed) - 1:
                        print(f"  ⛔ MIXED retry Key {r_idx+1} daily limit — trying next key...")
                        _time.sleep(KEY_SWITCH_PAUSE_DAILY)
                        continue
                    print(f"  ⚠️  MIXED retry unavailable ({e}). Using original narrative.")
                    break
            if raw_m is not None:
                narrative_m, ai_regime_m, regime_m = _parse_and_process(raw_m, tokens)
                mixed_v2 = [
                    m.group(0) for pat in MIXED_FORBIDDEN
                    for m in [_re.search(pat, narrative_m, _re.IGNORECASE)] if m
                ]
                if not mixed_v2:
                    print(f"  INFO MIXED retry resolved style violation. Regime: {regime_m}.")
                    narrative, regime = narrative_m, regime_m
                else:
                    print(f"  WARNING MIXED retry did not resolve style violation ({mixed_v2}). Using retry anyway.")
                    narrative, regime = narrative_m, regime_m

    # MIXED SAFE-HAVEN DETERMINISTIC GUARD — fires AFTER all retries.
    # If after all retries the narrative still contains "safe-haven bid" or "safe-haven demand"
    # in a MIXED regime, replace these phrases with carry + risk-premium equivalents.
    # This is a deterministic post-processing fallback — it fires when the model ignores
    # the Rule 3 instruction in the system prompt AND in the retry prefix.
    # The replacement preserves the geopolitical lead while correcting the framing.
    if regime in ("MIXED", "RISK-ON", "RISK_ON"):
        import re as _re_sh
        _SH_REPLACEMENTS = [
            # "safe-haven [CCY] bid" (e.g. "safe-haven USD bid") → "geopolitical risk premium"
            # The word-between pattern covers "safe-haven bid", "safe-haven USD bid",
            # "safe-haven JPY bid" — any variant where 0-2 words sit between the components.
            (r'\bsafe[\s\-]haven(?:\s+\w+){0,2}\s+bid\b', 'geopolitical risk premium'),
            # "safe-haven demand" → "risk-premium demand"
            (r'\bsafe[\s\-]haven\s+demand\b', 'risk-premium demand'),
            # "safe-haven flows? (into)" → "risk-premium flows (into)"
            (r'\bsafe[\s\-]haven\s+flows?\b', 'risk-premium flows'),
            # "JPY and CHF outperform" (safe-haven framing) → "JPY bid on carry + intervention risk"
            # Only fires when preceded by a risk-off lead (escalation drives / tensions drive)
            (r'(escalation|tensions|attack|conflict)\s+drives?\s+safe[\s\-]haven\s+bid\s+—\s+JPY\s+and\s+CHF\s+outperform',
             r'\1 introduces geopolitical risk premium — carry holds; JPY bid on 300bp Fed-BoJ differential, intervention risk above 160.00'),
        ]
        _narr_sh_orig = narrative
        for _sh_pat, _sh_rep in _SH_REPLACEMENTS:
            narrative = _re_sh.sub(_sh_pat, _sh_rep, narrative, flags=_re_sh.IGNORECASE)
        if narrative != _narr_sh_orig:
            print(f"  INFO {regime} safe-haven guard: replaced safe-haven framing with carry + risk-premium language.")


        on_hits, off_hits = _tone_hits(narrative)
        tone_mismatch = bool(on_hits or off_hits)

        if tone_mismatch:
            print(f"  WARNING Regime override: tone mismatch detected "
                  f"(LLM={ai_regime} -> {regime}). "
                  f"Risk-ON in text: {on_hits}. Risk-OFF in text: {off_hits}. Retrying...")

            retry_prefix = (
                f"REGIME PRE-DECLARED BY VALIDATOR: {regime}.\n"
                f"The cross-asset scoring (VIX={tokens.get('vix')}, "
                f"SPX={tokens.get('spx_pct')}, Gold={tokens.get('gold_pct')}) "
                f"does not meet the threshold for {ai_regime}. "
                f"Write the narrative with tone consistent with {regime}: "
                f"avoid strong directional USD language ('broadly offered', 'broadly bid', 'DXY retreats/firms'); "
                f"use 'USD mixed', 'USD consolidating', or pair-specific language without broad USD directionality. "
                f"IMPORTANT: 'DXY steady' must never be followed by a numeric value — write 'DXY steady' alone. "
                f"CRITICAL SAFE-HAVEN RULE for {regime}: VIX={tokens.get('vix')} is normal volatility — "
                f"do NOT use 'safe-haven bid', 'safe-haven demand', or 'safe-haven flows' for JPY or CHF. "
                f"Safe-haven framing requires RISK-OFF (VIX > 25). In {regime}, frame JPY/CHF moves as: "
                f"carry differential + geopolitical risk premium. "
                f"CORRECT example for MIXED + TIER-1 geopolitical event: "
                f"\"[Event] introduces geopolitical risk premium — [carry bp] carry sustains [pair] bid at [level]; "
                f"intervention risk above [threshold]. Gold {{{{gold_pct}}}}; equities {{{{spx_pct}}}}; "
                f"DXY {{{{dxy_pct}}}}; VIX {{{{vix}}}}. Watch [calendar event].\"\n"
                f"MANDATORY: open with the highest-tier catalyst in the news data — "
                f"TIER 1 (leads narrative): geopolitical shock, commodity supply disruption, "
                f"macro Tier-1 surprise (NFP, CPI, FOMC, GDP), systemic risk event. "
                f"TIER 2 (supporting only): PMI, retail sales, routine CB speech. "
                f"Use CB/yield differential ONLY when no TIER 1 event is in the headlines. "
                f"Do not list price levels without naming the specific driver.\n\n"
            )
            retry_prompt = f"{retry_prefix}Market data:\n\n{context}"

            # Retry key resolution: try current api_key first; if it hits DAILY_LIMIT,
            # rotate through _extra_keys (remaining keys passed in from main).
            # This avoids the retry being silently lost when api_key is already exhausted.
            retry_keys = [api_key] + (list(_extra_keys) if _extra_keys else [])
            raw2 = None
            for r_idx, r_key in enumerate(retry_keys):
                try:
                    if r_idx > 0:
                        print(f"  🔄 Retry switching to Key {r_idx+1} — pausing {KEY_SWITCH_PAUSE_DAILY}s...")
                        _time.sleep(KEY_SWITCH_PAUSE_DAILY)
                    raw2 = call_groq(r_key, NARRATIVE_SYSTEM, retry_prompt)
                    break
                except RuntimeError as e:
                    if "DAILY_LIMIT" in str(e) and r_idx < len(retry_keys) - 1:
                        print(f"  ⛔ Retry Key {r_idx+1} daily limit — trying next key...")
                        _time.sleep(KEY_SWITCH_PAUSE_DAILY)
                        continue
                    print(f"  ⚠️  Retry unavailable ({e}). Using original narrative.")
                    break

            if raw2 is None:
                return {
                    "narrative":    narrative,
                    "regime":       regime,
                    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "tokens":       tokens,
                }

            narrative2, ai_regime2, regime2 = _parse_and_process(raw2, tokens)

            on_hits2, off_hits2 = _tone_hits(narrative2)
            if not on_hits2 and not off_hits2:
                print(f"  INFO Retry resolved tone mismatch. Regime: {regime2}.")
            else:
                print(f"  WARNING Retry did not fully resolve tone mismatch "
                      f"(Risk-ON: {on_hits2}, Risk-OFF: {off_hits2}). Using retry narrative anyway.")
            narrative, regime = narrative2, regime2
        else:
            print(f"  INFO Regime override: LLM={ai_regime} -> {regime} "
                  f"(VIX={tokens.get('vix')}, SPX={tokens.get('spx_pct')}). "
                  f"Narrative tone is consistent with {regime} — no retry needed.")

    return {
        "narrative":    narrative,
        "regime":       regime,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tokens":       tokens,  # kept for JS live-refresh on subsequent renders
    }

def generate_signals(api_key: str, context: str, intraday_updated: str = "", system_prompt: str = None, _extra_keys=None) -> list:
    """Generate AI market signals with individual, staggered timestamps.

    system_prompt: override SIGNALS_SYSTEM (used for closed-market mode).

    Token budget: each Groq call's token usage is tracked against _SIGNALS_TOKEN_BUDGET.
    Optional repair retries (forbidden-language, safe-haven rewrite, empty-array retry)
    are skipped when the remaining budget is below their estimated cost. This prevents
    a single run from exhausting the signals key pool through cascading retries — the
    pattern that caused signals.json to freeze on high-activity days.

    Each signal gets a distinct time based on priority:
    - critical: anchored to most recent data update time
    - warning:  3 min earlier
    - info:     9 min earlier
    Post-processing guarantees no two signals share the same HH:MM.
    """
    global _signals_tokens_used
    from datetime import timedelta
    now_dt  = datetime.now(timezone.utc)
    now_utc = now_dt.strftime("%H:%M")

    def _tracked_call_groq(ak, sys_p, usr_p, max_tokens=800, mdl=MODEL):
        """call_groq wrapper that accumulates token usage into _signals_tokens_used."""
        global _signals_tokens_used
        result = call_groq(ak, sys_p, usr_p, max_tokens, mdl)
        _signals_tokens_used += getattr(call_groq, "_last_tokens", 0)
        return result

    def _budget_remaining() -> int:
        return max(0, _SIGNALS_TOKEN_BUDGET - _signals_tokens_used)

    def _budget_ok(estimated_cost: int) -> bool:
        """Return True if there is enough budget for a retry of estimated_cost tokens."""
        ok = _budget_remaining() >= estimated_cost
        if not ok:
            print(f"  INFO Budget guard: skipping optional retry (remaining={_budget_remaining()}, needed={estimated_cost})")
        return ok

    # Use intraday updated time as anchor if available and recent (< 30 min old)
    anchor_dt = now_dt
    if intraday_updated:
        try:
            upd = datetime.fromisoformat(intraday_updated.replace("Z", "+00:00"))
            age_min = (now_dt - upd).total_seconds() / 60
            if 0 <= age_min < 30:
                anchor_dt = upd
        except Exception:
            pass
    anchor_str = anchor_dt.strftime("%H:%M")

    active_system = system_prompt if system_prompt is not None else SIGNALS_SYSTEM

    # ── First-call robustness injections (signals) ────────────────────────────
    # Pre-compute and inject the constraints most violated on the first pass,
    # eliminating the downstream retry chains that drain the key pool.

    # Injection 1 — Regime + VIX (eliminates Rule-3 safe-haven rewrite retry)
    _sig_regime_meta = load_json(SITE_DIR / "ai-analysis" / "index.json", default={})
    _sig_regime   = (_sig_regime_meta.get("regime") or "MIXED").upper()
    _sig_tokens   = _sig_regime_meta.get("tokens") or {}
    _sig_vix      = _sig_tokens.get("vix", "N/A")
    _sig_spx      = _sig_tokens.get("spx_pct", "N/A")
    _regime_inj_s = (
        f"PRE-COMPUTED REGIME: {_sig_regime} | VIX: {_sig_vix} | SPX: {_sig_spx}\n"
    )
    try:
        _vix_float = float(_sig_vix)
    except (ValueError, TypeError):
        _vix_float = 99.0
    if _vix_float < 20 and _sig_regime in ("MIXED", "RISK-ON"):
        _regime_inj_s += (
            "SAFE-HAVEN FRAMING PROHIBITED (VIX < 20, regime not RISK-OFF):\n"
            "  Do NOT use: 'safe-haven demand', 'safe-haven bid', 'safe-haven flows' for JPY/CHF\n"
            "  Use carry-first framing: '[N]bp carry sustains bid; geopolitical risk premium secondary'\n"
        )
    elif _vix_float > 25:
        _regime_inj_s += "RISK-OFF confirmed (VIX > 25) — safe-haven framing for JPY/CHF is appropriate.\n"

    # Injection 2 — CB rates reminder (eliminates stale-rate violations without a separate guard)
    _sig_cb_lines = []
    _sig_cb_map = {"USD": "Fed", "EUR": "ECB", "GBP": "BoE", "JPY": "BoJ",
                   "AUD": "RBA", "CAD": "BoC", "CHF": "SNB", "NZD": "RBNZ"}
    for _sc_ccy, _sc_lbl in _sig_cb_map.items():
        _sc_rd = load_json(SITE_DIR / "rates" / f"{_sc_ccy}.json")
        if _sc_rd and _sc_rd.get("observations"):
            try:
                _sc_rate = float(_sc_rd["observations"][0]["value"])
                _sig_cb_lines.append(f"  {_sc_lbl} ({_sc_ccy}): {_sc_rate:.2f}%")
            except (ValueError, KeyError):
                pass
    _cb_inj_s = ""
    if _sig_cb_lines:
        _cb_inj_s = (
            "AUTHORITATIVE CB RATES — use ONLY these values, never training memory:\n"
            + "\n".join(_sig_cb_lines) + "\n"
        )

    # Injection 3 — COT extremes summary (pre-informs model; reduces neutral-COT signals)
    # Only pairs with |LF net| >= 10k are listed — the model knows anything below is not
    # a positioning extreme and should not generate a standalone COT signal for it.
    _cot_extreme_lines = []
    _cot_neutral_pairs = []   # used for pre-filter hint below
    _cot_ois_meeting = load_json(SITE_DIR / "meetings-data" / "meetings.json") or {}
    _cot_mtg_map = _cot_ois_meeting.get("meetings", {})
    for _ce_ccy in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]:
        _ce_d = load_json(SITE_DIR / "cot-data" / f"{_ce_ccy}.json")
        if not _ce_d:
            continue
        _ce_net = _ce_d.get("netPosition")
        if _ce_net is None:
            continue
        _ce_abs = abs(_ce_net)
        _ce_dir = "LONG" if _ce_net > 0 else "SHORT"
        _ce_lp  = _ce_d.get("longPositions", 0) or 0
        _ce_sp  = _ce_d.get("shortPositions", 0) or 0
        _ce_tot = _ce_lp + _ce_sp
        _ce_lp_pct = round(_ce_lp / _ce_tot * 100, 1) if _ce_tot > 0 else 50.0
        if _ce_abs >= 10_000:
            _bias_raw = _cot_mtg_map.get(_ce_ccy, {}).get("bias", "hold")
            _is_div = (_ce_net < -10_000 and _bias_raw == "hike") or (_ce_net > 10_000 and _bias_raw == "cut")
            _div_tag = " ⚠ COT-OIS DIVERGENCE" if _is_div else ""
            _cot_extreme_lines.append(
                f"  {_ce_ccy}: {_ce_dir} {_ce_net:+,} (long% {_ce_lp_pct}%){_div_tag}"
            )
        else:
            _cot_neutral_pairs.append(_ce_ccy)

    _cot_inj_s = ""
    if _cot_extreme_lines or _cot_neutral_pairs:
        _cot_inj_s = "COT POSITIONING SUMMARY (LF net ≥ 10k = actionable extreme; below = neutral, do NOT signal):\n"
        if _cot_extreme_lines:
            _cot_inj_s += "\n".join(_cot_extreme_lines) + "\n"
        if _cot_neutral_pairs:
            _cot_inj_s += (
                f"NEUTRAL (do NOT generate standalone COT signals for): "
                f"{', '.join(_cot_neutral_pairs)}\n"
            )

    _signals_prefix = (
        _regime_inj_s + "\n"
        + _cb_inj_s + "\n"
        + (_cot_inj_s + "\n" if _cot_inj_s else "")
    )

    user_prompt = (
        f"Current time UTC: {now_utc} | Most recent data update: {anchor_str}\n\n"
        + _signals_prefix
        + f"Market data:\n\n{context}"
    )
    raw = _tracked_call_groq(
        api_key, active_system,
        user_prompt,
        max_tokens=1100,  # raised — institutional multi-layer signals are 150-300 chars text + evidence
    )
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    start, end = raw.find("["), raw.rfind("]") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    try:
        signals = json.loads(raw)
    except json.JSONDecodeError as e:
        # JSON truncation recovery: retry once with an explicit token budget reminder.
        # Cause: multi-layer signal text is longer — 900 tokens may still be tight in rare cases.
        # Budget guard: only retry if we have enough budget remaining (~900 tokens estimated).
        if not _budget_ok(900):
            print(f"  WARNING Signals JSON parse error ({e}) — budget exhausted, skipping brevity retry.")
            return []
        print(f"  WARNING Signals JSON parse error ({e}) — retrying with brevity reminder...")
        brevity_prefix = (
            "IMPORTANT: Your previous response was truncated and produced invalid JSON. "
            "Keep each signal text under 180 characters. Keep evidence to 2 items max. "
            "The full JSON array MUST fit within the token budget — do not truncate.\n\n"
        )
        raw2 = _tracked_call_groq(
            api_key, active_system,
            brevity_prefix + user_prompt,
            max_tokens=900,
        )
        raw2 = raw2.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        s2, e2 = raw2.find("["), raw2.rfind("]") + 1
        if s2 >= 0 and e2 > s2:
            raw2 = raw2[s2:e2]
        signals = json.loads(raw2)  # raise if still broken

    # ── Degenerate-response guard ─────────────────────────────────────────────
    # If the LLM returned an empty array [] without any parse error, it produced
    # a degenerate response (common when a key is exhausted mid-quota but does not
    # raise a DAILY_LIMIT error — it simply silently returns []).
    # Budget guard: skip extra-key retry if budget is already tight.
    if not signals and _extra_keys:
        if not _budget_ok(1100):
            print(f"  WARNING LLM returned empty signals array — budget exhausted, skipping extra-key retry.")
        else:
            _empty_retry_keys = list(_extra_keys)
            for _ek_idx, _ek in enumerate(_empty_retry_keys):
                # Use a longer wait when retrying on extra keys — they may be under RPM
                # pressure from prior calls in the same run (narrative, drivers, etc.).
                _rpm_pause = 30 if _ek_idx == 0 else KEY_SWITCH_PAUSE_DAILY
                print(f"  WARNING LLM returned empty signals array — waiting {_rpm_pause}s then retrying "
                      f"with extra key {_ek_idx+1} ({mask_key(_ek)})...")
                time.sleep(_rpm_pause)
                try:
                    _raw_ek = _tracked_call_groq(_ek, active_system, user_prompt, max_tokens=1100)
                    _raw_ek = _raw_ek.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                    _s_ek, _e_ek = _raw_ek.find("["), _raw_ek.rfind("]") + 1
                    if _s_ek >= 0 and _e_ek > _s_ek:
                        _raw_ek = _raw_ek[_s_ek:_e_ek]
                    _sigs_ek = json.loads(_raw_ek)
                    if _sigs_ek:
                        print(f"  INFO Extra key {_ek_idx+1} produced {len(_sigs_ek)} signal(s) — using.")
                        signals = _sigs_ek
                        break
                    else:
                        print(f"  WARNING Extra key {_ek_idx+1} also returned empty — trying next.")
                except Exception as _ek_e:
                    print(f"  WARNING Extra key {_ek_idx+1} failed ({_ek_e}) — trying next.")
            if not signals:
                print("  WARNING All extra keys returned empty signals — proceeding with [].")

    # ── Resolve regime + tokens for Python-side guards ────────────────────────
    # generate_signals() runs after generate_narrative() writes ai-analysis/index.json.
    # Load regime and tokens from that file so safe-haven and COT-OIS guards have
    # the correct VIX value and market regime without touching call_with_key_rotation.
    _narrative_meta = load_json(SITE_DIR / "ai-analysis" / "index.json", default={})
    tokens = _narrative_meta.get("tokens") or {}
    regime = _narrative_meta.get("regime") or "MIXED"

    # ── Python-side signal validation ─────────────────────────────────────────
    # Enforce rules that the LLM violates despite prompt instructions.
    #
    # RULE 5 guard — drop signals whose text contains a neutral-positioning
    # admission ("neutral positioning", "flat positioning", "long%=4[0-9]" or
    # "long%=5[0-9]") at WARNING or CRITICAL priority.  INFO-level signals are
    # allowed through (they may still provide a technical read on the pair).
    import re as _re_guard
    NEUTRAL_PATTERNS = [
        r'neutral\s+positioning', r'flat\s+positioning',
        r'long%\s*=\s*[45]\d[\.\d]*\s*%?',  # 40–59% — trailing [-—] removed; LLM doesn't always include separator
        r'lf\s+net\s+(?:short|long)\s+[^\d]*(\d+)\b',  # caught via _is_neutral_cot_signal net-size check below
    ]
    def _is_neutral_cot_signal(s: dict) -> bool:
        if s.get("priority", "info").lower() not in ("critical", "warning"):
            return False
        text_lower = (s.get("text", "") + " " + s.get("title", "")).lower()
        if any(_re_guard.search(p, text_lower) for p in NEUTRAL_PATTERNS[:2]):  # text phrases
            return True
        if _re_guard.search(NEUTRAL_PATTERNS[2], text_lower):  # long% 40–59%
            return True
        # Numeric guard: catch near-zero net (absolute < 1 000 contracts) mentioned in text
        net_matches = _re_guard.findall(r'lf\s+net\s+(?:short|long)\s+[−\-]?([\d,]+)', text_lower)
        for nm in net_matches:
            try:
                if abs(int(nm.replace(',', ''))) < 1000:
                    return True
            except ValueError:
                pass
        return False

    # RULE 3 guard — safe-haven framing requires risk-off regime (VIX > 25).
    # When regime = MIXED or RISK-ON and VIX < 20, "safe-haven demand" framing is incorrect
    # (carry is the dominant driver) and the model must use carry-first framing.
    # This guard detects the violation and marks the signal for a targeted retry.
    # Matches: "safe-haven demand", "safe haven demand", "safe-haven bid" used as the
    # PRIMARY DRIVER (leading the text or immediately after the dash separator).
    # It does NOT strip signals that mention geopolitical risk as a SECONDARY layer
    # after carry-first framing (e.g. "...carry dominates; geopolitical risk secondary").
    import re as _re_sh
    _SH_LEAD_PATTERN = _re_sh.compile(
        r'^[^—–]*(—|–)\s*safe.haven\s+(demand|bid|flow)',
        _re_sh.IGNORECASE
    )
    _SH_ANY_PATTERN = _re_sh.compile(r'\bsafe.haven\s+(demand|bid|flow)\b', _re_sh.IGNORECASE)
    # Only apply when regime is MIXED or RISK-ON and VIX < 20
    _vix_val_sh   = float(tokens.get("vix", 99)) if tokens else 99
    _regime_sh    = (regime or "").upper()
    _sh_guard_active = _vix_val_sh < 20 and _regime_sh in ("MIXED", "RISK-ON", "RISK_ON")

    def _has_safe_haven_violation(s: dict) -> bool:
        """Return True if signal leads with safe-haven framing in a non-risk-off regime."""
        if not _sh_guard_active:
            return False
        text  = s.get("text", "")
        title = s.get("title", "")
        # Violation if safe-haven appears as the primary driver (leading the text)
        if _SH_LEAD_PATTERN.search(text):
            return True
        # Also flag if safe-haven appears in the title driver tag (e.g. "Geopolitical + safe-haven")
        if _re_sh.search(r'safe.haven', title, _re_sh.IGNORECASE) and _SH_ANY_PATTERN.search(text):
            return True
        return False

    # RULE 11 guard — keep only the first signal per FX pair (highest priority
    # wins; signals are already sorted critical→warning→info by the LLM).
    # Pair extracted from title prefix before " — ".
    def _pair_key(s: dict) -> str | None:
        title = s.get("title", "")
        m = _re_guard.match(r'^([A-Z]{3}/[A-Z]{3})', title)
        return m.group(1) if m else None

    seen_pairs: set[str] = set()
    _seen_topics: set[str] = set()   # Rule 11b — thematic title deduplication
    filtered_signals = []
    dropped_reasons  = []
    sh_violation_signals = []   # signals with safe-haven framing violations (need retry, not drop)
    for s in signals:
        # ── Regime-summary signal suppression ─────────────────────────────────
        # Drops info signals that duplicate what the Narrative panel already covers:
        # overall regime, VIX level, G8 scorecard rankings, retail sentiment overview.
        # Detection: info priority + empty/missing title + regime/VIX/scorecard keywords.
        _s_title    = s.get("title", "").strip()
        _s_text     = s.get("text",  "").lower()
        _s_priority = s.get("priority", "info")
        _REGIME_KEYWORDS = ("current regime", "regime is", "vix at", "spx up", "spx down",
                            "g8 scorecard", "g8 currency strength", "scorecard shows",
                            "retail traders", "90% short", "90% long", "leader with",
                            "laggard with", "market environment")
        _is_regime_summary = (
            _s_priority == "info"
            and not _s_title   # empty title is a strong signal
            and any(kw in _s_text for kw in _REGIME_KEYWORDS)
        )
        if _is_regime_summary:
            print(f"  INFO Regime-summary filter: dropped redundant context signal (duplicates Narrative panel)")
            dropped_reasons.append("'' — regime/VIX/scorecard summary (redundant with Narrative)")
            continue
        # Rule 5: drop neutral COT signals at WARN/CRIT
        if _is_neutral_cot_signal(s):
            print(f"  INFO Rule-5 filter: dropped neutral-COT {s.get('priority','?').upper()} signal — '{s.get('title','')}'")
            dropped_reasons.append(f"'{s.get('title','')}' — neutral COT positioning (not an extreme)")
            continue
        # G8 scorecard suppression — Python-side validator.
        # Drops any G8 outperformance/underperformance signal when none of the trigger
        # thresholds are met: individual avg ≥ ±0.50% OR bilateral spread ≥ 0.80%.
        # This is a hard guard because the prompt suppression rule alone is insufficient
        # when the model's context includes borderline values.
        _title_lower = s.get("title", "").lower()
        _is_g8_signal = ("g8" in _title_lower or "outperform" in _title_lower or "underperform" in _title_lower)
        if _is_g8_signal and _G8_SCORECARD:
            _sc = _G8_SCORECARD
            _leader_ok  = _sc.get("leader_avg", 0) >= 0.50
            _laggard_ok = _sc.get("laggard_avg", 0) <= -0.50
            _spread_ok  = _sc.get("spread", 0) >= 0.80
            if not (_leader_ok or _laggard_ok or _spread_ok):
                _why = (f"leader={_sc.get('leader_avg',0):+.3f}% (<+0.50%), "
                        f"laggard={_sc.get('laggard_avg',0):+.3f}% (>-0.50%), "
                        f"spread={_sc.get('spread',0):.3f}% (<0.80%) — all below trigger thresholds")
                print(f"  INFO G8-suppression: dropped '{s.get('title','')}' — {_why}")
                dropped_reasons.append(f"'{s.get('title','')}' — G8 thresholds not met ({_why})")
                continue
            # ── G8 currency-pair correction ───────────────────────────────────
            # If the trigger IS met (we're past the suppression above), verify the
            # LLM used the correct LEADER/LAGGARD pair from _G8_SCORECARD, not an
            # arbitrary pair comparison from the ranked table.
            # This catches the "EUR vs GBP" false comparison (spread 0.754%) when
            # the real leader/laggard pair is different (e.g. CAD vs EUR, 0.937%).
            _l_ccy_sc  = _sc.get("leader_ccy", "")
            _g_ccy_sc  = _sc.get("laggard_ccy", "")
            _l_avg_sc  = _sc.get("leader_avg", 0)
            _g_avg_sc  = _sc.get("laggard_avg", 0)
            _sp_sc     = _sc.get("spread", 0)
            _l_n_sc    = _sc.get("leader_n", 0)
            _g_n_sc    = _sc.get("laggard_n", 0)
            if _l_ccy_sc and _g_ccy_sc:
                _title_str = s.get("title", "")
                # Check if either the correct leader OR correct laggard appears in the title/evidence
                _ev_text = " ".join(s.get("evidence", []))
                _has_leader  = _l_ccy_sc in _title_str or _l_ccy_sc in _ev_text
                _has_laggard = _g_ccy_sc in _title_str or _g_ccy_sc in _ev_text
                if not (_has_leader and _has_laggard):
                    # Rewrite the evidence chips with the correct pair data
                    _sl = "+" if _l_avg_sc >= 0 else ""
                    _sg = "+" if _g_avg_sc >= 0 else ""
                    _correct_ev = [
                        f"{_l_ccy_sc} 1W avg: {_sl}{_l_avg_sc:.3f}% (n={_l_n_sc})",
                        f"{_g_ccy_sc} 1W avg: {_sg}{_g_avg_sc:.3f}% (n={_g_n_sc})",
                        f"G8 spread: {_sp_sc:.3f}%",
                    ]
                    # Rewrite the title to use the correct bilateral/individual format
                    _driver_suffix = ""
                    _m_suffix = _re_guard.search(r' — (.+)$', _title_str)
                    if _m_suffix:
                        _driver_suffix = " — " + _m_suffix.group(1)
                    if _spread_ok:
                        _correct_title = f"{_l_ccy_sc}/{_g_ccy_sc} — G8 Bilateral Theme{_driver_suffix}"
                    elif _leader_ok:
                        _correct_title = f"{_l_ccy_sc} — G8 Outperformance{_driver_suffix}"
                    else:
                        _correct_title = f"{_g_ccy_sc} — G8 Underperformance{_driver_suffix}"
                    print(f"  INFO G8-pair correction: '{_title_str}' → '{_correct_title}' "
                          f"(LLM used wrong comparison pair; corrected to {_l_ccy_sc}/{_g_ccy_sc})")
                    s["title"]    = _correct_title
                    s["evidence"] = _correct_ev
        # Rule 3: flag safe-haven framing violation in MIXED/RISK-ON + VIX < 20
        if _has_safe_haven_violation(s):
            print(f"  WARNING Rule-3 safe-haven framing violation (VIX={_vix_val_sh:.1f}, regime={_regime_sh}) "
                  f"— '{s.get('title','')}': safe-haven lead in non-risk-off regime. Queued for carry-first retry.")
            sh_violation_signals.append(s)
            # Keep the signal for now (don't drop) — it will be rewritten in the retry below
        # Rule 11: drop duplicate pairs (keep first / highest priority)
        pair = _pair_key(s)
        if pair:
            if pair in seen_pairs:
                print(f"  INFO Rule-11 filter: dropped duplicate-pair signal — '{s.get('title','')}'")
                dropped_reasons.append(f"'{s.get('title','')}' — duplicate pair (already covered)")
                continue
            seen_pairs.add(pair)
        # Rule 11b: drop duplicate thematic titles (e.g. two "Oil / Geopolitical" signals).
        # Normalize: lowercase, strip punctuation, collapse spaces → topic key.
        _topic_key = _re_guard.sub(r'[^a-z0-9]+', '_', s.get("title", "").lower()).strip('_')
        if _topic_key in _seen_topics:
            print(f"  INFO Rule-11b filter: dropped duplicate-topic signal — '{s.get('title','')}'")
            dropped_reasons.append(f"'{s.get('title','')}' — duplicate topic (already covered)")
            continue
        _seen_topics.add(_topic_key)
        filtered_signals.append(s)

    # ── CB evidence chip consolidation ────────────────────────────────────────
    # When a signal has two separate chips for central bank rates (e.g. "BoE: 3.75%"
    # and "Fed: 3.75%"), merge them into one chip: "BoE 3.75% vs Fed 3.75% — Xbp".
    # This enforces the evidence chip rule from GUIDELINES Rule 13 / SIGNALS_SYSTEM.
    import re as _re_chip
    _CB_LABELS = {"boe", "fed", "ecb", "boj", "rba", "rbnz", "boc", "snb", "pboc"}
    _RATE_CHIP_PAT = _re_chip.compile(
        r"^(BoE|Fed|ECB|BoJ|RBA|RBNZ|BoC|SNB|PBOC)[:\s]+(\d+\.\d+)%$", _re_chip.IGNORECASE
    )
    for _fs in filtered_signals:
        _ev = _fs.get("evidence")
        if not isinstance(_ev, list) or len(_ev) < 2:
            continue
        # Find all CB rate chips
        _cb_chips = {}   # label → (index, rate_float)
        for _ci, _chip in enumerate(_ev):
            _m = _RATE_CHIP_PAT.match(_chip.strip())
            if _m:
                _cb_chips[_m.group(1).upper()] = (_ci, float(_m.group(2)))
        if len(_cb_chips) < 2:
            continue
        # Build consolidated chip from the first two CB entries found
        _cb_items = list(_cb_chips.items())
        _lbl_a, (_idx_a, _rate_a) = _cb_items[0]
        _lbl_b, (_idx_b, _rate_b) = _cb_items[1]
        _bp = round(abs(_rate_a - _rate_b) * 100)
        _higher = _lbl_a if _rate_a >= _rate_b else _lbl_b
        _lower  = _lbl_b if _rate_a >= _rate_b else _lbl_a
        _r_high = max(_rate_a, _rate_b)
        _r_low  = min(_rate_a, _rate_b)
        _consolidated = (
            f"{_higher} {_r_high:.2f}% vs {_lower} {_r_low:.2f}% — {_bp}bp"
            if _bp > 0 else
            f"{_lbl_a} {_rate_a:.2f}% vs {_lbl_b} {_rate_b:.2f}% — 0bp"
        )
        # Replace the two separate chips with one consolidated chip at the lower index
        _keep_idx = min(_idx_a, _idx_b)
        _drop_idx = max(_idx_a, _idx_b)
        _ev[_keep_idx] = _consolidated
        del _ev[_drop_idx]
        print(f"  INFO CB-chip consolidation: merged '{_lbl_a}' + '{_lbl_b}' → '{_consolidated}' in '{_fs.get('title','')}'")


    # When Rule-3 violations were detected (safe-haven lead with VIX < 20 in MIXED/RISK-ON),
    # rewrite ONLY those signals in-place with carry-first framing.
    # Budget guard: skip rewrite if budget is tight — stale framing is acceptable over token exhaustion.
    if sh_violation_signals:
        if not _budget_ok(600):
            print(f"  INFO Budget guard: skipping Rule-3 safe-haven rewrite ({len(sh_violation_signals)} signals kept with warning).")
        else:
            violation_pairs = [s.get("title", "").split(" — ")[0] for s in sh_violation_signals]
            sh_retry_prefix = (
                f"RULE-3 CARRY-FIRST REWRITE REQUIRED.\n"
                f"Regime = {_regime_sh}. VIX = {_vix_val_sh:.1f} (normal volatility — NOT risk-off).\n"
                f"The following signal(s) used 'safe-haven demand/bid' as the primary driver, "
                f"which is incorrect when VIX < 20 and regime is MIXED or RISK-ON:\n"
                + "\n".join(f"  - {s.get('title','')} | text: \"{s.get('text','')}\"" for s in sh_violation_signals)
                + f"\n\nRewrite ONLY these signal(s) using CARRY-FIRST framing:\n"
                f"  • Lead with the carry differential (e.g. '300bp Fed-BoJ differential dominates')\n"
                f"  • Geopolitical/Middle East context is VALID as a secondary layer — name the risk\n"
                f"    and the specific threshold it puts in play (e.g. intervention risk at 160.00)\n"
                f"  • COT data (crowded carry shorts) is a valid amplifier\n"
                f"  • Do NOT use 'safe-haven demand', 'safe-haven bid', or 'safe-haven flow' in the text\n"
                f"  • CORRECT: 'USD/JPY at 159.78 — 300bp Fed-BoJ carry sustains bid; "
                f"Middle East tensions add intervention risk at 160.00. JPY COT LF short −65k (crowded).'\n"
                f"  • Keep the same priority level and evidence chips\n"
                f"Return ONLY a JSON array containing the rewritten signal(s) for: "
                f"{', '.join(violation_pairs)}. No other signals.\n\n"
            )
            raw_sh = _tracked_call_groq(api_key, active_system, sh_retry_prefix + user_prompt, max_tokens=600)
            raw_sh = raw_sh.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            ss, se = raw_sh.find("["), raw_sh.rfind("]") + 1
            if ss >= 0 and se > ss:
                raw_sh = raw_sh[ss:se]
            try:
                rewritten = json.loads(raw_sh)
                if isinstance(rewritten, list):
                    rewrite_map = {}
                    for r in rewritten:
                        if isinstance(r, dict):
                            rp = _pair_key(r)
                            if rp:
                                rewrite_map[rp] = r
                    applied = 0
                    for i, s in enumerate(filtered_signals):
                        rp = _pair_key(s)
                        if rp and rp in rewrite_map and _has_safe_haven_violation(s):
                            # Verify the rewrite doesn't still contain safe-haven lead
                            candidate = rewrite_map[rp]
                            if _has_safe_haven_violation(candidate):
                                print(f"  WARNING Rule-3 rewrite for {rp} still has safe-haven lead — keeping with warning.")
                            filtered_signals[i] = candidate
                            applied += 1
                            print(f"  INFO Rule-3 carry-first rewrite applied for {rp}.")
                    if applied == 0:
                        print(f"  WARNING Rule-3 carry-first rewrite: no signals matched for replacement.")
            except (json.JSONDecodeError, Exception) as _sh_e:
                print(f"  WARNING Rule-3 carry-first rewrite parse error ({_sh_e}) — original signals kept.")

    # ── Retry if post-filter count < 4 ────────────────────────────────────────
    # The model generated invalid signals that were dropped. Ask it to replace
    # them with valid alternatives, listing exactly what was rejected and why.
    # Budget guard: only retry if budget allows (~700 tokens estimated).
    MIN_SIGNALS = 4
    if len(filtered_signals) < MIN_SIGNALS and dropped_reasons:
        if not _budget_ok(700):
            print(f"  INFO Budget guard: skipping replacement retry (have {len(filtered_signals)} signals — budget exhausted).")
        else:
            n_needed = MIN_SIGNALS - len(filtered_signals)
            already_covered = ", ".join(seen_pairs) if seen_pairs else "none"
            already_themes = []
            for s in filtered_signals:
                title = s.get("title", "")
                if "G8 Bilateral" in title or "G8 Outperformance" in title:
                    already_themes.append("G8 Bilateral/Outperformance (already covered — do NOT generate another G8 signal)")
                    break
            already_themes_str = "; ".join(already_themes) if already_themes else "none"
            dropped_list = "\n".join(f"  - {r}" for r in dropped_reasons)
            retry_prefix = (
                f"Your previous response produced {len(signals)} signals, but {len(dropped_reasons)} were "
                f"invalid and discarded:\n{dropped_list}\n\n"
                f"Pairs already covered: {already_covered}.\n"
                f"Themes already covered: {already_themes_str}.\n"
                f"Generate EXACTLY {n_needed} additional valid signal(s) for DIFFERENT pairs not listed above. "
                f"Do NOT repeat the dropped signals. Do NOT generate signals for already-covered pairs. "
                f"Do NOT generate a G8 signal if one already exists — the G8 data is already represented. "
                f"Only include pairs where the data shows a genuine extreme, multi-layer read, or CB/geopolitical driver. "
                f"If no additional valid signals exist, return an empty array [].\n\n"
            )
            print(f"  INFO Retry: {len(filtered_signals)} signals after filter (need {MIN_SIGNALS}) — requesting {n_needed} replacement(s)...")
            raw_retry = _tracked_call_groq(
                api_key, active_system,
                retry_prefix + user_prompt,
                max_tokens=700,
            )
            raw_retry = raw_retry.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            rs, re_ = raw_retry.find("["), raw_retry.rfind("]") + 1
            if rs >= 0 and re_ > rs:
                raw_retry = raw_retry[rs:re_]
            try:
                retry_signals = json.loads(raw_retry)
                added = 0
                for s in retry_signals:
                    if not isinstance(s, dict):
                        continue
                    if _is_neutral_cot_signal(s):
                        print(f"  INFO Retry signal also failed Rule-5 — '{s.get('title','')}' — skipping.")
                        continue
                    pair = _pair_key(s)
                    if pair and pair in seen_pairs:
                        print(f"  INFO Retry signal also failed Rule-11 — '{s.get('title','')}' — skipping.")
                        continue
                    if pair:
                        seen_pairs.add(pair)
                    filtered_signals.append(s)
                    added += 1
                    if len(filtered_signals) >= MIN_SIGNALS:
                        break
                print(f"  INFO Retry added {added} replacement signal(s). Total: {len(filtered_signals)}.")
            except (json.JSONDecodeError, Exception) as e:
                print(f"  WARNING Retry parse error ({e}) — proceeding with {len(filtered_signals)} signals.")

    # Last-resort fallback: if still < 3 after retry, use unfiltered set
    if len(filtered_signals) < 3 and len(signals) >= 3:
        print(f"  WARNING Filters + retry left only {len(filtered_signals)} signals — using unfiltered set.")
        filtered_signals = signals

    # ── COT-OIS divergence enforcement ───────────────────────────────────────
    # A COT-OIS divergence (specs positioned AGAINST the CB forward bias) is the
    # highest-conviction structural signal in FX — it IS the signal per Rule 5.
    # When one exists and no filtered signal covers it, force a targeted retry.
    #
    # Divergence definition (aligned with SIGNALS_SYSTEM Rule 5):
    #   specs SHORT (netPosition < −10k)  AND  CB bias = hike   → divergence
    #   specs LONG  (netPosition > +10k)  AND  CB bias = cut    → divergence
    # Threshold 10k filters noise; genuine divergences in G8 are typically >>10k.
    #
    # Pair mapping: each CCY divergence maps to the primary USD pair to signal.
    _COT_OIS_PAIR_MAP = {
        "NZD": "NZD/USD", "AUD": "AUD/USD", "GBP": "GBP/USD",
        "EUR": "EUR/USD", "JPY": "USD/JPY", "CAD": "USD/CAD",
        "CHF": "USD/CHF",
    }
    _COT_MIN_ABS = 10_000  # contracts — below this is noise, not a signal

    try:
        _meetings_cot = load_json(SITE_DIR / "meetings-data" / "meetings.json") or {}
        _mtgs         = _meetings_cot.get("meetings", {})
        _active_divergences = []  # list of dicts: {ccy, pair, net, bias, hikeProb, cutProb}
        for _ccy, _pair in _COT_OIS_PAIR_MAP.items():
            _cot_d = load_json(SITE_DIR / "cot-data" / f"{_ccy}.json") or {}
            _net   = _cot_d.get("netPosition")
            if _net is None or abs(_net) < _COT_MIN_ABS:
                continue
            _mtg_d     = _mtgs.get(_ccy, {})
            _bias      = _mtg_d.get("bias", "hold")
            _hike_prob = _mtg_d.get("hikeProb") or 0
            _cut_prob  = _mtg_d.get("cutProb")  or 0
            # Divergence: specs short vs hike bias OR specs long vs cut bias
            _is_div = (
                (_net < -_COT_MIN_ABS and _bias == "hike") or
                (_net > +_COT_MIN_ABS and _bias == "cut")
            )
            if _is_div:
                _active_divergences.append({
                    "ccy":       _ccy,
                    "pair":      _pair,
                    "net":       _net,
                    "bias":      _bias,
                    "hike_prob": _hike_prob,
                    "cut_prob":  _cut_prob,
                })

        if _active_divergences:
            # Check which divergences are already covered by a filtered signal
            _covered_pairs = {_pair_key(s) for s in filtered_signals if _pair_key(s)}
            _uncovered = [d for d in _active_divergences if d["pair"] not in _covered_pairs]

            if _uncovered:
                _div_descriptions = []
                for _d in _uncovered:
                    _net_str  = f"{_d['net']:+,.0f}"
                    _dir_str  = "short" if _d["net"] < 0 else "long"
                    _bias_str = _d["bias"].upper()
                    _prob_str = ""
                    if _d["bias"] == "hike" and _d["hike_prob"]:
                        _prob_str = f" ({_d['hike_prob']}% priced)"
                    elif _d["bias"] == "cut" and _d["cut_prob"]:
                        _prob_str = f" ({_d['cut_prob']}% priced)"
                    _impl = (
                        "short-squeeze risk if CB delivers hike"
                        if _d["bias"] == "hike"
                        else "long-unwind risk if CB delivers cut"
                    )
                    _div_descriptions.append(
                        f"  {_d['pair']}: COT LF net {_dir_str} {_net_str} vs CB bias {_bias_str}{_prob_str} "
                        f"→ {_impl}"
                    )
                    print(f"  WARNING COT-OIS divergence uncovered: {_d['pair']} "
                          f"(net={_d['net']:+,}, bias={_d['bias']}) — requesting signal.")

                # ── Dedicated COT-OIS system prompt (minimal — no full SIGNALS_SYSTEM overhead) ──
                # Each divergence is called ONE AT A TIME with a self-contained ~300-token user
                # prompt instead of the full 11k context. This is the institutional pattern:
                # targeted single-asset call with only the data relevant to that signal.
                _CCY_CB_MAP_E = {
                    "NZD": "RBNZ", "AUD": "RBA", "CAD": "BoC", "GBP": "BoE",
                    "EUR": "ECB",  "JPY": "BoJ", "USD": "Fed", "CHF": "SNB",
                }
                _COT_OIS_SYSTEM = (
                    "You are an FX research analyst writing a single trading signal for a COT vs OIS divergence. "
                    "A COT-OIS divergence exists when speculative positioning (CFTC Commitment of Traders, "
                    "Large Futures net) is directionally opposed to the central bank's forward bias (OIS-implied). "
                    "This is the highest-conviction structural signal in FX — it means the market is positioned "
                    "AGAINST the direction the CB is signalling, creating an asymmetric squeeze risk.\n\n"
                    "OUTPUT: Return ONLY a valid JSON array containing exactly one signal object:\n"
                    "[\n"
                    "  {\n"
                    "    \"time\": \"\",\n"
                    "    \"priority\": \"warning\",\n"
                    "    \"title\": \"[PAIR] — COT vs OIS Divergence + [CB name]\",\n"
                    "    \"text\": \"[2-3 sentence signal. Lead with carry context and current pair level. "
                    "Name the COT net position (e.g. 'specs short −18.1k') and the CB bias explicitly. "
                    "State the squeeze/unwind implication. End with the next CB meeting date as catalyst — use ONLY the date from the DATA section provided; NEVER invent or recall a date from training.]\",\n"
                    "    \"evidence\": [\"COT LF net: [±Xk]\", \"[CB] bias: [hike/cut] ([X]% priced)\", \"[pair] at [level]\"]\n"
                    "  }\n"
                    "]\n\n"
                    "RULES:\n"
                    "- Write the text as a carry-first research note: establish WHY the position matters "
                    "(carry differential, fundamental backdrop) before naming the divergence.\n"
                    "- Only reference the CB for the pair being described. Never mention other CBs' hike/cut probabilities.\n"
                    "- priority = 'critical' if (hike/cut prob ≥ 80%) AND (|net| ≥ 30,000 contracts). Otherwise 'warning'.\n"
                    "- Complete the JSON fully. Do not truncate.\n"
                    "- CRITICAL: Use ONLY the dates provided in the user prompt DATA section. NEVER recall or invent CB meeting dates from training data.\n"
                    "- FORBIDDEN modal verbs: 'may', 'could', 'might'. Use decisive language: 'sets up', 'likely to', 'creates risk', 'points to'.\n"
                    "  Exception: 'May' as a month name is allowed (e.g. 'RBNZ meeting on 27 May').\n"
                    "- Return ONLY the JSON array. No preamble, no markdown."
                )

                _cot_keys = [api_key] + (list(_extra_keys) if _extra_keys else [])
                _added_cot_total = 0

                # ── Spot rate + carry rate lookup (loaded once, used per divergence) ────
                # Institutional standard: every COT-OIS signal must cite the current pair
                # level and carry differential — these are the two anchors a desk analyst
                # reads before writing any carry/positioning note.
                _SPOT_KEY_MAP = {
                    "NZD/USD": ("nzdusd", 4), "AUD/USD": ("audusd", 4),
                    "GBP/USD": ("gbpusd", 4), "EUR/USD": ("eurusd", 4),
                    "USD/JPY": ("usdjpy", 2), "USD/CAD": ("usdcad", 4),
                    "USD/CHF": ("usdchf", 4),
                }
                _cot_iq_raw = load_json(SITE_DIR / "intraday-data" / "quotes.json") or {}
                _cot_iq     = _cot_iq_raw.get("quotes", {})
                _cot_rates: dict = {}
                for _cr_ccy in list(_CCY_CB_MAP_E.keys()) + ["USD"]:
                    _rd = load_json(SITE_DIR / "rates" / f"{_cr_ccy}.json")
                    if _rd and _rd.get("observations"):
                        try:
                            _cot_rates[_cr_ccy] = float(_rd["observations"][0]["value"])
                        except (ValueError, KeyError):
                            pass

                for _d in _uncovered:
                    _d_ccy   = _d["ccy"]
                    _d_pair  = _d["pair"]
                    _d_net   = _d["net"]
                    _d_bias  = _d["bias"]
                    _d_hprob = _d.get("hike_prob", 0)
                    _d_cprob = _d.get("cut_prob", 0)
                    _d_cb    = _CCY_CB_MAP_E.get(_d_ccy, _d_ccy + " CB")
                    _d_dir   = "short" if _d_net < 0 else "long"
                    _d_net_k = f"{_d_net/1000:+.1f}k"
                    _d_impl  = "short-squeeze risk" if _d_bias == "hike" else "long-unwind risk"
                    _d_prob_str = (f" ({_d_hprob}% hike priced)" if _d_bias == "hike" and _d_hprob
                                   else f" ({_d_cprob}% cut priced)" if _d_bias == "cut" and _d_cprob
                                   else "")
                    _d_priority = (
                        "critical"
                        if ((_d_bias == "hike" and _d_hprob >= 80) or
                            (_d_bias == "cut"  and _d_cprob >= 80)) and abs(_d_net) >= 30_000
                        else "warning"
                    )

                    # Live spot rate from quotes.json
                    _spot_key, _spot_dec = _SPOT_KEY_MAP.get(_d_pair, (None, 4))
                    _d_spot = None
                    if _spot_key and _cot_iq.get(_spot_key, {}).get("close"):
                        try:
                            _d_spot = round(float(_cot_iq[_spot_key]["close"]), _spot_dec)
                        except (ValueError, TypeError):
                            pass
                    _d_spot_str = f"  Spot rate: {_d_pair} at {_d_spot}\n" if _d_spot else ""

                    # Carry differential: base_ccy rate vs quote_ccy rate
                    _d_carry_str = ""
                    _d_parts = _d_pair.split("/")
                    if len(_d_parts) == 2:
                        _base_ccy_c, _quote_ccy_c = _d_parts
                        _rb_c = _cot_rates.get(_base_ccy_c)
                        _rq_c = _cot_rates.get(_quote_ccy_c)
                        if _rb_c is not None and _rq_c is not None:
                            _diff_bp_c = round((_rb_c - _rq_c) * 100)
                            _carry_dir = _base_ccy_c if _diff_bp_c > 0 else (_quote_ccy_c if _diff_bp_c < 0 else "neutral")
                            _d_carry_str = (
                                f"  Carry: {_base_ccy_c} {_rb_c}% vs {_quote_ccy_c} {_rq_c}% "
                                f"= {_diff_bp_c:+d}bp (carry favors {_carry_dir})\n"
                            )

                    # Next CB meeting date
                    _d_next_mtg = _mtgs.get(_d_ccy, {}).get("nextMeeting", "")
                    _d_mtg_str  = f"  Next {_d_cb} meeting: {_d_next_mtg}\n" if _d_next_mtg else ""

                    # Self-contained user prompt — spot + carry + COT + CB bias.
                    # Every field that a desk analyst cites is present; LLM has no reason to hallucinate.
                    _d_user = (
                        f"Generate a COT vs OIS divergence signal for {_d_pair}.\n\n"
                        f"DATA (use exact values — do not substitute):\n"
                        f"  Pair: {_d_pair}\n"
                        f"{_d_spot_str}"
                        f"{_d_carry_str}"
                        f"  CFTC COT LF net: {_d_net_k} ({_d_dir})\n"
                        f"  Central bank: {_d_cb}\n"
                        f"  CB bias: {_d_bias}{_d_prob_str}\n"
                        f"{_d_mtg_str}"
                        f"  Divergence implication: {_d_impl}\n"
                        f"  Priority: {_d_priority}\n\n"
                        f"Write the signal as a JPMorgan or Goldman FX desk note. "
                        f"Lead with carry context and the current pair level. "
                        f"Name the COT net ({_d_net_k}) and CB bias explicitly. "
                        f"State the squeeze/unwind implication. "
                        f"End with the next {_d_cb} meeting as catalyst — use ONLY the date '{_d_next_mtg}' from the DATA above; NEVER invent a date. " if _d_next_mtg else f"End with the next {_d_cb} meeting as catalyst. "
                        f"2-3 sentences in text field."
                    )

                    _raw_cot_ois = None
                    for _cot_k_idx, _cot_k in enumerate(_cot_keys):
                        # Budget guard: each COT-OIS call costs ~550 tokens
                        if not _budget_ok(550):
                            print(f"  INFO Budget guard: skipping COT-OIS call for {_d_pair} (budget exhausted).")
                            break
                        try:
                            if _cot_k_idx > 0:
                                import time as _time_cot; _time_cot.sleep(KEY_SWITCH_PAUSE_DAILY)
                            _raw_cot_ois = _tracked_call_groq(
                                _cot_k, _COT_OIS_SYSTEM, _d_user,
                                max_tokens=550,
                            )
                            break
                        except RuntimeError as _cot_k_e:
                            _cot_k_e_str = str(_cot_k_e)
                            if _cot_k_idx < len(_cot_keys) - 1:
                                if "DAILY_LIMIT" in _cot_k_e_str:
                                    import time as _time_cot_dl; _time_cot_dl.sleep(KEY_SWITCH_PAUSE_DAILY)
                                    continue
                                elif "RATE_LIMIT" in _cot_k_e_str:
                                    import time as _time_cot_rl; _time_cot_rl.sleep(KEY_SWITCH_PAUSE_RATE)
                                    continue
                            print(f"  WARNING COT-OIS key {_cot_k_idx+1} unavailable ({_cot_k_e}).")
                            break

                    if _raw_cot_ois:
                        _raw_cot_ois = _raw_cot_ois.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                        _cs, _ce = _raw_cot_ois.find("["), _raw_cot_ois.rfind("]") + 1
                        if _cs >= 0 and _ce > _cs:
                            _raw_cot_ois = _raw_cot_ois[_cs:_ce]
                        try:
                            _cot_ois_signals = json.loads(_raw_cot_ois)
                            for _s in _cot_ois_signals:
                                if not isinstance(_s, dict):
                                    continue
                                _p = _pair_key(_s)
                                if _p and _p in _covered_pairs:
                                    print(f"  INFO COT-OIS returned already-covered pair {_p} — skipping.")
                                    continue
                                if _p:
                                    _covered_pairs.add(_p)
                                    seen_pairs.add(_p)
                                filtered_signals.append(_s)
                                _added_cot_total += 1
                            print(f"  INFO COT-OIS signal generated for {_d_pair}. Total: {len(filtered_signals)}.")
                            continue  # success — move to next divergence
                        except (json.JSONDecodeError, Exception) as _cot_e:
                            print(f"  WARNING COT-OIS parse error for {_d_pair} ({_cot_e}) — using structured fallback.")

                    # Structured fallback — fires when LLM call failed OR parse error.
                    # Institutional standard: lead with carry context + pair level (same structure
                    # as the LLM-generated version). All values come from DATA already computed above.
                    print(f"  INFO COT-OIS structured fallback for {_d_pair}.")

                    # Build carry sentence: "[PAIR] at [spot] — [X]bp carry favors [side]"
                    if _d_spot and _d_carry_str:
                        _carry_parts = _d_carry_str.strip().split("=")
                        _carry_bp_clause = _carry_parts[1].strip() if len(_carry_parts) > 1 else ""
                        _fb_lead = (
                            f"{_d_pair} at {_d_spot} — "
                            f"{_carry_bp_clause.split('(')[0].strip()} differential; "
                            f"specs {_d_dir} {_d_net_k} (CFTC LF) against {_d_cb} {_d_bias} bias{_d_prob_str}."
                        )
                    elif _d_spot:
                        _fb_lead = (
                            f"{_d_pair} at {_d_spot}; "
                            f"specs {_d_dir} {_d_net_k} (CFTC LF) against {_d_cb} {_d_bias} bias{_d_prob_str}."
                        )
                    else:
                        _fb_lead = (
                            f"Specs {_d_dir} {_d_net_k} (CFTC LF) against {_d_cb} {_d_bias} bias{_d_prob_str} in {_d_pair}."
                        )

                    _fb_catalyst = (
                        f"Watch {_d_next_mtg} {_d_cb} for directional resolution."
                        if _d_next_mtg
                        else f"Watch next {_d_cb} meeting for directional resolution."
                    )
                    _fb_text = (
                        f"{_fb_lead} "
                        f"This COT-OIS divergence creates {_d_impl} if {_d_cb} follows through on forward guidance. "
                        f"{_fb_catalyst}"
                    )

                    _fb_evidence = [f"COT LF net: {_d_net_k}", f"{_d_cb} bias: {_d_bias}{_d_prob_str}"]
                    if _d_spot:
                        _fb_evidence.append(f"{_d_pair} spot: {_d_spot}")

                    _fb_signal = {
                        "time": "",
                        "priority": _d_priority,
                        "title": f"{_d_pair} — COT vs OIS Divergence + {_d_cb}",
                        "text": _fb_text,
                        "evidence": _fb_evidence,
                    }
                    _p_fb = _pair_key(_fb_signal)
                    if not _p_fb or _p_fb not in _covered_pairs:
                        filtered_signals.append(_fb_signal)
                        if _p_fb:
                            _covered_pairs.add(_p_fb)
                            seen_pairs.add(_p_fb)
                        _added_cot_total += 1
                        print(f"  INFO COT-OIS fallback added for {_d_pair}. Total: {len(filtered_signals)}.")

                if _added_cot_total:
                    print(f"  INFO COT-OIS enforcement complete: {_added_cot_total} signal(s) added.")
    except Exception as _cot_ois_outer_e:
        print(f"  WARNING COT-OIS enforcement check failed ({_cot_ois_outer_e}) — skipping.")

    signals = filtered_signals

    # ── Weekend signal cap ────────────────────────────────────────────────────
    # When market is closed, cap at 6 signals max (terminal panel space + UX).
    # Priority order is preserved: critical → warning → info (LLM already sorted).
    # COT-OIS enforcement may add signals past the cap — trim here deterministically.
    MAX_SIGNALS_WEEKEND = 6
    if system_prompt is SIGNALS_SYSTEM_CLOSED and len(signals) > MAX_SIGNALS_WEEKEND:
        print(f"  INFO Weekend cap: trimming {len(signals)} → {MAX_SIGNALS_WEEKEND} signals.")
        signals = signals[:MAX_SIGNALS_WEEKEND]

    # Fallback stagger offsets (minutes before anchor) — used when AI assigns duplicate times
    PRIORITY_OFFSET = {"critical": 0, "warning": 3, "info": 9}
    seen_times = set()
    result = []
    for i, s in enumerate(signals):
        if not isinstance(s, dict):
            continue
        priority = s.get("priority", "info").lower()
        ai_time  = s.get("time", "")

        # Use AI time if valid and unique, otherwise compute from anchor
        time_str = ai_time if (ai_time and len(ai_time) == 5 and ":" in ai_time
                               and ai_time not in seen_times) else None
        if not time_str:
            base_offset = PRIORITY_OFFSET.get(priority, i * 3)
            candidate   = anchor_dt - timedelta(minutes=base_offset + i)
            attempts = 0
            while candidate.strftime("%H:%M") in seen_times and attempts < 10:
                candidate -= timedelta(minutes=1)
                attempts  += 1
            time_str = candidate.strftime("%H:%M")

        seen_times.add(time_str)
        # evidence[]: list of 2–4 "LABEL: VALUE" strings from the data that motivated this signal.
        # Validated: must be a non-empty list of strings; truncated to 4 entries max.
        raw_evidence = s.get("evidence", [])
        if isinstance(raw_evidence, list):
            evidence = [str(e)[:80] for e in raw_evidence if isinstance(e, str) and e.strip()][:4]
        else:
            evidence = []

        signal_text = str(s.get("text", ""))[:600]  # raised from 300 — institutional multi-layer text can reach 400-500 chars
        title = str(s.get("title", ""))[:40]  # assigned early — used by sanitizer log below

        # VIX label safety net — last-resort guard in case the LLM ignores the pre-computed
        # label injected in the context. This should rarely fire now that vix_label is supplied
        # as a data fact (Mejora 3 / v7.23.32). Kept as a narrow fallback only.
        import re as _re
        _vix_m = _re.search(r'VIX\s+(?:at\s+|of\s+)?([\d.]+)', signal_text, _re.IGNORECASE)
        if _vix_m:
            try:
                _v = float(_vix_m.group(1))
                _correct_label = (
                    "low volatility" if _v < 15 else
                    "normal volatility" if _v < 20 else
                    "elevated volatility" if _v <= 28 else
                    "high volatility / fear"
                )
                _fixed = _re.sub(
                    r'(?:very\s+)?(?:low|normal|elevated|moderate|high|extreme)\s+volatility(?:\s*/\s*(?:fear|panic))?',
                    _correct_label, signal_text, count=1, flags=_re.IGNORECASE
                )
                if _fixed != signal_text:
                    print(f"  INFO VIX label safety net fired (LLM ignored pre-computed label): {signal_text[:60]}...")
                    signal_text = _fixed
            except ValueError:
                pass

        # ── Weekend verb sanitizer ─────────────────────────────────────────
        # When market_closed=True (SIGNALS_SYSTEM_CLOSED path), replace present-
        # tense live-market verbs with Friday-close equivalents. These verbs
        # imply a live market and confuse users reading the terminal after close.
        # Runs after VIX safety net so it sees the final signal_text.
        # Patterns: verb + [at|above|below|near] + number (precise, non-greedy).
        # CB POLICY LABELS (Hold/Cut/Hike) are PROTECTED — never replaced.
        if system_prompt is SIGNALS_SYSTEM_CLOSED:
            _WEEKEND_VERB_SUBS = [
                # Live-market verbs ONLY when followed by at/above/below/near + number.
                # This prevents false positives on "gains toward 155" → "closed  toward 155".
                # Requires a digit after the preposition to confirm it's a price reference.
                (r'\b(hovers?|trades?|sits?|climbs?|falls?|rises?|drops?|edges?|'
                 r'extends?|rebounds?|pressures?|pushes?|pulls?|gains?|loses?)\s+'
                 r'(above|below|at|near)\s*(\$?[\d])',
                 r'closed \2 \3'),
                # "holds at/above/below/near [number]" separately (CB label protection:
                # "BoE Hold bias" or "Hold bias" must NOT be touched — only "holds [preposition] [price]")
                (r'\bholds?\s+(above|below|at|near)\s*(\$?[\d])',
                 r'closed \1 \2'),
                # "amid rising oil prices and US-Iran tensions" → stripped.
                (r'\s*\bamid\s+[^.;]{0,80}?(?=\s*[.;]|\s+(?:WTI|USD|EUR|GBP|JPY|AUD|CAD|CHF|NZD|Key|Watch)\b|$)', r''),
                (r'  +', r' '),  # collapse double spaces left by amid strip
                # "into the Tokyo/Sydney open" → "for the Tokyo/Sydney open"
                (r'\binto\s+(the\s+(?:Tokyo|Sydney|Monday|Sunday)\s+open)\b', r'for \1'),
            ]
            for _vpat, _vrep in _WEEKEND_VERB_SUBS:
                _fixed_st = _re.sub(_vpat, _vrep, signal_text, flags=_re.IGNORECASE)
                if _fixed_st != signal_text:
                    print(f"  INFO Weekend verb sanitizer: fixed present-tense verb in '{title[:30]}...'")
                    signal_text = _fixed_st.strip()
            # Also fix evidence strings
            evidence = [
                _re.sub(
                    r'\b(hovers?|trades?|sits?|climbs?|falls?|rises?|drops?|edges?|'
                    r'extends?|rebounds?|pressures?|pushes?|pulls?|gains?|loses?)\s+'
                    r'(above|below|at|near)\s*(\$?[\d])',
                    r'closed \2 \3', ev, flags=_re.IGNORECASE
                ).strip()
                for ev in evidence
            ]

            # ── Modal uncertainty verb sanitizer ──────────────────────────────
            # Replace "may/could/might" (modal uncertainty) with decisive alternatives.
            # Does NOT touch "May" as a month name (preceded by a number or capitalized
            # in date context: "27 May", "in May", "for May meeting").
            # Pattern: word boundary "may/could/might" NOT preceded by a digit+space (date).
            import re as _re_modal
            _MODAL_SUBS = [
                # "may [adverb] [verb]" — adverb-bridged modals (e.g. "may further impact", "may also influence")
                # These slip through the plain "may [verb]" patterns because an adverb sits between modal and verb.
                # Covers: further, also, still, now, soon, yet, already, again, ultimately, additionally, potentially
                (r'(?<!\d\s)\bmay\s+(?:further|also|still|now|soon|yet|already|again|ultimately|additionally|potentially)\s+'
                 r'(compress|sustain|retest|move|extend|target|reach|test|push|pull|'
                 r'drift|slide|rally|break|hold|continue|reverse|unwind|trigger|create|set\s+up|'
                 r'offset|influence|propel|drive|pressure|support|weigh|cap|limit|add|signal|indicate|suggest|'
                 r'impact|resolve|determine|affect|shape|dominate|dictate|guide|keep|introduce|stabilize|'
                 r'prevent|accelerate|delay|reinforce|amplify|reduce|increase|narrow|widen|compress|close)\b',
                 r'is likely to \1'),
                (r'\bcould\s+(?:further|also|still|now|soon|yet|already|again|ultimately|additionally|potentially)\s+'
                 r'(compress|sustain|retest|move|extend|target|reach|test|push|pull|'
                 r'drift|slide|rally|break|hold|continue|reverse|unwind|trigger|create|set\s+up|'
                 r'offset|influence|propel|drive|pressure|support|weigh|cap|limit|add|signal|indicate|suggest|'
                 r'impact|resolve|determine|affect|shape|dominate|dictate|guide|keep|introduce|stabilize|'
                 r'prevent|accelerate|delay|reinforce|amplify|reduce|increase|narrow|widen|compress|close)\b',
                 r'is likely to \1'),
                (r'\bmight\s+(?:further|also|still|now|soon|yet|already|again|ultimately|additionally|potentially)\s+'
                 r'(compress|sustain|retest|move|extend|target|reach|test|push|pull|'
                 r'drift|slide|rally|break|hold|continue|reverse|unwind|trigger|create|set\s+up|'
                 r'offset|influence|propel|drive|pressure|support|weigh|cap|limit|add|signal|indicate|suggest|'
                 r'impact|resolve|determine|affect|shape|dominate|dictate|guide|keep|introduce|stabilize|'
                 r'prevent|accelerate|delay|reinforce|amplify|reduce|increase|narrow|widen|compress|close)\b',
                 r'is likely to \1'),
                # "may compress/sustain/retest/move/extend/offset/influence/impact/resolve..." → "is likely to ..."
                (r'(?<!\d\s)\bmay\s+(compress|sustain|retest|move|extend|target|reach|test|push|pull|'
                 r'drift|slide|rally|break|hold|continue|reverse|unwind|trigger|create|set\s+up|'
                 r'offset|influence|propel|drive|pressure|support|weigh|cap|limit|add|signal|indicate|suggest|'
                 r'impact|resolve|determine|affect|shape|dominate|dictate|guide|keep|introduce|stabilize|'
                 r'prevent|accelerate|delay|reinforce|amplify|reduce|increase|narrow|widen|compress|close)\b',
                 r'is likely to \1'),
                # "could [verb]" → "is likely to [verb]"
                (r'\bcould\s+(compress|sustain|retest|move|extend|target|reach|test|push|pull|'
                 r'drift|slide|rally|break|hold|continue|reverse|unwind|trigger|create|set\s+up|'
                 r'offset|influence|propel|drive|pressure|support|weigh|cap|limit|add|signal|indicate|suggest|'
                 r'impact|resolve|determine|affect|shape|dominate|dictate|guide|keep|introduce|stabilize|'
                 r'prevent|accelerate|delay|reinforce|amplify|reduce|increase|narrow|widen|compress|close)\b',
                 r'is likely to \1'),
                # "might [verb]" → "is likely to [verb]"
                (r'\bmight\s+(compress|sustain|retest|move|extend|target|reach|test|push|pull|'
                 r'drift|slide|rally|break|hold|continue|reverse|unwind|trigger|create|set\s+up|'
                 r'offset|influence|propel|drive|pressure|support|weigh|cap|limit|add|signal|indicate|suggest|'
                 r'impact|resolve|determine|affect|shape|dominate|dictate|guide|keep|introduce|stabilize|'
                 r'prevent|accelerate|delay|reinforce|amplify|reduce|increase|narrow|widen|compress|close)\b',
                 r'is likely to \1'),
                # "may be set to shift" → "is set to shift"
                (r'(?<!\d\s)\bmay\s+be\s+set\s+to\b', r'is set to'),
                # "may be" → "is likely to be" (generic fallback for remaining "may be")
                (r'(?<!\d\s)\bmay\s+be\b', r'is likely to be'),
            ]
            for _mpat, _mrep in _MODAL_SUBS:
                _fixed_modal = _re_modal.sub(_mpat, _mrep, signal_text, flags=_re_modal.IGNORECASE)
                if _fixed_modal != signal_text:
                    print(f"  INFO Modal sanitizer: replaced uncertainty verb in '{title[:30]}...'")
                    signal_text = _fixed_modal.strip()

            # ── Subject-verb agreement (post-modal) ───────────────────────────
            # The modal sanitizer converts "tensions may weigh" → "tensions is likely to weigh".
            # Fix: plural FX nouns followed (within the same clause) by "is likely to" → "are likely to".
            import re as _re_sva
            _SVA_PLURAL = (
                r'(?:tensions|forces|pressures|flows|movements|moves|developments|'
                r'risks|concerns|factors|headwinds|tailwinds|conditions|dynamics)'
            )
            signal_text = _re_sva.sub(
                rf'\b({_SVA_PLURAL})\b([^.;]{{0,60}}?)\bis\s+likely\s+to\b',
                r'\1\2are likely to',
                signal_text, flags=_re_sva.IGNORECASE
            )

            # ── Stale past meeting date guard ─────────────────────────────────
            # Groq uses meeting dates from the data block but sometimes picks up a
            # past date (e.g. "next BoJ meeting on 28 Apr" when today is May 2).
            # Replace "[CB] meeting on [past date]" with "[CB] meeting" (no date).
            import re as _re_spd
            from datetime import datetime as _spd_dt, timezone as _spd_tz, date as _spd_date
            _spd_today = _spd_dt.now(_spd_tz.utc).date()
            _SPD_MONTHS = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                           'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
            _SPD_CBS    = r'(?:BoJ|BoE|RBA|RBNZ|SNB|ECB|Fed|FOMC|BoC)'
            _SPD_MON    = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
            _spd_pat    = _re_spd.compile(
                rf'\b(?:next\s+)?({_SPD_CBS})?\s*meeting(?:\s+is)?\s+(?:on\s+)?(?:(\d{{1,2}})\s+({_SPD_MON})|({_SPD_MON})\s+(\d{{1,2}}))\b',
                _re_spd.IGNORECASE
            )
            def _spd_replace(m):
                # group(1) = optional CB name (may be None when "next meeting on" has no CB prefix)
                cb = m.group(1) or "CB"
                if m.group(2) and m.group(3):
                    day_s, mon_s = m.group(2), m.group(3)
                elif m.group(4) and m.group(5):
                    mon_s, day_s = m.group(4), m.group(5)
                else:
                    return m.group(0)
                try:
                    mn = _SPD_MONTHS.get(mon_s.lower()[:3])
                    if mn is None:
                        return m.group(0)
                    d = _spd_date(_spd_today.year, mn, int(day_s))
                    if d < _spd_today:
                        print(f"  INFO Stale-date guard: removed past date '{d}' from '{title[:30]}...'")
                        return f"next {cb} meeting"
                except Exception:
                    pass
                return m.group(0)
            signal_text = _spd_pat.sub(_spd_replace, signal_text)


            # Deterministic Python guardrails for rules removed from the system prompt.
            # By moving mechanical transformations here we shorten the system prompt,
            # reducing attention diffusion and improving LLM compliance on the rules
            # that DO require judgment (directional coherence, COT direction logic).
            import re as _re_adv
            _ADV_SUBS = [
                # ── Direction artefact ──────────────────────────────────────────
                (r'\b(downside|upside)\s+closed\b', r'\1'),
                # ── CB "closed at [rate]%" artefact (closed-market mode) ─────────
                # Groq interprets 'market_closed=True' and writes "BoE closed at 3.75%"
                # The CB rate did not "close" — it simply IS that rate. Strip "closed".
                (r'\b(BoE|BoC|BoJ|RBA|RBNZ|SNB|ECB|Fed|FOMC)\s+closed\s+at\b', r'\1 at'),
                # ── CB terminology: Bloomberg labels only ───────────────────────
                (r'\b(BoE|BoC|BoJ|RBA|RBNZ|SNB|ECB|Fed|FOMC)\s+closed\s+bias\b', r'\1 Hold bias'),
                (r'\b(BoE|BoC|BoJ|RBA|RBNZ|SNB|ECB|Fed|FOMC)\s+neutral\s+bias\b', r'\1 Hold bias'),
                (r'\b(BoE|BoC|BoJ|RBA|RBNZ|SNB|ECB|Fed|FOMC)\s+on\s+hold\s+bias\b', r'\1 Hold bias'),
                # ── Pair convention: Bloomberg standard ─────────────────────────
                (r'\bJPY/USD\b', r'USD/JPY'),
                (r'\bCHF/USD\b', r'USD/CHF'),
                (r'\bCAD/USD\b', r'USD/CAD'),
                (r'\bJPY/EUR\b', r'EUR/JPY'),
                (r'\bJPY/GBP\b', r'GBP/JPY'),
                (r'\bJPY/AUD\b', r'AUD/JPY'),
                # ── Investment-advice phrases ───────────────────────────────────
                (r'\bfor\s+this\s+trade\b', r'as the key catalyst'),
                (r'\bfor\s+the\s+trade\b',  r'as the key catalyst'),
                (r'\bgo\s+long\b',          r'expect upside'),
                (r'\bgo\s+short\b',         r'expect downside'),
                (r'\bbuy\s+the\s+dip\b',    r'upside risk if support holds'),
                (r'\bsell\s+the\s+rally\b', r'downside risk at resistance'),
                (r'\bentry\s+point\b',      r'catalyst level'),
                (r'\btake\s+profit\b',      r'target level'),
                (r'\bstop\s+loss\b',        r'invalidation level'),
                (r'\breduce\s+exposure\b',  r'downside risk rises'),
                (r'\badd\s+risk\b',         r'upside risk rises'),
                (r'\bcut\s+positions\b',    r'unwind risk rises'),
                # ── Vague editorial / filler ────────────────────────────────────
                (r'\bwill\s+be\s+crucial\s+for\b', r'determines'),
                (r'\bwill\s+be\s+crucial\b',        r'determines direction'),
                (r'\bworth\s+watching\s+closely\b', r'the key setup'),
                (r'\bworth\s+watching\b',            r'notable'),
                (r'\bsetup\s+is\s+compelling\b',     r'setup is asymmetric'),
                (r'\bincreasingly\s+interesting\b',  r'notable'),
                # "especially considering" → "given" (brevity, institutional tone)
                (r'\bespecially\s+considering\b',    r'given'),
                # "considering the [CB]" → strip (adds nothing)
                (r',?\s*considering\s+the\s+\w+(?:\'s)?\s+\w+\s+stance\b', r''),
                # "creating an asymmetric [risk/setup]" → "creating [risk/setup]"
                (r'\bcreating\s+an\s+asymmetric\s+(squeeze\s+risk|unwind\s+risk|setup|risk)\b', r'creating \1'),
                # "asymmetric risk" (bare) → "unwind risk"
                (r'\ban\s+asymmetric\s+risk\b', r'unwind risk'),
                # "is likely to be the catalyst for this unwind" → "is the catalyst"
                (r'\bis\s+likely\s+to\s+be\s+the\s+catalyst\s+for\s+(?:this\s+)?(\w+)\b', r'is the catalyst for \1'),
                # "is likely to impact [the pair's/direction/outcome]" → "determines direction"
                # The modal guard converts "may impact" → "is likely to impact", but it's still vague
                (r'\bis\s+likely\s+to\s+impact\s+(?:the\s+)?(?:pair\'s\s+)?(?:direction|outcome|trend|bias)\b', r'determines direction'),
                (r'\bis\s+likely\s+to\s+impact\s+(?:the\s+)?(?:\w+\/\w+\s+)?(?:direction|outcome|trend|bias)\b', r'determines direction'),
                # "will be a key catalyst" / "will be the key catalyst" → "is the catalyst"
                (r'\bwill\s+be\s+(?:a|the)\s+key\s+catalyst\b', r'is the catalyst'),
                # "potential [unwind/squeeze/reversal/shift/catalyst/divergence/decrease/impact/increase]" → strip "potential"
                (r'\bpotential\s+(unwind|squeeze|reversal|rally|decline|move|shift|catalyst|divergence|upside|downside|carry\s+extension|decrease|increase|impact|drop|rise|compression|expansion|correction)\b', r'\1'),
                # "pointing to a [potential] X" → strip entire editorial wrapper
                (r',?\s*pointing\s+to\s+a\s+(?:potential\s+)?\w+(?:\s+\w+){0,3}', r''),
                # "important for resolving" → "determines" (shorter, more direct)
                (r'\bwill\s+be\s+important\s+for\s+resolving\b', r'determines'),
                # "The mechanism driving the move is [the/a] ..." → strip and keep the content
                (r'\bThe\s+mechanism\s+driving\s+the\s+move\s+is\s+(?:the\s+|a\s+)?', r''),
                # "[X] drives the move" / "[X] drives the pair" — generic filler tail
                # Replace with empty: "The BoC's 2.25% rate drives the move." → strip
                (r'\s+drives\s+the\s+(?:move|pair)\b\.?', r'.'),
                # "affecting [the X's value]" / "affecting the [CCY]'s value" → strip (verbose tail)
                (r',?\s*affecting\s+(?:the\s+)?(?:\w+\'s\s+)?value\.?\s*$', r''),
                # "affecting [CCY/pair] direction" → strip if verbose
                (r',?\s*affecting\s+(?:the\s+)?\w+(?:\/\w+)?\s+(?:pair\'s\s+)?(?:direction|movement|value)\b', r''),
                # ── First person ────────────────────────────────────────────────
                (r'\bwe\s+see\b',        r'data shows'),
                (r'\bwe\s+view\b',       r'the read is'),
                (r'\bour\s+view\s+is\b', r'the bias is'),
                (r'\bour\s+view\b',      r'the read'),
                # ── Duplicate adjacent phrases ───────────────────────────────────
                (r'\b(as the key catalyst)\s+\1\b', r'as the key catalyst'),
                (r'\b(the key catalyst)\s+\1\b',    r'the key catalyst'),
                (r'\b(as a catalyst)\s+\1\b',       r'as a catalyst'),
                # Groq sometimes writes "is a key catalyst as the key catalyst" when
                # ADV_SUBS partially replaced "will be a key catalyst" leaving both halves
                (r'\bis\s+(?:a|the)\s+key\s+catalyst\s+as\s+the\s+key\s+catalyst\b', r'is the catalyst'),
                # Generic: any "[X] catalyst [X] catalyst" dedup
                (r'\b(is the catalyst)\s+as\s+the\s+key\s+catalyst\b', r'is the catalyst'),
                # ── Collapse double spaces ──────────────────────────────────────
                (r'  +', r' '),
            ]
            for _apat, _arep in _ADV_SUBS:
                _fixed_adv = _re_adv.sub(_apat, _arep, signal_text, flags=_re_adv.IGNORECASE)
                if _fixed_adv != signal_text:
                    print(f"  INFO Advisory/style stripper: fixed phrase in '{title[:30]}...'")
                    signal_text = _fixed_adv.strip()
            # Also fix title (pair convention + CB terminology apply to titles too)
            for _apat, _arep in _ADV_SUBS:
                title = _re_adv.sub(_apat, _arep, title, flags=_re_adv.IGNORECASE)

        # ── CB rate stale-training guard ───────────────────────────────────────
        # Detects when the model used a rate from training memory instead of the
        # data block. Root cause: the model "remembers" a historically prominent
        # rate (e.g. BoE 4.50% from 2025) and ignores the current data block value.
        # This guard loads the actual CB rates from rates/*.json and replaces any
        # stale rate found in the signal text with the correct current value.
        # Only fires on mismatches — does not modify correct signals.
        import re as _re_cbr
        _CB_RATE_FILES = {
            "Fed":  "USD", "ECB":  "EUR", "BoE":  "GBP", "BoJ":  "JPY",
            "RBA":  "AUD", "BoC":  "CAD", "SNB":  "CHF", "RBNZ": "NZD",
        }
        for _cb_label, _ccy_code in _CB_RATE_FILES.items():
            # Find any rate written next to this CB label in signal text
            _rate_pattern = _re_cbr.compile(
                rf'\b{_re_cbr.escape(_cb_label)}\s+([\d.]+)%', _re_cbr.IGNORECASE
            )
            for _m in _rate_pattern.finditer(signal_text):
                _written_rate = float(_m.group(1))
                # Load actual current rate from rates/*.json
                try:
                    _rate_file = load_json(SITE_DIR / "rates" / f"{_ccy_code}.json")
                    _obs = (_rate_file or {}).get("observations", [])
                    _actual_rate = float(_obs[0]["value"]) if _obs else None
                except Exception:
                    _actual_rate = None
                if _actual_rate is not None and abs(_written_rate - _actual_rate) >= 0.10:
                    # Significant mismatch — replace with correct rate
                    _old_str = f"{_cb_label} {_written_rate:.2f}%".rstrip("0").rstrip(".")
                    _new_str = f"{_cb_label} {_actual_rate:.2f}%".rstrip("0").rstrip(".")
                    # Also fix the bp spread if present (e.g. "75bp" → recomputed)
                    # Find the counterpart rate to recompute the spread
                    _COUNTERPART = {
                        "BoE": ("Fed", "USD"), "ECB": ("Fed", "USD"),
                        "RBA": ("Fed", "USD"), "BoC": ("Fed", "USD"),
                        "RBNZ": ("Fed", "USD"), "BoJ": ("Fed", "USD"),
                        "SNB": ("Fed", "USD"), "Fed": ("ECB", "EUR"),
                    }
                    _cp_label, _cp_ccy = _COUNTERPART.get(_cb_label, ("", ""))
                    _cp_rate = None
                    if _cp_label:
                        try:
                            _cp_file = load_json(SITE_DIR / "rates" / f"{_cp_ccy}.json")
                            _cp_obs = (_cp_file or {}).get("observations", [])
                            _cp_rate = float(_cp_obs[0]["value"]) if _cp_obs else None
                        except Exception:
                            pass
                    _fixed_text = signal_text.replace(
                        f"{_cb_label} {_written_rate:.2f}%", f"{_cb_label} {_actual_rate:.2f}%"
                    )
                    # Also handle single-decimal format e.g. "BoE 4.5%" and integer "BoE 4%"
                    _rate_variants = [
                        str(_written_rate),           # "4.5"
                        f"{_written_rate:.2f}",       # "4.50"
                        f"{_written_rate:.1f}",       # "4.5"
                        str(int(_written_rate)) if _written_rate == int(_written_rate) else None,  # "4"
                    ]
                    for _rv in _rate_variants:
                        if _rv is None:
                            continue
                        _fixed_text = _re_cbr.sub(
                            rf'\b{_re_cbr.escape(_cb_label)}\s+{_re_cbr.escape(_rv)}%',
                            f"{_cb_label} {_actual_rate:.2f}%",
                            _fixed_text
                        )
                    # Recompute bp spread if counterpart rate is available
                    if _cp_rate is not None:
                        _old_bp = round(abs(_written_rate - _cp_rate) * 100)
                        _new_bp = round(abs(_actual_rate - _cp_rate) * 100)
                        if _old_bp != _new_bp:
                            _fixed_text = _re_cbr.sub(
                                rf'\b{_old_bp}bp\b', f"{_new_bp}bp", _fixed_text
                            )
                    if _fixed_text != signal_text:
                        print(f"  WARNING CB rate stale-training guard: {_cb_label} {_written_rate}% → {_actual_rate:.2f}% in '{title[:30]}'")
                        signal_text = _fixed_text
                    break  # one fix per CB label per signal

        # ── Hike/cut probability stale-training guard ──────────────────────────
        # If the signal mentions a hike or cut probability (e.g. "hike 100% priced"),
        # verify it against meetings.json hikeProb/cutProb. Correct if discrepancy ≥ 15pp.
        # IMPORTANT: only match patterns where the CB name appears explicitly near the
        # percentage — avoids false matches on unrelated numbers in the same signal.
        import re as _re2
        _meetings = load_json(SITE_DIR / "meetings-data" / "meetings.json") or {}
        _mtg_data = _meetings.get("meetings", {})
        # Map of CB label patterns → currency key in meetings.json
        _CB_CCY_MAP = {"RBNZ": "NZD", "RBA": "AUD", "BoC": "CAD",
                       "BoE": "GBP", "ECB": "EUR", "BoJ": "JPY",
                       "Fed": "USD", "FOMC": "USD", "SNB": "CHF"}
        for _cb, _ccy in _CB_CCY_MAP.items():
            _ccy_data = _mtg_data.get(_ccy, {})
            _actual_hike_p = _ccy_data.get("hikeProb")
            _actual_cut_p  = _ccy_data.get("cutProb")
            if _actual_hike_p is None: continue
            # Only match patterns that include the CB name explicitly — prevents
            # cross-CB contamination where a generic "40% priced" in an AUD signal
            # gets rewritten for BoC/BoJ/Fed because they iterate over the same text.
            _prob_re = _re2.compile(
                rf"(?:{_re2.escape(_cb)}\s+(?:hike|cut)\s+(\d{{1,3}})%"      # "RBA hike 40%"
                rf"|{_re2.escape(_cb)}\s+.*?(\d{{1,3}})%\s+(?:priced|probability)"  # "RBA bias (40% priced)"
                rf"|(\d{{1,3}})%\s+(?:priced|probability)(?:\s+for)?\s+(?:a\s+)?{_re2.escape(_cb)})",  # "40% priced for RBA"
                _re2.IGNORECASE | _re2.DOTALL
            )
            for _pm in _prob_re.finditer(signal_text):
                _written_p = int(next(g for g in _pm.groups() if g is not None))
                if abs(_written_p - _actual_hike_p) >= 15:
                    _fixed = signal_text.replace(f"{_written_p}%", f"{_actual_hike_p}%", 1)
                    if _fixed != signal_text:
                        print(f"  WARNING hike-prob guard: {_cb} {_written_p}% → {_actual_hike_p}% in '{title[:35]}'")
                        signal_text = _fixed
                break  # one correction per CB per signal

        # ── Evidence chip coherence guard ─────────────────────────────────────
        # When the hike-bias fix corrected the signal text to say "hold bias (0% priced)"
        # but Groq's original evidence chip still says "RBA hike bias: 40% priced",
        # the chip and text contradict each other. This pass detects the mismatch and
        # corrects the chip label to match the text's bias assertion.
        # Pattern: text says "hold bias" for a CB, but chip says "[CB] hike bias: X%"
        import re as _re_ev_coh
        _CB_EV_LABELS = {
            "RBA": "AUD", "RBNZ": "NZD", "BoC": "CAD", "BoE": "GBP",
            "ECB": "EUR", "BoJ": "JPY", "Fed": "USD", "FOMC": "USD", "SNB": "CHF",
        }
        for _ec_cb, _ec_ccy in _CB_EV_LABELS.items():
            # Detect "hold bias" in text for this CB
            _hold_in_text = bool(_re_ev_coh.search(
                rf'\b{_re_ev_coh.escape(_ec_cb)}\b.{{0,80}}?hold\s+bias',
                signal_text, _re_ev_coh.IGNORECASE
            ))
            if not _hold_in_text:
                # Also check "(0% priced for a hike)" pattern without explicit CB name in vicinity
                _hold_in_text = bool(_re_ev_coh.search(r'\(0%\s+priced\s+for\s+a\s+hike\)', signal_text, _re_ev_coh.IGNORECASE))
            if _hold_in_text:
                # Correct any evidence chip that says "[CB] hike bias: X%" for this CB
                for _ec_i, _ec_chip in enumerate(evidence):
                    _ec_m = _re_ev_coh.match(
                        rf'^{_re_ev_coh.escape(_ec_cb)}\s+hike\s+bias[:\s]+([\d.]+)%\s*(?:priced)?$',
                        _ec_chip.strip(), _re_ev_coh.IGNORECASE
                    )
                    if _ec_m:
                        evidence[_ec_i] = f"{_ec_cb} hold bias: 0% hike priced"
                        print(f"  INFO Evidence-chip coherence guard: corrected '{_ec_chip}' → '{evidence[_ec_i]}' in '{title[:30]}'")


        # ── OIS probability injection guard ───────────────────────────────────
        # When a signal references "next [CB] meeting" without a market-implied
        # probability, append the OIS probability from meetings.json.
        # Bloomberg standard: every CB meeting reference includes the market-priced
        # probability (e.g. "next BoE meeting — mkt 88% hold").
        # This guard fires only when: (a) a CB meeting mention exists in the signal,
        # (b) the signal lacks any "% priced", "% hold", "% hike", "% cut" nearby.
        import re as _re_ois_inj
        _OIS_INJ_MAP = {
            "BoE":  "GBP", "ECB":  "EUR", "BoJ":  "JPY", "RBA":  "AUD",
            "RBNZ": "NZD", "BoC":  "CAD", "SNB":  "CHF", "Fed":  "USD",
            "FOMC": "USD",
        }
        _ois_mtg_data = (load_json(SITE_DIR / "meetings-data" / "meetings.json") or {}).get("meetings", {})
        for _oi_cb, _oi_ccy in _OIS_INJ_MAP.items():
            # Only act if this CB meeting appears in the signal text
            if not _re_ois_inj.search(rf'\bnext\s+{_re_ois_inj.escape(_oi_cb)}\s+meeting\b', signal_text, _re_ois_inj.IGNORECASE):
                continue
            # Skip if OIS probability already present near the meeting mention
            _nearby_prob = _re_ois_inj.search(
                rf'\bnext\s+{_re_ois_inj.escape(_oi_cb)}\s+meeting\b.{{0,120}}?(\d{{1,3}})%\s+(?:priced|hold|hike|cut|probability)',
                signal_text, _re_ois_inj.IGNORECASE | _re_ois_inj.DOTALL
            )
            if _nearby_prob:
                continue
            # Build the OIS probability string from meetings.json
            _oi_mtg = _ois_mtg_data.get(_oi_ccy, {})
            _oi_bias_raw = _oi_mtg.get("bias", "hold")
            _oi_hike_p   = _oi_mtg.get("hikeProb") or 0
            _oi_cut_p    = _oi_mtg.get("cutProb")  or 0
            # Apply coherence: bias=hike + hikeProb=0 → hold
            if _oi_bias_raw == "hike" and _oi_hike_p == 0:
                _oi_bias_raw = "hold"
            if _oi_bias_raw == "cut" and _oi_cut_p == 0:
                _oi_bias_raw = "hold"
            if _oi_bias_raw == "hike" and _oi_hike_p > 0:
                _oi_prob_str = f"{int(_oi_hike_p)}% hike priced"
            elif _oi_bias_raw == "cut" and _oi_cut_p > 0:
                _oi_prob_str = f"{int(_oi_cut_p)}% cut priced"
            else:
                _hold_p = 100 - max(_oi_hike_p, _oi_cut_p)
                _oi_prob_str = f"{int(_hold_p)}% hold priced"
            # Inject after "next [CB] meeting" in the signal text
            _old_ref = _re_ois_inj.search(
                rf'\bnext\s+{_re_ois_inj.escape(_oi_cb)}\s+meeting\b', signal_text, _re_ois_inj.IGNORECASE
            )
            if _old_ref:
                _inj_pos = _old_ref.end()
                signal_text = signal_text[:_inj_pos] + f" (mkt: {_oi_prob_str})" + signal_text[_inj_pos:]
                print(f"  INFO OIS-probability guard: injected '{_oi_prob_str}' for {_oi_cb} in '{title[:30]}...'")


        # ── Squeeze direction coherence guard ─────────────────────────────────
        # "crowded [CCY] short sets up squeeze risk" is directionally ambiguous unless
        # the implication is stated explicitly.
        # Short-squeeze = shorts cover = the shorted CCY RISES.
        # For USD/JPY: crowded JPY short → short-squeeze → JPY rises → USD/JPY FALLS.
        # Bloomberg always disambiguates: "short-squeeze risk = USD/JPY downside risk."
        # Guard: when a JPY signal contains "crowded short" + "squeeze risk" without an
        # explicit direction note, replace "sets up squeeze risk" with a directional form.
        import re as _re_sq
        _is_jpy_signal = "JPY" in title.upper()
        if _is_jpy_signal:
            _has_crowded_short  = bool(_re_sq.search(r'\bcrowded\s+(?:\w+\s+)?short\b', signal_text, _re_sq.IGNORECASE))
            _has_squeeze        = bool(_re_sq.search(r'\bsqueeze\s+risk\b', signal_text, _re_sq.IGNORECASE))
            _has_direction_note = bool(_re_sq.search(
                r'\b(?:downside\s+risk|USD/JPY\s+(?:downside|lower|falls?)|pair\s+(?:falls?|retreats?)|'
                r'unwind\s+(?:risk|targets?)|short.squeeze\s+(?:risk|targets?)|squeeze\s+targets?)\b',
                signal_text, _re_sq.IGNORECASE
            ))
            if _has_crowded_short and _has_squeeze and not _has_direction_note:
                _fixed_sq = _re_sq.sub(
                    r'\bsets\s+up\s+squeeze\s+risk\b',
                    'sets up short-squeeze risk (USD/JPY downside if shorts cover)',
                    signal_text, flags=_re_sq.IGNORECASE
                )
                if _fixed_sq != signal_text:
                    signal_text = _fixed_sq
                    print(f"  INFO Squeeze-direction guard: added direction note to '{title[:35]}...'")

        # the pair/text so the signal is not silently dropped or displayed blank.
        if not title.strip():
            # Try to extract a pair from the text (e.g. "EUR/USD at 1.17")
            _pair_match = re.search(
                r'\b([A-Z]{3}/[A-Z]{3})\b', signal_text
            )
            if _pair_match:
                title = f"{_pair_match.group(1)} — Market Update"
            else:
                title = "FX Market Update"
            print(f"  WARNING Empty title guard: derived title '{title}' from signal text")

        # ── Append processed signal to result ──────────────────────────────────
        result.append({
            **s,
            "time":     time_str,
            "text":     signal_text,
            "title":    title[:120],
            "evidence": evidence,
            "priority": priority,
        })

    # ── Token budget summary ──────────────────────────────────────────────────
    print(f"  Signals token budget: {_signals_tokens_used:,} used / {_SIGNALS_TOKEN_BUDGET:,} limit "
          f"({'OK' if _signals_tokens_used <= _SIGNALS_TOKEN_BUDGET else 'EXCEEDED'})")

    return result

# ── Change detection ─────────────────────────────────────────────────────────
# Compares current market data against the snapshot saved in the previous run.
# Only analytically significant moves (above materiality thresholds) trigger
# a Groq call. This avoids regenerating identical content 12× per day when
# prices, yields, and headlines are unchanged between runs.
#
# Snapshot file: ai-analysis/context_snapshot.json
# Written at the end of every successful Groq run.
# Deleted on force-regenerate (workflow_dispatch or --force flag).

# Materiality thresholds — minimum move in each field to justify regeneration.
# Aligned with the NARRATIVE_SYSTEM materiality rules (DXY ≥0.15%, etc.).
_MATERIALITY = {
    "eurusd": 0.0020, "gbpusd": 0.0020, "usdjpy": 0.20,
    "audusd": 0.0020, "usdchf": 0.0020, "usdcad": 0.0020, "nzdusd": 0.0020,
    "vix": 0.50, "spx": 20.0, "gold": 15.0, "wti": 0.50,
    "dxy": 0.15, "move": 3.0, "us2y": 0.03, "us10y": 0.03,
}

def build_snapshot() -> dict:
    """Extract analytically significant market values for change detection."""
    snap = {}
    intraday = load_json(SITE_DIR / "intraday-data" / "quotes.json")
    q = (intraday or {}).get("quotes", {})

    FX_KEYS    = ["eurusd","gbpusd","usdjpy","audusd","usdchf","usdcad","nzdusd"]
    CROSS_KEYS = ["vix","spx","gold","wti","dxy","move","us2y","us10y"]

    snap["fx"] = {}
    for k in FX_KEYS:
        d = q.get(k)
        if d and d.get("close") is not None:
            snap["fx"][k] = float(d["close"])

    snap["cross"] = {}
    for k in CROSS_KEYS:
        d = q.get(k)
        if d and d.get("close") is not None:
            snap["cross"][k] = float(d["close"])

    # COT — changes weekly; always include weekEnding so a COT update forces regen
    snap["cot"] = {}
    for ccy in CURRENCIES:
        d = load_json(SITE_DIR / "cot-data" / f"{ccy}.json")
        if d:
            snap["cot"][ccy] = {
                "net": d.get("netPosition"),
                "we":  d.get("weekEnding") or d.get("reportDate"),
            }

    # News headlines — title+time; any new headline forces regeneration
    news_raw = load_json(SITE_DIR / "news-data" / "news.json", default={})
    articles = news_raw.get("articles", news_raw.get("items", [])) \
               if isinstance(news_raw, dict) else news_raw
    snap["headlines"] = [
        {"t": a.get("title", ""), "time": a.get("time", "")}
        for a in articles if a.get("impact") == "high"
    ][:8]

    return snap


def market_data_changed(prev: dict, curr: dict) -> tuple[bool, str]:
    """Compare two snapshots. Returns (changed: bool, reason: str).

    Any field that exceeds its materiality threshold triggers regeneration.
    Headlines are compared by set equality — any new or removed headline is
    treated as a change regardless of magnitude.
    Always returns True (with reason) when no previous snapshot exists.
    """
    # FX pairs
    prev_fx, curr_fx = prev.get("fx", {}), curr.get("fx", {})
    for k, threshold in _MATERIALITY.items():
        if k not in curr_fx and k not in prev.get("cross", {}):
            continue
        prev_val = prev_fx.get(k) or prev.get("cross", {}).get(k)
        curr_val = curr_fx.get(k) or curr.get("cross", {}).get(k)
        if prev_val is None or curr_val is None:
            continue
        delta = abs(curr_val - prev_val)
        if delta >= threshold:
            return True, f"{k} moved {curr_val - prev_val:+.4f} (threshold ±{threshold})"

    # COT — weekEnding change means new CFTC report published
    prev_cot, curr_cot = prev.get("cot", {}), curr.get("cot", {})
    for ccy in CURRENCIES:
        pc, cc = prev_cot.get(ccy, {}), curr_cot.get(ccy, {})
        if pc.get("we") != cc.get("we"):
            return True, f"COT weekEnding changed for {ccy}: {pc.get('we')} → {cc.get('we')}"

    # Headlines — set comparison on (title, time) tuples
    prev_hl = {(h["t"], h["time"]) for h in prev.get("headlines", [])}
    curr_hl = {(h["t"], h["time"]) for h in curr.get("headlines", [])}
    if prev_hl != curr_hl:
        new_hl = curr_hl - prev_hl
        sample = next(iter(new_hl), ("", ""))
        return True, f"Headlines changed — {len(new_hl)} new: '{sample[0][:60]}'"

    return False, "no material change detected"


def load_snapshot(out_dir: "Path") -> dict:
    return load_json(out_dir / "context_snapshot.json") or {}


def save_snapshot(out_dir: "Path", snap: dict) -> None:
    # FIX-30: Preserve cross_pct and closed_symbols written by fetch_intraday_quotes.py.
    # build_snapshot() creates a minimal change-detection dict (fx/cross/cot/headlines) —
    # it intentionally doesn't include cross_pct or closed_symbols. Without this guard,
    # every narrative run overwrites context_snapshot.json and silently discards those two
    # fields, causing the narrative prompt to see no pct data and generate "+0.00%" strings.
    snap_path = out_dir / "context_snapshot.json"
    if snap_path.exists():
        try:
            with open(snap_path) as _f:
                _existing = json.load(_f)
            for _preserve_key in ("cross_pct", "closed_symbols"):
                if _preserve_key not in snap and _preserve_key in _existing:
                    snap[_preserve_key] = _existing[_preserve_key]
        except Exception:
            pass  # If we can't read the existing file, just write the new snap as-is
    snap["saved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with open(snap_path, "w") as f:
            json.dump(snap, f, indent=2)
    except Exception as e:
        print(f"  WARNING Could not save snapshot: {e}")


# ── RSS / JSON Feed generator ─────────────────────────────────────────────────
# Called at the end of main() after narrative + signals are written.
# Maintains feed.xml (RSS 2.0) and feed.json (JSON Feed 1.1) in the site root.
# Each entry is one narrative run that produced new content. Max 20 items kept.

def _rfc822(dt: "datetime") -> str:
    """Format a datetime as RFC-822 for RSS <pubDate>."""
    return dt.strftime("%a, %d %b %Y %H:%M:%S +0000")


def _iso8601_to_dt(s: str) -> "datetime":
    """Parse an ISO-8601 UTC string to datetime; returns epoch on failure."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _regime_label(regime: str) -> str:
    labels = {
        "RISK-ON":  "[RISK-ON]",
        "RISK-OFF": "[RISK-OFF]",
        "CAUTION":  "[CAUTION]",
        "MIXED":    "[MIXED]",
        "NEUTRAL":  "[NEUTRAL]",
    }
    return labels.get((regime or "").upper(), f"[{regime}]" if regime else "[—]")


def write_feeds(site_dir: "Path", narrative: dict, signals: list) -> None:
    """
    Build/update feed.xml (RSS 2.0) and feed.json (JSON Feed 1.1) from the
    latest narrative + signals. Prepends a new item only when the narrative
    text has actually changed from the most recent feed entry.
    """
    import xml.etree.ElementTree as ET

    SITE_URL   = "https://globalinvesting.github.io"
    FEED_URL   = f"{SITE_URL}/feed.xml"
    JFEED_URL  = f"{SITE_URL}/feed.json"
    MAX_ITEMS  = 20
    FEED_TITLE = "Global Investing FX Terminal — AI Market Narrative"
    FEED_DESC  = "AI-generated FX market narrative and signals, updated at each major session transition."

    narr_text    = (narrative or {}).get("narrative", "").strip()
    regime       = (narrative or {}).get("regime", "")
    generated_at = (narrative or {}).get("generated_at",
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    if not narr_text:
        print("  ⏭  write_feeds: empty narrative — skipping feed update.")
        return

    pub_dt   = _iso8601_to_dt(generated_at)
    pub_rss  = _rfc822(pub_dt)
    pub_iso  = generated_at if generated_at.endswith("Z") else pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    snippet    = narr_text if len(narr_text) <= 80 else narr_text[:77] + "…"
    item_title = f"{_regime_label(regime)} {snippet}"

    top_sigs = [s for s in (signals or []) if s.get("priority") in ("critical", "warning")][:3]
    body_parts = [narr_text]
    if top_sigs:
        body_parts.append("\n\nKey signals:")
        for s in top_sigs:
            body_parts.append(f"\n• {s.get('title','')}: {s.get('text','')[:140]}")
    item_body = "".join(body_parts)

    item_url  = SITE_URL + "/"
    item_guid = f"{SITE_URL}/narrative/{pub_dt.strftime('%Y%m%dT%H%M%S')}"

    # ── RSS 2.0 ──────────────────────────────────────────────────────────────
    rss_file = site_dir / "feed.xml"
    existing_items: list = []

    if rss_file.exists():
        try:
            tree = ET.parse(rss_file)
            root = tree.getroot()
            channel = root.find("channel")
            if channel is not None:
                for item_el in channel.findall("item"):
                    existing_items.append({
                        "title":       item_el.findtext("title") or "",
                        "description": item_el.findtext("description") or "",
                        "link":        item_el.findtext("link") or item_url,
                        "guid":        item_el.findtext("guid") or "",
                        "pubDate":     item_el.findtext("pubDate") or "",
                    })
        except Exception as exc:
            print(f"  WARN Could not parse existing feed.xml: {exc} — rebuilding.")
            existing_items = []

    last_desc = existing_items[0].get("description", "") if existing_items else ""
    if last_desc[:200].strip() == item_body[:200].strip():
        print("  ⏭  write_feeds: narrative unchanged — feed not updated.")
        return

    new_item = {
        "title":       item_title,
        "description": item_body,
        "link":        item_url,
        "guid":        item_guid,
        "pubDate":     pub_rss,
    }
    all_items = ([new_item] + existing_items)[:MAX_ITEMS]

    rss_root   = ET.Element("rss", version="2.0")
    rss_root.set("xmlns:atom", "http://www.w3.org/2005/Atom")
    channel_el = ET.SubElement(rss_root, "channel")

    def _sub(parent, tag, text):
        el = ET.SubElement(parent, tag)
        el.text = text
        return el

    _sub(channel_el, "title",         FEED_TITLE)
    _sub(channel_el, "link",          SITE_URL)
    _sub(channel_el, "description",   FEED_DESC)
    _sub(channel_el, "language",      "en")
    _sub(channel_el, "lastBuildDate", pub_rss)
    _sub(channel_el, "ttl",           "120")
    atom_link = ET.SubElement(channel_el, "atom:link")
    atom_link.set("href", FEED_URL)
    atom_link.set("rel",  "self")
    atom_link.set("type", "application/rss+xml")

    for it in all_items:
        item_el = ET.SubElement(channel_el, "item")
        _sub(item_el, "title",       it["title"])
        _sub(item_el, "description", it["description"])
        _sub(item_el, "link",        it["link"])
        _sub(item_el, "pubDate",     it["pubDate"])
        guid_el = ET.SubElement(item_el, "guid")
        guid_el.text = it["guid"]
        guid_el.set("isPermaLink", "false")

    xml_bytes = ET.tostring(rss_root, encoding="unicode", xml_declaration=False)
    with open(rss_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(xml_bytes)
    print(f"  ✅ feed.xml updated — {len(all_items)} item(s)")

    # ── JSON Feed 1.1 ────────────────────────────────────────────────────────
    jfeed_file = site_dir / "feed.json"
    existing_jitems: list = []

    if jfeed_file.exists():
        try:
            existing_feed = json.loads(jfeed_file.read_text(encoding="utf-8"))
            existing_jitems = existing_feed.get("items", [])
        except Exception as exc:
            print(f"  WARN Could not parse existing feed.json: {exc} — rebuilding.")
            existing_jitems = []

    new_jitem = {
        "id":             item_guid,
        "url":            item_url,
        "title":          item_title,
        "content_text":   item_body,
        "date_published": pub_iso,
        "tags":           [regime] if regime else [],
    }
    all_jitems = ([new_jitem] + [i for i in existing_jitems if i.get("id") != item_guid])[:MAX_ITEMS]

    jfeed = {
        "version":       "https://jsonfeed.org/version/1.1",
        "title":         FEED_TITLE,
        "home_page_url": SITE_URL,
        "feed_url":      JFEED_URL,
        "description":   FEED_DESC,
        "language":      "en",
        "items":         all_jitems,
    }
    with open(jfeed_file, "w", encoding="utf-8") as f:
        json.dump(jfeed, f, indent=2, ensure_ascii=False)
    print(f"  ✅ feed.json updated — {len(all_jitems)} item(s)")

def main():
    import time
    global _signals_tokens_used
    _signals_tokens_used = 0   # reset per-process token counter for signals path

    # ── Mode selection ────────────────────────────────────────────────────────
    # --mode intraday   : generate narrative + signals only (default, 12×/day)
    # --mode structural : generate session context + currency drivers only (1×/day)
    # --mode all        : generate all four outputs (legacy / manual use)
    #
    # Tiered Refresh Architecture — each mode uses a dedicated key pool so that
    # structural calls (session context + drivers) never compete with intraday
    # calls (narrative + signals) for the same daily token quota.
    _mode_arg = "all"
    for _a in sys.argv[1:]:
        if _a.startswith("--mode="):
            _mode_arg = _a.split("=", 1)[1].strip().lower()
        elif _a == "--mode" and sys.argv.index(_a) + 1 < len(sys.argv):
            _mode_arg = sys.argv[sys.argv.index(_a) + 1].strip().lower()
    RUN_INTRADAY   = _mode_arg in ("all", "intraday")
    RUN_STRUCTURAL = _mode_arg in ("all", "structural")
    if _mode_arg not in ("all", "intraday", "structural"):
        print(f"ERROR: Unknown --mode '{_mode_arg}'. Use: intraday | structural | all", file=sys.stderr)
        sys.exit(1)
    print(f"Run mode: {_mode_arg.upper()}")

    keys = load_groq_keys()
    gemini_keys_available = load_gemini_keys()
    print(f"Gemini keys available: {len(gemini_keys_available)} (primary editor)")
    if keys:
        print(f"Groq keys available: {len(keys)} (fallback + structural)")
    else:
        print("Groq keys: none — Gemini-only mode (fallback unavailable)")

    if not gemini_keys_available and not keys:
        print("ERROR: No API keys found. Set GEMINI_API_KEY and/or GROQ_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # ── Groq key pool split (for fallback + structural only) ─────────────────
    # Groq is now the fallback editor and handles: currency drivers, session context.
    # Narrative + signals use Gemini as primary — Groq pool is backup only.
    # Pool split kept for compatibility; in Gemini-primary mode the split is less
    # critical since Groq is only called when all Gemini keys fail.
    #   5 keys  → narrative fallback: [0,1]  · signals fallback: [2,3,4]
    #   4 keys  → narrative: [0,1]            · signals: [2,3]
    #   3 keys  → narrative: [0,1]            · signals: [1,2]
    #   2 keys  → narrative: [0]              · signals: [1]
    #   1 key   → both share [0]
    if RUN_INTRADAY and len(keys) >= 5:
        _narrative_keys = keys[:2]
        _signals_keys   = keys[2:5]
    elif RUN_INTRADAY and len(keys) == 4:
        _narrative_keys = keys[:2]
        _signals_keys   = keys[2:4]
    elif RUN_INTRADAY and len(keys) == 3:
        _narrative_keys = keys[:2]
        _signals_keys   = keys[1:]
    elif RUN_INTRADAY and len(keys) == 2:
        _narrative_keys = keys[:1]
        _signals_keys   = keys[1:]
    else:
        _narrative_keys = keys
        _signals_keys   = keys
    if RUN_INTRADAY and keys:
        print(f"  Groq fallback pool — narrative: {len(_narrative_keys)} key(s), signals: {len(_signals_keys)} key(s)")

    out_dir = SITE_DIR / "ai-analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building context from available data files...")
    market_closed, next_session = is_market_closed()
    if market_closed:
        print(f"  Market status: CLOSED — using closed-market prompts (next open: {next_session})")
    else:
        print("  Market status: OPEN — using live prompts")
    context = build_context()
    print(f"Context length: {len(context)} chars")

    if "--debug-context" in sys.argv:
        print("\n--- CONTEXT ---")
        print(context[:4000])
        print("--- END ---\n")

    def call_with_key_rotation(fn_name, fn, *args, key_pool=None):
        """Call fn(api_key, *args) rotating through keys on DAILY_LIMIT.

        key_pool: explicit list of keys to use (defaults to `keys` for structural /
        single-key mode). Intraday narrative and signals pass their dedicated sub-pools
        (_narrative_keys / _signals_keys) so exhaustion of one pool cannot prevent
        the other from executing.

        For generate_narrative specifically: passes the remaining keys as _extra_keys
        so the internal retry can rotate without bubbling back up to this loop.
        This prevents the retry from being silently dropped when api_key is exhausted.

        Convention for generate_narrative args: (context, system_prompt)
          → called as fn(key, context, _extra_keys=extra, system_prompt=system_prompt)
        Convention for generate_signals args: (context, intraday_updated, system_prompt)
          → called as fn(key, context, intraday_updated, system_prompt=system_prompt)
        """
        _pool = key_pool if key_pool is not None else keys
        key_idx = 0
        while key_idx < len(_pool):
            key = _pool[key_idx]
            try:
                print(f"  Groq Key {key_idx+1} ({mask_key(key)})...")
                if fn is generate_narrative:
                    # args = (context, system_prompt)
                    ctx        = args[0]
                    sys_prompt = args[1] if len(args) > 1 else None
                    extra      = _pool[key_idx+1:] if key_idx+1 < len(_pool) else []
                    return fn(key, ctx, _extra_keys=extra, system_prompt=sys_prompt)
                elif fn is generate_signals:
                    # args = (context, intraday_updated, system_prompt)
                    ctx         = args[0]
                    intra_upd   = args[1] if len(args) > 1 else ""
                    sys_prompt  = args[2] if len(args) > 2 else None
                    extra       = _pool[key_idx+1:] if key_idx+1 < len(_pool) else []
                    result = fn(key, ctx, intraday_updated=intra_upd, system_prompt=sys_prompt, _extra_keys=extra)
                    # Treat an empty list as a degenerate response (key silently exhausted without
                    # raising DAILY_LIMIT). Rotate to next key so Python-side guards can run.
                    if not result:
                        key_idx += 1
                        if key_idx < len(_pool):
                            print(f"  WARNING generate_signals returned [] — degenerate response. Rotating to Key {key_idx+1}...")
                            time.sleep(KEY_SWITCH_PAUSE_DAILY)
                            continue
                        else:
                            # All pool keys returned [] (degenerate / silent exhaustion under RPM).
                            # Wait 60s for RPM windows to reset, then retry the full pool.
                            print(f"  WARNING generate_signals returned [] and all pool keys exhausted — ")
                            print(f"  INFO RPM recovery wait 60s then retrying pool keys...")
                            time.sleep(60)
                            ctx2       = args[0]
                            intra_upd2 = args[1] if len(args) > 1 else ""
                            sys_prompt2= args[2] if len(args) > 2 else None
                            for _rpm_k_idx, _rpm_k in enumerate(_pool):
                                try:
                                    print(f"  INFO RPM recovery — trying Key {_rpm_k_idx+1} ({mask_key(_rpm_k)})...")
                                    result2 = fn(_rpm_k, ctx2, intraday_updated=intra_upd2,
                                                 system_prompt=sys_prompt2, _extra_keys=[])
                                    if result2:
                                        print(f"  INFO RPM recovery Key {_rpm_k_idx+1} produced {len(result2)} signal(s).")
                                        return result2
                                    else:
                                        print(f"  WARNING RPM recovery Key {_rpm_k_idx+1} also returned [] — trying next.")
                                        time.sleep(KEY_SWITCH_PAUSE_RATE)
                                except Exception as _rpm_e:
                                    err_rpm = str(_rpm_e)
                                    if "DAILY_LIMIT" in err_rpm:
                                        print(f"  WARNING RPM recovery Key {_rpm_k_idx+1} daily limit — skipping.")
                                    else:
                                        print(f"  WARNING RPM recovery Key {_rpm_k_idx+1} failed ({_rpm_e}) — trying next.")
                                    time.sleep(KEY_SWITCH_PAUSE_RATE)
                            print("  WARNING RPM recovery exhausted all pool keys — returning [].")
                    return result
                elif fn is generate_currency_drivers:
                    # Pass ALL keys so the function can rotate internally on DAILY_LIMIT mid-loop.
                    # args = (context,)
                    ctx = args[0]
                    return fn(key, ctx, _all_keys=keys)
                elif fn is generate_session_context:
                    # Same pattern as generate_currency_drivers — pass all keys for intra-loop rotation.
                    ctx = args[0]
                    return fn(key, ctx, _all_keys=keys)
                return fn(key, *args)
            except RuntimeError as e:
                err = str(e)
                if "DAILY_LIMIT" in err or "RATE_LIMIT" in err:
                    is_daily = "DAILY_LIMIT" in err
                    label = "daily limit reached" if is_daily else "rate limit — rotating"
                    pause = KEY_SWITCH_PAUSE_DAILY if is_daily else KEY_SWITCH_PAUSE_RATE
                    print(f"  ⛔ Key {key_idx+1} {label}")
                    key_idx += 1
                    if key_idx < len(_pool):
                        print(f"  🔄 Switching to Key {key_idx+1} — pausing {pause}s...")
                        time.sleep(pause)
                    else:
                        print("  ⛔ All pool keys exhausted")
                        raise RuntimeError(f"{fn_name}: all Groq keys exhausted") from e
                else:
                    raise
        raise RuntimeError(f"{fn_name}: no keys available")

    narrative_file = out_dir / "index.json"
    signals_file   = out_dir / "signals.json"

    # Load existing content — kept if Groq is unavailable (keys exhausted/daily limit)
    existing_narrative = load_json(narrative_file)
    _existing_signals_raw = load_json(signals_file, default=[])
    # Handle both old format (plain array) and new format ({generated_at, signals})
    if isinstance(_existing_signals_raw, dict):
        existing_signals = _existing_signals_raw.get("signals", [])
        existing_signals_generated_at = _existing_signals_raw.get("generated_at")
    else:
        existing_signals = _existing_signals_raw  # backward compat: old plain array
        existing_signals_generated_at = None

    # ── Change detection — skip Groq if market data is unchanged ─────────────
    # Builds a snapshot of analytically significant fields (FX prices, cross-asset,
    # COT weekEnding, high-impact headlines) and compares against the previous run.
    # Only calls Groq when something material has changed — avoids regenerating
    # identical content on the 12×/day schedule when prices are flat.
    # Override with --force flag or workflow_dispatch (already handled by GitHub).
    force_regen = "--force" in sys.argv
    curr_snap   = build_snapshot()
    prev_snap   = load_snapshot(out_dir)

    # When market is closed: skip regeneration if previous run was also closed.
    # Weekend prices are static Friday closing levels — nothing changes between runs.
    if market_closed and prev_snap.get("market_closed") and not force_regen:
        print("  ⏭  Market closed and previous run was also closed — skipping Groq (no new data).")
        print("Done.")
        return

    changed, change_reason = market_data_changed(prev_snap, curr_snap)
    if not changed and not force_regen and existing_narrative and existing_signals:
        # Update checked_at so the frontend knows the run completed.
        # Also apply fix_direction_mismatch to the preserved narrative in case it was
        # generated before this fix was deployed (e.g. "gold higher -0.91%").
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        existing_narrative["checked_at"] = now_str
        tokens = extract_tokens_from_intraday()
        raw_text = existing_narrative.get("narrative", "")
        fixed = fix_direction_mismatch(raw_text, tokens)
        if fixed != raw_text:
            existing_narrative["narrative"] = fixed
            print(f"  INFO Direction mismatch corrected in preserved narrative.")
        with open(narrative_file, "w") as f:
            json.dump(existing_narrative, f, indent=2, ensure_ascii=False)
        print(f"  ⏭  Context unchanged ({change_reason}) — skipping Groq. "
              f"Preserved narrative (regime={existing_narrative.get('regime','?')}).")
        print("Done.")
        return

    if force_regen:
        print("  🔄 Force regeneration requested (--force flag).")
    else:
        print(f"  ✅ Market data changed: {change_reason} — proceeding with Gemini.")

    # ─────────────────────────────────────────────────────────────────────────

    # ── INTRADAY LAYER: Narrative + Signals ──────────────────────────────────
    # Runs 12×/day via generate-ai-narrative.yml.
    # Gemini is the primary editor (v7.54.0). Groq is the fallback.
    # Skipped entirely when --mode structural.
    #
    # narrative_prompt / signals_prompt are initialised here (before the guard)
    # so they are always bound — referencing them inside a nested `if RUN_INTRADAY`
    # block while the call sites sit outside caused UnboundLocalError in --mode structural
    # (fixed v7.64.3: hoist initialisation out of the inner guard).
    narrative_prompt = NARRATIVE_SYSTEM_CLOSED if market_closed else None  # None = use default
    signals_prompt   = SIGNALS_SYSTEM_CLOSED   if market_closed else None

    if RUN_INTRADAY:
        # ── Narrative — Gemini primary, Groq fallback ────────────────────────
        print("Generating narrative via Gemini...")
        narrative = None
        _gemini_keys_narr = load_gemini_keys()
        if _gemini_keys_narr:
            narrative = generate_narrative_via_gemini(
                _gemini_keys_narr, context, system_prompt=narrative_prompt
            )
        if narrative:
            narrative = _apply_narrative_guards(narrative)
            narrative["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(narrative_file, "w") as f:
                json.dump(narrative, f, indent=2, ensure_ascii=False)
            print(f"  Regime:    {narrative['regime']}")
            print(f"  Narrative: {narrative['narrative']}")
        else:
            # Gemini failed — try Groq as fallback (generate_narrative has guards built-in)
            print("  Gemini narrative failed — trying Groq fallback...")
            try:
                narrative = call_with_key_rotation("narrative", generate_narrative, context, narrative_prompt, key_pool=_narrative_keys)
                with open(narrative_file, "w") as f:
                    json.dump(narrative, f, indent=2, ensure_ascii=False)
                print(f"  Regime:    {narrative['regime']}")
                print(f"  Narrative: {narrative['narrative']}")
            except Exception as e:
                print(f"  ⚠️  Groq narrative fallback also failed: {e}", file=sys.stderr)
                if existing_narrative and existing_narrative.get("narrative", "").strip():
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

        # ── Signals — Gemini primary, Groq fallback ──────────────────────────
        intraday_meta = load_json(SITE_DIR / "intraday-data" / "quotes.json")
        intraday_updated = (intraday_meta or {}).get("updated", "") if intraday_meta else ""

        print(f"Pausing {KEY_SWITCH_PAUSE_RATE}s before signals call...")
        time.sleep(KEY_SWITCH_PAUSE_RATE)
        print("Generating signals via Gemini...")
        signals_generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        def _write_signals(sigs: list, source: str = "gemini") -> None:
            with open(signals_file, "w") as f:
                json.dump({"generated_at": signals_generated_at, "signals": sigs}, f,
                          indent=2, ensure_ascii=False)
            print(f"  Wrote {len(sigs)} signal(s) to signals.json (source: {source})")
            for s in sigs:
                print(f"    [{s.get('priority','?').upper()[:4]}] {s.get('title','?')}: {str(s.get('text',''))[:80]}")

        signals = []
        _gemini_keys_sig = load_gemini_keys()
        if _gemini_keys_sig:
            signals = generate_signals_via_gemini(_gemini_keys_sig, context, system_prompt=signals_prompt)
            if signals:
                signals = _apply_signal_guards(signals)

        if signals:
            _write_signals(signals, source="gemini")
        else:
            # Gemini failed — try Groq as fallback
            print("  Gemini signals failed — trying Groq fallback...")
            try:
                # Use a reduced context for Groq to avoid 413 Payload Too Large.
                # The full context (~16k chars) exceeds Groq's payload limit when
                # combined with the SIGNALS_SYSTEM prompt (~9k chars) and injections.
                # build_drivers_context() strips the largest sections (retail sentiment,
                # cross-asset correlations, headlines) that Groq doesn't use for signals
                # anyway — keeps CB rates, OIS, COT, intraday FX, and yield curve.
                _groq_signals_ctx = build_drivers_context(context)
                signals = call_with_key_rotation("signals", generate_signals, _groq_signals_ctx, intraday_updated, signals_prompt, key_pool=_signals_keys)
                if signals:
                    _write_signals(signals, source="groq-fallback")
                else:
                    if existing_signals:
                        print(f"  ✅ Preserved {len(existing_signals)} previous signals (Gemini+Groq both returned []).")
                    else:
                        print(f"  ⚠️  No previous signals and both Gemini and Groq failed — signals.json empty.")
                        with open(signals_file, "w") as f:
                            json.dump({"generated_at": signals_generated_at, "signals": []}, f)
            except Exception as e:
                print(f"  ⚠️  Groq signals fallback also failed: {e}", file=sys.stderr)
                if existing_signals:
                    print(f"  ✅ Preserved {len(existing_signals)} previous signals (Gemini+Groq both failed).")
                else:
                    with open(signals_file, "w") as f:
                        json.dump({"generated_at": signals_generated_at, "signals": []}, f)

    # ── Session Context — per-currency per-session notes (INTRADAY) ────────
    # Moved from structural layer (v7.64.6) — 1×/day at 06:00 UTC was stale by
    # 15 hours when Sydney opened at 21:00 UTC. Now regenerates every intraday
    # run (~every 2h), matching the Bloomberg/Eikon convention of refreshing
    # session notes at each major session open. Token cost: ~2,800 tokens/run
    # (Gemini primary — same key pool already allocated to this workflow).
    # currency-drivers.json remains in generate-structural-context.yml
    # (COT/CB/OIS inputs change weekly, not intraday).
    session_ctx_file = out_dir / "session-context.json"
    existing_session_ctx = load_json(session_ctx_file, default={})

    if market_closed:
        # Weekend: generate recap notes (Friday close / weekly range / Monday outlook).
        # Only regenerate once per weekend (check if existing file is already a closed recap
        # generated today to avoid burning Gemini tokens on repeated weekend runs).
        existing_closed_at = existing_session_ctx.get("generated_at", "")[:10] if existing_session_ctx.get("market_closed") else ""
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if existing_closed_at == today_str and not force_regen:
            print(f"  ⏭  Weekend session recap already generated today ({existing_closed_at}) — preserving.")
        else:
            print("Pausing 3s before weekend session context call...")
            time.sleep(3)
            print("Generating weekend session recap via Gemini (Groq fallback)...")
            session_ctx_generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                session_ctx = call_with_key_rotation(
                    "session_context_closed", generate_session_context_closed, context
                )
                if session_ctx:
                    payload = {
                        "generated_at": session_ctx_generated_at,
                        "sessions": session_ctx,
                        "market_closed": True,
                    }
                    with open(session_ctx_file, "w") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                    print(f"  Generated weekend session recap for: {', '.join(session_ctx.keys())}")
                else:
                    print("  WARNING Empty weekend session context — keeping previous file.")
            except Exception as e:
                print(f"  ⚠️  Gemini/Groq unavailable for weekend session context: {e}", file=sys.stderr)
                if existing_session_ctx.get("sessions"):
                    print("  ✅ Preserved previous session context.")
                else:
                    with open(session_ctx_file, "w") as f:
                        json.dump({"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                   "sessions": {}, "market_closed": True}, f)
    else:
        print("Pausing 3s before session context call...")
        time.sleep(3)
        print("Generating session context via Gemini (Groq fallback)...")
        session_ctx_generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            session_ctx = call_with_key_rotation("session_context", generate_session_context, context)
            if session_ctx:
                payload = {"generated_at": session_ctx_generated_at, "sessions": session_ctx}
                with open(session_ctx_file, "w") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                print(f"  Generated session context for: {', '.join(session_ctx.keys())}")
            else:
                print("  WARNING Empty session context response — keeping previous file.")
        except Exception as e:
            print(f"  ⚠️  Gemini/Groq unavailable for session context: {e}", file=sys.stderr)
            if existing_session_ctx.get("sessions"):
                print("  ✅ Preserved previous session context.")
            else:
                with open(session_ctx_file, "w") as f:
                    json.dump({"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                               "sessions": {}}, f)

    # END RUN_INTRADAY block

    # ── STRUCTURAL LAYER: Currency Drivers only ───────────────────────────
    # Runs 1×/day via generate-structural-context.yml (key 5).
    # Session context moved to RUN_INTRADAY block above (v7.64.6).
    # Skipped entirely when --mode intraday.

    # ── Currency Drivers — one-line Groq note per G8 currency ───────────────
    # Written to ai-analysis/currency-drivers.json.
    # Read by heatmap-modal.js to populate the Correlations tab driver notes.
    # Skipped when market is closed (prices are static Friday levels — no intraday driver to describe).
    #
    # STRUCTURAL CHANGE DETECTION — drivers are regenerated only when their
    # underlying structural inputs change, not on every intraday price move.
    # Inputs that drive driver content:
    #   - COT weekEnding (CFTC weekly — changes Fridays)
    #   - CB rate lastUpdate per currency (changes on meeting decisions)
    #   - OIS bias + biasUpdated per currency (changes weekly via workflow_meetings.yml)
    # Intraday FX price moves do NOT change the driver narrative — that context
    # lives in the narrative and signals. This mirrors how Bloomberg/Eikon handle
    # currency driver panels: structural notes update on events, not on pip moves.
    drivers_file = out_dir / "currency-drivers.json"
    existing_drivers = load_json(drivers_file, default={})

    def _build_drivers_snapshot() -> dict:
        """Snapshot of structural inputs that determine driver content."""
        snap = {}
        meetings_raw = load_json(SITE_DIR / "meetings-data" / "meetings.json") or {}
        meetings = meetings_raw.get("meetings", {})
        for ccy in CURRENCIES:
            entry = {}
            # COT weekEnding
            cot = load_json(SITE_DIR / "cot-data" / f"{ccy}.json") or {}
            entry["cot_we"] = cot.get("weekEnding") or cot.get("reportDate")
            # CB rate last update
            rates = load_json(SITE_DIR / "rates" / f"{ccy}.json") or {}
            entry["rate_updated"] = rates.get("lastUpdate") or rates.get("rateDate")
            # OIS bias + biasUpdated
            mtg = meetings.get(ccy, {})
            entry["bias"] = mtg.get("bias")
            entry["bias_updated"] = mtg.get("biasUpdated")
            snap[ccy] = entry
        return snap

    def _drivers_inputs_changed(prev_drv_snap: dict, curr_drv_snap: dict) -> tuple:
        """Return (changed: bool, reason: str) comparing two driver snapshots."""
        for ccy in CURRENCIES:
            prev = prev_drv_snap.get(ccy, {})
            curr = curr_drv_snap.get(ccy, {})
            for field, label in [("cot_we", "COT weekEnding"), ("rate_updated", "CB rate"),
                                  ("bias", "OIS bias"), ("bias_updated", "biasUpdated")]:
                pv, cv = prev.get(field), curr.get(field)
                if pv != cv:
                    return True, f"{ccy} {label} changed: {pv} → {cv}"
        return False, "no structural change"

    if RUN_STRUCTURAL:
        # Load driver snapshot from previous run (stored inside currency-drivers.json)
        prev_drivers_snap = existing_drivers.get("_inputs_snap", {})
        curr_drivers_snap = _build_drivers_snapshot()
        drivers_inputs_changed, drivers_change_reason = _drivers_inputs_changed(
            prev_drivers_snap, curr_drivers_snap
        )

        # On --force flag, always regenerate drivers too
        if force_regen:
            drivers_inputs_changed = True
            drivers_change_reason = "--force flag"

        # FIX-21: if _inputs_snap is empty ({}), the system is frozen from a previous
        # failed run that wrote the empty payload. Force regeneration to recover.
        if not drivers_inputs_changed and not prev_drivers_snap:
            drivers_inputs_changed = True
            drivers_change_reason = "empty _inputs_snap detected — recovering from frozen state"

        if market_closed:
            print("  ⏭  Market closed — skipping currency drivers (no intraday data).")
        elif not drivers_inputs_changed and existing_drivers.get("drivers"):
            print(f"  ⏭  Currency drivers unchanged ({drivers_change_reason}) — preserving existing notes.")
        else:
            if drivers_inputs_changed:
                print(f"  ✅ Driver inputs changed: {drivers_change_reason} — regenerating.")
            print("Pausing 4s before currency drivers call...")
            time.sleep(4)
            print("Generating currency drivers via Gemini (Groq fallback)...")
            drivers_generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                drivers = call_with_key_rotation("drivers", generate_currency_drivers, context)
                # FIX-19 (v7.79.0): check for actual content, not just truthiness.
                # generate_currency_drivers returns {"EUR": {}, "GBP": {}, ...} when ALL
                # Gemini + Groq keys fail — a truthy but empty dict. Previously this caused
                # the empty payload to be written AND _inputs_snap to be updated, so subsequent
                # runs saw "no structural change" and skipped forever, leaving drivers stale.
                # Fix: require at least one currency to have non-empty pair notes.
                # When all fail: keep the previous file and do NOT update _inputs_snap so
                # the next run retries the API calls.
                has_content = bool(drivers and any(v for v in drivers.values()))
                if has_content:
                    drivers = _sanitize_driver_notes(drivers)
                    drivers = _sanitize_cot_values(drivers, SITE_DIR)
                    covered = sum(1 for v in drivers.values() if v)
                    payload = {"generated_at": drivers_generated_at, "drivers": drivers, "_inputs_snap": curr_drivers_snap}
                    with open(drivers_file, "w") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                    print(f"  Generated drivers for: {', '.join(drivers.keys())} ({covered}/8 with content)")
                    for ccy, note in drivers.items():
                        print(f"    {ccy}: {note}")
                else:
                    print("  WARNING Empty drivers response (all keys failed) — keeping previous file.")
                    print("  NOTE: _inputs_snap NOT updated — next run will retry API calls.")
            except Exception as e:
                print(f"  ⚠️  Groq unavailable for currency drivers: {e}", file=sys.stderr)
                if existing_drivers.get("drivers"):
                    print(f"  ✅ Preserved previous currency drivers.")
                else:
                    with open(drivers_file, "w") as f:
                        json.dump({"generated_at": drivers_generated_at, "drivers": {}}, f)
    else:
        print("  ⏭  Skipping currency drivers (mode=intraday).")

    # Session context has been moved to the RUN_INTRADAY block above (v7.64.6).
    # Removed from structural layer — no action needed here.

    # ── Save snapshot for next run's change detection ───────────────────────
    # Also update RSS / JSON Feed with the latest narrative + signals.
    _feeds_narrative = load_json(out_dir / "index.json") or {}
    _feeds_signals_raw = load_json(out_dir / "signals.json") or {}
    _feeds_signals = (_feeds_signals_raw.get("signals", [])
                      if isinstance(_feeds_signals_raw, dict)
                      else _feeds_signals_raw)
    print("Updating feeds...")
    write_feeds(SITE_DIR, _feeds_narrative, _feeds_signals)

    curr_snap["market_closed"] = market_closed
    save_snapshot(out_dir, curr_snap)
    print(f"  📸 Context snapshot saved.")
    print("Done.")


if __name__ == "__main__":
    main()
