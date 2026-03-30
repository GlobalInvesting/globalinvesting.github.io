#!/usr/bin/env python3
"""
fetch_intraday_quotes.py  v1.0 — Intraday quotes via Twelve Data + Alpha Vantage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produces:  site/intraday-data/quotes.json
Schedule:  Every 15 min on weekdays, US/EU market hours (via GitHub Action)

SYMBOLS COVERED:
  VIX       — CBOE Volatility Index
  SPX       — S&P 500 Index
  GOLD      — XAU/USD (Gold spot)
  WTI       — WTI Crude Oil (front-month futures)
  US10Y     — US 10-Year Treasury Yield
  NIKKEI    — Nikkei 225
  STOXX     — Euro Stoxx 50
  DXY       — US Dollar Index

FETCH STRATEGY (per symbol):
  1. Twelve Data  — primary (most symbols, real-time delayed ~15min free tier)
  2. Alpha Vantage — fallback if Twelve Data fails for a symbol
  3. Previous repo value (extended-data/USD.json) — last resort, marks as stale

OUTPUT FORMAT:
  {
    "updated": "2026-03-30T14:30:00Z",
    "source": "twelve_data",          // or "alpha_vantage" / "mixed"
    "quotes": {
      "vix":   { "close": 18.5, "prev_close": 17.9, "chg": 0.6, "pct": 3.35 },
      "spx":   { "close": 5200, "prev_close": 5180, "chg": 20,  "pct": 0.39 },
      "gold":  { ... },
      "wti":   { ... },
      "us10y": { "close": 4.31, "prev_close": 4.28, "chg": 0.03, "pct": 0.70 },
      "nikkei":{ ... },
      "stoxx": { ... },
      "dxy":   { ... }
    }
  }
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import time
import requests
from datetime import datetime, timezone

# ── API Keys (from GitHub Secrets) ──────────────────────────────────────────
TWELVE_DATA_API_KEY   = os.environ.get("TWELVE_DATA_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

# ── Symbol maps ──────────────────────────────────────────────────────────────
# Twelve Data symbols (https://twelvedata.com/symbols)
TWELVE_DATA_SYMBOLS = {
    "vix":    "VIX",
    "spx":    "SPX",
    "gold":   "XAU/USD",
    "wti":    "WTI/USD",       # crude oil
    "us10y":  "TNX",           # CBOE 10Y Treasury Yield Index (x10 in TD)
    "nikkei": "NI225",
    "stoxx":  "STOXX50E",
    "dxy":    "DXY",
}

# Alpha Vantage symbols — used as fallback
# AV uses different symbol conventions per function
ALPHA_VANTAGE_SYMBOLS = {
    "vix":    {"function": "GLOBAL_QUOTE", "symbol": "^VIX"},
    "spx":    {"function": "GLOBAL_QUOTE", "symbol": "^GSPC"},
    "gold":   {"function": "CURRENCY_EXCHANGE_RATE", "from": "XAU", "to": "USD"},
    "wti":    {"function": "GLOBAL_QUOTE", "symbol": "USO"},   # WTI ETF proxy
    "us10y":  {"function": "GLOBAL_QUOTE", "symbol": "^TNX"},
    "nikkei": {"function": "GLOBAL_QUOTE", "symbol": "^N225"},
    "stoxx":  {"function": "GLOBAL_QUOTE", "symbol": "^STOXX50E"},
    "dxy":    {"function": "GLOBAL_QUOTE", "symbol": "DX-Y.NYB"},
}

# Yield symbols in Twelve Data are returned as percentages already (not ×10)
YIELD_SYMBOLS = {"us10y"}

# ── Twelve Data: batch quote ─────────────────────────────────────────────────

def fetch_twelve_data_batch(symbols_map: dict) -> dict:
    """
    Fetch all symbols in a single batch call to Twelve Data /quote endpoint.
    Returns dict: internal_id → {close, prev_close, chg, pct} or None on failure.
    """
    if not TWELVE_DATA_API_KEY:
        print("[TwelveData] No API key — skipping")
        return {}

    td_symbols = ",".join(symbols_map.values())
    url = "https://api.twelvedata.com/quote"
    params = {
        "symbol": td_symbols,
        "apikey": TWELVE_DATA_API_KEY,
        "dp": 4,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[TwelveData] Request failed: {e}")
        return {}

    results = {}
    # If only one symbol, TD returns the object directly (not wrapped in symbol key)
    if len(symbols_map) == 1:
        key = list(symbols_map.keys())[0]
        td_sym = list(symbols_map.values())[0]
        data = {td_sym: data}

    # Reverse map: TD symbol → internal id
    reverse_map = {v: k for k, v in symbols_map.items()}

    for td_sym, internal_id in reverse_map.items():
        raw = data.get(td_sym, {})
        if not raw or raw.get("status") == "error" or not raw.get("close"):
            print(f"[TwelveData] No data for {td_sym}: {raw.get('message', 'empty')}")
            results[internal_id] = None
            continue

        try:
            close      = float(raw["close"])
            prev_close = float(raw.get("previous_close") or raw.get("open") or close)
            chg        = close - prev_close
            pct        = (chg / prev_close * 100) if prev_close != 0 else 0.0

            # US10Y via TD TNX: value is already in percent (e.g. 4.31)
            # No scaling needed unlike Yahoo Finance (which ×10)

            results[internal_id] = {
                "close":      round(close, 4),
                "prev_close": round(prev_close, 4),
                "chg":        round(chg, 4),
                "pct":        round(pct, 4),
                "source":     "twelve_data",
            }
            print(f"[TwelveData] ✓ {internal_id} ({td_sym}): {close}")
        except (ValueError, TypeError) as e:
            print(f"[TwelveData] Parse error for {td_sym}: {e}")
            results[internal_id] = None

    return results


# ── Alpha Vantage: per-symbol fallback ───────────────────────────────────────

def fetch_alpha_vantage_quote(internal_id: str, cfg: dict) -> dict | None:
    """Fetch a single symbol from Alpha Vantage. Returns quote dict or None."""
    if not ALPHA_VANTAGE_API_KEY:
        print(f"[AlphaVantage] No API key — skipping {internal_id}")
        return None

    base_url = "https://www.alphavantage.co/query"
    function = cfg["function"]

    try:
        if function == "GLOBAL_QUOTE":
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": cfg["symbol"],
                "apikey": ALPHA_VANTAGE_API_KEY,
            }
            resp = requests.get(base_url, params=params, timeout=12)
            resp.raise_for_status()
            raw = resp.json().get("Global Quote", {})
            if not raw or not raw.get("05. price"):
                print(f"[AlphaVantage] Empty quote for {internal_id}")
                return None

            close      = float(raw["05. price"])
            prev_close = float(raw.get("08. previous close") or close)
            chg        = close - prev_close
            pct        = (chg / prev_close * 100) if prev_close != 0 else 0.0

            # TNX in Alpha Vantage is ×10, so divide to get actual yield %
            if internal_id == "us10y" and close > 10:
                close /= 10; prev_close /= 10; chg /= 10

        elif function == "CURRENCY_EXCHANGE_RATE":
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": cfg["from"],
                "to_currency": cfg["to"],
                "apikey": ALPHA_VANTAGE_API_KEY,
            }
            resp = requests.get(base_url, params=params, timeout=12)
            resp.raise_for_status()
            raw = resp.json().get("Realtime Currency Exchange Rate", {})
            if not raw or not raw.get("5. Exchange Rate"):
                print(f"[AlphaVantage] Empty FX for {internal_id}")
                return None
            close      = float(raw["5. Exchange Rate"])
            prev_close = close   # AV FX endpoint doesn't provide prev close
            chg        = 0.0
            pct        = 0.0
        else:
            return None

        result = {
            "close":      round(close, 4),
            "prev_close": round(prev_close, 4),
            "chg":        round(chg, 4),
            "pct":        round(pct, 4),
            "source":     "alpha_vantage",
        }
        print(f"[AlphaVantage] ✓ {internal_id}: {close}")
        return result

    except Exception as e:
        print(f"[AlphaVantage] Error for {internal_id}: {e}")
        return None


# ── Repo fallback: read previous extended-data/USD.json ──────────────────────

def load_repo_fallback(site_path: str) -> dict:
    """
    Read yesterday's extended-data values as a last-resort fallback.
    Returns dict: internal_id → {close, prev_close=close, chg=0, pct=0, stale=True}
    """
    path = os.path.join(site_path, "extended-data", "USD.json")
    try:
        with open(path) as f:
            usd = json.load(f)
        d = usd.get("data", {})
        fallbacks = {}
        if d.get("vix"):
            fallbacks["vix"] = {"close": d["vix"], "prev_close": d["vix"], "chg": 0, "pct": 0, "source": "repo", "stale": True}
        if d.get("bond10y"):
            fallbacks["us10y"] = {"close": d["bond10y"], "prev_close": d["bond10y"], "chg": 0, "pct": 0, "source": "repo", "stale": True}
        return fallbacks
    except Exception as e:
        print(f"[Repo] Could not read USD.json: {e}")
        return {}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    site_path = os.environ.get("SITE_PATH", ".")
    out_dir   = os.path.join(site_path, "intraday-data")
    out_file  = os.path.join(out_dir, "quotes.json")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"fetch_intraday_quotes.py  v1.0  —  {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"{'='*60}\n")

    # ── STEP 1: Twelve Data batch ──────────────────────────────────────────
    quotes = {}
    td_results = fetch_twelve_data_batch(TWELVE_DATA_SYMBOLS)

    for internal_id, result in td_results.items():
        if result is not None:
            quotes[internal_id] = result

    # ── STEP 2: Alpha Vantage fallback for any symbol TD missed ───────────
    # AV free tier: 25 calls/day and 5 calls/min — use only for gaps
    missing_ids = [k for k in TWELVE_DATA_SYMBOLS if k not in quotes]

    if missing_ids:
        print(f"\n[AlphaVantage] Fetching {len(missing_ids)} missing symbols: {missing_ids}")
        for internal_id in missing_ids:
            cfg = ALPHA_VANTAGE_SYMBOLS.get(internal_id)
            if cfg:
                result = fetch_alpha_vantage_quote(internal_id, cfg)
                if result:
                    quotes[internal_id] = result
                # AV free tier: 5 req/min — small delay between calls
                time.sleep(13)

    # ── STEP 3: Repo fallback for anything still missing ──────────────────
    still_missing = [k for k in TWELVE_DATA_SYMBOLS if k not in quotes]
    if still_missing:
        print(f"\n[Repo] Loading fallback for: {still_missing}")
        repo_fallbacks = load_repo_fallback(site_path)
        for internal_id in still_missing:
            if internal_id in repo_fallbacks:
                quotes[internal_id] = repo_fallbacks[internal_id]

    # ── STEP 4: Determine overall source label ────────────────────────────
    sources = set(q.get("source", "unknown") for q in quotes.values())
    if sources == {"twelve_data"}:
        source_label = "twelve_data"
    elif sources == {"alpha_vantage"}:
        source_label = "alpha_vantage"
    elif "repo" in sources and len(sources) == 1:
        source_label = "repo"
    else:
        source_label = "mixed"

    # ── STEP 5: Write output JSON ─────────────────────────────────────────
    output = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source":  source_label,
        "quotes":  quotes,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Written {len(quotes)}/{len(TWELVE_DATA_SYMBOLS)} symbols → {out_file}")
    print(f"   Source: {source_label}")
    for sym, q in quotes.items():
        stale = " [STALE]" if q.get("stale") else ""
        print(f"   {sym:8s}  {q['close']:>12.4f}  {q['pct']:+.2f}%  [{q['source']}]{stale}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
