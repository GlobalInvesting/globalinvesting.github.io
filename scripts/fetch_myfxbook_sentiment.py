#!/usr/bin/env python3
"""
fetch_myfxbook_sentiment.py  v3.0 — Myfxbook Community Outlook via REST API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce:  sentiment-data/myfxbook.json
Schedule: Cada 30 min en días hábiles (via GitHub Action en repo público)

API docs: https://www.myfxbook.com/api  (v1.38, oct 2025)
  - Login:   GET /api/login.json?email=X&password=Y  -> { session: "TOKEN" }
  - Outlook: GET /api/get-community-outlook.json?session=TOKEN
             -> { symbols: [ {name, shortPercentage, longPercentage, ...} ] }
  - Logout:  GET /api/logout.json?session=TOKEN

NOTAS TÉCNICAS:
  - El session token puede contener caracteres ya URL-encoded (ej: %2F, %3D).
    Hay que pasarlo RAW en la URL — no re-encodear, no usar params=.
  - Sessions son IP-bound (desde oct 2025). Usar una sola requests.Session()
    para login y todas las calls siguientes.
  - symbols es un ARRAY: [{name: "EURUSD", longPercentage, shortPercentage, ...}]
  - Limite free: 100 requests/24h.

CREDENCIALES (GitHub Secrets):
  MYFXBOOK_EMAIL    — email de la cuenta
  MYFXBOOK_PASSWORD — contraseña de la cuenta
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sys, json, time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("[ERROR] requests no instalado. Correr: pip install requests")
    sys.exit(1)

BASE_URL          = "https://www.myfxbook.com/api"
MYFXBOOK_EMAIL    = os.environ.get("MYFXBOOK_EMAIL", "")
MYFXBOOK_PASSWORD = os.environ.get("MYFXBOOK_PASSWORD", "")

SYMBOL_DISPLAY = {
    "EURUSD":"EUR/USD","GBPUSD":"GBP/USD","USDJPY":"USD/JPY","USDCHF":"USD/CHF",
    "EURCHF":"EUR/CHF","EURGBP":"EUR/GBP","USDCAD":"USD/CAD","EURJPY":"EUR/JPY",
    "EURCAD":"EUR/CAD","AUDUSD":"AUD/USD","AUDJPY":"AUD/JPY","GBPJPY":"GBP/JPY",
    "CHFJPY":"CHF/JPY","EURAUD":"EUR/AUD","NZDUSD":"NZD/USD","GBPCHF":"GBP/CHF",
    "EURNZD":"EUR/NZD","AUDCAD":"AUD/CAD","GBPCAD":"GBP/CAD","AUDCHF":"AUD/CHF",
    "GBPAUD":"GBP/AUD","AUDNZD":"AUD/NZD","CADJPY":"CAD/JPY","GBPNZD":"GBP/NZD",
}

PRIORITY_ORDER = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP",
    "EURJPY","GBPJPY","EURAUD","EURCAD","EURCHF","EURNZD","AUDJPY","AUDCAD",
    "AUDCHF","AUDNZD","GBPCHF","GBPAUD","GBPCAD","GBPNZD","CHFJPY","CADJPY",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

TIMEOUT = 20


def main():
    site_path = os.environ.get("SITE_PATH", ".")
    out_dir   = os.path.join(site_path, "sentiment-data")
    out_file  = os.path.join(out_dir, "myfxbook.json")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*60}")
    print(f"fetch_myfxbook_sentiment.py  v3.0  --  {ts}")
    print(f"{'='*60}\n")

    if not MYFXBOOK_EMAIL or not MYFXBOOK_PASSWORD:
        print("[ERROR] MYFXBOOK_EMAIL and MYFXBOOK_PASSWORD environment variables required.")
        print("        Set them as GitHub Secrets: MYFXBOOK_EMAIL, MYFXBOOK_PASSWORD")
        sys.exit(1)

    # Una sola Session para todas las requests (IP-bound sessions)
    http = requests.Session()
    http.headers.update(HEADERS)

    # ── STEP 1: Login ────────────────────────────────────────────────────────
    print(f"[Auth] Logging in as {MYFXBOOK_EMAIL}...")
    r = http.get(f"{BASE_URL}/login.json",
                 params={"email": MYFXBOOK_EMAIL, "password": MYFXBOOK_PASSWORD},
                 timeout=TIMEOUT)
    r.raise_for_status()
    login_data = r.json()

    if login_data.get("error"):
        print(f"[Auth] Login failed: {login_data.get('message', 'unknown')}")
        sys.exit(1)

    # IMPORTANTE: el token puede contener caracteres ya URL-encoded (%2F, %3D, etc.)
    # Guardarlo RAW — no pasar por urlencode ni quote() en calls posteriores.
    token = login_data.get("session", "")
    if not token:
        print(f"[Auth] No session token in response: {login_data}")
        sys.exit(1)

    print(f"[Auth] Login OK. Session: {token[:8]}...")
    time.sleep(1)

    # ── STEP 2: Community Outlook ────────────────────────────────────────────
    # Token raw en la URL — NO usar params= (requests re-encodearía el token)
    print("[API]  Fetching community outlook...")
    r2 = http.get(f"{BASE_URL}/get-community-outlook.json?session={token}", timeout=TIMEOUT)
    r2.raise_for_status()
    outlook_data = r2.json()

    if outlook_data.get("error"):
        print(f"[API]  Error: {outlook_data.get('message', 'unknown')}")
        sys.exit(1)

    # symbols es un ARRAY: [{name, longPercentage, shortPercentage, longVolume, shortVolume, ...}]
    symbols_list = outlook_data.get("symbols", [])
    if not symbols_list:
        print(f"[API]  No symbols in response. Keys: {list(outlook_data.keys())}")
        sys.exit(1)

    print(f"[API]  Got {len(symbols_list)} symbols")

    # ── STEP 3: Normalizar ───────────────────────────────────────────────────
    sym_map = {item.get("name", "").upper().replace("/", ""): item for item in symbols_list}
    pairs = []

    for api_name in PRIORITY_ORDER:
        raw = sym_map.get(api_name)
        if not raw:
            continue
        display   = SYMBOL_DISPLAY.get(api_name, api_name)
        long_pct  = round(float(raw.get("longPercentage",  0) or 0))
        short_pct = round(float(raw.get("shortPercentage", 0) or 0))
        long_vol  = round(float(raw.get("longVolume",  0) or 0), 2)
        short_vol = round(float(raw.get("shortVolume", 0) or 0), 2)

        total = long_pct + short_pct
        if total > 0 and total != 100:
            long_pct  = round(long_pct / total * 100)
            short_pct = 100 - long_pct

        long_pos  = int(raw.get("longPositions",  0) or 0)
        short_pos = int(raw.get("shortPositions", 0) or 0)
        total_pos = int(raw.get("totalPositions", long_pos + short_pos) or long_pos + short_pos)
        avg_long  = round(float(raw.get("avgLongPrice",  0) or 0), 5)
        avg_short = round(float(raw.get("avgShortPrice", 0) or 0), 5)

        pairs.append({
            "sym":       display,
            "long":      long_pct,
            "short":     short_pct,
            "longVol":   long_vol,
            "shortVol":  short_vol,
            "longPos":   long_pos,
            "shortPos":  short_pos,
            "totalPos":  total_pos,
            "avgLongPx": avg_long,
            "avgShortPx":avg_short,
        })

        bias = "LONG " if long_pct >= short_pct else "SHORT"
        print(f"  {display:10s}  long={long_pct:3d}%  short={short_pct:3d}%  [{bias}]")

    if not pairs:
        print("[ERROR] Could not normalize any pairs from API response")
        sys.exit(1)

    # ── STEP 4: Extract general stats ───────────────────────────────────────
    raw_general = outlook_data.get("general", {}) or {}
    general = {
        "profitablePercentage":    raw_general.get("profitablePercentage",    0),
        "nonProfitablePercentage": raw_general.get("nonProfitablePercentage", 0),
        "realAccountsPercentage":  raw_general.get("realAccountsPercentage",  0),
        "demoAccountsPercentage":  raw_general.get("demoAccountsPercentage",  0),
        "totalFunds":              raw_general.get("totalFunds",              ""),
        "averageDeposit":          raw_general.get("averageDeposit",          ""),
        "averageAccountProfit":    raw_general.get("averageAccountProfit",    ""),
        "averageAccountLoss":      raw_general.get("averageAccountLoss",      ""),
    } if raw_general else None

    # ── STEP 5: Logout ───────────────────────────────────────────────────────
    try:
        http.get(f"{BASE_URL}/logout.json?session={token}", timeout=10)
        print("\n[Auth] Logged out OK")
    except Exception as e:
        print(f"\n[Auth] Logout warning (non-fatal): {e}")

    # ── STEP 6: Escribir JSON ────────────────────────────────────────────────
    output = {"updated": ts, "source": "myfxbook", "pairs": pairs, "general": general}
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"OK {len(pairs)} pairs -> {out_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
