#!/usr/bin/env python3
"""
fetch_myfxbook_sentiment.py  v2.0 — Myfxbook Community Outlook via REST API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API docs: https://www.myfxbook.com/api
  - Login:   GET /api/login.json?email=X&password=Y  -> { session: "TOKEN" }
  - Outlook: GET /api/get-community-outlook.json?session=TOKEN
             -> { symbols: [ {name, shortPercentage, longPercentage, ...} ] }
  - Logout:  GET /api/logout.json?session=TOKEN

NOTAS (API v1.38, oct 2025):
  - Sesiones IP-bound, TTL 1 mes.
  - get-community-outlook solo acepta "session", NO "email".
  - symbols es un ARRAY con campo "name" (ej: "EURUSD"), no un dict.
  - Limite free: 100 requests/24h.
"""

import os, sys, json, time
from datetime import datetime, timezone
from urllib.parse import urlencode

try:
    import requests
except ImportError:
    print("[ERROR] requests no instalado.")
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

_http = requests.Session()
_http.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json",
})

def api_get(path, params):
    qs  = urlencode(params)
    url = f"{BASE_URL}/{path}?{qs}"
    r   = _http.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def main():
    site_path = os.environ.get("SITE_PATH", ".")
    out_dir   = os.path.join(site_path, "sentiment-data")
    out_file  = os.path.join(out_dir, "myfxbook.json")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*60}")
    print(f"fetch_myfxbook_sentiment.py  v2.0  --  {ts}")
    print(f"{'='*60}\n")

    if not MYFXBOOK_EMAIL or not MYFXBOOK_PASSWORD:
        print("[ERROR] MYFXBOOK_EMAIL and MYFXBOOK_PASSWORD environment variables required.")
        sys.exit(1)

    # STEP 1: Login
    print(f"[Auth] Logging in as {MYFXBOOK_EMAIL}...")
    login_data = api_get("login.json", {"email": MYFXBOOK_EMAIL, "password": MYFXBOOK_PASSWORD})
    print(f"[Auth] Response: error={login_data.get('error')} message='{login_data.get('message','')}'")

    if login_data.get("error"):
        print(f"[Auth] Login failed: {login_data.get('message','unknown')}")
        sys.exit(1)

    session_token = login_data.get("session", "")
    if not session_token:
        print(f"[Auth] No session token. Full response: {login_data}")
        sys.exit(1)

    print(f"[Auth] Login OK. Session: {session_token[:8]}... (len={len(session_token)})")
    time.sleep(2)

    # STEP 2: Community Outlook — solo session, sin email
    print("[API]  Fetching community outlook...")
    try:
        outlook_data = api_get("get-community-outlook.json", {"session": session_token})
    except Exception as e:
        print(f"[API]  Request failed: {e}")
        sys.exit(1)

    print(f"[API]  Response: error={outlook_data.get('error')} message='{outlook_data.get('message','')}'")

    if outlook_data.get("error"):
        print(f"[API]  Error from server: {outlook_data.get('message','unknown')}")
        print(f"[API]  Full response: {json.dumps(outlook_data)[:500]}")
        sys.exit(1)

    # symbols es un ARRAY: [{name, shortPercentage, longPercentage, ...}]
    symbols_list = outlook_data.get("symbols", [])
    if not symbols_list:
        print(f"[API]  No symbols. Response keys: {list(outlook_data.keys())}")
        sys.exit(1)

    print(f"[API]  Got {len(symbols_list)} symbols")

    # STEP 3: Normalizar
    sym_map = {item.get("name","").upper().replace("/",""): item for item in symbols_list}

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
        pairs.append({"sym": display, "long": long_pct, "short": short_pct, "longVol": long_vol, "shortVol": short_vol})
        bias = "LONG " if long_pct >= short_pct else "SHORT"
        print(f"  {display:10s}  long={long_pct:3d}%  short={short_pct:3d}%  [{bias}]")

    if not pairs:
        print(f"[ERROR] No pairs normalized. sym_map keys: {list(sym_map.keys())[:5]}")
        sys.exit(1)

    # STEP 4: Logout
    try:
        api_get("logout.json", {"session": session_token})
        print("\n[Auth] Logged out OK")
    except Exception as e:
        print(f"\n[Auth] Logout warning (non-fatal): {e}")

    # STEP 5: Escribir JSON
    output = {"updated": ts, "source": "myfxbook", "pairs": pairs}
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"OK {len(pairs)} pairs -> {out_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
