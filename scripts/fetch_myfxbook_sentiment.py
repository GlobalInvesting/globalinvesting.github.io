#!/usr/bin/env python3
"""
fetch_myfxbook_sentiment.py  v2.1 — debug IP-bound session issue
"""

import os, sys, json, time
from datetime import datetime, timezone

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

def get_public_ip(session):
    """Obtiene la IP pública que verá Myfxbook."""
    try:
        r = session.get("https://api.ipify.org?format=json", timeout=5)
        return r.json().get("ip", "unknown")
    except:
        return "unknown"

def main():
    site_path = os.environ.get("SITE_PATH", ".")
    out_dir   = os.path.join(site_path, "sentiment-data")
    out_file  = os.path.join(out_dir, "myfxbook.json")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*60}")
    print(f"fetch_myfxbook_sentiment.py  v2.1  --  {ts}")
    print(f"{'='*60}\n")

    if not MYFXBOOK_EMAIL or not MYFXBOOK_PASSWORD:
        print("[ERROR] MYFXBOOK_EMAIL and MYFXBOOK_PASSWORD environment variables required.")
        sys.exit(1)

    # Una sola Session para TODAS las requests — mismas cookies, mismo socket pool
    http = requests.Session()
    http.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "application/json",
    })

    # Verificar IP antes del login
    ip_before = get_public_ip(http)
    print(f"[Net]  Public IP before login: {ip_before}")

    # STEP 1: Login
    print(f"[Auth] Logging in as {MYFXBOOK_EMAIL}...")
    r = http.get(f"{BASE_URL}/login.json",
                 params={"email": MYFXBOOK_EMAIL, "password": MYFXBOOK_PASSWORD},
                 timeout=20)
    r.raise_for_status()
    login_data = r.json()
    print(f"[Auth] Response: error={login_data.get('error')} message='{login_data.get('message','')}'")

    if login_data.get("error"):
        print(f"[Auth] Login failed: {login_data.get('message','unknown')}")
        sys.exit(1)

    session_token = login_data.get("session", "")
    if not session_token:
        print(f"[Auth] No session token. Full response: {login_data}")
        sys.exit(1)

    print(f"[Auth] Login OK. Token len={len(session_token)}")
    print(f"[Auth] Token preview (first 16): {session_token[:16]}")
    print(f"[Auth] Token has special chars: {any(c in session_token for c in '/+= %')}")

    # Verificar IP después del login (debe ser la misma)
    ip_after = get_public_ip(http)
    print(f"[Net]  Public IP after login:  {ip_after}")
    if ip_before != ip_after:
        print(f"[Net]  WARNING: IP changed between requests! {ip_before} -> {ip_after}")
        print(f"[Net]  This is the cause of 'Invalid session' — sessions are IP-bound since Oct 2025")
    else:
        print(f"[Net]  IP stable: {ip_after}")

    time.sleep(1)

    # STEP 2: Community Outlook — pasar token con params= (requests maneja el encoding)
    print("\n[API]  Fetching community outlook...")
    print(f"[API]  URL: {BASE_URL}/get-community-outlook.json?session=***")
    
    r2 = http.get(f"{BASE_URL}/get-community-outlook.json",
                  params={"session": session_token},
                  timeout=20)
    
    print(f"[API]  HTTP status: {r2.status_code}")
    print(f"[API]  Final URL (redacted): {r2.url[:80]}...")
    
    r2.raise_for_status()
    outlook_data = r2.json()
    print(f"[API]  Response: error={outlook_data.get('error')} message='{outlook_data.get('message','')}'")

    if outlook_data.get("error"):
        print(f"[API]  Error: {outlook_data.get('message','unknown')}")
        # Intentar con token URL-encoded manualmente como fallback
        print("\n[API]  Retrying with manually encoded token...")
        from urllib.parse import quote
        encoded_token = quote(session_token, safe='')
        url_manual = f"{BASE_URL}/get-community-outlook.json?session={encoded_token}"
        print(f"[API]  Manual URL token preview: {url_manual[len(BASE_URL)+35:len(BASE_URL)+51]}...")
        r3 = http.get(url_manual, timeout=20)
        print(f"[API]  Retry HTTP status: {r3.status_code}")
        outlook_data = r3.json()
        print(f"[API]  Retry response: error={outlook_data.get('error')} message='{outlook_data.get('message','')}'")
        if outlook_data.get("error"):
            print("[API]  Both attempts failed.")
            sys.exit(1)

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
        sys.exit(1)

    # STEP 4: Logout
    try:
        http.get(f"{BASE_URL}/logout.json", params={"session": session_token}, timeout=10)
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
