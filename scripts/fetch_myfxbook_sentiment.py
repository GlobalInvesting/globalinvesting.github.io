#!/usr/bin/env python3
"""
fetch_myfxbook_sentiment.py  v2.2 — diagnóstico token + POST fallback
"""

import os, sys, json, time, base64
from datetime import datetime, timezone
from urllib.parse import quote, unquote

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

def try_outlook(http, token_str, label):
    """Intenta obtener community outlook con el token dado. Retorna data o None."""
    print(f"\n[Try]  Attempt: {label}")
    url = f"{BASE_URL}/get-community-outlook.json?session={token_str}"
    print(f"[Try]  URL session prefix (32 chars): {url[len(BASE_URL)+35:len(BASE_URL)+67]}")
    try:
        r = http.get(url, timeout=20)
        print(f"[Try]  HTTP {r.status_code}")
        data = r.json()
        print(f"[Try]  error={data.get('error')} message='{data.get('message','')}'")
        if not data.get("error"):
            return data
    except Exception as e:
        print(f"[Try]  Exception: {e}")
    return None

def normalize_pairs(symbols_list):
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
        pairs.append({"sym": display, "long": long_pct, "short": short_pct,
                      "longVol": long_vol, "shortVol": short_vol})
        bias = "LONG " if long_pct >= short_pct else "SHORT"
        print(f"  {display:10s}  long={long_pct:3d}%  short={short_pct:3d}%  [{bias}]")
    return pairs

def main():
    site_path = os.environ.get("SITE_PATH", ".")
    out_dir   = os.path.join(site_path, "sentiment-data")
    out_file  = os.path.join(out_dir, "myfxbook.json")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*60}")
    print(f"fetch_myfxbook_sentiment.py  v2.2  --  {ts}")
    print(f"{'='*60}\n")

    if not MYFXBOOK_EMAIL or not MYFXBOOK_PASSWORD:
        print("[ERROR] MYFXBOOK_EMAIL and MYFXBOOK_PASSWORD environment variables required.")
        sys.exit(1)

    http = requests.Session()
    http.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "application/json",
    })

    # STEP 1: Login
    print(f"[Auth] Logging in...")
    r = http.get(f"{BASE_URL}/login.json",
                 params={"email": MYFXBOOK_EMAIL, "password": MYFXBOOK_PASSWORD},
                 timeout=20)
    r.raise_for_status()
    login_data = r.json()

    if login_data.get("error"):
        print(f"[Auth] Login failed: {login_data.get('message','unknown')}")
        sys.exit(1)

    token_raw = login_data.get("session", "")
    if not token_raw:
        print(f"[Auth] No session token.")
        sys.exit(1)

    print(f"[Auth] Login OK. Token len={len(token_raw)}")
    # Log token encoded in base64 so we can see it without exposing it in plaintext
    token_b64 = base64.b64encode(token_raw.encode()).decode()
    print(f"[Auth] Token b64: {token_b64}")
    
    # Log each special character found
    special = [(i, c, hex(ord(c))) for i, c in enumerate(token_raw) if not c.isalnum()]
    print(f"[Auth] Special chars in token: {special}")

    time.sleep(1)

    # STEP 2: Probar múltiples formas de pasar el token
    outlook_data = None

    # Intento A: token tal cual (raw), construido a mano en la URL
    outlook_data = try_outlook(http, token_raw, "raw token in URL")

    # Intento B: token con quote(safe='') — encode todo
    if not outlook_data:
        outlook_data = try_outlook(http, quote(token_raw, safe=''), "quote(token, safe='')")

    # Intento C: token con quote(safe='=+/') — encode solo lo peligroso
    if not outlook_data:
        outlook_data = try_outlook(http, quote(token_raw, safe='=+/'), "quote(token, safe='=+/')")

    # Intento D: unquote primero, por si el token ya viene encoded
    if not outlook_data:
        outlook_data = try_outlook(http, unquote(token_raw), "unquote(token) then raw")

    # Intento E: unquote + re-encode
    if not outlook_data:
        outlook_data = try_outlook(http, quote(unquote(token_raw), safe=''), "unquote then quote")

    if not outlook_data:
        print("\n[FAIL] All token encoding attempts failed.")
        print("[INFO] This may be an IP-bound session issue on GitHub Actions.")
        print("[INFO] Consider using a self-hosted runner or a fixed egress IP.")
        sys.exit(1)

    symbols_list = outlook_data.get("symbols", [])
    if not symbols_list:
        print(f"[API]  No symbols. Keys: {list(outlook_data.keys())}")
        sys.exit(1)

    print(f"\n[API]  Got {len(symbols_list)} symbols")
    pairs = normalize_pairs(symbols_list)

    if not pairs:
        sys.exit(1)

    # Logout
    try:
        http.get(f"{BASE_URL}/logout.json", params={"session": token_raw}, timeout=10)
        print("\n[Auth] Logged out OK")
    except Exception as e:
        print(f"\n[Auth] Logout (non-fatal): {e}")

    output = {"updated": ts, "source": "myfxbook", "pairs": pairs}
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"OK {len(pairs)} pairs -> {out_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
