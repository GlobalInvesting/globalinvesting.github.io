#!/usr/bin/env python3
"""
fetch_myfxbook_sentiment.py  v4.0 — Myfxbook Community Outlook via REST API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce:  sentiment-data/myfxbook.json
Schedule: Each hour on business days (via GitHub Action in public repo)

API docs: https://www.myfxbook.com/api  (v1.38, oct 2025)
  - Login:   GET /api/login.json?email=X&password=Y  -> { session: "TOKEN" }
  - Outlook: GET /api/get-community-outlook.json?session=TOKEN
             -> { symbols: [ {name, shortPercentage, longPercentage, ...} ] }
  - Logout:  GET /api/logout.json?session=TOKEN

TECHNICAL NOTES:
  - Session token may contain URL-encoded characters (e.g. %2F, %3D).
    Pass RAW in the URL — do not re-encode, do not use params=.
  - Sessions are IP-bound (since oct 2025). Use a single requests.Session()
    for login and all subsequent calls.
  - symbols is an ARRAY: [{name: "EURUSD", longPercentage, shortPercentage, ...}]
  - Free limit: 100 requests/24h.

GRACEFUL DEGRADATION (v4.0):
  - On 403 / API block: exit 0 (non-fatal). Existing JSON is preserved with
    apiBlocked=true flag. Dashboard falls back to COT/Dukascopy sources.
  - On network timeout or transient error: 3 retries with exponential backoff,
    then graceful exit preserving existing data.
  - The workflow no longer fails when Myfxbook blocks GitHub Actions IPs.

CREDENTIALS (GitHub Secrets):
  MYFXBOOK_EMAIL    — account email
  MYFXBOOK_PASSWORD — account password
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sys, json, time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("[ERROR] requests not installed. Run: pip install requests")
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

# Rotate User-Agent strings — GitHub Actions IPs may be recognised and blocked
# when using a static UA. Cycling through common browser UAs reduces this risk.
USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
]

TIMEOUT      = 25
MAX_RETRIES  = 3
RETRY_DELAY  = 4   # seconds (doubles on each retry)


def build_headers(attempt=0):
    ua = USER_AGENTS[attempt % len(USER_AGENTS)]
    return {
        "User-Agent":      ua,
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
        "Referer":         "https://www.myfxbook.com/community/outlook",
        "Sec-Fetch-Dest":  "empty",
        "Sec-Fetch-Mode":  "cors",
        "Sec-Fetch-Site":  "same-origin",
    }


def preserve_existing(out_file, ts, reason):
    """
    Mark the existing JSON with apiBlocked=true so the dashboard knows
    the data is cached, then exit 0 (non-fatal — workflow succeeds).
    """
    existing = {}
    if os.path.exists(out_file):
        try:
            with open(out_file) as f:
                existing = json.load(f)
        except Exception:
            pass

    if existing.get("pairs"):
        existing["apiBlocked"]  = True
        existing["blockReason"] = reason
        existing["blockTs"]     = ts
        with open(out_file, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\n[Fallback] Existing data preserved with apiBlocked=true.")
        print(f"           Dashboard will fall back to COT / Dukascopy sources.")
        print(f"           Pairs in cache: {len(existing['pairs'])}")
        print(f"           Last successful fetch: {existing.get('updated', 'unknown')}")
    else:
        print(f"\n[Fallback] No existing data to preserve. Dashboard will use COT fallback.")

    print(f"\n[Exit] Exiting with code 0 (non-fatal) — reason: {reason}")
    sys.exit(0)


def attempt_login(http, attempt, out_file, ts):
    """
    Try to log in. Returns session token string on success,
    None on soft/retriable failure.
    Calls preserve_existing (exits 0) on unrecoverable errors.
    """
    http.headers.update(build_headers(attempt))
    delay = RETRY_DELAY * (2 ** attempt)

    print(f"[Auth] Login attempt {attempt + 1}/{MAX_RETRIES} as {MYFXBOOK_EMAIL}...")

    try:
        r = http.get(
            f"{BASE_URL}/login.json",
            params={"email": MYFXBOOK_EMAIL, "password": MYFXBOOK_PASSWORD},
            timeout=TIMEOUT,
        )
    except requests.exceptions.Timeout:
        print(f"[Auth] Timeout on attempt {attempt + 1}. Waiting {delay}s...")
        time.sleep(delay)
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"[Auth] Connection error on attempt {attempt + 1}: {e}. Waiting {delay}s...")
        time.sleep(delay)
        return None

    if r.status_code == 403:
        # 403 = Forbidden — Myfxbook is blocking this request.
        # Most common cause: GitHub Actions IP ranges are blocked by the provider.
        # Treat as non-fatal: preserve existing data and exit 0.
        print(f"\n[Auth] HTTP 403 Forbidden — Myfxbook API is blocking this request.")
        print(f"       Possible causes:")
        print(f"         1. GitHub Actions IP ranges are blocked by Myfxbook.")
        print(f"         2. Account credentials have changed.")
        print(f"         3. Myfxbook has changed the login endpoint.")
        print(f"       Action: preserving last known data, exiting non-fatally.")
        preserve_existing(out_file, ts, "HTTP 403 — API blocked GitHub Actions IPs")
        # preserve_existing calls sys.exit(0) — line below is unreachable
        return None

    if r.status_code == 429:
        print(f"[Auth] HTTP 429 Rate Limited. Waiting {delay}s...")
        time.sleep(delay)
        return None

    if r.status_code >= 500:
        print(f"[Auth] HTTP {r.status_code} Server Error. Waiting {delay}s...")
        time.sleep(delay)
        return None

    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"[Auth] HTTP error: {e}. Waiting {delay}s...")
        time.sleep(delay)
        return None

    try:
        login_data = r.json()
    except ValueError:
        print(f"[Auth] Non-JSON response (status {r.status_code}). Body: {r.text[:200]}")
        time.sleep(delay)
        return None

    if login_data.get("error"):
        msg = login_data.get("message", "unknown")
        print(f"[Auth] API returned error: {msg}")
        # Credential errors are non-retriable
        if any(kw in msg.lower() for kw in ("invalid", "password", "email", "wrong")):
            preserve_existing(out_file, ts, f"credential error: {msg}")
        time.sleep(delay)
        return None

    token = login_data.get("session", "")
    if not token:
        print(f"[Auth] No session token in response: {login_data}")
        time.sleep(delay)
        return None

    return token


def main():
    site_path = os.environ.get("SITE_PATH", ".")
    out_dir   = os.path.join(site_path, "sentiment-data")
    out_file  = os.path.join(out_dir, "myfxbook.json")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*60}")
    print(f"fetch_myfxbook_sentiment.py  v4.0  --  {ts}")
    print(f"{'='*60}\n")

    if not MYFXBOOK_EMAIL or not MYFXBOOK_PASSWORD:
        print("[ERROR] MYFXBOOK_EMAIL and MYFXBOOK_PASSWORD environment variables required.")
        print("        Set them as GitHub Secrets: MYFXBOOK_EMAIL, MYFXBOOK_PASSWORD")
        sys.exit(1)

    # Single Session for all requests (IP-bound sessions since oct 2025)
    http = requests.Session()

    # ── STEP 1: Login (with retries) ─────────────────────────────────────────
    token = None
    for attempt in range(MAX_RETRIES):
        result = attempt_login(http, attempt, out_file, ts)
        if result:
            token = result
            break

    if not token:
        print(f"[Auth] Login failed after {MAX_RETRIES} attempts.")
        preserve_existing(out_file, ts, f"login failed after {MAX_RETRIES} attempts")

    print(f"[Auth] Login OK. Session: {token[:8]}...")
    time.sleep(1)

    # ── STEP 2: Community Outlook ─────────────────────────────────────────────
    # Pass token RAW in URL — do NOT use params= (requests would re-encode the token)
    print("[API]  Fetching community outlook...")
    outlook_data = None
    for attempt in range(MAX_RETRIES):
        delay = RETRY_DELAY * (2 ** attempt)
        try:
            r2 = http.get(
                f"{BASE_URL}/get-community-outlook.json?session={token}",
                timeout=TIMEOUT,
            )
            if r2.status_code == 403:
                print(f"[API]  HTTP 403 on outlook fetch.")
                preserve_existing(out_file, ts, "HTTP 403 on outlook endpoint")
            if r2.status_code == 429:
                print(f"[API]  HTTP 429 Rate Limited. Waiting {delay}s...")
                time.sleep(delay)
                continue
            r2.raise_for_status()
            outlook_data = r2.json()
            break
        except requests.exceptions.Timeout:
            print(f"[API]  Timeout on attempt {attempt + 1}. Waiting {delay}s...")
            time.sleep(delay)
        except requests.exceptions.ConnectionError as e:
            print(f"[API]  Connection error on attempt {attempt + 1}: {e}. Waiting {delay}s...")
            time.sleep(delay)
        except requests.exceptions.HTTPError as e:
            print(f"[API]  HTTP error: {e}. Waiting {delay}s...")
            time.sleep(delay)
        except ValueError:
            print(f"[API]  Non-JSON response. Waiting {delay}s...")
            time.sleep(delay)

    if outlook_data is None:
        print(f"[API]  Could not fetch outlook after {MAX_RETRIES} attempts.")
        preserve_existing(out_file, ts, f"outlook fetch failed after {MAX_RETRIES} attempts")

    if outlook_data.get("error"):
        msg = outlook_data.get("message", "unknown")
        print(f"[API]  Error in outlook response: {msg}")
        preserve_existing(out_file, ts, f"outlook API error: {msg}")

    # symbols is an ARRAY: [{name, longPercentage, shortPercentage, longVolume, shortVolume, ...}]
    symbols_list = outlook_data.get("symbols", [])
    if not symbols_list:
        print(f"[API]  No symbols in response. Keys: {list(outlook_data.keys())}")
        preserve_existing(out_file, ts, "empty symbols list in outlook response")

    print(f"[API]  Got {len(symbols_list)} symbols")

    # ── STEP 3: Normalize ─────────────────────────────────────────────────────
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
        preserve_existing(out_file, ts, "could not normalize any pairs from response")

    # ── STEP 4: Extract general stats ─────────────────────────────────────────
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

    # ── STEP 5: Logout ────────────────────────────────────────────────────────
    try:
        http.get(f"{BASE_URL}/logout.json?session={token}", timeout=10)
        print("\n[Auth] Logged out OK")
    except Exception as e:
        print(f"\n[Auth] Logout warning (non-fatal): {e}")

    # ── STEP 6: Write JSON ────────────────────────────────────────────────────
    output = {
        "updated":    ts,
        "source":     "myfxbook",
        "apiBlocked": False,
        "pairs":      pairs,
        "general":    general,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"OK {len(pairs)} pairs -> {out_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
