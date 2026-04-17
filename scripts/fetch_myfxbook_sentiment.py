#!/usr/bin/env python3
"""
fetch_myfxbook_sentiment.py  v5.0 — Myfxbook via Cloudflare Worker proxy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce:  sentiment-data/myfxbook.json
Schedule: Each hour on business days (via GitHub Action in public repo)

WHY THIS EXISTS (v5.0):
  Myfxbook blocks GitHub Actions IP ranges at the Cloudflare WAF level
  (x-deny-reason: host_not_allowed). No authentication scheme or header
  change fixes this — the block is at the network layer.

  Solution: a Cloudflare Worker acts as proxy. GHA calls the Worker
  (CF edge IP, not on Myfxbook's denylist), the Worker forwards the
  request to Myfxbook using browser-like headers from CF's own IP range.

  Architecture:
    GHA (blocked IP) → CF Worker (CF edge IP) → Myfxbook API → Worker → GHA

CREDENTIALS (GitHub Secrets — set in repo Settings → Secrets → Actions):
  MYFXBOOK_EMAIL    — Myfxbook account email
  MYFXBOOK_PASSWORD — Myfxbook account password
  CF_WORKER_URL     — Full URL of deployed Cloudflare Worker
                      e.g. https://myfxbook-proxy.YOUR_SUBDOMAIN.workers.dev
  CF_WORKER_SECRET  — Shared secret (must match WORKER_SECRET env var in CF Worker)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sys, json, time
from datetime import datetime, timezone
from urllib.parse import urlencode

try:
    import requests
except ImportError:
    print("[ERROR] requests not installed. Run: pip install requests")
    sys.exit(1)

MYFXBOOK_EMAIL    = os.environ.get("MYFXBOOK_EMAIL", "")
MYFXBOOK_PASSWORD = os.environ.get("MYFXBOOK_PASSWORD", "")
CF_WORKER_URL     = os.environ.get("CF_WORKER_URL", "").rstrip("/")
CF_WORKER_SECRET  = os.environ.get("CF_WORKER_SECRET", "")

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

TIMEOUT     = 25
MAX_RETRIES = 3
RETRY_DELAY = 4   # seconds (doubles on each retry)


def worker_get(action, params, out_file, ts):
    """Call the CF Worker proxy. Returns parsed JSON on success, exits 0 on failure."""
    all_params = {"action": action, **params}
    url = f"{CF_WORKER_URL}?{urlencode(all_params)}"
    headers = {"X-Worker-Secret": CF_WORKER_SECRET}

    for attempt in range(MAX_RETRIES):
        delay = RETRY_DELAY * (2 ** attempt)
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
        except requests.exceptions.Timeout:
            print(f"[Worker] Timeout on {action} attempt {attempt + 1}. Waiting {delay}s...")
            time.sleep(delay)
            continue
        except requests.exceptions.ConnectionError as e:
            print(f"[Worker] Connection error on {action} attempt {attempt + 1}: {e}. Waiting {delay}s...")
            time.sleep(delay)
            continue

        if r.status_code == 401:
            print(f"[Worker] HTTP 401 — check CF_WORKER_SECRET matches Worker WORKER_SECRET env var.")
            preserve_existing(out_file, ts, "Worker 401 — CF_WORKER_SECRET mismatch")

        if r.status_code == 403:
            print(f"[Worker] HTTP 403 from upstream — Myfxbook blocking CF Worker IPs.")
            preserve_existing(out_file, ts, "HTTP 403 — Myfxbook blocking CF Worker IPs")

        if r.status_code == 429:
            print(f"[Worker] HTTP 429 Rate Limited. Waiting {delay}s...")
            time.sleep(delay)
            continue

        if r.status_code >= 500:
            print(f"[Worker] HTTP {r.status_code} server error. Waiting {delay}s...")
            time.sleep(delay)
            continue

        try:
            return r.json()
        except ValueError:
            print(f"[Worker] Non-JSON response on {action} (status {r.status_code}). Waiting {delay}s...")
            time.sleep(delay)
            continue

    print(f"[Worker] {action} failed after {MAX_RETRIES} attempts.")
    preserve_existing(out_file, ts, f"{action} failed after {MAX_RETRIES} attempts")


def preserve_existing(out_file, ts, reason):
    """Mark existing JSON with apiBlocked=true and exit 0 (non-fatal)."""
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
        print(f"           Dashboard will fall back to Dukascopy / static sources.")
        print(f"           Pairs in cache: {len(existing['pairs'])}")
        print(f"           Last successful fetch: {existing.get('updated', 'unknown')}")
    else:
        print(f"\n[Fallback] No existing data. Dashboard will use Dukascopy fallback.")

    print(f"\n[Exit] Exiting with code 0 (non-fatal) — reason: {reason}")
    sys.exit(0)


def main():
    site_path = os.environ.get("SITE_PATH", ".")
    out_dir   = os.path.join(site_path, "sentiment-data")
    out_file  = os.path.join(out_dir, "myfxbook.json")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*60}")
    print(f"fetch_myfxbook_sentiment.py  v5.0  --  {ts}")
    print(f"{'='*60}\n")

    missing = [v for v in ["MYFXBOOK_EMAIL","MYFXBOOK_PASSWORD","CF_WORKER_URL","CF_WORKER_SECRET"]
               if not os.environ.get(v)]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        print(f"        Set them as GitHub Secrets in repo Settings → Secrets → Actions.")
        sys.exit(1)

    print(f"[Config] Worker: {CF_WORKER_URL}")
    print(f"[Config] Account: {MYFXBOOK_EMAIL}")

    # STEP 1: Login via Worker
    print(f"\n[Auth] Logging in via CF Worker...")
    login_data = worker_get("login", {"email": MYFXBOOK_EMAIL, "password": MYFXBOOK_PASSWORD}, out_file, ts)
    if login_data.get("error"):
        preserve_existing(out_file, ts, f"Myfxbook login error: {login_data.get('message','unknown')}")
    token = login_data.get("session", "")
    if not token:
        preserve_existing(out_file, ts, "no session token in login response")
    print(f"[Auth] Login OK. Session: {token[:8]}...")
    time.sleep(1)

    # STEP 2: Community Outlook via Worker
    print("[API]  Fetching community outlook via CF Worker...")
    outlook_data = worker_get("outlook", {"session": token}, out_file, ts)
    if outlook_data.get("error"):
        preserve_existing(out_file, ts, f"Myfxbook outlook error: {outlook_data.get('message','unknown')}")
    symbols_list = outlook_data.get("symbols", [])
    if not symbols_list:
        preserve_existing(out_file, ts, "empty symbols list in outlook response")
    print(f"[API]  Got {len(symbols_list)} symbols")

    # STEP 3: Normalize
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
        long_pos  = int(raw.get("longPositions",  0) or 0)
        short_pos = int(raw.get("shortPositions", 0) or 0)
        total_pos = int(raw.get("totalPositions", long_pos + short_pos) or long_pos + short_pos)
        avg_long  = round(float(raw.get("avgLongPrice",  0) or 0), 5)
        avg_short = round(float(raw.get("avgShortPrice", 0) or 0), 5)
        pairs.append({
            "sym": display, "long": long_pct, "short": short_pct,
            "longVol": long_vol, "shortVol": short_vol,
            "longPos": long_pos, "shortPos": short_pos, "totalPos": total_pos,
            "avgLongPx": avg_long, "avgShortPx": avg_short,
        })
        bias = "LONG " if long_pct >= short_pct else "SHORT"
        print(f"  {display:10s}  long={long_pct:3d}%  short={short_pct:3d}%  [{bias}]")

    if not pairs:
        preserve_existing(out_file, ts, "could not normalize any pairs from response")

    # STEP 4: General stats
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

    # STEP 5: Logout via Worker (non-fatal if it fails)
    try:
        worker_get("logout", {"session": token}, out_file, ts)
        print("\n[Auth] Logged out OK")
    except Exception as e:
        print(f"\n[Auth] Logout warning (non-fatal): {e}")

    # STEP 6: Write JSON
    output = {
        "updated": ts, "source": "myfxbook",
        "apiBlocked": False, "pairs": pairs, "general": general,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"OK {len(pairs)} pairs -> {out_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
