#!/usr/bin/env python3
"""
fetch_myfxbook_sentiment.py  v8.1
=============================================================
Fetches Myfxbook Community Outlook sentiment data directly
from the Myfxbook API — no Cloudflare Worker needed.

Architecture:
  GitHub Actions (ubuntu-latest) → login.json → get-community-outlook.json → logout.json

Output format matches dashboard.js expectations exactly.
=============================================================
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from urllib.parse import urlencode, unquote

SCRIPT_VERSION = "8.1"
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"\n{'='*60}")
print(f"fetch_myfxbook_sentiment.py  v{SCRIPT_VERSION}  --  {TIMESTAMP}")
print(f"{'='*60}")

EMAIL       = os.environ.get("MYFXBOOK_EMAIL", "").strip()
PASSWORD    = os.environ.get("MYFXBOOK_PASSWORD", "").strip()
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "sentiment-data/myfxbook.json")
MYFXBOOK_BASE = "https://www.myfxbook.com/api"

# The 24 pairs the dashboard tracks, in order, with slash format
TRACKED_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF",
    "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY", "EUR/AUD", "EUR/CAD",
    "EUR/CHF", "EUR/NZD", "AUD/JPY", "AUD/CAD", "AUD/CHF", "AUD/NZD",
    "GBP/CHF", "GBP/AUD", "GBP/CAD", "GBP/NZD", "CHF/JPY", "CAD/JPY",
]

if not EMAIL or not PASSWORD:
    print("[Error] MYFXBOOK_EMAIL or MYFXBOOK_PASSWORD not set.")
    sys.exit(1)

print(f"[Config] Email: ********")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

existing_data = {}
if os.path.exists(OUTPUT_PATH):
    try:
        with open(OUTPUT_PATH) as f:
            existing_data = json.load(f)
    except Exception:
        pass

last_fetch = existing_data.get("updated", "never")
pair_count = len(existing_data.get("pairs", []))

def api_get(path, params):
    url = f"{MYFXBOOK_BASE}/{path}?{urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())

def save_fallback(reason):
    fallback = dict(existing_data)
    fallback["apiBlocked"] = True
    fallback["updated"]    = TIMESTAMP
    fallback["blockReason"] = reason
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(fallback, f, indent=2, ensure_ascii=False)
    print(f"[Fallback] Existing data preserved. Pairs in cache: {pair_count}. Last fetch: {last_fetch}")
    print(f"[Exit] Exiting with code 0 (non-fatal) — reason: {reason}")
    sys.exit(0)

# Login
print("[Login] Logging in to Myfxbook...")
try:
    data = api_get("login.json", {"email": EMAIL, "password": PASSWORD})
except Exception as e:
    save_fallback(f"Network error on login: {e}")

if data.get("error"):
    save_fallback(f"Login failed: {data.get('message', 'Unknown error')}")

session = unquote(data.get("session", ""))
print("[Login] Success.")

# Outlook
print("[Outlook] Fetching community outlook...")
try:
    outlook = api_get("get-community-outlook.json", {"session": session})
except Exception as e:
    save_fallback(f"Network error on outlook: {e}")

if outlook.get("error"):
    save_fallback(f"Outlook failed: {outlook.get('message', 'Unknown error')}")

# Logout
try:
    api_get("logout.json", {"session": session})
    print("[Logout] Done.")
except Exception:
    pass

# Build lookup by sym with slash format
raw_lookup = {}
for s in outlook.get("symbols", []):
    name = (s.get("name") or "").upper()
    # Convert EURUSD → EUR/USD (insert slash at position 3)
    if len(name) == 6:
        name = name[:3] + "/" + name[3:]
    raw_lookup[name] = s

# Build pairs list — only tracked pairs, in order
pairs_list = []
for sym in TRACKED_PAIRS:
    s = raw_lookup.get(sym)
    if not s:
        continue
    pairs_list.append({
        "sym":       sym,
        "long":      int(s.get("longPercentage")  or 0),
        "short":     int(s.get("shortPercentage") or 0),
        "longVol":   float(s.get("longVolume")    or 0),
        "shortVol":  float(s.get("shortVolume")   or 0),
        "longPos":   int(s.get("longPositions")   or 0),
        "shortPos":  int(s.get("shortPositions")  or 0),
        "totalPos":  int(s.get("totalPositions")  or 0),
        "avgLongPx": float(s.get("avgLongPrice")  or 0),
        "avgShortPx":float(s.get("avgShortPrice") or 0),
    })

general = outlook.get("general", {})
print(f"[Success] Received {len(pairs_list)} tracked pairs.")

output = {
    "updated":    TIMESTAMP,
    "source":     "myfxbook",
    "apiBlocked": False,
    "pairs":      pairs_list,
    "general": {
        "profitablePercentage":    general.get("profitablePercentage", 0),
        "nonProfitablePercentage": general.get("nonProfitablePercentage", 0),
        "realAccountsPercentage":  general.get("realAccountsPercentage", 0),
        "demoAccountsPercentage":  general.get("demoAccountsPercentage", 0),
        "totalFunds":              general.get("totalFunds", ""),
        "averageDeposit":          general.get("averageDeposit", ""),
        "averageAccountProfit":    general.get("averageAccountProfit", ""),
        "averageAccountLoss":      general.get("averageAccountLoss", ""),
    },
}

os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"[Output] Written to {OUTPUT_PATH}")
print(f"[Exit] Success.")
