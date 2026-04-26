#!/usr/bin/env python3
"""
fetch_myfxbook_sentiment.py  v8.0
=============================================================
Fetches Myfxbook Community Outlook sentiment data directly
from the Myfxbook API — no Cloudflare Worker needed.

Architecture:
  GitHub Actions (ubuntu-latest) → login.json → get-community-outlook.json → logout.json

On login failure:
  - apiBlocked is set to true in the output JSON
  - Dashboard falls back to Dukascopy / static sources
=============================================================
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from urllib.parse import urlencode

SCRIPT_VERSION = "8.0"
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"\n{'='*60}")
print(f"fetch_myfxbook_sentiment.py  v{SCRIPT_VERSION}  --  {TIMESTAMP}")
print(f"{'='*60}")

EMAIL       = os.environ.get("MYFXBOOK_EMAIL", "").strip()
PASSWORD    = os.environ.get("MYFXBOOK_PASSWORD", "").strip()
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "sentiment-data/myfxbook.json")
MYFXBOOK_BASE = "https://www.myfxbook.com/api"

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

last_fetch = existing_data.get("lastSuccessfulFetch", "never")
pair_count = len(existing_data.get("pairs", {}))

def api_get(path, params):
    url = f"{MYFXBOOK_BASE}/{path}?{urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())

def save_fallback(reason):
    fallback = dict(existing_data)
    fallback["apiBlocked"]  = True
    fallback["lastAttempt"] = TIMESTAMP
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

from urllib.parse import unquote
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

# Normalize
pairs = {}
for s in outlook.get("symbols", []):
    name = (s.get("name") or "").upper().replace("/", "")
    if not name:
        continue
    pairs[name] = {
        "longPct":   round(float(s.get("longsPercentage")  or 0), 1),
        "shortPct":  round(float(s.get("shortsPercentage") or 0), 1),
        "longVol":   float(s.get("longVolume")  or 0),
        "shortVol":  float(s.get("shortVolume") or 0),
        "traders":   int(s.get("tradersCount")  or 0),
        "positions": int(s.get("positionsCount") or 0),
    }

general = outlook.get("general", {})
print(f"[Success] Received {len(pairs)} pairs.")

output = {
    "apiBlocked":          False,
    "lastSuccessfulFetch": TIMESTAMP,
    "lastAttempt":         TIMESTAMP,
    "general": {
        "longPct":  round(float(general.get("longsPercentage")  or 0), 1),
        "shortPct": round(float(general.get("shortsPercentage") or 0), 1),
    },
    "pairs": pairs,
}

os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"[Output] Written to {OUTPUT_PATH}")
print(f"[Exit] Success.")
