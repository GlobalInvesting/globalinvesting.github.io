#!/usr/bin/env python3
"""
fetch_myfxbook_sentiment.py  v7.0
=============================================================
Fetches Myfxbook Community Outlook sentiment data via the
Cloudflare Worker proxy using a pre-authenticated session token.

Architecture:
  GitHub Actions → CF Worker (MYFXBOOK_SESSION token) → Myfxbook outlook API

The session token is obtained automatically by the self-hosted runner
workflow (renew-myfxbook-token.yml) running on a residential IP.
Token validity: ~30 days. Renewal runs every 20 days.

On token expiry (Worker returns 401 + tokenExpired=true):
  - apiBlocked is set to true in the output JSON
  - The renewal workflow is triggered via workflow_dispatch
  - Dashboard falls back to Dukascopy / static sources
=============================================================
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

SCRIPT_VERSION = "7.0"
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"\n{'='*60}")
print(f"fetch_myfxbook_sentiment.py  v{SCRIPT_VERSION}  --  {TIMESTAMP}")
print(f"{'='*60}")

# ── Config ───────────────────────────────────────────────────────────────────
WORKER_URL    = os.environ.get("MYFXBOOK_WORKER_URL", "").strip()
WORKER_SECRET = os.environ.get("MYFXBOOK_WORKER_SECRET", "").strip()
SESSION_TOKEN = os.environ.get("MYFXBOOK_SESSION", "").strip()
OUTPUT_PATH   = os.environ.get("OUTPUT_PATH", "sentiment-data/myfxbook.json")

if not WORKER_URL:
    print("[Error] MYFXBOOK_WORKER_URL is not set.")
    sys.exit(1)
if not WORKER_SECRET:
    print("[Error] MYFXBOOK_WORKER_SECRET is not set.")
    sys.exit(1)
if not SESSION_TOKEN:
    print("[Error] MYFXBOOK_SESSION is not set. Run the token renewal workflow.")
    sys.exit(1)

print(f"[Config] Worker:  {'*' * 8}")
print(f"[Config] Account: using pre-authenticated session token")

# ── Load existing cache ──────────────────────────────────────────────────────
existing_data = {}
if os.path.exists(OUTPUT_PATH):
    try:
        with open(OUTPUT_PATH, "r") as f:
            existing_data = json.load(f)
    except Exception:
        pass

last_fetch   = existing_data.get("lastSuccessfulFetch", "never")
cached_pairs = existing_data.get("pairs", {})
pair_count   = len(cached_pairs)

# ── Call Worker ──────────────────────────────────────────────────────────────
def call_worker(session: str, retries: int = 3) -> dict:
    payload = json.dumps({
        "secret":  WORKER_SECRET,
        "session": session,
    }).encode("utf-8")

    for attempt in range(1, retries + 1):
        try:
            print(f"[Worker] Attempt {attempt}/{retries} — sending session token to Worker...")
            req = urllib.request.Request(
                WORKER_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            body = {}
            try:
                body = json.loads(e.read().decode("utf-8"))
            except Exception:
                pass

            if e.code == 401 and body.get("tokenExpired"):
                print(f"[Worker] Session token expired or invalid.")
                return {"ok": False, "tokenExpired": True, "message": body.get("message", "Session invalid")}

            print(f"[Worker] HTTP {e.code} on attempt {attempt}: {body}")
            if e.code in (400, 401, 500):
                break  # non-retryable

        except Exception as err:
            print(f"[Worker] Error on attempt {attempt}: {err}")

        if attempt < retries:
            time.sleep(5 * attempt)

    return {"ok": False, "tokenExpired": False, "message": "All retries exhausted"}


result = call_worker(SESSION_TOKEN)

# ── Handle token expiry ──────────────────────────────────────────────────────
if not result.get("ok") and result.get("tokenExpired"):
    print(f"[Token] Session expired — flagging for renewal.")
    print(f"[Fallback] Existing data preserved with apiBlocked=true.")
    print(f"           Dashboard will fall back to Dukascopy / static sources.")
    print(f"           Pairs in cache: {pair_count}")
    print(f"           Last successful fetch: {last_fetch}")

    fallback = dict(existing_data)
    fallback["apiBlocked"]       = True
    fallback["tokenExpired"]     = True
    fallback["lastAttempt"]      = TIMESTAMP
    fallback["blockReason"]      = "session_expired"

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(fallback, f, indent=2, ensure_ascii=False)

    print(f"[Exit] Exiting with code 0 (non-fatal) — token expired, renewal workflow will be triggered.")
    sys.exit(0)

# ── Handle other errors ──────────────────────────────────────────────────────
if not result.get("ok"):
    reason = result.get("message", "Unknown error")
    print(f"[Worker] Worker reported failure: {reason}")
    print(f"[Fallback] Existing data preserved with apiBlocked=true.")
    print(f"           Pairs in cache: {pair_count}")
    print(f"           Last successful fetch: {last_fetch}")

    fallback = dict(existing_data)
    fallback["apiBlocked"]  = True
    fallback["tokenExpired"] = False
    fallback["lastAttempt"] = TIMESTAMP
    fallback["blockReason"] = reason

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(fallback, f, indent=2, ensure_ascii=False)

    print(f"[Exit] Exiting with code 0 (non-fatal) — reason: {reason}")
    sys.exit(0)

# ── Success ──────────────────────────────────────────────────────────────────
pairs   = result.get("pairs", {})
general = result.get("general", {})
fetched = result.get("fetchedAt", TIMESTAMP)

print(f"[Success] Received {len(pairs)} pairs from Worker.")

output = {
    "apiBlocked":         False,
    "tokenExpired":       False,
    "lastSuccessfulFetch": fetched,
    "lastAttempt":        TIMESTAMP,
    "general": {
        "longPct":  general.get("longPct",  0),
        "shortPct": general.get("shortPct", 0),
    },
    "pairs": pairs,
}

os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"[Output] Written to {OUTPUT_PATH}")
print(f"[Exit] Success.")
