#!/usr/bin/env python3
"""
fetch_myfxbook_sentiment.py  v7.0 — pre-authenticated session token
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce:  sentiment-data/myfxbook.json
Schedule: Each hour (via GitHub Action in public repo)

WHAT CHANGED IN v7.0 (vs v6.0):
  Myfxbook blocks login requests from Cloudflare Workers IP ranges.
  The login API works from a browser but returns "Wrong email/password"
  from CF datacenter IPs — regardless of credentials or architecture.

  Solution: remove the login step entirely.
  - The session token is obtained once per month by the user visiting:
      https://www.myfxbook.com/api/login.json?email=EMAIL&password=PASSWORD
    in their browser and copying the session value.
  - That token is stored as MYFXBOOK_SESSION in GitHub secrets.
  - This script POSTs the token to the CF Worker, which calls only the
    get-community-outlook endpoint (not blocked from CF IPs).
  - When the token expires (~1 month), the script exits gracefully with
    apiBlocked=true and logs a clear renewal instruction.

HOW TO RENEW THE SESSION TOKEN (once per month):
  1. Open in your browser:
     https://www.myfxbook.com/api/login.json?email=YOUR_EMAIL&password=YOUR_PASSWORD
  2. Copy the value of the "session" field from the JSON response.
  3. Go to GitHub repo → Settings → Secrets → Actions → MYFXBOOK_SESSION
  4. Update the secret with the new session value.
  The next workflow run will pick it up automatically.

SECRETS REQUIRED:
  MYFXBOOK_SESSION  — session token from manual browser login (renew monthly)
  CF_WORKER_URL     — Cloudflare Worker URL
  CF_WORKER_SECRET  — shared secret (matches WORKER_SECRET in CF env vars)

SECRETS NO LONGER NEEDED (can be deleted):
  MYFXBOOK_EMAIL, MYFXBOOK_PASSWORD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sys, json, time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("[ERROR] requests not installed. Run: pip install requests")
    sys.exit(1)

MYFXBOOK_SESSION = os.environ.get("MYFXBOOK_SESSION", "")
CF_WORKER_URL    = os.environ.get("CF_WORKER_URL",    "").rstrip("/")
CF_WORKER_SECRET = os.environ.get("CF_WORKER_SECRET", "")

TIMEOUT     = 30
MAX_RETRIES = 3
RETRY_DELAY = 5   # seconds (doubles on each retry)


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


def call_worker(out_file, ts):
    """POST session token to the CF Worker. Returns parsed data on success."""
    headers = {
        "X-Worker-Secret": CF_WORKER_SECRET,
        "Content-Type":    "application/json",
    }
    payload = {"session": MYFXBOOK_SESSION}

    for attempt in range(MAX_RETRIES):
        delay = RETRY_DELAY * (2 ** attempt)
        try:
            r = requests.post(CF_WORKER_URL, headers=headers, json=payload, timeout=TIMEOUT)
        except requests.exceptions.Timeout:
            print(f"[Worker] Timeout on attempt {attempt + 1}. Waiting {delay}s...")
            time.sleep(delay)
            continue
        except requests.exceptions.ConnectionError as e:
            print(f"[Worker] Connection error on attempt {attempt + 1}: {e}. Waiting {delay}s...")
            time.sleep(delay)
            continue

        if r.status_code == 401:
            preserve_existing(out_file, ts,
                "Worker returned 401 — check CF_WORKER_SECRET matches WORKER_SECRET in Cloudflare.")
        if r.status_code == 500:
            preserve_existing(out_file, ts,
                "Worker returned 500 — WORKER_SECRET env var not set in Cloudflare dashboard.")
        if r.status_code == 429:
            print(f"[Worker] HTTP 429 — rate limited. Waiting {delay}s...")
            time.sleep(delay)
            continue
        if r.status_code >= 500:
            print(f"[Worker] HTTP {r.status_code} server error. Waiting {delay}s...")
            time.sleep(delay)
            continue

        try:
            data = r.json()
        except ValueError:
            print(f"[Worker] Non-JSON response (status {r.status_code}). Waiting {delay}s...")
            time.sleep(delay)
            continue

        if not data.get("ok"):
            err = data.get("error", "unknown worker error")
            # Detect session expiry explicitly so the log message is actionable.
            if "expired" in err.lower() or "renew" in err.lower():
                print(f"\n[Session] TOKEN EXPIRED — renew MYFXBOOK_SESSION secret:")
                print(f"  1. Open in browser: https://www.myfxbook.com/api/login.json"
                      f"?email=YOUR_EMAIL&password=YOUR_PASSWORD")
                print(f"  2. Copy the 'session' value from the JSON response.")
                print(f"  3. Update GitHub secret MYFXBOOK_SESSION with the new value.")
            else:
                print(f"[Worker] Worker reported failure: {err}")
            preserve_existing(out_file, ts, err)

        return data

    preserve_existing(out_file, ts, f"Worker call failed after {MAX_RETRIES} attempts.")


def main():
    site_path = os.environ.get("SITE_PATH", ".")
    out_dir   = os.path.join(site_path, "sentiment-data")
    out_file  = os.path.join(out_dir, "myfxbook.json")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*60}")
    print(f"fetch_myfxbook_sentiment.py  v7.0  --  {ts}")
    print(f"{'='*60}\n")

    missing = [v for v in ["MYFXBOOK_SESSION", "CF_WORKER_URL", "CF_WORKER_SECRET"]
               if not os.environ.get(v)]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        if "MYFXBOOK_SESSION" in missing:
            print(f"\n  To get the session token, open this URL in your browser:")
            print(f"  https://www.myfxbook.com/api/login.json?email=YOUR_EMAIL&password=YOUR_PASSWORD")
            print(f"  Copy the 'session' value and save it as GitHub secret MYFXBOOK_SESSION.")
        sys.exit(1)

    print(f"[Config] Worker:  {CF_WORKER_URL}")
    print(f"[Config] Session: {MYFXBOOK_SESSION[:8]}...  (first 8 chars)")
    print(f"\n[Worker] POSTing session token — Worker will fetch community outlook...")

    data    = call_worker(out_file, ts)
    pairs   = data.get("pairs",   [])
    general = data.get("general", None)

    if not pairs:
        preserve_existing(out_file, ts, "Worker returned ok=true but empty pairs list.")

    print(f"[Worker] Got {len(pairs)} pairs:")
    for p in pairs:
        bias = "LONG " if p["long"] >= p["short"] else "SHORT"
        print(f"  {p['sym']:10s}  long={p['long']:3d}%  short={p['short']:3d}%  [{bias}]")

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
    print(f"OK  {len(pairs)} pairs  →  {out_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
