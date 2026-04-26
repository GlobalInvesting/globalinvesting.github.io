#!/usr/bin/env python3
"""
fetch_myfxbook_sentiment.py  v6.0 — single-call CF Worker proxy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce:  sentiment-data/myfxbook.json
Schedule: Each hour on business days (via GitHub Action in public repo)

WHAT CHANGED IN v6.0 (vs v5.0):
  v5.0 called the CF Worker three times: login → outlook → logout.
  Myfxbook sessions are IP-bound — that architecture was correct in
  principle (all three calls went through the same CF edge IP), but
  Myfxbook's anti-bot layer detected the automated multi-call pattern
  and responded with "Wrong email/password" — a generic block that
  makes credentials look wrong when they are not.

  v6.0 sends a single POST to the Worker with the credentials.
  The Worker handles login + outlook + logout internally and returns
  only the normalized pair data. The session token never leaves
  Cloudflare's network. One request = one browser-like session.

  The CF Worker must be updated to v2.0 to match this interface.

CREDENTIALS (GitHub Secrets — set in repo Settings → Secrets → Actions):
  MYFXBOOK_EMAIL    — Myfxbook account email
  MYFXBOOK_PASSWORD — Myfxbook account password
  CF_WORKER_URL     — Full URL of deployed Cloudflare Worker
                      e.g. https://myfxbook-proxy.YOUR_SUBDOMAIN.workers.dev
  CF_WORKER_SECRET  — Shared secret (must match WORKER_SECRET env var in CF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sys, json, time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("[ERROR] requests not installed. Run: pip install requests")
    sys.exit(1)

MYFXBOOK_EMAIL    = os.environ.get("MYFXBOOK_EMAIL",    "")
MYFXBOOK_PASSWORD = os.environ.get("MYFXBOOK_PASSWORD", "")
CF_WORKER_URL     = os.environ.get("CF_WORKER_URL",     "").rstrip("/")
CF_WORKER_SECRET  = os.environ.get("CF_WORKER_SECRET",  "")

TIMEOUT     = 30   # seconds — Worker needs time to complete its internal flow
MAX_RETRIES = 3
RETRY_DELAY = 5    # seconds (doubles on each retry)


# ── Helpers ────────────────────────────────────────────────────────────────

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
    """
    POST credentials to the CF Worker.
    The Worker handles login + outlook + logout and returns normalized data.
    Returns parsed JSON on success; calls preserve_existing() on any failure.
    """
    url     = CF_WORKER_URL
    headers = {
        "X-Worker-Secret": CF_WORKER_SECRET,
        "Content-Type":    "application/json",
    }
    payload = {"email": MYFXBOOK_EMAIL, "password": MYFXBOOK_PASSWORD}

    for attempt in range(MAX_RETRIES):
        delay = RETRY_DELAY * (2 ** attempt)
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        except requests.exceptions.Timeout:
            print(f"[Worker] Timeout on attempt {attempt + 1}. Waiting {delay}s...")
            time.sleep(delay)
            continue
        except requests.exceptions.ConnectionError as e:
            print(f"[Worker] Connection error on attempt {attempt + 1}: {e}. Waiting {delay}s...")
            time.sleep(delay)
            continue

        # Auth / config errors returned as 4xx/5xx — not retryable
        if r.status_code == 401:
            preserve_existing(out_file, ts,
                "Worker returned 401 — check CF_WORKER_SECRET matches Worker WORKER_SECRET env var.")
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

        # Worker always returns { ok: true|false }
        if not data.get("ok"):
            err = data.get("error", "unknown worker error")
            print(f"[Worker] Worker reported failure: {err}")
            preserve_existing(out_file, ts, err)

        return data   # { ok: true, pairs: [...], general: {...}|null }

    preserve_existing(out_file, ts, f"Worker call failed after {MAX_RETRIES} attempts.")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    site_path = os.environ.get("SITE_PATH", ".")
    out_dir   = os.path.join(site_path, "sentiment-data")
    out_file  = os.path.join(out_dir, "myfxbook.json")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*60}")
    print(f"fetch_myfxbook_sentiment.py  v6.0  --  {ts}")
    print(f"{'='*60}\n")

    # Validate required env vars
    missing = [v for v in ["MYFXBOOK_EMAIL", "MYFXBOOK_PASSWORD", "CF_WORKER_URL", "CF_WORKER_SECRET"]
               if not os.environ.get(v)]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        print(f"        Set them as GitHub Secrets in repo Settings → Secrets → Actions.")
        sys.exit(1)

    print(f"[Config] Worker:  {CF_WORKER_URL}")
    print(f"[Config] Account: {MYFXBOOK_EMAIL}")
    print(f"\n[Worker] Sending single POST — Worker will handle login + outlook + logout...")

    data  = call_worker(out_file, ts)
    pairs   = data.get("pairs",   [])
    general = data.get("general", None)

    if not pairs:
        preserve_existing(out_file, ts, "Worker returned ok=true but empty pairs list.")

    print(f"[Worker] Got {len(pairs)} pairs:")
    for p in pairs:
        bias = "LONG " if p["long"] >= p["short"] else "SHORT"
        print(f"  {p['sym']:10s}  long={p['long']:3d}%  short={p['short']:3d}%  [{bias}]")

    # Write JSON
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
