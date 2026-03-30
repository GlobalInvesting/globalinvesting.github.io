#!/usr/bin/env python3
"""
fetch_myfxbook_sentiment.py  v1.0 — Myfxbook Community Outlook via REST API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce:  sentiment-data/myfxbook.json
Schedule: Cada 30 min en días hábiles (via GitHub Action en repo público)

FLUJO:
  1. POST /api/login.json            → obtiene session token
  2. GET  /api/get-community-outlook.json  → descarga posicionamiento retail
  3. Normaliza a formato {sym, longPercent, shortPercent} para el terminal
  4. GET  /api/logout.json           → cierra la sesión (buenas prácticas)

CREDENCIALES:
  Leer de variables de entorno (GitHub Secrets en el Action):
    MYFXBOOK_EMAIL    — email de la cuenta Myfxbook
    MYFXBOOK_PASSWORD — contraseña de la cuenta Myfxbook

  ⚠️  No hardcodear credenciales. El script falla de forma limpia si no están.

NOTAS:
  • La API de Myfxbook no tiene límite de calls documentado para el plan free.
    Con 30 runs/día estamos muy por debajo de cualquier límite razonable.
  • El session token expira. Hacemos login en cada run para evitar tokens stale.
  • Este script NO requiere API keys de pago — solo una cuenta Myfxbook gratuita.
  • Vive en el repo PÚBLICO porque no contiene ventaja competitiva y aprovecha
    los minutos ilimitados de GitHub Actions del repo público.

FORMATO DE SALIDA (sentiment-data/myfxbook.json):
  {
    "updated": "2026-03-30T20:00:00Z",
    "source": "myfxbook",
    "pairs": [
      { "sym": "EUR/USD", "long": 56, "short": 44, "longVol": 1234567, "shortVol": 987654 },
      ...
    ]
  }
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import json
import time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("[ERROR] requests no instalado. Correr: pip install requests")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────────

BASE_URL = "https://www.myfxbook.com/api"

MYFXBOOK_EMAIL    = os.environ.get("MYFXBOOK_EMAIL", "")
MYFXBOOK_PASSWORD = os.environ.get("MYFXBOOK_PASSWORD", "")

# Symbol IDs to fetch — matches the current widget embed
# Format: myfxbook internal ID → display symbol
SYMBOL_MAP = {
    1:   "EUR/USD",
    2:   "GBP/USD",
    3:   "USD/JPY",
    4:   "USD/CHF",
    5:   "EUR/CHF",
    6:   "EUR/GBP",
    7:   "USD/CAD",
    9:   "EUR/JPY",
    10:  "EUR/CAD",
    11:  "AUD/USD",
    12:  "AUD/JPY",
    13:  "GBP/JPY",
    14:  "CHF/JPY",
    17:  "EUR/AUD",
    20:  "NZD/USD",
    24:  "GBP/CHF",
    25:  "EUR/NZD",
    26:  "AUD/CAD",
    27:  "GBP/CAD",
    28:  "AUD/CHF",
    29:  "GBP/AUD",
    46:  "AUD/NZD",
    47:  "CAD/JPY",
    49:  "GBP/NZD",
}

# Priority order for the terminal panel (top 10 shown by default)
PRIORITY_IDS = [1, 2, 3, 11, 7, 4, 20, 6, 9, 13]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.myfxbook.com/",
}

TIMEOUT = 15  # seconds

# ── Session (persiste cookies entre requests — requerido por Myfxbook) ────────

_session = requests.Session()
_session.headers.update(HEADERS)


def api_get(path, params=None, raw_params=None):
    url = f"{BASE_URL}/{path}"
    if raw_params:
        # Construir query string manualmente para evitar double-encoding
        from urllib.parse import urlencode
        qs = urlencode(raw_params, quote_via=lambda s, *a, **k: s)
        url = f"{url}?{qs}"
        r = _session.get(url, timeout=TIMEOUT)
    else:
        r = _session.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    site_path = os.environ.get("SITE_PATH", ".")
    out_dir   = os.path.join(site_path, "sentiment-data")
    out_file  = os.path.join(out_dir, "myfxbook.json")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*60}")
    print(f"fetch_myfxbook_sentiment.py  v1.0  —  {ts}")
    print(f"{'='*60}\n")

    # Validate credentials
    if not MYFXBOOK_EMAIL or not MYFXBOOK_PASSWORD:
        print("[ERROR] MYFXBOOK_EMAIL and MYFXBOOK_PASSWORD environment variables required.")
        print("        Set them as GitHub Secrets: MYFXBOOK_EMAIL, MYFXBOOK_PASSWORD")
        sys.exit(1)

    session_token = None

    try:
        # STEP 1 — Login
        print(f"[Auth] Logging in as {MYFXBOOK_EMAIL}...")
        login_resp = api_get("login.json", params={
            "email":    MYFXBOOK_EMAIL,
            "password": MYFXBOOK_PASSWORD,
        })

        if login_resp.get("error"):
            print(f"[Auth] Login failed: {login_resp.get('message', 'unknown error')}")
            sys.exit(1)

        session_token = login_resp.get("session")
        if not session_token:
            print("[Auth] No session token returned")
            sys.exit(1)

        # Decodificar el token si viene URL-encoded (ej: %2F → /)
        from urllib.parse import unquote
        session_token = unquote(session_token)

        print(f"[Auth] Login OK. Session: {session_token[:8]}...")
        time.sleep(1)  # Myfxbook necesita un momento para registrar la sesión

        # STEP 2 — Fetch community outlook
        print("[API]  Fetching community outlook...")
        outlook_resp = api_get("get-community-outlook.json", raw_params={
            "session": session_token,
            "email":   MYFXBOOK_EMAIL,
        })

        if outlook_resp.get("error"):
            print(f"[API]  Error: {outlook_resp.get('message', 'unknown')}")
            sys.exit(1)

        symbols_data = outlook_resp.get("symbols", {})
        if not symbols_data:
            print("[API]  No symbols data in response")
            sys.exit(1)

        # STEP 3 — Normalize
        # The API returns data keyed by symbol name, e.g.:
        # { "EURUSD": { "name": "EUR/USD", "shortPercentage": 44.2, "longPercentage": 55.8,
        #               "shortVolume": 987654, "longVolume": 1234567, ... } }
        pairs = []

        # Build lookup from symbol name → data
        sym_lookup = {}
        for key, val in symbols_data.items():
            name = val.get("name", key)
            # Normalize: "EURUSD" → "EUR/USD"
            display = name if "/" in name else key
            sym_lookup[display.replace("_", "/")] = val

        # Also try numeric ID lookup from our SYMBOL_MAP
        for sym_id in PRIORITY_IDS + [k for k in SYMBOL_MAP if k not in PRIORITY_IDS]:
            display = SYMBOL_MAP.get(sym_id)
            if not display:
                continue

            # Try to find by display name
            raw = sym_lookup.get(display)
            if raw is None:
                # Try without slash
                raw = sym_lookup.get(display.replace("/", ""))
            if raw is None:
                continue

            long_pct  = raw.get("longPercentage")  or raw.get("longPercent")  or 0
            short_pct = raw.get("shortPercentage") or raw.get("shortPercent") or 0
            long_vol  = raw.get("longVolume",  0)
            short_vol = raw.get("shortVolume", 0)

            # Normalize to integers
            long_pct  = round(float(long_pct))
            short_pct = round(float(short_pct))

            # Ensure they sum to 100
            if long_pct + short_pct != 100:
                total = long_pct + short_pct
                if total > 0:
                    long_pct  = round(long_pct  / total * 100)
                    short_pct = 100 - long_pct

            pairs.append({
                "sym":      display,
                "long":     long_pct,
                "short":    short_pct,
                "longVol":  int(long_vol),
                "shortVol": int(short_vol),
            })

            bias = "LONG" if long_pct >= short_pct else "SHORT"
            print(f"  {display:10s}  long={long_pct:3d}%  short={short_pct:3d}%  [{bias}]")

        if not pairs:
            print("[ERROR] Could not normalize any pairs from API response")
            sys.exit(1)

        # STEP 4 — Logout (best practice)
        try:
            api_get("logout.json", raw_params={"session": session_token})
            print("\n[Auth] Logged out OK")
        except Exception as e:
            print(f"\n[Auth] Logout warning (non-fatal): {e}")

        # STEP 5 — Write output
        output = {
            "updated": ts,
            "source":  "myfxbook",
            "pairs":   pairs,
        }

        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✅ {len(pairs)} pairs → {out_file}")
        print(f"{'='*60}\n")

    except requests.RequestException as e:
        print(f"\n[ERROR] Network error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
