#!/usr/bin/env python3
"""
fetch_frankfurter_cache.py
Fetches ECB/Frankfurter FX rates server-side (no CORS) and writes
site/fx-data/frankfurter.json for consumption by dashboard.js.

Output structure:
{
  "updated": "2026-04-05T08:00:00Z",   # ISO timestamp of this fetch
  "today": {                            # Latest available business day
    "date": "2026-04-04",
    "rates": { "EUR": 0.9182, "GBP": 0.7710, "JPY": 150.32, ... }  # USD base
  },
  "prev": {                             # Previous business day
    "date": "2026-04-03",
    "rates": { ... }
  },
  "series": {                           # Last 7 calendar days timeseries (EUR base: USD, GBP, JPY)
    "date_from": "2026-03-29",
    "date_to":   "2026-04-04",
    "rates": {
      "2026-03-31": { "USD": 1.0852, "GBP": 0.8393, "JPY": 163.81 },
      ...
    }
  }
}
"""

import json
import os
import sys
import requests
from datetime import datetime, timedelta, timezone


BASE_URL = "https://api.frankfurter.app"
OUTPUT_PATH = os.path.join("fx-data", "frankfurter.json")

# Currencies to fetch (USD base — mirrors what dashboard.js needs for FX table / STATE.rates)
ALL_CURRENCIES = "EUR,GBP,JPY,AUD,CAD,CHF,NZD,HKD,SEK,NOK,DKK,SGD,MXN,ZAR,PLN,HUF,CZK,TRY"
# EUR-base currencies for the ECB Reference Rates panel (USD must appear as a key)
ECB_CURRENCIES = "USD,GBP,JPY,AUD,CAD,CHF,NZD"
# Liquidity series currencies (EUR base — mirrors fetchLiquidityData)
SERIES_CURRENCIES = "USD,GBP,JPY"


def get_business_dates(n=2):
    """Return last n business dates (Mon-Fri) counting back from today."""
    dates = []
    d = datetime.now(timezone.utc).date()
    while len(dates) < n:
        if d.weekday() < 5:  # 0=Mon, 4=Fri
            dates.append(d.isoformat())
        d -= timedelta(days=1)
    return dates  # [today_or_latest_biz, prev_biz]


def fetch_json(url, timeout=15):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main():
    today_date, prev_date = get_business_dates(n=2)

    # Timeseries: 7 calendar days back (ensures ≥ 5 trading days)
    today_dt = datetime.strptime(today_date, "%Y-%m-%d")
    series_start = (today_dt - timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"  Fetching today={today_date}, prev={prev_date}, series={series_start}..{today_date}")

    # 1) Today rates (USD base → all major currencies) — used by STATE.rates / FX table
    today_data = fetch_json(f"{BASE_URL}/{today_date}?from=USD&to={ALL_CURRENCIES}")
    print(f"  ✓ today (USD base): {today_data['date']} — {len(today_data['rates'])} currencies")

    # 2) Previous business day (USD base)
    prev_data = fetch_json(f"{BASE_URL}/{prev_date}?from=USD&to={ALL_CURRENCIES}")
    print(f"  ✓ prev  (USD base): {prev_data['date']} — {len(prev_data['rates'])} currencies")

    # 3) Today EUR base — for ECB Reference Rates panel (USD must appear as a key)
    today_eur_data = fetch_json(f"{BASE_URL}/{today_date}?from=EUR&to={ECB_CURRENCIES}")
    print(f"  ✓ today (EUR base): {today_eur_data['date']} — {len(today_eur_data['rates'])} currencies")

    # 4) Previous EUR base — use the business day before the date the API ACTUALLY returned
    #    (not prev_date from the calendar). This handles the case where the ECB hasn't published
    #    today's fixing yet: the API returns yesterday's date for "today", so using calendar
    #    prev_date would produce the same date → Chg = 0.
    actual_today_eur = today_eur_data["date"]
    actual_today_dt = datetime.strptime(actual_today_eur, "%Y-%m-%d")
    # Step back one business day from the API-returned date
    prev_eur_dt = actual_today_dt - timedelta(days=1)
    while prev_eur_dt.weekday() >= 5:
        prev_eur_dt -= timedelta(days=1)
    prev_eur_date = prev_eur_dt.strftime("%Y-%m-%d")
    prev_eur_data = fetch_json(f"{BASE_URL}/{prev_eur_date}?from=EUR&to={ECB_CURRENCIES}")
    print(f"  ✓ prev  (EUR base): {prev_eur_data['date']} — {len(prev_eur_data['rates'])} currencies")

    # 5) Timeseries: EUR base → USD, GBP, JPY (for liquidity canvas vol scalar)
    series_data = fetch_json(
        f"{BASE_URL}/{series_start}..{today_date}?from=EUR&to={SERIES_CURRENCIES}"
    )
    print(f"  ✓ series: {len(series_data.get('rates', {}))} days ({series_start}..{today_date})")

    output = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # USD-base: used by STATE.rates / FX table / computeRate()
        "today": {
            "date": today_data["date"],
            "rates": today_data["rates"],
        },
        "prev": {
            "date": prev_data["date"],
            "rates": prev_data["rates"],
        },
        # EUR-base: used exclusively by the ECB Reference Rates panel
        # rates keys are USD, GBP, JPY, AUD, CAD, CHF, NZD
        # e.g. {"USD": 1.1715, "GBP": 0.8671, "JPY": 184.83, ...}
        "today_eur": {
            "date": today_eur_data["date"],
            "rates": today_eur_data["rates"],
        },
        "prev_eur": {
            "date": prev_eur_data["date"],
            "rates": prev_eur_data["rates"],
        },
        "series": {
            "date_from": series_start,
            "date_to": today_date,
            "rates": series_data.get("rates", {}),
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    print(f"  ✓ Written: {OUTPUT_PATH}")


def _load_existing_cache() -> dict | None:
    """Return existing frankfurter.json if it is valid and recent enough to use as fallback."""
    if not os.path.exists(OUTPUT_PATH):
        return None
    try:
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        # Must have a today block with at least one rate
        if data.get("today", {}).get("rates"):
            return data
    except Exception:
        pass
    return None


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"  ❌ Error: {e}", file=sys.stderr)
        # v8.4.5: Fallback to existing cache instead of exiting with code 1.
        # A 520/503/timeout from api.frankfurter.app is transient (Cloudflare hiccup,
        # ECB publishing delay). Failing the GitHub Actions step is disruptive and
        # misleading — the cached data from the previous run is valid for the dashboard.
        # If no prior cache exists, exit 1 so the problem is visible.
        existing = _load_existing_cache()
        if existing:
            # Bump the updated timestamp so downstream consumers know we checked.
            existing["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(existing, f, separators=(",", ":"))
            print(f"  ⚠️  API unavailable — preserved previous cache "
                  f"(today={existing['today']['date']}). Will retry on next scheduled run.",
                  file=sys.stderr)
            sys.exit(0)   # ← exit 0: step is yellow warning, not red failure
        else:
            print("  ❌ No prior cache available — cannot recover.", file=sys.stderr)
            sys.exit(1)
