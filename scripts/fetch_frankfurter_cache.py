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

    # 4) Previous EUR base — for ECB panel prev column
    prev_eur_data = fetch_json(f"{BASE_URL}/{prev_date}?from=EUR&to={ECB_CURRENCIES}")
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"  ❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
