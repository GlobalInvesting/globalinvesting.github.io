"""
backfill_fx_history.py
───────────────────────
Descarga tasas FX históricas semanales (lunes) desde Frankfurter/ECB
para el período 2024-01-01 → hoy.

Esto provee los datos de OUTCOME para el backtest retrospectivo:
cada snapshot de scores necesita una tasa FX de entrada y una de salida
1-6 semanas después.

Ejecutar UNA SOLA VEZ manualmente:
  python3 scripts/backfill_fx_history.py

Frankfurter tiene datos históricos completos desde 1999. Límite: ~1 req/seg.
"""

import json
import os
import time
import requests
from datetime import date, datetime, timedelta

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; ForexDashboard/3.0)',
    'Accept': 'application/json',
}

FX_OUT_DIR = "fx-history"
BASE_CURRENCY = "USD"
CURRENCIES = ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

START_DATE = date(2024, 1, 1)
END_DATE   = date.today()

os.makedirs(FX_OUT_DIR, exist_ok=True)


def fetch_rates(target_date, retries=3, backoff=5):
    """Fetch rates for a specific date from Frankfurter API. Retries on failure."""
    url = f"https://api.frankfurter.app/{target_date.isoformat()}?base={BASE_CURRENCY}"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            d = r.json()
            rates = d.get("rates", {})
            rates["USD"] = 1.0
            return rates, d.get("date", target_date.isoformat())
        except Exception as e:
            if attempt < retries:
                print(f"  RETRY {attempt}/{retries} for {target_date}: {e}")
                time.sleep(backoff * attempt)
            else:
                print(f"  ERROR {target_date} after {retries} attempts: {e}")
                return None, None


def get_mondays(start, end):
    """Return all Mondays between start and end dates."""
    mondays = []
    current = start
    # Advance to first Monday
    while current.weekday() != 0:  # 0 = Monday
        current += timedelta(days=1)
    while current <= end:
        mondays.append(current)
        current += timedelta(weeks=1)
    return mondays


def main():
    mondays = get_mondays(START_DATE, END_DATE)
    print(f"Backfilling FX history: {len(mondays)} Mondays from {START_DATE} to {END_DATE}")
    print(f"Output directory: {FX_OUT_DIR}/\n")

    skipped = 0
    fetched = 0
    errors  = 0

    for i, monday in enumerate(mondays):
        path = f"{FX_OUT_DIR}/{monday.isoformat()}.json"

        if os.path.exists(path):
            skipped += 1
            continue

        rates, actual_date = fetch_rates(monday)

        if rates:
            snapshot = {
                "date":        actual_date or monday.isoformat(),
                "rates_vs_usd": rates,
                "source":      "Frankfurter/ECB",
                "fetched_for": monday.isoformat(),
            }
            with open(path, "w") as f:
                json.dump(snapshot, f, indent=2)
            fetched += 1
            if fetched % 10 == 0:
                print(f"  [{i+1}/{len(mondays)}] {monday} → saved ({fetched} fetched, {skipped} skipped, {errors} errors)")
        else:
            errors += 1

        # Rate limit: ~1 request per second
        time.sleep(1.1)

    print(f"\n✅ Done: {fetched} new files | {skipped} already existed | {errors} errors")
    print(f"   Files in {FX_OUT_DIR}/: {len(os.listdir(FX_OUT_DIR))}")


if __name__ == "__main__":
    main()
