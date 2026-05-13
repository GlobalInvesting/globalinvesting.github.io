#!/usr/bin/env python3
"""
fetch_dtcc_fx.py
Fetches FX OTC transaction data from DTCC Global Trade Repository (GTR)
public dissemination API and writes dtcc-data/dtcc_fx.json.

Source: DTCC GTR CFTC Recast public dissemination
URL:    https://pddata.dtcc.com/gtr/api/fx/current
Delay:  T+1 (next business day after trade date)
No API key required — public under Dodd-Frank Section 2(a)(13).

Output schema:
{
  "fetched":    "2026-05-13",
  "trade_date": "2026-05-12",
  "source":     "DTCC GTR · CFTC Recast · public dissemination · T+1",
  "note":       "Notional capped at $250M per trade per CFTC rules. Represents a subset of total OTC FX volume.",
  "pairs": {
    "EUR/USD": {
      "notional_usd_bn": 42.3,
      "trade_count":     1821,
      "by_product": {
        "FxSwap":    { "notional_usd_bn": 28.1, "count": 1102 },
        "FxForward": { "notional_usd_bn": 12.4, "count":  634 },
        "FxSpot":    { "notional_usd_bn":  1.8, "count":   85 }
      }
    },
    ...
  },
  "totals": {
    "notional_usd_bn": 312.4,
    "trade_count":     14820,
    "by_product": { ... }
  }
}
"""

import json
import os
import sys
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict

API_URL    = "https://pddata.dtcc.com/gtr/api/fx/current"
OUTPUT_DIR = "dtcc-data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dtcc_fx.json")

# G8 pairs to track — canonical form as DTCC publishes them
G8_PAIRS = {
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CHF", "USD/CAD", "NZD/USD",
    # Common crosses
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "EUR/CHF",
    "EUR/AUD", "AUD/JPY", "USD/MXN", "USD/CNY",
}

# DTCC sometimes publishes the inverse (e.g. JPY/USD instead of USD/JPY)
INVERSE_MAP = {v: k for k, v in {
    "USD/JPY": "JPY/USD",
    "USD/CHF": "CHF/USD",
    "USD/CAD": "CAD/USD",
    "USD/MXN": "MXN/USD",
    "USD/CNY": "CNY/USD",
}.items()}

# Product type normalisation from DTCC taxonomy
PRODUCT_MAP = {
    "FX:FxSwap":    "FxSwap",
    "FX:FxForward": "FxForward",
    "FX:FxSpot":    "FxSpot",
    "FX:FxOption":  "FxOption",
    "FX:FxNDF":     "FxNDF",
}


def normalise_pair(raw: str) -> str | None:
    """Normalise pair to canonical form. Returns None if not a G8 pair."""
    p = raw.strip().upper().replace(" ", "")
    if p in G8_PAIRS:
        return p
    if p in INVERSE_MAP:
        return INVERSE_MAP[p]
    return None


def notional_value(record: dict) -> float:
    """
    Extract USD-equivalent notional from a DTCC record.
    CFTC rules cap individual trade notional at $250M for public dissemination;
    trades above this threshold are published as ROUNDED_NOTIONAL_AMOUNT_1.
    We use whichever field is populated, converting to billions.
    """
    # Try exact notional first, then rounded (large trades)
    for field in ("NOTIONAL_AMOUNT_1", "ROUNDED_NOTIONAL_AMOUNT_1"):
        v = record.get(field)
        if v is not None and v != "" and v != "null":
            try:
                return float(str(v).replace(",", "")) / 1e9
            except (ValueError, TypeError):
                pass
    return 0.0


def fetch_with_retry(url: str, retries: int = 3, timeout: int = 30) -> list:
    """Fetch JSON array from URL with retries."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, headers={
                "Accept": "application/json",
                "User-Agent": "globalinvesting-fx-terminal/1.0",
            })
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            # Some endpoints wrap in {"records": [...]}
            if isinstance(data, dict):
                for key in ("records", "data", "items", "trades"):
                    if isinstance(data.get(key), list):
                        return data[key]
            print(f"  Unexpected response shape: {str(data)[:200]}", file=sys.stderr)
            return []
        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt}/{retries}", file=sys.stderr)
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP {e.response.status_code} on attempt {attempt}/{retries}", file=sys.stderr)
            if e.response.status_code in (400, 404):
                break  # no point retrying
        except Exception as e:
            print(f"  Error on attempt {attempt}/{retries}: {e}", file=sys.stderr)
        if attempt < retries:
            import time
            time.sleep(2 ** attempt)
    return []


def aggregate(records: list) -> dict:
    """
    Aggregate raw DTCC records into per-pair notional volumes.
    Only counts ACTION=NEWT (new trades) to avoid double-counting
    amendments and terminations.
    """
    pairs: dict[str, dict] = defaultdict(lambda: {
        "notional_usd_bn": 0.0,
        "trade_count": 0,
        "by_product": defaultdict(lambda: {"notional_usd_bn": 0.0, "count": 0}),
    })
    totals = {
        "notional_usd_bn": 0.0,
        "trade_count": 0,
        "by_product": defaultdict(lambda: {"notional_usd_bn": 0.0, "count": 0}),
    }
    trade_dates: set[str] = set()

    for rec in records:
        # Only count new trades (NEWT), skip amendments/terminations
        action = str(rec.get("ACTION", "")).strip().upper()
        if action not in ("NEWT", "NEW", ""):
            continue

        raw_pair = str(rec.get("UNDERLYING_ASSET_1", "")).strip()
        pair = normalise_pair(raw_pair)
        if not pair:
            continue

        notional = notional_value(rec)
        if notional <= 0:
            continue

        raw_product = str(rec.get("SUB_ASSET_CLASS_FOR_OTHER_COMMODITY", "")).strip()
        product = PRODUCT_MAP.get(raw_product, raw_product.split(":")[-1] if ":" in raw_product else raw_product or "FxSwap")

        trade_date = str(rec.get("TRADE_DATE", rec.get("EXECUTION_TIMESTAMP", ""))[:10])
        if trade_date:
            trade_dates.add(trade_date)

        pairs[pair]["notional_usd_bn"] += notional
        pairs[pair]["trade_count"] += 1
        pairs[pair]["by_product"][product]["notional_usd_bn"] += notional
        pairs[pair]["by_product"][product]["count"] += 1

        totals["notional_usd_bn"] += notional
        totals["trade_count"] += 1
        totals["by_product"][product]["notional_usd_bn"] += notional
        totals["by_product"][product]["count"] += 1

    # Round and convert defaultdicts to regular dicts
    def clean_entry(entry: dict) -> dict:
        return {
            "notional_usd_bn": round(entry["notional_usd_bn"], 2),
            "trade_count": entry["trade_count"],
            "by_product": {
                k: {
                    "notional_usd_bn": round(v["notional_usd_bn"], 2),
                    "count": v["count"],
                }
                for k, v in sorted(
                    entry["by_product"].items(),
                    key=lambda x: x[1]["notional_usd_bn"],
                    reverse=True,
                )
            },
        }

    pairs_clean = {
        pair: clean_entry(data)
        for pair, data in sorted(
            pairs.items(),
            key=lambda x: x[1]["notional_usd_bn"],
            reverse=True,
        )
    }

    trade_date_str = sorted(trade_dates)[-1] if trade_dates else ""

    return pairs_clean, clean_entry(totals), trade_date_str


def main():
    print(f"  Fetching DTCC GTR FX public dissemination...")
    records = fetch_with_retry(API_URL)

    if not records:
        print("  ❌ No records returned — writing empty fallback", file=sys.stderr)
        # Write a graceful empty file so the frontend can show "data pending"
        output = {
            "fetched": datetime.now(timezone.utc).date().isoformat(),
            "trade_date": "",
            "source": "DTCC GTR · CFTC Recast · public dissemination · T+1",
            "note": "Notional capped at $250M per trade per CFTC rules. Represents a subset of total OTC FX volume.",
            "pairs": {},
            "totals": {"notional_usd_bn": 0.0, "trade_count": 0, "by_product": {}},
            "status": "unavailable",
        }
    else:
        print(f"  ✓ Fetched {len(records):,} raw records")
        pairs, totals, trade_date = aggregate(records)
        print(f"  ✓ Aggregated: {len(pairs)} G8 pairs, trade_date={trade_date}")
        print(f"  ✓ Total notional: ${totals['notional_usd_bn']:.1f}bn across {totals['trade_count']:,} new trades")
        for pair, data in list(pairs.items())[:5]:
            print(f"    {pair}: ${data['notional_usd_bn']:.2f}bn ({data['trade_count']} trades)")

        output = {
            "fetched":    datetime.now(timezone.utc).date().isoformat(),
            "trade_date": trade_date,
            "source":     "DTCC GTR · CFTC Recast · public dissemination · T+1",
            "note":       "Notional capped at $250M per trade per CFTC rules. Represents a subset of total OTC FX volume.",
            "pairs":      pairs,
            "totals":     totals,
            "status":     "ok",
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    print(f"  ✓ Written: {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"  ❌ Fatal: {e}", file=sys.stderr)
        sys.exit(1)
