#!/usr/bin/env python3
"""
fetch_dtcc_fx.py
Fetches FX OTC transaction data from the DTCC GTR CFTC Cumulative Slice Reports
(S3 public bucket) and writes dtcc-data/dtcc_fx.json.

Source:  DTCC GTR — CFTC Cumulative Slice Reports (Phase 2, December 2023+)
S3 URL:  https://kgc0418-tdw-data-0.s3.amazonaws.com/cftc/eod/CFTC_CUMULATIVE_FOREX_{YYYY}_{MM}_{DD}.zip
Delay:   T+1 (cumulative for prior business day)
No API key required — public under Dodd-Frank Section 2(a)(13).

The script tries yesterday first, then falls back up to 5 prior business days.
"""

import csv
import io
import json
import os
import sys
import zipfile
from datetime import date, datetime, timezone, timedelta
from collections import defaultdict

import requests

S3_BASE     = "https://kgc0418-tdw-data-0.s3.amazonaws.com/cftc/eod"
OUTPUT_DIR  = "dtcc-data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dtcc_fx.json")

G8_PAIRS = {
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CHF", "USD/CAD", "NZD/USD",
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "EUR/CHF",
    "EUR/AUD", "AUD/JPY", "USD/MXN", "USD/CNY",
}

INVERSE_MAP = {
    "JPY/USD": "USD/JPY", "CHF/USD": "USD/CHF", "CAD/USD": "USD/CAD",
    "MXN/USD": "USD/MXN", "CNY/USD": "USD/CNY",
    "JPY/EUR": "EUR/JPY", "GBP/EUR": "EUR/GBP", "CHF/EUR": "EUR/CHF",
    "AUD/EUR": "EUR/AUD", "JPY/GBP": "GBP/JPY", "JPY/AUD": "AUD/JPY",
}

PRODUCT_NORM = {
    "foreignexchange:fxswap":    "FxSwap",
    "foreignexchange:fxforward": "FxForward",
    "foreignexchange:fxspot":    "FxSpot",
    "foreignexchange:fxoption":  "FxOption",
    "foreignexchange:fxndf":     "FxNDF",
    "fx:fxswap":    "FxSwap",
    "fx:fxforward": "FxForward",
    "fx:fxspot":    "FxSpot",
    "fx:fxoption":  "FxOption",
    "fx:fxndf":     "FxNDF",
    "fxswap":       "FxSwap",
    "fxforward":    "FxForward",
    "fxspot":       "FxSpot",
}


def s3_url(d: date) -> str:
    return f"{S3_BASE}/CFTC_CUMULATIVE_FOREX_{d.year}_{d.month:02d}_{d.day:02d}.zip"


def prev_business_day(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def candidate_dates(today: date) -> list:
    yesterday = prev_business_day(today)
    candidates = [yesterday, today]
    d = prev_business_day(yesterday)
    for _ in range(5):
        candidates.append(d)
        d = prev_business_day(d)
    return candidates


def fetch_zip(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=60, headers={
            "User-Agent": "globalinvesting-fx-terminal/1.0",
        })
        if resp.status_code == 200 and len(resp.content) > 500:
            return resp.content
        if resp.status_code == 404:
            return None
        print(f"  HTTP {resp.status_code} for {url}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Error fetching {url}: {e}", file=sys.stderr)
        return None


def parse_csv_from_zip(zip_bytes: bytes) -> tuple:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        csv_name = next((n for n in names if n.lower().endswith('.csv')), None)
        if not csv_name:
            print(f"  No CSV in ZIP. Contents: {names}", file=sys.stderr)
            return [], ""
        with zf.open(csv_name) as f:
            text = f.read().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader), csv_name


def find_field(record: dict, *candidates: str) -> str:
    lower_map = {k.lower().strip(): v for k, v in record.items()}
    for c in candidates:
        v = lower_map.get(c.lower().strip())
        if v is not None:
            return str(v)
    return ""


def notional_value(record: dict) -> float:
    for field in (
        "notional amount-leg 1", "notional amount leg 1",
        "notional_amount_1",
        "rounded notional amount-leg 1", "rounded_notional_amount_1",
    ):
        v = find_field(record, field)
        if v and v not in ("", "null", "None", "N/A"):
            try:
                return float(str(v).replace(",", "")) / 1e9
            except (ValueError, TypeError):
                pass
    return 0.0


def normalise_pair(raw: str) -> str:
    p = raw.strip().upper().replace(" ", "").replace("-", "/")
    if "/" not in p and len(p) == 6:
        p = p[:3] + "/" + p[3:]
    if p in G8_PAIRS:
        return p
    if p in INVERSE_MAP:
        return INVERSE_MAP[p]
    return None


def aggregate(records: list) -> tuple:
    pairs = defaultdict(lambda: {
        "notional_usd_bn": 0.0, "trade_count": 0,
        "by_product": defaultdict(lambda: {"notional_usd_bn": 0.0, "count": 0}),
    })
    totals = {
        "notional_usd_bn": 0.0, "trade_count": 0,
        "by_product": defaultdict(lambda: {"notional_usd_bn": 0.0, "count": 0}),
    }
    trade_dates = set()

    for rec in records:
        action = find_field(rec, "action type", "action").strip().upper()
        if action and action not in ("NEWT", "NEW", ""):
            continue

        raw_pair = find_field(rec, "underlying asset name", "underlying_asset_1",
                               "underlier id-leg 1", "underlier id leg 1",
                               "strike price currency/currency pair").strip()
        pair = normalise_pair(raw_pair)
        if not pair:
            continue

        notional = notional_value(rec)
        if notional <= 0:
            continue

        raw_product = find_field(rec, "product name",
                                  "sub_asset_class_for_other_commodity").strip()
        key = raw_product.lower().replace(" ", "").replace("_", "")
        product = PRODUCT_NORM.get(key)
        if not product:
            tail = raw_product.split(":")[-1].strip().lower() if ":" in raw_product else key
            product = PRODUCT_NORM.get(tail, tail.capitalize() or "FxSwap")

        td = find_field(rec, "execution timestamp", "event timestamp", "effective date")[:10]
        if td and len(td) == 10:
            trade_dates.add(td)

        pairs[pair]["notional_usd_bn"] += notional
        pairs[pair]["trade_count"] += 1
        pairs[pair]["by_product"][product]["notional_usd_bn"] += notional
        pairs[pair]["by_product"][product]["count"] += 1
        totals["notional_usd_bn"] += notional
        totals["trade_count"] += 1
        totals["by_product"][product]["notional_usd_bn"] += notional
        totals["by_product"][product]["count"] += 1

    def clean(entry):
        return {
            "notional_usd_bn": round(entry["notional_usd_bn"], 2),
            "trade_count": entry["trade_count"],
            "by_product": {
                k: {"notional_usd_bn": round(v["notional_usd_bn"], 2), "count": v["count"]}
                for k, v in sorted(entry["by_product"].items(),
                                   key=lambda x: x[1]["notional_usd_bn"], reverse=True)
            },
        }

    pairs_clean = {
        pair: clean(data)
        for pair, data in sorted(pairs.items(),
                                  key=lambda x: x[1]["notional_usd_bn"], reverse=True)
    }
    trade_date_str = sorted(trade_dates)[-1] if trade_dates else ""
    return pairs_clean, clean(totals), trade_date_str


def main():
    today = datetime.now(timezone.utc).date()
    candidates = candidate_dates(today)
    zip_bytes = None
    fetched_date = None

    for d in candidates:
        url = s3_url(d)
        print(f"  Trying {url}")
        data = fetch_zip(url)
        if data:
            zip_bytes = data
            fetched_date = d
            print(f"  Found: {len(data):,} bytes for {d}")
            break

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not zip_bytes:
        print("  No DTCC data available for any candidate date", file=sys.stderr)
        output = {
            "fetched": today.isoformat(), "trade_date": "",
            "source": "DTCC GTR · CFTC Cumulative Slice · public dissemination · T+1",
            "note": "Notional capped at $250M per trade per CFTC rules.",
            "pairs": {}, "totals": {"notional_usd_bn": 0.0, "trade_count": 0, "by_product": {}},
            "status": "unavailable",
        }
    else:
        records, csv_name = parse_csv_from_zip(zip_bytes)
        print(f"  Parsed {len(records):,} rows from {csv_name}")
        if not records:
            print("  Empty CSV", file=sys.stderr)
            output = {
                "fetched": today.isoformat(), "trade_date": fetched_date.isoformat(),
                "source": "DTCC GTR · CFTC Cumulative Slice · public dissemination · T+1",
                "note": "Notional capped at $250M per trade per CFTC rules.",
                "pairs": {}, "totals": {"notional_usd_bn": 0.0, "trade_count": 0, "by_product": {}},
                "status": "unavailable",
            }
        else:
            pairs, totals, trade_date = aggregate(records)
            print(f"  {len(pairs)} G8 pairs | trade_date={trade_date} | "
                  f"${totals['notional_usd_bn']:.1f}bn / {totals['trade_count']:,} trades")
            for pair, d in list(pairs.items())[:5]:
                print(f"    {pair}: ${d['notional_usd_bn']:.2f}bn ({d['trade_count']} trades)")
            output = {
                "fetched":    today.isoformat(),
                "trade_date": trade_date or fetched_date.isoformat(),
                "source":     "DTCC GTR · CFTC Cumulative Slice · public dissemination · T+1",
                "note":       "Notional capped at $250M per trade per CFTC rules. Represents a subset of total OTC FX volume.",
                "pairs":      pairs,
                "totals":     totals,
                "status":     "ok",
            }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    print(f"  Written: {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
