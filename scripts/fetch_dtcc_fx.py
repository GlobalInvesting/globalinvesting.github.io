#!/usr/bin/env python3
"""
fetch_dtcc_fx.py
Fetches FX OTC transaction data from the DTCC GTR CFTC Cumulative Slice Reports
(S3 public bucket) and writes dtcc-data/dtcc_fx.json.

Source:  DTCC GTR — CFTC Cumulative Slice Reports (Phase 2, Jan 2024+)
         PPD User Guide v2.8, May 2025 — Appendix A confirmed field list
S3 URL:  https://kgc0418-tdw-data-0.s3.amazonaws.com/cftc/eod/CFTC_CUMULATIVE_FOREX_{YYYY}_{MM}_{DD}.zip
Delay:   T+1 (cumulative for prior business day)
Access:  Public — no API key required (Dodd-Frank §2(a)(13))

Phase 2 field mapping (confirmed from PPD User Guide v2.8 Appendix A):
  Currency pair:  "Notional currency-Leg 1" + "Notional currency-Leg 2" (swaps/fwds/spots/NDFs)
                  "Strike price currency/currency pair"                   (FX options)
  Notional:       "Notional amount-Leg 1"
  Product:        "Product name"
  Action:         "Action type"  (NEWT/NEW/"" = new trade; skip MODI/TERM/CORR/NOVA/REVI/VALU)
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
    "EUR/AUD", "AUD/JPY",
    "GBP/CHF", "GBP/AUD", "EUR/CAD",
}

# Build bidirectional lookup: EURUSD / EUR/USD / EUR-USD all → "EUR/USD"
# Inverse also maps to the canonical form (USDEUR → "EUR/USD")
PAIR_LOOKUP: dict = {}
for _p in G8_PAIRS:
    _left, _right = _p.split("/")
    for _sep in ("", "/", "-"):
        PAIR_LOOKUP[_left + _sep + _right] = _p
        PAIR_LOOKUP[_right + _sep + _left] = _p  # inverse → same canonical

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
    "fxswap":    "FxSwap",
    "fxforward": "FxForward",
    "fxspot":    "FxSpot",
    "fxoption":  "FxOption",
    "fxndf":     "FxNDF",
}

# Action types that represent lifecycle events — NOT new notional exposure
SKIP_ACTIONS = {"MODI", "TERM", "CORR", "NOVA", "REVI", "VALU", "PRTO", "EROR"}


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


def fetch_zip(url: str):
    try:
        resp = requests.get(
            url, timeout=90,
            headers={"User-Agent": "globalinvesting-fx-terminal/1.0"}
        )
        if resp.status_code == 200 and len(resp.content) > 500:
            return resp.content
        if resp.status_code == 404:
            return None
        print(f"  HTTP {resp.status_code} for {url}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  Error fetching {url}: {exc}", file=sys.stderr)
        return None


def parse_csv_from_zip(zip_bytes: bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        csv_name = next((n for n in names if n.lower().endswith(".csv")), None)
        if not csv_name:
            print(f"  No CSV in ZIP. Contents: {names}", file=sys.stderr)
            return [], []
        with zf.open(csv_name) as f:
            text = f.read().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    headers = list(reader.fieldnames or [])
    return rows, headers


def normalise_pair(raw: str):
    p = raw.strip().upper().replace(" ", "")
    return PAIR_LOOKUP.get(p)


def build_pair_from_legs(ccy1: str, ccy2: str):
    if not ccy1 or not ccy2:
        return None
    c1 = ccy1.strip().upper()[:3]
    c2 = ccy2.strip().upper()[:3]
    return PAIR_LOOKUP.get(c1 + c2) or PAIR_LOOKUP.get(c2 + c1)


def get(row_lower: dict, *candidates: str) -> str:
    for c in candidates:
        v = row_lower.get(c.lower().strip())
        if v is not None:
            return v.strip()
    return ""


def parse_notional(row_lower: dict) -> float:
    for field in (
        "notional amount-leg 1",
        "notional amount-leg1",
        "notional amount leg 1",
        "notional_amount_1",
    ):
        v = row_lower.get(field, "")
        if v and v not in ("", "null", "none", "n/a", "-"):
            try:
                return float(str(v).replace(",", "")) / 1e9
            except (ValueError, TypeError):
                pass
    return 0.0


def aggregate(records: list, headers: list):
    # Diagnostic header/row dump — helps confirm field names in workflow log
    print(f"  === CSV HEADERS ({len(headers)}) ===")
    for h in headers:
        print(f"    '{h}'")
    if records:
        first_lower = {k.lower().strip(): (v.strip() if v else "") for k, v in records[0].items()}
        print("  === FIRST ROW (non-empty fields only) ===")
        for k, v in first_lower.items():
            if v:
                print(f"    '{k}': '{v[:80]}'")
        # Explicitly log product name distribution across first 200 rows (diagnose product breakdown)
        from collections import Counter
        prod_sample = Counter()
        for rec in records[:200]:
            rl2 = {k.lower().strip(): (v.strip() if v else "") for k, v in rec.items()}
            raw = (rl2.get("product name") or rl2.get("sub_asset_class_for_other_commodity") or "")
            prod_sample[raw or "(blank)"] += 1
        print(f"  === PRODUCT NAME sample (first 200 rows) ===")
        for pname, cnt in prod_sample.most_common(10):
            print(f"    '{pname}': {cnt}")

    pairs = defaultdict(lambda: {
        "notional_usd_bn": 0.0, "trade_count": 0,
        "by_product": defaultdict(lambda: {"notional_usd_bn": 0.0, "count": 0}),
    })
    totals = {
        "notional_usd_bn": 0.0, "trade_count": 0,
        "by_product": defaultdict(lambda: {"notional_usd_bn": 0.0, "count": 0}),
    }
    trade_dates = set()
    cnt = {"skipped_action": 0, "skipped_pair": 0, "skipped_notional": 0, "accepted": 0}

    for rec in records:
        rl = {k.lower().strip(): (v.strip() if v else "") for k, v in rec.items()}

        # Skip lifecycle events — keep only new trades (NEWT, NEW, or blank)
        action = get(rl, "action type", "action").upper()
        if action in SKIP_ACTIONS:
            cnt["skipped_action"] += 1
            continue

        # --- Determine currency pair ---
        pair = None

        # 1. FX options: "Strike price currency/currency pair"
        raw_opt = get(rl,
            "strike price currency/currency pair",
            "strike price currency currency pair",
        )
        if raw_opt:
            pair = normalise_pair(raw_opt)

        # 2. All other FX products: build from Notional currency legs (Phase 2 primary)
        if not pair:
            ccy1 = get(rl,
                "notional currency-leg 1", "notional currency-leg1",
                "notional currency leg 1", "notional_currency_1",
            )
            ccy2 = get(rl,
                "notional currency-leg 2", "notional currency-leg2",
                "notional currency leg 2", "notional_currency_2",
            )
            pair = build_pair_from_legs(ccy1, ccy2)

        if not pair:
            cnt["skipped_pair"] += 1
            continue

        notional = parse_notional(rl)
        if notional <= 0:
            cnt["skipped_notional"] += 1
            continue

        # Normalise product name
        raw_product = get(rl, "product name", "sub_asset_class_for_other_commodity")
        key = raw_product.lower().replace(" ", "").replace("_", "").replace("-", "")
        product = PRODUCT_NORM.get(key)
        if not product:
            tail = key.split(":")[-1] if ":" in key else key
            product = PRODUCT_NORM.get(tail, tail.capitalize() or "FxSwap")

        # Trade date
        td = get(rl,
            "execution timestamp", "event timestamp",
            "dissemination timestamp", "effective date",
        )[:10]
        if td and len(td) == 10 and td[:4].isdigit():
            trade_dates.add(td)

        pairs[pair]["notional_usd_bn"] += notional
        pairs[pair]["trade_count"] += 1
        pairs[pair]["by_product"][product]["notional_usd_bn"] += notional
        pairs[pair]["by_product"][product]["count"] += 1
        totals["notional_usd_bn"] += notional
        totals["trade_count"] += 1
        totals["by_product"][product]["notional_usd_bn"] += notional
        totals["by_product"][product]["count"] += 1
        cnt["accepted"] += 1

    print(f"  Counts: accepted={cnt['accepted']} | "
          f"skipped_action={cnt['skipped_action']} | "
          f"skipped_pair={cnt['skipped_pair']} | "
          f"skipped_notional={cnt['skipped_notional']}")

    def clean(entry):
        return {
            "notional_usd_bn": round(entry["notional_usd_bn"], 2),
            "trade_count": entry["trade_count"],
            "by_product": {
                k: {"notional_usd_bn": round(v["notional_usd_bn"], 2), "count": v["count"]}
                for k, v in sorted(
                    entry["by_product"].items(),
                    key=lambda x: x[1]["notional_usd_bn"], reverse=True,
                )
            },
        }

    pairs_clean = {
        pair: clean(data)
        for pair, data in sorted(
            pairs.items(),
            key=lambda x: x[1]["notional_usd_bn"], reverse=True,
        )
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
        records, headers = parse_csv_from_zip(zip_bytes)
        print(f"  Parsed {len(records):,} rows from CSV")

        if not records:
            output = {
                "fetched": today.isoformat(),
                "trade_date": fetched_date.isoformat(),
                "source": "DTCC GTR · CFTC Cumulative Slice · public dissemination · T+1",
                "note": "Notional capped at $250M per trade per CFTC rules.",
                "pairs": {}, "totals": {"notional_usd_bn": 0.0, "trade_count": 0, "by_product": {}},
                "status": "unavailable",
            }
        else:
            pairs, totals, trade_date = aggregate(records, headers)
            print(f"  Result: {len(pairs)} G8 pairs | trade_date={trade_date} | "
                  f"${totals['notional_usd_bn']:.1f}bn / {totals['trade_count']:,} trades")
            for pair, d in list(pairs.items())[:8]:
                print(f"    {pair}: ${d['notional_usd_bn']:.2f}bn ({d['trade_count']} trades)")

            output = {
                "fetched":    today.isoformat(),
                "trade_date": trade_date or fetched_date.isoformat(),
                "source":     "DTCC GTR · CFTC Cumulative Slice · public dissemination · T+1",
                "note":       "Notional capped at $250M per trade per CFTC rules. Represents a subset of total OTC FX volume.",
                "pairs":      pairs,
                "totals":     totals,
                "status":     "ok" if pairs else "no_g8_pairs",
            }

    with open(OUTPUT_PATH, "w") as fh:
        json.dump(output, fh, separators=(",", ":"))
    print(f"  Written: {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
