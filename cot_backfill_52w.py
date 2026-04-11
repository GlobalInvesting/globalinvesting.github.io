#!/usr/bin/env python3
"""
COT HISTORICAL BACKFILL — 52 WEEKS
====================================
Downloads annual historical TFF files from CFTC.gov and builds up to 52 weeks
of history for each currency in cot-data/*.json.

CFTC publishes annual ZIPs at (comma-separated text):
  Options+Futures Combined: https://www.cftc.gov/files/dea/history/com_fin_txt_YYYY.zip
  Futures Only:             https://www.cftc.gov/files/dea/history/fut_fin_txt_YYYY.zip

Column layout (Disaggregated TFF annual CSV, 87 columns):
  [0]  Market_and_Exchange_Names
  [1]  As_of_Date_in_Form_YYMMDD          (e.g. 250408)
  [2]  Report_Date_as_YYYY-MM-DD           (e.g. 2025-04-08)
  [7]  Open_Interest_All
  [8]  Dealer_Positions_Long_All
  [9]  Dealer_Positions_Short_All
  [11] Asset_Mgr_Positions_Long_All
  [12] Asset_Mgr_Positions_Short_All
  [14] Lev_Money_Positions_Long_All
  [15] Lev_Money_Positions_Short_All

IMPORTANT — CFTC.GOV ACCESS:
  The CFTC blocks downloads from residential/non-datacenter IP addresses.
  Run this script via GitHub Actions (engine repo -> Actions -> 'Backfill COT History').
"""

import argparse
import csv
import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from datetime import date, datetime

import requests

# ── Configuration ────────────────────────────────────────────────────────────

TARGET_WEEKS   = 52
OUTPUT_DIR     = "cot-data"
CURRENT_YEAR   = date.today().year
YEARS_TO_FETCH = [CURRENT_YEAR - 2, CURRENT_YEAR - 1, CURRENT_YEAR]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; COT-Backfill-Bot/1.0; "
        "+https://globalinvesting.github.io/)"
    )
}

# Keywords to match in Market_and_Exchange_Names (case-insensitive substring)
# Using very short substrings to survive any capitalization or spacing variation
CURRENCY_PATTERNS = {
    "USD": ["usd index", "dollar index"],
    "EUR": ["euro fx"],
    "GBP": ["british pound"],
    "JPY": ["japanese yen"],
    "CAD": ["canadian dollar"],
    "CHF": ["swiss franc"],
    "AUD": ["australian dollar"],
    "NZD": ["new zealand", "n.z. dollar", "nz dollar"],
}

# Column indices (confirmed from live header row in GitHub Actions log)
COL_MARKET_NAME = 0
COL_DATE        = 2    # Report_Date_as_YYYY-MM-DD
COL_DATE_ALT    = 1    # As_of_Date_in_Form_YYMMDD  (fallback)
COL_DD_LONG     = 8
COL_DD_SHORT    = 9
COL_AM_LONG     = 11
COL_AM_SHORT    = 12
COL_LF_LONG     = 14
COL_LF_SHORT    = 15

# ── Helpers ───────────────────────────────────────────────────────────────────

def match_currency(market_name):
    """Return CCY code if market_name matches a tracked currency, else None."""
    name_lower = market_name.lower().strip()
    for ccy, patterns in CURRENCY_PATTERNS.items():
        for pat in patterns:
            if pat in name_lower:
                return ccy
    return None


def parse_int(val):
    try:
        return int(str(val).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def parse_date_col(val):
    """Parse date from col[2] (YYYY-MM-DD) or col[1] (YYMMDD or YYYYMMDD)."""
    val = str(val).strip()
    # Try ISO format first (col[2])
    if len(val) == 10 and val[4] == "-":
        try:
            datetime.strptime(val, "%Y-%m-%d")
            return val  # already in our target format
        except ValueError:
            pass
    # Try YYYYMMDD (col[1] sometimes)
    if len(val) == 8:
        try:
            return datetime.strptime(val, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Try YYMMDD (CFTC compact format)
    if len(val) == 6:
        try:
            return datetime.strptime(val, "%y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Legacy MM/DD/YYYY
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(val, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


# ── CSV parser ────────────────────────────────────────────────────────────────

def parse_annual_csv(content, verbose=False):
    """
    Parse an annual CFTC TFF CSV.
    Returns list of dicts: {weekEnding, ccy, levLong, levShort, levNet,
                             assetManagerNet, dealerNet}.
    """
    reader = csv.reader(io.StringIO(content))
    rows   = list(reader)

    if not rows:
        print("  ERROR: empty file")
        return []

    # Detect header row
    header_idx = 0
    for i, row in enumerate(rows[:5]):
        if row and "market" in row[0].lower():
            header_idx = i
            break

    if verbose:
        hdr = rows[header_idx]
        print(f"  Header row: {header_idx}  |  Total columns: {len(hdr)}")
        for idx in [COL_MARKET_NAME, COL_DATE, COL_DATE_ALT,
                    COL_DD_LONG, COL_DD_SHORT, COL_AM_LONG, COL_AM_SHORT,
                    COL_LF_LONG, COL_LF_SHORT]:
            if idx < len(hdr):
                print(f"    col[{idx:2d}] = {hdr[idx]}")

    # Print first 5 unique market names for diagnosis
    sample_names = []
    for row in rows[header_idx + 1: header_idx + 200]:
        if row and row[0].strip() and row[0].strip() not in sample_names:
            sample_names.append(row[0].strip())
        if len(sample_names) >= 8:
            break
    print(f"  Sample Market_and_Exchange_Names (first {len(sample_names)} unique):")
    for n in sample_names:
        ccy = match_currency(n)
        print(f"    {'[' + ccy + ']' if ccy else '[---]'} {repr(n[:80])}")

    # Also print first data row's date columns
    if len(rows) > header_idx + 1:
        first_data = rows[header_idx + 1]
        if len(first_data) > max(COL_DATE, COL_DATE_ALT):
            raw_date  = first_data[COL_DATE]
            raw_date2 = first_data[COL_DATE_ALT]
            parsed    = parse_date_col(raw_date) or parse_date_col(raw_date2)
            print(f"  First data row dates: col[{COL_DATE}]={repr(raw_date)}  "
                  f"col[{COL_DATE_ALT}]={repr(raw_date2)}  -> parsed={parsed}")

    results = []
    counts  = defaultdict(int)
    date_fail = 0
    int_fail  = 0

    for row in rows[header_idx + 1:]:
        if len(row) <= COL_LF_SHORT:
            continue

        market_name = row[COL_MARKET_NAME].strip()
        ccy = match_currency(market_name)
        if ccy is None:
            continue

        # Try primary date col, then fallback
        week_ending = parse_date_col(row[COL_DATE])
        if not week_ending:
            week_ending = parse_date_col(row[COL_DATE_ALT])
        if not week_ending:
            date_fail += 1
            continue

        lf_long  = parse_int(row[COL_LF_LONG])
        lf_short = parse_int(row[COL_LF_SHORT])
        if lf_long is None or lf_short is None:
            int_fail += 1
            continue
        lf_net = lf_long - lf_short

        am_long  = parse_int(row[COL_AM_LONG])
        am_short = parse_int(row[COL_AM_SHORT])
        am_net   = (am_long - am_short) if (am_long is not None and am_short is not None) else None

        dd_long  = parse_int(row[COL_DD_LONG])
        dd_short = parse_int(row[COL_DD_SHORT])
        dd_net   = (dd_long - dd_short) if (dd_long is not None and dd_short is not None) else None

        results.append({
            "weekEnding":      week_ending,
            "ccy":             ccy,
            "levLong":         lf_long,
            "levShort":        lf_short,
            "levNet":          lf_net,
            "assetManagerNet": am_net,
            "dealerNet":       dd_net,
        })
        counts[ccy] += 1

    if verbose:
        for ccy in sorted(CURRENCY_PATTERNS):
            print(f"  {ccy}: {counts[ccy]} rows parsed")
        if date_fail:
            print(f"  Date parse failures: {date_fail}")
        if int_fail:
            print(f"  Int parse failures:  {int_fail}")

    return results


# ── Download ──────────────────────────────────────────────────────────────────

def fetch_annual_zip(year, report_type="com"):
    url = f"https://www.cftc.gov/files/dea/history/{report_type}_fin_txt_{year}.zip"
    print(f"  GET {url}", end="", flush=True)
    try:
        r = requests.get(url, headers=HEADERS, timeout=90)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            txt_files = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not txt_files:
                print(f" FAILED (no .txt in ZIP)")
                return None
            content = z.read(txt_files[0]).decode("latin-1")
        print(f" OK ({len(r.content)//1024}KB, {len(content):,} chars)")
        return content
    except requests.HTTPError as e:
        status = e.response.status_code
        if status == 403:
            print(f" FAILED -- HTTP 403 (IP blocked by CFTC)")
            print("  ** Run the backfill via GitHub Actions (engine repo -> Actions). **")
        else:
            print(f" FAILED -- HTTP {status}")
        return None
    except Exception as e:
        print(f" FAILED -- {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COT Historical Backfill -- 52 weeks")
    parser.add_argument("--weeks",      type=int, default=TARGET_WEEKS)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print(f"COT HISTORICAL BACKFILL -- {args.weeks} weeks")
    print(f"Output: {args.output_dir}/   Dry-run: {args.dry_run}")
    print("=" * 65)

    all_blocks  = []
    any_blocked = False

    for year in YEARS_TO_FETCH:
        print(f"\n[{year}]")
        got_data = False
        for report_type in ("com", "fut"):
            content = fetch_annual_zip(year, report_type)
            if content:
                print(f"  Parsing ({report_type}) as CSV...")
                # Always pass verbose=True for backfill runs so we get diagnostics
                blocks = parse_annual_csv(content, verbose=True)
                unique_weeks = len(set(b["weekEnding"] for b in blocks))
                print(f"  -> {len(blocks)} records ({unique_weeks} unique weeks, "
                      f"{len(set(b['ccy'] for b in blocks))} currencies)")
                all_blocks.extend(blocks)
                got_data = True
                break
            else:
                any_blocked = True
        if not got_data:
            print(f"  WARNING: No data obtained for {year}")

    if not all_blocks:
        print()
        print("ERROR: No data downloaded.")
        if any_blocked:
            print("Most likely cause: CFTC.gov is blocking your IP address.")
            print("Solution: engine repo -> Actions -> 'Backfill COT History (52 weeks)' -> Run workflow")
        return 1

    # Group by currency, deduplicate, sort
    by_ccy = defaultdict(dict)
    for b in all_blocks:
        by_ccy[b["ccy"]][b["weekEnding"]] = b

    print(f"\n{'='*65}")
    print("DATA DOWNLOADED")
    print(f"{'='*65}")
    for ccy in sorted(CURRENCY_PATTERNS):
        entries = sorted(by_ccy[ccy].values(), key=lambda x: x["weekEnding"])
        if entries:
            print(f"  {ccy:3s}: {len(entries):2d} weeks  "
                  f"({entries[0]['weekEnding']} -> {entries[-1]['weekEnding']})")
        else:
            print(f"  {ccy:3s}: WARNING no data")

    print(f"\n{'='*65}")
    print("WRITING JSON FILES")
    print(f"{'='*65}")

    os.makedirs(args.output_dir, exist_ok=True)

    for ccy in CURRENCY_PATTERNS:
        path = os.path.join(args.output_dir, f"{ccy}.json")

        raw_entries = sorted(by_ccy[ccy].values(), key=lambda x: x["weekEnding"])
        if not raw_entries:
            print(f"  {ccy}: WARNING no data -- file unchanged")
            continue

        new_history = [
            {
                "weekEnding":      b["weekEnding"],
                "levNet":          b["levNet"],
                "levLong":         b["levLong"],
                "levShort":        b["levShort"],
                "assetManagerNet": b["assetManagerNet"],
                "dealerNet":       b["dealerNet"],
            }
            for b in raw_entries
        ]

        existing = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing = json.load(f)
            except Exception:
                pass

        existing_history = existing.get("history", [])
        combined = {h["weekEnding"]: h for h in new_history}
        for h in existing_history:
            if h["weekEnding"] not in combined:
                combined[h["weekEnding"]] = h

        history = sorted(combined.values(), key=lambda x: x["weekEnding"])[-args.weeks:]

        latest    = history[-1]
        lev_net   = latest["levNet"]
        lev_long  = latest.get("levLong")
        lev_short = latest.get("levShort")

        wow_net_change = None
        if len(history) >= 2:
            wow_net_change = lev_net - history[-2]["levNet"]

        lev_oi = (lev_long or 0) + (lev_short or 0)
        lev_net_pct_oi = round(lev_net / lev_oi * 100, 1) if lev_oi else None

        updated = {
            "netPosition":       lev_net,
            "longPositions":     lev_long,
            "shortPositions":    lev_short,
            "positionCategory":  "Leveraged Funds (speculative)",
            "assetManagerNet":   latest.get("assetManagerNet"),
            "assetManagerLong":  existing.get("assetManagerLong"),
            "assetManagerShort": existing.get("assetManagerShort"),
            "dealerNet":         latest.get("dealerNet"),
            "dealerLong":        existing.get("dealerLong"),
            "dealerShort":       existing.get("dealerShort"),
            "wowNetChange":      wow_net_change,
            "levNetPctOI":       lev_net_pct_oi,
            "history":           history,
            "sourceType":        existing.get("sourceType", "options_futures_combined"),
            "source":            "CFTC Official",
            "sourceUrl":         "https://www.cftc.gov/dea/options/financial_lof.htm",
            "reportDate":        date.today().isoformat(),
            "lastUpdate":        date.today().isoformat(),
            "weekEnding":        latest["weekEnding"],
        }
        updated = {k: v for k, v in updated.items() if v is not None}

        if args.dry_run:
            print(f"  {ccy}: [DRY-RUN] {len(history)}w  "
                  f"({history[0]['weekEnding']} -> {history[-1]['weekEnding']})")
        else:
            with open(path, "w") as f:
                json.dump(updated, f, indent=2)
            print(f"  {ccy}: OK {len(history)}w  "
                  f"({history[0]['weekEnding']} -> {history[-1]['weekEnding']})")

    print(f"\n{'='*65}")
    if args.dry_run:
        print("DRY-RUN complete. No files were written.")
    else:
        print("BACKFILL COMPLETE.")
        print()
        print("Next steps:")
        print("  1. Verify: python3 -c \"import json; d=json.load(open('cot-data/EUR.json')); print(len(d['history']), 'weeks')\"")
        print("  2. git add cot-data/ && git commit -m 'backfill: 52w COT history' && git push")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
