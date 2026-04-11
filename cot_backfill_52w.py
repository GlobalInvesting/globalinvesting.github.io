#!/usr/bin/env python3
"""
COT HISTORICAL BACKFILL — 52 WEEKS
====================================
Downloads annual historical TFF files from CFTC.gov and builds up to 52 weeks
of history for each currency in cot-data/*.json.

CFTC publishes annual ZIPs at (comma-separated text, NOT HTML):
  Options+Futures Combined: https://www.cftc.gov/files/dea/history/com_fin_txt_YYYY.zip
  Futures Only:             https://www.cftc.gov/files/dea/history/fut_fin_txt_YYYY.zip

Column layout (Disaggregated TFF, positions "All" = O+F combined or FO):
  [0]  Market_and_Exchange_Names
  [1]  As_of_Date_in_Form_YYMMDD
  [2]  Report_Date_as_MM_DD_YYYY       ← weekEnding source
  [3]  CFTC_Contract_Market_Code
  [4]  CFTC_Market_Code
  [5]  CFTC_Region_Code
  [6]  CFTC_Commodity_Code
  [7]  Open_Interest_All
  [8]  Dealer_Positions_Long_All
  [9]  Dealer_Positions_Short_All
  [10] Dealer_Positions_Spread_All
  [11] Asset_Mgr_Positions_Long_All
  [12] Asset_Mgr_Positions_Short_All
  [13] Asset_Mgr_Positions_Spread_All
  [14] Lev_Money_Positions_Long_All    ← primary signal
  [15] Lev_Money_Positions_Short_All   ← primary signal
  [16] Lev_Money_Positions_Spread_All
  ...

IMPORTANT — CFTC.GOV ACCESS:
  The CFTC blocks downloads from residential/non-datacenter IP addresses.
  If you get HTTP 403 or connection errors running this script locally,
  run it instead via the GitHub Actions workflow:
    engine repo -> Actions -> "Backfill COT History (52 weeks)" -> Run workflow

  GitHub Actions IPs are not blocked by CFTC. The production weekly workflow
  already proves this.

USAGE (from the root of globalinvesting.github.io repo):
  pip install requests
  python3 cot_backfill_52w.py

  # Dry run (no files written):
  python3 cot_backfill_52w.py --dry-run

  # Verbose output (shows week counts per currency):
  python3 cot_backfill_52w.py --dry-run --verbose
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from datetime import date, datetime

import requests

# ── Configuration ────────────────────────────────────────────────────────────

TARGET_WEEKS   = 52
OUTPUT_DIR     = "cot-data"
CURRENT_YEAR   = date.today().year
# Fetch 2 prior years + current year to guarantee 52 weeks regardless of
# when in the year the backfill is run. CFTC publishes YTD files for the
# current year at the same URL pattern as completed years.
YEARS_TO_FETCH = [CURRENT_YEAR - 2, CURRENT_YEAR - 1, CURRENT_YEAR]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; COT-Backfill-Bot/1.0; "
        "+https://globalinvesting.github.io/)"
    )
}

# Match substrings in the Market_and_Exchange_Names column (case-insensitive)
CURRENCY_PATTERNS = {
    "USD": ["usd index", "u.s. dollar index", "us dollar index"],
    "EUR": ["euro fx"],
    "GBP": ["british pound"],
    "JPY": ["japanese yen"],
    "CAD": ["canadian dollar"],
    "CHF": ["swiss franc"],
    "AUD": ["australian dollar"],
    "NZD": ["new zealand dollar", "n.z. dollar", "nz dollar"],
}

# Column indices in the annual CSV (0-based, confirmed from CFTC header row)
COL_MARKET_NAME  = 0
COL_DATE_MMDDYY  = 2   # Report_Date_as_MM_DD_YYYY  e.g. "04/08/2025"
COL_OI           = 7
COL_DD_LONG      = 8
COL_DD_SHORT     = 9
COL_AM_LONG      = 11
COL_AM_SHORT     = 12
COL_LF_LONG      = 14
COL_LF_SHORT     = 15

# ── CSV parser ───────────────────────────────────────────────────────────────

def match_currency(market_name):
    """Return the CCY code for a market name, or None if not a tracked pair."""
    name_lower = market_name.lower()
    for ccy, patterns in CURRENCY_PATTERNS.items():
        for pat in patterns:
            if pat in name_lower:
                return ccy
    return None


def parse_int(val):
    """Parse a formatted integer string like '123,456' or '123456'."""
    try:
        return int(str(val).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def parse_date(val):
    """Parse MM/DD/YYYY into YYYY-MM-DD."""
    val = str(val).strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(val, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def parse_annual_csv(content, verbose=False):
    """
    Parse an annual CFTC TFF CSV file (comma-separated, header row present).
    Returns a list of dicts: {weekEnding, ccy, levLong, levShort, levNet,
                               assetManagerNet, dealerNet}.
    """
    reader = csv.reader(io.StringIO(content))
    rows   = list(reader)

    if not rows:
        print("  ERROR: empty file")
        return []

    # Find the header row (first row that contains "Market_and_Exchange_Names")
    header_idx = None
    for i, row in enumerate(rows[:5]):
        if row and "market_and_exchange_names" in row[0].lower():
            header_idx = i
            break

    if header_idx is None:
        # No standard header — try to detect by column count (≥17 columns expected)
        header_idx = 0

    if verbose:
        print(f"  Header row: {header_idx}  |  Columns: {len(rows[header_idx])}")
        # Show the column names we care about
        hdr = rows[header_idx]
        for idx in [COL_MARKET_NAME, COL_DATE_MMDDYY, COL_OI,
                    COL_DD_LONG, COL_DD_SHORT, COL_AM_LONG, COL_AM_SHORT,
                    COL_LF_LONG, COL_LF_SHORT]:
            if idx < len(hdr):
                print(f"    col[{idx:2d}] = {hdr[idx]}")

    results = []
    counts  = defaultdict(int)

    for row in rows[header_idx + 1:]:
        if len(row) <= COL_LF_SHORT:
            continue

        market_name = row[COL_MARKET_NAME].strip()
        ccy = match_currency(market_name)
        if ccy is None:
            continue

        week_ending = parse_date(row[COL_DATE_MMDDYY])
        if not week_ending:
            continue

        lf_long  = parse_int(row[COL_LF_LONG])
        lf_short = parse_int(row[COL_LF_SHORT])
        if lf_long is None or lf_short is None:
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

    return results


# ── Download ─────────────────────────────────────────────────────────────────

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
            print()
            print("  ** CFTC blocks downloads from residential/non-datacenter IPs.    **")
            print("  ** Run the backfill via GitHub Actions instead:                  **")
            print("  ** engine repo -> Actions -> 'Backfill COT History (52 weeks)'  **")
            print("  ** Click 'Run workflow' -> Run workflow                          **")
        else:
            print(f" FAILED -- HTTP {status}")
        return None
    except Exception as e:
        print(f" FAILED -- {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

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
                blocks = parse_annual_csv(content, verbose=args.verbose)
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
            print()
            print("Most likely cause: CFTC.gov is blocking your IP address.")
            print("This is expected when running locally from a residential connection.")
            print()
            print("Solution -- run the backfill via GitHub Actions:")
            print("  1. Go to the ENGINE repo on GitHub")
            print("  2. Click Actions -> 'Backfill COT History (52 weeks)'")
            print("  3. Click 'Run workflow' -> Run workflow")
            print("  4. The workflow downloads data and commits it to the site repo")
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

        # Merge new history with any newer weeks already in the existing file
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
            "netPosition":    lev_net,
            "longPositions":  lev_long,
            "shortPositions": lev_short,
            "positionCategory": "Leveraged Funds (speculative)",
            "assetManagerNet":   latest.get("assetManagerNet"),
            "assetManagerLong":  existing.get("assetManagerLong"),
            "assetManagerShort": existing.get("assetManagerShort"),
            "dealerNet":   latest.get("dealerNet"),
            "dealerLong":  existing.get("dealerLong"),
            "dealerShort": existing.get("dealerShort"),
            "wowNetChange":   wow_net_change,
            "levNetPctOI":    lev_net_pct_oi,
            "history":        history,
            "sourceType":     existing.get("sourceType", "options_futures_combined"),
            "source":         "CFTC Official",
            "sourceUrl":      "https://www.cftc.gov/dea/options/financial_lof.htm",
            "reportDate":     date.today().isoformat(),
            "lastUpdate":     date.today().isoformat(),
            "weekEnding":     latest["weekEnding"],
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
