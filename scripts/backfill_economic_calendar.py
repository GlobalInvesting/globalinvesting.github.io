#!/usr/bin/env python3
"""
backfill_economic_calendar.py  v5.0
──────────────────────────────────────────────────────────────────────────────
One-shot historical backfill for calendar-data/calendar.json.

Industry-standard Economic Surprises coverage: targets ~8-10 indicator types
per currency so all 8 G8 currencies reach 80-120 FRED events/year vs USD's
similar depth. Based on the Citi/Bloomberg ESI methodology (GDP, CPI,
unemployment, trade balance, retail sales, industrial production, core CPI
as core indicators).

SOURCES
  1. FRED API (St Louis Fed) — api.stlouisfed.org
     Free public API. Requires FRED_API_KEY env var.
  2. Eurostat HICP series via FRED — current through 2026.

CHANGES vs v4.1
  ── Industrial Production: definitive OECD series ID fix ─────────────────
  The PRINTO01XXM659Y pattern (OECD MEI old format) returned HTTP 400 on
  FRED's public API for ALL non-EUR currencies. Replaced with the new OECD
  data format (country-prefixed GYSAM = Growth rate same period previous
  year, monthly, Seasonally Adjusted):

    OLD (400 on public API)  → NEW (confirmed current)
    PRINTO01GBM659Y          → GBRPRINTO01GYSAM  (Dec 2025, Mar 16 2026 ✓)
    PRINTO01CAM659Y          → CANPRINTO01GYSAM  (Dec 2025, Mar 16 2026 ✓)
    PRINTO01JPM659Y          → JPNPRINTO01GYSAM  (Mar 2026, May 2026 ✓)
    PRINTO01AUM659Y          → AUSPRINTO01GYSAM  (same OECD pattern ✓)
    PRINTO01EZM659Y          → REMOVED (EA19PRINTO01GYSAM stale: Oct 2023)

  This restores ~48 FRED events/year for GBP/CAD/JPY/AUD industrial production.
  EUR IP removed until a current Eurostat-on-FRED series is identified.

  ── Core CPI (ex-food-energy) added for 5 currencies ─────────────────────
  Pattern CPGRLE01XXM659N = Core CPI YoY%, monthly, confirmed through
  Mar-Apr 2025. Adds ~12 events/year per currency:

    CPGRLE01GBM659N  — UK Core CPI YoY (Mar 2025 ✓)
    CPGRLE01JPM659N  — Japan Core CPI YoY (Mar 2025 ✓)
    CPGRLE01AUM659N  — Australia Core CPI YoY (Mar 2025 ✓)
    CPGRLE01CAM659N  — Canada Core CPI YoY (Mar 2025 ✓)
    CPGRLE01CHM659N  — Switzerland Core CPI YoY (Mar 2025 ✓)
  (EUR already has TOTNRGFOODEA20MI15XM for Core HICP.)

  ── Expected FRED event yield per currency after v5.0 ─────────────────────
    USD: ~172/yr (unchanged — 14 monthly + 1 quarterly series)
    EUR: ~84/yr  (IP removed, CoreCPI already present as TOTNRGFOOD...)
    GBP: ~96/yr  (was ~72 — +IP restored +CoreCPI added = +2 monthly series)
    JPY: ~96/yr  (was ~72 — same)
    AUD: ~96/yr  (was ~72 — same)
    CAD: ~104/yr (was ~80 — +IP restored +CoreCPI added)
    CHF: ~92/yr  (was ~80 — +CoreCPI added)
    NZD: ~40/yr  (unchanged — quarterly-only economy)

CUMULATIVE HISTORY
  v4.0/v4.1: EUR CPI fix (Eurostat HICP), NZD unemployment fix, +25 new
    series for all G8 pairs (trade balance, retail sales, confidence, permits).
  v5.0: Industrial Production series ID fix + Core CPI added (see header).

MERGE STRATEGY
  - Default: existing events with actuals are protected (never overwritten)
  - --force: FRED events are overwritten regardless of actuals
  - ForexFactory events are NEVER overwritten (even with --force)
  - Deduplication key: (currency, dateISO, event_name_normalised)

HOW TO RUN
  FRED_API_KEY=<key> python scripts/backfill_economic_calendar.py
  FRED_API_KEY=<key> python scripts/backfill_economic_calendar.py --force
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from collections import Counter

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

FRED_API_KEY     = os.environ.get("FRED_API_KEY", "")
FRED_BASE        = "https://api.stlouisfed.org/fred/series/observations"
OUTPUT_PATH      = "calendar-data/calendar.json"
MAX_HISTORY_DAYS = 365
FETCH_TIMEOUT    = 25
RATE_LIMIT_SLEEP = 0.30   # seconds between FRED API calls

HEADERS = {
    "User-Agent": "globalinvesting-bot/4.0 (https://globalinvesting.github.io)",
}

FLAG_MAP = {
    "USD": "\U0001f1fa\U0001f1f8",
    "EUR": "\U0001f1ea\U0001f1fa",
    "GBP": "\U0001f1ec\U0001f1e7",
    "JPY": "\U0001f1ef\U0001f1f5",
    "AUD": "\U0001f1e6\U0001f1fa",
    "CAD": "\U0001f1e8\U0001f1e6",
    "CHF": "\U0001f1e8\U0001f1ed",
    "NZD": "\U0001f1f3\U0001f1ff",
}

# ── FRED series catalogue ─────────────────────────────────────────────────────
#
# as_change=True  → compute MoM or QoQ absolute change (level → diff)
# pct_change=True → compute % change (level → %Δ)
# quarterly=True  → use quarterly release lag (60 days vs 45)
# index_qoq=True  → compute QoQ % from consecutive level/index values
# scale_b=True    → divide raw value by 1e9 to display as billions
# scale_m=True    → divide raw value by 1e6 to display as millions
# direct_pct=True → series already returns % directly (no transformation needed)

FRED_SERIES = {
    # ══ USD ══════════════════════════════════════════════════════════════════
    # 13 series, mostly monthly. USD is the benchmark; all others aim for parity.
    "PAYEMS": {
        "event": "Non-Farm Payrolls", "currency": "USD", "impact": "high",
        "unit": "K", "as_change": True, "pct_change": False, "is_inverse": False,
    },
    "UNRATE": {
        "event": "Unemployment Rate", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    "CPIAUCSL": {
        "event": "CPI (MoM)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": True, "pct_change": True, "is_inverse": False,
    },
    "CPILFESL": {
        "event": "Core CPI (MoM)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": True, "pct_change": True, "is_inverse": False,
    },
    "PCEPI": {
        "event": "PCE Price Index (MoM)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": True, "pct_change": True, "is_inverse": False,
    },
    "PCEPILFE": {
        "event": "Core PCE Price Index (MoM)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": True, "pct_change": True, "is_inverse": False,
    },
    "RSXFS": {
        "event": "Retail Sales (MoM)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": True, "pct_change": True, "is_inverse": False,
    },
    "A191RL1Q225SBEA": {
        "event": "GDP (QoQ)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False, "quarterly": True,
        "direct_pct": True,
    },
    "BOPGSTB": {
        "event": "Trade Balance", "currency": "USD", "impact": "medium",
        "unit": "B", "as_change": False, "is_inverse": True,
        "scale_b": True,
    },
    "HOUST": {
        "event": "Housing Starts", "currency": "USD", "impact": "medium",
        "unit": "K", "as_change": False, "is_inverse": False,
        "scale_k": True,
    },
    "PERMIT": {
        "event": "Building Permits", "currency": "USD", "impact": "medium",
        "unit": "K", "as_change": False, "is_inverse": False,
        "scale_k": True,
    },
    "UMCSENT": {
        "event": "Michigan Consumer Sentiment", "currency": "USD", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False,
    },
    "PPIACO": {
        "event": "PPI (MoM)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": True, "pct_change": True, "is_inverse": False,
    },

    # ══ EUR ══════════════════════════════════════════════════════════════════
    # Target: 8 indicator types × 12 months ≈ 80-100 FRED events/year.
    #
    # CPI: Eurostat HICP — confirmed available through Feb 2026, returns YoY %
    # directly. The OECD series (CPALTT01EZM659N, CPHPTT01EZM659N) both 400 or
    # return 0 observations on the FRED public API because Euro Area OECD MEI
    # data is restricted. Eurostat HICP series are open-access on FRED.
    "CP0000EZ19M086NEST": {
        "event": "CPI (YoY)", "currency": "EUR", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
        # Eurostat HICP All-Items for Euro Area (19 Countries). Returns
        # "Percent Change from Year Ago" directly. Confirmed through Dec 2025.
    },
    # Core CPI (HICP ex food, energy, alcohol, tobacco) — ECB's preferred metric
    "TOTNRGFOODEA20MI15XM": {
        "event": "Core CPI (YoY)", "currency": "EUR", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
        # Eurostat HICP ex-energy-food-alcohol-tobacco EA-20. Confirmed Dec 2025.
    },
    # Unemployment: OECD MEI EZ code works for unemployment (unlike CPI)
    "LRHUTTTTEZM156S": {
        "event": "Unemployment Rate", "currency": "EUR", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    # GDP
    "NAEXKP01EZQ657S": {
        "event": "GDP (QoQ)", "currency": "EUR", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
    # Trade Balance — monthly, in EUR. Confirmed through Jan 2026.
    "XTNTVA01EZM664S": {
        "event": "Trade Balance", "currency": "EUR", "impact": "medium",
        "unit": "B", "as_change": False, "is_inverse": True,
        "scale_b": True,
        # OECD: Trade Balance commodities for Euro Area, EUR, SA, monthly.
    },
    # Retail Sales QoQ% — quarterly OECD MEI for Euro Area
    "SLRTTO01EZQ657S": {
        "event": "Retail Sales (QoQ)", "currency": "EUR", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
        # OECD Growth rate previous period, SA, quarterly.
    },
    # Industrial Production EUR monthly — EA19PRINTO01GYSAM last updated Oct 2023
    # (OECD MEI new format does not carry current data for EA19 on public FRED).
    # Removed in v5.0; no current Eurostat-on-FRED YoY series available yet.

    # ══ GBP ══════════════════════════════════════════════════════════════════
    # Target: 7+ indicator types.
    "CPALTT01GBM659N": {
        "event": "CPI (YoY)", "currency": "GBP", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },
    # Core CPI GBP monthly YoY% (ex-food-energy) — confirmed Mar 2025
    # CPGRLE01GBM659N: OECD CPI All Items Non-Food Non-Energy, YoY%, monthly.
    "CPGRLE01GBM659N": {
        "event": "Core CPI (YoY)", "currency": "GBP", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },
    "LRHUTTTTGBM156S": {
        "event": "Unemployment Rate", "currency": "GBP", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    "CLVMNACSCAB1GQUK": {
        "event": "GDP (QoQ)", "currency": "GBP", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
    },
    # Trade Balance UK monthly — in USD converted. Falls back gracefully if 400.
    "XTNTVA01GBM667S": {
        "event": "Trade Balance", "currency": "GBP", "impact": "medium",
        "unit": "B", "as_change": False, "is_inverse": True,
        "scale_b": True,
    },
    # Retail Sales QoQ% GBP quarterly
    "SLRTTO01GBQ657S": {
        "event": "Retail Sales (QoQ)", "currency": "GBP", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
    },
    # Industrial Production GBP monthly YoY% — OECD new format (v5.0 fix)
    # GBRPRINTO01GYSAM: Growth rate same period prev year, SA, monthly.
    # Confirmed Dec 2025, updated Mar 16 2026. Replaces PRINTO01GBM659Y (400).
    "GBRPRINTO01GYSAM": {
        "event": "Industrial Production (YoY)", "currency": "GBP", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },

    # ══ JPY ══════════════════════════════════════════════════════════════════
    # Target: 7+ indicator types.
    "CPALTT01JPM659N": {
        "event": "CPI (YoY)", "currency": "JPY", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },
    # Core CPI JPY monthly YoY% (ex-food-energy) — confirmed Mar 2025
    "CPGRLE01JPM659N": {
        "event": "Core CPI (YoY)", "currency": "JPY", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },
    "LRHUTTTTJPM156S": {
        "event": "Unemployment Rate", "currency": "JPY", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    "JPNRGDPEXP": {
        "event": "GDP (QoQ)", "currency": "JPY", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
    # Trade Balance JPY monthly
    "XTNTVA01JPM664S": {
        "event": "Trade Balance", "currency": "JPY", "impact": "medium",
        "unit": "B", "as_change": False, "is_inverse": True,
        "scale_b": True,
    },
    # Retail Sales QoQ% JPY quarterly
    "SLRTTO01JPQ657S": {
        "event": "Retail Sales (QoQ)", "currency": "JPY", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
    },
    # Industrial Production JPY monthly YoY% — OECD new format (v5.0 fix)
    # JPNPRINTO01GYSAM: Growth rate same period prev year, SA, monthly.
    # Confirmed current (Mar 2026 data). Replaces PRINTO01JPM659Y (400).
    "JPNPRINTO01GYSAM": {
        "event": "Industrial Production (YoY)", "currency": "JPY", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },

    # ══ AUD ══════════════════════════════════════════════════════════════════
    # Target: 7+ indicator types. AUD CPI is quarterly (ABS release schedule).
    "AUSCPIALLQINMEI": {
        "event": "CPI (QoQ)", "currency": "AUD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
    "LRHUTTTTAUM156S": {
        "event": "Unemployment Rate", "currency": "AUD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    "AUSGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "AUD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
    # Trade Balance AUD monthly
    "XTNTVA01AUM664S": {
        "event": "Trade Balance", "currency": "AUD", "impact": "medium",
        "unit": "B", "as_change": False, "is_inverse": True,
        "scale_b": True,
    },
    # Retail Sales QoQ% AUD quarterly
    "SLRTTO01AUQ657S": {
        "event": "Retail Sales (QoQ)", "currency": "AUD", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
    },
    # Industrial Production AUD monthly YoY% — OECD new format (v5.0 fix)
    # AUSPRINTO01GYSAM: Growth rate same period prev year, SA, monthly.
    # Same OECD MEI new-format pattern as GBR/CAN/JPN. Replaces PRINTO01AUM659Y (400).
    "AUSPRINTO01GYSAM": {
        "event": "Industrial Production (YoY)", "currency": "AUD", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },
    # Core CPI AUD monthly YoY% (ex-food-energy) — confirmed Mar 2025
    # AUD headline CPI is quarterly; OECD CPGRLE monthly core fills the gap.
    "CPGRLE01AUM659N": {
        "event": "Core CPI (YoY)", "currency": "AUD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },

    # ══ CAD ══════════════════════════════════════════════════════════════════
    # Target: 7+ indicator types.
    "CPALTT01CAM659N": {
        "event": "CPI (YoY)", "currency": "CAD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },
    # Core CPI CAD monthly YoY% (ex-food-energy) — confirmed Mar 2025
    "CPGRLE01CAM659N": {
        "event": "Core CPI (YoY)", "currency": "CAD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },
    "LRHUTTTTCAM156S": {
        "event": "Unemployment Rate", "currency": "CAD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    "CANGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "CAD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
    # Trade Balance CAD monthly
    "XTNTVA01CAM664S": {
        "event": "Trade Balance", "currency": "CAD", "impact": "medium",
        "unit": "B", "as_change": False, "is_inverse": True,
        "scale_b": True,
    },
    # Retail Sales QoQ% CAD quarterly
    "SLRTTO01CAQ657S": {
        "event": "Retail Sales (QoQ)", "currency": "CAD", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
    },
    # Industrial Production CAD monthly YoY% — OECD new format (v5.0 fix)
    # CANPRINTO01GYSAM: Growth rate same period prev year, SA, monthly.
    # Confirmed Dec 2025, updated Mar 16 2026. Replaces PRINTO01CAM659Y (400).
    "CANPRINTO01GYSAM": {
        "event": "Industrial Production (YoY)", "currency": "CAD", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },

    # ══ CHF ══════════════════════════════════════════════════════════════════
    # Target: 6+ indicator types. CHF has limited monthly data on FRED public API.
    "CPALTT01CHM659N": {
        "event": "CPI (YoY)", "currency": "CHF", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },
    # Core CPI CHF monthly YoY% (ex-food-energy) — confirmed Mar 2025
    "CPGRLE01CHM659N": {
        "event": "Core CPI (YoY)", "currency": "CHF", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "direct_pct": True,
    },
    # Unemployment: LRUN64TTCHM156S (15-64 age group) confirmed working in v3.0.
    # Also try LRHUTTTTCHM156S as primary; fall back gracefully if 400.
    "LRUN64TTCHM156S": {
        "event": "Unemployment Rate", "currency": "CHF", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    "CHEGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "CHF", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
    # Trade Balance CHF monthly (SA, in CHF). Confirmed through Jan 2026.
    # XTNTVA01CHM664S has data through Dec 2024 (SA); use non-SA variant:
    # XTNTVA01CHM664N has data through Jan 2026 (not SA).
    # Using USD-converted SA version: XTNTVA01CHM667S through Jan 2026.
    "XTNTVA01CHM667S": {
        "event": "Trade Balance", "currency": "CHF", "impact": "medium",
        "unit": "B", "as_change": False, "is_inverse": True,
        "scale_b": True,
        # USD-converted, SA, monthly. Through Jan 2026.
    },
    # Retail Sales QoQ% CHF quarterly
    "SLRTTO01CHQ657S": {
        "event": "Retail Sales (QoQ)", "currency": "CHF", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
    },

    # ══ NZD ══════════════════════════════════════════════════════════════════
    # Target: 6+ indicator types. NZD data is primarily quarterly (Stats NZ).
    # NZD CPI and Retail Sales are quarterly; unemployment is quarterly.
    "NZLCPIALLQINMEI": {
        "event": "CPI (QoQ)", "currency": "NZD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
    # Unemployment: LRUNTTTTNZQ156S — confirmed through Q4 2025.
    # (previous: LRHUTTTTDZM156S was wrong country code DZ ≠ NZ)
    "LRUNTTTTNZQ156S": {
        "event": "Unemployment Rate", "currency": "NZD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
        "quarterly": True,
        # OECD Infra-Annual Labor Statistics: Unemployment Rate Total: 15+
        # for New Zealand. Quarterly, SA. Confirmed through Q4 2025.
    },
    "NZLGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "NZD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
    # Trade Balance NZD monthly — confirmed through Jan 2026.
    "XTNTVA01NZM664S": {
        "event": "Trade Balance", "currency": "NZD", "impact": "medium",
        "unit": "B", "as_change": False, "is_inverse": True,
        "scale_b": True,
        # OECD: Trade Balance commodities NZL, NZD, SA, monthly. Through Jan 2026.
    },
    # Retail Sales QoQ% NZD quarterly — confirmed through Q4 2025.
    "SLRTTO01NZQ657S": {
        "event": "Retail Sales (QoQ)", "currency": "NZD", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
    },
    # Employment Rate NZD quarterly (15-64 age group) — through Q1 2026.
    "LREM64TTNZQ156S": {
        "event": "Employment Rate", "currency": "NZD", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True,
    },

    # ══ CONFIDENCE & PERMITS — confirmed current (2025-2026) ══════════════════
    #
    # Pattern: CSCICP02XXM460S = Consumer Confidence, % balance, SA, monthly
    #          BSCICP02XXM460S = Business Confidence, % balance, SA, monthly/quarterly
    #          XXXODCNPI03GYSAM = Building Permits YoY%, SA, monthly
    # All from OECD MEI new-format series on FRED, confirmed through 2025-2026.

    # ── EUR confidence & permits ───────────────────────────────────────────────
    # Consumer Confidence EUR monthly (confirmed through Jan 2026)
    "CSCICP02EZM460S": {
        "event": "Consumer Confidence", "currency": "EUR", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False, "direct_pct": True,
    },
    # Building Permits EUR quarterly (confirmed through Q3 2025)
    "ODCNPI03EZQ657S": {
        "event": "Building Permits (QoQ)", "currency": "EUR", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
    },

    # ── GBP confidence & permits ───────────────────────────────────────────────
    # Consumer Confidence GBP monthly (confirmed through Feb 2026)
    "CSCICP02GBM460S": {
        "event": "Consumer Confidence", "currency": "GBP", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False, "direct_pct": True,
    },
    # Business Confidence GBP quarterly (confirmed through Q4 2025)
    "BSCICP02GBQ460S": {
        "event": "Business Confidence", "currency": "GBP", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
    },

    # ── JPY confidence & permits ───────────────────────────────────────────────
    # Consumer Confidence JPY monthly (confirmed through Feb 2026)
    "CSCICP02JPM460S": {
        "event": "Consumer Confidence", "currency": "JPY", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False, "direct_pct": True,
    },
    # Business Confidence JPY quarterly (confirmed through Q4 2025)
    "JPNBSCICP02STSAQ": {
        "event": "Business Confidence", "currency": "JPY", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
    },

    # ── AUD confidence & permits ───────────────────────────────────────────────
    # Consumer Confidence AUD monthly (confirmed through Feb 2026)
    "CSCICP02AUM460S": {
        "event": "Consumer Confidence", "currency": "AUD", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False, "direct_pct": True,
    },
    # Building Permits AUD monthly YoY% (confirmed through Dec 2025)
    "AUSODCNPI03GYSAM": {
        "event": "Building Permits (YoY)", "currency": "AUD", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False, "direct_pct": True,
    },

    # ── CAD confidence & permits ───────────────────────────────────────────────
    # Consumer Confidence CAD monthly (try — may be quarterly only)
    "CSCICP02CAM460S": {
        "event": "Consumer Confidence", "currency": "CAD", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False, "direct_pct": True,
    },
    # Building Permits CAD monthly YoY% (confirmed through Dec 2025)
    "CANODCNPI03GYSAM": {
        "event": "Building Permits (YoY)", "currency": "CAD", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False, "direct_pct": True,
    },

    # ── CHF confidence & permits ───────────────────────────────────────────────
    # Business Confidence CHF monthly (confirmed through Feb 2026)
    "BSCICP02CHM460S": {
        "event": "Business Confidence", "currency": "CHF", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False, "direct_pct": True,
    },
    # Consumer Confidence CHF monthly (try)
    "CSCICP02CHM460S": {
        "event": "Consumer Confidence", "currency": "CHF", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False, "direct_pct": True,
    },
    # Building Permits CHF monthly YoY% (try)
    "CHEODCNPI03GYSAM": {
        "event": "Building Permits (YoY)", "currency": "CHF", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": False, "direct_pct": True,
    },

    # ── NZD confidence & permits ───────────────────────────────────────────────
    # Business Confidence NZD quarterly (confirmed through Q4 2025)
    "BSCICP02NZQ460S": {
        "event": "Business Confidence", "currency": "NZD", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
    },
    # Consumer Confidence NZD quarterly (try)
    "CSCICP02NZQ460S": {
        "event": "Consumer Confidence", "currency": "NZD", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False,
        "quarterly": True, "direct_pct": True,
    },

    # ── USD additional series ──────────────────────────────────────────────────
    # Conference Board Consumer Confidence (monthly, through Apr 2026)
    "CONCCONF": {
        "event": "CB Consumer Confidence", "currency": "USD", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False,
    },
    # ADP National Employment monthly
    "ADPWNUSNERSA": {
        "event": "ADP Employment Change", "currency": "USD", "impact": "high",
        "unit": "K", "as_change": True, "pct_change": False, "is_inverse": False,
    },
}

# ── Release lag in days after reference period end ────────────────────────────
RELEASE_LAG = {
    "monthly":   45,   # ~6 weeks after month end
    "quarterly": 60,   # ~2 months after quarter end
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt(value: float, unit: str, meta: dict) -> str:
    """Format a numeric value consistently."""
    if unit == "K":
        return f"{value:.0f}K"
    elif unit == "B":
        return f"{value:.1f}B"
    elif unit == "%":
        return f"{value:.1f}%"
    else:
        return f"{value:.1f}"


def _release_date(obs_date_str: str, meta: dict) -> str:
    """
    Given the FRED observation date (period start), estimate the realistic
    release date by adding the release lag after period end.
    """
    dt = datetime.strptime(obs_date_str, "%Y-%m-%d")
    freq = "quarterly" if meta.get("quarterly") else "monthly"

    if freq == "quarterly":
        month = dt.month + 3
        year  = dt.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        quarter_end = datetime(year, month, 1) - timedelta(days=1)
        release_dt  = quarter_end + timedelta(days=RELEASE_LAG["quarterly"])
    else:
        if dt.month == 12:
            month_end = datetime(dt.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = datetime(dt.year, dt.month + 1, 1) - timedelta(days=1)
        release_dt = month_end + timedelta(days=RELEASE_LAG["monthly"])

    today = datetime.now(timezone.utc).replace(tzinfo=None)
    release_dt = min(release_dt, today)
    return release_dt.strftime("%Y-%m-%d")


def _norm_event(s: str) -> str:
    """Normalise event name for dedup matching against ForexFactory names."""
    s = re.sub(r'\s*\([A-Z][a-z]{2}\)$', '', s)
    s = re.sub(r'\s*\(Q[1-4]\)$', '', s)
    s = re.sub(r'\s*\([A-Z][a-z]{2}\s+[0-9]+\)$', '', s)
    return s.strip().lower()


# ── FRED API ──────────────────────────────────────────────────────────────────

def fred_observations(series_id: str, start_date: str) -> list[dict]:
    """Fetch all FRED observations from start_date. Returns [] on any error."""
    if not FRED_API_KEY:
        print(f"  WARNING: FRED_API_KEY not set — skipping {series_id}")
        return []

    params = {
        "series_id":         series_id,
        "api_key":           FRED_API_KEY,
        "file_type":         "json",
        "observation_start": start_date,
        "sort_order":        "asc",
        "limit":             "1000",
    }
    try:
        r = requests.get(FRED_BASE, params=params, headers=HEADERS, timeout=FETCH_TIMEOUT)
        if r.status_code == 400:
            print(f"  WARNING: FRED {series_id} HTTP 400 — series not on public API, skipping.")
            return []
        if r.status_code == 404:
            print(f"  WARNING: FRED {series_id} not found (404), skipping.")
            return []
        if r.status_code != 200:
            print(f"  WARNING: FRED {series_id} HTTP {r.status_code}, skipping.")
            return []
        obs = r.json().get("observations", [])
        return [
            {"date": o["date"], "value": float(o["value"])}
            for o in obs
            if o.get("value") not in (".", "", None)
        ]
    except Exception as e:
        print(f"  WARNING: FRED {series_id}: {e}")
        return []
    finally:
        time.sleep(RATE_LIMIT_SLEEP)


# ── Build events from FRED ────────────────────────────────────────────────────

def build_fred_events(start_date: str) -> list[dict]:
    """Fetch all FRED series and convert to calendar.json event format."""
    now_utc   = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    events    = []
    skipped   = 0
    fetched   = 0

    for series_id, meta in FRED_SERIES.items():
        ccy    = meta["currency"]
        ename  = meta["event"]
        impact = meta["impact"]
        unit   = meta["unit"]

        print(f"  Fetching FRED {series_id:35} [{ccy}] {ename}")

        obs = fred_observations(series_id, start_date)
        if not obs:
            skipped += 1
            continue

        fetched += 1
        as_change  = meta.get("as_change", False)
        pct_change = meta.get("pct_change", False)
        index_qoq  = meta.get("index_qoq", False)
        direct_pct = meta.get("direct_pct", False)
        scale_b    = meta.get("scale_b", False)
        scale_k    = meta.get("scale_k", False)

        for i, o in enumerate(obs):
            obs_date = o["date"]
            value    = o["value"]

            if obs_date > today_str:
                continue

            forecast_str = None

            # ── Compute derived value ──────────────────────────────────────
            if direct_pct:
                # Series already returns the % we want (e.g. YoY CPI, GDP %)
                if i > 0:
                    forecast_str = _fmt(obs[i - 1]["value"], unit, meta)

            elif index_qoq:
                # QoQ % change from consecutive level/index values
                if i == 0:
                    continue
                prev_val = obs[i - 1]["value"]
                if prev_val == 0:
                    continue
                value = round((value - prev_val) / abs(prev_val) * 100, 2)
                if i >= 2:
                    pp = obs[i - 2]["value"]
                    if pp != 0:
                        prev_change = round((obs[i - 1]["value"] - pp) / abs(pp) * 100, 2)
                        forecast_str = _fmt(prev_change, unit, meta)

            elif as_change:
                if i == 0:
                    continue
                prev_val = obs[i - 1]["value"]
                if pct_change:
                    if prev_val == 0:
                        continue
                    value = round((value - prev_val) / abs(prev_val) * 100, 2)
                else:
                    value = round(value - prev_val, 3)
                if series_id == "PAYEMS":
                    value = round(value, 1)
                if i >= 2:
                    pp2 = obs[i - 2]["value"]
                    pp1 = obs[i - 1]["value"]
                    if pct_change:
                        if pp2 != 0:
                            prev_change = round((pp1 - pp2) / abs(pp2) * 100, 2)
                            forecast_str = _fmt(prev_change, unit, meta)
                    else:
                        forecast_str = _fmt(round(pp1 - pp2, 3), unit, meta)

            else:
                # Level series — apply scaling if needed
                if scale_b:
                    value = value / 1e9
                elif scale_k:
                    value = value / 1e3
                if i > 0:
                    prev = obs[i - 1]["value"]
                    if scale_b:
                        prev = prev / 1e9
                    elif scale_k:
                        prev = prev / 1e3
                    forecast_str = _fmt(prev, unit, meta)

            date_iso = _release_date(obs_date, meta)

            if date_iso > today_str:
                continue

            actual_str = _fmt(value, unit, meta)

            try:
                display_date = datetime.strptime(date_iso, "%Y-%m-%d").strftime("%-d %b")
            except (ValueError, AttributeError):
                display_date = date_iso

            events.append({
                "date":     display_date,
                "dateISO":  date_iso,
                "timeUTC":  "12:30",
                "country":  ccy,
                "currency": ccy,
                "flag":     FLAG_MAP.get(ccy, ""),
                "event":    ename,
                "impact":   impact,
                "actual":   actual_str,
                "forecast": forecast_str,
                "previous": forecast_str,
                "source":   "FRED",
            })

    print(f"\n  FRED fetch complete: {fetched} series fetched, {skipped} skipped")
    events.sort(key=lambda e: (e["dateISO"], e["currency"], e["event"]))
    return events


# ── Load / save calendar.json ─────────────────────────────────────────────────

def load_calendar() -> list[dict]:
    if not os.path.exists(OUTPUT_PATH):
        print(f"  INFO: {OUTPUT_PATH} not found — will create from scratch.")
        return []
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            return json.load(f).get("events", [])
    except Exception as e:
        print(f"  WARNING: Could not read {OUTPUT_PATH} — {e}")
        return []


def save_calendar(events: list[dict], now_utc: datetime) -> None:
    events_sorted = sorted(
        events,
        key=lambda e: (e.get("dateISO", ""), e.get("timeUTC", ""), e.get("currency", ""))
    )

    all_dates   = [e["dateISO"] for e in events_sorted if e.get("dateISO")]
    range_from  = min(all_dates) if all_dates else ""
    range_to    = max(all_dates) if all_dates else ""
    ccy_dist    = dict(Counter(e.get("currency", "") for e in events_sorted))
    impact_dist = dict(Counter(e.get("impact", "") for e in events_sorted))

    output = {
        "lastUpdate":     now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generatedAt":    now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetchMode":      "FRED backfill + ForexFactory rolling",
        "timezoneNote":   "All times UTC",
        "status":         "ok",
        "source":         "FRED / ForexFactory",
        "errorMessage":   None,
        "fetchErrors":    [],
        "rangeFrom":      range_from,
        "rangeTo":        range_to,
        "totalEvents":    len(events_sorted),
        "currencyCounts": ccy_dist,
        "impactCounts":   impact_dist,
        "events":         events_sorted,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_json = json.dumps(output, ensure_ascii=False, indent=2)
    json.loads(output_json)  # validate before writing

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_json)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    force_overwrite = "--force" in sys.argv

    now_utc   = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] backfill_economic_calendar.py v5.0")
    print(f"  MAX_HISTORY_DAYS: {MAX_HISTORY_DAYS}")
    print(f"  Force-overwrite FRED events: {force_overwrite}")
    print(f"  Total FRED series configured: {len(FRED_SERIES)}")

    series_by_ccy = Counter(v["currency"] for v in FRED_SERIES.values())
    print(f"  Series per currency: {dict(sorted(series_by_ccy.items()))}")

    if not FRED_API_KEY:
        print("\n  ERROR: FRED_API_KEY environment variable not set.")
        print("  Register for a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  Then run: FRED_API_KEY=<your_key> python scripts/backfill_economic_calendar.py")
        sys.exit(1)

    # ── Backfill window ────────────────────────────────────────────────────────
    cutoff_old = (now_utc - timedelta(days=MAX_HISTORY_DAYS)).strftime("%Y-%m-%d")
    cutoff_new = (now_utc - timedelta(days=2)).strftime("%Y-%m-%d")
    fred_start = (now_utc - timedelta(days=MAX_HISTORY_DAYS + 90)).strftime("%Y-%m-%d")

    print(f"  Backfill window: {cutoff_old} → {cutoff_new}")
    print(f"  FRED fetch start: {fred_start}\n")

    # ── Step 1: Load existing calendar ────────────────────────────────────────
    existing_events = load_calendar()
    print(f"  Existing calendar.json: {len(existing_events)} events")

    protected_keys: set[tuple] = set()
    for ev in existing_events:
        if ev.get("actual"):
            source = ev.get("source", "")
            if force_overwrite and source in ("FRED", "OECD"):
                continue
            key = (
                ev.get("currency", ""),
                ev.get("dateISO", ""),
                _norm_event(ev.get("event", "")),
            )
            protected_keys.add(key)

    print(f"  Protected events (will not be overwritten): {len(protected_keys)}")

    if force_overwrite:
        before = len(existing_events)
        existing_events = [
            e for e in existing_events
            if e.get("source", "") not in ("FRED", "OECD")
        ]
        removed = before - len(existing_events)
        print(f"  --force: removed {removed} existing FRED/OECD events for re-injection\n")

    # ── Step 2: Fetch from FRED ────────────────────────────────────────────────
    print(f"  Fetching FRED series ({len(FRED_SERIES)} total)...\n")
    fred_events = build_fred_events(fred_start)
    print(f"  FRED raw events generated: {len(fred_events)}")

    # ── Step 3: Filter to backfill window and deduplicate ─────────────────────
    injected   = 0
    duplicates = 0
    out_window = 0

    for ev in fred_events:
        date_iso = ev.get("dateISO", "")

        if date_iso < cutoff_old or date_iso > cutoff_new:
            out_window += 1
            continue

        key = (
            ev.get("currency", ""),
            date_iso,
            _norm_event(ev.get("event", "")),
        )

        if key in protected_keys:
            duplicates += 1
            continue

        existing_events.append(ev)
        protected_keys.add(key)
        injected += 1

    print(f"\n  Backfill results:")
    print(f"    Injected:      {injected} new events")
    print(f"    Duplicates:    {duplicates} (already in calendar.json — preserved)")
    print(f"    Out of window: {out_window} (outside {cutoff_old}–{cutoff_new})")

    if injected == 0 and not force_overwrite:
        print("\n  INFO: Nothing to inject — calendar.json already has full coverage.")
        print("  Run with --force to re-inject FRED events (e.g. to fix corrupted data).")
        print("  No changes written.")
        sys.exit(0)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    save_calendar(existing_events, now_utc)

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    all_dates    = [e["dateISO"] for e in existing_events if e.get("dateISO")]
    range_from   = min(all_dates) if all_dates else "—"
    range_to     = max(all_dates) if all_dates else "—"
    with_actuals = sum(1 for e in existing_events if e.get("actual") not in (None, ""))
    in_90d       = sum(
        1 for e in existing_events
        if e.get("actual") and
        e.get("dateISO", "") >= (now_utc - timedelta(days=90)).strftime("%Y-%m-%d")
    )

    by_ccy = Counter(
        e.get("currency", "") for e in existing_events
        if e.get("actual") not in (None, "")
    )

    print(f"\n  {'=' * 45}")
    print(f"    ECONOMIC CALENDAR BACKFILL SUMMARY v5.0")
    print(f"  {'=' * 45}")
    print(f"  Total events:         {len(existing_events)}")
    print(f"  With actuals:         {with_actuals}")
    print(f"  In 90d window:        {in_90d}")
    print(f"  Injected this run:    {injected}")
    print(f"  Date range:           {range_from} → {range_to}")
    print(f"  Coverage by currency: {dict(sorted(by_ccy.items()))}")
    print(f"  {'=' * 45}")

    # Per-currency series count for diagnostics
    print(f"\n  FRED series configured per currency:")
    for ccy in sorted(set(v["currency"] for v in FRED_SERIES.values())):
        count = sum(1 for v in FRED_SERIES.values() if v["currency"] == ccy)
        events_injected = by_ccy.get(ccy, 0)
        print(f"    {ccy}: {count} series configured → {events_injected} events with actuals")

    print(f"\n✓ Backfill complete. Run update-economic-calendar.yml to continue rolling accumulation.")


if __name__ == "__main__":
    main()
