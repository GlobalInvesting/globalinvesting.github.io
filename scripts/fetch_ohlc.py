#!/usr/bin/env python3
"""
fetch_ohlc.py  v1.9 — Daily OHLC history for Lightweight Charts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Downloads 3 years of daily OHLC bars via yfinance for all symbols
used by the Lightweight Charts panel (replaces TradingView widget).

Symbols covered:
  FX Majors   (7): EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, NZD/USD
  FX Crosses (21): all pairs in PAIRS[] that have chart click support
  Cross-asset (4): Gold (GC=F), WTI (CL=F), BTC (BTC-USD), US 10Y (^TNX)
  Indices     (4): S&P 500 (^GSPC), Nasdaq 100 (^NDX), Nikkei 225 (^N225), EuroStoxx 50 (^STOXX50E)
  Crypto      (1): ETH/USD (ETH-USD)
  FX Index    (1): DXY (DX-Y.NYB)

Output:
  ohlc-data/{id}.json   — [{time, open, high, low, close}] newest-last
  ohlc-data/meta.json   — {updated_at, symbol_count, errors:[]}

Each file is REPLACED on every run (not appended) — repo stays fixed size.
Total size estimate: ~38 files × ~45 KB raw = ~1.7 MB, ~560 KB gzipped.

Schedule: daily at 22:30 UTC (after NY close, before Sydney open).
          Runs in the ENGINE private repo — pushes output to PUBLIC site repo.

Data integrity — FX daily bar construction (v1.3):
  FX symbols:     Daily bars are BUILT from 1H bars aggregated over the FX trading day
                  (17:00 NY → 17:00 NY next day, i.e. 21:00 UTC → 21:00 UTC).
                  This is the industry-standard session boundary used by Bloomberg,
                  TradingView (FX daily charts), and Reuters for all FX pairs.

                  Why 1H aggregation instead of Yahoo's native 1D bars:
                    • Yahoo's native 1D FX bars use a UTC midnight cutoff (00:00–24:00 UTC),
                      which does NOT correspond to any real FX session boundary. The result
                      is that H/L values are materially wrong: a breakout that occurs between
                      17:00 and 00:00 UTC appears in the prior day's bar; a breakout between
                      00:00 and 17:00 UTC appears in the next day's bar. These H/L errors
                      cause candle wicks to be systematically too short or too long.
                    • Yahoo's native 1D FX open field stores the last tick of the prior UTC
                      day (not the real 17:00 NY open), causing candle bodies to contradict
                      the actual day's direction in ~40–50% of bars.
                    • Aggregating 1H bars from 21:00 UTC to 21:00 UTC (17:00 NY session)
                      captures the correct intraday range: open = first 1H bar open, high =
                      max of all 1H highs, low = min of all 1H lows, close = last 1H close.
                      This matches the H/L shown on Yahoo Finance's own 4H and 1H charts,
                      TradingView daily FX bars, and Bloomberg FX daily candles.

  Non-FX (all):   Native yfinance 1D bars — Gold, WTI, SPX, Nasdaq, Stoxx, Nikkei,
                  US10Y, VIX, BTC, ETH, DXY all use the same path as WTI (always worked).
                  1H aggregation was previously used for Gold (CME stub-bar quality) and
                  selected equity indices, but any fixed UTC session boundary causes the
                  session guard to discard today’s bar at the 22:30 UTC run (session_date
                  today > run_date_utc yesterday). Native 1D already contains the
                  in-progress today bar at 22:30 UTC for all non-FX sessions. The intraday
                  today-bar injection in dashboard.js updates it with live data from
                  quotes.json — the proven pattern WTI has used since launch.
                  Gold stub-bar quality (prev. reason for 1H): accepted tradeoff. The
                  intraday injection overwrites today’s bar with real H/L from yfinance.
  DXY only:       HL_MAX_SPREAD guard drops bars with impossible intraday ranges
  Nasdaq:         Uses ^NDX (Nasdaq 100) to match the CFI:US100 chart tab; ^IXIC
                  (Composite) has different constituents and price levels (~19k vs ~5.8k).

yfinance 1H window — 720-day limit (v1.9):
  yfinance enforces a hard 730-day limit for 1H bars, evaluated in Yahoo's server
  timezone (America/New_York, UTC-4 in summer). When the script runs before midnight
  NY time (i.e. before 04:00 UTC), the NY date is still "yesterday". A start window
  of `now_utc - 729 days` expressed as a date string can land exactly on the 730-day
  boundary as seen from NY, triggering:
    "1h data not available ... The requested range must be within the last 730 days."
  When this happens yfinance silently falls through to the native 1D path for the
  entire 1H request, producing 350-425 structurally invalid bars per FX pair (vs the
  normal ~120-155 from the known open-tick artifact) — the root cause of the nightly
  FX candle bug that persisted through v7.74.35.
  Fix (v1.9): use `now_utc - 720 days` (9-day safety margin). The 10-day gap in
  1H coverage (days 720-730) is seamlessly covered by the Part B native 1D bars,
  which always cover the full 3-year period. No visible impact on chart quality.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────

SITE_DIR = Path(os.environ.get("SITE_DIR", "."))   # public repo root (workflow sets SITE_DIR=.)
OUT_DIR  = SITE_DIR / "ohlc-data"

# 3 years of daily bars (trading days only, ~756 bars)
PERIOD = "3y"
INTERVAL = "1d"

# Symbol map: dashboard ID → yfinance ticker
SYMBOLS: dict[str, str] = {
    # FX Majors
    "eurusd": "EURUSD=X",
    "gbpusd": "GBPUSD=X",
    "usdjpy": "USDJPY=X",
    "audusd": "AUDUSD=X",
    "usdcad": "USDCAD=X",
    "usdchf": "USDCHF=X",
    "nzdusd": "NZDUSD=X",
    # FX Crosses
    "eurgbp": "EURGBP=X",
    "eurjpy": "EURJPY=X",
    "eurchf": "EURCHF=X",
    "eurcad": "EURCAD=X",
    "euraud": "EURAUD=X",
    "eurnzd": "EURNZD=X",
    "gbpjpy": "GBPJPY=X",
    "gbpchf": "GBPCHF=X",
    "gbpcad": "GBPCAD=X",
    "gbpaud": "GBPAUD=X",
    "gbpnzd": "GBPNZD=X",
    "audjpy": "AUDJPY=X",
    "audnzd": "AUDNZD=X",
    "audchf": "AUDCHF=X",
    "audcad": "AUDCAD=X",
    "cadjpy": "CADJPY=X",
    "cadchf": "CADCHF=X",
    "nzdjpy": "NZDJPY=X",
    "nzdcad": "NZDCAD=X",
    "nzdchf": "NZDCHF=X",
    "chfjpy": "CHFJPY=X",
    # Cross-asset
    "gold":  "GC=F",       # Gold front-month futures (CME); XAUUSD=X is spot — GC=F matches institutional convention
    "wti":   "CL=F",       # WTI Crude front-month futures
    "btc":   "BTC-USD",    # Bitcoin vs USD
    "us10y": "^TNX",       # US 10-Year Treasury yield
    "us5y":  "^FVX",       # US 5-Year Treasury yield
    "us2y":  "^IRX",       # US 2-Year Treasury yield (^IRX = 13-week T-Bill, best available proxy)
    # Equity indices
    "spx":    "^GSPC",     # S&P 500
    "nasdaq": "^NDX",       # Nasdaq 100 index — matches CFI:US100 chart tab; ^IXIC (Composite) has different levels (~19k vs ~5.8k)
    "nikkei": "^N225",     # Nikkei 225
    "stoxx":  "^STOXX50E", # EuroStoxx 50
    "vix":    "^VIX",      # CBOE Volatility Index
    "move":   "^MOVE",     # ICE BofA MOVE Index — bond market volatility (Treasury options)
    # Crypto
    "eth":   "ETH-USD",    # Ethereum vs USD
    # FX Index
    "dxy":   "DX-Y.NYB",   # US Dollar Index (ICE futures)
    # Additional instruments
    "silver": "SI=F",       # Silver front-month futures (CME)
    "brent":  "BZ=F",       # Brent Crude front-month futures (ICE)
    "dax":    "^GDAXI",     # DAX Performance Index (Frankfurt)
    "ftse":   "^FTSE",      # FTSE 100 Index (London)
    "hsi":    "^HSI",       # Hang Seng Index (Hong Kong)
    "dji":    "^DJI",       # Dow Jones Industrial Average
}

# Decimal precision per symbol (for display only — not stored in OHLC)
DECIMALS: dict[str, int] = {
    "eurusd": 5, "gbpusd": 5, "audusd": 5, "usdchf": 5, "usdcad": 5,
    "nzdusd": 5, "eurgbp": 5, "eurchf": 5, "eurcad": 5, "euraud": 5,
    "eurnzd": 5, "gbpchf": 5, "gbpcad": 5, "gbpaud": 5, "gbpnzd": 5,
    "audnzd": 5, "audchf": 5, "audcad": 5, "cadchf": 5, "nzdcad": 5,
    "nzdchf": 5,
    "usdjpy": 3, "eurjpy": 3, "gbpjpy": 3, "audjpy": 3, "cadjpy": 3,
    "nzdjpy": 3, "chfjpy": 3,
    "gold":  2, "wti": 2, "btc": 2, "us10y": 4, "us5y": 4, "us2y": 4,
    "spx": 2, "nasdaq": 2, "nikkei": 2, "stoxx": 2, "vix": 2,
    "move": 2,
    "eth": 2, "dxy": 3,
    "silver": 2, "brent": 2, "dax": 2, "ftse": 2, "hsi": 2, "dji": 2,
}

# Plausibility guards — reject bars outside these ranges
GUARDS: dict[str, tuple[float, float]] = {
    "gold":  (500.0,   8000.0),
    "wti":   (10.0,    300.0),
    "btc":   (100.0,   500000.0),
    "us10y": (0.01,    25.0),
    "us5y":  (0.01,    25.0),
    "us2y":  (0.01,    25.0),
    "spx":   (500.0,   15000.0),
    "nasdaq":(1000.0,  30000.0),  # ^NDX (Nasdaq 100): historical range ~1k to ~22k; upper headroom for growth
    "nikkei":(5000.0,  80000.0),
    "stoxx": (1000.0,  8000.0),
    "vix":   (5.0,     90.0),      # VIX historically ranges 5-90
    "move":  (20.0,    400.0),     # MOVE Index historically ranges ~30–200; headroom for spikes
    "eth":   (10.0,    20000.0),
    "dxy":   (60.0,    150.0),
    "silver": (5.0,    500.0),
    "brent":  (10.0,   300.0),
    "dax":    (3000.0, 30000.0),
    "ftse":   (2000.0, 12000.0),
    "hsi":    (5000.0, 60000.0),
    "dji":    (5000.0, 70000.0),
    # JPY pairs — quoted in yen, must not use FX_GUARD (0.1–50)
    "usdjpy":(70.0,    300.0),
    "eurjpy":(100.0,   400.0),
    "gbpjpy":(100.0,   400.0),
    "audjpy":(50.0,    200.0),
    "cadjpy":(50.0,    200.0),
    "nzdjpy":(40.0,    200.0),
    "chfjpy":(80.0,    300.0),
}
FX_GUARD = (0.1, 50.0)   # applies to non-JPY FX pairs

# Maximum tolerated H/L spread as a fraction of Low, per symbol.
# yfinance DX-Y.NYB occasionally produces bars that span two contract months at
# roll dates, creating an impossible intraday range. DXY does not apply Panama
# back-adjustment, so the guard remains as a hard filter.
#
# Gold (GC=F) and WTI (CL=F) previously had guards here (4% and 8% respectively),
# but those were removed in v7.47.19 because:
#   1. The Panama back-adjustment (applied below) already corrects inter-bar roll
#      gaps — the primary artifact. A single bar with a wide H/L is NOT a roll
#      artifact; it is a real volatile trading session.
#   2. During high-volatility geopolitical events (e.g. Strait of Hormuz crisis,
#      2026) WTI legitimately traded with 8–15% daily ranges and Gold with 4–6%.
#      The old guards silently dropped these bars, creating multi-day gaps in the
#      chart (March 2026: 14 of 22 WTI bars dropped; Jan–Feb 2026: 13 Gold bars
#      dropped). The result was a sparse, visually broken candlestick chart.
#   3. DXY genuinely does not move > 5% in a single session — the guard is safe
#      there and catches actual roll artifacts without collateral damage.
HL_MAX_SPREAD: dict[str, float] = {
    "dxy":   0.05,   # 5% — DX futures roll artifacts (no Panama applied to DXY)
}

# FX session boundary — industry standard (Bloomberg, TradingView, Reuters):
# The FX trading day runs from 17:00 New York time to 17:00 New York time next day.
# New York is UTC-5 (EST) or UTC-4 (EDT). Because we always need 17:00 NY in UTC:
#   EST (Nov–Mar):  17:00 NY = 22:00 UTC
#   EDT (Mar–Nov):  17:00 NY = 21:00 UTC
# We use a fixed 21:00 UTC cutoff. During EST this is 16:00 NY (1h early) which
# occasionally misses the last hour, but is the safe choice accepted across the
# industry when DST transitions are not individually tracked. TradingView and
# Bloomberg both use a fixed 21:00 UTC daily cutoff for FX daily bar construction.
#
# Yahoo's native 1D FX bars use 00:00–24:00 UTC — NOT a real FX session boundary.
# This causes H/L values to be materially wrong and open values to be unreliable.
# Fix: download 1H bars and aggregate them over the 21:00 UTC → 21:00 UTC window.
# Result: open/high/low/close exactly match what Yahoo Finance shows on its own
# 4H chart, and match TradingView and Bloomberg FX daily candles.
NON_FX_SYMBOLS = {'wti', 'btc', 'us10y', 'us5y', 'us2y', 'spx', 'nasdaq', 'nikkei', 'stoxx', 'vix', 'move', 'eth', 'dxy', 'gold', 'silver', 'brent', 'dax', 'ftse', 'hsi', 'dji'}
# Equity indices routed to fetch_equity_ohlc_from_1h (1H aggregation over 21:00 UTC boundary).
# These instruments close well before the 22:30 UTC workflow run, so their 1H bars are fully
# available. Nikkei: TSE closes ~06:00 UTC. EuroStoxx: Euronext closes ~15:30 UTC.
# EuroStoxx was moved here because yfinance native-1D has a data-availability lag for
# European exchanges when the workflow runs at 22:30 UTC (= next calendar day in CEST),
# causing the most recent session bar to be excluded from period="3y" results.
EQUITY_1H_SYMBOLS = {'nikkei', 'stoxx'}
# Symbols routed to fetch_gold_ohlc_from_1h (CME 22:00/23:00 UTC session boundary).
# Gold is pulled OUT of native 1D because yfinance 1D for GC=F uses UTC midnight
# as its cutoff: the CME session runs 22:00/23:00 UTC → next day 21:00/22:00 UTC,
# spanning two UTC calendar days. Native 1D therefore produces bars where
# open ≈ prev_close (no real session boundary gap) and H/L miss the first hour of
# trading (23:00–00:00 UTC). 1H aggregation over the CME boundary produces
# correct session opens, proper H/L ranges, and real overnight gaps — matching
# Bloomberg and TradingView CME Gold daily candles.
CME_SYMBOLS = {'gold', 'wti', 'dxy'}
#
# Routing rationale — three paths:
#   FX pairs (28):    1H aggregation over the 21:00 UTC NY session boundary.
#                     Required because yfinance native 1D FX bars use UTC midnight as
#                     their cutoff, producing materially wrong H/L and unreliable opens.
#                     1H aggregation matches Bloomberg, TradingView and Reuters FX candles.
#
#   CME/ICE instruments: 1H aggregation over the DST-aware session boundary.
#   (gold, wti, dxy)  Gold/WTI: CME boundary = 17:00 ET (21:00 UTC EDT / 22:00 UTC EST).
#                     DXY: ICE boundary = 17:00 ET — identical to CME in practice.
#                     Both WTI and DXY were previously on native 1D which used UTC
#                     midnight as its cutoff, producing bars with wrong session opens,
#                     compressed H/L, and "ayer y hoy" mismatches vs TradingView.
#                     Same hybrid approach as FX and Gold: 1H for recent 730 days,
#                     native 1D for older history. dashboard.js strips the last bar
#                     and injects the live today-bar from quotes.json.
#
#   Non-FX (8):       Native yfinance 1D — BTC, ETH, US10Y, SPX, Nasdaq, Stoxx,
#                     Nikkei (via equity 1H), VIX. Their session boundaries coincide
#                     closely enough with UTC midnight that native 1D is adequate.
FX_SYMBOLS = set(SYMBOLS.keys()) - NON_FX_SYMBOLS

# Crypto trades 24/7 — yfinance only returns Saturday/Sunday bars when an explicit
# start/end date range is passed. Using period="3y" silently omits weekends because
# yfinance applies a NYSE/NASDAQ business-day calendar as the default date filter.
CRYPTO_SYMBOLS = {'btc', 'eth'}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _guard(id_: str, val: float) -> bool:
    lo, hi = GUARDS.get(id_, FX_GUARD)
    return lo <= val <= hi


# ── FX 1H → daily aggregation ───────────────────────────────────────────────

def fetch_fx_ohlc_from_1h(id_: str, ticker_sym: str) -> list[dict] | None:
    """
    Build FX daily bars by aggregating 1H bars over the industry-standard FX
    session boundary: 21:00 UTC → 21:00 UTC (≈ 17:00 New York → 17:00 New York).

    This produces H/L values that match Yahoo Finance's own 4H chart, TradingView
    daily FX candles, and Bloomberg FX daily bars — all of which use the NY session
    boundary, NOT a UTC midnight cutoff.

    yfinance returns 1H data for up to 730 days. Bars older than 730 days fall back
    to the native 1D endpoint with prev_close open correction. The crossover is
    seamless — only H/L differ, and the 1H portion (most recent ~2 years) is always
    the correct, faithful representation of real FX session ranges.
    """
    try:
        ticker = yf.Ticker(ticker_sym)
        dec    = DECIMALS.get(id_, 5)

        # ── PART A: 1H bars → aggregate to daily (FX session boundary 21:00 UTC) ──
        # 720-day window (not 729/730): yfinance enforces a 730-day hard limit evaluated
        # in NY time (UTC-4). Before 04:00 UTC the NY date is still yesterday — a 729-day
        # UTC window lands exactly on the 730-day NY boundary and triggers a silent fallback
        # to native 1D, producing ~400 clamped bars per pair vs the normal ~130. 9-day
        # safety margin; gap covered by Part B native 1D. (v1.9)
        _end_1h   = datetime.now(timezone.utc) + timedelta(days=1)
        _start_1h = datetime.now(timezone.utc) - timedelta(days=720)
        hist_1h = ticker.history(
            start=_start_1h.strftime("%Y-%m-%d"),
            end=_end_1h.strftime("%Y-%m-%d"),
            interval="1h",
            auto_adjust=True,
        )

        # Session-complete guard: only include a session bar if the session has closed.
        # FX session closes at 21:00 UTC. The scheduled run at 01:30 UTC runs well after
        # the close — session_date == run_date_utc is the fully-completed bar.
        # session_date > run_date_utc is the next partial session — discard.
        # If triggered via workflow_dispatch before 21:00 UTC (mid-session), today's
        # bucket is still open — discard it by treating yesterday as the effective cutoff.
        _now_utc = datetime.now(timezone.utc)
        _fx_session_closed = _now_utc.hour >= 21   # FX closes 21:00 UTC
        if _fx_session_closed:
            run_date_utc = (_end_1h - timedelta(days=1)).date()  # today UTC = last complete session
        else:
            run_date_utc = (_end_1h - timedelta(days=2)).date()  # yesterday UTC (today's session still open)

        day_buckets: dict[str, dict] = {}
        if not hist_1h.empty:
            for ts, row in hist_1h.iterrows():
                if ts.tzinfo is None:
                    ts_utc = ts.replace(tzinfo=timezone.utc)
                else:
                    ts_utc = ts.astimezone(timezone.utc)

                # Assign 1H bar to FX session date (session closes at 21:00 UTC)
                # Hours 00–20 belong to the session closing today (UTC date)
                # Hours 21–23 open a new session that closes tomorrow (UTC date)
                if ts_utc.hour < 21:
                    session_date = ts_utc.date()
                else:
                    session_date = (ts_utc + timedelta(days=1)).date()

                # Drop any 1H bar that belongs to a session not yet completed.
                # session_date == run_date_utc = today's session (complete — closed at 21:00 UTC).
                # session_date > run_date_utc = tomorrow's partial session — discard.
                if session_date > run_date_utc:
                    continue

                date_str = session_date.strftime("%Y-%m-%d")
                o = float(row["Open"])
                h = float(row["High"])
                l = float(row["Low"])
                c = float(row["Close"])

                if not (_guard(id_, c) and _guard(id_, o) and h >= l):
                    continue
                if o == h == l == c:
                    continue

                if date_str not in day_buckets:
                    # Include open in H/L so the session open price is always within range.
                    # yfinance 1H high/low reflect ticks WITHIN that hour, not the open tick
                    # itself. If the open is above the hour's high (gap-up open) or below
                    # the hour's low (gap-down open), ignoring it produces H < O or L > O —
                    # structurally impossible bars that render as deformed candles.
                    day_buckets[date_str] = {"open": o, "high": max(o, h), "low": min(o, l), "close": c}
                else:
                    b = day_buckets[date_str]
                    b["high"]  = max(b["high"], h)
                    b["low"]   = min(b["low"],  l)
                    b["close"] = c  # overwrite with the chronologically latest 1H close

        bars_1h: list[dict] = []
        for date_str in sorted(day_buckets.keys()):
            b = day_buckets[date_str]
            d = datetime.strptime(date_str, "%Y-%m-%d")
            if d.weekday() in (5, 6):  # Saturday, Sunday — FX closed
                continue
            if not _guard(id_, b["close"]):
                continue
            o_r = round(b["open"],  dec)
            c_r = round(b["close"], dec)
            # ── OHLC structural integrity clamp ──────────────────────────────
            # The bucket open = first 1H bar's Open (which yfinance FX 1H reports
            # as prev_session_close on most bars). On gap-down sessions the
            # prev_session_close can exceed all 1H highs accumulated during the day,
            # producing high < open — a structurally impossible candle. Clamping
            # guarantees H >= max(O,C) and L <= min(O,C) without discarding the
            # real intraday range. The same clamp is applied in _lwBuildTodayBar
            # in dashboard.js for the live today-bar.
            h_r = round(max(b["high"],  o_r, c_r), dec)
            l_r = round(min(b["low"],   o_r, c_r), dec)
            bars_1h.append({
                "time":   date_str,
                "open":   o_r,
                "high":   h_r,
                "low":    l_r,
                "close":  c_r,
                "volume": 0,  # 1H FX tick-volume from Yahoo is not meaningful; omit
            })

        # ── PART B: native 1D bars for period older than 730-day 1H limit ────────
        hist_1d = ticker.history(period=PERIOD, interval=INTERVAL, auto_adjust=True)
        bars_1d_old: list[dict] = []
        earliest_1h = bars_1h[0]["time"] if bars_1h else None

        if not hist_1d.empty:
            for ts, row in hist_1d.iterrows():
                date_str = ts.strftime("%Y-%m-%d")
                if earliest_1h and date_str >= earliest_1h:
                    continue  # covered by more-accurate 1H aggregation
                o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
                if not (_guard(id_, c) and _guard(id_, o) and h >= l and h >= c and l <= c):
                    continue
                if o == h == l == c:
                    continue
                vol = int(row["Volume"]) if "Volume" in row and not (row["Volume"] != row["Volume"]) else 0
                bars_1d_old.append({
                    "time":   date_str,
                    "open":   round(o, dec),
                    "high":   round(h, dec),
                    "low":    round(l, dec),
                    "close":  round(c, dec),
                    "volume": vol,
                })
            # Apply prev_close open correction to legacy 1D bars
            for i in range(1, len(bars_1d_old)):
                bars_1d_old[i]["open"] = bars_1d_old[i - 1]["close"]

        # ── Merge: legacy 1D + 1H-aggregated (chronological) ─────────────────────
        combined = bars_1d_old + bars_1h

        seen: set[str] = set()
        deduped: list[dict] = []
        for bar in reversed(combined):
            if bar["time"] not in seen:
                seen.add(bar["time"])
                deduped.append(bar)
        deduped.reverse()  # oldest to newest

        if len(deduped) < 30:
            print(f"  WARN [{id_}]: only {len(deduped)} valid bars after 1H aggregation - skipping")
            return None

        return deduped

    except Exception as exc:
        print(f"  ERROR [{id_}] (1H agg): {exc}")
        return None


# ── Gold CME 1H → daily aggregation ─────────────────────────────────────────

def fetch_gold_ohlc_from_1h(id_: str, ticker_sym: str) -> list[dict] | None:
    """
    Build Gold daily bars by aggregating 1H bars over the CME COMEX session boundary:
    22:00 UTC → 22:00 UTC (17:00 New York → 17:00 New York, EST fixed).

    CME Gold (GC=F) trades Sunday 18:00 ET to Friday 17:00 ET with a 60-minute daily
    maintenance break (17:00–18:00 ET). The session boundary for daily bar construction
    is 17:00 ET, which converts to:
      22:00 UTC during EDT (UTC−4, April–October)
      23:00 UTC during EST (UTC−5, November–March)

    The boundary is computed dynamically from the New York UTC offset to remain
    correct across DST transitions — avoiding the winter/summer mismatch that would
    cause bars at the boundary hour to be assigned to the wrong session.

    yfinance native 1D bars for GC=F are severely degraded because Yahoo constructs
    them using a UTC midnight cutoff, not the CME session boundary. This causes:
      - open ≈ prev_close on most bars (no real session gap visible)
      - H/L missing the first hour of each session (23:00–00:00 UTC)
    1H aggregation over the correct CME boundary produces faithful session OHLC.

    Hybrid approach (same as FX):
    - 1H bars cover the most recent 730 days (yfinance limit) with correct H/L.
    - Native 1D bars cover the period older than 730 days (adequate for historical context;
      the artifact rate is lower in older data since the pattern worsened with the 2024–2026
      gold volatility regime).
    - No prev_close open correction: Gold futures have a genuine electronic session open
      at 18:00 ET, not a continuous market like FX. The open from the first 1H bar of
      each session is the real market open — no correction needed.
    - No back-adjustment: roll gaps appear as-is (institutional standard for front-month
      futures where absolute price level matters for spread and carry analysis).
    """
    try:
        ticker = yf.Ticker(ticker_sym)
        dec = DECIMALS.get(id_, 2)

        # ── PART A: 1H bars → aggregate to daily (CME session boundary 22:00 UTC) ─
        _end_1h   = datetime.now(timezone.utc) + timedelta(days=1)
        _start_1h = datetime.now(timezone.utc) - timedelta(days=720)
        hist_1h = ticker.history(
            start=_start_1h.strftime("%Y-%m-%d"),
            end=_end_1h.strftime("%Y-%m-%d"),
            interval="1h",
            auto_adjust=True,
        )

        # ── DST-aware CME session boundary ────────────────────────────────────
        # CME Gold closes at 17:00 ET and reopens at 18:00 ET.
        # In EDT (UTC−4, Apr–Oct): close = 21:00 UTC, reopen = 22:00 UTC.
        # In EST (UTC−5, Nov–Mar): close = 22:00 UTC, reopen = 23:00 UTC.
        # Using a hardcoded boundary causes bars in the boundary hour to be
        # misassigned across DST transitions. Compute from the NY UTC offset.
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        _ny_now = datetime.now(ZoneInfo("America/New_York"))
        _ny_offset_h = int(_ny_now.utcoffset().total_seconds() / 3600)  # −4 EDT, −5 EST
        _cme_close_utc  = 17 + (-_ny_offset_h)   # 21 UTC in EDT, 22 UTC in EST
        _cme_reopen_utc = 18 + (-_ny_offset_h)   # 22 UTC in EDT, 23 UTC in EST

        _now_utc = datetime.now(timezone.utc)
        _gold_session_closed = _now_utc.hour >= _cme_close_utc
        if _gold_session_closed:
            run_date_utc = (_end_1h - timedelta(days=1)).date()  # today UTC = last complete session
        else:
            run_date_utc = (_end_1h - timedelta(days=2)).date()  # yesterday (today's session still open)

        day_buckets: dict[str, dict] = {}
        if not hist_1h.empty:
            for ts, row in hist_1h.iterrows():
                if ts.tzinfo is None:
                    ts_utc = ts.replace(tzinfo=timezone.utc)
                else:
                    ts_utc = ts.astimezone(timezone.utc)

                # CME session boundary: 17:00 ET (DST-aware UTC hour = _cme_reopen_utc).
                # Hours 00–(_cme_reopen_utc−1) belong to the session closing today.
                # Hours _cme_reopen_utc–23 open a new session that closes tomorrow.
                if ts_utc.hour < _cme_reopen_utc:
                    session_date = ts_utc.date()
                else:
                    session_date = (ts_utc + timedelta(days=1)).date()

                # Drop any 1H bar that belongs to a session not yet completed.
                # session_date == run_date_utc = today's session (complete — closed at 22:00 UTC).
                # session_date > run_date_utc = tomorrow's partial session — discard.
                if session_date > run_date_utc:
                    continue

                date_str = session_date.strftime("%Y-%m-%d")
                o = float(row["Open"])
                h = float(row["High"])
                l = float(row["Low"])
                c = float(row["Close"])

                if not (_guard(id_, c) and _guard(id_, o) and h >= l):
                    continue
                if o == h == l == c:
                    continue

                if date_str not in day_buckets:
                    # Include open in H/L: same fix as FX — yfinance 1H H/L excludes the
                    # open tick, so O can exceed H or fall below L on the first bar.
                    day_buckets[date_str] = {"open": o, "high": max(o, h), "low": min(o, l), "close": c}
                else:
                    b = day_buckets[date_str]
                    b["high"]  = max(b["high"], h)
                    b["low"]   = min(b["low"],  l)
                    b["close"] = c  # latest 1H bar close = session close

        bars_1h: list[dict] = []
        for date_str in sorted(day_buckets.keys()):
            b = day_buckets[date_str]
            d = datetime.strptime(date_str, "%Y-%m-%d")
            # CME Gold closes Friday 17:00 ET and reopens Sunday 18:00 ET.
            # Saturday has no trading session — drop it.
            if d.weekday() == 5:  # Saturday
                continue
            if not _guard(id_, b["close"]):
                continue
            o_r = round(b["open"],  dec)
            c_r = round(b["close"], dec)
            h_r = round(max(b["high"],  o_r, c_r), dec)  # clamp: H >= max(O,C)
            l_r = round(min(b["low"],   o_r, c_r), dec)  # clamp: L <= min(O,C)
            bars_1h.append({
                "time":   date_str,
                "open":   o_r,
                "high":   h_r,
                "low":    l_r,
                "close":  c_r,
                "volume": 0,
            })

        # ── PART B: native 1D bars for period older than 730-day 1H limit ────────
        hist_1d = ticker.history(period=PERIOD, interval=INTERVAL, auto_adjust=True)
        bars_1d_old: list[dict] = []
        earliest_1h = bars_1h[0]["time"] if bars_1h else None

        if not hist_1d.empty:
            for ts, row in hist_1d.iterrows():
                date_str = ts.strftime("%Y-%m-%d")
                if earliest_1h and date_str >= earliest_1h:
                    continue  # covered by more-accurate 1H aggregation
                o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
                if not (_guard(id_, c) and _guard(id_, o) and h >= l and h >= c and l <= c):
                    continue
                if o == h == l == c:
                    continue
                vol = int(row["Volume"]) if "Volume" in row and not (row["Volume"] != row["Volume"]) else 0
                bars_1d_old.append({
                    "time":   date_str,
                    "open":   round(o, dec),
                    "high":   round(h, dec),
                    "low":    round(l, dec),
                    "close":  round(c, dec),
                    "volume": vol,
                })
            # No prev_close open correction for legacy 1D — Gold futures open field
            # in older yfinance data is generally reliable (session open tick).

        # ── Merge: legacy 1D + 1H-aggregated (chronological) ─────────────────────
        combined = bars_1d_old + bars_1h

        seen: set[str] = set()
        deduped: list[dict] = []
        for bar in reversed(combined):
            if bar["time"] not in seen:
                seen.add(bar["time"])
                deduped.append(bar)
        deduped.reverse()  # oldest to newest

        if len(deduped) < 30:
            print(f"  WARN [{id_}]: only {len(deduped)} valid bars after 1H aggregation - skipping")
            return None

        return deduped

    except Exception as exc:
        print(f"  ERROR [{id_}] (gold 1H agg): {exc}")
        return None


# ── WTI crude oil + DXY dollar index: 1H → daily aggregation (CME/ICE boundary) ──

def fetch_wti_dxy_ohlc_from_1h(id_: str, ticker_sym: str) -> list[dict] | None:
    """
    Build WTI and DXY daily bars by aggregating 1H bars over the DST-aware CME/ICE
    session boundary (17:00 ET = 21:00 UTC in EDT / 22:00 UTC in EST).

    WTI (CME CL=F): trades Sunday 18:00 ET – Friday 17:00 ET. Session closes daily at
    17:00 ET. yfinance native 1D uses UTC midnight, producing bars that mix two
    consecutive sessions — open ≈ prev_close, H/L wrong relative to TradingView.

    DXY (ICE DX-Y.NYB): same 17:00 ET daily boundary. Same UTC-midnight artifact.
    At the 22:30 UTC OHLC run, native 1D for DXY returns a bar that starts at 00:00
    UTC (mid-session) instead of at the 22:00/23:00 UTC session open — the opens are
    systematically wrong and H/L exclude the early post-midnight session movement.

    Hybrid approach (identical to Gold):
    - 1H bars cover the most recent 730 days with correct session OHLC.
    - Native 1D bars cover history older than 730 days (legacy; artifact rate lower).
    - No prev_close open correction: futures have a genuine electronic session open.
    - No back-adjustment: roll gaps appear as-is (front-month convention).

    Weekend handling:
    - WTI: Saturday has no trading session → Saturday bars dropped.
    - DXY: Same — Saturday dropped.
    - Sunday bars (from 18:00 ET Sunday open) fold into Monday's session_date.
    """
    try:
        ticker = yf.Ticker(ticker_sym)
        dec = DECIMALS.get(id_, 2)

        # ── PART A: 1H bars → aggregate to daily (CME/ICE 17:00 ET boundary) ──
        _end_1h   = datetime.now(timezone.utc) + timedelta(days=1)
        _start_1h = datetime.now(timezone.utc) - timedelta(days=720)
        hist_1h = ticker.history(
            start=_start_1h.strftime("%Y-%m-%d"),
            end=_end_1h.strftime("%Y-%m-%d"),
            interval="1h",
            auto_adjust=True,
        )

        # ── DST-aware session boundary (identical to Gold) ────────────────────
        # CME WTI and ICE DXY both close at 17:00 ET daily.
        # EDT (UTC−4, Apr–Oct): close = 21:00 UTC, reopen = 22:00 UTC.
        # EST (UTC−5, Nov–Mar): close = 22:00 UTC, reopen = 23:00 UTC.
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        _ny_now      = datetime.now(ZoneInfo("America/New_York"))
        _ny_offset_h = int(_ny_now.utcoffset().total_seconds() / 3600)  # −4 EDT, −5 EST
        _close_utc   = 17 + (-_ny_offset_h)   # 21 UTC in EDT, 22 UTC in EST
        _reopen_utc  = 18 + (-_ny_offset_h)   # 22 UTC in EDT, 23 UTC in EST

        _now_utc = datetime.now(timezone.utc)
        _session_closed = _now_utc.hour >= _close_utc
        if _session_closed:
            run_date_utc = (_end_1h - timedelta(days=1)).date()  # today = last complete session
        else:
            run_date_utc = (_end_1h - timedelta(days=2)).date()  # yesterday (today still open)

        day_buckets: dict[str, dict] = {}
        if not hist_1h.empty:
            for ts, row in hist_1h.iterrows():
                if ts.tzinfo is None:
                    ts_utc = ts.replace(tzinfo=timezone.utc)
                else:
                    ts_utc = ts.astimezone(timezone.utc)

                # Session boundary: 17:00 ET (DST-aware UTC hour = _reopen_utc).
                # Hours 00–(_reopen_utc−1) belong to the session closing today.
                # Hours _reopen_utc–23 open a new session that closes tomorrow.
                if ts_utc.hour < _reopen_utc:
                    session_date = ts_utc.date()
                else:
                    session_date = (ts_utc + timedelta(days=1)).date()

                if session_date > run_date_utc:
                    continue

                date_str = session_date.strftime("%Y-%m-%d")
                o = float(row["Open"])
                h = float(row["High"])
                l = float(row["Low"])
                c = float(row["Close"])

                if not (_guard(id_, c) and _guard(id_, o) and h >= l):
                    continue
                if o == h == l == c:
                    continue
                if id_ in HL_MAX_SPREAD and l > 0:
                    if (h - l) / l > HL_MAX_SPREAD[id_]:
                        continue

                if date_str not in day_buckets:
                    # Include open in H/L: same fix as FX — yfinance 1H H/L excludes the
                    # open tick, so O can exceed H or fall below L on the first bar.
                    day_buckets[date_str] = {"open": o, "high": max(o, h), "low": min(o, l), "close": c}
                else:
                    b = day_buckets[date_str]
                    b["high"]  = max(b["high"], h)
                    b["low"]   = min(b["low"],  l)
                    b["close"] = c

        bars_1h: list[dict] = []
        for date_str in sorted(day_buckets.keys()):
            b = day_buckets[date_str]
            d = datetime.strptime(date_str, "%Y-%m-%d")
            if d.weekday() == 5:  # Saturday — no CME/ICE session
                continue
            if not _guard(id_, b["close"]):
                continue
            o_r = round(b["open"],  dec)
            c_r = round(b["close"], dec)
            h_r = round(max(b["high"],  o_r, c_r), dec)  # clamp: H >= max(O,C)
            l_r = round(min(b["low"],   o_r, c_r), dec)  # clamp: L <= min(O,C)
            bars_1h.append({
                "time":   date_str,
                "open":   o_r,
                "high":   h_r,
                "low":    l_r,
                "close":  c_r,
                "volume": 0,
            })

        # ── PART B: native 1D bars for period older than 730-day 1H limit ────
        hist_1d = ticker.history(period=PERIOD, interval=INTERVAL, auto_adjust=True)
        bars_1d_old: list[dict] = []
        earliest_1h = bars_1h[0]["time"] if bars_1h else None

        if not hist_1d.empty:
            for ts, row in hist_1d.iterrows():
                date_str = ts.strftime("%Y-%m-%d")
                if earliest_1h and date_str >= earliest_1h:
                    continue
                o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
                if not (_guard(id_, c) and _guard(id_, o) and h >= l and h >= c and l <= c):
                    continue
                if o == h == l == c:
                    continue
                if id_ in HL_MAX_SPREAD and l > 0:
                    if (h - l) / l > HL_MAX_SPREAD[id_]:
                        continue
                vol = int(row["Volume"]) if "Volume" in row and not (row["Volume"] != row["Volume"]) else 0
                bars_1d_old.append({
                    "time":   date_str,
                    "open":   round(o, dec),
                    "high":   round(h, dec),
                    "low":    round(l, dec),
                    "close":  round(c, dec),
                    "volume": vol,
                })

        # ── Merge: legacy 1D + 1H-aggregated (chronological) ─────────────────
        combined = bars_1d_old + bars_1h
        seen: set[str] = set()
        deduped: list[dict] = []
        for bar in reversed(combined):
            if bar["time"] not in seen:
                seen.add(bar["time"])
                deduped.append(bar)
        deduped.reverse()  # oldest to newest

        if len(deduped) < 30:
            print(f"  WARN [{id_}]: only {len(deduped)} valid bars after 1H aggregation - skipping")
            return None

        return deduped

    except Exception as exc:
        print(f"  ERROR [{id_}] (wti/dxy 1H agg): {exc}")
        return None


# ── Equity indices + US10Y + VIX: 1H → daily aggregation (NYSE 21:00 UTC) ───

def fetch_equity_ohlc_from_1h(id_: str, ticker_sym: str) -> list[dict] | None:
    """
    Build daily bars for early-closing equity indices (Nikkei 225, EuroStoxx 50) by
    aggregating 1H bars over the 21:00 UTC session boundary.

    Why 1H aggregation for these instruments?

    Nikkei 225 (TSE): closes ~06:00 UTC — well before the 22:30 UTC workflow run.
    EuroStoxx 50 (Euronext): closes ~15:30 UTC — also well before the 22:30 UTC run.

    Both instruments are fully traded out before the run executes. 1H aggregation
    gives faithful H/L and avoids a yfinance native-1D data-availability lag that
    affects European exchanges specifically:

      Root cause (EuroStoxx): yfinance.history(period="3y") computes its end-date in
      UTC. The workflow runs at 22:30 UTC which is already the NEXT calendar day in
      CEST (UTC+2). yfinance can exclude the current UTC-date's bar for a European
      exchange even when the session completed hours earlier (15:30 UTC), returning
      data only through the prior calendar day. 1H bars have no such lag — they are
      available immediately after each candle closes. This made stoxx.json end at
      2026-05-05 when it should end at 2026-05-06 after the May 6 run.

    SPX, Nasdaq, US10Y, and VIX are intentionally excluded and routed to native 1D
    yfinance. Their NYSE/CBOE sessions close at 20:00-21:00 UTC, and yfinance's
    native 1D feed already includes the in-progress today bar by 22:30 UTC. Routing
    them through 1H aggregation caused the session guard to drop today's bar (the 1H
    buckets for the current day aggregate into session_date = today, which is >
    run_date_utc = yesterday at a 22:30 UTC run), leaving the JSON one day behind.

    Session boundary: 21:00 UTC → 21:00 UTC
      * Nikkei 225 (TSE):     close ~06:00 UTC → fully included before run
      * EuroStoxx 50 (ENX):   close ~15:30 UTC → fully included before run

    Hybrid approach (same as FX / Gold):
    - 1H bars cover the most recent 730 days with correct H/L.
    - Native 1D bars cover the period older than 730 days.
    - Weekend bars excluded (no equity trading Sat/Sun).
    """
    try:
        ticker = yf.Ticker(ticker_sym)
        dec = DECIMALS.get(id_, 2)

        # PART A: 1H bars -> aggregate to daily (NYSE boundary 21:00 UTC)
        _end_1h   = datetime.now(timezone.utc) + timedelta(days=1)
        _start_1h = datetime.now(timezone.utc) - timedelta(days=720)
        hist_1h = ticker.history(
            start=_start_1h.strftime("%Y-%m-%d"),
            end=_end_1h.strftime("%Y-%m-%d"),
            interval="1h",
            auto_adjust=True,
        )

        # Session-complete guard: only include a session bar if the session has closed.
        # NYSE/Nasdaq close at 20:00 UTC; EuroStoxx at 15:30 UTC; Nikkei at ~06:00 UTC;
        # VIX/US10Y settle at ~21:00 UTC. The 21:00 UTC boundary means all equity
        # sessions are complete by 21:00 UTC. The scheduled run at 22:30 UTC is always safe.
        # session_date > run_date_utc = tomorrow's partial session — discard.
        # If triggered via workflow_dispatch before 21:00 UTC (mid-session), today's
        # partial bucket must be discarded to prevent incomplete bars in the JSON.
        _now_utc = datetime.now(timezone.utc)
        _equity_session_closed = _now_utc.hour >= 21  # All equity sessions closed by 21:00 UTC
        if _equity_session_closed:
            run_date_utc = (_end_1h - timedelta(days=1)).date()  # today UTC = last complete session
        else:
            run_date_utc = (_end_1h - timedelta(days=2)).date()  # yesterday UTC (today's session still open)

        day_buckets: dict[str, dict] = {}
        if not hist_1h.empty:
            for ts, row in hist_1h.iterrows():
                if ts.tzinfo is None:
                    ts_utc = ts.replace(tzinfo=timezone.utc)
                else:
                    ts_utc = ts.astimezone(timezone.utc)

                # NYSE boundary 21:00 UTC:
                # Hours 00-20 belong to the session closing today (UTC date).
                # Hours 21-23 open a new session that closes tomorrow.
                if ts_utc.hour < 21:
                    session_date = ts_utc.date()
                else:
                    session_date = (ts_utc + timedelta(days=1)).date()

                # Drop future/partial sessions only
                if session_date > run_date_utc:
                    continue

                # Drop weekends (no equity trading)
                if session_date.weekday() >= 5:
                    continue

                date_str = session_date.strftime("%Y-%m-%d")
                o = float(row["Open"])
                h = float(row["High"])
                l = float(row["Low"])
                c = float(row["Close"])

                if not (_guard(id_, c) and _guard(id_, o) and h >= l):
                    continue
                if o == h == l == c:
                    continue

                if date_str not in day_buckets:
                    # Include open in H/L: same fix as FX — yfinance 1H H/L excludes the
                    # open tick, so O can exceed H or fall below L on the first bar.
                    day_buckets[date_str] = {"open": o, "high": max(o, h), "low": min(o, l), "close": c}
                else:
                    b = day_buckets[date_str]
                    b["high"]  = max(b["high"], h)
                    b["low"]   = min(b["low"],  l)
                    b["close"] = c

        bars_1h: list[dict] = []
        for date_str in sorted(day_buckets.keys()):
            b = day_buckets[date_str]
            if not _guard(id_, b["close"]):
                continue
            o_r = round(b["open"],  dec)
            c_r = round(b["close"], dec)
            h_r = round(max(b["high"],  o_r, c_r), dec)  # clamp: H >= max(O,C)
            l_r = round(min(b["low"],   o_r, c_r), dec)  # clamp: L <= min(O,C)
            bars_1h.append({
                "time":   date_str,
                "open":   o_r,
                "high":   h_r,
                "low":    l_r,
                "close":  c_r,
                "volume": 0,
            })

        # PART B: native 1D bars for period older than 730-day 1H limit
        hist_1d = ticker.history(period=PERIOD, interval=INTERVAL, auto_adjust=True)
        bars_1d_old: list[dict] = []
        earliest_1h = bars_1h[0]["time"] if bars_1h else None

        if not hist_1d.empty:
            for ts, row in hist_1d.iterrows():
                date_str = ts.strftime("%Y-%m-%d")
                if earliest_1h and date_str >= earliest_1h:
                    continue
                o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
                if not (_guard(id_, c) and _guard(id_, o) and h >= l and h >= c and l <= c):
                    continue
                if o == h == l == c:
                    continue
                vol = int(row["Volume"]) if "Volume" in row and not (row["Volume"] != row["Volume"]) else 0
                bars_1d_old.append({
                    "time":   date_str,
                    "open":   round(o, dec),
                    "high":   round(h, dec),
                    "low":    round(l, dec),
                    "close":  round(c, dec),
                    "volume": vol,
                })

        # Merge: legacy 1D + 1H-aggregated (chronological)
        combined = bars_1d_old + bars_1h
        seen: set[str] = set()
        deduped: list[dict] = []
        for bar in reversed(combined):
            if bar["time"] not in seen:
                seen.add(bar["time"])
                deduped.append(bar)
        deduped.reverse()

        if len(deduped) < 30:
            print(f"  WARN [{id_}]: only {len(deduped)} valid bars after equity 1H agg - skipping")
            return None

        return deduped

    except Exception as exc:
        print(f"  ERROR [{id_}] (equity 1H agg): {exc}")
        return None

def fetch_ohlc(id_: str, ticker_sym: str) -> list[dict] | None:
    """
    Download 3 years of daily bars for ticker_sym.
    Returns a list of {time, open, high, low, close} dicts sorted oldest to newest,
    or None on failure.

    Three paths:
      FX pairs (28):      delegates to fetch_fx_ohlc_from_1h() — 1H aggregation over the
                          21:00 UTC NY session boundary. H/L match Bloomberg, TradingView,
                          Reuters FX daily candles. Required because native 1D FX bars use
                          UTC midnight cutoff, producing wrong H/L and unreliable opens.

      Equity 1H (2):      Nikkei 225 (TSE) and EuroStoxx 50 (Euronext) via
                          fetch_equity_ohlc_from_1h(). Both close well before the 22:30 UTC
                          run. EuroStoxx is here because yfinance native-1D has a data-
                          availability lag for European exchanges at 22:30 UTC (which is
                          already the next calendar day in CEST), causing the most recent
                          completed bar to be excluded. 1H bars have no such lag.

      Non-FX (9):         native yfinance 1D bars — BTC, ETH, SPX, Nasdaq, US10Y, VIX,
                          WTI, DXY, Gold. SPX/Nasdaq/US10Y/VIX close at 20:00–21:00 UTC;
                          native 1D includes their in-progress today bar at 22:30 UTC.
                          dashboard.js strips the today bar and re-injects it live from
                          quotes.json.
    """
    if id_ in FX_SYMBOLS:
        return fetch_fx_ohlc_from_1h(id_, ticker_sym)
    if id_ in CME_SYMBOLS:
        if id_ == 'gold':
            return fetch_gold_ohlc_from_1h(id_, ticker_sym)
        return fetch_wti_dxy_ohlc_from_1h(id_, ticker_sym)
    if id_ in EQUITY_1H_SYMBOLS:
        return fetch_equity_ohlc_from_1h(id_, ticker_sym)

    try:
        ticker = yf.Ticker(ticker_sym)
        # Crypto (BTC, ETH) trades 24/7 — use explicit start/end so yfinance
        # includes Saturday and Sunday bars. With period="3y", yfinance applies
        # the NYSE business-day calendar and silently drops all weekend bars.
        if id_ in CRYPTO_SYMBOLS:
            _end   = datetime.now(timezone.utc) + timedelta(days=1)
            _start = _end - timedelta(days=3 * 365 + 3)
            hist = ticker.history(
                start=_start.strftime("%Y-%m-%d"),
                end=_end.strftime("%Y-%m-%d"),
                interval=INTERVAL,
                auto_adjust=True,
            )
        else:
            hist = ticker.history(period=PERIOD, interval=INTERVAL, auto_adjust=True)
        if hist.empty:
            print(f"  WARN [{id_}]: empty history from yfinance")
            return None

        # Determine today's UTC date for the today-bar guard below.
        _today_utc = datetime.now(timezone.utc).date()

        bars: list[dict] = []
        for ts, row in hist.iterrows():
            # ── Timezone-safe date extraction ──────────────────────────────────
            # yfinance returns crypto (BTC, ETH) timestamps as timezone-aware
            # datetime objects in America/New_York (UTC-4/UTC-5). A bar stamped
            # "2026-04-29 00:00:00-04:00" is the Apr 29 UTC session but
            # ts.strftime("%Y-%m-%d") renders as "2026-04-28" in ET.
            # Converting to UTC first ensures the correct calendar date.
            if id_ in CRYPTO_SYMBOLS:
                if ts.tzinfo is not None:
                    date_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
                else:
                    date_str = ts.strftime("%Y-%m-%d")
            else:
                date_str = ts.strftime("%Y-%m-%d")

            o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])

            if not (_guard(id_, c) and _guard(id_, o) and h >= l and h >= c and l <= c):
                continue
            if o == h == l == c:
                continue
            if id_ in HL_MAX_SPREAD and l > 0:
                if (h - l) / l > HL_MAX_SPREAD[id_]:
                    continue

            dec = DECIMALS.get(id_, 5)
            vol = int(row["Volume"]) if "Volume" in row and not (row["Volume"] != row["Volume"]) else 0
            bars.append({
                "time":   date_str,
                "open":   round(o, dec),
                "high":   round(h, dec),
                "low":    round(l, dec),
                "close":  round(c, dec),
                "volume": vol,
            })

        # Keep FIRST bar per date (not last/reversed).
        # For instruments that reopen before UTC midnight (DXY 22:00 UTC, Gold/WTI 23:00 UTC),
        # yfinance can return TWO rows with the same calendar date:
        #   Row 1 (earlier):  completed session, correct full-day OHLC  ← KEEP THIS
        #   Row 2 (later):    new in-progress session, partial/wrong data ← DISCARD
        # dashboard.js strips the JSON's last bar and replaces it with the live today-bar
        # from quotes.json, so the in-progress row is not needed in the JSON at all.
        # Iterating forward and keeping the first occurrence preserves the completed session.
        # For all other symbols (one bar per date) behaviour is unchanged.
        seen: set[str] = set()
        deduped: list[dict] = []
        for bar in bars:
            if bar["time"] not in seen:
                seen.add(bar["time"])
                deduped.append(bar)
        # bars is already oldest-to-newest; no reverse needed.

        if len(deduped) < 30:
            print(f"  WARN [{id_}]: only {len(deduped)} valid bars - skipping")
            return None

        # ── Crypto gap-fill guard ──────────────────────────────────────────────
        # Root cause: yfinance has a finalization lag for crypto daily bars. The
        # weekend run (Sat/Sun 23:30 UTC) requests bars with an explicit start/end
        # range. yfinance returns the new day's in-progress bar (e.g. 2026-05-09)
        # but may omit the just-completed prior bar (e.g. 2026-05-08) because it
        # has not yet been finalized in the Yahoo Finance data feed.
        # Result: the completed bar is absent from deduped, replaced only by the
        # new in-progress bar. dashboard.js strips the in-progress bar and injects
        # a live today-bar — but the prior completed bar is still missing, leaving
        # a 1-day gap in the chart (07/05 → 09/05, skipping 08/05).
        #
        # Fix: when a gap of >= 2 days is detected between the last two bars in the
        # new yfinance response, read the EXISTING JSON file from disk and backfill
        # any bars that fall within the gap but are present in the prior JSON.
        # This preserves correctly finalized bars that yfinance temporarily omits.
        #
        # Safety: bars backfilled from the existing JSON are inserted verbatim (they
        # were written by a prior run that did pass all guards). The new yfinance
        # data takes priority for any date present in both sources — the backfill
        # only fills dates that are ABSENT from the new response.
        if id_ in CRYPTO_SYMBOLS and len(deduped) >= 2:
            from datetime import date as _date
            _last_date = _date.fromisoformat(deduped[-1]["time"])
            _prev_date = _date.fromisoformat(deduped[-2]["time"])
            _gap = (_last_date - _prev_date).days
            if _gap >= 2:
                print(f"  WARN [{id_}]: {_gap}-day gap between {deduped[-2]['time']} and {deduped[-1]['time']} — "
                      f"attempting backfill from existing JSON.")
                # Backfill from existing JSON on disk
                _existing_path = OUT_DIR / f"{id_}.json"
                if _existing_path.exists():
                    try:
                        with open(_existing_path, "r", encoding="utf-8") as _ef:
                            _existing_bars: list[dict] = json.load(_ef)
                        # Build a set of dates already present in the new deduped list
                        _new_dates: set[str] = {b["time"] for b in deduped}
                        # Collect bars from the existing JSON that fall within the gap
                        # (strictly between _prev_date and _last_date, exclusive) and
                        # are absent from the new yfinance response.
                        _gap_start = deduped[-2]["time"]  # exclusive lower bound
                        _gap_end   = deduped[-1]["time"]  # exclusive upper bound
                        _recovered: list[dict] = [
                            b for b in _existing_bars
                            if _gap_start < b["time"] < _gap_end
                            and b["time"] not in _new_dates
                        ]
                        if _recovered:
                            print(f"  INFO [{id_}]: backfilled {len(_recovered)} bar(s) from existing JSON: "
                                  + ", ".join(b["time"] for b in _recovered))
                            # Merge: deduped (without last bar) + recovered + last bar
                            deduped = deduped[:-1] + _recovered + [deduped[-1]]
                            # Re-sort chronologically (recovered bars should already be in order,
                            # but sort defensively to guarantee oldest-to-newest invariant)
                            deduped.sort(key=lambda b: b["time"])
                        else:
                            print(f"  WARN [{id_}]: no recoverable bars found in existing JSON for gap "
                                  f"{_gap_start} → {_gap_end}.")
                    except Exception as _gfe:
                        print(f"  WARN [{id_}]: backfill failed ({_gfe}) — gap remains.")
                else:
                    print(f"  WARN [{id_}]: no existing JSON at {_existing_path} — gap cannot be filled.")

        # ── Today-bar partial-data guard ───────────────────────────────────────
        # fetch_ohlc runs at 22:30 UTC. For most non-FX symbols the last bar in
        # the yfinance response is today's in-progress bar (volume will be partial
        # for US markets, or a completed bar for Asian sessions that close by
        # 08:00 UTC). dashboard.js strips it and replaces it with the live
        # intraday bar from quotes.json. That is the intended design.
        #
        # Problem: if the workflow is triggered via workflow_dispatch BEFORE the
        # relevant market closes (e.g. Nikkei closes ~07:00 UTC, but dispatch
        # is run at 06:00 UTC), the today bar would have clearly wrong/partial
        # data. To prevent writing a bad today-bar for Nikkei and similar early-
        # close markets we do NOT strip here — the JS always strips the today-bar
        # from the JSON regardless. So the guard here is intentionally a no-op:
        # we KEEP the today-bar in the JSON so the JS can strip it and replace
        # with the live feed. This is the correct pipeline architecture.
        #
        # No FX open correction here — handled by fetch_fx_ohlc_from_1h.
        # No back-adjustment for futures — removed v7.47.19.

        return deduped

    except Exception as exc:
        print(f"  ERROR [{id_}]: {exc}")
        return None

def clamp_bars(bars: list[dict], id_: str) -> list[dict]:
    """
    Apply OHLC structural integrity clamp to every bar in the final output list.

    Guarantees H >= max(O, C) and L <= min(O, C) for every bar.

    Root cause of violations in FX 1H-aggregated data:
      - The bucket open = first 1H bar's Open, which yfinance FX reports as the
        prev_session_close on most bars (the last tick before session open, not the
        first real tick of the new session). On gap-down sessions prev_session_close
        can exceed all 1H highs accumulated during the day → H < O.
      - The Part B 1D legacy bars apply prev_close open correction AFTER individual
        bar validation, which can also introduce H < O when bars are merged.
      - The per-bucket clamp in each fetch_*_ohlc_from_1h already handles the 1H
        portion, but the merge step and deduplication can still expose violations.

    This function is the definitive last-line-of-defense: it processes every bar
    regardless of how it was produced, ensuring the JSON written to disk is always
    structurally valid. LightweightCharts renders H < O or L > C as visually
    deformed candles — this clamp prevents that entirely.

    The clamp is idempotent and only extends wicks outward — it never compresses
    the real intraday range. A session where the open genuinely gaps above/below
    the intraday range will correctly show an upper/lower wick extending to the
    gap-open level.
    """
    dec = DECIMALS.get(id_, 5)
    clamped = 0
    result = []
    for bar in bars:
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        new_h = round(max(h, o, c), dec)
        new_l = round(min(l, o, c), dec)
        if new_h != h or new_l != l:
            clamped += 1
            bar = dict(bar)
            bar["high"] = new_h
            bar["low"]  = new_l
        result.append(bar)
    if clamped:
        print(f"    clamp: fixed {clamped} structurally invalid bars")
    return result


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


# ── Main ───────────────────────────────────────────────────────────────────────



# ── Intraday (H1 / H4) OHLC builder ───────────────────────────────────────────
# Generates ohlc-data/h1/{id}.json and ohlc-data/h4/{id}.json for all FX pairs
# plus key non-FX instruments (BTC, ETH, Gold, WTI, SPX, DXY).
#
# Bar format: {"time": <unix_seconds_int>, "open": …, "high": …, "low": …, "close": …}
# LightweightCharts requires unix timestamps (not YYYY-MM-DD) for intraday series.
# H4 boundaries align to UTC 4-hour blocks (00, 04, 08, 12, 16, 20).
# yfinance 1H limit: 730 days.

INTRADAY_SYMBOLS: dict[str, str] = {
    **{id_: sym for id_, sym in SYMBOLS.items() if id_ not in NON_FX_SYMBOLS},
    "btc": "BTC-USD", "eth": "ETH-USD", "gold": "GC=F",
    "wti": "CL=F", "spx": "^GSPC", "dxy": "DX-Y.NYB",
}


def build_intraday_ohlc() -> None:
    """Fetch 1H bars from yfinance and write H1 + H4 JSONs for all intraday symbols."""
    now_utc  = datetime.now(timezone.utc)
    h1_dir   = OUT_DIR / "h1"
    h4_dir   = OUT_DIR / "h4"
    h1_dir.mkdir(parents=True, exist_ok=True)
    h4_dir.mkdir(parents=True, exist_ok=True)

    end_dt   = now_utc + timedelta(days=1)
    start_dt = now_utc - timedelta(days=720)
    written_h1 = written_h4 = 0
    errors_intra: list[str] = []

    for id_, ticker_sym in INTRADAY_SYMBOLS.items():
        print(f"  [{id_:10s}] {ticker_sym} H1/H4 ...", end=" ", flush=True)
        try:
            ticker = yf.Ticker(ticker_sym)
            dec    = DECIMALS.get(id_, 5)
            hist   = ticker.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="1h", auto_adjust=True,
            )
            if hist.empty:
                print("SKIP (empty)")
                continue

            h1_bars: list[dict] = []
            h4_buckets: dict[int, dict] = {}

            for ts, row in hist.iterrows():
                ts_utc  = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
                unix_ts = int(ts_utc.timestamp())
                o = round(float(row["Open"]),  dec)
                h = round(float(row["High"]),  dec)
                l = round(float(row["Low"]),   dec)
                c = round(float(row["Close"]), dec)
                if not (_guard(id_, c) and _guard(id_, o) and h >= l):
                    continue
                if o == h == l == c:
                    continue
                h = round(max(h, o, c), dec)
                l = round(min(l, o, c), dec)
                h1_bars.append({"time": unix_ts, "open": o, "high": h, "low": l, "close": c})

                # H4: align to UTC 4-hour block
                block_hour  = (ts_utc.hour // 4) * 4
                block_start = ts_utc.replace(hour=block_hour, minute=0, second=0, microsecond=0)
                block_unix  = int(block_start.timestamp())
                if block_unix not in h4_buckets:
                    h4_buckets[block_unix] = {"open": o, "high": h, "low": l, "close": c}
                else:
                    b = h4_buckets[block_unix]
                    b["high"]  = round(max(b["high"], h), dec)
                    b["low"]   = round(min(b["low"],  l), dec)
                    b["close"] = c

            # Drop the current (incomplete) H4 block
            if h4_buckets:
                del h4_buckets[max(h4_buckets)]

            # Deduplicate H1 and sort
            seen_h1: dict[int, dict] = {b["time"]: b for b in h1_bars}
            h1_final = sorted(seen_h1.values(), key=lambda b: b["time"])

            # Build sorted H4 list with integrity clamp
            h4_final = [
                {"time": ts, "open": b["open"],
                 "high": round(max(b["high"], b["open"], b["close"]), dec),
                 "low":  round(min(b["low"],  b["open"], b["close"]), dec),
                 "close": b["close"]}
                for ts, b in sorted(h4_buckets.items())
            ]

            if len(h1_final) < 50:
                print(f"SKIP (only {len(h1_final)} H1 bars)")
                errors_intra.append(id_)
                continue

            write_json(h1_dir / f"{id_}.json", h1_final)
            write_json(h4_dir / f"{id_}.json", h4_final)
            written_h1 += 1; written_h4 += 1
            print(f"OK  H1:{len(h1_final)} H4:{len(h4_final)}")

        except Exception as exc:
            print(f"ERROR: {exc}")
            errors_intra.append(id_)
        time.sleep(0.4)

    print(f"  Intraday: {written_h1} H1 files, {written_h4} H4 files written.")
    if errors_intra:
        print(f"  Intraday errors: {', '.join(errors_intra)}")

def main() -> None:
    now_utc = datetime.now(timezone.utc)
    print(f"fetch_ohlc.py — {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Output dir   : {OUT_DIR.resolve()}")
    print(f"Symbols      : {len(SYMBOLS)}")
    print()

    errors: list[str] = []
    written = 0

    for id_, ticker_sym in SYMBOLS.items():
        print(f"  [{id_:10s}] {ticker_sym} ...", end=" ", flush=True)
        bars = fetch_ohlc(id_, ticker_sym)
        if bars is None:
            errors.append(id_)
            print("FAILED")
        else:
            bars = clamp_bars(bars, id_)
            write_json(OUT_DIR / f"{id_}.json", bars)
            print(f"OK  ({len(bars)} bars, {OUT_DIR / f'{id_}.json'} {(OUT_DIR / f'{id_}.json').stat().st_size // 1024}KB)")
            written += 1
        # Polite rate limit — yfinance is free/unauthenticated
        time.sleep(0.4)

    # Write metadata file
    meta = {
        "updated_at":    now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "period":        PERIOD,
        "interval":      INTERVAL,
        "symbol_count":  written,
        "errors":        errors,
    }
    write_json(OUT_DIR / "meta.json", meta)

    print()
    print(f"Done — {written}/{len(SYMBOLS)} symbols written.")
    if errors:
        print(f"Errors ({len(errors)}): {', '.join(errors)}")
    # Build H1/H4 intraday OHLC files
    print()
    print("Building H1/H4 intraday OHLC …")
    build_intraday_ohlc()

    # Exit non-zero only if majority of D1 symbols failed
    if written < len(SYMBOLS) // 2:
        sys.exit(1)


if __name__ == "__main__":
    main()
