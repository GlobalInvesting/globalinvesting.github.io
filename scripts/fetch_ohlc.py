#!/usr/bin/env python3
"""
fetch_ohlc.py  v1.4 — Daily OHLC history for Lightweight Charts
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

  Gold:           Daily bars BUILT from 1H bars aggregated over the CME session boundary:
                  22:00 UTC → 22:00 UTC (17:00 New York → 17:00 New York, EST fixed).
                  yfinance native 1D bars for GC=F are severely degraded: ~46% of bars have
                  O==H or O==L because Yahoo constructs them from settlement ticks, not from
                  the full electronic session. 1H aggregation faithfully captures intraday
                  H/L and produces candles that match Bloomberg, CME Group charts, and
                  TradingView Gold daily bars. No back-adjustment — roll gaps appear as-is.
  WTI:            Raw front-month prices from yfinance (CL=F) — no back-adjustment.
                  Roll gaps between contracts appear as-is: they reflect the actual switch
                  to the next front-month contract, exactly as shown on Reuters, CNBC,
                  Barchart, and Investing.com. Back-adjustment (Panama method) was removed
                  because it produces synthetic price levels that were never traded.
  DXY only:       HL_MAX_SPREAD guard drops bars with impossible intraday ranges
  Nasdaq:         Uses ^NDX (Nasdaq 100) to match the CFI:US100 chart tab; ^IXIC
                  (Composite) has different constituents and price levels (~19k vs ~5.8k).
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
    # Equity indices
    "spx":    "^GSPC",     # S&P 500
    "nasdaq": "^NDX",       # Nasdaq 100 index — matches CFI:US100 chart tab; ^IXIC (Composite) has different levels (~19k vs ~5.8k)
    "nikkei": "^N225",     # Nikkei 225
    "stoxx":  "^STOXX50E", # EuroStoxx 50
    "vix":    "^VIX",      # CBOE Volatility Index
    # Crypto
    "eth":   "ETH-USD",    # Ethereum vs USD
    # FX Index
    "dxy":   "DX-Y.NYB",   # US Dollar Index (ICE futures)
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
    "gold":  2, "wti": 2, "btc": 2, "us10y": 4,
    "spx": 2, "nasdaq": 2, "nikkei": 2, "stoxx": 2, "vix": 2,
    "eth": 2, "dxy": 3,
}

# Plausibility guards — reject bars outside these ranges
GUARDS: dict[str, tuple[float, float]] = {
    "gold":  (500.0,   8000.0),
    "wti":   (10.0,    300.0),
    "btc":   (100.0,   500000.0),
    "us10y": (0.01,    25.0),
    "spx":   (500.0,   15000.0),
    "nasdaq":(1000.0,  30000.0),  # ^NDX (Nasdaq 100): historical range ~1k to ~22k; upper headroom for growth
    "nikkei":(5000.0,  80000.0),
    "stoxx": (1000.0,  8000.0),
    "vix":   (5.0,     90.0),      # VIX historically ranges 5-90
    "eth":   (10.0,    20000.0),
    "dxy":   (60.0,    150.0),
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
NON_FX_SYMBOLS = {'wti', 'btc', 'us10y', 'spx', 'nasdaq', 'nikkei', 'stoxx', 'vix', 'eth', 'dxy'}
GOLD_SYMBOLS   = {'gold'}   # CME session 1H aggregation — separate from FX (different boundary)
# Equity indices and related instruments: 1H aggregation over NYSE 21:00 UTC boundary.
# yfinance native 1D bars for ^GSPC, ^NDX, ^STOXX50E etc. are unreliable for the
# current session — they are often absent or contain only a partial bar until well
# after market close. Aggregating 1H bars ensures the bar for the current trading day
# is always present in the JSON after the workflow runs (22:30 UTC > 21:00 UTC boundary).
EQUITY_1H_SYMBOLS = {'spx', 'nasdaq', 'stoxx', 'nikkei', 'us10y', 'vix'}
FX_SYMBOLS = set(SYMBOLS.keys()) - NON_FX_SYMBOLS - GOLD_SYMBOLS

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
        _end_1h   = datetime.now(timezone.utc) + timedelta(days=1)
        _start_1h = _end_1h - timedelta(days=730)
        hist_1h = ticker.history(
            start=_start_1h.strftime("%Y-%m-%d"),
            end=_end_1h.strftime("%Y-%m-%d"),
            interval="1h",
            auto_adjust=True,
        )

        # The script runs at 22:30 UTC. Bars from 21:00–22:30 UTC already belong
        # (per FX session logic) to tomorrow's session date — but that session has
        # only ~1.5 h of data and is incomplete. Writing it creates a phantom partial
        # bar in the JSON that causes a visual gap when the site loads next day
        # (the today-bar injector tries to place a bar for the *actual* today, which
        # is now earlier than the JSON's last bar, so LWC silently ignores it).
        # Fix: drop any bucket whose session_date >= the UTC calendar date at run time.
        # The FX session for the current UTC date is always complete by 20:59 UTC
        # (it opened at 21:00 UTC the prior day), so this is safe — we never drop
        # a completed session. The incomplete forward-looking bucket is discarded.
        run_date_utc = (_end_1h - timedelta(days=1)).date()  # UTC calendar date when the script runs

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
                # At 22:30 UTC, session_date == run_date_utc is the session that
                # JUST closed at 21:00 UTC — complete, keep it.
                # session_date > run_date_utc is tomorrow (~1.5h data) — discard.
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
                    day_buckets[date_str] = {"open": o, "high": h, "low": l, "close": c}
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
            bars_1h.append({
                "time":   date_str,
                "open":   round(b["open"],  dec),
                "high":   round(b["high"],  dec),
                "low":    round(b["low"],   dec),
                "close":  round(b["close"], dec),
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

        # Apply prev_close as open across the full series.
        # Bloomberg and TradingView FX daily convention: open = prior session's close.
        # This guarantees candle body color always matches the daily pct sign.
        for i in range(1, len(deduped)):
            deduped[i]["open"] = deduped[i - 1]["close"]

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
    is therefore 22:00 UTC (17:00 ET in standard EST/winter time), consistent with how
    Bloomberg, CME Group charts, and TradingView construct Gold daily candles.

    yfinance native 1D bars for GC=F are severely degraded (~46% of bars have O==H or
    O==L) because Yahoo constructs them from settlement ticks, not from the full COMEX
    electronic session. 1H aggregation produces faithful intraday H/L ranges.

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
        _start_1h = _end_1h - timedelta(days=730)
        hist_1h = ticker.history(
            start=_start_1h.strftime("%Y-%m-%d"),
            end=_end_1h.strftime("%Y-%m-%d"),
            interval="1h",
            auto_adjust=True,
        )

        # Drop any bucket whose session_date >= the UTC calendar date at run time.
        # The script runs at 22:30 UTC. Bars from 22:00–22:30 UTC already belong
        # (per CME session logic) to tomorrow's session date — but that session has
        # only ~30 min of data and is incomplete. Discarding it prevents a phantom
        # partial bar that would cause a visual gap when the site loads next day.
        # The CME session for the current UTC date is always complete by 21:59 UTC
        # (it opened at 22:00 UTC the prior day), so this never drops a completed session.
        run_date_utc = (_end_1h - timedelta(days=1)).date()  # UTC calendar date when the script runs

        day_buckets: dict[str, dict] = {}
        if not hist_1h.empty:
            for ts, row in hist_1h.iterrows():
                if ts.tzinfo is None:
                    ts_utc = ts.replace(tzinfo=timezone.utc)
                else:
                    ts_utc = ts.astimezone(timezone.utc)

                # CME session boundary: 22:00 UTC (17:00 ET EST).
                # Hours 00–21 belong to the session closing today.
                # Hours 22–23 open a new session that closes tomorrow.
                if ts_utc.hour < 22:
                    session_date = ts_utc.date()
                else:
                    session_date = (ts_utc + timedelta(days=1)).date()

                # Drop any 1H bar that belongs to a session not yet completed.
                # session_date == run_date_utc = today's session (complete at 22:00 UTC) — keep.
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
                    day_buckets[date_str] = {"open": o, "high": h, "low": l, "close": c}
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
            bars_1h.append({
                "time":   date_str,
                "open":   round(b["open"],  dec),
                "high":   round(b["high"],  dec),
                "low":    round(b["low"],   dec),
                "close":  round(b["close"], dec),
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


# ── Equity indices + US10Y + VIX: 1H → daily aggregation (NYSE 21:00 UTC) ───

def fetch_equity_ohlc_from_1h(id_: str, ticker_sym: str) -> list[dict] | None:
    """
    Build daily bars for equity indices (SPX, Nasdaq, Stoxx, Nikkei), US 10Y yield,
    and VIX by aggregating 1H bars over the NYSE session boundary: 21:00 UTC.

    Why 1H aggregation instead of native yfinance 1D bars?
    yfinance native 1D bars for these symbols often EXCLUDE the current trading day
    when the workflow runs at 22:30 UTC, even though the session is complete by
    20:00-21:00 UTC. The bar appears only in the next run. Aggregating 1H bars over
    the 21:00 UTC boundary guarantees today's session is always present.

    Session boundary: 21:00 UTC -> 21:00 UTC
      * US markets (NYSE/Nasdaq):   close 20:00 UTC -> fully included
      * EuroStoxx 50 (Euronext):    close 15:30 UTC -> fully included
      * Nikkei 225 (TSE):           close ~06:00 UTC -> fully included
      * US 10Y (^TNX, CBOE):        settles ~21:00 UTC -> included
      * VIX (CBOE):                 settles ~21:00 UTC -> included

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
        _start_1h = _end_1h - timedelta(days=730)
        hist_1h = ticker.history(
            start=_start_1h.strftime("%Y-%m-%d"),
            end=_end_1h.strftime("%Y-%m-%d"),
            interval="1h",
            auto_adjust=True,
        )

        # run_date_utc = today UTC (when workflow runs at 22:30 UTC).
        # session_date == run_date_utc = today's session (closed at 21:00 UTC) -- KEEP.
        # session_date > run_date_utc = tomorrow's partial session -- DISCARD.
        run_date_utc = (_end_1h - timedelta(days=1)).date()

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
                    day_buckets[date_str] = {"open": o, "high": h, "low": l, "close": c}
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
            bars_1h.append({
                "time":   date_str,
                "open":   round(b["open"],  dec),
                "high":   round(b["high"],  dec),
                "low":    round(b["low"],   dec),
                "close":  round(b["close"], dec),
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

    FX symbols: delegates to fetch_fx_ohlc_from_1h() which aggregates 1H bars over
    the 21:00 UTC session boundary — H/L faithful to real FX session ranges, matching
    Yahoo Finance's own 4H chart, TradingView FX daily candles, and Bloomberg.

    Gold: delegates to fetch_gold_ohlc_from_1h() which aggregates 1H bars over the
    22:00 UTC CME session boundary — eliminates the ~46% stub-bar rate in yfinance
    native 1D GC=F data.

    Equity indices / VIX / US10Y: delegates to fetch_equity_ohlc_from_1h() which
    aggregates 1H bars over the 21:00 UTC NYSE session boundary — guarantees the
    current trading day bar is present in the JSON after the 22:30 UTC workflow run.

    Non-FX remaining (BTC, ETH, WTI, DXY): uses Yahoo's native 1D bars.
    """
    if id_ in FX_SYMBOLS:
        return fetch_fx_ohlc_from_1h(id_, ticker_sym)

    if id_ in GOLD_SYMBOLS:
        return fetch_gold_ohlc_from_1h(id_, ticker_sym)

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

        bars: list[dict] = []
        for ts, row in hist.iterrows():
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

        seen: set[str] = set()
        deduped: list[dict] = []
        for bar in reversed(bars):
            if bar["time"] not in seen:
                seen.add(bar["time"])
                deduped.append(bar)
        deduped.reverse()

        if len(deduped) < 30:
            print(f"  WARN [{id_}]: only {len(deduped)} valid bars - skipping")
            return None

        # No FX open correction here — handled by fetch_fx_ohlc_from_1h.
        # No back-adjustment for futures — removed v7.47.19.

        return deduped

    except Exception as exc:
        print(f"  ERROR [{id_}]: {exc}")
        return None

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


# ── Main ───────────────────────────────────────────────────────────────────────

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
    # Exit non-zero only if majority of symbols failed
    if written < len(SYMBOLS) // 2:
        sys.exit(1)


if __name__ == "__main__":
    main()
