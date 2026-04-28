#!/usr/bin/env python3
"""
fetch_ohlc.py  v1.2 — Daily OHLC history for Lightweight Charts
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

Data integrity:
  FX symbols:     open replaced with prev bar's close (Yahoo FX open is unreliable)
  Gold/WTI:       Panama back-adjustment applied to eliminate contract roll gaps.
                  yfinance GC=F / CL=F switch front-month contracts each month,
                  creating inter-bar gaps that look like vertical price spikes in charts.
                  Back-adjustment (proportional, backward pass) makes the series
                  continuous — equivalent to TradingView's GC1! / CL1! continuous contracts.
  All futures:    HL_MAX_SPREAD guard drops bars with impossible intraday ranges
                  (secondary defense after back-adjustment).
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
# yfinance front-month futures (GC=F, CL=F, DX-Y.NYB) occasionally produce bars
# that span TWO contract months at roll dates — the High reflects the expiring
# contract's last prints and the Low reflects the incoming contract's first prints
# (or vice versa), creating a single bar with an impossible intraday range.
# These "roll-artifact" bars cause prominent visual spikes on the chart that are
# absent on continuous-contract sources (e.g. TradingView's GC1!).
#
# PRIMARY fix for roll gaps: back-adjustment (Panama method) applied in fetch_ohlc()
# for FUTURES_ADJUST symbols (gold, wti). This eliminates inter-bar gaps entirely.
#
# SECONDARY fix: bars where the H/L spread exceeds the threshold are still dropped.
# These represent bars where BOTH the High and Low are contaminated by the roll
# (impossible intraday range), which back-adjustment cannot fix.
# Normal gold intraday range: 0.3–2.5%. Threshold at 4% catches the impossible bars.
# WTI/CL=F: normal 1–5%; threshold at 8% catches severe multi-contract bars.
# DXY/DX-Y.NYB and index futures: normal < 2%; threshold at 5%.
HL_MAX_SPREAD: dict[str, float] = {
    "gold":  0.04,   # 4% — secondary guard after back-adjustment
    "wti":   0.08,   # 8% — secondary guard after back-adjustment
    "dxy":   0.05,   # 5% — DX futures roll artifacts
}

# FX spot symbols — daily open values from Yahoo Finance are NOT reliable for these.
# Yahoo reports the last tick of the previous UTC day as the open, which is close to
# but not exactly prev_close. This creates candle bodies whose color (open vs close)
# contradicts the day's actual direction (close vs prev_close) in ~40-50% of bars.
# Fix: replace each bar's open with the previous bar's close when writing the JSON.
# This ensures candle color always matches the daily pct sign, consistent with how
# TradingView renders FX daily bars and with the today-bar logic in dashboard.js.
NON_FX_SYMBOLS = {'gold', 'wti', 'btc', 'us10y', 'spx', 'nasdaq', 'nikkei', 'stoxx', 'vix', 'eth', 'dxy'}
FX_SYMBOLS = set(SYMBOLS.keys()) - NON_FX_SYMBOLS

# Crypto trades 24/7 — yfinance only returns Saturday/Sunday bars when an explicit
# start/end date range is passed. Using period="3y" silently omits weekends because
# yfinance applies a NYSE/NASDAQ business-day calendar as the default date filter.
CRYPTO_SYMBOLS = {'btc', 'eth'}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _guard(id_: str, val: float) -> bool:
    lo, hi = GUARDS.get(id_, FX_GUARD)
    return lo <= val <= hi



def fetch_ohlc(id_: str, ticker_sym: str) -> list[dict] | None:
    """
    Download 3 years of daily bars for ticker_sym.
    Returns a list of {time, open, high, low, close} dicts sorted oldest to newest,
    or None on failure.

    For FX symbols, each bar's open is replaced with the previous bar's close before
    writing to JSON. Yahoo Finance daily FX open values are unreliable — they store the
    last tick of the prior UTC session rather than the real session open, causing candle
    body color to contradict the day's actual direction (close vs prev_close) in roughly
    half of all bars. Replacing open with prev_close guarantees consistency:
      green candle ↔ close > prev_close ↔ pct > 0 (always)
      red candle   ↔ close < prev_close ↔ pct < 0 (always)
    This matches how the today-bar is built in dashboard.js and how TradingView renders
    FX daily candles. Non-FX symbols (BTC, SPX, Gold, etc.) use Yahoo's raw open because
    those assets have a real exchange session open that is meaningful for intraday context.
    """
    try:
        ticker = yf.Ticker(ticker_sym)
        # Crypto (BTC, ETH) trades 24/7 — use explicit start/end so yfinance
        # includes Saturday and Sunday bars. With period="3y", yfinance applies
        # the NYSE business-day calendar and silently drops all weekend bars,
        # creating the visual gap on the chart between Friday close and Monday open.
        if id_ in CRYPTO_SYMBOLS:
            _end   = datetime.now(timezone.utc) + timedelta(days=1)  # end is exclusive — add 1 day to include today
            _start = _end - timedelta(days=3 * 365 + 3)  # 3y + buffer
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

            # Skip bars with invalid OHLC relationships or out-of-range prices
            if not (_guard(id_, c) and _guard(id_, o) and h >= l and h >= c and l <= c):
                continue
            # Skip bars where all four values are identical (stale/placeholder bar)
            if o == h == l == c:
                continue
            # Skip front-month futures roll artifacts — bars where the H/L spread
            # exceeds the physical limit for a real trading session. yfinance GC=F and
            # CL=F occasionally produce bars that span two contract months at roll
            # dates, creating an impossibly wide range that renders as a visual spike
            # absent on continuous-contract sources (e.g. TradingView GC1!).
            if id_ in HL_MAX_SPREAD and l > 0:
                if (h - l) / l > HL_MAX_SPREAD[id_]:
                    continue

            dec = DECIMALS.get(id_, 5)
            # Volume: FX has tick-volume (number of ticks/quotes), non-FX has real traded volume
            # For FX, Yahoo returns tick count which is a reasonable proxy for activity
            vol = int(row["Volume"]) if "Volume" in row and not (row["Volume"] != row["Volume"]) else 0
            bars.append({
                "time":   date_str,
                "open":   round(o, dec),
                "high":   round(h, dec),
                "low":    round(l, dec),
                "close":  round(c, dec),
                "volume": vol,
            })

        # Deduplicate by date (keep last occurrence - intraday quirk for today's bar)
        seen: set[str] = set()
        deduped: list[dict] = []
        for bar in reversed(bars):
            if bar["time"] not in seen:
                seen.add(bar["time"])
                deduped.append(bar)
        deduped.reverse()  # oldest to newest

        if len(deduped) < 30:
            print(f"  WARN [{id_}]: only {len(deduped)} valid bars - skipping")
            return None

        # FX open correction: replace each bar's open with the previous bar's close.
        # Yahoo's daily FX open values are unreliable — they store the last tick of the
        # prior UTC session, not the actual FX session open. This causes candle body
        # color (open vs close) to contradict the day's actual direction (close vs
        # prev_close) in roughly half of all bars across all FX pairs.
        # Using prev_close as open guarantees: green candle ↔ pct > 0, red ↔ pct < 0.
        # Note: this can place open outside the bar's H-L range because Yahoo's H/L data
        # also has a timezone offset vs close data. LW Charts renders these correctly.
        # The first bar has no predecessor — its open is left as-is.
        if id_ in FX_SYMBOLS:
            for i in range(1, len(deduped)):
                deduped[i]["open"] = deduped[i - 1]["close"]

        # ── Back-adjustment for front-month futures (Panama method) ──────────────
        # yfinance GC=F (Gold) and CL=F (WTI) return bars from the current front-month
        # contract. Each month when the contract expires, yfinance silently switches to
        # the next contract. The new contract's price ≠ the old contract's price, creating
        # a "roll gap" between prev_close and next_open. These gaps are NOT real price
        # moves — they are an artifact of switching contracts — but they appear as violent
        # vertical jumps in the chart (e.g. WTI jumping 29% overnight on 2026-03-13).
        #
        # TradingView and Bloomberg solve this with back-adjusted continuous contracts
        # (GC1!, CL1!) where the historical series is retroactively shifted so every bar
        # closes at the opening price of the next bar. We replicate this here.
        #
        # Algorithm (Panama / proportional back-adjustment, backward pass):
        #   1. Scan forward; when close-to-open gap > ROLL_GAP_THRESHOLD, record a roll.
        #   2. Walk backward from each roll: multiply all prior OHLC values by the ratio
        #      curr_open / prev_close so the series connects seamlessly.
        #   3. Because we adjust proportionally (multiply), all percentage moves and candle
        #      body shapes are preserved exactly — only the absolute price level shifts.
        #
        # Threshold: 3.5% (WTI 3% normal intraday swings; Gold 0.3-2.5% normal swings).
        # Gold's rolls are typically 2-5%; WTI's can be 10-30%.
        #
        # Effect: the adjusted series looks identical to TradingView's continuous contracts.
        # The absolute price levels in the oldest bars will differ from spot/cash prices by
        # the cumulative adjustment factor — this is correct and expected (Bloomberg standard).
        FUTURES_ADJUST = {"gold", "wti"}
        ROLL_GAP_THRESHOLD = 0.035  # 3.5% — above normal overnight moves for these assets

        if id_ in FUTURES_ADJUST and len(deduped) >= 2:
            roll_count = 0
            for i in range(1, len(deduped)):
                prev_c = deduped[i - 1]["close"]
                curr_o = deduped[i]["open"]
                if prev_c <= 0:
                    continue
                gap = abs(curr_o - prev_c) / prev_c
                if gap > ROLL_GAP_THRESHOLD:
                    # Proportional adjustment factor: make prev_close == curr_open
                    factor = curr_o / prev_c
                    # Adjust all bars before this roll (backward pass)
                    for j in range(i):
                        for field in ("open", "high", "low", "close"):
                            deduped[j][field] = round(deduped[j][field] * factor, DECIMALS.get(id_, 2))
                    roll_count += 1
                    print(f"  [back-adj] {id_} roll at {deduped[i]['time']}: "
                          f"gap {gap*100:.1f}% factor={factor:.5f} "
                          f"(prev_c={prev_c:.2f} → curr_o={curr_o:.2f})")
            if roll_count:
                print(f"  [back-adj] {id_}: {roll_count} rolls adjusted — series is now continuous")

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
