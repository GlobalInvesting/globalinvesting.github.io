#!/usr/bin/env python3
"""
fetch_ohlc.py  v4.0  —  Daily OHLC bars via yfinance (native 1D, all symbols)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produces:  ohlc-data/{id}.json   — [{time, open, high, low, close, volume}] newest-last
           ohlc-data/meta.json   — {updated_at, symbol_count, errors:[]}

Architecture (v4.0 — simplified):
  All 39 symbols use yfinance native 1D bars. No 1H aggregation, no session
  guards, no boundary logic. yfinance defines the trading day; dashboard.js
  renders it exactly as received and updates the last bar live from quotes.json.

  FX pairs:    Native 1D. Yahoo UTC-midnight convention (00:00–24:00 UTC).
               Candles are consistent with what Yahoo Finance shows by default.
               The LW chart applies the "Forex" session overlay so bars align
               with the real 17:00 NY session boundary visually.
  Non-FX:      Native 1D — unchanged from prior versions.
  Crypto:      Explicit start/end date so yfinance returns Saturday/Sunday bars
               (period="3y" applies NYSE calendar and silently drops weekends).

Schedule: Mon–Fri 21:35 UTC and 22:30 UTC (update-ohlc.yml).
Placement: PUBLIC repo — yfinance only, no API keys.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

SITE_DIR = Path(os.environ.get("SITE_DIR", "."))
OUT_DIR  = SITE_DIR / "ohlc-data"

PERIOD   = "3y"
INTERVAL = "1d"

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
    "gold":   "GC=F",
    "wti":    "CL=F",
    "btc":    "BTC-USD",
    "us10y":  "^TNX",
    # Equity indices
    "spx":    "^GSPC",
    "nasdaq": "^NDX",
    "nikkei": "^N225",
    "stoxx":  "^STOXX50E",
    "vix":    "^VIX",
    # Crypto
    "eth":    "ETH-USD",
    # FX Index
    "dxy":    "DX-Y.NYB",
}

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

GUARDS: dict[str, tuple[float, float]] = {
    "gold":   (500.0,   8000.0),
    "wti":    (10.0,    300.0),
    "btc":    (100.0,   500000.0),
    "us10y":  (0.01,    25.0),
    "spx":    (500.0,   15000.0),
    "nasdaq": (1000.0,  30000.0),
    "nikkei": (5000.0,  80000.0),
    "stoxx":  (1000.0,  8000.0),
    "vix":    (5.0,     90.0),
    "eth":    (10.0,    20000.0),
    "dxy":    (60.0,    150.0),
    "usdjpy": (70.0,    300.0),
    "eurjpy": (100.0,   400.0),
    "gbpjpy": (100.0,   400.0),
    "audjpy": (50.0,    200.0),
    "cadjpy": (50.0,    200.0),
    "nzdjpy": (40.0,    200.0),
    "chfjpy": (80.0,    300.0),
}
FX_GUARD = (0.1, 50.0)

# DXY only: reject bars with impossible intraday ranges (futures roll artifacts)
HL_MAX_SPREAD: dict[str, float] = {
    "dxy": 0.05,
}

# Crypto needs explicit date range for weekend bars
CRYPTO_SYMBOLS = {"btc", "eth"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _guard(id_: str, val: float) -> bool:
    lo, hi = GUARDS.get(id_, FX_GUARD)
    return lo <= val <= hi


# ── Fetch — native 1D for all symbols ─────────────────────────────────────────

def fetch_ohlc(id_: str, ticker_sym: str) -> list[dict] | None:
    """
    Download up to 3 years of native 1D bars for ticker_sym via yfinance.
    Returns a list of {time, open, high, low, close, volume} dicts sorted
    oldest to newest, or None on failure.

    All symbols use native yfinance 1D bars. The bar date and OHLC values are
    exactly what yfinance returns — no session-boundary remapping, no open
    corrections, no 1H aggregation. dashboard.js renders them as-is and applies
    the LW Charts Forex session overlay for FX pairs to align candles visually
    with the 17:00 NY session boundary without modifying the underlying data.

    Crypto (BTC, ETH) uses an explicit start/end date range so that Saturday and
    Sunday bars are included. With period="3y", yfinance applies the NYSE
    business-day calendar and silently omits all weekend bars.
    """
    try:
        ticker = yf.Ticker(ticker_sym)
        dec    = DECIMALS.get(id_, 5)

        if id_ in CRYPTO_SYMBOLS:
            # Explicit date range so yfinance returns weekend bars
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
            # Timezone-safe date extraction.
            # Crypto timestamps from yfinance are timezone-aware (America/New_York).
            # A bar stamped "2026-04-29 00:00:00-04:00" is the Apr 29 UTC session;
            # converting to UTC first avoids a one-day-behind date for crypto bars.
            if id_ in CRYPTO_SYMBOLS and ts.tzinfo is not None:
                date_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
            else:
                date_str = ts.strftime("%Y-%m-%d")

            o = float(row["Open"])
            h = float(row["High"])
            l = float(row["Low"])
            c = float(row["Close"])

            if not (_guard(id_, c) and _guard(id_, o) and h >= l and h >= c and l <= c):
                continue
            if o == h == l == c:
                continue
            if id_ in HL_MAX_SPREAD and l > 0:
                if (h - l) / l > HL_MAX_SPREAD[id_]:
                    continue

            vol = int(row["Volume"]) if "Volume" in row and row["Volume"] == row["Volume"] else 0
            bars.append({
                "time":   date_str,
                "open":   round(o, dec),
                "high":   round(h, dec),
                "low":    round(l, dec),
                "close":  round(c, dec),
                "volume": vol,
            })

        # Deduplicate by date (keep last occurrence — most recent yfinance value)
        seen: set[str] = set()
        deduped: list[dict] = []
        for bar in reversed(bars):
            if bar["time"] not in seen:
                seen.add(bar["time"])
                deduped.append(bar)
        deduped.reverse()

        if len(deduped) < 30:
            print(f"  WARN [{id_}]: only {len(deduped)} valid bars — skipping")
            return None

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
            size_kb = (OUT_DIR / f"{id_}.json").stat().st_size // 1024
            print(f"OK  ({len(bars)} bars, {size_kb}KB)")
            written += 1
        time.sleep(0.4)

    meta = {
        "updated_at":   now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "period":       PERIOD,
        "interval":     INTERVAL,
        "symbol_count": written,
        "errors":       errors,
    }
    write_json(OUT_DIR / "meta.json", meta)

    print()
    print(f"Done — {written}/{len(SYMBOLS)} symbols written.")
    if errors:
        print(f"Errors ({len(errors)}): {', '.join(errors)}")
    if written < len(SYMBOLS) // 2:
        sys.exit(1)


if __name__ == "__main__":
    main()
