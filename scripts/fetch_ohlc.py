#!/usr/bin/env python3
"""
fetch_ohlc.py  v1.1 — Daily OHLC history for Lightweight Charts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Downloads 3 years of daily OHLC bars via yfinance for all symbols
used by the Lightweight Charts panel (replaces TradingView widget).

Symbols covered:
  FX Majors   (7): EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, NZD/USD
  FX Crosses (21): all pairs in PAIRS[] that have chart click support
  Cross-asset (4): Gold (XAUUSD=X), WTI (CL=F), BTC (BTC-USD), US 10Y (^TNX)
  Indices     (4): S&P 500 (^GSPC), Nasdaq (^IXIC), Nikkei 225 (^N225), EuroStoxx 50 (^STOXX50E)
  Crypto      (1): ETH/USD (ETH-USD)
  FX Index    (1): DXY (DX-Y.NYB)

Output:
  ohlc-data/{id}.json   — [{time, open, high, low, close}] newest-last
  ohlc-data/meta.json   — {updated_at, symbol_count, errors:[]}

Each file is REPLACED on every run (not appended) — repo stays fixed size.
Total size estimate: ~38 files × ~45 KB raw = ~1.7 MB, ~560 KB gzipped.

Schedule: daily at 22:30 UTC (after NY close, before Sydney open).
          Runs in the ENGINE private repo — pushes output to PUBLIC site repo.
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
    "gold":  "GC=F",       # Gold front-month futures (XAUUSD=X delisted on Yahoo Finance)
    "wti":   "CL=F",       # WTI Crude front-month futures
    "btc":   "BTC-USD",    # Bitcoin vs USD
    "us10y": "^TNX",       # US 10-Year Treasury yield
    # Equity indices
    "spx":    "^GSPC",     # S&P 500
    "nasdaq": "^IXIC",     # Nasdaq Composite
    "nikkei": "^N225",     # Nikkei 225
    "stoxx":  "^STOXX50E", # EuroStoxx 50
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
    "spx": 2, "nasdaq": 2, "nikkei": 2, "stoxx": 2,
    "eth": 2, "dxy": 3,
}

# Plausibility guards — reject bars outside these ranges
GUARDS: dict[str, tuple[float, float]] = {
    "gold":  (500.0,   8000.0),
    "wti":   (10.0,    300.0),
    "btc":   (100.0,   500000.0),
    "us10y": (0.01,    25.0),
    "spx":   (500.0,   15000.0),
    "nasdaq":(500.0,   30000.0),
    "nikkei":(5000.0,  80000.0),
    "stoxx": (1000.0,  8000.0),
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

# ── Helpers ────────────────────────────────────────────────────────────────────

def _guard(id_: str, val: float) -> bool:
    lo, hi = GUARDS.get(id_, FX_GUARD)
    return lo <= val <= hi


# FX spot pairs — yfinance returns corrupt open values (open ≈ close, not true session open).
# These need the prev-close reconstruction pass below. Cross-asset symbols are NOT included:
# BTC/ETH/SPX/etc. receive proper open prices from their respective exchanges.
_FX_SPOT_IDS: frozenset[str] = frozenset({
    "eurusd","gbpusd","usdjpy","audusd","usdcad","usdchf","nzdusd",
    "eurgbp","eurjpy","eurchf","eurcad","euraud","eurnzd","gbpjpy",
    "gbpchf","gbpcad","gbpaud","gbpnzd","audjpy","audnzd","audchf",
    "audcad","cadjpy","cadchf","nzdjpy","nzdcad","nzdchf","chfjpy","dxy",
})


def fetch_ohlc(id_: str, ticker_sym: str) -> list[dict] | None:
    """
    Download 3 years of daily bars for ticker_sym.
    Returns a list of {time, open, high, low, close} dicts sorted oldest→newest,
    or None on failure.

    FX open reconstruction
    ─────────────────────
    Yahoo Finance FX daily bars carry a known data-quality defect: the Open
    field is frequently set to the same value as Close (or within 0.03% of it),
    even when the High-Low range shows meaningful intraday movement.  This
    produces hundreds of doji/hammer candles that misrepresent actual price
    action.

    Root cause: Yahoo uses the last tick of the previous UTC day as the open
    price for FX spot pairs, making open ≈ previous-close ≈ current-close on
    low-gap days, and open ≈ close (duplicate) on high-gap days.

    Fix (FX only, applied after deduplication):
      For any bar where |open − close| / close < 0.03 % AND the wick (H−L)
      is non-trivial (> 0.05 % of close), replace open with the previous
      bar's close clamped to [low, high].  Clamping ensures the resulting
      OHLC relationship is always valid.  Cross-asset symbols (BTC, SPX,
      gold, etc.) are excluded — their open values are correct.
    """
    try:
        ticker = yf.Ticker(ticker_sym)
        hist = ticker.history(period=PERIOD, interval=INTERVAL, auto_adjust=True)
        if hist.empty:
            print(f"  WARN [{id_}]: empty history from yfinance")
            return None

        bars: list[dict] = []
        for ts, row in hist.iterrows():
            # ts is a pandas Timestamp — convert to date string
            date_str = ts.strftime("%Y-%m-%d")
            o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])

            # Skip bars with invalid OHLC relationships or out-of-range prices
            if not (_guard(id_, c) and _guard(id_, o) and h >= l and h >= c and l <= c):
                continue
            # Skip bars where all four values are identical (stale/placeholder bar)
            if o == h == l == c:
                continue

            dec = DECIMALS.get(id_, 5)
            bars.append({
                "time":  date_str,
                "open":  round(o, dec),
                "high":  round(h, dec),
                "low":   round(l, dec),
                "close": round(c, dec),
            })

        # Deduplicate by date (keep last occurrence — intraday quirk for today's bar)
        seen: set[str] = set()
        deduped: list[dict] = []
        for bar in reversed(bars):
            if bar["time"] not in seen:
                seen.add(bar["time"])
                deduped.append(bar)
        deduped.reverse()   # oldest → newest

        # ── FX open reconstruction ────────────────────────────────────────────
        # Applied after deduplication so prev_close is always the preceding
        # calendar bar (no duplicate-date interference).
        if id_ in _FX_SPOT_IDS:
            for i in range(1, len(deduped)):
                b    = deduped[i]
                prev = deduped[i - 1]
                wick = b["high"] - b["low"]
                body = abs(b["open"] - b["close"])
                # Threshold: body < 0.03 % of close price AND wick > 0.05 % of close
                # (filters genuine low-volatility dojis from the data-artifact dojis)
                if wick > 0 and body / b["close"] < 0.0003 and wick / b["close"] > 0.0005:
                    dec  = DECIMALS.get(id_, 5)
                    # Use prev_close clamped to [low, high] — always a valid open
                    new_open = max(b["low"], min(b["high"], prev["close"]))
                    deduped[i] = {**b, "open": round(new_open, dec)}

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
