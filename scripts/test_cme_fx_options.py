#!/usr/bin/env python3
"""
test_cme_fx_options.py — CME FX futures options via yfinance
============================================================
Tests whether yfinance can fetch ATM IV from CME FX futures option chains.

This is a DROP-IN REPLACEMENT investigation for the current ETF-based approach
in fetch_intraday_quotes.py (FXE/FXB/FXY/FXA/FXF/FXC).

Candidate sources:
  A) Continuous front-month:   6E=F, 6B=F, 6J=F, 6A=F, 6C=F, 6S=F, 6N=F
  B) Specific contract:        6EJ26, 6BJ26, 6JJ26 ... (current Apr 2026)
  C) Next quarterly:           6EM26, 6BM26, 6JM26 ... (Jun 2026 — more liquid)

Run on GitHub Actions:
  pip install yfinance --break-system-packages
  python3 test_cme_fx_options.py

Output: prints per-symbol result — IV%, expiry, ATM strike, source label.
"""

import sys
from datetime import date, timedelta

try:
    import yfinance as yf
    print(f"[INFO] yfinance {yf.__version__}")
except ImportError:
    print("[ERROR] yfinance not installed")
    sys.exit(1)

# ── Symbol map: pair → candidates to try in order ─────────────────────────────
today = date.today()

# IMM month codes
MONTH_CODE = {1:'F',2:'G',3:'H',4:'J',5:'K',6:'M',
               7:'N',8:'Q',9:'U',10:'V',11:'X',12:'Z'}

def imm_contracts(root, n=3):
    """Return next n quarterly IMM contract symbols (Mar/Jun/Sep/Dec)."""
    contracts = []
    yr, mo = today.year, today.month
    for _ in range(n * 4):
        mo += 1
        if mo > 12:
            mo = 1
            yr += 1
        if mo in (3, 6, 9, 12):
            code = MONTH_CODE[mo]
            contracts.append(f"{root}{code}{str(yr)[2:]}")
        if len(contracts) >= n:
            break
    return contracts

CME_FX_MAP = {
    # pair_id  root    invert   note
    "eurusd": ("6E",  False,   "EUR/USD — 125,000 EUR contract"),
    "gbpusd": ("6B",  False,   "GBP/USD — 62,500 GBP contract"),
    "usdjpy": ("6J",  True,    "JPY/USD → invert for USD/JPY — 12.5M JPY contract"),
    "audusd": ("6A",  False,   "AUD/USD — 100,000 AUD contract"),
    "usdcad": ("6C",  True,    "CAD/USD → invert for USD/CAD — 100,000 CAD contract"),
    "usdchf": ("6S",  True,    "CHF/USD → invert for USD/CHF — 125,000 CHF contract"),
    "nzdusd": ("6N",  False,   "NZD/USD — 100,000 NZD contract"),
}

MIN_DAYS_TO_EXPIRY = 4
IV_MIN, IV_MAX = 3.0, 40.0

# ── Helper: fetch option chain and extract ATM IV ──────────────────────────────

def fetch_iv_from_symbol(sym):
    """
    Try to get ATM IV from yfinance option chain for a given symbol.
    Returns dict with iv, expiry, atm_strike, source — or None.
    """
    try:
        ticker = yf.Ticker(sym)

        # 1. Resolve spot price
        hist = ticker.history(period="2d", interval="1d")
        if hist.empty:
            print(f"    {sym}: no price history")
            return None
        spot = float(hist["Close"].iloc[-1])
        if spot <= 0:
            return None

        # 2. Get available expirations
        exps = ticker.options
        if not exps:
            print(f"    {sym}: no options listed")
            return None

        # 3. Pick nearest expiry ≥ MIN_DAYS_TO_EXPIRY
        chosen_exp = None
        for exp_str in exps:
            try:
                exp_date = date.fromisoformat(exp_str)
                if (exp_date - today).days >= MIN_DAYS_TO_EXPIRY:
                    chosen_exp = exp_str
                    break
            except ValueError:
                continue

        if not chosen_exp:
            print(f"    {sym}: no valid expiry (all < {MIN_DAYS_TO_EXPIRY}d)")
            return None

        # 4. Fetch option chain
        chain = ticker.option_chain(chosen_exp)
        calls = chain.calls

        if calls.empty:
            print(f"    {sym}: empty calls chain")
            return None

        # 5. ATM strike
        strikes = calls["strike"].dropna().tolist()
        if not strikes:
            return None
        atm_strike = min(strikes, key=lambda s: abs(s - spot))

        # 6. Extract IV from ATM call; fallback to put
        atm_calls = calls[calls["strike"] == atm_strike]
        iv_raw = None
        if not atm_calls.empty:
            v = atm_calls["impliedVolatility"].iloc[0]
            if v and v == v and v > 0:
                iv_raw = v

        if iv_raw is None:
            puts = chain.puts
            atm_puts = puts[puts["strike"] == atm_strike]
            if not atm_puts.empty:
                v = atm_puts["impliedVolatility"].iloc[0]
                if v and v == v and v > 0:
                    iv_raw = v

        if iv_raw is None:
            print(f"    {sym}: IV is null at ATM strike {atm_strike}")
            return None

        # yfinance IV is decimal (0.074 → 7.4%)
        iv_pct = round(float(iv_raw) * 100, 2)

        if not (IV_MIN <= iv_pct <= IV_MAX):
            print(f"    {sym}: IV {iv_pct}% outside [{IV_MIN},{IV_MAX}] — skipping")
            return None

        days_to_exp = (date.fromisoformat(chosen_exp) - today).days
        print(f"    {sym}: ✓ IV={iv_pct:.2f}%  spot={spot:.5f}  K={atm_strike}  exp={chosen_exp} ({days_to_exp}d)  calls={len(calls)}")
        return {
            "iv":      iv_pct,
            "expiry":  chosen_exp,
            "atm":     atm_strike,
            "spot":    round(spot, 6),
            "source":  f"CME {sym} options",
            "n_calls": len(calls),
        }

    except Exception as e:
        print(f"    {sym}: exception — {e}")
        return None


# ── Main test loop ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("CME FX FUTURES OPTIONS — yfinance probe")
print(f"Date: {today}")
print("=" * 65)

results = {}

for pair_id, (root, invert, note) in CME_FX_MAP.items():
    print(f"\n[{pair_id.upper()}] {note}")

    # Candidate symbols: continuous, then next 2 IMM quarterlies
    continuous = f"{root}=F"
    quarterlies = imm_contracts(root, n=2)
    candidates = [continuous] + quarterlies

    print(f"  Trying: {candidates}")
    best = None
    best_sym = None

    for sym in candidates:
        result = fetch_iv_from_symbol(sym)
        if result:
            best = result
            best_sym = sym
            break  # take first that works

    if best:
        results[pair_id] = best
        print(f"  → PASS  {pair_id}: IV={best['iv']:.2f}%  source={best['source']}")
    else:
        results[pair_id] = None
        print(f"  → FAIL  {pair_id}: no IV retrieved from any candidate")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
passed = 0
for pair_id, r in results.items():
    if r:
        passed += 1
        print(f"  ✓ {pair_id:8s}  IV={r['iv']:5.2f}%  exp={r['expiry']}  K={r['atm']}  ({r['source']})")
    else:
        print(f"  ✗ {pair_id:8s}  UNAVAILABLE")

print(f"\n{passed}/{len(CME_FX_MAP)} pairs resolved via CME futures options")
if passed >= 4:
    print("→ VIABLE: sufficient coverage to replace ETF IV source")
elif passed >= 2:
    print("→ PARTIAL: usable for some pairs, ETF fallback needed for others")
else:
    print("→ NOT VIABLE: yfinance cannot access CME FX option chains from GH Actions")
