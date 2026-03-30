#!/usr/bin/env python3
"""
fetch_intraday_quotes.py  v2.0 — Intraday quotes via yfinance + Twelve Data (forex/gold)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce:  intraday-data/quotes.json
Schedule: Cada 15 min en días de semana, horario de mercado (via GitHub Action)

FUENTES POR SÍMBOLO:
  VIX     → yfinance  ^VIX          (CBOE, real-time diferido)
  SPX     → yfinance  ^GSPC         (S&P 500)
  GOLD    → yfinance  GC=F          (Gold futures) + fallback Twelve Data XAU/USD
  WTI     → yfinance  CL=F          (WTI Crude front-month futures)
  US10Y   → yfinance  ^TNX          (CBOE 10Y Treasury Yield — ya en %, no ×10)
  NIKKEI  → yfinance  ^N225
  STOXX   → yfinance  ^STOXX50E
  DXY     → yfinance  DX-Y.NYB

Por qué yfinance y no Twelve Data/Alpha Vantage:
  • yfinance funciona server-side en GitHub Actions sin CORS ni proxies
  • Cubre todos los índices, commodities y yields sin costo
  • TD free = solo forex/crypto/US stocks (índices y WTI requieren plan pago)
  • AV free = no tiene VIX, SPX, Nikkei, Stoxx

Twelve Data (API key opcional):
  • Se usa SOLO como fallback para XAU/USD si yfinance falla en gold
  • Consume 1 crédito máximo por ejecución (de los 800/día del plan free)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import sys
from datetime import datetime, timezone

try:
    import yfinance as yf
except ImportError:
    print("[ERROR] yfinance no instalado. Correr: pip install yfinance")
    sys.exit(1)

try:
    import requests
except ImportError:
    requests = None

TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")

YFINANCE_SYMBOLS = {
    "vix":    "^VIX",
    "spx":    "^GSPC",
    "gold":   "GC=F",
    "wti":    "CL=F",
    "us10y":  "^TNX",
    "nikkei": "^N225",
    "stoxx":  "^STOXX50E",
    "dxy":    "DX-Y.NYB",
}

VALIDATORS = {
    "vix":    lambda v: 5 < v < 100,
    "spx":    lambda v: v > 1000,
    "gold":   lambda v: v > 500,
    "wti":    lambda v: 10 < v < 300,
    "us10y":  lambda v: 0 < v < 20,
    "nikkei": lambda v: v > 5000,
    "stoxx":  lambda v: v > 500,
    "dxy":    lambda v: 50 < v < 130,
}


def fetch_yfinance_all(symbols_map):
    yf_tickers = list(symbols_map.values())
    print(f"[yfinance] Descargando: {' '.join(yf_tickers)}")

    results = {}
    try:
        # Descargar cada ticker individualmente para evitar problemas de columnas multinivel
        for internal_id, yf_sym in symbols_map.items():
            try:
                ticker = yf.Ticker(yf_sym)
                hist = ticker.history(period="5d", interval="1d", auto_adjust=True)

                if hist.empty:
                    print(f"[yfinance] Sin datos para {yf_sym}")
                    results[internal_id] = None
                    continue

                closes = hist["Close"].dropna()
                if len(closes) < 1:
                    results[internal_id] = None
                    continue

                close      = float(closes.iloc[-1])
                prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else close
                chg        = close - prev_close
                pct        = (chg / prev_close * 100) if prev_close != 0 else 0.0

                validator = VALIDATORS.get(internal_id)
                if validator and not validator(close):
                    print(f"[yfinance] {internal_id} fuera de rango: {close}")
                    results[internal_id] = None
                    continue

                results[internal_id] = {
                    "close":      round(close, 4),
                    "prev_close": round(prev_close, 4),
                    "chg":        round(chg, 4),
                    "pct":        round(pct, 4),
                    "source":     "yfinance",
                }
                print(f"[yfinance] ✓ {internal_id:8s} ({yf_sym}): {close:.4f}  {pct:+.2f}%")

            except Exception as e:
                print(f"[yfinance] Error en {yf_sym}: {e}")
                results[internal_id] = None

    except Exception as e:
        print(f"[yfinance] Error general: {e}")

    return results


def fetch_td_gold():
    if not TWELVE_DATA_API_KEY or requests is None:
        return None

    print("[TwelveData] Fallback para gold (XAU/USD)...")
    try:
        r = requests.get(
            "https://api.twelvedata.com/quote",
            params={"symbol": "XAU/USD", "apikey": TWELVE_DATA_API_KEY, "dp": 4},
            timeout=10,
        )
        r.raise_for_status()
        raw = r.json()
        if raw.get("status") == "error" or not raw.get("close"):
            print(f"[TwelveData] Error: {raw.get('message', 'sin datos')}")
            return None

        close      = float(raw["close"])
        prev_close = float(raw.get("previous_close") or raw.get("open") or close)
        chg        = close - prev_close
        pct        = (chg / prev_close * 100) if prev_close != 0 else 0.0

        if not VALIDATORS["gold"](close):
            return None

        print(f"[TwelveData] ✓ gold: {close:.4f}")
        return {"close": round(close,4), "prev_close": round(prev_close,4),
                "chg": round(chg,4), "pct": round(pct,4), "source": "twelve_data"}
    except Exception as e:
        print(f"[TwelveData] Error: {e}")
        return None


def load_repo_fallback(site_path):
    path = os.path.join(site_path, "extended-data", "USD.json")
    try:
        with open(path) as f:
            d = json.load(f).get("data", {})
        fb = {}
        if d.get("vix") and VALIDATORS["vix"](d["vix"]):
            fb["vix"]   = {"close": d["vix"],     "prev_close": d["vix"],     "chg": 0, "pct": 0, "source": "repo", "stale": True}
        if d.get("bond10y") and VALIDATORS["us10y"](d["bond10y"]):
            fb["us10y"] = {"close": d["bond10y"], "prev_close": d["bond10y"], "chg": 0, "pct": 0, "source": "repo", "stale": True}
        print(f"[Repo] {len(fb)} fallbacks desde USD.json")
        return fb
    except Exception as e:
        print(f"[Repo] No se pudo leer USD.json: {e}")
        return {}


def main():
    site_path = os.environ.get("SITE_PATH", ".")
    out_dir   = os.path.join(site_path, "intraday-data")
    out_file  = os.path.join(out_dir, "quotes.json")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*60}\nfetch_intraday_quotes.py  v2.0  —  {ts}\n{'='*60}\n")

    quotes = {}

    # PASO 1: yfinance (fuente principal — todos los símbolos)
    yf_results = fetch_yfinance_all(YFINANCE_SYMBOLS)
    for k, v in yf_results.items():
        if v is not None:
            quotes[k] = v

    # PASO 2: Fallback TD para gold si yfinance falló
    if "gold" not in quotes:
        td = fetch_td_gold()
        if td:
            quotes["gold"] = td

    # PASO 3: Repo fallback para lo restante
    missing = [k for k in YFINANCE_SYMBOLS if k not in quotes]
    if missing:
        print(f"\n[Repo] Sin datos para: {missing}")
        for k, v in load_repo_fallback(site_path).items():
            if k in missing:
                quotes[k] = v

    sources = set(q.get("source") for q in quotes.values())
    source_label = "yfinance" if sources <= {"yfinance"} else ("repo" if sources == {"repo"} else "mixed")

    output = {"updated": ts, "source": source_label, "quotes": quotes}
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ {len(quotes)}/{len(YFINANCE_SYMBOLS)} símbolos → {out_file}  [fuente: {source_label}]")
    for sym in YFINANCE_SYMBOLS:
        q = quotes.get(sym)
        if q:
            stale = " [STALE]" if q.get("stale") else ""
            print(f"   {sym:8s}  {q['close']:>12.4f}  {q['pct']:+.2f}%  [{q['source']}]{stale}")
        else:
            print(f"   {sym:8s}  — sin datos")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
