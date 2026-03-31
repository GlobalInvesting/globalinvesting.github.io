#!/usr/bin/env python3
"""
fetch_intraday_quotes.py  v2.3 — Intraday quotes via yfinance (+ FX pairs)
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
  US2Y    → yfinance  ^IRX          (13-week T-Bill, proxy de 2Y; el real us2y viene de FRED en repo)
  US3M    → yfinance  ^IRX          (13-week T-Bill = 3M, mismo símbolo)
  US5Y    → yfinance  ^FVX          (US 5Y Treasury yield)
  US30Y   → yfinance  ^TYX          (US 30Y Treasury yield)
  MOVE    → yfinance  ^MOVE         (ICE BofA Bond Volatility Index)
  BTC     → yfinance  BTC-USD       (Bitcoin)
  FX pairs→ yfinance  EURUSD=X etc  (21 pares — reemplaza Stooq bloqueado por CORS)

Por qué yfinance y no Twelve Data/Alpha Vantage:
  • yfinance funciona server-side en GitHub Actions sin CORS ni proxies
  • Cubre todos los índices, commodities, yields y el MOVE sin costo
  • TD free = solo forex/crypto/US stocks (índices y WTI requieren plan pago)
  • AV free = no tiene VIX, SPX, Nikkei, Stoxx, MOVE

Twelve Data (API key opcional):
  • Se usa SOLO como fallback para XAU/USD si yfinance falla en gold
  • Consume 1 crédito máximo por ejecución (de los 800/día del plan free)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import sys
import math
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
    # Cross-Asset panel
    "vix":    "^VIX",
    "spx":    "^GSPC",
    "gold":   "GC=F",
    "wti":    "CL=F",
    "us10y":  "^TNX",
    "nikkei": "^N225",
    "stoxx":  "^STOXX50E",
    "dxy":    "DX-Y.NYB",
    # Risk panel — yield curve
    "us3m":   "^IRX",     # Yahoo ^IRX = 13-week T-Bill yield (3M)
    "us2y":   "^IRX",     # mismo símbolo — proxy de 2Y (el real us2y viene de FRED en repo)
    "us5y":   "^FVX",     # US 5Y Treasury yield (×10 en Yahoo — se divide abajo)
    "us30y":  "^TYX",     # US 30Y Treasury yield (×10 en Yahoo — se divide abajo)
    # Risk panel — bond vol + crypto
    "move":   "^MOVE",    # ICE BofA MOVE Index (bond market volatility)
    "btc":    "BTC-USD",  # Bitcoin — topbar + cross-asset panel
    # FX Majors — reemplazan Stooq (bloqueado por CORS) como fuente para heatmap + quote bar
    "eurusd": "EURUSD=X",
    "gbpusd": "GBPUSD=X",
    "usdjpy": "JPY=X",
    "audusd": "AUDUSD=X",
    "usdchf": "CHF=X",
    "usdcad": "CAD=X",
    "nzdusd": "NZDUSD=X",
    # FX Crosses
    "eurgbp": "EURGBP=X",
    "eurjpy": "EURJPY=X",
    "eurchf": "EURCHF=X",
    "eurcad": "EURCAD=X",
    "euraud": "EURAUD=X",
    "gbpjpy": "GBPJPY=X",
    "gbpchf": "GBPCHF=X",
    "gbpcad": "GBPCAD=X",
    "audjpy": "AUDJPY=X",
    "audnzd": "AUDNZD=X",
    "audchf": "AUDCHF=X",
    "cadjpy": "CADJPY=X",
    "chfjpy": "CHFJPY=X",
    "nzdjpy": "NZDJPY=X",
}

# Yields que Yahoo devuelve ×10 (^TNX=43.42 significa 4.342%) — dividir por 10
# Nota: ^TNX YA está corregido en el map original (yfinance lo devuelve en % directo)
# ^IRX, ^FVX, ^TYX también vienen en % directamente desde yfinance (a diferencia de Yahoo API)
YIELD_DIVIDE_10 = set()   # yfinance Ticker.history ya normaliza — no hace falta dividir

VALIDATORS = {
    "vix":    lambda v: 5 < v < 100,
    "spx":    lambda v: v > 1000,
    "gold":   lambda v: v > 500,
    "wti":    lambda v: 10 < v < 300,
    "us10y":  lambda v: 0 < v < 20,
    "nikkei": lambda v: v > 5000,
    "stoxx":  lambda v: v > 500,
    "dxy":    lambda v: 50 < v < 130,
    "us3m":   lambda v: 0 < v < 20,
    "us2y":   lambda v: 0 < v < 20,
    "us5y":   lambda v: 0 < v < 20,
    "us30y":  lambda v: 0 < v < 20,
    "move":   lambda v: 10 < v < 400,
    "btc":    lambda v: v > 1000,
    # FX pairs
    "eurusd": lambda v: 0.8 < v < 1.6,
    "gbpusd": lambda v: 0.9 < v < 2.0,
    "usdjpy": lambda v: 80 < v < 200,
    "audusd": lambda v: 0.4 < v < 1.2,
    "usdchf": lambda v: 0.5 < v < 1.5,
    "usdcad": lambda v: 0.9 < v < 1.8,
    "nzdusd": lambda v: 0.3 < v < 1.0,
    "eurgbp": lambda v: 0.6 < v < 1.0,
    "eurjpy": lambda v: 100 < v < 200,
    "eurchf": lambda v: 0.8 < v < 1.5,
    "eurcad": lambda v: 1.0 < v < 2.0,
    "euraud": lambda v: 1.0 < v < 2.5,
    "gbpjpy": lambda v: 100 < v < 250,
    "gbpchf": lambda v: 0.8 < v < 1.8,
    "gbpcad": lambda v: 1.0 < v < 2.5,
    "audjpy": lambda v: 50 < v < 120,
    "audnzd": lambda v: 0.8 < v < 1.5,
    "audchf": lambda v: 0.4 < v < 1.0,
    "cadjpy": lambda v: 60 < v < 130,
    "chfjpy": lambda v: 100 < v < 220,
    "nzdjpy": lambda v: 40 < v < 110,
}


# FX pairs para los que calculamos HV30 (los mismos que aparecen en la tabla de Majors)
HV30_FX_PAIRS = [
    "eurusd", "gbpusd", "usdjpy", "audusd", "usdchf", "usdcad", "nzdusd",
    "eurgbp", "eurjpy", "eurchf", "eurcad", "euraud",
    "gbpjpy", "gbpchf", "gbpcad",
    "audjpy", "audnzd", "audchf",
    "cadjpy", "chfjpy", "nzdjpy",
]


def compute_hv30(closes_series):
    """
    Calcula la Historical Volatility anualizada de 30 días a partir de una
    serie de cierres diarios (lista o pandas Series).
    Retorna el valor en porcentaje (ej: 6.4 para 6.4%) o None si hay pocos datos.
    """
    try:
        prices = [float(c) for c in closes_series if c is not None and float(c) > 0]
        # Necesitamos al menos 22 cierres para 21 retornos diarios
        if len(prices) < 22:
            return None
        # Usar los últimos 31 precios → 30 retornos
        window = prices[-31:]
        returns = [math.log(window[i] / window[i - 1]) for i in range(1, len(window))]
        n = len(returns)
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
        hv_daily = math.sqrt(variance)
        hv_annual = hv_daily * math.sqrt(252) * 100  # en %
        return round(hv_annual, 2)
    except Exception:
        return None


# Pairs for which we compute rolling 60-day Pearson correlation
CORRELATION_PAIRS = [
    ("eurusd", "dxy"),
    ("audusd", "gold"),
    ("usdjpy", "us10y"),
    ("gbpusd", "eurusd"),
    ("usdcad", "wti"),
]

# Human-readable labels for the frontend
CORRELATION_LABELS = {
    ("eurusd", "dxy"):    ("EUR/USD", "DXY"),
    ("audusd", "gold"):   ("AUD/USD", "Gold"),
    ("usdjpy", "us10y"):  ("USD/JPY", "US 10Y"),
    ("gbpusd", "eurusd"): ("GBP/USD", "EUR/USD"),
    ("usdcad", "wti"):    ("USD/CAD", "WTI Oil"),
}

# All unique symbols needed for correlation (merged with FX pairs)
CORR_SYMBOLS = {k: v for k, v in YFINANCE_SYMBOLS.items()
                if any(k in pair for pair in CORRELATION_PAIRS)}


def pearson(x, y):
    """Compute Pearson correlation coefficient between two equal-length lists."""
    n = len(x)
    if n < 10:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return None
    return round(num / (den_x * den_y), 3)


def fetch_correlations():
    """
    Downloads 90 days of daily closes for each symbol used in correlation pairs
    and computes rolling 60-day Pearson correlation.
    Returns a list of dicts ready for quotes.json:
      [{ "a": "EUR/USD", "b": "DXY", "corr": -0.97, "n": 60 }, ...]
    """
    print("\n[Correlations] Computing rolling 60-day correlations...")

    # Fetch 90d history for all needed symbols
    series = {}
    for sym_id, yf_sym in CORR_SYMBOLS.items():
        try:
            ticker = yf.Ticker(yf_sym)
            hist = ticker.history(period="3mo", interval="1d", auto_adjust=True)
            if hist.empty or len(hist) < 15:
                print(f"[Correlations] {sym_id}: insufficient data ({len(hist)} days)")
                continue
            series[sym_id] = hist["Close"].dropna().tolist()
            print(f"[Correlations] ✓ {sym_id:8s} ({yf_sym}): {len(series[sym_id])} closes")
        except Exception as e:
            print(f"[Correlations] Error fetching {sym_id}: {e}")

    results = []
    for sym_a, sym_b in CORRELATION_PAIRS:
        labels = CORRELATION_LABELS[(sym_a, sym_b)]
        if sym_a not in series or sym_b not in series:
            print(f"[Correlations] Skipping {sym_a}/{sym_b} — missing data")
            results.append({"a": labels[0], "b": labels[1], "corr": None, "n": 0})
            continue
        # Align to shortest series, use last 60 points
        sa, sb = series[sym_a], series[sym_b]
        n = min(len(sa), len(sb), 60)
        sa60, sb60 = sa[-n:], sb[-n:]
        corr = pearson(sa60, sb60)
        status = f"{corr:+.3f}" if corr is not None else "N/A"
        print(f"[Correlations] {labels[0]:8s} vs {labels[1]:8s}: {status}  (n={n})")
        results.append({"a": labels[0], "b": labels[1], "corr": corr, "n": n})

    return results


def fetch_hv30_fx(fx_pairs):
    """
    Descarga 90 días de historia diaria para cada par FX y calcula HV30.
    Retorna dict { pair_id: hv30_value_or_None }.
    """
    print("\n[HV30] Calculando volatilidad histórica 30d para pares FX...")
    hv30_results = {}
    yf_map = {k: v for k, v in YFINANCE_SYMBOLS.items() if k in fx_pairs}

    for pair_id, yf_sym in yf_map.items():
        try:
            ticker = yf.Ticker(yf_sym)
            hist = ticker.history(period="3mo", interval="1d", auto_adjust=True)
            if hist.empty or len(hist) < 22:
                print(f"[HV30] {pair_id}: insuficientes datos ({len(hist)} días)")
                hv30_results[pair_id] = None
                continue
            closes = hist["Close"].dropna().tolist()
            hv = compute_hv30(closes)
            hv30_results[pair_id] = hv
            status = f"{hv:.2f}%" if hv is not None else "N/A"
            print(f"[HV30] ✓ {pair_id:8s} ({yf_sym}): {status}  ({len(closes)} cierres)")
        except Exception as e:
            print(f"[HV30] Error en {pair_id}: {e}")
            hv30_results[pair_id] = None

    return hv30_results


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

                # high/low del día más reciente disponible
                highs = hist["High"].dropna()
                lows  = hist["Low"].dropna()
                day_high = round(float(highs.iloc[-1]), 4) if len(highs) >= 1 else None
                day_low  = round(float(lows.iloc[-1]),  4) if len(lows)  >= 1 else None

                results[internal_id] = {
                    "close":      round(close, 4),
                    "prev_close": round(prev_close, 4),
                    "chg":        round(chg, 4),
                    "pct":        round(pct, 4),
                    "high":       day_high,
                    "low":        day_low,
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
    print(f"\n{'='*60}\nfetch_intraday_quotes.py  v2.3  —  {ts}\n{'='*60}\n")

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

    # PASO 4 (FIX W-05): Reemplazar us2y con el Treasury 2Y real desde FRED:DGS2
    # ^IRX es el T-Bill de 13 semanas (3M), no el Treasury 2Y.
    # extended-data/USD.json contiene bond2y desde FRED:DGS2 (daily), que es el dato correcto.
    try:
        usd_ext_path = os.path.join(site_path, "extended-data", "USD.json")
        with open(usd_ext_path) as f:
            usd_ext = json.load(f)
        bond2y = usd_ext.get("data", {}).get("bond2y")
        bond2y_date = usd_ext.get("dates", {}).get("bond2y", "")
        if bond2y and isinstance(bond2y, (int, float)) and 0 < bond2y < 20:
            existing_us2y = quotes.get("us2y", {})
            quotes["us2y"] = {
                "close":      round(float(bond2y), 4),
                "prev_close": existing_us2y.get("prev_close", round(float(bond2y), 4)),
                "chg":        existing_us2y.get("chg", 0.0),
                "pct":        existing_us2y.get("pct", 0.0),
                "source":     "fred_dgs2",
                "sourceDate": bond2y_date,
            }
            print(f"[us2y] FRED:DGS2 = {bond2y:.4f}% ({bond2y_date}) — reemplaza proxy ^IRX")
        else:
            print("[us2y] bond2y no disponible en USD.json — manteniendo proxy ^IRX")
    except Exception as e:
        print(f"[us2y] No se pudo leer extended-data/USD.json: {e}")

    sources = set(q.get("source") for q in quotes.values())
    source_label = "yfinance" if sources <= {"yfinance"} else ("repo" if sources == {"repo"} else "mixed")

    # PASO 5: Calcular HV30 para pares FX e inyectar en cada quote
    hv30_data = fetch_hv30_fx(HV30_FX_PAIRS)
    hv30_output = {}
    for pair_id, hv in hv30_data.items():
        hv30_output[pair_id] = hv  # None si no se pudo calcular
        if pair_id in quotes and hv is not None:
            quotes[pair_id]["hv30"] = hv

    # PASO 6: Calcular correlaciones rolling 60d
    correlations = fetch_correlations()

    output = {"updated": ts, "source": source_label, "quotes": quotes,
              "hv30": hv30_output, "correlations": correlations}
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ {len(quotes)}/{len(YFINANCE_SYMBOLS)} símbolos → {out_file}  [fuente: {source_label}]")
    for sym in YFINANCE_SYMBOLS:
        q = quotes.get(sym)
        if q:
            stale = " [STALE]" if q.get("stale") else ""
            hi  = f"  H:{q['high']:.4f}"  if q.get("high")  is not None else ""
            lo  = f"  L:{q['low']:.4f}"   if q.get("low")   is not None else ""
            hv  = f"  HV30:{q['hv30']:.1f}%" if q.get("hv30") is not None else ""
            print(f"   {sym:8s}  {q['close']:>12.4f}  {q['pct']:+.2f}%  [{q['source']}]{hi}{lo}{hv}{stale}")
        else:
            print(f"   {sym:8s}  — sin datos")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
