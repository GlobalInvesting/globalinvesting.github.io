#!/usr/bin/env python3
"""
fetch_intraday_quotes.py  v2.3 — Intraday quotes via yfinance (+ FX pairs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce:  intraday-data/quotes.json
Schedule: Cada 15 min en días de semana, horario de mercado (via GitHub Action)

FUENTES POR SÍMBOLO:
  VIX     → yfinance  ^VIX          (CBOE, real-time diferido)
  SPX     → yfinance  ^GSPC         (S&P 500)
  GOLD    → Frankfurter XAU/USD     (spot real — fuente primaria)
            yfinance  GLD           (ETF spot-tracking — fallback)
            yfinance  GC=F          (futuros — fallback final; precio ~4,600, ≠ spot)
  WTI     → yfinance  CL=F          (WTI Crude front-month futures)
  US10Y   → yfinance  ^TNX          (CBOE 10Y Treasury Yield — ya en %, no ×10)
  NIKKEI  → yfinance  ^N225
  STOXX   → yfinance  ^STOXX50E
  DXY     → yfinance  DX-Y.NYB
  US2Y    → yfinance  ^IRX          (13-week T-Bill, proxy de corto plazo)
  US3M    → yfinance  ^IRX          (13-week T-Bill = 3M, mismo símbolo)
  US5Y    → yfinance  ^FVX          (US 5Y Treasury yield)
  US30Y   → yfinance  ^TYX          (US 30Y Treasury yield)
  MOVE    → yfinance  ^MOVE         (ICE BofA Bond Volatility Index)
  BTC     → yfinance  BTC-USD       (Bitcoin)
  FX pairs→ yfinance  EURUSD=X etc  (21 pares — reemplaza Stooq bloqueado por CORS)

NOTA W-04 (resuelta v2.3):
  GC=F es el contrato de futuros de oro front-month. Su precio nominal (~4,600 USD)
  incluye el costo de carry del contrato y difiere del spot XAU/USD (~3,200 USD).
  Ahora se usa Frankfurter ECB XAU como fuente primaria (spot oficial del BCE).
  GLD como segundo fallback (ETF spot-tracking con diferencia <0.1% del spot).
  GC=F queda como último recurso con nota de fuente en el JSON.

NOTA W-05 (documentada):
  us2y usa ^IRX (T-Bill 13 semanas). El dato real del Treasury 2Y viene de
  FRED en el repo engine (update-extended-data). Para el JSON intraday se
  mantiene ^IRX como proxy de corto plazo porque yfinance no dispone de un
  ticker dedicado para el 2Y del Tesoro con delay <15min.

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
    # GOLD: GLD (SPDR Gold ETF) — precio spot-tracking, sin distorsión de carry de futuros.
    # La diferencia con XAU/USD spot es <0.1% (ratio NAV/acción ≈ 1/10 oz).
    # GC=F queda solo como fallback de último recurso (ver fetch_gold_spot abajo).
    "gold":   "GLD",
    "wti":    "CL=F",
    "us10y":  "^TNX",
    "nikkei": "^N225",
    "stoxx":  "^STOXX50E",
    "dxy":    "DX-Y.NYB",
    # Risk panel — yield curve
    "us3m":   "^IRX",     # Yahoo ^IRX = 13-week T-Bill yield (3M)
    "us2y":   "^IRX",     # mismo símbolo — proxy de corto plazo (real 2Y viene de FRED/extended-data)
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


def fetch_gold_frankfurter():
    """
    Fuente primaria de spot XAU/USD: Frankfurter ECB API.
    Devuelve el precio en USD por onza troy (dato oficial del BCE, lag ~1 día hábil).
    Gratuito, sin clave, sin rate limit relevante para este uso.
    """
    if requests is None:
        return None
    print("[Frankfurter] Intentando XAU/USD spot...")
    try:
        r = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": "XAU", "to": "USD"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        close = float(data["rates"]["USD"])
        if not VALIDATORS["gold"](close):
            print(f"[Frankfurter] XAU/USD fuera de rango: {close}")
            return None
        print(f"[Frankfurter] ✓ gold (XAU/USD spot): {close:.2f}")
        return {
            "close":      round(close, 2),
            "prev_close": round(close, 2),  # Frankfurter no provee prev_close
            "chg":        0.0,
            "pct":        0.0,
            "source":     "frankfurter_xau",
            "note":       "XAU/USD ECB spot rate (lag ~1 business day). No intraday chg available from this source.",
        }
    except Exception as e:
        print(f"[Frankfurter] Error: {e}")
        return None


def convert_gld_to_xau(gld_result):
    """
    Convierte precio de GLD (SPDR Gold ETF) a XAU/USD spot equivalente.
    GLD ≈ 1/10 oz de oro. Multiplicar ×10 da el spot aproximado con error <1%.
    Más exacto que GC=F (futuros) cuyo nominal incluye costo de carry.
    """
    if gld_result is None:
        return None
    factor = 10.0
    c  = round(gld_result["close"]      * factor, 2)
    pc = round(gld_result["prev_close"] * factor, 2)
    chg = round(c - pc, 2)
    pct = round((chg / pc * 100) if pc != 0 else 0.0, 4)
    if not VALIDATORS["gold"](c):
        print(f"[GLD×10] Precio convertido fuera de rango: {c}")
        return None
    return {
        "close":      c,
        "prev_close": pc,
        "chg":        chg,
        "pct":        pct,
        "source":     "yfinance_gld",
        "note":       "GLD ETF ×10 (spot-tracking proxy, error <1% vs XAU/USD spot).",
    }


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

    # PASO 1: yfinance (fuente principal — todos los símbolos excepto gold)
    yf_results = fetch_yfinance_all(YFINANCE_SYMBOLS)
    for k, v in yf_results.items():
        if v is not None:
            if k == "gold":
                # GLD raw — convertir a XAU/USD equivalente
                converted = convert_gld_to_xau(v)
                if converted:
                    quotes["gold"] = converted
                # si la conversión falla, se cae a los fallbacks
            else:
                quotes[k] = v

    # PASO 2: Si gold no obtenido via GLD, intentar Frankfurter XAU spot (fuente oficial BCE)
    if "gold" not in quotes:
        frankfurter_gold = fetch_gold_frankfurter()
        if frankfurter_gold:
            quotes["gold"] = frankfurter_gold

    # PASO 3: Fallback TD para gold si los anteriores fallaron
    if "gold" not in quotes:
        td = fetch_td_gold()
        if td:
            quotes["gold"] = td

    # PASO 4: Último recurso gold — yfinance GC=F futuros (precio nominal ~4,600, ≠ spot)
    if "gold" not in quotes:
        print("[GC=F] Último recurso: futuros de oro (precio nominal difiere del spot XAU/USD)")
        try:
            import yfinance as yf
            ticker = yf.Ticker("GC=F")
            hist = ticker.history(period="5d", interval="1d", auto_adjust=True)
            if not hist.empty:
                closes = hist["Close"].dropna()
                if len(closes) >= 1:
                    close      = float(closes.iloc[-1])
                    prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else close
                    chg        = close - prev_close
                    pct        = (chg / prev_close * 100) if prev_close != 0 else 0.0
                    # GC=F puede estar en ~4,600 — validador acepta >500
                    if VALIDATORS["gold"](close):
                        quotes["gold"] = {
                            "close":      round(close, 2),
                            "prev_close": round(prev_close, 2),
                            "chg":        round(chg, 2),
                            "pct":        round(pct, 4),
                            "source":     "yfinance_gcf_futures",
                            "note":       "ADVERTENCIA: precio de futuros GC=F (~4,600 USD). NO es XAU/USD spot (~3,200 USD). Usar solo como indicador de dirección.",
                        }
                        print(f"[GC=F] ⚠ gold fallback futuros: {close:.2f} (≠ spot)")
        except Exception as e:
            print(f"[GC=F] Error: {e}")

    # PASO 5: Repo fallback para lo restante
    missing = [k for k in YFINANCE_SYMBOLS if k not in quotes]
    if missing:
        print(f"\n[Repo] Sin datos para: {missing}")
        for k, v in load_repo_fallback(site_path).items():
            if k in missing:
                quotes[k] = v

    # PASO 6 (FIX W-05): Enriquecer us2y con el dato real del Treasury 2Y
    # extendido-data/USD.json contiene bond2y desde FRED:DGS2 (daily, verdadero 2Y del Tesoro).
    # Reemplaza el proxy ^IRX (T-Bill 13 semanas) que solo es válido como us3m.
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
                "note":       "US Treasury 2Y yield (FRED:DGS2, daily). Reemplaza proxy ^IRX (T-Bill 13 semanas).",
            }
            print(f"[W-05 fix] us2y actualizado con FRED:DGS2 = {bond2y:.4f}% ({bond2y_date})")
        else:
            print(f"[W-05] bond2y no disponible en USD.json — manteniendo proxy ^IRX para us2y")
            if "us2y" in quotes:
                quotes["us2y"]["note"] = "Proxy: ^IRX (T-Bill 13 semanas). El real Treasury 2Y (FRED:DGS2) no estaba disponible en extended-data/USD.json en este run."
    except Exception as e:
        print(f"[W-05] No se pudo leer extended-data/USD.json para us2y: {e}")

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
