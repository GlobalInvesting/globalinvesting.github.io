#!/usr/bin/env python3
"""
fetch_intraday_quotes.py  v3.0 — Intraday quotes via yfinance (+ FX pairs, ETF IV, extended correlations + historical norms)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce:  intraday-data/quotes.json
Schedule: Cada 15 min en días de semana, horario de mercado (via GitHub Action)

MEJORAS v3.0 vs v2.9:
  - pct1m (cambio % vs cierre de hace ~30 días) añadido a los 7 pares FX majors.
    Calculado dentro de fetch_hv30_fx() reutilizando el historial 3mo ya descargado
    (sin llamada API extra). Ventana de búsqueda ±2d para absorber festivos y fines
    de semana. Permite eliminar update-fx-performance.yml como fuente de datos del
    script de narrativa AI — generate_narrative_signals.py ahora lee spot, 1W y 1M
    directamente de quotes.json (yfinance, actualización c/5 min) en lugar de
    fx-performance/*.json (ECB/Frankfurter, actualización diaria).

MEJORAS v2.9 vs v2.8:
  - Sess H/L sanity check: no anula H=L en fin de semana (H=L=close del viernes es válido).

MEJORAS v2.8 vs v2.7:
  - fetch_correlations() ahora descarga 252d (1 año) en vez de 90d.
  - Calcula norm: media de Pearson en ventanas de 30d sobre el año histórico.
  - Calcula z_score: (corr60d - norm) / std — cuántas sigmas fuera de norma.
  - Agrega 2 pares nuevos: DXY/SPX y Gold/DXY (señales de régimen).
  - PASO 6b: genera signals de rotura de correlación en ai-analysis/signals.json
    cuando |z_score| > 1.5 (determinista, sin LLM). Se limpian en cada ejecución.

MEJORAS v2.7 vs v2.6:
  - 1D % change ahora usa ticker.info["regularMarketChangePercent"] como fuente primaria
    (mismo número que muestra Yahoo Finance) — elimina definitivamente el bug +0.00% en FX.
  - Cascade: info → fast_info.previous_close → daily history (fallback final).
  - fast_info.previous_close en STEP B en vez de closes.iloc[-1] — evita la contaminación
    de la barra de reapertura del domingo 22:00 UTC para pares FX en lunes.

FUENTES POR SÍMBOLO:
  VIX     → yfinance  ^VIX          (CBOE, real-time diferido)
  SPX     → yfinance  ^GSPC         (S&P 500)
  GOLD    → yfinance  GC=F          (Gold futures) + fallback Twelve Data XAU/USD
  WTI     → yfinance  CL=F          (WTI Crude front-month futures)
  US10Y   → yfinance  ^TNX          (CBOE 10Y Treasury Yield — ya en %, no ×10)
  NIKKEI  → yfinance  ^N225
  STOXX   → yfinance  ^STOXX50E
  DXY     → yfinance  DX-Y.NYB
  US2Y    → FRED DGS2               (real Treasury 2Y; ^IRX solo como placeholder inicial, reemplazado en PASO 4)
  US3M    → yfinance  ^IRX          (13-week T-Bill = 3M, mismo símbolo)
  US5Y    → yfinance  ^FVX          (US 5Y Treasury yield)
  US30Y   → yfinance  ^TYX          (US 30Y Treasury yield)
  MOVE    → yfinance  ^MOVE         (ICE BofA Bond Volatility Index)
  BTC     → yfinance  BTC-USD       (Bitcoin)
  FTSE    → yfinance  ^FTSE         (FTSE 100 — para correlación GBP/USD)
  ASX     → yfinance  ^AXJO         (ASX 200  — para correlación AUD/USD)
  NZX     → yfinance  ^NZ50         (NZX 50   — para correlación NZD/USD)
  FX pairs→ yfinance  EURUSD=X etc  (21 pares — reemplaza Stooq bloqueado por CORS)

SALIDAS CLAVE:
  quotes        — precios intraday de todos los símbolos
  hv30          — volatilidad histórica 30d por par FX
  correlations  — Pearson 60d rolling (10 pares, incluye GBP/FTSE, AUD/ASX, NZD/NZX, EUR/STOXX)
  fx_etf_iv     — IV real de ATM options en ETFs de divisa CBOE (FXE, FXB, FXY, FXA, FXF, FXC)

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
from datetime import datetime, timezone, timedelta

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
    "us2y":   "^IRX",     # T-Bill 13 semanas (3M) — placeholder inicial; reemplazado por FRED DGS2 (real 2Y) en PASO 4
    "us5y":   "^FVX",     # US 5Y Treasury yield (×10 en Yahoo — se divide abajo)
    "us30y":  "^TYX",     # US 30Y Treasury yield (×10 en Yahoo — se divide abajo)
    # Risk panel — bond vol + crypto
    "move":   "^MOVE",    # ICE BofA MOVE Index (bond market volatility)
    "btc":    "BTC-USD",  # Bitcoin — topbar + cross-asset panel
    # Equity indices for extended correlations (GBP, AUD, NZD) — not shown in panel
    "ftse":   "^FTSE",    # FTSE 100 — GBP/USD correlation
    "asx":    "^AXJO",    # ASX 200  — AUD/USD correlation
    "nzx":    "^NZ50",    # NZX 50   — NZD/USD correlation
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
    # New crosses — completing all 21 G8 cross pairs
    "eurnzd": "EURNZD=X",
    "gbpaud": "GBPAUD=X",
    "gbpnzd": "GBPNZD=X",
    "audcad": "AUDCAD=X",
    "cadchf": "CADCHF=X",
    "nzdcad": "NZDCAD=X",
    "nzdchf": "NZDCHF=X",
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
    "eurnzd": lambda v: 1.0 < v < 2.5,
    "gbpaud": lambda v: 1.4 < v < 2.5,
    "gbpnzd": lambda v: 1.6 < v < 2.8,
    "audcad": lambda v: 0.7 < v < 1.2,
    "cadchf": lambda v: 0.50 < v < 0.85,
    "nzdcad": lambda v: 0.67 < v < 1.11,
    "nzdchf": lambda v: 0.40 < v < 0.71,
}


# FX pairs para los que calculamos HV30 (los mismos que aparecen en la tabla de Majors)
HV30_FX_PAIRS = [
    "eurusd", "gbpusd", "usdjpy", "audusd", "usdchf", "usdcad", "nzdusd",
    "eurgbp", "eurjpy", "eurchf", "eurcad", "euraud",
    "gbpjpy", "gbpchf", "gbpcad",
    "audjpy", "audnzd", "audchf",
    "cadjpy", "chfjpy", "nzdjpy",
    "eurnzd", "gbpaud", "gbpnzd", "audcad", "cadchf", "nzdcad", "nzdchf",
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


def compute_var_cvar(closes_series, confidence=0.95, lookback=252):
    """
    Historical VaR and CVaR from daily log-returns.

    Parameters
    ----------
    closes_series : list or pandas Series
        Daily closing prices (any order, ascending preferred).
    confidence : float
        Confidence level, e.g. 0.95 for 95%, 0.99 for 99%.
    lookback : int
        Number of trading days to use (default 252 = 1 year).

    Returns
    -------
    dict with keys:
        var_pct  : VaR as a positive percentage of price (e.g. 0.83 means 0.83%)
        cvar_pct : CVaR (Expected Shortfall) as a positive percentage
        n        : number of daily returns used
        confidence: confidence level used
    or None if insufficient data.

    Methodology — Historical Simulation (non-parametric):
        1. Compute log-returns for the lookback window.
        2. Sort ascending (worst first).
        3. VaR  = -percentile(returns, 1 - confidence)  [the cut-off loss]
        4. CVaR = -mean(returns < VaR cut-off)           [mean of tail losses]
    Both expressed as % of price (multiply by 100).
    No distributional assumption: captures fat tails present in actual FX returns.
    """
    try:
        prices = [float(c) for c in closes_series if c is not None and float(c) > 0]
        # Need at least lookback + 1 prices to get lookback returns
        if len(prices) < max(lookback + 1, 30):
            return None
        window = prices[-(lookback + 1):]
        returns = [math.log(window[i] / window[i - 1]) for i in range(1, len(window))]
        n = len(returns)
        if n < 20:
            return None

        sorted_r = sorted(returns)   # ascending: worst losses first

        # VaR: the loss at the (1 - confidence) quantile
        # e.g. for 95%: cutoff index = floor(0.05 * n)
        cutoff_idx = max(0, int(math.floor((1 - confidence) * n)) - 1)
        var_return = sorted_r[cutoff_idx]   # negative number (a loss)
        var_pct    = round(-var_return * 100, 4)   # positive %

        # CVaR: mean of all returns worse than (or equal to) the VaR cut-off
        tail = sorted_r[: cutoff_idx + 1]
        cvar_return = sum(tail) / len(tail) if tail else var_return
        cvar_pct    = round(-cvar_return * 100, 4)   # positive %

        # Rolling 60-day VaR for regime comparison (shorter window)
        var60_pct = None
        if n >= 60:
            window60  = sorted(returns[-60:])
            ci60_idx  = max(0, int(math.floor((1 - confidence) * 60)) - 1)
            var60_pct = round(-window60[ci60_idx] * 100, 4)

        return {
            "var_pct":    var_pct,
            "cvar_pct":   cvar_pct,
            "var60_pct":  var60_pct,   # 60d rolling VaR — regime shift indicator
            "n":          n,
            "confidence": confidence,
        }
    except Exception:
        return None


# Pairs for which we compute rolling 60-day Pearson correlation
CORRELATION_PAIRS = [
    ("eurusd", "dxy"),
    ("audusd", "gold"),
    ("usdjpy", "us10y"),
    ("usdjpy", "vix"),
    ("usdcad", "wti"),
    # Extended pairs — added v2.4
    ("gbpusd", "ftse"),    # GBP/USD vs FTSE 100 (BoE/UK equity link)
    ("audusd", "asx"),     # AUD/USD vs ASX 200  (domestic equity proxy)
    ("nzdusd", "nzx"),     # NZD/USD vs NZX 50   (domestic equity proxy)
    ("eurusd", "stoxx"),   # EUR/USD vs EuroStoxx 50 (ECB/EU equity link)
    ("gbpusd", "gold"),    # GBP/USD vs Gold      (safe-haven vs sterling)
    # Dynamic correlation pairs — added v2.8 (regime-break signals)
    ("dxy", "spx"),        # DXY vs SPX: positive = USD funding stress (breaks normal negative)
    ("gold", "dxy"),       # Gold vs DXY: positive = safe-haven model broken or real inflation
]

# Human-readable labels for the frontend
CORRELATION_LABELS = {
    ("eurusd", "dxy"):    ("EUR/USD", "DXY"),
    ("audusd", "gold"):   ("AUD/USD", "Gold"),
    ("usdjpy", "us10y"):  ("USD/JPY", "US 10Y"),
    ("usdjpy", "vix"):    ("USD/JPY", "VIX"),
    ("usdcad", "wti"):    ("USD/CAD", "WTI Oil"),
    # Extended
    ("gbpusd", "ftse"):   ("GBP/USD", "FTSE 100"),
    ("audusd", "asx"):    ("AUD/USD", "ASX 200"),
    ("nzdusd", "nzx"):    ("NZD/USD", "NZX 50"),
    ("eurusd", "stoxx"):  ("EUR/USD", "EuroStoxx"),
    ("gbpusd", "gold"):   ("GBP/USD", "Gold"),
    # Dynamic
    ("dxy", "spx"):       ("DXY", "SPX"),
    ("gold", "dxy"):      ("Gold", "DXY"),
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


def fetch_var_cvar():
    """
    Downloads 270 days of daily closes for key instruments and computes
    Historical VaR 95% and CVaR 95% (1-day, expressed as % of price).

    Instruments covered:
        FX:          EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, USD/CAD, NZD/USD
        Cross-asset: SPX, Gold, WTI, DXY, US 10Y yield, BTC
        Vol indices: VIX, MOVE

    Returns a dict  { instrument_id: { var_pct, cvar_pct, var60_pct, n, confidence } }
    ready to be written into quotes.json under key "var_cvar".
    """
    VAR_SYMBOLS = {
        # FX majors
        "eurusd": "EURUSD=X",
        "gbpusd": "GBPUSD=X",
        "usdjpy": "JPY=X",
        "audusd": "AUDUSD=X",
        "usdchf": "CHF=X",
        "usdcad": "CAD=X",
        "nzdusd": "NZDUSD=X",
        # Cross-asset
        "spx":    "^GSPC",
        "gold":   "GC=F",
        "wti":    "CL=F",
        "dxy":    "DX-Y.NYB",
        "us10y":  "^TNX",
        "btc":    "BTC-USD",
        "vix":    "^VIX",
        "move":   "^MOVE",
    }
    LOOKBACK = 252     # 1 year of trading days
    PERIOD   = "390d"  # download buffer: ~15 months -> reliable 252-day window

    print("\n[VaR/CVaR] Computing Historical VaR 95% and CVaR 95%...")
    results = {}
    try:
        tickers = list(set(VAR_SYMBOLS.values()))
        raw = yf.download(tickers, period=PERIOD, auto_adjust=True, progress=False)
        closes_df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw

        for inst_id, ticker in VAR_SYMBOLS.items():
            try:
                if hasattr(closes_df, 'columns') and ticker in closes_df.columns:
                    series = closes_df[ticker].dropna().tolist()
                else:
                    series = []
                if not series:
                    print(f"  {inst_id:10s}: no data")
                    continue
                vc = compute_var_cvar(series, confidence=0.95, lookback=LOOKBACK)
                if vc:
                    results[inst_id] = vc
                    regime = ""
                    if vc.get("var60_pct") and vc["var60_pct"] > vc["var_pct"] * 1.25:
                        regime = "  [!] 60d VaR elevated vs 252d baseline"
                    print(f"  {inst_id:10s}: VaR95={vc['var_pct']:.3f}%  CVaR95={vc['cvar_pct']:.3f}%  n={vc['n']}{regime}")
                else:
                    print(f"  {inst_id:10s}: insufficient data (got {len(series)} closes)")
            except Exception as e:
                print(f"  {inst_id:10s}: error -- {e}")
    except Exception as e:
        print(f"[VaR/CVaR] Download failed: {e}")

    return results


def fetch_correlations():
    """
    Downloads 270 days of daily closes for each symbol used in correlation pairs.
    Computes:
      - corr:    rolling 60-day Pearson (current regime)
      - norm:    mean of all 30-day rolling Pearson windows over 252 days (historical baseline)
      - z_score: (corr - norm) / std_dev — how many std devs current is from historical norm
                 |z| > 1.5 = correlation break worth flagging as a signal
    Returns a list of dicts ready for quotes.json.
    """
    print("\n[Correlations] Computing rolling correlations with historical norm...")

    # Fetch 270d history (252 trading days + buffer) for all needed symbols
    series = {}
    for sym_id, yf_sym in CORR_SYMBOLS.items():
        try:
            ticker = yf.Ticker(yf_sym)
            hist = ticker.history(period="1y", interval="1d", auto_adjust=True)
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
            results.append({"a": labels[0], "b": labels[1], "corr": None, "n": 0,
                            "norm": None, "z_score": None})
            continue

        sa, sb = series[sym_a], series[sym_b]
        aligned = min(len(sa), len(sb))
        sa_all, sb_all = sa[-aligned:], sb[-aligned:]

        # Current correlations at three windows — 30d, 60d, 90d
        n30 = min(aligned, 30)
        n60 = min(aligned, 60)
        n90 = min(aligned, 90)
        corr30 = pearson(sa_all[-n30:], sb_all[-n30:])
        corr   = pearson(sa_all[-n60:], sb_all[-n60:])   # default window (60d)
        corr90 = pearson(sa_all[-n90:], sb_all[-n90:])

        # Historical norm: compute all 30-day rolling Pearson windows over last 252 points
        # Use 30-day windows (not 60) for the norm to get more samples and smoother baseline
        norm_window = 30
        hist_corrs = []
        hist_points = min(aligned, 252)
        sa_hist, sb_hist = sa_all[-hist_points:], sb_all[-hist_points:]
        for i in range(hist_points - norm_window + 1):
            c = pearson(sa_hist[i:i + norm_window], sb_hist[i:i + norm_window])
            if c is not None:
                hist_corrs.append(c)

        norm = None
        z_score = None
        if len(hist_corrs) >= 10:
            norm = round(sum(hist_corrs) / len(hist_corrs), 3)
            variance = sum((c - norm) ** 2 for c in hist_corrs) / (len(hist_corrs) - 1)
            std = math.sqrt(variance)
            if std > 0 and corr is not None:
                z_score = round((corr - norm) / std, 2)

        status = f"{corr:+.3f}" if corr is not None else "N/A"
        norm_s = f"norm={norm:+.3f}" if norm is not None else "norm=N/A"
        z_s = f"z={z_score:+.2f}" if z_score is not None else "z=N/A"
        c30s = f"{corr30:+.3f}" if corr30 is not None else "N/A"
        c90s = f"{corr90:+.3f}" if corr90 is not None else "N/A"
        print(f"[Correlations] {labels[0]:8s} vs {labels[1]:8s}: 30d={c30s}  60d={status}  90d={c90s}  {norm_s}  {z_s}")

        results.append({
            "a": labels[0], "b": labels[1],
            "corr30": corr30, "n30": n30,
            "corr":   corr,   "n":   n60,
            "corr90": corr90, "n90": n90,
            "norm": norm, "z_score": z_score,
        })

    return results


def fetch_hv30_fx(fx_pairs):
    """
    Descarga 90 días de historia diaria para cada par FX y calcula HV30 y pct1m.
    Retorna dict { pair_id: {"hv30": float|None, "pct1m": float|None, "pct1m_date": str|None} }.

    pct1m — cambio % vs cierre de hace ~30 días naturales (convención Bloomberg):
      Busca el primer cierre disponible en el rango [today-32d, today-28d] — ventana
      de ±2d para absorber fines de semana y festivos sin saltar a un mes diferente.
      Se calcula aquí porque fetch_yfinance_all() solo descarga 10d (suficiente para
      1W pero no para 1M).  El historial 3mo ya está disponible en este paso.
    """
    print("\n[HV30] Calculando HV30 y pct1m para pares FX...")
    hv30_results = {}
    yf_map = {k: v for k, v in YFINANCE_SYMBOLS.items() if k in fx_pairs}

    for pair_id, yf_sym in yf_map.items():
        try:
            ticker = yf.Ticker(yf_sym)
            hist = ticker.history(period="3mo", interval="1d", auto_adjust=True)
            if hist.empty or len(hist) < 22:
                print(f"[HV30] {pair_id}: insuficientes datos ({len(hist)} días)")
                hv30_results[pair_id] = {"hv30": None, "pct1m": None, "pct1m_date": None}
                continue

            closes = hist["Close"].dropna()
            hv = compute_hv30(closes.tolist())
            status = f"{hv:.2f}%" if hv is not None else "N/A"
            print(f"[HV30] ✓ {pair_id:8s} ({yf_sym}): {status}  ({len(closes)} cierres)")

            # ── pct1m: cierre de hace ~30 días naturales ──────────────────────
            # Busca el primer cierre disponible en la ventana [today-32d, today-28d].
            # La ventana de ±2d absorbe fines de semana y festivos sin saltar de mes.
            pct1m = None
            pct1m_date = None
            try:
                today_date = datetime.now(timezone.utc).date()
                target_lo  = today_date - timedelta(days=32)
                target_hi  = today_date - timedelta(days=28)
                hist_dates = [d.date() if hasattr(d, "date") else d for d in closes.index]
                # Buscar el cierre más cercano a 30d dentro de la ventana
                ref_close = None
                ref_date  = None
                for i, d in enumerate(hist_dates):
                    if target_lo <= d <= target_hi:
                        ref_close = float(closes.iloc[i])
                        ref_date  = d
                        break  # primer match en orden cronológico = más antiguo = más correcto
                if ref_close and ref_close != 0:
                    current_close = float(closes.iloc[-1])
                    pct1m = round((current_close / ref_close - 1.0) * 100.0, 4)
                    pct1m_date = str(ref_date)
                    print(f"[1M] ✓ {pair_id:8s}: {current_close:.4f} vs {ref_date} {ref_close:.4f} → {pct1m:+.4f}%")
                else:
                    print(f"[1M] ⚠ {pair_id:8s}: no se encontró cierre en ventana [{target_lo}, {target_hi}]")
            except Exception as _e1m:
                print(f"[1M] ⚠ {pair_id:8s}: {_e1m}")

            hv30_results[pair_id] = {"hv30": hv, "pct1m": pct1m, "pct1m_date": pct1m_date}

        except Exception as e:
            print(f"[HV30] Error en {pair_id}: {e}")
            hv30_results[pair_id] = {"hv30": None, "pct1m": None, "pct1m_date": None}

    return hv30_results


# ── FX ETF Implied Volatility ─────────────────────────────────────────────────
# Source: CBOE-listed currency ETF options via yfinance (free, no key)
# ETF → FX pair mapping:
#   FXE → EUR/USD    FXB → GBP/USD    FXY → USD/JPY (inverted ETF convention)
#   FXA → AUD/USD    FXF → USD/CHF (inverted)    FXC → USD/CAD (inverted)
# Method: ATM implied vol from nearest expiry with ≥4 days to expiration.
# Limitation: ETF option liquidity is lower than OTC FX interbank options.
#   IV may deviate 1–3 vol points from true OTC 25d RR. Label accordingly.
# Output in quotes.json under key "fx_etf_iv":
#   { "eurusd": { "iv": 7.4, "expiry": "2025-04-18", "source": "FXE options" }, ... }
FX_ETF_MAP = {
    "eurusd": {"etf": "FXE", "invert": False},
    "gbpusd": {"etf": "FXB", "invert": False},
    "usdjpy": {"etf": "FXY", "invert": True },  # FXY = JPY/USD → invert for USD/JPY
    "audusd": {"etf": "FXA", "invert": False},
    "usdchf": {"etf": "FXF", "invert": True },  # FXF = CHF/USD
    "usdcad": {"etf": "FXC", "invert": True },  # FXC = CAD/USD
}

def fetch_fx_etf_iv():
    """
    Fetches ATM implied volatility from CBOE-listed FX ETF option chains via yfinance.
    Returns dict { pair_id: { "iv": float_pct, "expiry": str, "source": str } | None }.
    Falls back gracefully to None per pair if options are illiquid or unavailable.
    """
    from datetime import date as _date
    print("\n[ETF-IV] Fetching FX ETF implied volatility...")
    results = {}
    today = _date.today()

    for pair_id, cfg in FX_ETF_MAP.items():
        etf_sym = cfg["etf"]
        try:
            ticker = yf.Ticker(etf_sym)
            exps = ticker.options
            if not exps:
                print(f"[ETF-IV] {etf_sym}: no options available")
                results[pair_id] = None
                continue

            # Pick nearest expiry with ≥4 days to go (avoid pin-risk distortion near expiry)
            chosen_exp = None
            for exp_str in exps:
                try:
                    exp_date = _date.fromisoformat(exp_str)
                    if (exp_date - today).days >= 4:
                        chosen_exp = exp_str
                        break
                except ValueError:
                    continue

            if not chosen_exp:
                print(f"[ETF-IV] {etf_sym}: no valid expiry found")
                results[pair_id] = None
                continue

            chain = ticker.option_chain(chosen_exp)
            calls = chain.calls

            # Current spot for ATM strike selection
            spot_info = ticker.history(period="2d", interval="1d")
            if spot_info.empty:
                results[pair_id] = None
                continue
            spot = float(spot_info["Close"].iloc[-1])

            # Nearest strike to spot = ATM
            strikes = calls["strike"].dropna().tolist()
            if not strikes:
                results[pair_id] = None
                continue
            atm_strike = min(strikes, key=lambda s: abs(s - spot))

            # IV from ATM call; fallback to ATM put if call IV is bad
            atm_call = calls[calls["strike"] == atm_strike]
            iv_raw = None
            if not atm_call.empty:
                iv_raw = atm_call["impliedVolatility"].iloc[0]

            if iv_raw is None or (iv_raw != iv_raw) or iv_raw <= 0:
                puts = chain.puts
                atm_put = puts[puts["strike"] == atm_strike]
                if not atm_put.empty:
                    iv_raw = atm_put["impliedVolatility"].iloc[0]

            if iv_raw is None or (iv_raw != iv_raw) or iv_raw <= 0 or iv_raw > 1.5:
                print(f"[ETF-IV] {etf_sym}: IV out of range ({iv_raw})")
                results[pair_id] = None
                continue

            # yfinance returns IV as decimal (0.074 → 7.4%)
            iv_pct = round(float(iv_raw) * 100, 1)
            # FX ETF IV plausibility gate: 3–40%
            if not (3.0 <= iv_pct <= 40.0):
                print(f"[ETF-IV] {etf_sym}: IV {iv_pct}% outside plausible range")
                results[pair_id] = None
                continue

            results[pair_id] = {
                "iv":     iv_pct,
                "expiry": chosen_exp,
                "source": f"{etf_sym} options",
                "atm":    round(atm_strike, 4),
            }
            print(f"[ETF-IV] ✓ {pair_id:8s} ({etf_sym}): ATM IV={iv_pct:.1f}%  exp={chosen_exp}  K={atm_strike}")

        except Exception as e:
            print(f"[ETF-IV] {etf_sym} ({pair_id}): {e}")
            results[pair_id] = None

    valid = sum(1 for v in results.values() if v is not None)
    print(f"[ETF-IV] {valid}/{len(FX_ETF_MAP)} pairs with real IV")
    return results


def fetch_yfinance_all(symbols_map):
    yf_tickers = list(symbols_map.values())
    print(f"[yfinance] Descargando: {' '.join(yf_tickers)}")

    results = {}
    try:
        # Descargar cada ticker individualmente para evitar problemas de columnas multinivel
        for internal_id, yf_sym in symbols_map.items():
            try:
                ticker = yf.Ticker(yf_sym)
                hist = ticker.history(period="10d", interval="1d", auto_adjust=True)

                if hist.empty:
                    print(f"[yfinance] Sin datos para {yf_sym}")
                    results[internal_id] = None
                    continue

                closes = hist["Close"].dropna()
                if len(closes) < 1:
                    results[internal_id] = None
                    continue

                # ── 1D % change: Strategy v2.7 ─────────────────────────────────────────
                #
                # Root cause of the persistent +0.00% bug for FX pairs:
                #   v2.6 used closes.iloc[-1] as prev_close, but for FX on Monday,
                #   yfinance daily history includes a bar from Sunday 22:00 UTC (market reopen)
                #   with Friday's price as its close. So closes.iloc[-1] == fast_info.last_price
                #   → chg = 0 → pct = 0.00%.
                #
                # Fix (v2.7) — cascade strategy, same logic Yahoo Finance uses:
                #
                #   STEP A (primary): ticker.info["regularMarketChangePercent"]
                #     • This is Yahoo Finance's own official 1D % — identical to what the
                #       user sees on finance.yahoo.com. No bar alignment issues.
                #     • ticker.info["regularMarketPrice"] → close
                #     • ticker.info["regularMarketPreviousClose"] → prev_close
                #     • ticker.info["regularMarketChange"] → chg (pre-calculated by Yahoo)
                #     • Downside: .info() is slower (~1–2s per ticker). Acceptable since
                #       we already fetch individual tickers for HV30 and correlations.
                #
                #   STEP B (fallback): fast_info.previous_close (NOT closes.iloc[-1])
                #     • fast_info.previous_close is Yahoo's official prev-session close,
                #       the same value used in regularMarketChangePercent calculations.
                #     • fast_info.last_price / fast_info.previous_close → correct chg/pct.
                #     • Avoids the closes.iloc[-1] Monday-reopen-bar contamination.
                #
                #   STEP C (last resort): closes.iloc[-1] vs closes.iloc[-2]
                #     • Pure yfinance daily history as final fallback.
                #     • Less accurate on Mondays due to the reopen bar, but better than 0%.

                close    = None
                prev_close = None
                chg      = None
                pct      = None
                day_high = None
                day_low  = None

                # STEP A: ticker.info (most accurate — same numbers as Yahoo Finance website)
                try:
                    info = ticker.info
                    rmp  = info.get("regularMarketPrice")
                    rmpc = info.get("regularMarketPreviousClose")
                    rmch = info.get("regularMarketChange")
                    rmpct= info.get("regularMarketChangePercent")
                    rmdh = info.get("dayHigh")
                    rmdl = info.get("dayLow")

                    if rmp and rmpc and VALIDATORS.get(internal_id, lambda x: True)(float(rmp)):
                        close      = float(rmp)
                        prev_close = float(rmpc)
                        chg        = float(rmch)  if rmch  is not None else (close - prev_close)
                        pct        = float(rmpct) if rmpct is not None else ((chg / prev_close * 100) if prev_close else 0.0)
                        if rmdh:
                            day_high = round(float(rmdh), 4)
                        if rmdl:
                            day_low  = round(float(rmdl), 4)
                        print(f"[yfinance] ✓ {internal_id:8s} ({yf_sym}): {close:.4f}  {pct:+.2f}%  [via info]")
                except Exception:
                    pass  # fall through to STEP B

                # STEP B: fast_info with fast_info.previous_close (not closes.iloc[-1])
                if close is None:
                    try:
                        fi = ticker.fast_info
                        lp = fi.get("last_price")      if hasattr(fi, "get") else getattr(fi, "last_price",      None)
                        pc = fi.get("previous_close")  if hasattr(fi, "get") else getattr(fi, "previous_close",  None)
                        dh = fi.get("day_high")        if hasattr(fi, "get") else getattr(fi, "day_high",        None)
                        dl = fi.get("day_low")         if hasattr(fi, "get") else getattr(fi, "day_low",         None)

                        if lp and pc and VALIDATORS.get(internal_id, lambda x: True)(float(lp)):
                            close      = float(lp)
                            prev_close = float(pc)
                            chg        = close - prev_close
                            pct        = (chg / prev_close * 100) if prev_close != 0 else 0.0
                            if dh:
                                day_high = round(float(dh), 4)
                            if dl:
                                day_low  = round(float(dl), 4)
                            print(f"[yfinance] ✓ {internal_id:8s} ({yf_sym}): {close:.4f}  {pct:+.2f}%  [via fast_info]")
                    except Exception:
                        pass  # fall through to STEP C

                # STEP C: daily history fallback (closes.iloc[-1] vs iloc[-2])
                if close is None:
                    if len(closes) >= 2:
                        close      = float(closes.iloc[-1])
                        prev_close = float(closes.iloc[-2])
                    elif len(closes) == 1:
                        close      = float(closes.iloc[-1])
                        prev_close = close
                    chg = close - prev_close
                    pct = (chg / prev_close * 100) if prev_close != 0 else 0.0
                    highs = hist["High"].dropna()
                    lows  = hist["Low"].dropna()
                    day_high = round(float(highs.iloc[-1]), 4) if len(highs) >= 1 else None
                    day_low  = round(float(lows.iloc[-1]),  4) if len(lows)  >= 1 else None
                    print(f"[yfinance] ✓ {internal_id:8s} ({yf_sym}): {close:.4f}  {pct:+.2f}%  [via daily hist fallback]")

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
                    "high":       day_high,
                    "low":        day_low,
                    "source":     "yfinance",
                }

                # ── 1W CHG: prior-Friday-close convention (Bloomberg/TradingView) ──────
                # Only computed for FX major pairs. Uses the daily history already
                # downloaded — no extra API call needed.
                # Finds the most recent Friday in the history that is strictly before
                # today's session (i.e. last week's Friday close).
                FX_MAJORS_1W = {"eurusd", "gbpusd", "usdjpy", "audusd", "usdchf", "usdcad", "nzdusd"}
                if internal_id in FX_MAJORS_1W:
                    try:
                        # hist index is tz-aware; normalize to date for weekday comparison
                        hist_dates = [d.date() if hasattr(d, "date") else d for d in hist.index]
                        hist_closes = hist["Close"].dropna()
                        today_date = datetime.now(timezone.utc).date()
                        # Find the most recent Friday strictly before today
                        prior_friday = None
                        prior_friday_close = None
                        for i in range(len(hist_dates) - 1, -1, -1):
                            d = hist_dates[i]
                            if d < today_date and d.weekday() == 4:  # 4 = Friday
                                prior_friday = d
                                prior_friday_close = float(hist_closes.iloc[i])
                                break
                        if prior_friday_close and prior_friday_close != 0:
                            pct1w = round((close / prior_friday_close - 1.0) * 100.0, 4)
                            results[internal_id]["pct1w"] = pct1w
                            results[internal_id]["pct1w_date"] = str(prior_friday)
                            print(f"[1W] ✓ {internal_id:8s}: {close:.4f} vs Friday {prior_friday} {prior_friday_close:.4f} → {pct1w:+.4f}%")
                        else:
                            results[internal_id]["pct1w"] = None
                            print(f"[1W] ⚠ {internal_id:8s}: no prior Friday found in {len(hist_dates)}d history")
                    except Exception as _e1w:
                        results[internal_id]["pct1w"] = None
                        print(f"[1W] ⚠ {internal_id:8s}: {_e1w}")

                # Sanity check: high == low means yfinance returned an incomplete intraday
                # bar (e.g. a stale tick where H=L=close).
                # Exception: on weekends H=L=close is valid (last Friday close) — keep it
                # so the frontend can show last-close range in Sess H/L.
                _is_weekend_hl = datetime.now(timezone.utc).weekday() >= 5  # 5=Sat, 6=Sun
                if (not _is_weekend_hl
                        and results[internal_id]["high"] is not None
                        and results[internal_id]["low"]  is not None
                        and results[internal_id]["high"] == results[internal_id]["low"]):
                    results[internal_id]["high"] = None
                    results[internal_id]["low"]  = None
                    print(f"[yfinance] ⚠ {internal_id:8s}: high==low ({day_high}) — clearing H/L (incomplete bar)")
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
    print(f"\n{'='*60}\nfetch_intraday_quotes.py  v3.0  —  {ts}\n{'='*60}\n")

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

    # PASO 5: Calcular HV30 + pct1m para pares FX e inyectar en cada quote
    hv30_data = fetch_hv30_fx(HV30_FX_PAIRS)
    hv30_output = {}
    for pair_id, result in hv30_data.items():
        hv       = result.get("hv30")
        p1m      = result.get("pct1m")
        p1m_date = result.get("pct1m_date")
        hv30_output[pair_id] = hv  # None si no se pudo calcular
        if pair_id in quotes:
            if hv is not None:
                quotes[pair_id]["hv30"] = hv
            if p1m is not None:
                quotes[pair_id]["pct1m"]      = p1m
                quotes[pair_id]["pct1m_date"] = p1m_date

    # PASO 6: Calcular correlaciones rolling 60d con norma histórica
    correlations = fetch_correlations()

    # PASO 6b: Generar signals de rotura de correlación para Alerts & Market Signals
    # Regla: |z_score| > 1.5 = correlación fuera de norma → signal determinista (no LLM)
    # Se acumulan en signals.json junto con los signals de AI (se añaden al inicio de la lista)
    try:
        import os as _os
        _signals_path = _os.path.join(site_path, "ai-analysis", "signals.json")
        _existing_signals = []
        if _os.path.exists(_signals_path):
            with open(_signals_path) as _sf:
                _existing_signals = json.load(_sf)
            # Remove stale corr-break signals from previous run (re-computed each time)
            _existing_signals = [s for s in _existing_signals
                                  if not s.get("source") == "corr_break"]

        _now_utc = datetime.now(timezone.utc)
        _time_str = _now_utc.strftime("%H:%M")

        # Descriptions for each pair explaining why the break matters for FX
        _corr_context = {
            ("USD/JPY", "VIX"):    {
                "normal": "negative",
                "break_pos": "USD/JPY acting as risk asset, not safe haven. JPY not being bought on stress — unusual.",
                "break_neg": "USD/JPY falling despite low volatility. JPY bid for reasons outside risk sentiment.",
            },
            ("DXY", "SPX"):        {
                "normal": "negative",
                "break_pos": "DXY and equities rising together — USD funding stress or stagflation signal.",
                "break_neg": "DXY falling with equities — broad risk-on, USD losing safe-haven premium.",
            },
            ("Gold", "DXY"):       {
                "normal": "negative",
                "break_pos": "Gold and USD rising together — real inflation or deep safe-haven demand.",
                "break_neg": "Gold falling with USD — risk-on, safe-haven unwind.",
            },
            ("EUR/USD", "EuroStoxx"): {
                "normal": "positive",
                "break_pos": "EUR/USD and EuroStoxx in unusually strong lockstep — EUR purely tracking risk appetite.",
                "break_neg": "EUR/USD falling while European equities hold — ECB policy divergence signal.",
            },
            ("AUD/USD", "Gold"):   {
                "normal": "positive",
                "break_pos": "AUD and Gold unusually correlated — commodity FX regime dominant.",
                "break_neg": "Gold rising but AUD falling — China/domestic risk overriding commodity link.",
            },
        }

        _new_corr_signals = []
        for c in correlations:
            z = c.get("z_score")
            if z is None or abs(z) < 1.5:
                continue
            pair_key = (c["a"], c["b"])
            ctx = _corr_context.get(pair_key, {})
            direction = "above" if z > 0 else "below"
            norm_str = f"{c['norm']:+.2f}" if c.get("norm") is not None else "hist. avg"
            corr_str = f"{c['corr']:+.2f}" if c.get("corr") is not None else "—"

            # Select description based on direction of break
            if ctx:
                desc = ctx["break_pos"] if z > 0 else ctx["break_neg"]
            else:
                norm_dir = "positive" if (c.get("norm") or 0) > 0 else "negative"
                desc = f"Correlation {direction} historical norm ({norm_str}). Normal relationship is {norm_dir}."

            priority = "critical" if abs(z) > 2.5 else "warning"
            _new_corr_signals.append({
                "source":   "corr_break",
                "time":     _time_str,
                "priority": priority,
                "title":    f"{c['a']} / {c['b']} correlation break",
                "text":     desc,
                "evidence": [
                    f"60d corr: {corr_str}",
                    f"Hist. norm: {norm_str}",
                    f"Z-score: {z:+.2f}σ",
                ],
            })
            print(f"[CorrSignal] {c['a']} vs {c['b']}: z={z:+.2f} → {priority} signal")

        if _new_corr_signals:
            _final_signals = _new_corr_signals + _existing_signals
            with open(_signals_path, "w") as _sf:
                json.dump(_final_signals, _sf, indent=2)
            print(f"[CorrSignal] {len(_new_corr_signals)} correlation break signal(s) added to signals.json")
        else:
            print("[CorrSignal] No correlation breaks detected (all |z| < 1.5)")
            # Still write to remove stale corr_break signals from previous run
            if _os.path.exists(_signals_path):
                with open(_signals_path, "w") as _sf:
                    json.dump(_existing_signals, _sf, indent=2)
    except Exception as _e:
        print(f"[CorrSignal] Error generating correlation signals: {_e}")

    # PASO 7: Implied volatility real desde FX ETF options (CBOE vía yfinance)
    fx_etf_iv = fetch_fx_etf_iv()

    # PASO 7b: Acumular historial semanal de IV (máx 52 entradas) y calcular IV Rank / IV Percentile.
    # iv_rank   = (current_iv - min_52w) / (max_52w - min_52w) × 100  → 0–100
    # iv_pct    = % of weekly snapshots in the last 52w where IV was below current IV → 0–100
    # Only runs once per calendar week (ISO week): guards against inflating history on intraday re-runs.
    # History file: intraday-data/iv_history.json
    try:
        import os as _os
        from datetime import date as _date
        _iv_hist_path = _os.path.join(_os.path.dirname(out_file), "iv_history.json")
        _today = _date.today()
        _iso_week = _today.isocalendar()[:2]  # (year, week)

        # Load existing history
        _iv_hist = {}
        if _os.path.exists(_iv_hist_path):
            with open(_iv_hist_path) as _f:
                _iv_hist = json.load(_f)

        _last_week = tuple(_iv_hist.get("_last_week", [0, 0]))
        _history_by_pair = _iv_hist.get("pairs", {})

        # Append this week's snapshot (once per ISO week)
        if tuple(_iso_week) != _last_week:
            _week_key = f"{_iso_week[0]}-W{_iso_week[1]:02d}"
            for _pair_id, _iv_entry in (fx_etf_iv or {}).items():
                if _iv_entry and _iv_entry.get("iv") is not None:
                    if _pair_id not in _history_by_pair:
                        _history_by_pair[_pair_id] = []
                    _history_by_pair[_pair_id].append({
                        "week": _week_key,
                        "iv":   _iv_entry["iv"],
                    })
                    # Keep last 52 weeks only
                    _history_by_pair[_pair_id] = _history_by_pair[_pair_id][-52:]
            _iv_hist = {"_last_week": list(_iso_week), "pairs": _history_by_pair}
            with open(_iv_hist_path, "w") as _f:
                json.dump(_iv_hist, _f, indent=2)
            print(f"[IV-Rank] Snapshot appended for ISO week {_week_key}")
        else:
            print(f"[IV-Rank] Already have snapshot for ISO week {_last_week[0]}-W{_last_week[1]:02d} — skipping append")

        # Compute iv_rank and iv_pct for each pair (uses all available history, even < 52w)
        for _pair_id, _iv_entry in (fx_etf_iv or {}).items():
            if not _iv_entry or _iv_entry.get("iv") is None:
                continue
            _hist = _history_by_pair.get(_pair_id, [])
            if len(_hist) < 4:
                # Need at least 4 data points for a meaningful rank
                _iv_entry["iv_rank"]     = None
                _iv_entry["iv_pct_rank"] = None
                _iv_entry["iv_hist_n"]   = len(_hist)
                continue
            _ivs = [h["iv"] for h in _hist]
            _cur = _iv_entry["iv"]
            _lo, _hi = min(_ivs), max(_ivs)
            _iv_entry["iv_rank"]     = round((_cur - _lo) / (_hi - _lo) * 100, 1) if _hi > _lo else 50.0
            _iv_entry["iv_pct_rank"] = round(sum(1 for v in _ivs if v < _cur) / len(_ivs) * 100, 1)
            _iv_entry["iv_hist_n"]   = len(_hist)
            print(f"[IV-Rank] {_pair_id}: IV={_cur:.1f}% rank={_iv_entry['iv_rank']:.0f} pct={_iv_entry['iv_pct_rank']:.0f} (n={len(_ivs)})")

    except Exception as _e:
        print(f"[IV-Rank] Error computing IV history/rank: {_e}")

    # PASO 8: Historical VaR 95% and CVaR 95% per instrument
    # Throttle: only recompute once per calendar day (UTC). VaR uses 252 days of
    # daily closes — running it every 5 minutes is wasteful and the download can
    # time out under load, silently wiping the last good values with an empty dict.
    # Strategy:
    #   1. Load the existing quotes.json to recover the last good var_cvar.
    #   2. If the existing data was computed today (UTC), reuse it (skip fetch).
    #   3. Otherwise run fetch_var_cvar(); if it returns a non-empty result, use it.
    #   4. If it returns empty (download failure), fall back to the preserved value.
    _prev_var_cvar = {}
    _prev_var_date = None
    try:
        if os.path.exists(out_file):
            with open(out_file) as _pf:
                _prev = json.load(_pf)
            _prev_var_cvar = _prev.get("var_cvar") or {}
            _prev_ts = _prev.get("updated", "")
            _prev_var_date = _prev_ts[:10] if _prev_ts else None  # "YYYY-MM-DD"
    except Exception:
        pass

    _today_utc = ts[:10]  # "YYYY-MM-DD" from the current run timestamp

    if _prev_var_cvar and _prev_var_date == _today_utc:
        # Already computed today — reuse without fetching
        var_cvar_data = _prev_var_cvar
        print(f"[VaR/CVaR] Reusing today's cached result ({len(var_cvar_data)} instruments)")
    else:
        # New day (or first run) — fetch fresh data
        var_cvar_data = fetch_var_cvar()
        if not var_cvar_data and _prev_var_cvar:
            # Fetch failed; preserve last good values rather than writing empty dict
            var_cvar_data = _prev_var_cvar
            print(f"[VaR/CVaR] Fetch returned empty — preserving last good data ({len(var_cvar_data)} instruments)")

    output = {"updated": ts, "source": source_label, "quotes": quotes,
              "hv30": hv30_output, "correlations": correlations,
              "fx_etf_iv": fx_etf_iv, "var_cvar": var_cvar_data}
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
