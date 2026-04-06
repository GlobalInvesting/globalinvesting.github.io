#!/usr/bin/env python3
"""
fetch_fx_liquidity.py  v1.0 — FX Liquidity profile via yfinance H-L range proxy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Produce:  fx-data/fx-liquidity.json
Schedule: Cada hora en días de semana (via GitHub Actions)

METODOLOGÍA:
  El volumen de transacciones FX OTC real (CLS Hourly) no está disponible
  públicamente sin suscripción institucional. El proxy más robusto disponible
  con fuentes gratuitas es el rango intraday High-Low por hora (H-L range),
  que captura la actividad realizada del mercado: a mayor actividad de trading,
  mayor dispersión de precios.

  Referencia académica: "Time of day is the factor that influences [FX] liquidity
  the most" — IEEE, Intraday forex bid/ask spread patterns (2013).

FUENTES:
  EUR/USD  → yfinance EURUSD=X  (par más líquido, benchmark principal)
  GBP/USD  → yfinance GBPUSD=X  (London session proxy)
  USD/JPY  → yfinance USDJPY=X  (Asia session proxy)
  USD/CHF  → yfinance USDCHF=X  (diversificación)
  AUD/USD  → yfinance AUDUSD=X  (Sydney/Asia open proxy)

  Se usa la mediana del H-L range de los 5 pares, normalizada 0–100, para
  producir un índice de actividad compuesto robusto a outliers en un par.

OUTPUTS (fx-data/fx-liquidity.json):
  updated        — ISO timestamp del último run exitoso
  today          — array de 24 valores (horas UTC 0–23) del día actual
                   cada valor: rango H-L mediano normalizado (0–100)
  baseline_30d   — array de 24 valores: promedio 30 días del mismo índice
                   reemplaza el LIQ_BASE hardcodeado en dashboard.js
  hours_complete — cuántas horas del día actual tienen datos reales (0–23)
  source         — string de atribución para el label del panel
  fallback       — true si se usó el baseline histórico por falta de datos

FALLBACK:
  Si yfinance falla (rate limit, timeout), se usa el LIQ_BASE canónico del
  código anterior para no romper el panel. El campo fallback=true lo señala.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import sys
import os
from datetime import datetime, timezone, timedelta
from statistics import median

try:
    import yfinance as yf
except ImportError:
    print("[ERROR] yfinance no instalado. Correr: pip install yfinance")
    sys.exit(1)

# ── Configuración ──────────────────────────────────────────────────────────────

PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"]

# Baseline de fallback (LIQ_BASE canónico — mismos valores que dashboard.js)
# Usado si yfinance no devuelve datos suficientes
LIQ_BASE_FALLBACK = [18,14,11,10,12,20,30,42,58,68,72,70,72,82,95,100,95,80,68,55,42,30,22,20]

OUTPUT_PATH = os.path.join(
    os.environ.get("SITE_PATH", "."),
    "fx-data", "fx-liquidity.json"
)

# ── Helpers ───────────────────────────────────────────────────────────────────




def build_hourly_profile(df_1h, n_days: int) -> list[float]:
    """
    Dado un DataFrame de barras horarias (yfinance 1h, columnas High/Low),
    calcula el H-L range mediano por hora UTC durante los últimos n_days.

    Devuelve lista de 24 floats (horas 0–23 UTC).
    Horas sin datos devuelven 0.0.
    """
    if df_1h is None or df_1h.empty:
        return [0.0] * 24

    # Asegurar timezone UTC
    df = df_1h.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Filtrar al período solicitado
    cutoff = datetime.now(timezone.utc) - timedelta(days=n_days)
    df = df[df.index >= cutoff]

    if df.empty:
        return [0.0] * 24

    df["hl_range"] = df["High"] - df["Low"]
    # Para JPY el rango es en yenes — normalizar por close para hacerlo comparable
    # (porcentaje del precio) antes de medianear con otros pares
    if "Close" in df.columns and df["Close"].median() > 50:
        df["hl_range"] = df["hl_range"] / df["Close"] * 100

    hourly = [[] for _ in range(24)]
    for ts, row in df.iterrows():
        h = ts.hour
        if row["hl_range"] > 0:
            hourly[h].append(row["hl_range"])

    return [median(v) if v else 0.0 for v in hourly]


def fetch_pair_data(symbol: str, period: str, interval: str):
    """Descarga datos de yfinance con manejo de errores."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df is None or df.empty:
            print(f"  [WARN] {symbol}: sin datos")
            return None
        return df
    except Exception as e:
        print(f"  [WARN] {symbol}: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    now_utc = datetime.now(timezone.utc)
    print(f"[fetch_fx_liquidity] Run: {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # ── PASO 1: Descargar datos horarios de los 5 pares ───────────────────────
    # Pedimos 35 días para tener >30 días completos incluso con gaps de fin de semana
    print("Descargando datos horarios 1h (35d)...")
    profiles_30d = []
    profiles_today = []

    for symbol in PAIRS:
        print(f"  {symbol}...", end=" ")
        df = fetch_pair_data(symbol, period="35d", interval="1h")
        if df is None:
            print("skip")
            continue
        print(f"OK ({len(df)} filas)")

        # Perfil 30d (baseline)
        p30 = build_hourly_profile(df, n_days=30)
        profiles_30d.append(p30)

        # Perfil hoy (solo filas del día UTC actual)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        today_str = now_utc.strftime("%Y-%m-%d")
        df_today = df[df.index.strftime("%Y-%m-%d") == today_str]

        p_today = build_hourly_profile(df_today, n_days=1)
        profiles_today.append(p_today)

    # ── PASO 2: Componer índice mediano entre pares ───────────────────────────
    fallback = False

    if len(profiles_30d) < 2:
        print("[WARN] Menos de 2 pares con datos — usando fallback LIQ_BASE")
        baseline_30d = list(LIQ_BASE_FALLBACK)
        fallback = True
    else:
        # Mediana hora a hora entre pares (robusta a outliers)
        baseline_raw = []
        for h in range(24):
            vals = [p[h] for p in profiles_30d if p[h] > 0]
            baseline_raw.append(median(vals) if vals else 0.0)

        # Si alguna hora tiene 0 (fin de semana distorsiona la media), interpolar
        # entre vecinos para no dejar huecos
        for h in range(24):
            if baseline_raw[h] == 0.0:
                prev_v = next((baseline_raw[(h - i) % 24] for i in range(1, 12) if baseline_raw[(h - i) % 24] > 0), 0)
                next_v = next((baseline_raw[(h + i) % 24] for i in range(1, 12) if baseline_raw[(h + i) % 24] > 0), 0)
                baseline_raw[h] = (prev_v + next_v) / 2 if prev_v and next_v else (prev_v or next_v)

        baseline_max = max(baseline_raw) if any(v > 0 for v in baseline_raw) else 1.0
        baseline_30d = [round(v / baseline_max * 100, 1) for v in baseline_raw]

    # Today profile — normalized with same denominator as baseline_30d
    # Avoids artificial 100% spike when only 1–2 hours of data exist
    if len(profiles_today) < 2:
        today_profile = [0.0] * 24
    else:
        today_raw = []
        for h in range(24):
            vals = [p[h] for p in profiles_today if p[h] > 0]
            today_raw.append(median(vals) if vals else 0.0)
        norm_max = baseline_max if not fallback else max((max(today_raw) if today_raw else 1.0), 1.0)
        today_profile = [round(v / norm_max * 100, 1) for v in today_raw]

    # Cuántas horas del día actual tienen datos reales.
    # La hora actual (now_utc.hour) ya tiene una vela cerrada en yfinance 1h,
    # por eso se usa hour + 1 como límite superior exclusivo.
    hours_complete = now_utc.hour + 1

    # ── PASO 3: Serializar ────────────────────────────────────────────────────
    payload = {
        "updated": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "today": today_profile,
        "baseline_30d": baseline_30d,
        "hours_complete": hours_complete,
        "source": "yfinance · H-L range proxy · 5 pairs · 30d avg",
        "fallback": fallback
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"\n✅ Escrito: {OUTPUT_PATH}")
    print(f"   baseline_30d peak hour: {baseline_30d.index(max(baseline_30d))}:00 UTC  (val={max(baseline_30d)})")
    print(f"   hours_complete today:   {hours_complete}")
    print(f"   fallback:               {fallback}")


if __name__ == "__main__":
    main()
