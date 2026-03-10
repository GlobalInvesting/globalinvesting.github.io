"""
generate_historical_scores.py
─────────────────────────────
Generates historical score snapshots for every date in fx-history/.

MODO DE DATOS (en orden de prioridad):
  1. economic-data-history/{CURRENCY}/{indicator}.json  → serie histórica real
     (generada por fetch_historical_econ_data.py con Playwright)
  2. Fallback: valor actual de economic-data/{CURRENCY}.json (proxy estático)

Indicadores con datos históricamente precisos:
  - interestRate    (10%): rates/ con historial mensual
  - rateMomentum    ( 7%): derivado de rates/
  - outlookScore    ( 5%): derivado de rates/
  - fxPerformance1M ( 8%): reconstruido desde fx-history/
  - Todos los demás (70%): economic-data-history/ si existe, sino proxy estático
"""

import json
import os
import sys
from datetime import datetime, timedelta, date

sys.path.insert(0, os.path.dirname(__file__))
from save_weekly_scores import (
    calculate_score, load_json, CURRENCIES, RATES_DIR,
    ECON_DIR, FX_DIR as FX_PERF_DIR, COT_DIR, EXT_DIR,
    TRADING_WEIGHTS, score_indicator, normalize
)

SCORES_DIR   = "scores-history"
FX_HIST_DIR  = "fx-history"
HIST_ECON_DIR = "economic-data-history"   # salida de fetch_historical_econ_data.py

os.makedirs(SCORES_DIR, exist_ok=True)

MIN_PAIR_DIFF      = 12.0   # minimum score differential for active pairs
MIN_PAIR_DIFF_HIGH = 20.0   # high-confidence threshold


# ─── Historical economic data loader ─────────────────────────────────────────

# Cache global: {currency: {indicator: [(date_str, value), ...]}}
_HIST_CACHE = {}

def _load_hist_series(currency, indicator):
    """
    Carga y cachea la serie histórica de economic-data-history/{currency}/{indicator}.json.
    Retorna lista ordenada de (date_str, value) o None si no existe.
    """
    key = f"{currency}/{indicator}"
    if key in _HIST_CACHE:
        return _HIST_CACHE[key]

    path = f"{HIST_ECON_DIR}/{currency}/{indicator}.json"
    if not os.path.exists(path):
        _HIST_CACHE[key] = None
        return None

    try:
        with open(path) as f:
            d = json.load(f)
        obs = d.get("observations", [])
        # Ordenar por fecha ascendente
        series = sorted(
            [(o["date"], o["value"]) for o in obs if o.get("date") and o.get("value") is not None],
            key=lambda x: x[0]
        )
        _HIST_CACHE[key] = series if series else None
        return _HIST_CACHE[key]
    except Exception:
        _HIST_CACHE[key] = None
        return None


def get_indicator_at_date(currency, indicator, target_date_str):
    """
    Devuelve el valor de un indicador para una divisa en una fecha dada.
    Usa la serie histórica si existe; de lo contrario retorna None (el caller
    usará el valor estático actual como fallback).

    Lógica: devuelve el valor más reciente disponible en o antes de target_date.
    Esto replica cómo un trader vería el dato en ese momento (dato con lag de publicación).
    """
    series = _load_hist_series(currency, indicator)
    if not series:
        return None

    target = target_date_str
    best_value = None
    for date_str, value in series:
        if date_str <= target:
            best_value = value
        else:
            break  # ordenado, podemos parar

    return best_value


def has_hist_data(currency, indicator):
    """Retorna True si existe serie histórica para este par divisa/indicador."""
    return _load_hist_series(currency, indicator) is not None


def log_data_sources(snap_date):
    """Log en el primer snapshot qué indicadores usan datos históricos vs proxy."""
    if snap_date != sorted([
        f.replace(".json", "") for f in os.listdir(FX_HIST_DIR) if f.endswith(".json")
    ])[0]:
        return  # Solo loguear una vez

    HISTORICAL_INDICATORS = [
        "gdpGrowth", "inflation", "unemployment", "currentAccount",
        "production", "tradeBalance", "retailSales", "wageGrowth",
        "manufacturingPMI", "servicesPMI", "bond10y",
        "consumerConfidence", "businessConfidence", "cotPositioning",
    ]
    print("\n  ── Fuentes de datos para el backtest ──")
    for cur in CURRENCIES:
        hist_count = sum(1 for ind in HISTORICAL_INDICATORS if has_hist_data(cur, ind))
        total      = len(HISTORICAL_INDICATORS)
        status     = "✓ histórico" if hist_count == total else f"⚠ {hist_count}/{total} histórico"
        print(f"    {cur}: {status}")
    print()


def get_rate_at_date(currency, target_date_str):
    """
    Returns the policy rate and 12-month momentum as of target_date.
    Uses the rates/ JSON which has monthly history.
    """
    d = load_json(f"{RATES_DIR}/{currency}.json")
    obs = d.get("observations", [])
    if not obs:
        return None, None

    target = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    obs_sorted = sorted(obs, key=lambda x: x["date"])

    # Find the most recent observation on or before target date
    current = None
    for o in obs_sorted:
        o_date = datetime.strptime(o["date"], "%Y-%m-%d").date()
        if o_date <= target:
            current = float(o["value"])

    if current is None:
        return None, None

    # Find observation ~12 months before target
    rate_12m = None
    target_12m = target - timedelta(days=365)
    best_delta = None
    for o in obs_sorted:
        o_date = datetime.strptime(o["date"], "%Y-%m-%d").date()
        if o_date <= target:
            delta = abs((o_date - target_12m).days)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                if abs((o_date - target_12m).days) <= 45:  # within 45 days of 12M ago
                    rate_12m = float(o["value"])

    momentum = round(current - rate_12m, 4) if (rate_12m is not None) else None
    return current, momentum


def derive_outlook(rm12, rm24=None):
    """
    Replicates index.html outlook derivation logic.
    Returns numeric score: Hawkish=90, Neutral=50, Dovish=10
    """
    rm_composite = rm24 if (rm24 is not None and rm24 >= 0.25) else rm12
    if rm_composite is None:
        return 50

    if rm_composite >= 0.25:
        outlook = "Hawkish"
    elif rm_composite <= -0.80:
        outlook = "Dovish"
    elif rm12 is not None and rm12 < -0.20:
        outlook = "Dovish"
    else:
        outlook = "Neutral"

    return {"Hawkish": 90, "Neutral": 50, "Dovish": 10}[outlook]




def get_fx_perf_at_date(currency, snap_date_str):
    """
    Reconstruct fxPerformance1M for a currency at snap_date.
    Uses fx-history/ to compute (rate_4w_ago / rate_now - 1) * 100.
    Returns None if data unavailable (first 4 weeks of history).
    """
    if currency == "USD":
        return 0.0  # USD is the base currency

    def load_rates_at(date_str, tolerance=7):
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
        for delta in range(tolerance + 1):
            for sign in [0, 1, -1]:
                d = (target + timedelta(days=delta * sign)).isoformat()
                p = f"{FX_HIST_DIR}/{d}.json"
                if os.path.exists(p):
                    return json.load(open(p)).get("rates_vs_usd", {})
        return {}

    snap_dt = datetime.strptime(snap_date_str, "%Y-%m-%d")
    date_4w_ago = (snap_dt - timedelta(weeks=4)).date().isoformat()

    rates_now = load_rates_at(snap_date_str)
    rates_4w  = load_rates_at(date_4w_ago)

    if not rates_now or not rates_4w:
        return None

    rate_now = rates_now.get(currency)
    rate_4w  = rates_4w.get(currency)

    if not rate_now or not rate_4w or rate_now == 0:
        return None

    # rates are per-USD (foreign units per 1 USD)
    # lower = stronger currency vs USD
    # perf = (rate_4w_ago / rate_now - 1) * 100
    return round((rate_4w / rate_now - 1) * 100, 4)

def calculate_score_at_date(currency, snap_date, econ_data, fx_data, cot_data, ext_data, all_econ):
    """
    Calculate score using historically accurate data for snap_date.

    Prioridad por indicador:
      - interestRate, rateMomentum, outlookScore: siempre desde rates/ (histórico preciso)
      - fxPerformance1M: reconstruido desde fx-history/ (histórico preciso)
      - demás indicadores: economic-data-history/ si existe, sino proxy estático actual
    """
    data = {}

    # ── Indicadores económicos: histórico si existe, proxy si no ──────────────
    ECON_INDICATORS = [
        "gdpGrowth", "inflation", "unemployment", "currentAccount",
        "tradeBalance", "debt", "production", "retailSales",
        "wageGrowth", "manufacturingPMI", "servicesPMI", "termsOfTrade",
    ]
    EXT_INDICATORS = [
        "bond10y", "capitalFlows", "consumerConfidence", "businessConfidence",
    ]

    # Cargar base estática (fallback)
    static_econ = econ_data.get("data", {}) if econ_data else {}
    static_ext  = ext_data.get("data", {})  if ext_data  else {}

    for ind in ECON_INDICATORS:
        hist_val = get_indicator_at_date(currency, ind, snap_date)
        data[ind] = hist_val if hist_val is not None else static_econ.get(ind)

    for ind in EXT_INDICATORS:
        hist_val = get_indicator_at_date(currency, ind, snap_date)
        data[ind] = hist_val if hist_val is not None else static_ext.get(ind)

    # COT: histórico si existe, proxy actual si no
    cot_hist = get_indicator_at_date(currency, "cotPositioning", snap_date)
    if cot_hist is not None:
        data["cotPositioning"] = cot_hist
    elif cot_data:
        data["cotPositioning"] = cot_data.get("netPosition")

    # ── HISTÓRICAMENTE PRECISO: tasa de interés y momentum ────────────────────
    rate, momentum = get_rate_at_date(currency, snap_date)
    data["interestRate"] = rate
    data["rateMomentum"] = momentum

    # outlookScore derivado del momentum histórico
    rm24_ext = static_ext.get("rateMomentum24M")
    data["outlookScore"] = derive_outlook(momentum, rm24_ext)

    # ── HISTÓRICAMENTE PRECISO: FX performance 1M ─────────────────────────────
    data["fxPerformance1M"] = get_fx_perf_at_date(currency, snap_date)

    data["economicSurprise"] = 50  # placeholder neutral

    # ── Cross-sectional normalization: all_data con tasas históricas ──────────
    all_data = {}
    for cur in CURRENCIES:
        ed2  = load_json(f"{ECON_DIR}/{cur}.json").get("data", {})
        ext2 = load_json(f"{EXT_DIR}/{cur}.json").get("data", {}) if os.path.exists(f"{EXT_DIR}/{cur}.json") else {}
        rate2, _ = get_rate_at_date(cur, snap_date)

        # Para normalización cross-sectional, usar histórico si existe
        merged = {**ed2, **ext2, "interestRate": rate2}
        for ind in ECON_INDICATORS + EXT_INDICATORS:
            hist_val = get_indicator_at_date(cur, ind, snap_date)
            if hist_val is not None:
                merged[ind] = hist_val
        all_data[cur] = merged

    # ── Score ponderado ────────────────────────────────────────────────────────
    weighted_sum = 0.0
    total_weight = 0.0
    contributions = {}
    indicators_with_data = 0

    for key, cfg in TRADING_WEIGHTS.items():
        w = cfg["weight"]
        if w == 0:
            continue
        val = data.get(key)
        s   = score_indicator(key, val, all_data, currency)
        contributions[key] = {
            "value":        val,
            "score":        s,
            "weight":       w,
            "contribution": round(s * w, 4) if s is not None else None,
            "source":       "historical" if (
                key in ECON_INDICATORS + EXT_INDICATORS + ["cotPositioning"]
                and get_indicator_at_date(currency, key, snap_date) is not None
            ) else "proxy",
        }
        if s is not None:
            weighted_sum        += s * w
            total_weight        += w
            indicators_with_data += 1

    score = round(weighted_sum / total_weight, 2) if total_weight > 0 else 50
    return score, contributions


def generate_pairs(scores_dict):
    """Generate ranked pairs from scores, mimicking carry trade logic."""
    pairs = []
    currencies = list(scores_dict.keys())
    for i, cur_a in enumerate(currencies):
        for cur_b in currencies[i+1:]:
            score_a = scores_dict[cur_a]
            score_b = scores_dict[cur_b]
            diff = abs(score_a - score_b)
            if diff >= MIN_PAIR_DIFF:
                long_cur  = cur_a if score_a > score_b else cur_b
                short_cur = cur_b if score_a > score_b else cur_a
                conf = "Alta" if diff >= MIN_PAIR_DIFF_HIGH else "Media"
                pairs.append({
                    "long":       long_cur,
                    "short":      short_cur,
                    "diff":       round(diff, 2),
                    "confidence": conf,
                })
    pairs.sort(key=lambda p: p["diff"], reverse=True)
    return pairs[:10]  # top 10 pairs


def main():
    # Load static data (used for all snapshots as proxy)
    econ_data, fx_data, cot_data, ext_data = {}, {}, {}, {}
    all_econ = {}
    for cur in CURRENCIES:
        econ_data[cur] = load_json(f"{ECON_DIR}/{cur}.json")
        fx_data[cur]   = load_json(f"{FX_PERF_DIR}/{cur}.json")
        cot_data[cur]  = load_json(f"{COT_DIR}/{cur}.json")
        ext_data[cur]  = load_json(f"{EXT_DIR}/{cur}.json")
        all_econ[cur]  = econ_data[cur].get("data", {}) if econ_data[cur] else {}

    # Get all dates from fx-history
    fx_dates = sorted([
        f.replace(".json", "")
        for f in os.listdir(FX_HIST_DIR)
        if f.endswith(".json")
    ])

    print(f"Generating historical score snapshots for {len(fx_dates)} dates...")
    print(f"Date range: {fx_dates[0]} → {fx_dates[-1]}")
    print()

    # Loguear qué indicadores tienen datos históricos reales
    log_data_sources(fx_dates[0])

    all_snapshots = []
    skipped = 0

    for snap_date in fx_dates:
        scores = {}
        contributions_all = {}

        for cur in CURRENCIES:
            try:
                score, contribs = calculate_score_at_date(
                    cur, snap_date,
                    econ_data[cur], fx_data[cur], cot_data[cur], ext_data[cur],
                    all_econ
                )
                scores[cur] = score
                contributions_all[cur] = contribs
            except Exception as e:
                scores[cur] = 50.0
                skipped += 1

        pairs = generate_pairs(scores)

        snapshot = {
            "date":    snap_date,
            "scores":  {cur: {"score": scores[cur]} for cur in CURRENCIES},
            "pairs":   pairs,
            "note":    "interestRate+rateMomentum+outlook historically accurate; other indicators are current-data proxy",
        }
        all_snapshots.append(snapshot)

        # Show progress every 20
        if len(all_snapshots) % 20 == 0:
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            print(f"  [{snap_date}] {top[:3]}")

    # Save all.json
    history_data = {
        "generated":  date.today().isoformat(),
        "version":    "v6.3-retrospective-full",
        "snapshots":  all_snapshots,
        "note":       (
            "interestRate (10%), rateMomentum (7%), outlookScore (5%), fxPerformance1M (8%) "
            "son siempre históricamente precisos. "
            "GDP, inflación, PMIs, desempleo, COT y demás indicadores usan "
            "economic-data-history/ si está disponible (fetch_historical_econ_data.py), "
            "de lo contrario proxy estático actual."
        ),
    }
    with open(f"{SCORES_DIR}/all.json", "w") as f:
        json.dump(history_data, f, indent=2)

    print(f"\n✅ Generated {len(all_snapshots)} snapshots → {SCORES_DIR}/all.json")
    print(f"   Skipped errors: {skipped}")

    # Show final score distribution
    last = all_snapshots[-1]
    print(f"\nLatest scores ({last['date']}):")
    for cur, s in sorted(last["scores"].items(), key=lambda x: x[1]["score"], reverse=True):
        print(f"  {cur}: {s['score']:.1f}")
    print(f"\nActive pairs: {len(last['pairs'])}")
    for p in last["pairs"][:5]:
        print(f"  {p['long']}/{p['short']} diff={p['diff']} conf={p['confidence']}")


if __name__ == "__main__":
    main()
