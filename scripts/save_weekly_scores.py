"""
save_weekly_scores.py
─────────────────────
Ejecutado cada lunes por el workflow save-weekly-scores.yml.

1. Lee los JSONs actuales de economic-data/, fx-performance/, rates/, cot-data/
2. Replica el modelo de scoring de index.html (TRADING_WEIGHTS v6.3)
3. Guarda el snapshot en:
   - scores-history/YYYY-MM-DD.json   → scores del día + metadatos
   - scores-history/all.json          → historial acumulado completo
   - fx-history/YYYY-MM-DD.json       → tasas FX del día (para medir resultados)

El backtester (backtest_retrospective.py) luego cruza:
  score diferencial (semana N) × variación FX (semanas N+1, N+2, N+4, N+6)
para calcular hit rate, Sharpe y drawdown.
"""

import json
import os
import math
from datetime import date, datetime

TODAY = date.today().isoformat()

# ─── Paths ────────────────────────────────────────────────────────────────────
ECON_DIR   = "economic-data"
FX_DIR     = "fx-performance"
RATES_DIR  = "rates"
COT_DIR    = "cot-data"
EXT_DIR    = "extended-data"
OUT_DIR    = "scores-history"
FX_OUT_DIR = "fx-history"

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FX_OUT_DIR, exist_ok=True)


# ─── TRADING_WEIGHTS (v6.3 — must match index.html exactly) ──────────────────
TRADING_WEIGHTS = {
    "interestRate":      {"weight": 0.10},
    "fxPerformance1M":   {"weight": 0.08},
    "inflation":         {"weight": 0.07},
    "rateMomentum":      {"weight": 0.07},
    "currentAccount":    {"weight": 0.07},
    "gdpGrowth":         {"weight": 0.07},
    "cotPositioning":    {"weight": 0.06},
    "outlookScore":      {"weight": 0.05},
    "unemployment":      {"weight": 0.05},
    "tradeBalance":      {"weight": 0.04},
    "debt":              {"weight": 0.04},
    "termsOfTrade":      {"weight": 0.04},
    "production":        {"weight": 0.04},
    "servicesPMI":       {"weight": 0.04},
    "economicSurprise":  {"weight": 0.04},
    "retailSales":       {"weight": 0.03},
    "manufacturingPMI":  {"weight": 0.03},
    "bond10y":           {"weight": 0.03},
    "consumerConfidence":{"weight": 0.02},
    "businessConfidence":{"weight": 0.02},
    "wageGrowth":        {"weight": 0.01},
    "capitalFlows":      {"weight": 0.00},
}

# ─── Helper: load JSON safely ─────────────────────────────────────────────────
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ─── Helper: get current policy rate and 12-month momentum ───────────────────
def get_rate_and_momentum(currency):
    d = load_json(f"{RATES_DIR}/{currency}.json")
    obs = d.get("observations", [])
    if not obs:
        return None, None
    obs_sorted = sorted(obs, key=lambda x: x["date"], reverse=True)
    current = float(obs_sorted[0]["value"]) if obs_sorted else None
    # Find observation ~12 months ago
    rate_12m = None
    for o in obs_sorted:
        o_date = datetime.strptime(o["date"], "%Y-%m-%d").date()
        delta = (date.today() - o_date).days
        if delta >= 335:
            rate_12m = float(o["value"])
            break
    momentum = round(current - rate_12m, 4) if (current is not None and rate_12m is not None) else None
    return current, momentum


# ─── Helper: normalize a value to 0-100 given min/max ────────────────────────
def normalize(val, min_val, max_val, invert=False):
    if val is None:
        return None
    if max_val == min_val:
        return 50
    score = (val - min_val) / (max_val - min_val) * 100
    score = max(0, min(100, score))
    return round(100 - score if invert else score, 2)


# ─── Score individual indicators (replicates index.html logic) ───────────────
def score_indicator(key, value, all_data, currency):
    """
    Returns a 0-100 score for a single indicator.
    Mirrors the scoring functions in index.html calculateStrength().
    """
    if value is None:
        return None

    if key == "interestRate":
        # Relative to G8 median: higher = better
        all_rates = [d.get("interestRate") for d in all_data.values() if d.get("interestRate") is not None]
        if not all_rates:
            return 50
        median = sorted(all_rates)[len(all_rates) // 2]
        diff = value - median
        # ±4pp range → 0-100
        return round(max(0, min(100, 50 + diff * 12.5)), 2)

    elif key == "rateMomentum":
        # -2% to +2% → 0-100, with hawkish pause boost applied externally
        return round(max(0, min(100, 50 + value * 25)), 2)

    elif key == "inflation":
        # Target 2%: deviation in either direction is negative
        # Near-target (1.5-2.5%) = high score; deflation or >5% = low
        target = 2.0
        dist = abs(value - target)
        if dist <= 0.5:
            return round(90 - dist * 10, 2)
        elif dist <= 2.0:
            return round(80 - dist * 20, 2)
        else:
            return round(max(0, 40 - dist * 8), 2)

    elif key == "gdpGrowth":
        # >2.5% = strong, <0% = recession
        if value >= 3.0:   return 95
        if value >= 2.5:   return 85
        if value >= 1.5:   return 70
        if value >= 0.5:   return 50
        if value >= 0.0:   return 30
        return max(0, round(20 + value * 10, 2))

    elif key == "unemployment":
        # Lower = better; ~3.5-4% = neutral; >6% = poor
        if value <= 3.5:   return 90
        if value <= 4.5:   return 75
        if value <= 5.5:   return 55
        if value <= 7.0:   return 35
        return max(0, round(20 - (value - 7) * 4, 2))

    elif key == "currentAccount":
        # % GDP: positive = surplus bullish; -3% = neutral; <-5% = bearish
        return round(max(0, min(100, 50 + value * 6)), 2)

    elif key == "tradeBalance":
        # Relative to G8 — normalize cross-sectionally
        all_vals = [d.get("tradeBalance") for d in all_data.values() if d.get("tradeBalance") is not None]
        if len(all_vals) < 2:
            return 50
        return normalize(value, min(all_vals), max(all_vals))

    elif key == "debt":
        # Lower = better; <60% GDP = strong; >120% = weak
        if value <= 40:    return 90
        if value <= 60:    return 75
        if value <= 90:    return 55
        if value <= 120:   return 35
        return max(0, round(20 - (value - 120) * 0.3, 2))

    elif key == "production":
        # YoY industrial production: >3% = strong, <-2% = weak
        return round(max(0, min(100, 50 + value * 8)), 2)

    elif key == "retailSales":
        return round(max(0, min(100, 50 + value * 10)), 2)

    elif key == "wageGrowth":
        # 2-4% = healthy; too high = inflationary pressure (mixed)
        if 2.0 <= value <= 4.0: return 70
        if value > 4.0: return round(70 - (value - 4) * 5, 2)
        return round(max(0, 50 + value * 10), 2)

    elif key == "manufacturingPMI" or key == "servicesPMI":
        # >55 = strong, 50 = neutral, <45 = weak
        if value >= 55:   return round(min(100, 75 + (value - 55) * 2.5), 2)
        if value >= 50:   return round(50 + (value - 50) * 5, 2)
        if value >= 45:   return round(25 + (value - 45) * 5, 2)
        return max(0, round((value - 35) * 2.5, 2))

    elif key == "consumerConfidence" or key == "businessConfidence":
        # Relative to G8
        all_vals = [d.get(key) for d in all_data.values() if d.get(key) is not None]
        if len(all_vals) < 2:
            return 50
        return normalize(value, min(all_vals), max(all_vals))

    elif key == "termsOfTrade":
        all_vals = [d.get("termsOfTrade") for d in all_data.values() if d.get(key) is not None]
        if len(all_vals) < 2:
            return 50
        return normalize(value, min(all_vals), max(all_vals))

    elif key == "capitalFlows":
        return 50  # weight=0, value irrelevant

    elif key == "cotPositioning":
        # Normalized net position: >50k = bullish, <-50k = bearish
        return round(max(0, min(100, 50 + value / 1000)), 2)

    elif key == "outlookScore":
        # Pre-coded: Hawkish=85, Neutral=50, Dovish=20
        return float(value) if value is not None else 50

    elif key == "bond10y":
        # Relative to G8 — higher yield = more attractive carry
        all_vals = [d.get("bond10y") for d in all_data.values() if d.get("bond10y") is not None]
        if len(all_vals) < 2:
            return 50
        return normalize(value, min(all_vals), max(all_vals))

    elif key == "fxPerformance1M":
        # -5% to +5% → 0-100
        return round(max(0, min(100, 50 + value * 10)), 2)

    elif key == "economicSurprise":
        # ESI: already 0-100 scale from calculateESI
        return float(value) if value is not None else 50

    return 50


# ─── Main scoring function ────────────────────────────────────────────────────
def calculate_score(currency, econ_data, fx_data, cot_data, ext_data, all_econ):
    """
    Replicates calculateStrength() from index.html.
    Returns dict with score, contributions, data_quality, indicators.
    """
    data = {}

    # Economic indicators
    if econ_data:
        ed = econ_data.get("data", {})
        data.update({
            "gdpGrowth":          ed.get("gdpGrowth"),
            "inflation":          ed.get("inflation"),
            "unemployment":       ed.get("unemployment"),
            "currentAccount":     ed.get("currentAccount"),
            "tradeBalance":       ed.get("tradeBalance"),
            "debt":               ed.get("debt"),
            "production":         ed.get("production"),
            "retailSales":        ed.get("retailSales"),
            "wageGrowth":         ed.get("wageGrowth"),
            "manufacturingPMI":   ed.get("manufacturingPMI"),
            "servicesPMI":        ed.get("servicesPMI"),
            "consumerConfidence": ed.get("consumerConfidence"),
            "businessConfidence": ed.get("businessConfidence"),
            "termsOfTrade":       ed.get("termsOfTrade"),  # BUG FIX: was missing
        })

    # Extended data — BUG FIX: values are nested inside ["data"] subkey
    if ext_data:
        ext = ext_data.get("data", {})
        data["bond10y"]           = ext.get("bond10y")
        data["capitalFlows"]      = ext.get("capitalFlows")
        data["consumerConfidence"]= ext.get("consumerConfidence")
        data["businessConfidence"]= ext.get("businessConfidence")
        # rateMomentum: prefer ext-data (12M calculated) over rates-history fallback
        data["_rm_ext"]           = ext.get("rateMomentum")
        data["_rm24_ext"]         = ext.get("rateMomentum24M")

    # termsOfTrade: comes from economic-data, already loaded above (correct)

    # Policy rate and momentum
    rate, momentum_rates = get_rate_and_momentum(currency)
    data["interestRate"] = rate
    # Use ext rateMomentum if available (more precise), fall back to rates-history
    data["rateMomentum"] = data.pop("_rm_ext", None) or momentum_rates

    # outlookScore: derive from rateMomentum24M + rateMomentum (mirrors index.html logic)
    # index.html: Hawkish=90, Neutral=50, Dovish=10
    rm12  = data["rateMomentum"] or 0
    rm24  = data.pop("_rm24_ext", None) or rm12
    # Replicate FIX-A composite: use 24M only to amplify hawkish signal
    rm_composite = rm24 if rm24 >= 0.25 else rm12
    if rm_composite >= 0.25:
        outlook_str = "Hawkish"
    elif rm_composite <= -0.80:
        outlook_str = "Dovish"
    elif rm12 < -0.20:
        outlook_str = "Dovish"
    else:
        outlook_str = "Neutral"
    # Map to numeric score (matches index.html case 'outlook_direction')
    data["outlookScore"] = {"Hawkish": 90, "Neutral": 50, "Dovish": 10}[outlook_str]

    # COT
    if cot_data:
        data["cotPositioning"] = cot_data.get("netPosition")

    # FX performance
    if fx_data:
        data["fxPerformance1M"] = fx_data.get("fxPerformance1M")

    # ESI: use 50 as neutral default (prospective system will have real value)
    data["economicSurprise"] = data.get("economicSurprise", 50)

    # Build all_data dict for cross-sectional normalization
    # BUG FIX: include interestRate so the median-relative scoring works correctly
    all_data = {}
    for cur in CURRENCIES:
        ed2   = load_json(f"{ECON_DIR}/{cur}.json").get("data", {})
        ext2  = load_json(f"{EXT_DIR}/{cur}.json").get("data", {}) if os.path.exists(f"{EXT_DIR}/{cur}.json") else {}
        rate2, _ = get_rate_and_momentum(cur)
        all_data[cur] = {**ed2, **ext2, "interestRate": rate2}

    # Score each indicator
    weighted_sum = 0.0
    total_weight = 0.0
    contributions = {}
    indicators_with_data = 0

    for key, cfg in TRADING_WEIGHTS.items():
        w = cfg["weight"]
        if w == 0:
            continue
        val = data.get(key)
        s = score_indicator(key, val, all_data, currency)
        contributions[key] = {
            "value":  val,
            "score":  s,
            "weight": w,
            "contribution": round(s * w, 4) if s is not None else None
        }
        if s is not None:
            weighted_sum += s * w
            total_weight += w
            indicators_with_data += 1

    # Normalize by available weight
    score = round(weighted_sum / total_weight * 100 / 100, 2) if total_weight > 0 else 50
    # total_weight is already fraction of 1.0; weighted_sum is already 0-100 weighted
    score = round(weighted_sum / total_weight, 2) if total_weight > 0 else 50

    data_quality = round(indicators_with_data / (len(TRADING_WEIGHTS) - 1) * 100, 1)  # -1 for capitalFlows=0

    return {
        "score":          score,
        "data_quality":   data_quality,
        "indicators":     indicators_with_data,
        "contributions":  contributions,
    }


# ─── Collect FX spot rates for outcome measurement ───────────────────────────
def collect_fx_rates():
    """Fetch current FX rates from Frankfurter API for outcome tracking."""
    try:
        import urllib.request
        url = "https://api.frankfurter.app/latest?base=USD"
        with urllib.request.urlopen(url, timeout=10) as r:
            d = json.loads(r.read())
        rates = d.get("rates", {})
        rates["USD"] = 1.0
        return rates
    except Exception as e:
        print(f"  FX rates fetch failed: {e}")
        return {}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"=== Weekly Score Snapshot — {TODAY} ===\n")

    scores = {}
    for currency in CURRENCIES:
        econ = load_json(f"{ECON_DIR}/{currency}.json")
        fx   = load_json(f"{FX_DIR}/{currency}.json")
        cot  = load_json(f"{COT_DIR}/{currency}.json")
        ext  = load_json(f"{EXT_DIR}/{currency}.json") if os.path.exists(f"{EXT_DIR}/{currency}.json") else {}

        result = calculate_score(currency, econ, fx, cot, ext, None)
        scores[currency] = result
        sentiment = "🟢 Alcista" if result["score"] > 60 else ("🔴 Bajista" if result["score"] < 50 else "⚪ Neutral")
        print(f"  {currency}: {result['score']:.1f} {sentiment}  (calidad: {result['data_quality']:.0f}%)")

    # Rank currencies
    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    print(f"\n  Ranking: {' > '.join(c for c, _ in ranked)}")

    # Generate pair signals (top 3 vs bottom 3)
    pairs = []
    top3    = [c for c, _ in ranked[:3]]
    bottom3 = [c for c, _ in ranked[-3:]]
    for long_cur in top3:
        for short_cur in bottom3:
            diff = round(scores[long_cur]["score"] - scores[short_cur]["score"], 2)
            if diff >= 12:
                confidence = "Alta" if diff >= 20 else "Media"
                pairs.append({
                    "pair":       f"{long_cur}/{short_cur}",
                    "long":       long_cur,
                    "short":      short_cur,
                    "diff":       diff,
                    "confidence": confidence,
                })
    pairs.sort(key=lambda x: -x["diff"])
    print(f"\n  Señales activas: {len(pairs)} pares")
    for p in pairs[:5]:
        print(f"    LONG {p['long']} / SHORT {p['short']}  diff={p['diff']}  [{p['confidence']}]")

    # Save FX rates snapshot
    fx_rates = collect_fx_rates()
    if fx_rates:
        fx_snapshot = {"date": TODAY, "rates_vs_usd": fx_rates, "source": "Frankfurter/ECB"}
        with open(f"{FX_OUT_DIR}/{TODAY}.json", "w") as f:
            json.dump(fx_snapshot, f, indent=2)
        print(f"\n  FX snapshot guardado → fx-history/{TODAY}.json")

    # Build daily snapshot
    snapshot = {
        "date":        TODAY,
        "version":     "6.3.0",
        "scores":      {c: {"score": v["score"], "data_quality": v["data_quality"]} for c, v in scores.items()},
        "ranking":     [c for c, _ in ranked],
        "pairs":       pairs,
        "contributions": {c: v["contributions"] for c, v in scores.items()},
        "fx_rates":    fx_rates,
    }

    # Save daily file
    daily_path = f"{OUT_DIR}/{TODAY}.json"
    with open(daily_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"  Snapshot diario guardado → {daily_path}")

    # Update all.json (append or create)
    all_path = f"{OUT_DIR}/all.json"
    if os.path.exists(all_path):
        with open(all_path) as f:
            history = json.load(f)
    else:
        history = {"snapshots": [], "generated": TODAY}

    # Remove existing entry for today if re-running
    history["snapshots"] = [s for s in history["snapshots"] if s.get("date") != TODAY]
    # Append lightweight entry (no contributions — too heavy)
    history["snapshots"].append({
        "date":     TODAY,
        "scores":   snapshot["scores"],
        "ranking":  snapshot["ranking"],
        "pairs":    snapshot["pairs"],
        "fx_rates": fx_rates,
    })
    history["snapshots"].sort(key=lambda x: x["date"])
    history["last_update"] = TODAY
    history["total_snapshots"] = len(history["snapshots"])

    with open(all_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Historial acumulado → {all_path}  ({len(history['snapshots'])} snapshots)")
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
