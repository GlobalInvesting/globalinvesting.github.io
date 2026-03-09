"""
backtest_retrospective.py
──────────────────────────
Backtesting RETROSPECTIVO del modelo de scoring forex v6.3.

METODOLOGÍA:
  1. Carga el historial de scores (scores-history/all.json)
  2. Para cada snapshot semanal, toma los pares activos (diff >= 12pt)
  3. Mide la variación FX real en la ventana de 1-6 semanas siguientes
     usando fx-history/
  4. Una señal LONG A/SHORT B es correcta si A sube vs B en el período
  5. Calcula: Hit Rate, Sharpe Ratio, Max Drawdown, avg ganancia/pérdida

NOTA SOBRE LOOK-AHEAD BIAS:
  El backtesting retrospectivo aproximado NO tiene look-ahead bias en los
  scores (cada snapshot usa solo datos disponibles en esa fecha), pero SÍ
  puede tener sesgos en los datos económicos si los indicadores se revisaron
  a posteriori. Documentamos esto explícitamente en los resultados.

USO:
  python3 scripts/backtest_retrospective.py
  python3 scripts/backtest_retrospective.py --min-diff 15 --window 4
  python3 scripts/backtest_retrospective.py --output backtest_results.json
"""

import json
import os
import math
import argparse
from datetime import date, datetime, timedelta

SCORES_DIR  = "scores-history"
FX_DIR      = "fx-history"
RESULTS_DIR = "backtest-results"

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]


# ─── Load all snapshots ────────────────────────────────────────────────────────
def load_history():
    path = f"{SCORES_DIR}/all.json"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run save_weekly_scores.py first.")
        return []
    with open(path) as f:
        d = json.load(f)
    snapshots = d.get("snapshots", [])
    print(f"Loaded {len(snapshots)} snapshots ({snapshots[0]['date'] if snapshots else 'none'} → {snapshots[-1]['date'] if snapshots else 'none'})")
    return snapshots


# ─── Load FX rates for a given date (nearest available) ──────────────────────
def load_fx_rates(target_date_str, tolerance_days=5):
    """
    Returns rates dict {currency: rate_vs_usd} for the target date.
    Searches within tolerance_days if exact date not found.
    """
    target = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    best_path, best_delta = None, None

    for i in range(tolerance_days + 1):
        for sign in [1, -1]:
            candidate = (target + timedelta(days=i * sign)).isoformat()
            path = f"{FX_DIR}/{candidate}.json"
            if os.path.exists(path):
                delta = i
                if best_delta is None or delta < best_delta:
                    best_path, best_delta = path, delta

    if not best_path:
        return None, None

    with open(best_path) as f:
        d = json.load(f)
    return d.get("rates_vs_usd", {}), d.get("date")


# ─── Calculate return for a pair ─────────────────────────────────────────────
def calculate_pair_return(long_cur, short_cur, rates_entry, rates_exit):
    """
    Calculates the return of LONG long_cur / SHORT short_cur.
    All rates are vs USD. Returns % gain (positive = correct direction).
    """
    if not rates_entry or not rates_exit:
        return None

    def get_rate(rates, cur):
        if cur == "USD":
            return 1.0
        return rates.get(cur)

    entry_long  = get_rate(rates_entry, long_cur)
    entry_short = get_rate(rates_entry, short_cur)
    exit_long   = get_rate(rates_exit, long_cur)
    exit_short  = get_rate(rates_exit, short_cur)

    if None in [entry_long, entry_short, exit_long, exit_short]:
        return None
    if entry_long == 0 or entry_short == 0:
        return None

    # Cross rate: long_cur / short_cur
    entry_cross = entry_long / entry_short
    exit_cross  = exit_long  / exit_short

    return round((exit_cross / entry_cross - 1) * 100, 4)


# ─── Statistics helpers ───────────────────────────────────────────────────────
def sharpe_ratio(returns, risk_free=0.0):
    """Weekly Sharpe (annualized ×√52)."""
    if len(returns) < 2:
        return None
    mean   = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std    = math.sqrt(variance)
    if std == 0:
        return None
    return round((mean - risk_free) / std * math.sqrt(52), 4)


def max_drawdown(equity_curve):
    """Maximum peak-to-trough drawdown from cumulative return series."""
    if len(equity_curve) < 2:
        return None
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
    return round(max_dd * 100, 2)


def win_rate(results):
    wins = sum(1 for r in results if r > 0)
    return round(wins / len(results) * 100, 1) if results else None


# ─── Main backtest logic ──────────────────────────────────────────────────────
def run_backtest(snapshots, min_diff=12, windows_weeks=(1, 2, 4, 6)):
    """
    For each snapshot, evaluate all active pair signals against future FX moves.
    Returns detailed trade log and summary statistics.
    """
    trades = []   # All individual trades
    skipped = 0

    snap_dates = [s["date"] for s in snapshots]

    for i, snap in enumerate(snapshots):
        snap_date   = snap["date"]
        scores      = snap.get("scores", {})
        pairs       = snap.get("pairs", [])
        fx_entry, _ = load_fx_rates(snap_date)

        if fx_entry is None:
            skipped += 1
            continue

        # Filter to pairs meeting minimum differential
        active_pairs = [p for p in pairs if p.get("diff", 0) >= min_diff]

        for pair_info in active_pairs:
            long_cur  = pair_info["long"]
            short_cur = pair_info["short"]
            diff      = pair_info["diff"]
            conf      = pair_info.get("confidence", "Media")

            # Evaluate at each horizon
            for w in windows_weeks:
                exit_date = (datetime.strptime(snap_date, "%Y-%m-%d") + timedelta(weeks=w)).date().isoformat()
                fx_exit, actual_exit_date = load_fx_rates(exit_date)

                if fx_exit is None:
                    continue  # Exit data not yet available

                ret = calculate_pair_return(long_cur, short_cur, fx_entry, fx_exit)
                if ret is None:
                    continue

                trades.append({
                    "entry_date":      snap_date,
                    "exit_date":       actual_exit_date or exit_date,
                    "long":            long_cur,
                    "short":           short_cur,
                    "pair":            f"{long_cur}/{short_cur}",
                    "score_diff":      diff,
                    "confidence":      conf,
                    "horizon_weeks":   w,
                    "return_pct":      ret,
                    "correct":         ret > 0,
                    "long_score":      scores.get(long_cur, {}).get("score"),
                    "short_score":     scores.get(short_cur, {}).get("score"),
                })

    print(f"\n  Total trades evaluated: {len(trades)} | Skipped snapshots: {skipped}")
    return trades


# ─── Aggregate results by window ─────────────────────────────────────────────
def aggregate_results(trades, windows_weeks=(1, 2, 4, 6)):
    summary = {}

    for w in windows_weeks:
        w_trades = [t for t in trades if t["horizon_weeks"] == w]
        if not w_trades:
            summary[f"{w}w"] = {"trades": 0}
            continue

        returns     = [t["return_pct"] for t in w_trades]
        correct     = [t["return_pct"] for t in w_trades if t["correct"]]
        incorrect   = [t["return_pct"] for t in w_trades if not t["correct"]]

        # Equity curve (cumulative returns)
        equity = [100.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r / 100))

        # By confidence level
        by_conf = {}
        for conf in ["Alta", "Media"]:
            conf_trades = [t for t in w_trades if t["confidence"] == conf]
            if conf_trades:
                conf_rets = [t["return_pct"] for t in conf_trades]
                by_conf[conf] = {
                    "trades":    len(conf_trades),
                    "hit_rate":  win_rate(conf_rets),
                    "avg_return": round(sum(conf_rets) / len(conf_rets), 4),
                    "sharpe":    sharpe_ratio(conf_rets),
                }

        summary[f"{w}w"] = {
            "trades":          len(w_trades),
            "hit_rate":        win_rate(returns),
            "avg_return_pct":  round(sum(returns) / len(returns), 4),
            "avg_win_pct":     round(sum(correct) / len(correct), 4) if correct else 0,
            "avg_loss_pct":    round(sum(incorrect) / len(incorrect), 4) if incorrect else 0,
            "sharpe_annualized": sharpe_ratio(returns),
            "max_drawdown_pct":  max_drawdown(equity),
            "total_return_pct":  round(equity[-1] - 100, 2),
            "by_confidence":   by_conf,
        }

    # Overall (all windows combined, deduplicate by entry date + pair)
    seen = set()
    unique_trades = []
    for t in trades:
        if t["horizon_weeks"] == 4:  # Use 4-week window as primary
            key = (t["entry_date"], t["pair"])
            if key not in seen:
                seen.add(key)
                unique_trades.append(t)

    if unique_trades:
        all_rets = [t["return_pct"] for t in unique_trades]
        summary["overall_4w"] = {
            "trades":     len(unique_trades),
            "hit_rate":   win_rate(all_rets),
            "sharpe":     sharpe_ratio(all_rets),
            "avg_return": round(sum(all_rets) / len(all_rets), 4),
        }

    return summary


# ─── Print report ─────────────────────────────────────────────────────────────
def print_report(summary, trades, min_diff):
    print("\n" + "═" * 60)
    print(f"  BACKTEST RESULTS  |  min_diff={min_diff}pt  |  {len(trades)} total trades")
    print("═" * 60)

    for window, stats in summary.items():
        if window == "overall_4w":
            continue
        if stats.get("trades", 0) == 0:
            print(f"\n  [{window}]  No data yet")
            continue

        print(f"\n  [{window}]  {stats['trades']} trades")
        print(f"    Hit Rate:        {stats['hit_rate']}%")
        print(f"    Avg Return:      {stats['avg_return_pct']:+.3f}%")
        print(f"    Avg Win:         {stats['avg_win_pct']:+.3f}%")
        print(f"    Avg Loss:        {stats['avg_loss_pct']:+.3f}%")
        print(f"    Sharpe (ann.):   {stats['sharpe_annualized']}")
        print(f"    Max Drawdown:    -{stats['max_drawdown_pct']}%")
        print(f"    Total Return:    {stats['total_return_pct']:+.2f}%")
        if stats.get("by_confidence"):
            for conf, cs in stats["by_confidence"].items():
                print(f"    [{conf}]  HR={cs['hit_rate']}%  avg={cs['avg_return']:+.3f}%  Sharpe={cs['sharpe']}")

    if "overall_4w" in summary and summary["overall_4w"]["trades"] > 0:
        o = summary["overall_4w"]
        print(f"\n  [RESUMEN 4W]  trades={o['trades']}  HR={o['hit_rate']}%  Sharpe={o['sharpe']}  avg={o['avg_return']:+.3f}%")

    print("\n" + "─" * 60)
    print("  NOTA: Backtest retrospectivo con scores reconstruidos.")
    print("  Los scores usan datos económicos actuales como proxy de")
    print("  los valores vigentes en cada fecha (look-ahead bias menor")
    print("  para datos macro de baja frecuencia, mayor para PMI/COT).")
    print("─" * 60)


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-diff",  type=float, default=12.0,
                        help="Minimum score differential to include a pair (default: 12)")
    parser.add_argument("--windows",   type=str,   default="1,2,4,6",
                        help="Evaluation windows in weeks (default: 1,2,4,6)")
    parser.add_argument("--output",    type=str,   default=None,
                        help="Optional path to save results JSON")
    args = parser.parse_args()

    windows = [int(w) for w in args.windows.split(",")]

    print(f"=== Backtest Retrospectivo — {date.today().isoformat()} ===")
    print(f"    min_diff={args.min_diff}pt  |  windows={windows}w\n")

    snapshots = load_history()
    if not snapshots:
        print("No hay snapshots disponibles. El sistema prospectivo aún no ha acumulado datos.")
        print("Los resultados estarán disponibles después de varias semanas de ejecución.")
        return

    trades  = run_backtest(snapshots, min_diff=args.min_diff, windows_weeks=windows)
    summary = aggregate_results(trades, windows_weeks=windows)
    print_report(summary, trades, args.min_diff)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        "generated":   date.today().isoformat(),
        "parameters":  {"min_diff": args.min_diff, "windows": windows},
        "summary":     summary,
        "trades":      trades,
        "note":        "Retrospective backtest. Minor look-ahead bias possible for low-frequency macro data.",
    }

    # Always save latest
    latest_path = f"{RESULTS_DIR}/latest.json"
    with open(latest_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n  Resultados guardados → {latest_path}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"  También guardado → {args.output}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
