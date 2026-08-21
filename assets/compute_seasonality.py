#!/usr/bin/env python3
"""
compute_seasonality.py  v2.0 — Daily-granularity seasonality for the beta Seasonality panel

WHY THIS FILE EXISTS
─────────────────────
Dool requested Seasonality (2026-08-20 email thread — recurring statistical
tendencies: fiscal year-end repatriation, commodity cycles, tax-payment
flows, etc.). Per Santiago's reply, this needs no new data source: it runs
entirely on the daily OHLC history the terminal already stores
(`ohlc-data/{pair}.json`), widened from 3y to 10y in v8.192.0 specifically
so this feature would have a real multi-year sample instead of a thin one.

SCOPE HISTORY — v1.0 was monthly-only, v2.0 is day-of-year
─────────────────────
v1.0 shipped a 13-point (Jan-1 baseline + 12 monthly) curve, flagged in its
own docstring as a real precision gap versus reference tools (analytics-
fx.com, Seasonax, EquityClock) that plot day-level curves (~250-260 points/
year) with visible day-to-day noise. Santiago compared the two side by side
and asked for the day-level version now that 10y of daily history is
confirmed live (see the 2026-08-21 `update-ohlc.yml` run: 51/51 symbols,
`Period: 10y / 1d`, no errors). v2.0 replaces the monthly `curve` with a
day-of-year curve built the same way — average-of-averages, additive, not
compounded (see METHODOLOGY) — just at daily instead of monthly resolution.
The `windows` table ("Strongest recurring windows") is intentionally left
on its existing monthly cadence: a day-level version of that table would
mean day-level start/end window search, which is a materially different
(and much noisier — many more candidate windows to multiple-test against)
feature that wasn't asked for here. Only the line chart's granularity
changed.

METHODOLOGY
─────────────────────
Curve (day-of-year, NEW in v2.0):
  - daily return(date) = pct change from the PREVIOUS bar in the series
    (chronological, whatever bar actually precedes it) to this bar's
    close. A Friday->Monday gap is captured entirely in Monday's return —
    no separate weekend/holiday point is fabricated; Saturday and Sunday
    simply have no bar and so contribute nothing (see below).
  - Returns are grouped by calendar (month, day), ignoring year — e.g.
    every "Mar 15" close-to-close return across all available years goes
    into one bucket. Feb 29 is its own bucket (fewer qualifying years,
    same as any other calendar day with fewer trading-day matches).
  - The curve walks a full reference leap year, Jan 1 -> Dec 31 (366 days,
    so Feb 29 is included), in chronological order. Each point's cum_pct
    = running sum of that calendar day's AVERAGE return across all years
    with data for that day. A calendar day with zero qualifying years
    (every weekend, most holidays) contributes avg=0 — i.e. "no trading
    day fell on this calendar date," not "data missing for a day that
    traded." This is the same additive-average approach v1.0 used for
    monthly buckets, applied at daily resolution instead — not a switch
    to compounding, so it stays a smooth cumulative path rather than
    compounding day-of-year averages that were never actually adjacent
    within a single year.
  - `n_years` on the top-level output is the count of distinct calendar
    years contributing at least one qualifying MONTHLY return (unchanged
    definition from v1.0 — see below), not the daily count; the two are
    typically identical or very close (daily coverage is a superset,
    since it only needs 2 consecutive bars vs. monthly's two full
    month-end anchors) but the monthly definition is kept as the single
    source of truth for the "Xy lookback" label and the MIN_YEARS gate,
    to avoid the same metric silently meaning two different things
    depending on which part of the file reads it.

Windows (monthly, UNCHANGED from v1.0):
  For each pair, and each calendar year with enough daily bars in that
  year's boundary months:
    - monthly return(year, month) = pct change from that month's LAST
      closed bar to the PRIOR calendar month's last closed bar (chained,
      so it compounds correctly across a window, not a naive sum).
    - A year only contributes to month M if both that month's and the
      prior month's last-bar-of-month exist in the data (handles the
      2016-08-22 start mid-month cleanly).
  Every contiguous 1-12 month window (same calendar year, no wraparound
  past December) is tested; per-year window return is the CHAINED
  compound of that window's monthly returns for years where every month
  in the window qualifies. avg_return = mean of those per-year window
  returns. win_rate = % of qualifying years whose window return shares
  avg_return's sign. sharpe = avg_return / stdev(per-year window
  returns), unannualized (a same-length relative ranking aid only, not
  cross-asset-class-comparable). Only windows with win_rate >= 70% are
  kept; top 3 by |avg_return| are written — matches
  _sznRenderWindows() in dashboard-beta.js.

MIN_YEARS = 5 (GUIDELINES data-integrity rule — a pair without a real
multi-year sample gets no output file at all, not an estimated one; the
beta JS panel already handles a missing file by saying so explicitly,
see dashboard-beta.js's _sznLoad() catch branch). This gate is evaluated
against the monthly qualifying-years count, same as v1.0 — the daily
curve is only built for pairs that already clear this bar.

PAIRS LIST
─────────────────────
Same 32 FX pairs as log_fair_value_inputs.py's PAIRS (identical list,
duplicated here rather than imported — same "no shared-module pattern in
this repo yet" reasoning already documented in log_dollar_smile_inputs.py's
header).

Runs once daily, gated to the same D1-finalization schedule as
calculate_technical_levels.py (only meaningful to recompute right after
ohlc-data actually refreshes) — see update-ohlc.yml's step.
"""

import json
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SITE_DIR = Path(os.environ.get("SITE_DIR", "."))
OHLC_DIR = SITE_DIR / "ohlc-data"
OUT_DIR = SITE_DIR / "seasonality-data"

MIN_YEARS = 5
MIN_WIN_RATE = 70.0
TOP_N_WINDOWS = 3

# Reference leap year purely for walking a full Jan-1..Dec-31 calendar
# (366 days, so Feb 29 is included) in chronological order when building
# the day-of-year curve — no bar from this specific year is ever read;
# it's only used for its calendar shape via timedelta arithmetic.
_REF_LEAP_YEAR = 2000

# Mirrors log_fair_value_inputs.py's PAIRS exactly (32 FX pairs tracked by
# the terminal) — see header note above re: dashboard-beta.js's SZN_PAIRS.
PAIRS = [
    "eurusd", "gbpusd", "usdjpy", "audusd", "usdchf", "usdcad", "nzdusd",
    "usdnok", "usdsek", "eurnok", "eursek", "eurgbp", "eurjpy", "eurchf",
    "eurcad", "euraud", "gbpjpy", "gbpchf", "gbpcad", "audjpy", "audnzd",
    "audchf", "cadjpy", "chfjpy", "nzdjpy", "eurnzd", "gbpaud", "gbpnzd",
    "audcad", "cadchf", "nzdcad", "nzdchf",
]


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  WARN: could not read {path}: {e}")
        return None


def _month_end_closes(bars):
    """
    Returns {(year, month): last_close_in_that_month}, keyed by the
    calendar month the bar's own date falls in (not a fixed trading-
    calendar assumption — just the actual last bar seen per (y, m) in
    the data, whatever weekday/holiday pattern that year had).
    """
    out = {}
    for b in bars:
        try:
            d = datetime.strptime(b["time"], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
        close = b.get("close")
        if close is None:
            continue
        key = (d.year, d.month)
        # bars are already time-ordered ascending in ohlc-data/*.json
        # (confirmed against fetch_ohlc.py's own write order) — last
        # write per key wins, i.e. the last bar of that month.
        out[key] = close
    return out


def _monthly_returns(month_end):
    """
    {(year, month): pct_return} — pct change from the PRIOR calendar
    month's month-end close to this month's month-end close. Only
    emitted where both anchors exist (see module docstring).
    """
    returns = {}
    for (y, m), close in month_end.items():
        py, pm = (y - 1, 12) if m == 1 else (y, m - 1)
        prev = month_end.get((py, pm))
        if prev is None or prev == 0:
            continue
        returns[(y, m)] = (close / prev - 1.0) * 100.0
    return returns


def _daily_returns(bars):
    """
    {date: pct_return} — chronological close-to-close return, keyed by
    the LATER bar's own date. Uses whatever bar actually precedes it in
    the series (bars are already time-ordered ascending — confirmed
    against fetch_ohlc.py's write order), so a Friday->Monday gap is
    captured entirely as Monday's return; no separate weekend/holiday
    point is fabricated for the days in between (see module docstring).
    """
    out = {}
    prev_close = None
    for b in bars:
        try:
            d = datetime.strptime(b["time"], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
        close = b.get("close")
        if close is None:
            continue
        if prev_close is not None and prev_close != 0:
            out[d] = (close / prev_close - 1.0) * 100.0
        prev_close = close
    return out


def _group_by_calendar_day(daily_returns):
    """{(month, day): [pct_return, ...]} across all years, keyed by
    calendar day ignoring year. Feb 29 is its own key (fewer qualifying
    years than every other day, same as any calendar day with fewer
    trading-day matches — not special-cased or merged into Feb 28)."""
    grouped = {}
    for d, ret in daily_returns.items():
        grouped.setdefault((d.month, d.day), []).append(ret)
    return grouped


def _build_daily_curve(grouped):
    """
    367-point curve: index 0 = pre-Jan-1 zero baseline, the remaining
    366 points walk Jan 1 -> Dec 31 of a reference leap year in
    chronological order (so Feb 29 is included). Each point's cum_pct is
    the running sum of that calendar day's AVERAGE return across all
    years with data for that day (average-of-averages, additive — same
    approach v1.0 used for monthly buckets, just at daily resolution).
    A calendar day with 0 qualifying years (most weekends/holidays)
    contributes avg=0 and is marked n_years=0, so the frontend can tell
    "no trading day fell here" apart from "1 lone qualifying year" if it
    ever needs to (currently only used for display, not filtering).
    """
    curve = [{"month": 0, "day": 0, "cum_pct": 0.0, "n_years": 0}]
    cum = 0.0
    start = datetime(_REF_LEAP_YEAR, 1, 1)
    for i in range(366):
        d = start + timedelta(days=i)
        vals = grouped.get((d.month, d.day), [])
        avg = statistics.mean(vals) if vals else 0.0
        cum += avg
        curve.append({"month": d.month, "day": d.day, "cum_pct": round(cum, 4), "n_years": len(vals)})
    return curve


def _window_returns(monthly_returns, start_m, end_m):
    """Per-year compounded return for the contiguous [start_m, end_m]
    window (same calendar year, no wraparound — see module docstring),
    only for years where every month in the window qualifies."""
    years = sorted({y for (y, m) in monthly_returns})
    out = []
    for y in years:
        months = range(start_m, end_m + 1)
        if not all((y, m) in monthly_returns for m in months):
            continue
        compound = 1.0
        for m in months:
            compound *= (1.0 + monthly_returns[(y, m)] / 100.0)
        out.append((compound - 1.0) * 100.0)
    return out


def _find_windows(monthly_returns):
    candidates = []
    for start_m in range(1, 13):
        for end_m in range(start_m, 13):
            rets = _window_returns(monthly_returns, start_m, end_m)
            if len(rets) < MIN_YEARS:
                continue
            avg = statistics.mean(rets)
            if avg == 0:
                continue
            same_sign = sum(1 for r in rets if (r >= 0) == (avg >= 0))
            win_rate = round(100.0 * same_sign / len(rets), 1)
            if win_rate < MIN_WIN_RATE:
                continue
            sharpe = round(avg / statistics.stdev(rets), 2) if len(rets) > 1 and statistics.stdev(rets) > 0 else None
            candidates.append({
                "start_month": start_m,
                "end_month": end_m,
                "dir": "Long" if avg >= 0 else "Short",
                "win_rate": win_rate,
                "avg_return": round(avg, 2),
                "sharpe": sharpe,
                "n_years": len(rets),
            })
    candidates.sort(key=lambda w: abs(w["avg_return"]), reverse=True)
    return candidates[:TOP_N_WINDOWS]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written, skipped = [], []
    for pair in PAIRS:
        src = OHLC_DIR / f"{pair}.json"
        bars = _load_json(src)
        if not bars or not isinstance(bars, list):
            skipped.append((pair, "no ohlc-data file"))
            continue

        month_end = _month_end_closes(bars)
        monthly_returns = _monthly_returns(month_end)
        n_years = len({y for (y, m) in monthly_returns})

        if n_years < MIN_YEARS:
            skipped.append((pair, f"only {n_years}y qualifying — needs {MIN_YEARS}y"))
            continue

        windows = _find_windows(monthly_returns)

        daily_returns = _daily_returns(bars)
        grouped = _group_by_calendar_day(daily_returns)
        curve = _build_daily_curve(grouped)

        out = {
            "pair": pair,
            "years": n_years,
            "curve": curve,
            "windows": windows,
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        (OUT_DIR / f"{pair}.json").write_text(json.dumps(out, separators=(",", ":")))
        written.append(pair)

    print(f"Done — {len(written)}/{len(PAIRS)} pairs written to {OUT_DIR}/.")
    if skipped:
        print("Skipped:")
        for pair, reason in skipped:
            print(f"  {pair}: {reason}")


if __name__ == "__main__":
    main()
