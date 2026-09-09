#!/usr/bin/env python3
"""
fetch_capital_flows.py  v1.1 — Capital Flows panel (TIC top foreign holders +
ICI weekly fund flows)

WHY THIS FILE EXISTS
─────────────────────
Santiago asked (2026-09-09 session) for a Capital Flows module — cross-border
demand for dollar assets — positioned between Economic Matrix and FX Fair
Value. Two real, public, verified sources cover this:

  1. US Treasury TIC "Major Foreign Holders of Treasury Securities" (MFH),
     released monthly as Table 5:
     https://treasury.gov/resource-center/data-chart-center/tic/Documents/slt_table5.txt
     Tab-delimited. Header row is "Country" + up to 13 month columns
     (most-recent first). Country rows are followed by "All Other",
     "Grand Total" and three "Of Which: ..." aggregate rows that must be
     excluded from the per-country ranking.

  2. ICI "Combined Estimated Long-Term Flows and ETF Net Issuance", released
     weekly:
     https://www.ici.org/research/stats/combined_flows
     HTML page with a 5-column-of-history table (Equity/Domestic/World/
     Hybrid/Bond/Taxable/Municipal/Commodity/Total), millions of USD, most
     recent week first.

SIGNAL METHODOLOGY (industry standard, not a guessed threshold)
─────────────────────
Same principle already used by FX Fair Value (Z-score column) and the
Correlation panel (Z-score filter): a rolling z-score of the period-over-
period change, not a fixed-dollar or fixed-% threshold. A fixed-dollar
threshold is meaningless here because Japan's $1.1T position and Norway's
$0.2T position have completely different natural month-to-month noise —
the same $10bn move is a rounding error for one and a regime shift for the
other. z-score normalizes for that automatically, exactly like Fair Value's
Z-score column already does for spot-vs-model deviation.

  MoM Δ (or WoW Δ for ICI) → z = (Δ_latest − mean(Δ_trailing)) / stdev(Δ_trailing)
  z >= +1.5  → "Accumulating"
  z <= -1.5  → "Reducing"
  otherwise  → "Neutral"

REJECTED ALTERNATIVE: fixed-% MoM change (e.g. ">2% = accumulating"). Same
problem as above — normalizes for direction but not for each country's own
volatility regime. A cross-sectional percentile rank across countries was
also considered and rejected: it would say "China is accumulating" any month
China's Δ happens to rank higher than most other countries, even if China's
own Δ is unremarkable versus its own history — the wrong question for a
per-country conviction badge.

HISTORY-GATING (same UX pattern as FX Fair Value's "Accumulating business-
day history — 0/60d"): the TIC source file only carries a rolling 13-month
window, not enough to fit a stable trailing-window z-score on its own. This
script therefore builds its own history by appending each run's snapshot to
capital-flows-data/tic_history.json / ici_history.json, and gates the badge
on a minimum window (12 monthly observations for TIC, 12 weekly for ICI)
before computing z — before that, the JSON exposes progress
(history_len/gate) and the frontend should show "Neutral" / no badge rather
than a z-score fit on too few points.

OUTPUT
─────────────────────
  capital-flows-data/capital_flows.json
    tic.top10        — top 10 countries by latest holdings ($bn), each with
                        holdings, mom_change, z (or null if pre-gate),
                        signal ("Accumulating"/"Reducing"/"Neutral"/null)
    tic.as_of         — YYYY-MM of latest TIC column
    tic.gate          — {ready: bool, have: int, need: int}
    ici.weekly        — last N weeks, each {week_ending, equity, bond,
                        money_market_proxy: null (ICI does not publish a
                        standalone MMF weekly-flow line in this release;
                        see NOTE below), total, z, signal}
    ici.as_of         — week-ending date of latest row
    ici.gate          — {ready: bool, have: int, need: int}
    generated_at      — ISO8601 UTC

NOTE ON MONEY MARKET: the "Combined Estimated Long-Term Flows and ETF Net
Issuance" release explicitly EXCLUDES money market funds (it is long-term
flows only). A separate ICI release ("Weekly Money Market Fund Assets")
covers MMF, but was not fetched in v1.0 — money_market_proxy is left null,
disclosed rather than silently filled with a long-term-fund number that
would misrepresent it. Wiring that release is a follow-up, not a v1.0 gap
papered over.

USAGE
    python fetch_capital_flows.py
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from urllib.request import Request, urlopen

from log_utils import safe_print

TIC_URL = "https://treasury.gov/resource-center/data-chart-center/tic/Documents/slt_table5.txt"
ICI_URL = "https://www.ici.org/research/stats/combined_flows"

OUT_DIR = "capital-flows-data"
TIC_HISTORY_PATH = os.path.join(OUT_DIR, "tic_history.json")
ICI_HISTORY_PATH = os.path.join(OUT_DIR, "ici_history.json")
OUT_PATH = os.path.join(OUT_DIR, "capital_flows.json")

TIC_GATE = 12   # minimum monthly observations before z-score is trusted
ICI_GATE = 12   # minimum weekly observations before z-score is trusted
Z_ACCUM = 1.5
Z_REDUCE = -1.5

EXCLUDE_ROWS = {
    "all other", "grand total",
    "of which: foreign official",
    "of which: foreign official treasury bills",
    "of which: foreign official t-bonds & notes",
}

# ISO 3166-1 alpha-2 for flag-icons — only the set that actually appears in
# the MFH top rows historically; extend if a new country enters the top 10.
COUNTRY_ISO2 = {
    "japan": "jp", "united kingdom": "gb", "china, mainland": "cn",
    "belgium": "be", "canada": "ca", "cayman islands": "ky",
    "luxembourg": "lu", "france": "fr", "ireland": "ie", "taiwan": "tw",
    "switzerland": "ch", "singapore": "sg", "hong kong": "hk",
    "norway": "no", "india": "in", "brazil": "br", "saudi arabia": "sa",
    "korea, south": "kr", "united arab emirates": "ae", "israel": "il",
}


def _fetch(url: str, timeout: int = 20) -> str:
    req = Request(url, headers={
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _load_history(path: str) -> list[dict]:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            safe_print(f"[WARN] history file unreadable, starting fresh: {path}")
    return []


def _save_history(path: str, history: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, separators=(",", ":"))


def _zscore_signal(deltas: list[float], gate: int) -> tuple[float | None, str | None, int]:
    """deltas: chronological list of period-over-period changes, most recent last."""
    n = len(deltas)
    if n < gate:
        return None, None, n
    trailing = deltas[:-1][-(gate - 1):] if n > gate else deltas[:-1]
    if len(trailing) < 3:
        return None, None, n
    mean = statistics.mean(trailing)
    stdev = statistics.pstdev(trailing)
    if stdev == 0:
        return 0.0, "Neutral", n
    z = (deltas[-1] - mean) / stdev
    if z >= Z_ACCUM:
        signal = "Accumulating"
    elif z <= Z_REDUCE:
        signal = "Reducing"
    else:
        signal = "Neutral"
    return round(z, 2), signal, n


def parse_tic(raw: str) -> tuple[str, dict[str, float]]:
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Country\t"))
    header = lines[header_idx].split("\t")
    months = [h.strip() for h in header[1:] if h.strip()]
    latest_month = months[0]

    holdings: dict[str, float] = {}
    for ln in lines[header_idx + 1:]:
        cols = ln.split("\t")
        name = cols[0].strip()
        if not name or name.lower() in EXCLUDE_ROWS:
            continue
        if len(cols) < 2:
            continue
        try:
            value = float(cols[1].strip())
        except ValueError:
            continue
        holdings[name] = value
    return latest_month, holdings


def parse_ici(html: str) -> tuple[str, dict[str, float]]:
    """DOM-parsed per GUIDELINES.md's standing rule (v8.231.0): an HTML table
    scraper must use real DOM parsing, not a hand-written regex, even an
    anchored one — a regex still encodes an unverified guess about exact
    markup shape. Requires: pip install beautifulsoup4 lxml
    (this workflow's pip-install step must add both, same as
    update-calendar-completeness.yml already does for its own bs4 scrape)."""
    from bs4 import BeautifulSoup

    date_re = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
    num_re = re.compile(r"^-?[\d,]+$")

    soup = BeautifulSoup(html, "lxml")
    for table in soup.find_all("table"):
        rows = [
            [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
            for tr in table.find_all("tr")
        ]
        rows = [r for r in rows if r]
        if not rows:
            continue
        header = rows[0]
        date_cols = [i for i, c in enumerate(header) if date_re.match(c)]
        if not date_cols:
            continue  # not the flows table — keep looking
        latest_col = min(date_cols)  # most-recent week is the first date column
        week_ending = header[latest_col]

        row = {}
        for r in rows[1:]:
            if not r or latest_col >= len(r):
                continue
            label = r[0].lower()
            val_txt = r[latest_col]
            if not num_re.match(val_txt):
                continue
            val = float(val_txt.replace(",", ""))
            if label == "equity":
                row["equity"] = val
            elif label == "bond":
                row["bond"] = val
            elif label == "total":
                row["total"] = val
        if row:
            return week_ending, row

    raise ValueError("ICI release table not found — page layout may have changed")


def build_tic() -> dict:
    raw = _fetch(TIC_URL)
    latest_month, holdings = parse_tic(raw)

    history = _load_history(TIC_HISTORY_PATH)
    if not history or history[-1].get("month") != latest_month:
        history.append({"month": latest_month, "holdings": holdings})
        history = history[-36:]  # keep 3yr of monthly snapshots — plenty for a 12mo gate
        _save_history(TIC_HISTORY_PATH, history)

    top10_names = sorted(holdings, key=holdings.get, reverse=True)[:10]
    top10 = []
    for name in top10_names:
        series = [snap["holdings"].get(name) for snap in history if name in snap["holdings"]]
        series = [v for v in series if v is not None]
        deltas = [series[i] - series[i - 1] for i in range(1, len(series))]
        z, signal, have = _zscore_signal(deltas, TIC_GATE)
        mom_change = deltas[-1] if deltas else None
        top10.append({
            "country": name,
            "iso2": COUNTRY_ISO2.get(name.lower()),
            "holdings_bn": holdings[name],
            "mom_change_bn": round(mom_change, 1) if mom_change is not None else None,
            "z": z,
            "signal": signal,
        })

    have = len(history)
    return {
        "top10": top10,
        "as_of": latest_month,
        "gate": {"ready": have >= TIC_GATE, "have": have, "need": TIC_GATE},
    }


def build_ici() -> dict:
    html = _fetch(ICI_URL)
    week_ending, row = parse_ici(html)

    history = _load_history(ICI_HISTORY_PATH)
    if not history or history[-1].get("week_ending") != week_ending:
        history.append({"week_ending": week_ending, **row})
        history = history[-104:]  # 2yr of weekly snapshots
        _save_history(ICI_HISTORY_PATH, history)

    series = [snap.get("total") for snap in history if snap.get("total") is not None]
    deltas = [series[i] - series[i - 1] for i in range(1, len(series))]
    z, signal, have_pts = _zscore_signal(deltas, ICI_GATE)

    weekly = [
        {
            "week_ending": snap["week_ending"],
            "equity": snap.get("equity"),
            "bond": snap.get("bond"),
            "total": snap.get("total"),
        }
        for snap in history[-8:]
    ]

    have = len(history)
    return {
        "weekly": weekly,
        "latest_signal": {"z": z, "signal": signal},
        "as_of": week_ending,
        "gate": {"ready": have >= ICI_GATE, "have": have, "need": ICI_GATE},
    }


def main() -> None:
    doc: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    try:
        doc["tic"] = build_tic()
        safe_print(f"[OK] TIC — {doc['tic']['as_of']}, {len(doc['tic']['top10'])} countries, "
                    f"gate {doc['tic']['gate']['have']}/{doc['tic']['gate']['need']}")
    except Exception as exc:
        safe_print(f"[ERROR] TIC fetch/parse failed: {exc}")
        doc["tic"] = None

    try:
        doc["ici"] = build_ici()
        safe_print(f"[OK] ICI — {doc['ici']['as_of']}, "
                    f"gate {doc['ici']['gate']['have']}/{doc['ici']['gate']['need']}")
    except Exception as exc:
        safe_print(f"[ERROR] ICI fetch/parse failed: {exc}")
        doc["ici"] = None

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(doc, f, separators=(",", ":"))

    if doc["tic"] is None and doc["ici"] is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
