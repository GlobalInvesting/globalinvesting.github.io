# Global Investing — FX Terminal

A professional-grade foreign exchange monitoring platform for serious market participants. Consolidates live FX prices (Finnhub WebSocket, tick-by-tick), central bank policy rates, institutional positioning, cross-asset flows, and AI-assisted market narrative into a single unified dashboard.

**[globalinvesting.github.io](https://globalinvesting.github.io/)**

![Status](https://img.shields.io/badge/Status-Live-success) ![License](https://img.shields.io/badge/License-Proprietary-red)

---

## What it covers

- **Real-time price feeds** — Major FX pairs, commodities (XAU, WTI), crypto (BTC), DXY, and US 10Y yield
- **Currency strength heatmap** — 8×8 grid showing major currency performance across all pairs
- **AI market narrative** — 2–3 sentence regime summary updated 4× daily alongside market signals
- **CFTC COT positioning** — Leveraged Funds net positioning (Disaggregated TFF, Options+Futures Combined) from CFTC.gov, updated weekly
- **US Treasury yield curve** — 3M, 2Y, 5Y, 10Y, 30Y, updated daily
- **Cross-asset risk monitor** — SPX, Gold, WTI, BTC, DXY, Nikkei, Stoxx correlations with stress scoring
- **Central bank policy rates** — All 8 major currencies CBs with rate cycle direction
- **CB Rate Expectations** — OIS-derived forward consensus direction (Cut/Hold/Hike) and 30-day CIP forward rate for each major central bank at their next meeting; sourced from CME FedWatch, ECB SDW ESTER, BoE SONIA, BoJ TONA, ASX Rate Indicator, BoC CORRA, SNB SARON, and RBNZ OCR overnight
- **FX liquidity & sessions** — 24-hour liquidity profile with live session indicator
- **Positioning Bias** — ATM implied volatility from CBOE-listed FX ETF options (FXE, FXB, FXY, FXA) combined with COT Leveraged Funds directional bias and 25-delta Risk Reversals from Saxo Bank (1M tenor, indicative mid-market). When ≥4 weeks of IV history are available, an IV Rank column (0–100 scale, where 100 = historically expensive vol) replaces the COT bias column.
- **Pair detail panel** — Linked right-panel showing price, 1W change, HV30, ATM IV, IV−HV, LF net, AM net, carry differential, and LF/AM alignment badge for the currently selected chart pair (Eikon-style linked panel). Hover tooltips on every metric cell explain the data source and interpretation.
- **ETF Options IV** — 8-row implied volatility panel: VIX, VIX9D, VVIX, MOVE, GLD IV, TLT IV, EEM IV, EFA IV. Colour-coded bar (red = elevated, amber = mid, green = low). Sources: CBOE indices and yfinance ETF options. Refreshes every 10 minutes.
- **Configurable price alerts** — Threshold alerts for VIX, EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CHF, Gold, US 10Y, and MOVE. Direction (`>` / `<`) and numeric threshold configurable per alert. Fires browser Notifications API on trigger. Persisted in localStorage across sessions. Checks every 5 minutes.
- **Panel data export** — CSV and JSON export for FX pairs, COT positioning, yield curve, and carry data; timestamped filenames, downloaded client-side from in-memory caches
- **Keyboard navigation** — `G`/`C`/`R`/`X`/`M`/`Y`/`K`/`D`/`N`/`B` jump to panels; `↑`/`↓` navigate FX table rows and load chart; `?` opens shortcut legend
- **Per-panel timestamps** — Every data panel displays its source and last update time in the user's local timezone. Panels with AI-generated signals also show when the data was loaded from the engine.
- **AI signal evidence traceability** — Each AI-generated market signal carries an `evidence[]` field listing the exact data values that motivated it (e.g. "VIX: 23.9", "Fed rate: 4.50%"). Evidence chips are hidden by default and revealed by clicking the signal row.
- **Economic calendar** — TradingView embed with real-time event actuals
- **Economic Surprises** — CESI-style normalised surprise index for all 8 major currencies, computed from ForexFactory calendar actual-vs-consensus over a 90-day rolling window. Clicking any currency row opens a full-screen detail modal with: (1) LightweightCharts v5 area series showing the 30-day rolling index history stepped weekly; (2) metrics bar (index, beats, misses, N, beat rate); (3) individual events table with beat / miss / in-line badges for the full 90-day window. Methodology mirrors Citi CESI convention — beat rate scaled to [−100, +100].
- **News Feed** — Dedicated tab (shortcut: N) with 52 FX-relevant headlines from FXStreet, ForexLive, Reuters FX, ECB, BoE, BoJ, RBA, RBNZ, BoC, SNB, Federal Reserve, Finnhub, NewsData.io, Marc to Market and others. Single-row accordion layout with impact dot, currency tag, relative age indicator, and full-headline tooltip. Filterable by currency (8 major currencies) and impact level (High / Med / All). Engine updates hourly; terminal re-checks every 2 minutes via HTTP ETag.
- **Bank Research** — Dedicated tab (shortcut: B) showing institutional FX research notes from ING Think, Saxo Bank (SaxoStrats), MUFG Research, DailyFX (IG Group), and BIS central banker speeches. Metadata-only pipeline: title · bank · date · currency tags · source URL — no content reproduction (copyright compliant). Bank-colour-coded row badges, series labels (FX Daily, FX Weekly, FX Talking), category chips (MACRO / TRADE / TECH / FLOW), and accordion drawer with direct link to the original note. Filterable by bank and by currency. Pipeline runs every 4 hours via GitHub Actions; zero LLM calls. Top 20 recent notes are also injected into the AI narrative context so bank views enrich the daily narrative without an additional model call.
- **Market signals** — 4–5 AI-generated signals updated 4× daily
- **Carry Trade Ranking** — 8 major currencies carry-to-vol ranking for all 28 pairs, sorted by nominal rate differential divided by 30-day realised volatility. Clicking any row opens the Real Rate Carry Analysis modal with three tabs: (1) Rates Breakdown — 8 currencies sorted by real rate (nominal CB rate minus inflation expectation), with OIS bias chip and data-age transparency; (2) Real Rate Matrix — 8×8 differential grid color-coded by carry sustainability; (3) Pair Detail — nominal carry, real carry, carry-to-vol on both bases, OIS probability for each leg, and carry sustainability assessment (sustainable / moderate / carry trap). Inflation expectations sourced live from FRED (T5YIE for USD, T5YIFR for EUR) and from `extended-data/*.json` batch for remaining currencies. Matches Bloomberg FXFR / FXFC layout convention.
- **Derivatives section** — Implied forwards (CIP, 1M/3M/6M/1Y), 25-delta Risk Reversal term structure (Saxo Bank), realized vol vs RR skew, ECB official reference exchange rates (daily fixing, 7 EUR pairs), and FX OTC notional volume by pair and product type (DTCC GTR CFTC Recast public dissemination, T+1).

---

## Architecture

Two repositories work together:

| Repo | Role |
|------|------|
| `globalinvesting-engine` (private) | Python scripts and GitHub Actions that fetch, process, and write JSON data files to this repo on schedule |
| `globalinvesting.github.io` (this repo) | Static HTML/CSS/JS terminal + JSON data files served via GitHub Pages |

The frontend reads all data via `fetch()` from JSON files committed to this repo. No API keys are exposed in the client. TradingView widgets provide live charting, heatmap, economic calendar, and economic map without requiring any API key.

---

## Currencies covered

USD · EUR · GBP · JPY · AUD · CAD · CHF · NZD — the eight major currencies, covering the substantial majority of global daily FX turnover.

---

## Data directories

| Directory | Contents | Updated by |
|-----------|----------|------------|
| `ai-analysis/` | `index.json` — AI regime label + narrative; `signals.json` — market signals | Engine — 4× daily |
| `calendar-data/` | Economic calendar events — `calendar.json` consumed by `update_extended_data.py` for PMI/CPI enrichment; `ff_calendar.json` (ForexFactory, 8 major currencies high-impact, this week + next) consumed by `generate_narrative_signals.py` → `build_calendar_block()` to ground session context notes in real catalysts | Engine — daily |
| `cot-data/` | CFTC COT positioning (Leveraged Funds + Asset Manager + Dealer) + 26-week `history[]` rolling window | Engine — weekly (Saturday) |
| `economic-data/` | Macro indicators | Engine — daily |
| `extended-data/` | IV, carry, cross-asset fallback, inflation expectations | Engine — daily / Site — weekly (inflation expectations) |
| `fx-data/` | `frankfurter.json` (ECB rates cache) + `fx-liquidity.json` | Engine — daily |
| `intraday-data/` | `quotes.json` — intraday source for non-FX instruments (VIX, SPX, Gold, WTI, etc.) and HV30/H-L/pct1w fields for FX pairs; `iv_history.json` — rolling 52-week IV snapshots | Engine — every 5 min (quotes); weekly append (iv_history) |
| `meetings-data/` | CB meeting schedules | Engine — weekly |
| `news-data/` | News feed headlines | Engine — hourly |
| `research-data/` | Bank & institutional FX research notes — `bank-research.json` (ING, Saxo, MUFG, DailyFX, BIS) | Engine — every 4h |
| `rates/` | FX rates cache | Engine — daily |
| `rr-data/` | `rr.json` — 25-delta Risk Reversal quotes (4 major pairs, 1M tenor) from Saxo Bank | Engine — Mon–Fri daily |
| `sentiment-data/` | `myfxbook.json` — retail positioning | Engine — hourly |
| `dtcc-data/` | `dtcc_fx.json` — FX OTC notional volume by pair and product type (DTCC GTR CFTC Recast) | Site — Mon–Fri 14:00 UTC |

### Site repo workflows

| Workflow | Schedule (UTC) | Output |
|---|---|---|
| `update-bond-yields.yml` | 23:00 daily | `extended-data/{USD,EUR,GBP,JPY}.json` — bond10y, bond2y, bond5y, vix (FRED + ECB + BOE) |
| `update-inflation-expectations.yml` | Monday 06:00 | `extended-data/{USD,EUR,GBP,JPY,AUD,CAD,CHF,NZD}.json` — inflationExpectations (FRED + World Bank) |
| `update-ohlc.yml` | Every :30 Mon–Fri (H1/H4); 01:30 Tue–Sat (D1 finalization); 23:30 Sat–Sun (crypto) | `ohlc-data/`, `ohlc-data/h1/`, `ohlc-data/h4/` — OHLC bars for all 38 chart symbols |
| `fetch-bank-research.yml` | Every 4h at :15 UTC (Mon–Sun) | `research-data/bank-research.json` — institutional FX research metadata (ING, Saxo, MUFG, DailyFX, BIS); injects `research_context` block into `ai-analysis/context_snapshot.json` |

---

## Pages

| Page | Description |
|------|-------------|
| `index.html` | Main FX Terminal dashboard |
| `about.html` | Product overview |
| `contact.html` | Contact information |
| `terms.html` | Terms of use |
| `privacy.html` | Privacy policy |
| `guide-dashboard.html` | How to use the terminal |
| `guide-cross-asset-risk.html` | Cross-asset risk monitor guide |
| `guide-rates-yield-curve.html` | Rates & yield curve guide |
| `guide-market-sentiment.html` | Market sentiment guide |
| `guide-fx-liquidity.html` | FX liquidity & sessions guide |
| `guide-cot.html` | COT / CFTC positioning guide |

---

## Disclaimer

Content on this terminal is informational and educational only. It does not constitute financial advice, an investment recommendation, or an offer to buy or sell any financial instrument. FX trading involves significant risk.

© 2026 Santiago Plá Casuriaga · Global Investing. All rights reserved.
