# Global Investing — FX Terminal

A professional-grade foreign exchange monitoring platform for serious market participants. Consolidates real-time price data, central bank policy rates, institutional positioning, cross-asset flows, and AI-assisted market narrative into a single unified dashboard.

**[globalinvesting.github.io](https://globalinvesting.github.io/)**

![Status](https://img.shields.io/badge/Status-Live-success) ![License](https://img.shields.io/badge/License-Proprietary-red)

---

## What it covers

- **Real-time price feeds** — Major FX pairs, commodities (XAU, WTI), crypto (BTC), DXY, and US 10Y yield
- **Currency strength heatmap** — 8×8 grid showing G8 currency performance across all pairs
- **AI market narrative** — 2–3 sentence regime summary updated 12× daily alongside market signals
- **CFTC COT positioning** — Leveraged Funds net positioning (Disaggregated TFF, Options+Futures Combined) from CFTC.gov, updated weekly
- **US Treasury yield curve** — 3M, 2Y, 5Y, 10Y, 30Y, updated daily
- **Cross-asset risk monitor** — SPX, Gold, WTI, BTC, DXY, Nikkei, Stoxx correlations with stress scoring
- **Central bank policy rates** — All G8 CBs with rate cycle direction
- **CB Rate Expectations** — OIS-derived forward consensus direction (Cut/Hold/Hike) and 30-day CIP forward rate for each G8 central bank at their next meeting; sourced from CME FedWatch, ECB SDW ESTER, BoE SONIA, BoJ TONA, ASX Rate Indicator, BoC CORRA, SNB SARON, and RBNZ OCR overnight
- **FX liquidity & sessions** — 24-hour liquidity profile with live session indicator
- **Positioning Bias** — ATM implied volatility from CBOE-listed FX ETF options (FXE, FXB, FXY, FXA) combined with COT Leveraged Funds directional bias and 25-delta Risk Reversals from Saxo Bank (1M tenor, indicative mid-market). When ≥4 weeks of IV history are available, an IV Rank column (0–100 scale, where 100 = historically expensive vol) replaces the COT bias column.
- **Pair detail panel** — Linked right-panel showing price, 1W change, HV30, ATM IV, IV−HV, LF net, AM net, carry differential, and LF/AM alignment badge for the currently selected chart pair (Eikon-style linked panel). Hover tooltips on every metric cell explain the data source and interpretation.
- **ETF Options IV** — 8-row implied volatility panel: VIX, VIX9D, VVIX, MOVE, GLD IV, TLT IV, EEM IV, EFA IV. Colour-coded bar (red = elevated, amber = mid, green = low). Sources: CBOE indices and yfinance ETF options. Refreshes every 10 minutes.
- **Configurable price alerts** — Threshold alerts for VIX, EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CHF, Gold, US 10Y, and MOVE. Direction (`>` / `<`) and numeric threshold configurable per alert. Fires browser Notifications API on trigger. Persisted in localStorage across sessions. Checks every 5 minutes.
- **Panel data export** — CSV and JSON export for FX pairs, COT positioning, yield curve, and carry data; timestamped filenames, downloaded client-side from in-memory caches
- **Keyboard navigation** — `G`/`C`/`R`/`X`/`M`/`Y`/`K` jump to panels; `↑`/`↓` navigate FX table rows and load chart; `?` opens shortcut legend
- **Per-panel timestamps** — Every data panel displays its source and last update time in the user's local timezone. Panels with AI-generated signals also show when the data was loaded from the engine.
- **AI signal evidence traceability** — Each AI-generated market signal carries an `evidence[]` field listing the exact data values that motivated it (e.g. "VIX: 23.9", "Fed rate: 4.50%"). Evidence chips are hidden by default and revealed by clicking the signal row.
- **Economic calendar** — TradingView embed with real-time event actuals
- **News feed** — FXStreet, ForexLive, Investing.com and others, filtered by FX relevance, updated hourly
- **Market signals** — 4–5 AI-generated signals updated 12× daily

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

USD · EUR · GBP · JPY · AUD · CAD · CHF · NZD — the eight G8 major currencies, covering the substantial majority of global daily FX turnover.

---

## Data directories

| Directory | Contents | Updated by |
|-----------|----------|------------|
| `ai-analysis/` | `index.json` — AI regime label + narrative; `signals.json` — market signals | Engine — 12× daily |
| `calendar-data/` | Economic calendar events — written by engine, consumed by `update_extended_data.py` for PMI/CPI enrichment (not consumed directly by frontend) | Engine — daily |
| `cot-data/` | CFTC COT positioning (Leveraged Funds + Asset Manager + Dealer) + 26-week `history[]` rolling window | Engine — weekly (Saturday) |
| `economic-data/` | Macro indicators | Engine — daily |
| `extended-data/` | IV, carry, cross-asset fallback | Engine — daily |
| `fx-data/` | `frankfurter.json` (ECB rates cache) + `fx-liquidity.json` | Engine — daily |
| `intraday-data/` | `quotes.json` — primary intraday source for cross-asset panels; `iv_history.json` — rolling 52-week IV snapshots per FX pair for IV Rank calculation | Engine — every 5 min (quotes); weekly append (iv_history) |
| `meetings-data/` | CB meeting schedules | Engine — weekly |
| `news-data/` | News feed headlines | Engine — hourly |
| `rates/` | FX rates cache | Engine — daily |
| `rr-data/` | `rr.json` — 25-delta Risk Reversal quotes (4 G8 pairs, 1M tenor) from Saxo Bank | Engine — Mon–Fri daily |
| `sentiment-data/` | `myfxbook.json` — retail positioning | Engine — hourly |

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
