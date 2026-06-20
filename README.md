# Global Investing — FX Terminal

A professional-grade foreign exchange monitoring platform for serious market participants. Consolidates live FX prices (Finnhub WebSocket, tick-by-tick), central bank policy rates, institutional positioning, cross-asset flows, and AI-assisted market narrative into a single unified dashboard.

**[globalinvesting.github.io](https://globalinvesting.github.io/)**

![Status](https://img.shields.io/badge/Status-Live-success) ![License](https://img.shields.io/badge/License-Proprietary-red)

---

## What it covers

- **Real-time price feeds** — Major FX pairs, commodities (XAU, WTI), crypto (BTC), DXY, and US 10Y yield
- **Currency strength heatmap** — 10×10 grid showing major currency performance across all pairs
- **AI market narrative** — 2–3 sentence regime summary updated 4× daily alongside market signals
- **CFTC COT positioning** — Leveraged Funds net positioning (Disaggregated TFF, Options+Futures Combined) from CFTC.gov, updated weekly
- **US Treasury yield curve** — 3M, 2Y, 5Y, 10Y, 30Y, updated daily
- **Cross-asset risk monitor** — SPX, Gold, WTI, BTC, DXY, Nikkei, Stoxx correlations with stress scoring
- **Central bank policy rates** — All 10 G10 currencies' CBs with rate cycle direction
- **CB Rate Expectations** — OIS-derived forward consensus direction (Cut/Hold/Hike) and 30-day CIP forward rate for each central bank at their next meeting; sourced from CME FedWatch, ECB SDW ESTER, BoE SONIA, BoJ TONA, ASX Rate Indicator, BoC CORRA, SNB SARON, RBNZ OCR overnight, and OECD overnight rate references for NOK/SEK
- **FX liquidity & sessions** — 24-hour liquidity profile with live session indicator
- **Positioning Bias** — ATM implied volatility from CBOE-listed FX ETF options (FXE, FXB, FXY, FXA) combined with COT Leveraged Funds directional bias and 25-delta Risk Reversals from Saxo Bank (1M tenor, indicative mid-market). When ≥4 weeks of IV history are available, an IV Rank column (0–100 scale, where 100 = historically expensive vol) replaces the COT bias column.
- **Pair detail panel** — Linked right-panel showing price, 1W change, HV30, ATM IV, IV−HV, LF net, AM net, carry differential, and LF/AM alignment badge for the currently selected chart pair (Eikon-style linked panel). Hover tooltips on every metric cell explain the data source and interpretation.
- **ETF Options IV** — 8-row implied volatility panel: VIX, VIX9D, VVIX, MOVE, GLD IV, TLT IV, EEM IV, EFA IV. Colour-coded bar (red = elevated, amber = mid, green = low). Sources: CBOE indices and yfinance ETF options. Refreshes every 10 minutes.
- **Configurable price alerts** — Threshold alerts for VIX, EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CHF, Gold, US 10Y, and MOVE. Direction (`>` / `<`) and numeric threshold configurable per alert. Fires browser Notifications API on trigger. Persisted in localStorage across sessions. Checks every 5 minutes.
- **Panel data export** — CSV and JSON export for FX pairs, COT positioning, yield curve, and carry data; timestamped filenames, downloaded client-side from in-memory caches
- **Keyboard navigation** — `G`/`C`/`R`/`X`/`M`/`Y`/`K`/`D`/`N`/`B` jump to panels; `↑`/`↓` navigate FX table rows and load chart; `?` opens shortcut legend
- **Per-panel timestamps** — Every data panel displays its source and last update time in the user's local timezone. Panels with AI-generated signals also show when that data was last refreshed.
- **AI signal evidence traceability** — Each AI-generated market signal carries an `evidence[]` field listing the exact data values that motivated it (e.g. "VIX: 23.9", "Fed rate: 4.50%"). Evidence chips are hidden by default and revealed by clicking the signal row.
- **Economic calendar** — TradingView embed with real-time event actuals
- **Economic Surprises** — CESI-style normalised surprise index for all 8 major currencies, computed from ForexFactory calendar actual-vs-consensus over a 90-day rolling window. Clicking any currency row opens a full-screen detail modal with: (1) LightweightCharts v5 area series showing the 30-day rolling index history stepped weekly; (2) metrics bar (index, beats, misses, N, beat rate); (3) individual events table with beat / miss / in-line badges for the full 90-day window. Methodology mirrors Citi CESI convention — beat rate scaled to [−100, +100].
- **News Feed** — Dedicated tab (shortcut: N) with 52 FX-relevant headlines from FXStreet, ForexLive, Reuters FX, ECB, BoE, BoJ, RBA, RBNZ, BoC, SNB, Federal Reserve, Finnhub, NewsData.io, Marc to Market and others. Single-row accordion layout with impact dot, currency tag, relative age indicator, and full-headline tooltip. Filterable by currency (8 major currencies) and impact level (High / Med / All). Headlines refresh hourly; the terminal re-checks every 2 minutes via HTTP ETag.
- **Bank Research** — Dedicated tab (shortcut: B) showing institutional FX research notes from ING Think, Saxo Bank (SaxoStrats), MUFG Research, DailyFX (IG Group), and BIS central banker speeches. Metadata-only: title · bank · date · currency tags · source URL — no content reproduction (copyright compliant). Bank-colour-coded row badges, series labels (FX Daily, FX Weekly, FX Talking), category chips (MACRO / TRADE / TECH / FLOW), and accordion drawer with direct link to the original note. Filterable by bank and by currency. Refreshes every 4 hours. Top 20 recent notes also enrich the daily AI narrative with institutional context.
- **Market signals** — 4–5 AI-generated signals updated 4× daily
- **Carry Trade Ranking** — G10 currencies carry-to-vol ranking for all 45 pairs, sorted by nominal rate differential divided by 30-day realised volatility. Clicking any row opens the Real Rate Carry Analysis modal with three tabs: (1) Rates Breakdown — 10 currencies sorted by real rate (nominal CB rate minus inflation expectation), with OIS bias chip and data-age transparency; (2) Real Rate Matrix — 10×10 differential grid color-coded by carry sustainability; (3) Pair Detail — nominal carry, real carry, carry-to-vol on both bases, OIS probability for each leg, and carry sustainability assessment (sustainable / moderate / carry trap). Inflation expectations sourced live from FRED (T5YIE for USD, T5YIFR for EUR) and from `extended-data/*.json` batch for remaining currencies. Matches Bloomberg FXFR / FXFC layout convention.
- **Derivatives section** — Implied forwards (CIP, 1M/3M/6M/1Y), 25-delta Risk Reversal term structure (Saxo Bank), realized vol vs RR skew, ECB official reference exchange rates (daily fixing, 7 EUR pairs), and FX OTC notional volume by pair and product type (DTCC GTR CFTC Recast public dissemination, T+1).

---

## Architecture

The terminal is a static frontend. All data is delivered as JSON files fetched directly by the browser — no backend server, no API keys exposed to the client. TradingView widgets provide live charting, heatmap, economic calendar, and economic map without requiring any API key.

Data is refreshed on a regular cadence behind the scenes and committed to this repo, so every panel reflects current market conditions without the user needing to do anything.

---

## Currencies covered

USD · EUR · GBP · JPY · AUD · CAD · CHF · NZD · NOK · SEK — the ten G10 currencies, covering the substantial majority of global daily FX turnover.

---

## Data directories

| Directory | Contents |
|-----------|----------|
| `ai-analysis/` | AI regime label, market narrative, and market signals |
| `calendar-data/` | Economic calendar events used to ground session context and macro enrichment |
| `cot-data/` | CFTC COT positioning (Leveraged Funds + Asset Manager + Dealer), with a rolling weekly history window |
| `economic-data/` | Macro indicators |
| `extended-data/` | Implied volatility, carry trade inputs, cross-asset fallback values, inflation expectations |
| `fx-data/` | ECB reference rates cache and FX liquidity reference data |
| `intraday-data/` | Intraday quotes for non-FX instruments (VIX, SPX, Gold, WTI, etc.), HV30/H-L/weekly-change fields for FX pairs, and rolling IV history |
| `meetings-data/` | Central bank meeting schedules and OIS-implied policy bias |
| `news-data/` | News feed headlines |
| `research-data/` | Bank and institutional FX research note metadata (title, bank, date, currency tags, source link — no content reproduction) |
| `rates/` | Central bank policy rates |
| `rr-data/` | 25-delta Risk Reversal quotes |
| `sentiment-data/` | Retail positioning sentiment |
| `dtcc-data/` | FX OTC notional volume by pair and product type |

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
