# Global Investing — FX Terminal

A professional-grade foreign exchange monitoring platform for serious market participants. Consolidates real-time price data, central bank policy rates, institutional positioning, cross-asset flows, and AI-assisted market narrative into a single unified dashboard.

**[globalinvesting.github.io](https://globalinvesting.github.io/)**

![Status](https://img.shields.io/badge/Status-Live-success) ![License](https://img.shields.io/badge/License-Proprietary-red)

---

## What it covers

- **Real-time price feeds** — Major FX pairs, commodities (XAU, WTI), crypto (BTC), DXY, and US 10Y yield
- **Currency strength heatmap** — 8×8 grid showing G8 currency performance across all pairs
- **AI market narrative** — 2–3 sentence regime summary updated 8× daily alongside market signals
- **CFTC COT positioning** — Leveraged Funds net positioning (Disaggregated TFF, Options+Futures Combined) from CFTC.gov, updated weekly
- **US Treasury yield curve** — 3M, 2Y, 5Y, 10Y, 30Y, updated daily
- **Cross-asset risk monitor** — SPX, Gold, WTI, BTC, DXY, Nikkei, Stoxx correlations with stress scoring
- **Central bank policy rates** — All G8 CBs with rate cycle direction
- **FX liquidity & sessions** — 24-hour liquidity profile with live session indicator
- **Positioning Bias** — ATM implied volatility from CBOE-listed FX ETF options (FXE, FXB, FXY, FXA) combined with COT Leveraged Funds directional bias
- **Economic calendar** — TradingView embed with real-time event actuals
- **News feed** — FXStreet, ForexLive, Investing.com and others, filtered by FX relevance, updated hourly
- **Market signals** — 5–7 AI-generated signals updated 8× daily

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
| `ai-analysis/signals.json` | AI narrative and market signals | Engine — 8× daily |
| `calendar-data/` | Economic calendar events | Engine — daily |
| `cot-data/` | CFTC COT positioning (Leveraged Funds + Asset Manager + Dealer) | Engine — weekly (Saturday) |
| `economic-data/` | Macro indicators | Engine — daily |
| `extended-data/` | IV, carry, cross-asset fallback | Engine — daily |
| `fx-performance/` | FX pair performance data | Engine — daily |
| `intraday-data/` | `quotes.json` — primary intraday source for cross-asset panels | Engine — every 15 min |
| `meetings-data/` | CB meeting schedules | Engine — weekly |
| `news-data/` | News feed headlines | Engine — hourly |
| `rates/` | FX rates cache | Engine — daily |

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
