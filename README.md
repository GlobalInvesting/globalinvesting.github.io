# Global Investing — Forex Fundamental Analysis Dashboard

A quantitative fundamental analysis platform for the 8 major G8 currencies (USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD). Built with real macroeconomic data, institutional COT positioning, and a live economic calendar.

**[globalinvesting.github.io](https://globalinvesting.github.io/)** · [Versión en español](https://globalinvesting.github.io/) · [English version](https://globalinvesting.github.io/en.html)

![Status](https://img.shields.io/badge/Status-Live-success) ![Model](https://img.shields.io/badge/Model-v6.5-informational) ![Indicators](https://img.shields.io/badge/Indicators-21-blue) ![License](https://img.shields.io/badge/License-Proprietary-red)

---

## What it does

- Calculates a **fundamental strength score (0–100)** per currency using 21 weighted macroeconomic indicators
- Displays an **interactive heat map** of all indicators across all 8 currencies in a single view
- Generates **LONG/SHORT pair signals** with quantified confidence levels
- Includes **carry trade analysis** with live rate differential rankings and global risk regime
- Integrates **institutional COT positioning** (CFTC Leveraged Funds) with contrarian logic
- Provides a **live economic calendar** with actuals vs forecasts and automatic timezone conversion
- Generates **AI-written fundamental narratives** per currency (updated daily)
- Publishes **RSS feeds** for news, AI analysis, and G8 strength scores

---

## Scoring model — v6.5

The model distributes 100 points across 6 thematic tiers:

| Tier | Theme | Weight | Key indicators |
|------|-------|--------|----------------|
| 1 | Monetary Policy | 29% | Interest rate, rate momentum, inflation, CB outlook |
| 2 | External Balance | 19% | Current account, trade balance, public debt, terms of trade |
| 3 | Growth & Employment | 16% | GDP growth, unemployment, industrial production |
| 4 | Market Sentiment | 22% | COT positioning, manufacturing PMI, services PMI, FX performance, ESI |
| 5 | Bond & Confidence | 11% | 10Y bond yield, consumer confidence, business confidence |
| 6 | Consumption | 3% | Retail sales |

**Score interpretation:** < 45 = Bearish · 45–65 = Neutral · > 65 = Bullish (top ~25% of G8 universe)

**Minimum spread for a signal:** 12 points differential between the long and short currency.

---

## Data sources

All data comes from official APIs — no scraping of financial data providers.

| Data type | Frequency | Source |
|-----------|-----------|--------|
| Exchange rates | Every 60s | 6 forex API providers (cascade fallback) |
| Economic indicators | Daily 06:00 UTC | FRED (St. Louis Fed), OECD, World Bank, IMF |
| Extended data (bonds, confidence) | Daily 06:30 UTC | FRED, World Bank |
| Interest rates | Daily 08:00 UTC | global-rates.com + FRED |
| FX Performance 1M | Daily 07:00 UTC | Frankfurter API (ECB) |
| COT positioning | Weekly (Friday) | CFTC Official — financial_lf.htm |
| Economic calendar | 3× daily | Investing.com + FXStreet |
| AI analysis | Daily | Groq AI (llama-3.3-70b-versatile) |

---

## Public RSS feeds

The dashboard publishes three RSS feeds updated automatically:

| Feed | URL | Frequency |
|------|-----|-----------|
| Forex news | [`/feed.xml`](https://globalinvesting.github.io/feed.xml) | 3× daily |
| AI analysis per currency | [`/feed-analysis.xml`](https://globalinvesting.github.io/feed-analysis.xml) | Daily |
| G8 strength scores | [`/feed-scores.xml`](https://globalinvesting.github.io/feed-scores.xml) | Daily |

---

## Architecture

The platform is split into two repositories:

- **[globalinvesting.github.io](https://github.com/GlobalInvesting/globalinvesting.github.io)** (public) — Static site served via GitHub Pages. All JSON data files, HTML pages, and RSS feeds live here.
- **globalinvesting-engine** (private) — 21 GitHub Actions workflows that fetch, process, and publish data to the public repo on automated schedules.

### Automated pipeline (daily)

```
06:00 UTC  update-economic-data     → economic-data/{currency}.json   (FRED + OECD + IMF)
07:00 UTC  update-extended-data     → extended-data/{currency}.json   (bonds, confidence)
07:30 UTC  update-fx-performance    → fx-performance/{currency}.json  (Frankfurter/ECB)
08:00 UTC  update-rates             → rates/{currency}.json           (with validation)
08:00 UTC  generate-ai-analysis     → ai-analysis/{currency}.json     (Groq AI)
08:45 UTC  update-pmi-from-calendar → economic-data/ (PMI + CPI patch)
09:00 UTC  calculate-scores         → strength-scores/latest.json     (model v6.5)
09:30 UTC  generate-rss             → feed.xml, feed-analysis.xml, feed-scores.xml
11:00 UTC  monitor-data-health      → data quality check + email alert on failures
```

### Weekly (Mondays)

```
09:30 UTC  save-weekly-scores  → scores-history/all.json (backtest data)
10:00 UTC  run-backtest        → backtest-results/latest.json
10:00 UTC  workflow-meetings   → meetings-data/meetings.json (central bank dates)
```

---

## Pages

| URL | Description |
|-----|-------------|
| [`/`](https://globalinvesting.github.io/) | Main dashboard — strength scores, heat map, signals, calendar |
| [`/news.html`](https://globalinvesting.github.io/news.html) | Forex news feed with AI summaries |
| [`/carry-trade.html`](https://globalinvesting.github.io/carry-trade.html) | Carry trade rate differentials |
| [`/guia-score-fortaleza.html`](https://globalinvesting.github.io/guia-score-fortaleza.html) | Guide: How to read the strength score |
| [`/guia-carry-trade.html`](https://globalinvesting.github.io/guia-carry-trade.html) | Guide: How carry trade works |
| [`/guia-cot.html`](https://globalinvesting.github.io/guia-cot.html) | Guide: COT report interpretation |
| [`/guia-bancos-centrales.html`](https://globalinvesting.github.io/guia-bancos-centrales.html) | Guide: The 8 G8 central banks |
| [`/data.html`](https://globalinvesting.github.io/data.html) | Public RSS feeds and data resources |

---

## Backtest

The model runs a weekly retrospective backtest using `scores-history/all.json` (weekly snapshots since January 2024). Results are published to [`backtest-results/latest.json`](https://globalinvesting.github.io/backtest-results/latest.json).

Evaluation windows: 1W, 2W, 4W, 6W. Minimum spread threshold: 12 points.

---

## Author

Built and maintained by Santiago Plá Casuriaga.  
Contact: [globalinvestingmarkets@gmail.com](mailto:globalinvestingmarkets@gmail.com)

---

*Content is informational and educational. Does not constitute financial advice. Forex trading involves significant risk.*
