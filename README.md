# Global Investing FX Terminal

Professional-grade foreign exchange intelligence for active traders and macro analysts. Consolidates live FX prices, central bank policy data, institutional positioning, derivatives analytics, and AI-assisted market narrative into a single unified platform — available as a native MetaTrader 5 overlay and a companion web terminal.

**[globalinvesting.github.io](https://globalinvesting.github.io/)** &nbsp;·&nbsp; **[Get Access on MQL5](https://www.mql5.com/en/market/product/180326)**

![Status](https://img.shields.io/badge/Status-Live-success) ![License](https://img.shields.io/badge/License-Proprietary-red)

---

## Platform Access

Access to the Global Investing FX Terminal is granted through one of two independent paths: the **MT5 Expert Advisor**, available for rent on the MQL5 Marketplace, or a verified live trading account opened through a partner broker. Either path unlocks full web terminal access on its own — no need to combine them. The EA additionally runs the native MT5 chart overlay; the partner-broker path does not require MT5 at all.

| Product | Platform | Link |
|---|---|---|
| GI FX Terminal EA | MetaTrader 5 (overlay) + Web Terminal | [MQL5 Market →](https://www.mql5.com/en/market/product/180326) |
| Institutional Risk Manager | MetaTrader 5 | [MQL5 Market →](https://www.mql5.com/en/market/product/180324) |
| CSI Currency Strength | MetaTrader 5 | [MQL5 Market →](https://www.mql5.com/en/market/product/180317) |
| Carry Trade Monitor | MetaTrader 5 | [MQL5 Market →](https://www.mql5.com/en/market/product/180322) |
| Monte Carlo Equity Projection | MetaTrader 5 | [MQL5 Market →](https://www.mql5.com/en/market/product/186259) |

The EA runs natively inside MetaTrader 5 as a zero-flicker canvas overlay — live broker swap rates, MT5 push alerts, and the full macro intelligence suite without leaving your trading platform. The companion web terminal covers the complete analytical suite from any browser. See [Pricing & Access](https://globalinvesting.github.io/access.html) for details on both paths.

---

## Coverage

**Universe:** USD · EUR · GBP · JPY · AUD · CAD · CHF · NZD · NOK · SEK — the ten G10 currencies, covering the substantial majority of global daily FX turnover.

### Real-Time Data
- Tick-by-tick FX prices across major, minor, and G10 cross pairs
- Currency strength heatmap with per-currency pair breakdown, live carry, COT bias, realized vol, and correlations
- FX liquidity profile with live session indicator (Sydney · Tokyo · London · New York)
- Configurable price alerts with browser notifications

### Central Bank Intelligence
- Policy rates for all 10 G10 central banks with rate cycle direction
- OIS-derived forward rate expectations (Cut / Hold / Hike consensus) at each CB's next meeting
- CIP-adjusted 30-day forward rate per currency pair

### Institutional Positioning
- CFTC Commitments of Traders — Leveraged Funds net positioning (Traders in Financial Futures / TFF, Options+Futures Combined), updated weekly
- 52-week positioning history with momentum scoring
- Full Breakdown view (fullscreen): per-symbol Open Interest/Contract Value/Net Position table across FX, equity indices, and commodities; a pairwise Leveraged Funds Strength Index grid; and a Net Exposure % Rank chart showing where current positioning sits within its own trailing range

### Macro Analytics
- Economic Surprise Index — CESI-style normalized surprise for all 10 G10 currencies, 90-day rolling window, beat-rate scaled [−100, +100] per Citi convention
- US Treasury yield curve (3M · 2Y · 5Y · 10Y · 30Y)
- Cross-asset risk monitor (SPX · Gold · WTI · BTC · DXY · Nikkei · Stoxx) with stress scoring, plus US HY/IG credit spreads (ICE BofA OAS) and 20-day spread-direction tracking
- Carry Trade Ranking — G10 carry-to-vol for all 45 pairs with real rate breakdown and sustainability assessment
- Volatility Leaderboard — top 5 of all 28 G10 pairs ranked by current ATM implied volatility (direct CBOE/CME FX Volatility Index for USD majors, triangulated for crosses), surfacing where options-market-priced movement is highest right now
- Economic Matrix — latest GDP, headline CPI (YoY and MoM), core CPI, unemployment, industrial production, business confidence, retail sales, current account, trade balance, and PCE (US-specific) for all 10 G10 economies, each with its reference period and date, alongside 10-year sovereign yield and central bank policy rate, sourced from the live economic calendar
- FX Fair Value (v8.191.0) — rate-differential + risk-sentiment model for all 32 tracked pairs, with a 60-day rolling regression once enough daily history has accumulated; shows real daily inputs and honest accumulation progress rather than an estimate before that window is met
- Seasonality (v8.219.0, significance gate FDR-corrected v8.260.0) — monthly recurring return patterns per FX pair, up to 20-year lookback (real per-pair coverage varies, widened from 10y via `fetch_ohlc.py` v1.15): a day-of-year seasonal curve plus a table of recurring windows gated on Benjamini-Hochberg FDR-corrected significance (q<0.05, applied across each pair's ~78 correlated candidate windows — not a raw per-test p<0.05) with both p-value and q-value shown, and a descriptive month-by-month average return table
- Dollar Smile (v8.219.0) — Stephen Jen & Fatih Yilmaz's (2001) growth-differential framework: USD tends to strengthen in a genuine global risk-off/crisis regime and when US growth significantly outperforms the rest of the G10, and underperform in the calm middle; regime classification combines the real GDP growth differential (FRED, quarterly) with a VIX-based crisis override for the left tail

### Derivatives & Flow
- 25-delta Risk Reversal term structure (1M · 3M · 6M · 1Y) with skew vs realized vol
- Implied CIP forwards (1M · 3M · 6M · 1Y)
- ECB official reference exchange rates (daily fixing, 7 EUR pairs)
- FX OTC notional volume by pair and product type (DTCC GTR public dissemination, T+1)

### News & Research
- FX news feed — 50+ headlines refreshed every 10 minutes from major FX newswires, central bank communications, and institutional sources; filterable by currency and impact
- Bank research — Institutional FX notes from ING, Saxo Bank, MUFG, DailyFX, and BIS; metadata-only with direct links to source publications
- AI market narrative — 2–3 sentence regime summary updated 4× daily at major FX session transitions, with supporting signal evidence

---

## Currencies & Instruments

In addition to the 10 G10 FX currencies, the terminal monitors cross-asset context through: XAU/USD (Gold), WTI Crude, BTC/USD, DXY (US Dollar Index), SPX, Nikkei 225, Euro Stoxx 50, and US Treasury yields (3M–30Y).

---

## Guides

Full documentation is available at [globalinvesting.github.io](https://globalinvesting.github.io/):

- [Terminal Guide](https://globalinvesting.github.io/guide-dashboard.html) — How to use the dashboard
- [COT Positioning Guide](https://globalinvesting.github.io/guide-cot.html) — Reading CFTC data
- [Rates & Yield Curve Guide](https://globalinvesting.github.io/guide-rates-yield-curve.html)
- [Cross-Asset Risk Guide](https://globalinvesting.github.io/guide-cross-asset-risk.html)
- [Economic Surprises Guide](https://globalinvesting.github.io/guide-economic-surprises.html)
- [FX Liquidity Guide](https://globalinvesting.github.io/guide-fx-liquidity.html)
- [MT5 EA Guide](https://globalinvesting.github.io/guide-mt5-ea.html) — Setup and features
- [Currency Strength Index (CSI) Guide](https://globalinvesting.github.io/guide-csi-indicator.html) — Methodology and MT5 indicator setup
- [Monte Carlo Equity Projection Guide](https://globalinvesting.github.io/guide-monte-carlo-simulator.html) — Bootstrap simulation methodology and MT5 indicator setup

---

## Disclaimer

Content on this platform is informational and educational only. It does not constitute financial advice, an investment recommendation, or an offer to buy or sell any financial instrument. FX and CFD trading involves significant risk of loss and may not be suitable for all investors.

© 2026 Santiago Plá Casuriaga · Global Investing. All rights reserved.
