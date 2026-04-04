# Global Investing FX Terminal — Editorial & Design Guidelines

This document defines the non-negotiable editorial, design, and content standards for the FX Terminal and all satellite pages. Read this before making any changes to the frontend or backend scripts.

---

## Product identity

**What this is:** A professional-grade FX market monitoring terminal. The benchmark is institutional-quality tools (Bloomberg, Eikon, FXSSI Pro). Every decision — copy, design, features — must serve that standard.

**What this is not:** A personal blog, a casual dashboard, a free tool that looks amateur.

---

## Language & tone

- All pages are in **English only**. No Spanish content anywhere.
- Tone is **concise, precise, professional**. No marketing superlatives, no casual phrasing.
- No emojis anywhere — not in HTML, not in copy, not in headings, not in cards.
- No invented social media handles or accounts. Only link to accounts that actually exist.
- No phrases like "We're happy to hear from you", "Feel free to...", etc. Keep it businesslike.

---

## What NOT to expose publicly

### No pricing or "free" references
- Never mention that the terminal is "free to use", "zero cost", "at no cost", or similar.
- Never add a "COST" or "PRICE" column to data source tables.
- Pricing and access model may change. The About page describes *what* we offer, not *at what price*.

### No backend/architecture details
- Do not describe the internal architecture (engine repo, GitHub Actions, Python scripts, private repos, etc.).
- Do not mention specific model names beyond what is necessary (e.g. "Groq LLM" is acceptable; "llama-3.3-70b-versatile" is internal detail).
- Do not explain that the site has "zero marginal cost" or "no runtime cost" — these are internal business details.
- Do not describe GitHub Actions workflows, cron schedules, or how data pipelines work internally.
- The About page should describe the **product** (what users experience), not the **backend** (how it's built).

### No speculative claims
- Do not claim the terminal covers "85% of global FX turnover" or similar statistics without a verifiable, current citation.
- Replace with: "the substantial majority of global daily FX turnover".

---

## Data integrity — non-negotiable rules

### No artificial noise in displayed data
- **Never use `Math.random()` to vary displayed market data**, indicators, or chart values. Any value shown to the user must be deterministic and reproducible.
- This applies to: option skew estimates, sentiment fallbacks, liquidity chart baselines, regime indicators, spread estimations, or any other panel.
- If real data is unavailable, show the static reference value clearly labeled (e.g. "Historical avg · live feed unavailable"). Do not add noise to make it look "live".
- Liquidity chart baselines (`LIQ_BASE`) are historically-derived constants. Interpolate them smoothly; do not perturb with random offsets.

### No investment advice language
- **Never use language that implies a recommended action** (e.g. "reduce exposure", "go long", "add risk", "cut positions").
- Regime and signal subtexts must be **descriptive, not prescriptive**:
  - Correct: "High fear · VIX elevated" — describes market conditions
  - Correct: "Inverted curve · recession risk" — describes macro environment
  - Correct: "Elevated volatility" — describes the environment
  - Wrong: "Reduce exposure" — investment advice
  - Wrong: "Add risk here" — investment advice
- This applies to all panels: Regime, Alerts & Market Signals, AI narrative, Cross-Asset indicators.

### Estimated data must be labeled
- Any value that is estimated, derived, or not sourced from a live/real feed **must carry a visible label** indicating its nature.
- Required labels by panel:
  - Positioning Bias — COT bias column → `CFTC Disaggregated TFF · Leveraged Funds · Options+Futures Combined` (net positioning of the Leveraged Funds category, not real options prices)
  - Positioning Bias — ATM IV column → `ETF options IV · CBOE` when live data available; falls back to COT-derived proxy, labeled `est. via COT`
  - Market Sentiment fallback → `Historical avg · live feed unavailable`
  - Session Vol ±pip table → `5yr historical avg · fixed reference`
  - FX Liquidity baseline → `5yr historical avg · fixed reference`
  - EUR/USD IV fallback → `est. via VIX` (scaled from VIX when real options data unavailable)
- Do not remove or downplay these labels to make the data appear more authoritative than it is.

### Source labels must be accurate
- Panel subtitles must reflect the actual active data source, not a legacy or aspirational one.
- When yfinance is the active source, say "yfinance", not "TradingView live" or "stooq".
- When a yield comes from the repo's `extended-data/` (daily batch), do not label it "live".
- Acceptable source label patterns: `yfinance · ~15min delay`, `yfinance + stooq`, `COT/CFTC · weekly`, `CFTC · week ending YYYY-MM-DD`, `Historical avg · fixed reference`.

### Yield data — 2Y must come from FRED DGS2
- The US 2Y yield must be sourced from FRED series `DGS2` via `update_extended_data.py`.
- Do not use `^IRX` (13-week T-Bill) as a proxy for the 2Y. The two diverge by up to 50bp during policy transitions and the discrepancy is visible to rates traders.
- The `bond2y` field in `extended-data/USD.json` is the authoritative source consumed by the frontend.

---

## Regime badge — architecture rules

The terminal has **two** regime indicators that must behave differently:

| Element | ID | Source | Rule |
|---|---|---|---|
| Risk Monitor badge | `#risk-regime` | Always live (VIX stress score) | Updates on every `renderRiskData()` call |
| Narrative badge | `#narrative-regime` | AI JSON when fresh; live when stale | Never downgraded by live score alone unless `liveRank ≥ 2` (RISK-OFF) |

### Boot sequence — non-negotiable order
```
boot()
  await loadIntradayQuotes()   ← populates intraday cache
  await loadAIRegime()         ← sets _aiRegimeFresh, paints narrative badge BEFORE fetchRiskData
  fetchRiskData()              ← sees _aiRegimeFresh=true, does not overwrite narrative badge
  buildRichNarrative()         ← fills narrative text asynchronously
```

`loadAIRegime()` must be awaited before `fetchRiskData()`. Reversing this order causes the RISK-ON → CAUTION flip on first load (VIX > 25 always scores CAUTION; without `_aiRegimeFresh`, it overwrites the narrative badge).

### Override rules for the narrative badge
```js
const shouldOverride = isCurrentStale        // AI JSON > 4h old
  || !_aiRegimeFresh                         // AI JSON not loaded yet
  || (liveRank > currentRank && liveRank >= 2); // live says RISK-OFF
```
This is the only condition under which `renderRiskData()` may write to `#narrative-regime`.

---

## Script placement — public vs private repo

### Rule: if no competitive advantage, place in the public repo
New Python scripts that fetch or transform publicly available data (e.g. yfinance, CFTC COT, Frankfurter API, economic calendars) **should live in the public site repo** (`globalinvesting.github.io`) unless they contain API keys, proprietary logic, or competitive differentiators.

**Rationale:** The public repo has unlimited free GitHub Actions minutes. The private engine repo has a finite monthly allowance. Placing scripts without competitive value in the private repo wastes that budget unnecessarily.

| Script type | Where it belongs |
|---|---|
| Fetches yfinance / free public APIs, no API keys required | **Public repo** (`scripts/`) |
| Fetches data requiring paid/secret API keys (Groq, Twelve Data, Alpha Vantage) | **Private engine repo** |
| Generates AI narratives / signals | **Private engine repo** |
| Processes COT/CFTC data from CFTC.gov (public source, no auth) | **Public repo** — unless the workflow must write to a *different* repo, which requires a `PAT_TOKEN` and therefore belongs in the private engine repo |
| Aggregates or transforms data with proprietary weighting/scoring | **Private engine repo** |

### GitHub Actions — public repo guidelines
- Public repo workflows use `GITHUB_TOKEN` — no PAT needed when the workflow and its output data live in the same repo.
- Secrets like `TWELVE_DATA_API_KEY` are acceptable in the public repo when needed as fallbacks — GitHub Actions secrets are never exposed in logs or to the public.
- Schedule windows should be documented in the workflow file with rationale for the chosen hours.
- **COT workflow specifically:** runs at `30 0 * * 6` (Saturday 00:30 UTC). Rationale: CFTC publishes Fridays at ~15:30 ET (≈20:30 UTC). Saturday 00:30 UTC provides a 4-hour buffer, eliminating the risk of fetching before the report is live.

---

## Frontend data loading — performance rules

### Intraday JSON must be pre-loaded before all other fetches
- `boot()` must `await loadIntradayQuotes()` **before** launching `fetchRiskData()`, `fetchCrossAssetData()`, or any other function that consumes the intraday cache.
- This guarantees the JSON (same-origin, ~5ms) is in cache before any external API calls start, enabling instant first render for all panels.

### fetchRiskData and fetchCrossAssetData must run in parallel
- Never chain `fetchCrossAssetData()` inside `.then()` of `fetchRiskData()`.
- Both functions read the intraday JSON cache independently. Chaining forces the cross-asset panel to wait for Stooq/Yahoo (2–8 seconds) before rendering.
- Correct pattern — applies to both `boot()` and the refresh `setInterval`:
  ```js
  // Correct — parallel
  fetchRiskData();
  fetchCrossAssetData();
  fetchCommodityQuotes();

  // Wrong — chained, causes 2-8s delay on cross-asset panel
  fetchRiskData().then(() => { fetchCrossAssetData(); fetchCommodityQuotes(); });
  ```

### Intraday JSON is the primary source — external APIs are enrichment only
- The intraday `quotes.json` (updated every 15 min via GitHub Action) is the **primary and fastest** source for all symbols it covers: VIX, SPX, Gold, WTI, US10Y, Nikkei, Stoxx, DXY, US2Y, US3M, US5Y, US30Y, MOVE, BTC.
- Stooq, Yahoo Finance (via proxies), and CoinGecko are **secondary enrichers** — they may update values silently after the first render, but must never block the initial paint.
- Any new panel that needs a symbol already in `quotes.json` must read it from the intraday cache in `Step 1.5` of the relevant fetch function, not wait for Stooq.

---

## Automated tests — non-negotiable

`assets/dashboard.test.js` is the canonical test suite. Run with `node assets/dashboard.test.js`.

### Rules
- **All 81 tests must pass before any deployment.** A failing test is a deployment blocker.
- When adding a new calculation to `dashboard.js`, add corresponding tests to `dashboard.test.js` before merging.
- Tests must be deterministic — no `Math.random()`, no `Date.now()` dependency without a fixed input.
- The HV30 implementation in `computeHV30()` (JS) must mirror `compute_hv30()` (Python engine) exactly: minimum 22 closes, last 31 prices window, sample variance (`n-1`), annualise with `√252 × 100`.

### What is tested (current coverage)
| Module | Tests | Notes |
|---|---|---|
| `fmt`, `clsDir`, `pctStr` | 15 | All null/NaN/boundary cases |
| `isOpen` | 12 | Including midnight wrap-around (Sydney) |
| `computeRate` | 7 | Direct, inverted, cross, null legs |
| Stress scoring | 14 | All VIX thresholds + gold/SPX/MOVE/curve boundary values |
| `localizeSignalTime` | 6 | Null, passthrough, bad format, midnight, 23:59 |
| Business dates | 7 | Mon–Fri, Sat, Sun, Mon→Fri prev |
| Yield spreads | 4 | Normal, inverted, flat, US-DE |
| `computeHV30` | 9 | Min 22 closes, annualisation, alternating-return known result |
| Pearson correlation | 7 | ±1, orthogonal, bounds, lengths, EUR/USD-DXY |

---

## Accessibility — WCAG 2.1 AA standards

The terminal targets WCAG 2.1 Level AA. The following are non-negotiable and must be preserved in all future changes.

### Landmarks
- `<header id="topbar" role="banner">` — topbar is the page banner
- `<footer id="statusbar" role="contentinfo">` — status bar is the page footer
- `<main id="main" role="main" aria-label="Dashboard">` — main content area
- `<nav class="top-nav" aria-label="Dashboard sections">` — primary navigation
- `<aside id="sidebar" aria-label="FX Pairs sidebar">`
- `<aside id="rightpanel" aria-label="Market data panels">`

### Skip link (WCAG 2.4.1)
- `<a href="#main" class="skip-link">Skip to main content</a>` must be the **first focusable element** in `<body>`.
- The `.skip-link` CSS class positions it off-screen until `:focus`, at which point it appears at `top: 8px`.
- Do not remove or reorder this element.

### Tables (WCAG 1.3.1)
- Every `<table>` must have an `aria-label` describing its content.
- Every `<th>` must have `scope="col"` (or `scope="row"` for row headers).
- Do not add new tables without both attributes.

### Buttons (WCAG 4.1.2)
- Every icon-only button (scroll arrows, menu toggle) must have `aria-label`.
- Chart tab buttons must have `role="tab"` and `aria-selected="true/false"`.
- The tab container must have `role="tablist"` and `aria-label`.
- The site menu button must have `aria-expanded` updated on hover/focus via the `initA11y()` function in `dashboard.js`.

### Live regions (WCAG 4.1.3)
| Element | `aria-live` | `role` | Purpose |
|---|---|---|---|
| `#narrative` | `polite` | `region` | Market narrative text |
| `#risk-regime` | `assertive` | — | Regime badge changes |
| `#alerts-container` | `polite` | `log` | New signals added |
| `#sr-announce` | `polite` | `status` | Generic SR announcements |

### Focus visibility (WCAG 2.4.7)
- The global CSS reset strips `outline`. The `:focus-visible` rule in `dashboard.css` must remain — it restores `2px solid var(--accent)` outlines on all interactive elements.
- Do not add `outline: none` or `outline: 0` to any interactive element.

### Navigation (WCAG 2.4.3, 2.4.6)
- `aria-current="location"` must be set on the active top-nav link and updated on click by the `initA11y()` function.

---

## Architecture — file structure

The frontend is split into four files. Do not collapse them back into a single `index.html`.

| File | Contents | Lines |
|---|---|---|
| `index.html` | HTML structure only — no inline JS, no inline CSS | ~620 |
| `assets/dashboard.css` | All styles including GDPR panel, skip link, sr-only | ~914 |
| `assets/dashboard.js` | All application logic + `initA11y()` module | ~2 913 |
| `assets/gdpr.js` | GDPR consent banner logic | 74 |
| `assets/dashboard.test.js` | Automated test suite (81 tests) | ~565 |

### CSP rules
- `script-src` must **not** contain `unsafe-inline`.
- Allowed external script origins: `s3.tradingview.com`, `widgets.tradingview-widget.com`, `pagead2.googlesyndication.com`, `cdn.jsdelivr.net`.
- TradingView widget `<script>` tags use the standard embed mechanism (src + JSON body) — this does not require `unsafe-inline`.

---

## Design standards

### Typography
- Base font size: **14px minimum** on all satellite pages (about, contact, guides, legal).
- Body text: `var(--text2)` — `#787b86`
- Headings: `var(--text)` — `#d1d4dc`
- Never go below 13px for body copy. 12px acceptable only for metadata, labels, table cells.

### No emojis
- No Unicode emoji characters in any HTML file. Period.
- Use SVG icons or plain text indicators instead.

### Diagrams and visualizations
- Use **SVG diagrams** for layout illustrations and technical visualizations. No ASCII art diagrams.
- The terminal layout diagram in `guide-dashboard.html` must use the SVG version.

### Color palette — always use CSS variables
```
--bg:        #131722   (main background)
--bg2:       #1e222d   (panel/card background)
--bg3:       #2a2e39   (hover, tertiary)
--border:    #2a2e39
--border2:   #363c4e
--text:      #d1d4dc   (primary text)
--text2:     #787b86   (secondary text)
--text3:     #6b7280   (muted/label text)
--up:        #26a69a   (green — positive)
--down:      #ef5350   (red — negative)
--blue:      #2962ff   (accent, links)
--orange:    #f6941c   (warnings, disclaimers)
--head-bg:   #1a1d29   (topbar, statusbar)
```

### Redundant widgets
- Do not include third-party widgets whose content is already covered by native panels.
- The TradingView ticker tape (`embed-widget-ticker-tape.js`) was removed — the quote bar already shows all FX pairs and cross-asset prices natively, with better styling control and no iframe overhead.
- Before adding any third-party widget, confirm it provides data or functionality not already present in the terminal.

---

## Site menu — canonical link list

Every page must use this exact site menu. No deviations:

```
Company
  About         → about.html
  Contact       → contact.html
  Terms of Use  → terms.html
  Privacy Policy→ privacy.html

Guides
  How to Use the Terminal      → guide-dashboard.html
  Cross-Asset Risk Monitor     → guide-cross-asset-risk.html
  Rates & Yield Curve          → guide-rates-yield-curve.html
  Market Sentiment             → guide-market-sentiment.html
  FX Liquidity & Sessions      → guide-fx-liquidity.html
  COT / CFTC Positioning       → guide-cot.html
```

Never link to: `advertise.html`, `guide-cb-rates.html`, `guia-*.html`, `carry-trade.html`, `news.html`, `en.html`, `data.html`, or any other deleted page.

---

## Page checklist — before publishing any satellite page

- [ ] `lang="en"` on `<html>` tag
- [ ] Base font size ≥ 14px on body
- [ ] No emojis anywhere in the file
- [ ] No "free", "at no cost", "zero cost" references
- [ ] No backend/architecture exposure
- [ ] Site menu uses the canonical link list above
- [ ] Current page highlighted in site menu with `class="current"`
- [ ] No links to deleted pages
- [ ] Topbar consistent with all other pages
- [ ] `robots.txt`, canonical URL, and OG tags present and correct
- [ ] Disclaimer present on About, Terms, Privacy
- [ ] No invented social media accounts or links

## README maintenance — non-negotiable

Both repos have a README that must stay in sync with the actual state of the project:

| File | Repo | What to keep current |
|---|---|---|
| `README.md` | Public site (`globalinvesting.github.io`) | Panel list, data directories table, pages table, currencies description |
| `README.md` | Engine (`globalinvesting-engine`) | Architecture diagram, scripts list, workflow schedule table, data sources table, secrets table |

### Rules
- **Any change to a workflow schedule** → update the schedule table in the engine README.
- **Any new or removed workflow** → update the workflow table and the architecture diagram in the engine README.
- **Any new or removed script** → update the scripts block in the engine README architecture diagram.
- **Any new or removed panel in the terminal** → update the "What it covers" list in the site README.
- **Any new or removed data directory** → update the data directories table in the site README and the engine README.
- **Any change to the COT data schema or source** → update the COT row in both READMEs.
- **No cost, pricing, or "free" references** in either README (applies to the data sources table — omit the Cost column entirely).
- **No internal model names** (e.g. `llama-3.3-70b-versatile`) in either README. Use "Groq LLM" or "LLM" only.
- **No "85% of global FX turnover"** — use "the substantial majority of global daily FX turnover".
- README updates must be included in the same CHANGELOG entry as the change that triggered them.

---

## Checklist — before adding or modifying any data panel

- [ ] Data source is accurately labeled in the panel subtitle
- [ ] If estimated or derived, a visible label indicates this (see Data integrity section)
- [ ] No `Math.random()` used anywhere in the rendering path
- [ ] No investment advice language in subtexts or signal copy
- [ ] If the symbol is in `quotes.json`, it is read from the intraday cache in Step 1.5 (not from Stooq/Yahoo alone)
- [ ] New Python scripts placed in the correct repo per the Script placement rules above

## Checklist — before any frontend deployment

- [ ] `node assets/dashboard.test.js` → 81 passed, 0 failed
- [ ] `assets/dashboard.js` does not end with `</script>` or any HTML tag (extract-only artifact from prior sessions)
- [ ] `assets/dashboard.css` does not contain `<style>` or `</style>` tags
- [ ] `index.html` references `assets/dashboard.css`, `assets/dashboard.js`, `assets/gdpr.js` (no inline CSS/JS blocks)
- [ ] Skip link `<a href="#main" class="skip-link">` is first child of `<body>`
- [ ] All tables have `aria-label` and `scope="col"` on headers
- [ ] `loadAIRegime()` is awaited before `fetchRiskData()` in `boot()`
- [ ] Both READMEs reviewed and updated if any panel, directory, workflow, or schedule changed
- [ ] CHANGELOG.md updated with new version entry
- [ ] GUIDELINES.md version footer bumped if any rule changed

---

## Sitemap — active pages only

```
https://globalinvesting.github.io/                          (index.html — terminal)
https://globalinvesting.github.io/about.html
https://globalinvesting.github.io/contact.html
https://globalinvesting.github.io/terms.html
https://globalinvesting.github.io/privacy.html
https://globalinvesting.github.io/guide-dashboard.html
https://globalinvesting.github.io/guide-cross-asset-risk.html
https://globalinvesting.github.io/guide-rates-yield-curve.html
https://globalinvesting.github.io/guide-market-sentiment.html
https://globalinvesting.github.io/guide-fx-liquidity.html
https://globalinvesting.github.io/guide-cot.html
```

---

## Data directories — do not delete

These directories are actively consumed by the terminal frontend:

| Directory | Used by |
|---|---|
| `ai-analysis/` | AI signals panel (keep `signals.json`) |
| `calendar-data/` | Macro Calendar right panel |
| `cot-data/` | COT Positioning section |
| `economic-data/` | Macro data panel |
| `economic-data-history/` | Historical charts (verify before deleting) |
| `extended-data/` | IV panel, carry trade sidebar, cross-asset fallback |
| `fx-performance/` | FX performance data |
| `intraday-data/` | `quotes.json` — primary intraday source for all cross-asset panels |
| `meetings-data/` | CB meeting calendar |
| `news-data/` | News feed and ticker |
| `rates/` | FX rates cache |

---

*Last updated: April 2026 — v7.5.0*
