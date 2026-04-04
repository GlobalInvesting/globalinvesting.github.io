# Changelog — globalinvesting-engine

---

## v7.10.0 (2026-04-04) — Alerts popover fix, carry rank blank space, tooltip column anchoring

### Frontend — assets/dashboard.css
- **Alerts popover clipped by statusbar overflow:** `#statusbar` had `overflow: hidden` which silently clipped the upward-opening alerts popover, making the button appear non-functional. Changed to `overflow: visible`.
- **Pair-detail tooltip anchoring:** Replaced `:first-child` / `:last-child` edge rules with `nth-child(odd)` (left-anchor) and `nth-child(even)` (right-anchor) to correctly fix all 8 cells in the 2-column grid — previously only the first and last cells were anchored, leaving cells 2–7 overflowing the rightpanel.

### Frontend — index.html
- **Alerts button emoji removed:** Replaced `🔔` with plain text `alerts` matching the `kb-hint-btn` style of the adjacent `? shortcuts` button — consistent with the no-emoji editorial rule.
- **Carry rank blank space:** Removed the static `Loading…` placeholder div from `#carry-rank-rows` that created a visible gap before `fetchCarryRanking()` populated the container.

---

## v7.9.0 (2026-04-04) — ETF Options IV panel, configurable alerts, pair-detail tooltips

### Frontend — assets/dashboard.js
- **`fetchEtfIV()`** replaces `fetchCarryRanking()`. Reads `intraday-data/quotes.json` (`etf_iv` block) with fallback to `STOOQ_RT_CACHE`. Renders 8 rows: VIX, VIX9D, VVIX, MOVE, GLD IV, TLT IV, EEM IV, EFA IV. Each row shows label, proportional colour bar (red > 30, amber > 18, green ≤ 18), numeric value, and 1-day % change. Click opens the ticker in TradingView. Interval reduced to 10 min (was 30 min for carry ranking).
- **`ETF_IV_MANIFEST`** constant: array of `{ key, label, desc, tvSym }` objects — single source of truth for the IV panel. Keys match `etf_iv` block in quotes.json and `STOOQ_RT_CACHE`.
- **Configurable alerts engine** (`initAlerts`, `alertsLoad`, `alertsSave`, `alertsCheck`, `alertsRender`, `alertsRemove`, `alertsAddFromUI`): Full localStorage-backed alert system. Alert object schema: `{ id, sym, dir:'above'|'below', threshold, label, fired, firedAt }`. `alertsCheck()` reads live values from `STOOQ_RT_CACHE` and DOM (US 10Y). Fires `Notification` (Notifications API) on first trigger with `tag` deduplication. `alertsCheck()` called on boot + every 5 min via `setInterval`. Permission requested automatically on first alert add.
- **`updatePairDetail()` tooltips:** All 8 `pd-cell` elements now carry `data-tip` attributes with institutional-grade descriptions (e.g. "CFTC Leveraged Funds net contracts (speculative)", "IV minus HV: +ve = options expensive vs realised"). Rendered via pure CSS `::after` pseudo-element — no JS overhead.
- **Latency label fix:** Three instances of `~15min delay` corrected to `~5min delay` (yield curve sub-title, risk monitor sub-title, pair-detail source label). Now consistent with the actual 5-minute GitHub Actions fetch cadence.
- **`exportPanel()` UX:** Added visual feedback on both success (flash `✓` green) and empty-cache warning (flash `NO DATA` orange) on the triggering button.
- **Boot wiring:** `fetchEtfIV()` and `initAlerts()` added to `boot()`. `setInterval(fetchEtfIV, 10 * 60 * 1000)` replaces carry ranking interval.

### Frontend — index.html
- **ETF Options IV panel:** `#carry-rank-rows` container and "Carry Trade Ranking" heading replaced with `#etf-iv-rows` and "Options IV" heading. Sub-label: "ETF · CBOE · 30-day". Footer note: "Implied vol · 30-day ATM · CBOE/yfinance".
- **Configurable alerts panel** (`#alerts-panel`): Inserted above "Central Bank Rates" in `#rightpanel`. Contains `#alerts-rows` (dynamic), `.alert-add-row` with instrument `<select>`, direction `<select>` (`>` / `<`), numeric `<input>`, and `+ Add` button. `#alerts-fired-badge` counter on section heading. Footer note: "Alerts check every 5 min · require browser permission".

### Frontend — assets/dashboard.css
- **`.etf-iv-row`, `.etf-iv-lbl`, `.etf-iv-bar-wrap`, `.etf-iv-bar`, `.etf-iv-bar-high/mid/low`, `.etf-iv-val`, `.etf-iv-chg`:** Full ETF IV panel layout. Bar colours: `var(--down)` (high), `var(--orange)` (mid), `var(--up)` (low).
- **`.pd-cell[data-tip]::after`:** CSS-only tooltip. Positioned above cell, `z-index:999`, `var(--bg2)` background, border, shadow. Edge cells pinned left/right to stay inside viewport.
- **Alert panel styles:** `#alerts-panel`, `.alert-row`, `.alert-row-active`, `.alert-lbl`, `.alert-val`, `.alert-fired`, `.alert-add-row`, `.alert-select`, `.alert-input`, `.alert-add-btn`, `.alert-del`, `.alert-notif-badge`. Consistent with terminal design system.

---

## v7.8.0 (2026-04-04) — Keyboard shortcuts + CSV/JSON panel export

### Frontend — assets/dashboard.js
- **`initKeyboardShortcuts()` IIFE:** Module-level keyboard handler. `G` → FX Pairs, `C` → COT, `R` → Risk, `X` → Cross-Asset, `M` → Macro, `Y` → Rates, `K` → Calendar. Fires `.click()` on the matching `.top-nav` link, reusing the existing scroll + active-state logic. `↑`/`↓` navigate rows in `#fx-pairs-tbody` and call `loadTVChart()` on each step. `?` toggles an accessible legend overlay (`role="dialog"`, `aria-modal="true"`). Handler skipped when focus is on `input`, `textarea`, `select`, or `contentEditable` elements.
- **`exportPanel(type, format)` function:** Client-side export of panel data directly from in-memory caches — no server round-trip. Types: `fx` (STOOQ_RT_CACHE + FX_PERF_CACHE), `cot` (COT_DATA_CACHE), `yield` (DOM yield-tbody), `carry` (STATE.cbRates). Formats: `csv` (default) or `json`. Filename pattern: `gi_{panel}_{ISO-timestamp}.{ext}`. Triggers download via Blob + `URL.createObjectURL`. COT_DATA_CACHE is already populated by the `prefetchCOT()` IIFE added in v7.7.0 — no additional fetch needed.

### Frontend — index.html
- **Export buttons on FX Pairs panel:** `CSV` and `JSON` buttons added to `#section-fxtable` panel-head via `.export-btns` wrapper. Call `exportPanel('fx')` and `exportPanel('fx','json')` respectively.
- **Export buttons on COT panel:** Same pattern on `#section-positioning` panel-head. Call `exportPanel('cot')` and `exportPanel('cot','json')`.
- **`? shortcuts` button in statusbar:** Compact `.kb-hint-btn` in the statusbar right slot. Dispatches a synthetic `?` keydown event to trigger the legend overlay — same code path as the keyboard handler.

### Frontend — assets/dashboard.css
- **`#kb-legend`:** Full-screen dimmed overlay (backdrop-filter blur) containing `.kbl-inner` card. Grid layout: key badge + description. `cursor:pointer` on overlay dismisses on click.
- **`.kb-focus`:** Highlight for the active FX row during keyboard navigation — `var(--bg3)` background + `1px solid var(--blue)` outline.
- **`.export-btn` / `.export-btns`:** Compact monospace buttons with hover transition (`var(--border2)` → `var(--blue)`).
- **`.kb-hint-btn`:** Matching style for the statusbar shortcut trigger button.

---

## v7.7.0 (2026-04-04) — Pair detail panel + full G8 carry trade ranking

### Frontend — index.html
- **`#pair-detail`:** New linked panel at top of `#rightpanel`. Shows placeholder ("Select a pair to view detail") on load; populated on every pair click. `aria-live="polite"` for screen reader updates.
- **Carry Trade Ranking (`#carry-rank-rows`):** New section in `#rightpanel` between pair detail and CB rates. Top 10 G8 pairs by CB rate differential with proportional bar. Each row calls `loadTVChart()`.

### Frontend — assets/dashboard.js
- **`COT_DATA_CACHE`:** Module-level cache. Self-invoking `prefetchCOT()` populates all 8 G8 currencies in parallel on load.
- **`pairMetaFromSym(tvSym)`:** Maps any TradingView symbol string to the matching PAIRS entry.
- **`updatePairDetail(tvSym)`:** Reads STOOQ_RT_CACHE, FX_PERF_CACHE, COT_DATA_CACHE, STATE.cbRates, and loadIntradayQuotes() cache. Renders: price + 1D% + session H/L, 2×4 data grid (1W · HV30 · ATM IV · IV−HV · LF Net · AM Net · Carry · Base Rate), LF≡AM / LF≠AM alignment badge, COT week date.
- **`loadTVChart(sym)`:** Extended to call `updatePairDetail(sym)` on every invocation.
- **`fetchCarryRanking()`:** Builds all 28 G8 pair combinations from STATE.cbRates (fallback: rates/*.json fetch). Sorts by differential descending, renders top 10 with proportional bar. Refreshes every 30 minutes.
- **Boot sequence:** `fetchCarryRanking()` in `boot()`. `setTimeout(() => updatePairDetail('FX_IDC:EURUSD'), 1500)` pre-populates panel.

### Frontend — assets/dashboard.css
- Pair detail panel styles: `.pd-header`, `.pd-sym`, `.pd-price-block`, `.pd-grid`, `.pd-cell`, `.pd-badge`, `.pd-aligned`, `.pd-diverge`, etc.
- Carry ranking styles: `.carry-rank-row`, `.cr-rank`, `.cr-pair`, `.cr-bar-wrap`, `.cr-bar`, `.cr-diff`.

---

## v7.6.0 (2026-04-04) — COT sparklines, LF/AM divergence indicator, panel timestamps

### Engine — .github/workflows/update-cot-cftc-all.yml
- **26-week history accumulator:** Workflow now reads existing `cot-data/*.json` before writing, preserves the `history[]` array, appends a new `{weekEnding, levNet, levLong, levShort}` snapshot, and trims to the last 26 entries. First run starts with 1 point; full window builds over 26 weeks.

### Frontend — assets/dashboard.js
- **`cotSparkline(history)`:** Renders a 52×18px SVG polyline from `history[].levNet`. Scales to min/max of the visible window. Color inherits `--up`/`--down` from the final net position.
- **LF/AM divergence dot:** Each COT row shows `●` (filled, `--up`) when LF and AM net positions have the same sign, `○` (hollow, `--orange`) when they diverge. Reads `assetManagerNet` from the existing JSON schema.
- **Panel timestamps:** Risk Monitor, Yield Curve, and Signals panels now show local time of last update in their subtitle.

### Frontend — index.html
- COT grid column definition updated to include the `6M` sparkline column.

---

## v7.5.1 (2026-04-04) — HTML audit: accessibility fixes + prohibited copy corrections

### Guides — guide-cot.html
- **Prohibited "free" references removed:** Two instances of language describing CFTC data as "free" replaced with "publicly available" and "published by the CFTC" respectively. Complies with GUIDELINES rule prohibiting "free", "at no cost", and similar references.

### Frontend — index.html
- **Added `#sr-announce` live region:** `<div id="sr-announce" role="status" aria-live="polite" aria-atomic="true" class="sr-only">` added before `</body>`. Required by GUIDELINES Accessibility section (WCAG 4.1.3) as the generic screen reader announcement region.
- **Fixed `scope` attribute on 5 `<th>` elements:** Tables for Trading Sessions, Session Volatility, and CB Rate Expectations had column headers without `scope="col"`. All 44 `<th>` elements now carry the correct scope attribute. Complies with GUIDELINES WCAG 1.3.1 rule.

---

## v7.5.0 (2026-04-04) — README maintenance rules + both READMEs updated

### GUIDELINES.md
- **New section "README maintenance"** added before the data panel checklist. Defines which README to update for each type of change (workflow schedule, new panel, new directory, COT schema, etc.). Prohibits cost/pricing references and internal model names in READMEs.
- **Pre-deployment checklist** extended with two new items: both READMEs reviewed, and CHANGELOG + GUIDELINES footer updated.
- Version bumped to v7.5.0.

### README — globalinvesting.github.io (public site)
- "AI market narrative" updated: "3× daily" → "8× daily".
- "CFTC COT positioning" updated to reflect Leveraged Funds / Disaggregated TFF / Options+Futures Combined source.
- "Option skew — 25-delta risk reversals" replaced with accurate "Positioning Bias — ATM IV from CBOE ETF options + COT Leveraged Funds directional bias".
- "News feed" sources updated to match actual active feeds.
- "Market signals" updated: "4–6" → "5–7", "3× daily" → "8× daily".
- `cot-data/` directory description updated to reflect extended schema (Leveraged Funds + Asset Manager + Dealer).
- `intraday-data/` and `fx-performance/` directories added to data directories table.
- Removed "85% of global FX turnover" stat (not verifiable per GUIDELINES) — replaced with "substantial majority".

### README — globalinvesting-engine (private engine)
- Architecture diagram updated: scripts list and workflow list reflect current state of both repos.
- `update_cot_cftc.py` removed (script is inline in the workflow, not a standalone file).
- AI section: removed specific model name ("llama-3.3-70b-versatile") per GUIDELINES — uses "Groq LLM" only; run count updated to 8× daily.
- Data sources table: removed Cost column entirely per GUIDELINES (no pricing references); all sources and frequencies updated to match current workflows.
- Workflow schedule table: fully rewritten to reflect all 12 current workflows with accurate UTC schedules. COT corrected from "Saturday 04:00" to "Saturday 00:30".
- `intraday-data/quotes.json` added to public repo side of architecture diagram.
- Documentation section added pointing to GUIDELINES.md, CHANGELOG.md, SETUP.md.

### Skill — globalinvesting-site
- Step 3 (new): Review and update both READMEs before presenting outputs.
- Step 4 and 5 (renumbered): CHANGELOG and GUIDELINES footer updates.
- Output checklist extended with README review item.
- Quick reference table updated with README paths for both repos.

---

## v7.4.0 (2026-04-04) — COT source upgrade: Disaggregated TFF + Options+Futures Combined

### Engine — .github/workflows/update-cot-cftc-all.yml
- **Primary source switched:** `financial_lf.htm` (Futures Only) → `financial_lof.htm` (Options+Futures Combined). Delta-adjusted options exposure now folded into each category's net — more complete, particularly for EUR and JPY where the options market is active.
- **Extended JSON schema:** `cot-data/*.json` now includes `assetManagerNet/Long/Short`, `dealerNet/Long/Short`, and `sourceType` (`options_futures_combined` or `futures_only`) in addition to the backward-compatible `netPosition/longPositions/shortPositions` (Leveraged Funds).
- **Schedule changed:** `0 20 * * 5` (Friday 20:00 UTC) → `30 0 * * 6` (Saturday 00:30 UTC). Provides 4-hour buffer after CFTC publication (~20:30 UTC Friday), eliminating risk of fetching before the report is live.

### Engine — scripts/generate_narrative_signals.py
- LLM context now includes `lev_net`, `am_net`, and `dd_net` per currency, plus `[O+F]`/`[F]` source tag. Enables the model to detect alignment/divergence between Leveraged Funds and Asset Managers.

### Frontend — assets/dashboard.js
- Positioning Bias header tooltip updated: "CFTC net speculative positioning" → "CFTC Disaggregated TFF net positioning of Leveraged Funds — Options+Futures Combined source".
- Per-pair tooltip `ex:` field updated: removed references to 1W/1M columns (COT fallback format); now describes Leveraged Funds vs Asset Manager convergence/divergence signal.

### Guides — guide-cot.html
- "Trader Categories" section rewritten: replaced legacy Non-Commercial/Commercial split with accurate Disaggregated TFF four-category breakdown (Leveraged Funds, Asset Manager/Institutional, Dealer/Intermediary, Other Reportables), including interpretation guidance for each.
- Added explanation of Options+Futures Combined source and why it is more complete than Futures Only.

### Guides — guide-dashboard.html
- Positioning Bias description updated to name "Leveraged Funds (hedge funds, CTAs)" and "CFTC Disaggregated TFF — Options+Futures Combined" explicitly.
- News Feed mock corrected: removed HIGH/MED/LOW impact labels (not shown in production); layout now matches actual format (timestamp · headline · currency tag · source · date).

### Frontend — index.html
- FAQ schema text updated: "updated weekly every Friday" → "The CFTC publishes on Friday afternoons; the terminal updates overnight Friday–Saturday."

### GUIDELINES.md
- Source label rules updated for Positioning Bias COT column and ATM IV column.
- Script placement table: added PAT_TOKEN exception for cross-repo COT workflow.
- Schedule windows: documented Saturday 00:30 UTC rationale for COT workflow.
- Version bumped to v7.4.0.

---

## v7.3.0 (2026-03-31) — Audit closure: tests, accessibility, architecture

### Frontend — assets/dashboard.js
- **Fix: regime badge no longer flips RISK-ON → CAUTION on page load.**
  Root cause: `renderRiskData()` was called twice in boot before `buildRichNarrative()` had set `_aiRegimeFresh`. With VIX > 25, the live stress score (CAUTION) always overwrote the AI regime badge.
  Fix: `loadAIRegime()` is now `await`ed in `boot()` before `fetchRiskData()`, guaranteeing `_aiRegimeFresh = true` is set before any `renderRiskData()` call touches the narrative badge.
- **Fix: CAUTION/MIXED regime now renders in `var(--orange)` instead of `var(--down)` (red).**
  The `isOn` branch was binary (RISK-ON / RISK-OFF); intermediate states inherited the red color.
- **Fix: narrative badge override logic tightened.**
  `shouldOverride` condition changed from `isCurrentStale || liveRank > currentRank || !_aiRegimeFresh` to `isCurrentStale || !_aiRegimeFresh || (liveRank > currentRank && liveRank >= 2)`. Prevents CAUTION from overriding a fresh AI RISK-ON except when live regime reaches RISK-OFF.
- **Fix: `_narrativeGeneratedAt` scoped correctly.**
  Variable was declared inside `if (narRes.ok)` block but referenced outside; extracted to outer scope.
- **Fix: `localizeSignalTime()` helper added.**
  Converts UTC `HH:MM` timestamps from `signals.json` to the user's local timezone using `toLocaleTimeString()`.
- **Accessibility (WCAG 2.1 AA):**
  - `<header role="banner">` and `<footer role="contentinfo">` landmarks added.
  - Skip-to-content link (`#main`) added as first focusable element (WCAG 2.4.1).
  - `<nav aria-label="Dashboard sections">` and `<aside aria-label>` on sidebar and right panel.
  - All 9 data tables now have `aria-label` and `scope="col"` on every `<th>`.
  - Chart tab strip: `role="tablist"`, `role="tab"`, `aria-selected` on each tab button; selection state synced on click via JS.
  - Quote bar and chart tab scroll buttons: `aria-label` added.
  - Site menu button: `aria-expanded` attribute synced with hover/focus state via JS.
  - Top-nav: `aria-current="location"` applied to active link, updated on click.
  - `role="log" aria-live="polite"` on `#alerts-container`.
  - `aria-live="assertive"` on `#risk-regime`.
  - `role="region" aria-live="polite"` on `#narrative`.
  - `.sr-only` utility class and `role="status"` announcement div added for screen reader price updates.
  - `:focus-visible` restored — the global CSS reset had stripped all focus outlines (WCAG 2.4.7).

### Frontend — assets/dashboard.css
- Skip link (`.skip-link`) styles: hidden off-screen, animates to visible position on `:focus`.
- `:focus-visible` and `button:focus-visible` / `a:focus-visible` rules added with `var(--accent)` outline.
- `.sr-only` utility class (standard visually-hidden pattern).

### Frontend — assets/ architecture (from v7.2.0, carried forward)
- `index.html` reduced from 4 423 → 620 lines by extracting all CSS and JS to `assets/`.
- `unsafe-inline` removed from `script-src` in `netlify.toml` and `_headers`; TradingView CDN domains added explicitly.

### Engine — scripts/generate_narrative_signals.py
- `generate_signals()` now receives `intraday_updated` as anchor timestamp.
- AI prompt updated: signals must carry individual times relative to their priority (critical = most recent, warning = −3 min, info = −9 min).
- Post-processing step guarantees no two signals share the same `HH:MM` timestamp.

### Tests — assets/dashboard.test.js (new file)
Automated test suite runnable with `node assets/dashboard.test.js`. 81 tests, 0 failures.
Covers:
- `fmt`, `clsDir`, `pctStr` — formatting utilities (15 tests)
- `isOpen` — session open/close logic with midnight wrap-around (12 tests)
- `computeRate` — direct, inverted, and cross FX rate calculation; null handling (7 tests)
- Stress scoring — all VIX thresholds (18, 25, 30), all boundary values for gold/SPX/MOVE/curve (14 tests)
- `localizeSignalTime` — null, passthrough, invalid format, midnight edge (6 tests)
- `getLatestBizDate` / `getPrevBizDate` — weekday, Saturday, Sunday, Monday edge cases (7 tests)
- Yield spreads — normal, inverted, flat curves; US-DE spread (4 tests)
- `computeHV30` — minimum 22 closes (mirrors Python engine), annualisation with √252, constant-return zero-variance case, alternating-return known result (9 tests)
- Pearson correlation — perfect ±1, orthogonal, bounds, mismatched lengths, EUR/USD–DXY scenario (7 tests)

---

## v7.2.0 (2026-03-31) — Monolith split + CSP hardening + P2/P3 audit items

### Architecture
- Extracted `assets/dashboard.css` (869 lines), `assets/dashboard.js` (2 857 lines), `assets/gdpr.js` (74 lines) from single-file `index.html`.
- `index.html` reduced from 4 423 to ~620 lines.

### Security
- `unsafe-inline` removed from `script-src` in `netlify.toml` and `_headers`.
- `s3.tradingview.com` and `widgets.tradingview-widget.com` added to explicit `script-src` allowlist.

### UX
- Latency disclaimer `~15 MIN · NOT FOR EXECUTION` added to status bar footer in `var(--orange)`.
- `aria-live="polite"` on `#alerts-container` (`role="log"`).
- `aria-live="assertive"` on `#risk-regime`.
- `aria-live="polite"` on `#narrative` (`role="region"`).
- `role="main" aria-label="Dashboard"` on `<main>`.

---

## v7.0.0 (2026-03-29) — Cleanup: remove legacy scoring system

### Removed
- `calculate-scores.yml`, `save-weekly-scores.yml`, `run-backtest.yml` — entire scoring system workflows
- `generate-ai-analysis.yml` — called Groq 8× per day writing `ai-analysis/{ccy}.json` files (removed from frontend)
- `generate-rss.yml` — RSS feeds removed from frontend
- `backfill-fx-history.yml`, `backfill-historical-rates.yml` — one-shot backfill workflows
- `playwright-visual.yml` — tested `news.html` and `carry-trade.html` (both removed from frontend)
- `fetch-econ-data-apis.yml` — wrote only to `economic-data-history/` (deleted from frontend)
- `scripts/calculate_scores.py`, `scripts/backtest_retrospective.py`, `scripts/generate_historical_scores.py`
- `scripts/save_weekly_scores.py`, `scripts/generate_ai_analysis.py`, `scripts/generate_rss.py`
- `scripts/generate_summaries.py`, `scripts/backfill_fx_history.py`, `scripts/backfill_historical_rates.py`
- `scripts/fetch_econ_data_apis.py` — only used by deleted workflow
- `cleanup_engine.sh` — cleanup script that already ran

### Fixed
- `lighthouse-ci.yml` — replaced `carry-trade.html` and `news.html` URLs with `about.html` and `guide-dashboard.html`
- `forex-news.yml` — removed `generate_summaries.py` step (script deleted; `summaries.json` not consumed by frontend)
- `generate_narrative_signals.py` — removed `strength-scores/latest.json` read (directory deleted from frontend)
- `update_pmi_from_calendar.py` — replaced `fx-history/` with `fx-performance/` as FX rate source (`rateNow` field)
- `fx_config.py` — updated docstring to remove references to deleted scripts
- `monitor_data_health.py` — removed `check_ai_analysis()` function (was checking deleted `ai-analysis/{ccy}.json` files)
- `tests/test_all_scripts.py` — removed imports and test classes for deleted scripts
- `SETUP.md` — removed `strength-scores/` and `scores-history/` from generated outputs list

---

## v6.8.0 (2026-03-27) — Industry upgrade: scoring, signals, backtest, tests

### A1 — `_adj_stale_score`: more aggressive and earlier penalty
- Activation threshold: 8 → 6 weeks
- Maximum penalty: -6 → -8 pts
- Convergence scale: 20 → 16 weeks
- Score threshold: 68 → 66 pts

### A2 — New `_adj_delta_score` function (score momentum)
- Hypothesis AUDIT-4.1: score **change** predicts better than absolute level
- Score rose ≥5 pts in 4 weeks → bonus up to +4 pts
- Score fell ≥5 pts in 4 weeks → penalty up to -4 pts

### A3 — Adjustment cap expanded: ±15 → ±18 pts

### B — FX confirmation gate in signal generation
- Signal confidence `"High"` requires spread ≥ 20 pts **and** price aligned
- New fields `"priceAligned"` and `"fxGateNote"` per signal

### C — Backtest: additional metrics and per-currency histogram
- `sortino_annualized`, `calmar_ratio`, `by_currency` breakdown

### D — Tests: coverage improved from C+ to B
- 14 new integration and regression tests

---

## v6.5 (March 2026) — Data fixes + industry thresholds

- Bullish threshold calibrated to industry standard: >65 (was >60)
- Trade Balance normalisation fix (gdpM divisor)
- Trade Balance FRED series switched to USD denomination
- USD Retail Sales series replaced (level → MoM%)
- GDP threshold fixes in safe-haven adjustment

---

## v6.4 (March 2026) — Commodity ToT + Reserve Currency + Weights

- P1 — Commodity Score via ToT z-score (AUD, CAD, NZD)
- P2 — Reserve Currency Premium data-driven
- servicesPMI weight: 4% → 5%

---

## v6.3 (March 2026) — ESI Proxy + Thresholds

- ESI Proxy (Economic Surprise Index): 22nd model indicator, weight 4%
- Bullish threshold: >65, bearish: <45, neutral zone: 45–65

---

## v6.2 (March 2026) — Services PMI

- servicesPMI: 22nd model indicator, weight 5%

---

## v6.1 (March 2026) — Hawkish Pause + Stagflation

- FIX-6: Hawkish Pause Boost in rateMomentum
- FIX-7: Stagflation Risk contextual adjustment

---

## v6.0 (March 2026) — Fundamental Rebalance

- fxPerformance1M: 28% → 8% (Confirmer, not driver)
- interestRate: 7% → 10%
- rateMomentum: 4% → 7%
- currentAccount: 5% → 7%
- inflationExpectations removed (0%)

---

## v5.11 (March 2026) — Inflation asymmetry + deflation risk zone

---

## v5.9

- FX Performance basket-corrected (vs other 7 currencies, not vs USD)
- Safe haven attenuation when FX 1M performance is negative

---

## v5.7

- COT price-confirmation filter: extreme positioning attenuated when price diverges
