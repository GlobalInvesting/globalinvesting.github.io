# Changelog — globalinvesting-engine

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
