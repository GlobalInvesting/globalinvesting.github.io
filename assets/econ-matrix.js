/**
 * econ-matrix.js v2.5.3 — Native Economic Matrix panel
 *
 * ── v2.5.3 (2026-08-26) — New "Emp Chg" column (net jobs created / employment
 *    change), added per direct feedback from an institutional-background
 *    client: he correctly flagged that G10 labor-market data was missing —
 *    not accurate. "Unemp" already existed and shipped in v2.0.0; what was
 *    genuinely absent was the flow/leading counterpart (net jobs created),
 *    which markets and central banks watch alongside the unemployment
 *    rate/stock figure. Wired using the SAME calendar.json source as every
 *    other column (fetch_ff_calendar.py / calendar-watcher.js already carry
 *    these event titles — no new fetcher needed), with each currency's
 *    title verified directly against live calendar.json before wiring, not
 *    assumed (see CATS entries). USD → 'Non Farm Payrolls' (the headline
 *    print, deliberately not blended with the distinct 'ADP Employment
 *    Change' preview series). GBP/AUD/CAD → bare 'Employment Change'
 *    (excludes each release's Full/Part-Time sub-components). NZD →
 *    'Employment Change QoQ' (NZ's native quarterly cadence, matching its
 *    GDP/CPI columns). EUR → 'Euro Area Employment Change QoQ'/YoY.
 *    Confirmed genuine gaps, left blank rather than guessed: JPY and CHF
 *    have no employment-change-equivalent title in the current source at
 *    all; NOK/SEK's only labor-flow titles ('Unemployed Persons(<Mon>)' /
 *    'Employed Persons(<Mon>)') are raw headcount LEVELS, not a
 *    change/rate figure, and wiring them would silently misrepresent the
 *    column's stated semantics — same principle as not conflating NZD's
 *    PPI Input/Output series elsewhere in this file. index.html's <thead>
 *    and all 10 skeleton <tbody> rows updated in the same change, in the
 *    same position (between PPI and Unemp) — see COLUMNS array warning
 *    comment above.
 *
 * ── v2.5.2 (2026-08-22) — NZD Ind Prod cell: repointed CATS.NZD.prod from
 *    the dead 'Manufacturing Sales YoY' mapping (comment falsely claimed it
 *    was "injected by fetch_supplementary_indicators.py" — that script does
 *    not exist in any repo; the intended replacement never matched a live
 *    NZD event in a full year of calendar.json, audited this session) to
 *    'Industrial Production YoY', fed by new fetch_te_nzd_ind_prod.py
 *    (third-party vendor scrape). Also corrected the header doc block's
 *    "Ind Prod: AUD, NZD, CAD — none of the three..." bullet, stale for
 *    AUD/CAD which were already wired via their own proxies. See
 *    GUIDELINES.md v8.231.0 for the full incident. No other change.
 *
 * ── v2.5.1 (2026-08-19) — ECONMX_POLL_MS 3min → 90s. Backend latency audit
 *    (this session) found the client-side poll was the single heaviest link
 *    in the end-to-end publish→display chain (up to 3min of a ~9-13min
 *    worst-case total) and, unlike the upstream calendar-watcher.js CF Worker
 *    poll (external Myfxbook source, already at its sane floor), carries
 *    effectively no rate-limit risk — this panel only re-reads GlobalInvesting's
 *    own calendar.json off GitHub Pages/CDN, built to serve exactly this kind
 *    of frequent cheap polling. Lowered to 90s to shave ~1.5min off the
 *    worst case without touching any third-party-facing cadence. Synced with
 *    the equivalent change in calendar-panel.js's fetchEconomicCalendar
 *    interval this same session so both panels refresh calendar.json on the
 *    same cadence again (previously 3min here vs 2min there — a drift left
 *    over from calendar-panel.js's own v1.3 2026-06-10 reduction that was
 *    never mirrored here).
 *
 * ── v2.5.0 (2026-08-17) — 10Y Yld / CB Rate went stale for the life of the
 *    session: v2.4.0's periodic refresh only re-fetched calendar.json and
 *    re-rendered against a `_y10Cache`/`_cbCache` snapshot taken once at
 *    first scroll-into-view — reasoned at the time that policy/yield data
 *    "changes far less often" than calendar actuals. False for 10Y specifically:
 *    `extended-data/{CCY}.json` bond10y is written daily by
 *    `fetch_bond_yields.py`, so any tab left open across that daily update
 *    (or open when a fix like the AUD/CAD orphaned-bond10y-field one landed)
 *    kept showing the pre-fix value indefinitely — confirmed live: The client's
 *    screenshot showed AUD 4.83%/CAD 3.59%/NOK 4.20% while the underlying
 *    `extended-data/*.json` already had fresh AUD 5.05% (18 Aug)/CAD 3.72%
 *    (17 Aug)/NOK 4.40% (17 Aug) from the v2.10.2 fix — the panel simply
 *    never re-asked. Fixed: the periodic refresh now re-fetches 10Y (cheap,
 *    ten small JSON files) and recomputes CB Rate (no fetch — `getCBRate()`
 *    already reads live `window._STATE_cbRates`, kept via dashboard.js's own
 *    5-min `health.json` sentinel poll; only the econ-matrix.js snapshot of
 *    it was stale) on every tick, same as calendar.json. `load10y()` also
 *    picked up `{cache:'no-store'}` to match `loadCalendarData()`, so the
 *    browser/SW HTTP cache can't hand back a stale copy inside the 3-min
 *    window either. See GUIDELINES.md/CHANGELOG.md v8.154.3.
 * ── v2.4.0 (2026-08-16) — Live polling: panel used to fetch calendar.json
 *    once (on first scroll-into-view) and never again, so a new actual
 *    required a full page reload to appear. Now re-fetches calendar.json
 *    every 3 min (matching calendar-panel.js's cadence) and re-renders with
 *    cached 10Y/CB values — see GUIDELINES.md/CHANGELOG.md v8.163.0.
 * ── v2.3.0 (2026-08-15) — CB Rate subtext date fixed (was always "01 Aug"
 *    for every currency); Unemp column colored as an inverted indicator ──
 * Two issues the client flagged after reviewing a live screenshot:
 *
 * (1) CB RATE subtext showed the same day-of-month ("Aug 01") for every
 *     single currency, every session — not a rendering bug (fmtDateShort()
 *     itself is correct, proven by the 10Y column using it fine) but a
 *     data-shape mismatch: `getCBRate()`'s date came straight from
 *     `rates/{CCY}.json` observations[0].date, and that file is a FRED-style
 *     MONTHLY series where every observation is stamped to the 1st of its
 *     month by construction (confirmed directly: rates/USD.json's most
 *     recent three observations are 2026-08-01 / 07-01 / 06-01, not real
 *     decision dates) — so "Aug 01" wasn't wrong data, it was real monthly-
 *     bucket data mislabeled with day-level precision it doesn't have.
 *     Fix: `getCBRate()` now also resolves the actual last-meeting date
 *     from `meetings-data/meetings.json`'s per-currency `allMeetings` (real
 *     ISO decision dates, e.g. USD's 2026-07-29 FOMC date) — the most
 *     recent entry not after today — and uses THAT for the subtext instead.
 *     This is the same file `cbrates-modal.js`'s click-through already
 *     reads (`window._STATE_meetings`), so it's reused here the same way
 *     `_STATE_cbRates` already is, with an independent fetch fallback if
 *     that global isn't populated yet. Falls back to the old obs[0].date
 *     behavior only if no meeting data exists for a currency at all.
 *
 * (2) Unemp column reused the generic `trendClass()` (actual > previous →
 *     'up' → var(--up), green) with no inversion — meaning a RISING
 *     unemployment rate rendered green, the same color as rising GDP,
 *     exactly backwards from every other panel in this app. This app
 *     already has one audited, canonical "inverse indicator" keyword list
 *     for this exact problem — `CAL_INVERSE_KW` in calendar-panel.js
 *     (mirrored as `INVERSE_KW` in dashboard.js and `_ESM_INVERSE_KW` in
 *     econ-surprises-modal.js): `['unemployment', 'unemployed', 'jobless',
 *     'claims', 'deficit']` — but econ-matrix.js never adopted it; it was
 *     built independently and this gap was never audited against it until
 *     now. Added the same list here (`MX_INVERSE_KW`) and `trendClass()`
 *     now takes the cell's event title, flipping up/down to down/up when
 *     it matches. Audited every other column's underlying event titles
 *     against this list: GDP/CPI/Core CPI/PPI/Ind Prod/Bus Cond/Rtl Sales/
 *     PCE none match (correct as non-inverted). Cur Acct / Trade Bal titles
 *     ("Current Account", "Trade Balance") also don't literally contain
 *     "deficit" so aren't auto-flagged by this list — their sign is already
 *     baked into the value itself (a worsening balance prints a more
 *     negative number, which parseNum() already reads correctly as "down"),
 *     so no separate inversion is needed there; only Unemp was actually
 *     affected in this file.
 *
 * ── v2.2.9 (2026-08-15) — dropped redundant "10Y"/"Policy" prefix from the
 *    10Y Yld / CB Rate subtext ─────────────────────────────────────────────
 * The client flagged two things about v2.2.8's fix: (1) the "Policy" word in
 * the CB Rate subtext is unnecessary — the column header already says
 * "CB RATE"; (2) "10Y · 30 Jul" repeats "10Y", which the column header
 * ("10Y YLD") already states, unlike the calendar-driven columns where the
 * event name genuinely varies row to row. Both cells now show just the
 * date ("30 Jul", "05 Aug") in econmx-ref, nothing else. fmtDateShort()
 * itself is untouched — only the two literal prefix strings in rowHTML()
 * were removed.
 *
 * ── v2.2.8 (2026-08-15) — 10Y Yld / CB Rate cells given the same
 *    value+subtext structure as every other column ──────────────────────
 * The client flagged that 10Y Yld and CB Rate were the only two columns
 * without a date/period line under the value, breaking the pattern every
 * other column follows. Root cause: rowHTML() built those two cells with
 * a bare '<td>{value}%</td>' (date only in the title tooltip) instead of
 * cellHTML()'s '<div class="econmx-val">/<div class="econmx-ref">'
 * two-line structure used everywhere else. Fixed by giving both cells that
 * same structure, with a new fmtDateShort() helper reused from refLabel()'s
 * 'DD Mon' date format so the subtext reads identically to the calendar-
 * driven columns ("10Y · 30 Jul", "Policy · 05 Aug"). No CSS change needed
 * — .econmx-val/.econmx-ref are already generic rules, not scoped to
 * calendar cells specifically. Purely a rendering fix: load10y()/getCBRate()
 * and their underlying sources are untouched, no data or trend-coloring
 * regression risk.
 *
 * ── v2.2.7 (2026-08-14) — SEK Core CPI wired; corrects a wrong "no vendor
 *    equivalent" gap note; CHF/NZD fetcher live-validated ──────────
 * CATS.SEK.core wired to ['Core Inflation Rate YoY'], same source and
 * event title as CHF/NZD. v2.2.6's SEK note was wrong: it only checked
 * Myfxbook (correctly finding no page there) and concluded no vendor
 * equivalent existed either, without actually checking. The vendor does carry
 * this series (see the private script's header for the URL, labelled
 * "CPIF excl. Energy YoY" \u2014 a different display name than CHF/NZD's
 * pages use, which is likely why the earlier pass missed it). Confirmed
 * server-rendered and live-scraped successfully this session \u2014 see
 * fetch_te_core_inflation.py v3.0, which now generalizes the row-label
 * match per currency instead of assuming "Core Inflation Rate" is
 * universal.
 * CHF/NZD's own fetcher wiring (v2.2.6, UNVALIDATED at the time) is now
 * confirmed working end-to-end: the guest:guest API path it was built
 * against returned HTTP 410 (live-tested this session), so the fetcher
 * was rewritten to scrape the public page directly instead \u2014 both cells
 * populated successfully via a production GH Actions run this session.
 *
 * ── v2.2.6 (2026-08-14) — CHF/NZD Core CPI wired to a new non-Myfxbook
 *    source (a third-party vendor, unvalidated pending a live run) ──────────
 * Wired CATS.CHF.core and CATS.NZD.core to ['Core Inflation Rate YoY'],
 * fed by the new fetch_te_core_inflation.py (globalinvesting-scripts repo)
 * rather than Myfxbook, since no Myfxbook page exists for either (re-
 * confirmed this session). IMPORTANT: NZD's series is the vendor's
 * own "Core Inflation Rate" (RBNZ-sourced, ex-gasoline) \u2014 explicitly NOT
 * the RBNZ Sectoral Factor Model figure quoted in financial press (2.7%
 * YoY Q2 2026 vs this series' ~3.2% YoY) \u2014 see fetch_te_core_inflation.py
 * header for the full distinction; do not conflate the two in copy. Both
 * wirings are UNVALIDATED as of this version: the fetcher's live
 * guest:guest access could not be tested from the session's sandbox
 * (no network path to the vendor's domain) \u2014 run it manually once before
 * scheduling it in a workflow. Until then, or if guest access turns out
 * not to cover these indicators, both cells simply render blank, same as
 * before this change \u2014 no regression risk either way.
 * SEK core (CPIF ex Energy) investigated in the same pass \u2014 no equivalent
 * vendor indicator found either, left unwired, still a
 * documented genuine gap (v2.2.4 finding stands).
 * AUD rtl investigated per the client's report of a still-blank cell despite
 * v2.2.3's fix \u2014 confirmed NOT a wiring bug: the live Myfxbook page itself
 * (australia/retail-sales-mom) has no observation newer than 2025-07-31.

 * No code change; CATS.AUD.rtl stays as wired in v2.2.3.
 * ── v2.2.5 (2026-08-14) — CHF CPI MoM: same wiring-gap pattern as v2.2.4's
 *    NZD PPI fix. CATS.CHF.cpimom was hardcoded to [] under a stale
 *    "no MoM headline release in current source" comment that was never
 *    re-verified — FSO publishes MoM alongside YoY every release and a live
 *    Myfxbook page exists. Wired to ['Inflation Rate MoM']; one-time value
 *    backfilled via backfill_supplementary_events.py v1.2 (FSO-cited) since
 *    Myfxbook RSS's rolling ~24h window means the live pipeline can't
 *    retroactively pull in a release that already passed. CAD PPI YoY/MoM
 *    and NZD PPI Output QoQ (already correctly wired since v2.2.2/v2.2.4)
 *    got the same one-time backfill treatment for the same reason — see
 *    backfill_supplementary_events.py v1.2 for full citations on all three.
 * ──────────────────────────────────────────────────────────────────────────
 * ── v2.2.4 (2026-08-14) — NZD PPI: wiring gap, not a source gap; upstream fix
 *    (v3.43) had already covered it but econ-matrix.js's CATS list was never
 *    updated to match. SEK core (CPIF ex Energy) re-confirmed as a genuine
 *    gap ──────────────────────────────────────────────────────────────────
 * Continues the gap sweep started in v2.2.3, per the client's request to chase
 * the two items that pass explicitly deferred (NZD ppi, SEK core).
 *   - NZD ppi: re-checked live against myfxbook.com/forex-economic-calendar/
 *     new-zealand — found a live page for "PPI Output QoQ" (Low impact,
 *     quarterly, Source: Statistics New Zealand) that v2.2.3's sweep missed.
 *     Crucially, the upstream impact-filter fix that would let this event
 *     through (_IMPACT_UPGRADES's "ppi" substring, fetch_ff_calendar.py
 *     v3.43) was ALREADY shipped before this sweep — this was never an
 *     upstream gap, only a downstream wiring miss in econ-matrix.js's own
 *     CATS.NZD.ppi list, which stayed `[]` after the source-side fix landed.
 *     Wired: ppi: ['PPI Output QoQ']. A second live page also exists under
 *     "PPI Input QoQ" — deliberately not wired, since Input measures what
 *     producers pay (not what they receive) and is a distinct series, not an
 *     alternate cadence of Output — every other currency's ppi column is the
 *     output/producer-price concept.
 *   - SEK core (CPIF Excluding Energy): re-checked against the full live
 *     Sweden Myfxbook calendar listing (checked through its ~Sep 2026
 *     horizon) — no calendar page exists under any title for this series.
 *     The underlying data is real (Riksbank/SCB publish it, confirmed via
 *     search) but Myfxbook doesn't carry a page for it. Re-confirmed genuine
 *     gap, left as `[]`.
 * No upstream (fetch_ff_calendar.py / calendar-watcher.js) changes needed
 * this pass — NZD PPI only needed the matrix-side wiring, not a new
 * _IMPACT_UPGRADES entry.
 *
 * ── v2.2.3 (2026-08-14) — Full gap sweep: CHF PPI, JPY CPI MoM, AUD Retail
 *    Sales were the same pipeline bug as v2.2.2's PPI fix, not source gaps ──
 * Prompted by the client asking for a complete sweep of every "—" cell in the
 * matrix after the v2.2.2 PPI fix shipped. Rather than trust each field's
 * existing "confirmed gap" comment, every one was re-verified against LIVE
 * Myfxbook pages (not calendar.json — see the GUIDELINES.md rule from
 * v8.144.0 on why a derived-file gap check doesn't prove source absence).
 * Three more fields turned out to be the identical impact-filter bug:
 *   - CHF ppi: v2.2.2 had already found the real title ("Producer & Import
 *     Prices YoY/MoM") while investigating the "ppi" substring fix, but
 *     never gave it its own upgrade entry — so it stayed unreachable even
 *     though the title was known. Confirmed live at myfxbook.com/forex-
 *     economic-calendar/switzerland/producer-import-prices-yoy|mom. Fixed
 *     upstream (fetch_ff_calendar.py v3.44, calendar-watcher.js v5.30) and
 *     wired here: ppi: ['Producer & Import Prices YoY', 'Producer & Import
 *     Prices MoM'].
 *   - JPY cpimom: "Inflation Rate MoM" has a live Myfxbook page (japan/
 *     inflation-rate-mom) — YoY already worked because Myfxbook tags it
 *     medium+, MoM is tagged Low and was silently dropped. Fixed upstream,
 *     wired here: cpimom: ['Inflation Rate MoM'].
 *   - AUD rtl: "Retail Sales MoM" has a live Myfxbook page (australia/
 *     retail-sales-mom) — every other G10 currency's retail sales is
 *     already medium+; AUD's alone is Low. Fixed upstream, wired here:
 *     rtl: ['Retail Sales MoM'].
 * All three upstream substring additions were checked against the full live
 * title corpus for accidental collisions before shipping — none found (see
 * fetch_ff_calendar.py v3.44 header for the corpus check).
 * NOT changed (re-investigated, left as-is): NZD ppi — re-confirmed no
 * Myfxbook page exists for NZ producer prices under any title, genuine gap.
 * SEK core (CPIF excluding Energy) — the underlying data is real (Riksbank/
 * SCB actively publish it, confirmed via search), but no Myfxbook calendar
 * page could be located for it in this pass; left as a documented gap
 * rather than wiring an unverified title. Flagged for a follow-up pass with
 * direct Myfxbook access if the client wants it chased further.
 *
 * ── v2.2.2 (2026-08-14) — GBP/JPY/CAD PPI: fixed pipeline bug misdiagnosed
 *    as a source gap in v2.2.0 ──────────────────────────────────────────
 * v2.2.0 documented GBP/JPY/CHF/CAD/NZD as "confirmed genuine gaps — no
 * PPI release in the current source." Re-audited against live Myfxbook
 * pages per currency (not assumed): three of the five were wrong — the
 * releases exist, but fetch_ff_calendar.py's/calendar-watcher.js's
 * IMPACT_UPGRADES table only matched the literal phrase "producer price
 * index", which never appears in Myfxbook's actual titles (always the
 * short form: "PPI YoY", "United Kingdom PPI Output YoY", "Canada PPI
 * MoM"). Myfxbook tags GBP/JPY/CAD's PPI Low impact, so all three were
 * silently dropped by the pipeline's impact filter before ever reaching
 * calendar.json — USD/EUR/AUD/NOK/SEK only ever worked because Myfxbook
 * already tags those medium/high directly, which masked the bug. Fixed
 * upstream (fetch_ff_calendar.py v3.43, calendar-watcher.js v5.29, engine
 * repo) and wired the real titles here: GBP → ['PPI Output YoY', 'PPI
 * Output MoM'], JPY/CAD → ['PPI YoY', 'PPI MoM']. CHF and NZD verified
 * as genuine gaps — CHF's real title ("Producer & Import Prices YoY")
 * doesn't match either the old or new impact-upgrade entry, and NZD has no
 * PPI/producer-price page on Myfxbook at all — both left as-is (empty
 * array, GAP_TITLE stands). See CHANGELOG.md v8.144.0.
 *
 * ── v2.2.1 (2026-08-14) — CRITICAL: fixed index.html column desync caused
 *    by v2.2.0's new PPI column ─────────────────────────────────────────
 * v2.2.0 added 'ppi' to COLUMNS (11 → 12 categories) but index.html's
 * <thead> and 10 skeleton <tbody> rows were never updated to add the
 * matching PPI <th>/<td> — a step this file cannot enforce on its own, since
 * rowHTML() builds cells purely by iterating COLUMNS in order with no
 * awareness of what index.html's static markup declares. Effect: every
 * column from PPI onward rendered shifted one position left of its header —
 * PCE data appeared under the "10Y Yld" header (confirmed by a client
 * screenshot showing a "United States PCE Price Index YoY" tooltip on that
 * cell), the real 10Y yield appeared under "CB Rate", and CB Rate itself was
 * pushed off the end of the table. Reported by the client from a live
 * screenshot. Fixed in index.html only (no logic in this file was wrong) —
 * see CHANGELOG.md v8.143.0 for the full incident and the new GUIDELINES.md
 * rule requiring COLUMNS-array changes and index.html's thead/skeleton rows
 * to be edited in the same change.
 *
 * ── v2.2.0 (2026-08-14) — closing out remaining industry-standard gaps:
 *    CHF GDP dead title, EUR Bus Cond gap resolved, new PPI column ────────
 * Prompted by the client asking to close out every remaining item after
 * v2.1.0, rather than leave anything flagged-but-unfixed. Three changes:
 *   (1) CHF gdp had the same class of bug as v2.1.0's GBP fix, inverted:
 *       'GDP Growth Rate QoQ Flash' matched ZERO events in the feed (the
 *       real title is 'GDP Growth Rate QoQ', no "Flash") \u2014 QoQ was
 *       silently unreachable for CHF regardless of freshness, not a
 *       deliberate YoY-first policy choice. Fixed the title; the visible
 *       output is unchanged today (YoY still wins most quarters on
 *       freshness \u2014 verified against the live feed), but the fallback
 *       chain the column's tooltip promises now actually works.
 *   (2) EUR Bus Cond ("\u2014" since v2.0.0, see that note) reinvestigated
 *       rather than left as a standing gap: the feed's more recent entries
 *       (~Jun 2026 onward) now carry country-prefixed titles that didn't
 *       exist when v2.0.0 shipped (vendor formatting change, confirmed via
 *       an identical-value overlap date, not new/different data). Germany's
 *       Ifo Business Climate \u2014 a single, clean, continuous, widely-cited
 *       series \u2014 is now shown as EUR's proxy, the same pattern already
 *       used for AUD's RBA Trimmed Mean CPI. Also found the same drift
 *       affects PPI (see below).
 *   (3) New PPI column, per the client's original request (Dimitrius's
 *       "IPP (infla\u00e7\u00e3o do produtor)"). Verified per-currency coverage
 *       against the live feed before wiring anything: real data exists for
 *       USD (MoM only \u2014 no YoY title in this feed), AUD (QoQ \u2014 ABS's
 *       genuine native cadence, not a fallback), NOK (YoY), SEK (YoY+MoM,
 *       YoY preferred), and EUR (Germany's national PPI, per the same
 *       investigation as (2) \u2014 the bare "PPI YoY" title turned out to
 *       ALSO be Germany's series, not a Euro Area aggregate, confirmed by
 *       the identical overlap-date value). GBP/JPY/CHF/CAD/NZD are
 *       confirmed gaps \u2014 no PPI release in the current source \u2014 not
 *       guessed or left ambiguous.
 * Column tooltip and header-comment "intentionally blank" list both
 * updated to match. Not touched this session (out of scope, no client
 * signal either): PMI Services/Composite \u2014 checked against the live
 * feed and found only 2 of 10 currencies (USD, SEK) carry a genuine
 * Services PMI title, and zero carry a Composite PMI title at all. Adding
 * a column that reads "\u2014" for 8-9 of 10 rows would be a worse outcome
 * than the informational PMI-scale note now added to the Bus Cond tooltip
 * instead \u2014 flagged in CHANGELOG as a deliberate non-addition with the
 * reasoning, not silently dropped.
 *
 * ── v2.1.0 (2026-08-14) — GDP column: fixed GBP QoQ omission, tagged USD's
 *    SAAR convention ──────────────────────────────────────────────────────
 * Prompted by a follow-up question from the client after v2.0.0 shipped: is it
 * industry-standard for the GDP column to show different periodicities
 * (QoQ/MoM/YoY) across different currency rows? Investigation found three
 * distinct things bundled under that one question:
 *   (1) GBP bug: the CATS.GBP.gdp prefix list was `['GDP MoM']` only — it
 *       never included the QoQ title at all, even though the UK's ONS DOES
 *       publish a quarterly GDP figure, released the SAME DAY as the
 *       monthly print 3 of every 4 months (confirmed in calendar.json:
 *       "United Kingdom GDP Growth Rate QoQ" and "...GDP MoM" both dated
 *       2026-08-13). This contradicted the column's own documented policy
 *       ("QoQ where published") and meant the cell could never show the
 *       quarterly headline even on a release day. Fixed by adding
 *       'GDP Growth Rate QoQ' ahead of 'GDP MoM' in the prefix list —
 *       findLatestGeneric()'s existing same-date tie-break now resolves to
 *       QoQ on release days, and falls back to MoM (the freshest real data)
 *       in the two in-between months, same as it always did.
 *   (2) CAD reviewed and left as-is: its quarterly print is genuinely stale
 *       (last Q1 2026, from 2026-05-29) relative to the monthly print
 *       (2026-07-31), so the existing freshest-date selection already
 *       surfaces the right cell — not a bug, no change needed.
 *   (3) Bigger institutional-comparability issue, found during this
 *       investigation (not what was asked, but material to "industry
 *       standard"): USD's matched GDP title, "GDP Growth Rate QoQ", is by
 *       the BEA's own convention already seasonally-adjusted ANNUALIZED
 *       (SAAR) — the familiar "grew at an annualized pace of X%" figure.
 *       Every other currency's "GDP Growth Rate QoQ" is the raw,
 *       non-annualized quarterly change. Both were tagged identically as
 *       "QoQ" in the per-cell subtext, which would read as directly
 *       comparable magnitudes when they are not (US 1.5% SAAR vs EA 0.4%
 *       raw QoQ implies the US grew ~4x faster than it actually did in
 *       comparable terms — the non-annualized equivalent is ~0.37%, in
 *       line with the EA print). Fixed by special-casing periodLabel() for
 *       gdp/USD/qoq to render "QoQ SAAR" instead of "QoQ" — every other
 *       cell's tag is unaffected. Column tooltip updated to disclose this
 *       explicitly rather than relying on the reader to already know US
 *       GDP convention.
 * Not changed: CHF's gdp list still prioritizes YoY over its QoQ Flash —
 * this wasn't part of what was asked and needs its own verification pass
 * before touching (see "On the horizon" in CHANGELOG.md).
 *
 *
 * Replaces the third-party TradingView Economic Map widget (tv-economic-map.js)
 * with a native table in the style of an institutional regional economic matrix
 * (e.g. Bloomberg ECMX), built entirely from data the terminal already fetches
 * elsewhere — no new backend script or workflow required.
 *
 * ── v2.0.0 (2026-08-14) — institutional-user data-accuracy audit ──────────
 * Prompted by a client (Dimitrius, via the operator) flagging: (1) USD CPI YoY
 * showing a stale 4.2% print instead of the then-current 3.4%; (2) AUD CPI
 * showing "102.03" — an index level in points, not a %-rate; (3) no MoM
 * alongside YoY, and no visible reference date per cell; (4) no core/PCE
 * measure. Investigation (see CHANGELOG v8.140.0) found the root cause was
 * NOT stale source data — calendar.json already carried the correct latest
 * prints — but a title-matching bug in `findLatest()`:
 *   - Myfxbook titles the SAME release inconsistently: sometimes bare
 *     ("Inflation Rate YoY"), sometimes country-prefixed ("United States
 *     Inflation Rate YoY"). The old CATS prefix lists only covered a few
 *     observed variants per currency, so on any date where the upstream
 *     title format drifted, `findLatest()` silently fell through to an
 *     older event that happened to match a listed variant — this is what
 *     surfaced the stale-looking 4.2% (a real June print, just not the
 *     latest one).
 *   - AUD CPI: "Australia CPI" (the raw index, e.g. 102.03) and "Australia
 *     Inflation Rate YoY" (the %, e.g. 3.8%) are two DIFFERENT releases that
 *     print on the SAME day. The old code had no way to prefer one over the
 *     other when both matched on the same date, so it non-deterministically
 *     picked whichever appeared later in the day's insertion order — the
 *     index level, on this occasion.
 *   - EUR bare "Business Confidence" was found to silently interleave THREE
 *     unrelated national surveys (three different value ranges — ~87-89,
 *     ~96-105, and negative ~-3 to -6 — with no country field to
 *     disambiguate) under one identical title. This was already a latent
 *     bug, not something the client reported, caught during this audit.
 * Fix (see below): (1) country-prefix canonicalisation before matching, so
 * "United States X" and bare "X" key to the same series regardless of which
 * form Myfxbook used that day; (2) EUR is matched WITHOUT canonicalisation,
 * against explicit "Euro Area " literal prefixes only — this is a
 * deliberate asymmetry, not an oversight, because EUR is the one currency
 * where stripping the country prefix would blend in member-state prints
 * (Germany, France, Italy...); (3) strict prefix match (exact, or with a
 * trailing "(" for parenthetical month/quarter suffixes) instead of loose
 * startsWith, so "CPI" never matches "CPI Trimmed-Mean"; (4) deterministic
 * same-day tie-break — the FIRST prefix in a category's list that has a
 * match on the single most-recent date wins, so prefix order is now a real,
 * documented priority ranking, not an accident of iteration order; (5)
 * dropped "Australia CPI" from the AUD cpi prefix list entirely — the index
 * level has no place in a %-rate column; (6) EUR "Bus Cond" is left blank
 * with an explanatory tooltip rather than showing an unverifiable blend of
 * three surveys (see GUIDELINES.md "Data integrity" — never display a value
 * that isn't reliably attributable to a single named release).
 *
 * New in v2.0.0, per client request:
 *   - CPI MoM added alongside CPI YoY (a YoY figure can mask a recent trend
 *     reversal from base effects — e.g. an energy spike 12 months ago still
 *     weighing on the annual figure without reflecting current conditions).
 *   - Core CPI YoY added — central banks weight core more than headline,
 *     since headline is dominated by volatile/seasonal food & energy.
 *     AUD uses the RBA Trimmed Mean CPI as its core-equivalent (see RBA's
 *     own published rationale: trimming extreme price moves gives a
 *     cleaner read on persistent underlying inflation — this is the
 *     standard cross-market substitute for "core CPI" in AUD, since
 *     Australia doesn't publish an ex-food-and-energy core CPI the way the
 *     US/UK/EA do). CHF/NZD/SEK have no core measure in the current feed —
 *     left blank with a tooltip rather than guessed at.
 *   - PCE YoY added as a currency-specific column, populated for USD only
 *     (the Fed's preferred inflation gauge) — blank for all other
 *     currencies with a tooltip explaining PCE is a US-specific series;
 *     other central banks target CPI/HICP-based measures instead (e.g. the
 *     RBA's Trimmed Mean, already shown in the Core CPI column, or the
 *     ECB's HICP, already the basis of the EUR CPI columns here).
 *   - Every calendar-derived cell now shows its reference period + date as
 *     a small subtext line under the value (e.g. "YoY · 12 Aug"), so YoY
 *     and QoQ/MoM prints are never visually ambiguous (NZD CPI, for
 *     instance, is quarterly-only and now reads "QoQ · 20 Jul" rather than
 *     a bare "1.5%" that could be mistaken for an annual figure) and
 *     staleness is visible at a glance without opening the tooltip.
 *
 * Column sourcing:
 *   GDP, CPI YoY, CPI MoM, Core CPI, Unemp, Ind Prod, Bus Cond, Rtl Sales,
 *   Cur Acct, Trade Bal, PCE
 *     → calendar-data/calendar.json — latest "actual" print per category per
 *       currency. This file carries ~1yr of history with real released actuals
 *       (unlike economic-data/{CCY}.json, disabled in v7.24.1 for staleness —
 *       see GUIDELINES.md "Data directories").
 *   10Y Yld
 *     → extended-data/{CCY}.json `bond10y` — same field already used by
 *       yc-modal.js for the Yield Curve detail modal. No color/trend shown,
 *       matching the established precedent there (extended-data carries no
 *       intraday delta for this field).
 *   CB Rate
 *     → window._STATE_cbRates (populated by fetchCBRates() in dashboard.js) +
 *       computeCBTrend() for the trend arrow color — reused as-is so this
 *       panel never disagrees with the CB Rates table elsewhere on the page.
 *
 * Documented deviations from the Bloomberg ECMX column set (see CHANGELOG
 * v8.23.0 for full rationale, extended in v8.140.0):
 *   - "Bud %GDP" (fiscal budget balance) has no recurring calendar release
 *     outside the US in the current source, so the column is replaced with
 *     Trade Balance, which the calendar carries for all 10 currencies and is
 *     directly FX-relevant.
 *   - "CA %GDP" is shown as "Cur Acct" — the calendar's raw latest actual, in
 *     each currency's own native reporting units, rather than a %GDP ratio.
 *     The values are not uniformly unit-tagged at the source (some carry a
 *     currency prefix, some don't), so dividing by a GDP denominator would
 *     manufacture false precision. Trend coloring still works (see below).
 *
 * Cells are intentionally left blank ("—") where the underlying release does
 * not exist in the current source for that currency, or where the source
 * data cannot be reliably attributed to a single named release:
 *   - Ind Prod: none of AUD/NZD/CAD have a standalone Ind Prod release on
 *     Myfxbook (this feed's primary source), but all three are populated
 *     via proxies/direct fetches, none of them blank in current production:
 *     AUD via Ai Group Industry Index (Myfxbook), CAD via Manufacturing
 *     Sales MoM (Myfxbook), NZD via the vendor's own "Industrial
 *     Production" series (fetch_te_nzd_ind_prod.py — Myfxbook's RSS feed
 *     never surfaces NZD's equivalent release despite a live page existing
 *     for it; see that script's module docstring for the full incident).
 *     This bullet previously (through v2.4.x) claimed all three were
 *     genuine gaps — stale even for AUD/CAD, which had already been wired;
 *     corrected here per the same-session NZD audit (GUIDELINES.md v8.231.0).
 *   - Rtl Sales: AUD — not currently tracked in the source feed (a feed gap,
 *     the ABS does publish retail trade figures).
 *   - Cur Acct: EUR, AUD — not currently tracked in the source feed for
 *     these two (a feed gap, not a "doesn't exist" fact — both the ECB and
 *     the ABS do publish a current account series).
 *   - Core CPI: CHF, NZD, SEK — no core/underlying-inflation release in the
 *     current source for these three.
 *   - PCE: every currency except USD — PCE is a US-specific series (see
 *     above); this is a genuine "doesn't exist for this economy" fact, not
 *     a feed gap.
 *   - PPI: GBP, JPY, CHF, CAD, NZD — no producer-price release in the
 *     current source for these five (see v2.2.0 note below).
 *
 * Bus Cond fallback — these currencies have no Manufacturing PMI in the
 * current source, so the column falls back to each economy's standard
 * business/industrial confidence survey instead (per the column definition
 * above):
 *   - GBP: CBI Industrial Trends Orders (CBI manufacturing orders survey)
 *   - JPY: Tankan Large Manufacturers Index (BoJ's quarterly tankan survey —
 *     the benchmark Japanese manufacturer-sentiment gauge)
 *
 * Color convention: every calendar-derived cell is colored by the direction
 * of change vs. the previous reading (delta = actual − previous), not by the
 * raw sign of the level. This is purely descriptive (mirrors how price/D%/W%
 * deltas are colored elsewhere in the terminal) and avoids any "good/bad"
 * value judgement on a given print, consistent with GUIDELINES' ban on
 * investment-advice-flavored signal language.
 */
(function () {
  'use strict';

  const CCY_ORDER = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CHF', 'CAD', 'NZD', 'NOK', 'SEK'];
  const FLAG = { USD: 'us', EUR: 'eu', GBP: 'gb', JPY: 'jp', AUD: 'au', CHF: 'ch', CAD: 'ca', NZD: 'nz', NOK: 'no', SEK: 'se' };

  // ⚠ COLUMNS length/order is NOT self-enforcing against index.html. Adding,
  // removing, or reordering an entry here means index.html's Economic
  // Matrix <thead> (<th scope="col">) AND all 10 skeleton <tbody> rows'
  // <td class="flat">—</td> placeholders must be edited in the SAME change,
  // in the same order, or every column from the edit point onward silently
  // renders shifted under the wrong header (see v2.2.1 incident above —
  // this exact bug shipped with v2.2.0's PPI column).
  const COLUMNS = [
    { key: 'gdp',     label: 'GDP',       title: 'Latest GDP growth rate \u2014 QoQ where published, YoY otherwise, falling back to the freshest monthly print for GBP/CAD between quarterly releases (see subtext on each cell for the period actually shown). Note: USD\u2019s QoQ is seasonally-adjusted ANNUALIZED (SAAR) per BEA convention \u2014 tagged \u201cQoQ SAAR\u201d, not directly comparable in magnitude to the raw non-annualized QoQ shown for other currencies.' },
    { key: 'cpi',     label: 'CPI YoY',   title: 'Latest headline CPI / inflation rate, year-on-year' },
    { key: 'cpimom',  label: 'CPI MoM',   title: 'Latest headline CPI / inflation rate, month-on-month \u2014 can reveal a trend reversal the YoY figure masks via base effects' },
    { key: 'core',    label: 'Core CPI',  title: 'Latest core/underlying inflation, year-on-year \u2014 excludes volatile food & energy components; the measure central banks weight most heavily. AUD shows the RBA Trimmed Mean CPI, Australia\u2019s standard core-equivalent.' },
    { key: 'ppi',     label: 'PPI',       title: 'Latest producer-price inflation \u2014 YoY where published, QoQ/MoM otherwise (see subtext on each cell for the period actually shown). EUR shows Germany\u2019s national PPI as a proxy \u2014 no genuine Euro Area-aggregate PPI title exists in the current source. Blank where the currency\u2019s economy has no standalone PPI release in the current source.' },
    // v2.5.3: added per client feedback (institutional trader, 2026-08-26) —
    // Unemp already existed, but the flow/leading indicator (net jobs
    // created) did not, and central banks + markets watch both. Verified
    // per-currency titles directly against live calendar-data/calendar.json
    // (not assumed) before wiring — see CATS entries for per-currency
    // sourcing notes and confirmed gaps (JPY/CHF/NOK/SEK).
    { key: 'emp',     label: 'Emp Chg',   title: 'Latest employment change \u2014 net jobs created, a flow/leading labor-market indicator distinct from the Unemployment Rate (a stock/lagging indicator). US shows Non-Farm Payrolls, the single most market-watched G10 print. Blank where the currency\u2019s economy has no standalone employment-change release in the current source \u2014 see column-specific per-currency notes.' },
    { key: 'unemp',   label: 'Unemp',     title: 'Latest unemployment rate' },
    { key: 'prod',    label: 'Ind Prod',  title: 'Latest industrial / manufacturing production change' },
    { key: 'conf',    label: 'Bus Cond',  title: 'Latest manufacturing PMI, or the economy\u2019s standard business/industrial confidence survey where no PMI is published. PMI readings are on a 0\u2013100 scale where 50 is the expansion/contraction cutoff \u2014 non-PMI substitutes (Ifo, NAB Business Confidence, Industrial Confidence, etc.) use their own survey-specific scale with no fixed 50 threshold.' },
    { key: 'rtl',     label: 'Rtl Sales', title: 'Latest retail sales change' },
    { key: 'ca',      label: 'Cur Acct',  title: 'Latest current account, native reporting units (see GUIDELINES \u2014 not normalized to %GDP)' },
    { key: 'trade',   label: 'Trade Bal', title: 'Latest trade balance, native reporting units' },
    { key: 'pce',     label: 'PCE YoY',   title: 'Latest PCE Price Index, year-on-year \u2014 the U.S. Federal Reserve\u2019s preferred inflation gauge. US-specific; other economies target CPI/HICP-based measures shown in the CPI/Core CPI columns instead.' },
  ];

  // Country-name prefixes Myfxbook/ForexFactory sometimes (not always)
  // prepend to an otherwise-bare event title, e.g. "United States Inflation
  // Rate YoY" vs bare "Inflation Rate YoY" for the identical release. Used to
  // canonicalise BEFORE matching so title-format drift on any given day never
  // causes findLatest() to silently fall back to an older event (see v2.0.0
  // note above \u2014 this was the actual root cause of the reported stale-CPI
  // symptom). Deliberately mirrors calendar-panel.js's `_CAL_CCY_PFXS` so the
  // two independent matchers stay conceptually in sync; EUR is excluded on
  // purpose (see EUR handling below).
  const CCY_PFXS = {
    USD: 'united states ', GBP: 'united kingdom ', JPY: 'japan ', AUD: 'australia ',
    CAD: 'canada ', CHF: 'switzerland ', NZD: 'new zealand ', NOK: 'norway ', SEK: 'sweden ',
  };
  function canon(ccy, title) {
    const pfx = CCY_PFXS[ccy];
    if (pfx && title.toLowerCase().indexOf(pfx) === 0) return title.slice(pfx.length);
    return title;
  }

  // Strict prefix match: exact title match, or prefix immediately followed by
  // a parenthetical suffix such as "(Apr)" / "(Q1)" that NOK/SEK titles carry.
  // Deliberately NOT a loose startsWith \u2014 "CPI" must never match "CPI
  // Trimmed-Mean" or "CPIF", and "Inflation Rate YoY" must never match
  // "Inflation Rate YoY Flash" (a different, preliminary release).
  function strictMatch(title, prefix) {
    if (title.length < prefix.length || title.slice(0, prefix.length) !== prefix) return false;
    const rest = title.slice(prefix.length);
    return rest === '' || rest.charAt(0) === '(';
  }

  // Union of accepted ForexFactory/Myfxbook event-title prefixes per
  // category, per currency, in bare (country-prefix-stripped) canonical
  // form \u2014 except EUR, which is matched separately (see findLatestEUR).
  // Order is a real priority ranking: for a same-day tie between two
  // prefixes in one category, the prefix listed FIRST wins (see AUD cpi \u2014
  // deliberately does NOT include "CPI", which is the raw index level in
  // points, not a rate; see v2.0.0 note above). Empty array = confirmed gap
  // for that currency (see header comment) \u2014 renders "\u2014".
  const CATS = {
    USD: {
      gdp:   ['GDP Growth Rate QoQ'],
      cpi:   ['Inflation Rate YoY'],
      cpimom:['Inflation Rate MoM'],
      core:  ['Core Inflation Rate YoY'],
      ppi:   ['PPI MoM'], // confirmed only MoM published in the current source \u2014 no PPI YoY title observed
      // v2.5.3: 'Non Farm Payrolls' only \u2014 deliberately NOT mixed with 'ADP
      // Employment Change' (a distinct private-sector preview series released
      // ~2 days before NFP, not an alternate cadence of the same release,
      // same non-mixing principle as NZD's PPI Input/Output note below). NFP
      // is the headline print markets and the Fed anchor to.
      emp:   ['Non Farm Payrolls'],
      unemp: ['Unemployment Rate'],
      prod:  ['Industrial Production MoM'],
      conf:  ['ISM Manufacturing PMI'],
      rtl:   ['Retail Sales MoM'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade', 'Goods Trade Balance'],
      pce:   ['PCE Price Index YoY'],
    },
    GBP: {
      // GDP: QoQ is the UK's headline growth figure and IS published — ONS
      // bundles it into the same release as the monthly print 3 of every 4
      // months (see v2.1.0 note above). Listed first so the same-day tie
      // resolves to QoQ; MoM is still what's freshest in the two
      // in-between months and remains the correct fallback there.
      gdp:   ['GDP Growth Rate QoQ', 'GDP MoM'],
      cpi:   ['Inflation Rate YoY'],
      cpimom:['Inflation Rate MoM'],
      core:  ['Core Inflation Rate YoY'],
      // v2.2.2: NOT a genuine gap \u2014 was a pipeline bug, not a missing source.
      // Myfxbook's real title is "PPI Output YoY"/"PPI Output MoM" (UK's PPI
      // has separate Output/Input series; "Output" is the headline one, same
      // convention BoE/ONS coverage uses). fetch_ff_calendar.py/calendar-
      // watcher.js's impact-upgrade table only matched the literal phrase
      // "producer price index", which never appears in Myfxbook's actual
      // titles \u2014 GBP's PPI (Myfxbook-tagged Low impact) was silently
      // dropped before it ever reached calendar.json. Fixed upstream in
      // fetch_ff_calendar.py v3.43 / calendar-watcher.js v5.29; wiring the
      // real title here now that the source will actually carry it.
      ppi:   ['PPI Output YoY', 'PPI Output MoM'],
      // v2.5.3: bare 'Employment Change'.
      emp:   ['Employment Change'],
      unemp: ['Unemployment Rate'],
      prod:  ['Industrial Production MoM'],
      conf:  ['S&P Global Manufacturing PMI', 'CBI Industrial Trends Orders'],
      rtl:   ['Retail Sales MoM'],
      ca:    ['Current Account'],
      trade: ['Goods Trade Balance', 'Balance of Trade'],
      pce:   [],
    },
    JPY: {
      gdp:   ['GDP Growth Rate QoQ Final', 'GDP Growth Rate QoQ Prel', 'GDP Growth Rate QoQ'],
      cpi:   ['Inflation Rate YoY'],
      // v2.2.3: NOT a genuine gap — same pipeline bug as ppi below. Myfxbook's
      // real title is bare "Inflation Rate MoM" (live page confirmed:
      // japan/inflation-rate-mom) — tagged Low impact while YoY is medium+,
      // so MoM was silently dropped. Fixed upstream in fetch_ff_calendar.py
      // v3.44 / calendar-watcher.js v5.30.
      cpimom:['Inflation Rate MoM'],
      core:  ['Core Inflation Rate YoY'],
      // v2.2.2: NOT a genuine gap \u2014 same pipeline bug as GBP above.
      // Myfxbook's real title is bare "PPI YoY"/"PPI MoM" (canon() strips
      // any "Japan " prefix drift the same way it does elsewhere). Fixed
      // upstream in fetch_ff_calendar.py v3.43 / calendar-watcher.js v5.29.
      ppi:   ['PPI YoY', 'PPI MoM'],
      // v2.5.3: confirmed gap \u2014 no 'Employment Change'-equivalent title
      // found in calendar.json for JPY. Unemployment Rate remains Japan's
      // primary published labor-market indicator in the current source.
      emp:   [],
      unemp: ['Unemployment Rate'],
      prod:  ['Industrial Production MoM Prel', 'Industrial Production MoM'],
      conf:  ['Jibun Bank Manufacturing PMI', 'Tankan Large Manufacturers Index'],
      rtl:   ['Retail Sales YoY'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade'],
      pce:   [],
    },
    AUD: {
      gdp:   ['GDP Growth Rate QoQ', 'GDP Growth Rate YoY'],
      // Deliberately excludes "CPI" \u2014 that title is the raw index level in
      // points (e.g. 102.03), not a %-rate. See v2.0.0 note above.
      cpi:   ['Inflation Rate YoY'],
      cpimom:['Inflation Rate MoM'],
      // RBA Trimmed Mean CPI is AUD's standard core-equivalent (see column def).
      core:  ['RBA Trimmed Mean CPI YoY', 'Quarterly RBA Trimmed Mean CPI YoY'],
      ppi:   ['PPI QoQ'], // ABS publishes PPI quarterly, not monthly \u2014 QoQ is the genuine native cadence, not a fallback
      // v2.5.3: bare 'Employment Change' (net, seasonally adjusted) \u2014
      // deliberately excludes 'Full Time Employment Chg', a sub-component
      // of the same release, not an alternate cadence (same non-mixing
      // principle as NZD's PPI Input/Output note below).
      emp:   ['Employment Change'],
      unemp: ['Unemployment Rate'],
      prod:  ['Ai Group Industry Index'], // Ai Group Performance of Manufacturing \u2014 published monthly by Ai Group Australia
      conf:  ['NAB Business Confidence'],
      // v2.2.3: NOT a genuine gap \u2014 same pipeline bug as JPY cpimom above.
      // Myfxbook's real title is bare "Retail Sales MoM" (live page confirmed:
      // australia/retail-sales-mom) \u2014 tagged Low impact while every other
      // G10 currency's retail sales is medium+, so AUD's alone was silently
      // dropped. Fixed upstream in fetch_ff_calendar.py v3.44 /
      // calendar-watcher.js v5.30.
      rtl:   ['Retail Sales MoM'],
      ca:    ['Current Account'], // ABS BOP quarterly \u2014 injected by fetch_supplementary_indicators.py
      trade: ['Balance of Trade'],
      pce:   [],
    },
    CAD: {
      gdp:   ['GDP MoM', 'GDP Growth Rate Annualized'],
      cpi:   ['Inflation Rate YoY'],
      cpimom:['Inflation Rate MoM'],
      core:  ['Core Inflation Rate YoY'],
      // v2.2.2: NOT a genuine gap \u2014 same pipeline bug as GBP/JPY above.
      // Myfxbook's real title is bare "PPI YoY"/"PPI MoM" (StatCan's IPPI;
      // canon() strips any "Canada " prefix drift the same way it does
      // elsewhere). Fixed upstream in fetch_ff_calendar.py v3.43 /
      // calendar-watcher.js v5.29.
      ppi:   ['PPI YoY', 'PPI MoM'],
      // v2.5.3: bare 'Employment Change' \u2014 excludes 'Full/Part Time
      // Employment Chg' sub-components, same reasoning as AUD above.
      emp:   ['Employment Change'],
      unemp: ['Unemployment Rate'],
      prod:  ['Manufacturing Sales MoM', 'Manufacturing Sales YoY'], // StatCan via FRED (MoM) or OECD MEI (YoY fallback) \u2014 injected by fetch_supplementary_indicators.py
      conf:  ['Ivey PMI s.a', 'S&P Global Manufacturing PMI'],
      rtl:   ['Retail Sales MoM', 'Retail Sales MoM Final', 'Retail Sales Ex Autos MoM'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade'],
      pce:   [],
    },
    CHF: {
      // v2.2.0: 'GDP Growth Rate QoQ Flash' never matched anything in the
      // feed (real title is 'GDP Growth Rate QoQ') — QoQ was silently
      // unreachable for CHF, not a deliberate YoY-first choice. Both real
      // titles now considered; YoY currently wins on freshness most
      // quarters (its release date consistently trails QoQ's by ~2 weeks
      // in this feed), same outcome as before but for the right reason.
      gdp:   ['GDP Growth Rate QoQ', 'GDP Growth Rate YoY'],
      cpi:   ['Inflation Rate YoY'],
      // v2.2.5: NOT a genuine gap \u2014 the prior "no MoM headline release in
      // current source" comment was never re-verified against live Myfxbook.
      // FSO publishes MoM alongside YoY every release, and a live page
      // exists (myfxbook.com/forex-economic-calendar/switzerland/
      // inflation-rate-mom). One-time backfilled via
      // backfill_supplementary_events.py v1.2 (FSO-cited); live pipeline
      // will pick up future releases automatically \u2014 canon()'s existing
      // "Switzerland " prefix stripping needs no new logic.
      cpimom:['Inflation Rate MoM'],
      // v2.2.6: no Myfxbook page exists for this (re-confirmed) \u2014 wired to a
      // new non-Myfxbook source instead of left blank. The vendor's
      // "Switzerland Core Inflation Rate" (FSO-sourced), fetched by
      // fetch_te_core_inflation.py v1.0. UNVALIDATED as of v2.2.6 \u2014 that
      // script's live guest:guest access was never confirmed to actually
      // return this indicator (see its header). If it never populates a
      // matching event, this cell simply stays blank, same as before \u2014 see
      // fetch_te_core_inflation.py's header before assuming it's broken.
      core:  ['Core Inflation Rate YoY'],
      // v2.2.3: NOT a genuine gap \u2014 same pipeline bug as GBP/JPY/CAD (v2.2.2)
      // above. Myfxbook's real title is "Producer & Import Prices YoY/MoM"
      // (live page confirmed: switzerland/producer-import-prices-yoy|mom) \u2014
      // this was already known in v2.2.2 while investigating the "ppi"
      // substring fix, but never got its own upgrade entry so it stayed
      // unreachable. Fixed upstream in fetch_ff_calendar.py v3.44 /
      // calendar-watcher.js v5.30 (new "producer & import prices" entry).
      // Relies on canon()'s existing "Switzerland " prefix stripping \u2014 no
      // new stripping logic needed.
      ppi:   ['Producer & Import Prices YoY', 'Producer & Import Prices MoM'],
      // v2.5.3: confirmed gap \u2014 no 'Employment Change'-equivalent title
      // found in calendar.json for CHF.
      emp:   [],
      unemp: ['Unemployment Rate'],
      prod:  ['Industrial Production YoY'],
      conf:  ['procure.ch Manufacturing PMI'],
      rtl:   ['Retail Sales YoY'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade'],
      pce:   [],
    },
    NZD: {
      gdp:   ['GDP Growth Rate QoQ'],
      cpi:   ['Inflation Rate QoQ'], // NZ publishes quarterly (not monthly/annual) CPI under this title \u2014 see subtext "QoQ" tag
      cpimom:[], // confirmed gap \u2014 NZ does not publish a monthly CPI
      // v2.2.6: no Myfxbook page exists for this (re-confirmed) \u2014 wired to
      // The vendor's "New Zealand Core Inflation Rate" (NZCIR,
      // RBNZ-sourced, ex-gasoline), via fetch_te_core_inflation.py v1.0.
      // \u26a0\ufe0f THIS IS NOT THE RBNZ SECTORAL FACTOR MODEL quoted in financial
      // press after each CPI release (that reading was 2.7% YoY Q2 2026) \u2014
      // The vendor's NZCIR is a different, older ex-fuel core measure (~3.2% YoY
      // Q4 2025 at time of writing). Do not relabel this as "Sectoral
      // Factor Model" anywhere \u2014 see fetch_te_core_inflation.py header for
      // the full explanation. UNVALIDATED as of v2.2.6 \u2014 live guest:guest
      // access for this indicator was never confirmed; if the fetcher
      // never populates a matching event this cell just stays blank.
      core:  ['Core Inflation Rate YoY'],
      // v2.2.4: NOT a genuine gap \u2014 live Myfxbook page confirmed at
      // myfxbook.com/forex-economic-calendar/new-zealand/ppi-output-qoq
      // (quarterly, Low impact, Source: Statistics New Zealand). The upstream
      // \"ppi\" substring in _IMPACT_UPGRADES (fetch_ff_calendar.py v3.43)
      // already covers this title and has since v3.43 shipped \u2014 this cell
      // was left blank purely because econ-matrix.js's own CATS list was
      // never updated to match, even after the upstream fix. A live Myfxbook
      // page for NZ also exists under \"PPI Input QoQ\" (what producers pay
      // for inputs) \u2014 deliberately NOT wired here: it is a distinct series
      // from Output PPI, not an alternate cadence of the same series (unlike
      // CHF's YoY/MoM pair), and every other currency's ppi column reports
      // the output/producer-price concept, not an input-cost index.
      ppi:   ['PPI Output QoQ'],
      // v2.5.3: NZ publishes employment quarterly, same cadence as its
      // GDP/CPI columns above \u2014 'Employment Change QoQ'.
      emp:   ['Employment Change QoQ'],
      unemp: ['Unemployment Rate'],
      // v2.5.0: was ['Manufacturing Sales YoY'], commented as "injected by
      // fetch_supplementary_indicators.py" \u2014 that script does not exist in
      // any repo (dead reference; the actual intended replacement,
      // fetch_ff_calendar.py's "manufacturing sales" _IMPACT_UPGRADES entry,
      // never once matched a live NZD event across a full year of
      // calendar.json, confirmed by audit). Repointed to the vendor's
      // "Industrial Production" series (fetch_te_nzd_ind_prod.py), which is
      // both live and the genuinely correctly-named series for this column
      // \u2014 not a proxy substitution like AUD/CAD's mappings. See
      // GUIDELINES.md v8.231.0 for the full incident.
      prod:  ['Industrial Production YoY'],
      conf:  ['Business NZ PMI'],
      rtl:   ['Retail Sales QoQ'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade'],
      pce:   [],
    },
    SEK: {
      gdp:   ['GDP Growth Rate QoQ'],
      cpi:   ['CPIF YoY'],
      cpimom:['CPIF MoM'],
      // CORRECTION (v2.2.7, 2026-08-14): v2.2.4's "genuine gap" note below
      // was checking the wrong source. Myfxbook indeed has no calendar page
      // for this series, but the vendor does \u2014 see the private script's
      // header for the URL, labelled "CPIF excl. Energy YoY" (the vendor's
      // own display name, not "Core Inflation Rate" like CHF/NZD's pages).
      // Confirmed server-rendered and live-scraped successfully the same
      // session this was caught \u2014 see fetch_te_core_inflation.py v3.0.
      // Original v2.2.4 note preserved below for the historical record of
      // what was actually checked (Myfxbook) and why it looked like a gap.
      //
      // v2.2.4 (superseded): full live Sweden calendar listing
      // (myfxbook.com/forex-economic-calendar/sweden, checked through its
      // Sep 2026 horizon) carries CPIF YoY/MoM (headline) and Inflation Rate
      // YoY/MoM but no separate \"CPIF Excluding Energy\" / core title under
      // any name on Myfxbook specifically.
      core:  ['Core Inflation Rate YoY'],
      ppi:   ['PPI YoY', 'PPI MoM'], // both published; YoY preferred per column policy
      // v2.5.3: confirmed gap, NOT wired to 'Employed Persons(<Mon>)' \u2014
      // that title is a raw LEVEL (headcount), not a change/rate figure, and
      // would silently misrepresent this column's stated semantics (net
      // jobs created). No genuine employment-CHANGE title found in the
      // current source for SEK.
      emp:   [],
      unemp: ['Unemployment Rate'],
      prod:  ['Industrial Production YoY'],
      conf:  ['Swedbank Manufacturing PMI'],
      rtl:   ['Retail Sales YoY'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade'],
      pce:   [],
    },
    NOK: {
      gdp:   ['GDP Growth Rate QoQ'],
      cpi:   ['Inflation Rate YoY'],
      cpimom:['Inflation Rate MoM'],
      core:  ['Core Inflation Rate YoY'],
      ppi:   ['PPI YoY'], // parenthetical-month title style (e.g. "PPI YoY(May)") \u2014 strictMatch already handles this
      // v2.5.3: confirmed gap, same reasoning as SEK above \u2014
      // 'Unemployed Persons(<Mon>)' in the current source is a raw LEVEL,
      // not a change figure, so it is not wired here.
      emp:   [],
      unemp: ['Unemployment Rate'],
      prod:  ['Manufacturing Production MoM'],
      conf:  ['Industrial Confidence'],
      rtl:   ['Retail Sales MoM'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade'],
      pce:   [],
    },
  };

  // EUR is matched WITHOUT country-prefix stripping, against explicit
  // "Euro Area " literal prefixes only \u2014 deliberate asymmetry vs. the other
  // nine currencies (see v2.0.0 note above: stripping "Euro Area " would let
  // member-state prints for Germany/France/Italy/etc. leak into the EA
  // aggregate column). conf is intentionally empty \u2014 see header comment on
  // the "Business Confidence" contamination finding.
  const CATS_EUR = {
    gdp:   ['Euro Area GDP Growth Rate QoQ'],
    cpi:   ['Euro Area Inflation Rate YoY'],
    cpimom:['Euro Area Inflation Rate MoM'],
    core:  ['Euro Area Core Inflation Rate YoY'],
    // v2.2.0: no genuine "Euro Area PPI" title exists in the current
    // source. The bare "PPI YoY" title was checked (not assumed) against
    // "Germany PPI YoY" — identical value on their one overlapping date
    // (2026-06-19, both 2.2%) confirms bare "PPI YoY" is actually GERMANY's
    // national PPI, not an EA aggregate, just without the country prefix
    // the vendor only started adding partway through the feed's history
    // (same drift pattern as Ifo above). Shown as Germany's national PPI
    // \u2014 same proxy pattern as conf above \u2014 rather than mislabeled as
    // an EA-wide figure.
    ppi:   ['Germany PPI YoY', 'PPI YoY'],
    // v2.5.3: QoQ listed first (tie-break priority only) to match this
    // block's own GDP-column convention of QoQ-first for EA headline flow
    // releases; YoY included as the same-day secondary candidate.
    emp:   ['Euro Area Employment Change QoQ', 'Euro Area Employment Change YoY'],
    unemp: ['Euro Area Unemployment Rate'],
    prod:  ['Euro Area Industrial Production MoM'],
    // v2.2.0: was previously an intentional gap (see v2.0.0 header note —
    // bare "Business Confidence" blended 3+ unlabeled national surveys with
    // no way to disambiguate). Re-investigated: the feed's more recent
    // entries (from ~Jun 2026) now carry country-prefixed titles
    // ("France Business Confidence", "Italy Business Confidence", "Germany
    // Ifo Business Climate", ...) that weren't present when v2.0.0 shipped
    // — confirmed via calendar.json this is a genuine vendor formatting
    // change, not new data. "Ifo Business Climate" (bare) and "Germany Ifo
    // Business Climate" share an identical value on their one overlapping
    // date (2026-06-19, both 2.2%) confirming they're the SAME continuous
    // series, just the country-prefix drift already documented for other
    // currencies. Germany's Ifo is used as the EUR proxy — the same pattern
    // as AUD's RBA Trimmed Mean CPI: a well-defined, single-attributable
    // national release standing in for a pan-EA business-confidence
    // aggregate the feed doesn't otherwise carry, chosen because Ifo is
    // itself the most widely cited Eurozone business-sentiment bellwether
    // in FX/macro coverage (Germany being the bloc's largest economy) —
    // not an arbitrary pick among the now-available national surveys.
    conf:  ['Germany Ifo Business Climate', 'Ifo Business Climate'],
    rtl:   ['Euro Area Retail Sales MoM'],
    ca:    ['Euro Area Current Account'],
    trade: ['Euro Area Balance of Trade'],
    pce:   [],
  };

  const GAP_TITLE = {
    prod: 'Not published as a standalone release in the current source for this currency',
    ca:   'Not currently tracked in the source feed for this currency',
    rtl:  'Not currently tracked in the source feed for this currency',
    core: 'No core/underlying inflation release in the current source for this currency',
    cpimom: 'No monthly headline CPI release in the current source for this currency',
    pce:  'PCE is a U.S.-specific series (the Fed\u2019s preferred inflation gauge) \u2014 not published for this economy. See the CPI / Core CPI columns for this currency\u2019s targeted measure.',
    ppi:  'No producer-price release in the current source for this currency',
  };

  // ── Period-label detection \u2014 for the per-cell reference-date subtext ──────
  // Purely a display convenience so YoY/QoQ/MoM/Annualized prints are never
  // visually ambiguous (e.g. NZD CPI is QoQ-only; several GDP prints are YoY
  // where a currency has no QoQ release). Detected from the event's own
  // title text, not asserted independently, so it can never drift out of
  // sync with what was actually matched.
  // v2.1.0: USD's "GDP Growth Rate QoQ" title is, by the BEA's own reporting
  // convention, already seasonally-adjusted ANNUALIZED (SAAR) — unlike every
  // other currency's "GDP Growth Rate QoQ", which is the raw, non-annualized
  // quarterly change. Tagging both identically as "QoQ" would silently make
  // US growth look ~4x stronger than an equivalent EA/UK/JPY print in the
  // same column. Flagged only for gdp/USD/qoq — every other cell's title
  // already carries an unambiguous, correctly-scaled period tag.
  function periodLabel(title, ccy, colKey) {
    const t = title.toLowerCase();
    if (colKey === 'gdp' && ccy === 'USD' && t.indexOf('qoq') !== -1) return 'QoQ SAAR';
    if (t.indexOf('qoq') !== -1) return 'QoQ';
    if (t.indexOf('yoy') !== -1) return 'YoY';
    if (t.indexOf('mom') !== -1) return 'MoM';
    if (t.indexOf('annualized') !== -1) return 'Annualized';
    if (t.indexOf('3-month avg') !== -1) return '3M Avg';
    return '';
  }

  // Formats a dateISO ('YYYY-MM-DD') as 'DD Mon' for the subtext line, or
  // extracts a parenthetical month/quarter tag ("(Apr)"/"(Q1)") from the
  // NOK/SEK title style when present, since that's the vendor's own stated
  // reference period and is more precise than the release/print date.
  function refLabel(ev) {
    const m = /\(([^)]+)\)\s*$/.exec(ev.event);
    if (m) return m[1];
    const d = new Date(ev.dateISO + 'T00:00:00Z');
    if (isNaN(d)) return ev.dateISO;
    return d.toLocaleDateString('en', { day: '2-digit', month: 'short', timeZone: 'UTC' });
  }

  // ── Value parsing \u2014 sign-aware, unit-agnostic ──────────────────────────────
  // Calendar "actual"/"previous" strings carry inconsistent prefixes (none,
  // "$", "CHF", "NZ$", "-SEK", ...) and suffixes ("%", "B"). We only need a
  // signed numeric value to compute trend direction for coloring \u2014 the cell
  // itself always displays the original string verbatim, so no precision is
  // invented and no unit conversion is attempted.
  function parseNum(s) {
    if (s == null) return null;
    const str = String(s).trim().replace(/,/g, '');
    const digitIdx = str.search(/\d/);
    if (digitIdx === -1) return null;
    const prefix = str.slice(0, digitIdx);
    const neg = prefix.indexOf('-') !== -1 || prefix.indexOf('(') !== -1;
    const m = str.slice(digitIdx).match(/\d+\.?\d*/);
    if (!m) return null;
    const v = parseFloat(m[0]);
    return neg ? -v : v;
  }

  // Same canonical list as calendar-panel.js's CAL_INVERSE_KW / dashboard.js's
  // INVERSE_KW / econ-surprises-modal.js's _ESM_INVERSE_KW — an indicator
  // whose title matches one of these reads "worse" when it goes up (rising
  // unemployment, more jobless claims, a wider deficit), so its up/down
  // coloring must be flipped relative to every other column. See v2.3.0
  // module header note for the audit of which of this file's columns are
  // actually affected (only Unemp, as of this file's current COLUMNS list).
  const MX_INVERSE_KW = ['unemployment', 'unemployed', 'jobless', 'claims', 'deficit'];

  function trendClass(actual, previous, eventTitle) {
    const a = parseNum(actual), p = parseNum(previous);
    if (a == null || p == null) return '';
    const titleLower = (eventTitle || '').toLowerCase();
    const inverse = MX_INVERSE_KW.some(kw => titleLower.indexOf(kw) !== -1);
    if (a > p) return inverse ? 'down' : 'up';
    if (a < p) return inverse ? 'up' : 'down';
    return 'flat';
  }

  // ── Build latest-actual index from calendar-data/calendar.json ────────────
  // Deterministic same-day tie-break: finds the single most-recent date on
  // which ANY listed prefix matches, then returns the match for the
  // FIRST prefix (in priority order) that hit on that date \u2014 so a same-day
  // clash between two different releases (e.g. AUD's index-level "CPI" vs.
  // its "Inflation Rate YoY" on the same print day) always resolves to the
  // category's documented priority, not iteration-order luck.
  // NOTE: a scheduled-but-not-yet-printed release carries `actual: null` in
  // the feed (e.g. the NZD Business NZ PMI for the day this file shipped —
  // dated, but not released yet). Both functions below only consider events
  // that have actually printed, so a pending release never masks the last
  // real reading — otherwise "latest by date" would return an empty cell for
  // a currency that in fact has a perfectly good recent print available.
  function findLatestGeneric(ccy, byCcy, prefixes) {
    if (!prefixes || !prefixes.length) return null;
    const list = byCcy[ccy];
    if (!list) return null;
    let bestDate = null;
    for (let i = 0; i < list.length; i++) {
      if (list[i].actual == null || list[i].actual === '') continue;
      const c = canon(ccy, list[i].event);
      for (let j = 0; j < prefixes.length; j++) {
        if (strictMatch(c, prefixes[j])) {
          if (bestDate === null || list[i].dateISO > bestDate) bestDate = list[i].dateISO;
          break;
        }
      }
    }
    if (bestDate === null) return null;
    for (let j = 0; j < prefixes.length; j++) {
      for (let i = 0; i < list.length; i++) {
        if (list[i].dateISO === bestDate && list[i].actual != null && list[i].actual !== '' &&
            strictMatch(canon(ccy, list[i].event), prefixes[j])) {
          return list[i];
        }
      }
    }
    return null;
  }

  function findLatestEUR(byCcy, prefixes) {
    if (!prefixes || !prefixes.length) return null;
    const list = byCcy.EUR;
    if (!list) return null;
    let bestDate = null;
    for (let i = 0; i < list.length; i++) {
      if (list[i].actual == null || list[i].actual === '') continue;
      for (let j = 0; j < prefixes.length; j++) {
        if (strictMatch(list[i].event, prefixes[j])) {
          if (bestDate === null || list[i].dateISO > bestDate) bestDate = list[i].dateISO;
          break;
        }
      }
    }
    if (bestDate === null) return null;
    for (let j = 0; j < prefixes.length; j++) {
      for (let i = 0; i < list.length; i++) {
        if (list[i].dateISO === bestDate && list[i].actual != null && list[i].actual !== '' &&
            strictMatch(list[i].event, prefixes[j])) {
          return list[i];
        }
      }
    }
    return null;
  }

  async function loadCalendarData() {
    const res = await fetch('./calendar-data/calendar.json', { cache: 'no-store' }).catch(() => null);
    if (!res || !res.ok) return null;
    const data = await res.json().catch(() => null);
    if (!data || !Array.isArray(data.events)) return null;

    const byCcy = {};
    data.events.forEach(ev => {
      if (!ev || !ev.currency || !ev.dateISO || !ev.event) return;
      if (!byCcy[ev.currency]) byCcy[ev.currency] = [];
      byCcy[ev.currency].push(ev);
    });
    Object.keys(byCcy).forEach(c => byCcy[c].sort((a, b) => a.dateISO < b.dateISO ? -1 : 1));

    const out = {};
    CCY_ORDER.forEach(ccy => {
      out[ccy] = {};
      const cats = ccy === 'EUR' ? CATS_EUR : (CATS[ccy] || {});
      COLUMNS.forEach(col => {
        const prefixes = cats[col.key];
        out[ccy][col.key] = ccy === 'EUR'
          ? findLatestEUR(byCcy, prefixes)
          : findLatestGeneric(ccy, byCcy, prefixes);
      });
    });
    return { byCategory: out, lastUpdate: data.lastUpdate || null };
  }

  // Formats a plain 'YYYY-MM-DD' (or any Date-parseable) string as 'DD Mon'
  // for the 10Y Yld / CB Rate subtext line \u2014 same 'DD Mon' shape as
  // refLabel() below uses for the calendar-driven columns, so all 16
  // columns share one visual subtext pattern. Unlike refLabel(), this has
  // no parenthetical-tag case to check (10y/CB rate dates never carry one)
  // and tolerates an already-short/unparseable string by returning it as-is
  // rather than showing nothing.
  function fmtDateShort(dateStr) {
    if (!dateStr) return '';
    const d = new Date(/^\d{4}-\d{2}-\d{2}$/.test(dateStr) ? dateStr + 'T00:00:00Z' : dateStr);
    if (isNaN(d)) return dateStr;
    return d.toLocaleDateString('en', { day: '2-digit', month: 'short', timeZone: 'UTC' });
  }

  // ── 10Y yield \u2014 extended-data/{CCY}.json, same field as yc-modal.js ────────
  // {cache:'no-store'} added in v2.5.0 to match loadCalendarData() below \u2014
  // bond10y is written daily (fetch_bond_yields.py) and this fetch is now
  // re-run every ECONMX_POLL_MS tick, so a stale HTTP-cached copy could
  // otherwise re-serve the exact same value the periodic refresh exists to
  // replace. See CHANGELOG.md v8.154.3.
  async function load10y(ccy) {
    const ext = await fetch('./extended-data/' + ccy + '.json', { cache: 'no-store' }).then(r => r.ok ? r.json() : null).catch(() => null);
    const v = ext && ext.data && ext.data.bond10y;
    if (v == null || isNaN(v)) return null;
    const date = (ext.dates && ext.dates.bond10y) || '';
    return { value: v, date };
  }

  // ── CB policy rate \u2014 reuse window._STATE_cbRates + computeCBTrend ─────────
  function waitForCBRates(timeoutMs) {
    return new Promise(resolve => {
      const start = Date.now();
      (function poll() {
        if (window._STATE_cbRates && Object.keys(window._STATE_cbRates).length) {
          resolve(window._STATE_cbRates);
        } else if (Date.now() - start > timeoutMs) {
          resolve(window._STATE_cbRates || null);
        } else {
          setTimeout(poll, 200);
        }
      }());
    });
  }

  function simpleTrend(obs) {
    if (!obs || obs.length < 2) return 'flat';
    const a = parseFloat(obs[0].value), b = parseFloat(obs[1].value);
    if (isNaN(a) || isNaN(b)) return 'flat';
    if (a > b) return 'up';
    if (a < b) return 'down';
    return 'flat';
  }

  // rates/{CCY}.json is a FRED-style MONTHLY series — every observation is
  // stamped to the 1st of its month by construction, never a real decision
  // date (confirmed: rates/USD.json's latest three rows are 2026-08-01 /
  // 07-01 / 06-01). meetings-data/meetings.json's per-currency `allMeetings`
  // carries the actual ISO decision dates instead; this resolves the most
  // recent one not after today, i.e. the last meeting that could plausibly
  // have set the currently-displayed rate. Reuses window._STATE_meetings
  // (populated by dashboard.js) the same way waitForCBRates() reuses
  // _STATE_cbRates, with an independent fetch fallback.
  function waitForMeetings(timeoutMs) {
    return new Promise(resolve => {
      const start = Date.now();
      (function poll() {
        if (window._STATE_meetings && window._STATE_meetings.meetings) {
          resolve(window._STATE_meetings.meetings);
        } else if (Date.now() - start > timeoutMs) {
          resolve((window._STATE_meetings && window._STATE_meetings.meetings) || null);
        } else {
          setTimeout(poll, 200);
        }
      }());
    });
  }

  async function lastMeetingDate(ccy) {
    let meetings = await waitForMeetings(1500);
    if (!meetings) {
      const data = await fetch('./meetings-data/meetings.json').then(r => r.ok ? r.json() : null).catch(() => null);
      meetings = data && data.meetings;
    }
    const rec = meetings && meetings[ccy];
    const all = rec && rec.allMeetings;
    if (!all || !all.length) return null;
    const todayISO = new Date().toISOString().slice(0, 10);
    const past = all.filter(d => d <= todayISO);
    if (!past.length) return null;
    return past[past.length - 1]; // allMeetings is chronological; last past entry = most recent
  }

  async function getCBRate(ccy) {
    const store = await waitForCBRates(3000);
    const rec = store && store[ccy.toLowerCase()];
    const meetingDate = await lastMeetingDate(ccy);
    if (rec && rec.rate != null) {
      const trend = (typeof window.computeCBTrend === 'function') ? window.computeCBTrend(rec.obs) : simpleTrend(rec.obs);
      return { rate: rec.rate, date: meetingDate || rec.date, trend };
    }
    // Fallback: independent fetch if STATE never populated (e.g. CB Rates
    // panel failed to load before this one came into view).
    const data = await fetch('./rates/' + ccy + '.json').then(r => r.ok ? r.json() : null).catch(() => null);
    const obs = data && data.observations;
    if (!obs || !obs.length) return null;
    const rate = parseFloat(obs[0].value);
    if (isNaN(rate)) return null;
    return { rate, date: meetingDate || obs[0].date, trend: simpleTrend(obs) };
  }

  // ── Render ──────────────────────────────────────────────────────────────────
  function cellHTML(ev, gapKey, ccy) {
    if (!ev) {
      const title = (gapKey && GAP_TITLE[gapKey]) || 'No data available';
      return '<td class="flat" title="' + title.replace(/"/g, '&quot;') + '">\u2014</td>';
    }
    const cls = trendClass(ev.actual, ev.previous, ev.event);
    const period = periodLabel(ev.event, ccy, gapKey);
    const ref = refLabel(ev);
    const sub = (period ? period + ' \u00b7 ' : '') + ref;
    const title = ev.event + ' \u00b7 ' + ev.dateISO + (ev.previous != null ? ' \u00b7 prev ' + ev.previous : '');
    return '<td' + (cls ? ' class="' + cls + '"' : '') + ' title="' + title.replace(/"/g, '&quot;') + '">' +
      '<div class="econmx-val">' + (ev.actual != null ? ev.actual : '\u2014') + '</div>' +
      '<div class="econmx-ref">' + sub + '</div>' +
      '</td>';
  }

  function rowHTML(ccy, calRow, y10, cb) {
    const flag = FLAG[ccy] ? '<span class="fi fi-' + FLAG[ccy] + '" style="margin-right:5px;border-radius:2px;"></span>' : '';
    let html = '<tr><td style="white-space:nowrap;">' + flag + '<span style="font-size:10px;">' + ccy + '</span></td>';
    COLUMNS.forEach(col => {
      html += cellHTML(calRow[col.key], col.key, ccy);
    });
    if (y10) {
      const y10ref = fmtDateShort(y10.date);
      const y10sub = y10ref || '\u2014';
      html += '<td class="flat" title="10Y \u00b7 as of ' + (y10.date || '\u2014') + '">' +
        '<div class="econmx-val">' + y10.value.toFixed(2) + '%</div>' +
        '<div class="econmx-ref">' + y10sub + '</div>' +
        '</td>';
    } else {
      html += '<td class="flat" title="No data available">\u2014</td>';
    }
    if (cb) {
      const cls = cb.trend === 'up' ? 'up' : cb.trend === 'down' ? 'down' : 'flat';
      const cbref = fmtDateShort(cb.date);
      const cbsub = cbref || '\u2014';
      html += '<td class="' + cls + '" title="CB policy rate \u00b7 as of ' + (cb.date || '\u2014') + '">' +
        '<div class="econmx-val">' + cb.rate.toFixed(2) + '%</div>' +
        '<div class="econmx-ref">' + cbsub + '</div>' +
        '</td>';
    } else {
      html += '<td class="flat" title="No data available">\u2014</td>';
    }
    html += '</tr>';
    return html;
  }

  // v2.4.0 \u2014 live polling. Previously loadEconMatrix() was gated by a
  // permanent `_loaded` flag, so the panel only ever fetched once (on first
  // scroll-into-view) and never again \u2014 a page reload was the only way to
  // see a new actual (see GUIDELINES.md / CHANGELOG.md v8.163.0 for the
  // incident this fixes). calendar.json itself now refreshes near-real-time
  // upstream (v8.162.0's repository_dispatch bridge, ~2\u20134 min end-to-end),
  // so the panel re-fetches it on the same cadence. 10Y yields and CB policy
  // v2.5.0: the periodic refresh now re-fetches calendar.json, 10Y yield
  // (extended-data/{CCY}.json \u2014 written daily, ten small files, cheap to
  // re-ask every tick) and recomputes CB Rate (no network cost \u2014 reads the
  // live window._STATE_cbRates that dashboard.js's own 5-min health.json
  // sentinel already keeps fresh) every tick, instead of freezing y10/cb at
  // whatever they were on first scroll-into-view. See CHANGELOG.md v8.154.3
  // for the live incident (stale AUD/CAD/NOK 10Y shown after a same-day
  // backend fix) this replaced the old "cache forever" comment/behavior for.
  const ECONMX_POLL_MS = 90 * 1000; // v2.5.1: 3min → 90s, matches calendar-panel.js's cadence

  let _loading = false;
  let _y10Cache = null;
  let _cbCache  = null;

  function renderMatrix(cal, y10All, cbAll) {
    const tbody = document.getElementById('econmx-tbody');
    const sub   = document.getElementById('econmx-sub');
    if (!cal) {
      if (sub) sub.textContent = 'Economic Calendar \u00b7 data unavailable';
      return;
    }

    const rows = CCY_ORDER.map((ccy, i) => rowHTML(ccy, cal.byCategory[ccy] || {}, y10All[i], cbAll[i]));
    if (tbody) tbody.innerHTML = rows.join('');

    if (sub) {
      let label = 'Economic Calendar \u00b7 latest actuals \u00b7 G10';
      if (cal.lastUpdate) {
        const d = new Date(cal.lastUpdate);
        if (!isNaN(d)) label += ' \u00b7 updated ' + d.toLocaleDateString('en', { day: '2-digit', month: 'short' });
      }
      sub.textContent = label;
    }
  }

  // Full load \u2014 fetches calendar.json + every 10Y/CB source. Runs once,
  // on first visibility, and caches y10/cb for subsequent light refreshes.
  async function loadEconMatrix() {
    if (_loading) return;
    _loading = true;
    const sub = document.getElementById('econmx-sub');
    try {
      const [cal, y10All, cbAll] = await Promise.all([
        loadCalendarData(),
        Promise.all(CCY_ORDER.map(load10y)),
        Promise.all(CCY_ORDER.map(getCBRate)),
      ]);
      _y10Cache = y10All;
      _cbCache  = cbAll;
      renderMatrix(cal, y10All, cbAll);
    } catch (e) {
      if (sub) sub.textContent = 'Economic Calendar \u00b7 data unavailable';
    } finally {
      _loading = false;
    }
  }

  // Light refresh \u2014 re-fetches calendar.json, 10Y yield, and CB Rate every
  // tick (v2.5.0; previously only calendar.json, leaving y10/cb frozen at
  // their first-load values for the rest of the session \u2014 see CHANGELOG.md
  // v8.154.3). Still skipped if the first full load hasn't completed yet, or
  // if a load is already in flight. On a failed re-fetch, falls back to the
  // last good y10/cb snapshot rather than blanking the panel.
  async function refreshPanel() {
    if (_loading || !_y10Cache || !_cbCache) return;
    _loading = true;
    try {
      const [cal, y10All, cbAll] = await Promise.all([
        loadCalendarData(),
        Promise.all(CCY_ORDER.map(load10y)),
        Promise.all(CCY_ORDER.map(getCBRate)),
      ]);
      if (y10All.some(v => v != null)) _y10Cache = y10All;
      if (cbAll.some(v => v != null)) _cbCache = cbAll;
      if (cal) renderMatrix(cal, _y10Cache, _cbCache);
    } catch (e) {
      // Silent \u2014 keep showing the last good render rather than blanking
      // a working panel over a transient background-refresh failure.
    } finally {
      _loading = false;
    }
  }

  function attach() {
    const section = document.getElementById('section-econmap');
    if (!section) return;

    function start() {
      loadEconMatrix();
      setInterval(refreshPanel, ECONMX_POLL_MS);
    }

    if (typeof IntersectionObserver === 'undefined') {
      start();
      return;
    }
    const io = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          start();
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: '150px' });
    io.observe(section);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach);
  } else {
    attach();
  }
}());
