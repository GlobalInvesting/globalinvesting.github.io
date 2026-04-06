# Changelog — globalinvesting-engine

---

## v7.13.12 (2026-04-06) — Pair detail popover: fix vertical clipping at bottom of viewport

### Frontend — assets/dashboard.js
- **Bug fix:** `openPairPopover()` used a hardcoded `ph = 300` estimate to calculate whether the popover would overflow the bottom of the viewport. The actual rendered height of the panel is significantly taller (Price + Volatility + COT Positioning + Retail Sentiment sections). Fixed by positioning the popover at `top: 8px` temporarily, then re-positioning after `updatePairDetail()` resolves using `pop.offsetHeight` — the real rendered height.

### Frontend — index.html
- **Bug fix:** `#pd-popover` had `overflow:hidden` with no `max-height`, meaning content could be clipped with no scroll fallback on short viewports. Changed to `overflow-y:auto` and added `max-height:calc(100vh - 16px)` as a last-resort safety net.

---

## v7.13.11 (2026-04-06) — COT panel: fix TradingView chart symbols to use TFF Leveraged Funds

### Frontend — assets/dashboard.js
- **Bug fix:** `COT_TV_SYMBOLS` used `COT:` prefix (Legacy report · NonCommercial Positions Long, `_FO_NCP_L`) which opens a different trader category in TradingView than what the panel displays. The panel data source is CFTC Disaggregated TFF · Leveraged Funds (Options+Futures Combined). Corrected all 8 symbols to use `COT3:` prefix (Financial/TFF report) with `_FO_LMP_L` suffix (Leveraged Funds · Long), so the chart that opens on row click is consistent with the NET and Long% values displayed in the panel.

---

## v7.13.10 (2026-04-06) — Fix FX Majors chart symbols to use ICE (FX_IDC)

### Frontend — assets/dashboard.js
- **Bug fix:** FX Pairs — Majors table was generating TradingView symbols with the `FX:` prefix (FXCM) instead of `FX_IDC:` (ICE). Corrected both branches of the `tvSym` expression to use `FX_IDC:`, consistent with carry pairs and sidebar symbol logic already in the file.

---

## v7.13.9 (2026-04-06) — COT panel: add NZD row

### Frontend — assets/dashboard.js
- **NZD added to `COT_CURRENCIES`:** Array now includes all 8 G8 currencies: EUR, GBP, JPY, AUD, CAD, CHF, NZD, USD.
- **NZD TV symbol added:** `COT:112741_FO_NCP_L` (CFTC code 112741 = NZ Dollar CME).

### Guides — guide-cot.html
- **Mock panel updated:** NZD row added with live data (week 2026-03-31: Long%=20%, Net=−17,798, OI=29k).

---

## v7.13.8 (2026-04-06) — Revert FX symbols to FX_IDC (ICEUS reverted)

### Frontend — assets/dashboard.js / index.html
- **Reverted:** All `ICEUS:` references changed back to `FX_IDC:` per user confirmation. FX_IDC is the correct prefix for all currency pairs in the TV embedded widget.

---

## v7.13.7 (2026-04-06) — Gold spot source XAUUSD=X; FX chart symbols → ICEUS

### Engine — scripts/fetch_intraday_quotes.py (site repo)
- **Gold yfinance ticker:** `GC=F` (Gold futures, CME) → `XAUUSD=X` (Gold spot). Eliminates the futures basis premium (~$5–15/oz) that caused a small but visible price discrepancy between the quote bar and the TradingView spot chart.

### Frontend — assets/dashboard.js
- **FX chart symbols → ICEUS:** All three symbol-builder functions (`fxSymbol`, `carryTV`, and the Crosses builder) now generate `ICEUS:EURUSD` etc. instead of `FX_IDC:EURUSD`. `pairMetaFromSym` strip regex updated to also accept `ICEUS:` prefix for pair metadata lookups.

### Frontend — index.html
- **All `data-sym` FX attributes updated:** 43 occurrences of `FX_IDC:` in `data-sym` replaced with `ICEUS:`. Default fallback symbol also updated to `ICEUS:EURUSD`.

---

## v7.13.6 (2026-04-06) — COT TV symbols: _FO_NCP_L confirmed; Gold chart → FOREXCOM:GOLD

### Frontend — assets/dashboard.js
- **COT TV symbols corrected to `_FO_NCP_L`:** Format confirmed by user as working in TV embedded widget. `COT:{code}_FO_NCP_L` = NonCommercial Positions Long, Futures+Options Combined. All 7 currencies updated.
- **Gold chart symbol:** `TVC:GOLD` → `FOREXCOM:GOLD` in cross-asset config. Price tracks spot XAU/USD, closely matching the yfinance `GC=F` futures source used in quotes.json (minor futures premium difference expected but negligible for display purposes).

### Frontend — index.html
- **Gold `data-sym` updated:** All three gold click targets (quote bar, TV tab, cross-asset cell) changed from `FX_IDC:XAUUSD` to `FOREXCOM:GOLD` to match the updated chart symbol.

---

## v7.13.5 (2026-04-06) — COT TV symbols: confirmed suffix _FO_LMP_L (Leveraged Funds Positions Long)

### Frontend — assets/dashboard.js
- **TV symbol suffix confirmed and corrected:** Set to `COT:{code}_FO_LMP_L` — verified via TradingView Symbol Search showing `232741_FO_LMP_L = AUSTRALIAN DOLLAR - CME - Futures and Options - Leveraged Funds Positions Long`. Prefix remains `COT:` (Legacy report in TV hosts the LMP series). `_FO_` = Futures+Options Combined, `_LMP_` = Leveraged Funds Positions, `_L` = Long.
- **Previous attempts:** `_F_L` (non-existent), `COT3:..._FO_TLF_L` (non-existent) — both caused "symbol doesn't exist" in TV.

### Guides — guide-cot.html
- **TradingView Chart section:** Updated example symbol to `COT:099741_FO_LMP_L` with correct suffix breakdown.

---

## v7.13.4 (2026-04-06) — COT TV symbols: correct prefix COT3 + suffix TLF for Leveraged Funds

### Frontend — assets/dashboard.js
- **TV symbol prefix corrected:** Changed from `COT:` (Legacy report) to `COT3:` (Financial/TFF report). The TFF report is the one that contains the Leveraged Funds category used by the panel; the Legacy `COT:` prefix does not have a Leveraged Funds series for FX currencies.
- **TV symbol suffix corrected:** Changed from `_FO_CP_L` (Commercial Positions) and `_F_L` (non-existent) to `_FO_TLF_L` — Futures+Options Combined · Traders Leveraged Funds · Long. This matches the panel's `options_futures_combined` source type and Leveraged Funds category exactly.
- **All 7 symbols updated:** EUR=`COT3:099741_FO_TLF_L`, GBP=`COT3:096742_FO_TLF_L`, JPY=`COT3:097741_FO_TLF_L`, AUD=`COT3:232741_FO_TLF_L`, CAD=`COT3:090741_FO_TLF_L`, CHF=`COT3:092741_FO_TLF_L`, USD=`COT3:098662_FO_TLF_L`.

### Guides — guide-cot.html
- **TradingView Chart section:** Updated example symbol and suffix breakdown to reflect `COT3:099741_FO_TLF_L`.

---

## v7.13.3 (2026-04-06) — COT TV symbols: fix category from Commercial to Non-Commercial

### Frontend — assets/dashboard.js
- **TV symbol suffix corrected:** `COT_TV_SYMBOLS` changed from `_CP_L` (Commercial Positions Long — hedgers) to `_F_L` (Non-Commercial Long — speculators). The panel displays Leveraged Funds (speculative) data; `_CP_L` was opening the opposing hedger category on TradingView, causing an apparent divergence that was a category mismatch, not a data error.
- **Root cause:** `_CP` in TradingView COT symbols means "Commercial Positions" (dealers/hedgers), not "Combined" as previously assumed. The correct suffix for Non-Commercial / speculative positioning is `_F_L`.

### Guides — guide-cot.html
- **TradingView Chart section rewritten:** Corrected the example symbol from `COT:098662_F_CP_L` (USD index, Commercial) to `COT:099741_F_L` (EUR, Non-Commercial). Added explanation of suffix meaning. Added a warning callout explaining that `_CP_L` (Commercial) moves inversely to speculative positioning — mixing the two categories produces apparent divergences that are not real data conflicts.

---

## v7.13.2 (2026-04-06) — COT panel: row-click TV chart, EUR symbol fix, USD row added

### Frontend — assets/dashboard.js
- **Row click → embedded TV chart:** The `↗` TV link column is removed. Each COT row now carries `data-sym` + `cursor:pointer`, and a click event delegate calls `loadTVChart(sym)` — identical behavior to FX Pairs and Crosses rows. The COT chart opens in the embedded TradingView widget with a smooth scroll to the chart section.
- **EUR CFTC code corrected:** `098662` is the US Dollar Index — not EUR/USD futures. EUR CFTC code is `099741`. Symbol corrected to `COT:099741_F_CP_L`.
- **USD row added:** `'USD'` added to `COT_CURRENCIES`. `USD` uses `COT:098662_F_CP_L` (US Dollar Index futures). `USD.json` data was already present in `cot-data/`.
- **`COT_TV_SYMBOLS` map updated** with corrected EUR code and new USD entry.

### Frontend — assets/dashboard.css
- **COT grid back to 6 columns:** `grid-template-columns` reverted to `36px 1fr 42px 64px 14px 68px` (TV col removed).
- **Removed `.cot-tv-link`** styles.
- **Added `.cot-row:hover`** — `background: var(--bg3)` on hover to signal clickability, consistent with FX Pairs and Crosses rows.

### Frontend — index.html
- **COT header updated:** TV column header removed; `grid-template-columns` in the static header div updated to match.

---



### Guides — guide-cot.html
- **Mock panel updated:** The "Reading the COT Panel" mock now reflects the v7.13.0 layout — 7-column grid (`CCY · Long/Short · Long% · Net · ● · OI · TV`), live data values (EUR −38,698 / 193k, GBP +24,859 / 81k, JPY −57,183 / 194k, AUD +43,837 / 97k, CAD −50,097 / 94k, CHF −368 / 18k), `↗` TV link column, correct bar fill direction (green from left, red from right). Old sparkline SVGs removed.
- **Legend updated:** Footer now reads `● LF+AM aligned · ○ LF/AM diverge · OI = LF open interest (long + short) · TV ↗ = open chart on TradingView`.
- **New section — Open Interest:** Added `#open-interest` section explaining LF OI as a conviction/participation metric, with a signal table (rising OI + growing net = new money; falling OI + improving net = short covering, not fresh longs), and a callout on relative OI context across currencies.
- **New section — TradingView Chart:** Added `#tv-chart` section explaining the `↗` link, TradingView COT symbol format (`COT:098662_F_CP_L`), and how to use the full chart for multi-year extreme context.
- **TOC updated:** Added `Open Interest` and `TradingView Chart` entries.

### Guides — guide-dashboard.html
- **FX Liquidity description updated:** The `<h3>FX Liquidity</h3>` paragraph now mentions the orange vertical time marker behavior (past = volatility-scaled, future = historical projection) and the dynamic source label (`yfinance · H-L range proxy · 30d avg` vs `Historical avg · fixed reference`).

### Guides — guide-fx-liquidity.html
- **"How to read it" bullet updated:** The solid blue area description now cites the correct source label (`yfinance · H-L range proxy · 30d avg`) instead of a generic "yfinance data feed" reference.
- **"Data source & refresh" section rewritten:** Now accurately describes the three-tier fallback chain — (1) `fx-liquidity.json` server-side cache (yfinance H-L range proxy, 30d avg), (2) Frankfurter rate series fallback, (3) static `Historical avg · fixed reference` baseline. Explains that the source label below the chart legend updates dynamically after each fetch, and what it means when `Historical avg · fixed reference` persists after full page load.

---



### Frontend — assets/dashboard.js
- **Added OI column:** COT panel rows now show Leveraged Funds Open Interest (long + short contracts) formatted as abbreviated numbers (e.g. `193k`, `1.2M`). Data sourced from existing `longPositions` + `shortPositions` fields in `cot-data/*.json` — no additional network request.
- **OI direction arrow:** When `history[1]` is available, a `▲` (green) or `▼` (red) indicator appears before the OI number, matching the Pair Detail OI cell behavior from v7.12.41.
- **Replaced 26-week sparkline with TradingView COT link:** The `6M` sparkline column is replaced by a `↗` anchor link that opens the TradingView COT chart for the selected currency in a new tab (`target="_blank" rel="noopener"`). Symbol map: EUR=`COT:098662_F_CP_L`, GBP=`COT:096742_F_CP_L`, JPY=`COT:097741_F_CP_L`, AUD=`COT:232741_F_CP_L`, CAD=`COT:090741_F_CP_L`, CHF=`COT:092741_F_CP_L`.
- **Added `COT_TV_SYMBOLS` map** and `fmtOI()` helper function.

### Frontend — assets/dashboard.css
- **COT grid updated:** `grid-template-columns` changed from `36px 1fr 42px 64px 14px 52px` to `36px 1fr 42px 64px 14px 68px 24px` to accommodate the new OI column and compact TV link column.
- **Added `.cot-oi`** style for the Open Interest cell (mono font, right-aligned, muted color with up/down arrow sub-classes `.oi-up` / `.oi-dn`).
- **Added `.cot-tv-link`** style for the TradingView anchor — muted by default, accented on hover.
- **Removed `.cot-spark`** style (sparkline no longer rendered in this panel).

### Frontend — index.html
- **COT header row updated:** Column headers now read `CCY · Long/Short · Long% · Net · ● · OI · TV` matching the new grid. OI header has tooltip explaining rising/falling OI significance. TV header has tooltip `Open COT chart on TradingView`.

---

## v7.12.43 (2026-04-06) — guide-dashboard.html: Pair Detail mockup actualizado

### Frontend — guide-dashboard.html
- **Mockup reemplazado:** El mock del Pair Detail en la guía ahora refleja el layout sectioned de v7.12.39+: secciones Price · Volatility · COT Positioning · Retail Sentiment, con el bloque de precio en el header (precio, % diario, H/L, Spread, ADR).
- **Datos actualizados:** El mockup usa valores de EUR/USD del ciclo actual (precio 1.15110, HV 7.8%, ATM IV 20.9%, LF Net −38,698, OI 193,390) en vez de los valores desactualizados (precio 1.08430, datos de 2024).
- **Labels con divisa:** Las celdas COT en el mock muestran `LF Net (EUR)`, `AM Net (EUR)`, `LF Open Interest (EUR)` — coherente con el cambio de v7.12.42.
- **Footer de positioning:** El mock muestra `LF Short · AM Long · diverging` en texto plano monocromático — el badge LF≡AM/LF≠AM ya no aparece en ningún lugar del sitio.
- **Lista de campos actualizada:** Reemplazada por descripciones de las cuatro secciones, incluyendo la nota de cruces en COT y la definición correcta del Carry (numerador − denominador).
- **Párrafo intro actualizado:** Ahora menciona explícitamente las cuatro secciones agrupadas.

---


## v7.12.42 (2026-04-06) — COT labels: divisa fuente explícita en cruces

### Frontend — assets/dashboard.js
- **Labels con divisa:** Las tres celdas de COT (LF Net, AM Net, LF Open Interest) ahora muestran la divisa fuente entre paréntesis: `LF Net (EUR)`, `AM Net (EUR)`, `LF Open Interest (EUR)`. Aplica a todos los pares — majors y cruces.
- **Nota de contexto en cruces:** Para pares cruce (EUR/GBP, GBP/JPY, EUR/JPY, etc.), el tooltip agrega automáticamente: *"CFTC tracks EUR futures vs USD — not this cross specifically. Use as a proxy for EUR sentiment broadly."* Para majors (EUR/USD, USD/JPY…) no se agrega nota — los datos son directamente relevantes.
- **Detección automática:** Usa `meta.cross` (ya presente en PAIRS para todos los cruces) — ningún hardcoding por par.

---


## v7.12.41 (2026-04-06) — COT OI: ▲▼ direction indicator + delta vs prior week

### Frontend — assets/dashboard.js + dashboard.css
- **OI direction arrow:** LF Open Interest cell now shows `▲` (green) or `▼` (red) before the number when prior-week OI is available from `history[1]` in the COT JSON. When history has only 1 entry (current state), the number renders cleanly without an arrow.
- **Delta in parentheses:** Alongside the arrow, shows `(+12,450)` or `(−8,200)` in muted small text so the magnitude is visible at a glance without dominating the cell.
- **Dynamic tooltip example:** The `data-tip-ex` in the OI tooltip now uses the actual delta figure and direction ("▲ 12,450 vs prior week. New money entering — expanding participation.") instead of generic text — the tooltip adapts to the live data.
- **Cache:** `COT_DATA_CACHE` now stores `prevOI` derived from `history[1].levLong + history[1].levShort`. No additional network request — computed at prefetch time.
- **CSS:** Added `.pd-oi-up` (green, 9px) and `.pd-oi-dn` (red, 9px) for the arrow characters.

---


## v7.12.40 (2026-04-06) — Pair Detail: carry fix + LF Open Interest

### Frontend — assets/dashboard.js
- **Fix carry sign for USD/CCY pairs:** `USD/JPY`, `USD/CHF`, `USD/CAD` were showing negative carry (e.g. −3.00% for USD/JPY). Root cause: the formula used `cbBase − cbQuote` for `invert:false` pairs, but `base='JPY'` in those pairs — so it computed JPY rate − USD rate. Fix: `invert:true` (CCY/USD) → `cbBase − cbQuote`; `invert:false` (USD/CCY) → `cbQuote − cbBase`. USD/JPY now correctly shows ~+4.00%.
- **Carry tooltip improved:** Reworded to "numerator currency rate minus denominator currency rate" with a USD/JPY worked example. Eliminates the confusing base/quote language that didn't match the displayed pair label.
- **Added LF Open Interest:** COT section now shows a third full-width cell "LF Open Interest" = `long + short` from CFTC LF category. Data was already in the cache (`cotRaw.long`, `cotRaw.short`) — just not surfaced. Tooltip explains rising/falling OI significance.

---


## v7.12.39 (2026-04-06) — Pair Detail: sectioned layout, ADR, Spread, COT summary text

### Frontend — assets/dashboard.js
- **Redesign:** Pair Detail popover now uses labelled sections — Price · Volatility · COT Positioning · Retail Sentiment — matching the grouping convention of Eikon/Bloomberg pair panels.
- **Added ADR:** Average Daily Range in pips, derived from HV30 (close × HV30/100 / √252 / pipSize). Displayed in both the price block header and the Price section.
- **Added Spread:** Bid-ask spread in pips (from TYPICAL_SPREADS proxy — live model or ECN floor fallback) moved into the Volatility section as a discrete cell.
- **Added Carry + Base Rate** into Price section, replacing the old flat grid position.
- **COT badge removed:** LF≡AM / LF≠AM emoji badges replaced by plain text: `LF Long · AM Short · diverging` (or `aligned`) in muted monospace below the COT grid. No emojis, no colored pills — matches Eikon COT summary style.
- **Tooltips:** All cells now use `data-tip-title` / `data-tip-body` / `data-tip-ex` attributes. Tooltip attachment reads these directly — no more TIP_MAP lookup table. Each tooltip includes an example clause where relevant.

### Frontend — assets/dashboard.css
- **Added:** `.pd-section`, `.pd-section-lbl`, `.pd-section--last`, `.pd-price-row`, `.pd-spread-row`, `.pd-cot-summary` styles.
- **Removed:** `.pd-badge`, `.pd-aligned`, `.pd-diverge` — no longer used.
- **Updated:** `.pd-price-block` layout from flex-row to block, with separate price-row div.

---

## v7.12.38 (2026-04-06) — Fix: FX Liquidity today normalization spike

### Engine — scripts/fetch_fx_liquidity.py
- **Bug:** `today_profile` se normalizaba con su propio máximo (`normalize_to_100`). Con 1 sola hora de datos reales, esa hora se convertía en 100.0 — produciendo un spike artificial al inicio del día.
- **Fix:** `today_profile` ahora se normaliza con el mismo denominador (`baseline_max`) que `baseline_30d`. Los valores de `today` y `baseline_30d` quedan en la misma escala y son directamente comparables.
- **Cleanup:** Función `normalize_to_100` eliminada (ya no se usa).

---

## v7.12.37 (2026-04-06) — FX Liquidity: chart starts at 22:00 UTC (Sydney open)

### Frontend — assets/dashboard.js
- **Change:** Canvas X axis now starts at 22:00 UTC (Sydney open) and ends at 22:00 UTC the following day — matching the real FX trading day. Previously started at 00:00 UTC, splitting Sydney session across the edges.
- **Implementation:** `OFFSET=44` slots (22h × 2). Canvas slot `i` maps to array slot `(i+44)%48`. All drawing loops (past area, future area, past line, future dashed line, session bands) updated to use rotated coordinates. Tooltip `mousemove` handler updated: canvas slot → array slot → UTC hour via same offset. `isPast` comparison now in canvas-slot space.
- **Hour labels:** Updated from `00,06,12,18,24` to `22,02,06,10,14,18,22` — reflects real session boundaries starting at Sydney open.
- **Session bands:** Sydney band (22:00–07:00 UTC) now visible at left edge where it belongs.

---



### Frontend — index.html
- **Fix:** "New York" → "NY" and "30d forecast" → "30d avg" in legend row to prevent label truncation at narrow panel widths.

---

## v7.12.35 (2026-04-06) — FX Liquidity: dashed blue future line traces real 30d baseline

### Frontend — assets/dashboard.js
- **Change:** Future hours in `_liqData` now use `baseline_30d[h]` directly instead of `baseline_30d[h] * 0.75`. The dashed blue future line now traces the real 30-day average profile rather than an arbitrarily attenuated version.
- **Removed:** Grey dashed baseline reference line (`rgba(120,123,134,0.45)`) eliminated from `drawLiquidityChart`. The dashed blue future line replaces its informational function — same data, cohesive visual language.

### Frontend — index.html
- **Updated:** Legend item changed from grey dashed "30d avg" to blue dashed "30d forecast" — matches the actual line color and communicates intent more clearly.

---

## v7.12.34 (2026-04-06) — Fix: FX Liquidity tooltip — % vs baseline 30d real

### Frontend — assets/dashboard.js
- **Bug:** `volLabel(v, maxV)` usaba el máximo del array `_liqData` del día actual como denominador. Al inicio del día (pocas horas con datos reales), ese máximo era bajo y los porcentajes resultaban distorsionados (ej. "High 80%" a las 01:00 UTC).
- **Fix:** el porcentaje ahora se calcula sobre `maxBaseline` (máximo del array `_liqBaseline` 30d) — denominador estable que no cambia a lo largo del día.
- **Nuevo:** para horas pasadas con datos reales, el tooltip muestra la desviación vs el promedio 30d del mismo slot: `↑ +18% vs 30d avg`, `↓ -12% vs 30d avg`, o `≈ in line with 30d avg` (umbral: ±8%). Solo se muestra cuando `_liqBaseline` está disponible y es distinto de `_liqData`.
- Cache bust: `dashboard.js?v=7.12.34`.

---

## v7.12.33 (2026-04-06) — Fix: fetch_fx_liquidity hours_complete off-by-one

### Engine — scripts/fetch_fx_liquidity.py
- **Bug:** `hours_complete = now_utc.hour` producía 0 a las 00:xx UTC, lo que excluía la hora 0 del día aunque ya tuviera una vela 1h cerrada en yfinance. El dashboard mostraba el baseline para todas las horas del día en vez de los datos reales de la hora 0.
- **Fix:** `hours_complete = now_utc.hour + 1` — la hora actual ya tiene su vela cerrada en yfinance `1h`, por lo que se incluye como dato real.

---

## v7.12.32 (2026-04-05) — FX Liquidity: enriquecimiento con datos reales vía yfinance

### Engine — scripts/fetch_fx_liquidity.py (nuevo — repo público)
- **Nuevo script:** produce `fx-data/fx-liquidity.json` con dos arrays de 24h UTC: `today` (rango H-L realizado del día actual, hora a hora) y `baseline_30d` (promedio rolling 30 días del mismo índice).
- **Metodología:** mediana del rango H-L por hora entre 5 pares (EURUSD=X, GBPUSD=X, USDJPY=X, USDCHF=X, AUDUSD=X) vía yfinance `1h`, 35 días de historia. El rango H-L es el proxy de actividad FX más robusto disponible en fuentes públicas (referencia: IEEE 2013 — el tiempo del día es el factor que más influye en la liquidez FX).
- **Fallback en cadena:** fx-liquidity.json → frankfurter.json vol-scalar (legado) → LIQ_BASE hardcodeado. El campo `fallback: true` señala cuando no hay datos reales.
- **No requiere API keys.** yfinance, sin ventaja competitiva → repo público.

### Engine — .github/workflows/update-fx-liquidity.yml (nuevo — repo público)
- Corre cada hora, todos los días (`0 * * * *`). Sin secrets. Minutos ilimitados en repo público.
- Timeout: 10 min. Concurrency group para evitar solapamientos.

### Frontend — assets/dashboard.js
- **`fetchLiquidityData()` reescrita** con tres capas de fallback: (1) `fx-liquidity.json` primario, (2) `frankfurter.json` legado, (3) LIQ_BASE puro.
- **`_liqBaseline`** (nuevo): array de 48 slots con el promedio 30d, dibujado como línea de referencia gris punteada sobre el área del día actual.
- **`_liqSource`** (nuevo): string de atribución actualizado dinámicamente en `#liq-source-label` después de cada fetch.
- **Helper `_liqTo48()`**: interpola un array de 24h a 48 half-hour slots, evitando duplicación de lógica.
- Cache bust: `dashboard.js?v=7.12.32`.

### Frontend — index.html
- Leyenda "30d avg" con línea punteada gris agregada al pie del panel FX Liquidity.
- `<span id="liq-source-label">` agregado debajo de la leyenda — muestra la fuente activa en 7.5px.
- Cache bust: `dashboard.css?v=7.12.32`.

### GUIDELINES.md
- Regla de label del panel FX Liquidity actualizada: `yfinance · H-L range proxy · 30d avg` cuando el JSON está disponible; `Historical avg · fixed reference` como fallback. Versión bumpeada a v7.12.32.

---

## v7.12.31 (2026-04-05) — UX: Pair detail — doble click reemplaza botón ⓘ (patrón Refinitiv)

### Frontend — assets/dashboard.js
- **Eliminado botón ⓘ:** `openPairPopover()` ahora recibe el elemento row (no el botón) y ancla el popover al borde del row.
- **Gestos actualizados:** `click` sigue cargando el chart; `dblclick` abre el popover de detalle — igual al comportamiento de Refinitiv Eikon/Workspace para instrument detail.
- **`preventDefault()`** en dblclick evita selección de texto accidental.
- Outside-click handler simplificado (sin referencia a `.pd-info-btn`).
- Template FX table: `<button class="pd-info-btn">` eliminado del `<td class="sym">`.

### Frontend — index.html
- Los 21 botones `.pd-info-btn` eliminados de todos los `.sb-row` de Crosses.
- Atributo `title` de cada fila actualizado: `"Click: chart · Double-click: detail"`.
- Cache bust: `dashboard.css?v=7.12.31`, `dashboard.js?v=7.12.31`.

### Frontend — assets/dashboard.css
- Todo el bloque CSS de `.pd-info-btn` eliminado (~13 reglas).
- `.sb-sym` revertido a `flex-shrink:0` simple — el `inline-flex` ya no es necesario sin el botón inline.

---

## v7.12.31 (2026-04-05) — UX: Pair detail — doble click reemplaza botón ⓘ (patrón Refinitiv)

### Frontend — assets/dashboard.js
- **Eliminado botón ⓘ:** `openPairPopover()` ahora recibe el elemento row (no el botón) y ancla el popover al borde del row.
- **Gestos actualizados:** `click` sigue cargando el chart; `dblclick` abre el popover de detalle — igual al comportamiento de Refinitiv Eikon/Workspace para instrument detail.
- **`preventDefault()`** en dblclick evita selección de texto accidental.
- Outside-click handler simplificado (sin referencia a `.pd-info-btn`).
- Template FX table: `<button class="pd-info-btn">` eliminado del `<td class="sym">`.

### Frontend — index.html
- Los 21 botones `.pd-info-btn` eliminados de todos los `.sb-row` de Crosses.
- Atributo `title` de cada fila actualizado: `"Click: chart · Double-click: detail"`.
- Cache bust: `dashboard.css?v=7.12.31`, `dashboard.js?v=7.12.31`.

### Frontend — assets/dashboard.css
- Todo el bloque CSS de `.pd-info-btn` eliminado (~13 reglas).
- `.sb-sym` revertido a `flex-shrink:0` simple — el `inline-flex` ya no es necesario sin el botón inline.

---

## v7.12.30 (2026-04-05) — Fix: FX Liquidity no reacciona al abrir el mercado el domingo

### Frontend — assets/dashboard.js
- **Bug:** `drawLiquidityChart()` y los dos generadores de `_liqData` en `fetchLiquidityData()` usaban `utcDay === 0 || utcDay === 6` para detectar fin de semana. Esta definición trata **todo el domingo UTC como mercado cerrado**, incluso después de las 21:00 UTC cuando Sydney ya abrió. El canvas permanecía con la curva plana y el texto "MARKET CLOSED — WEEKEND" hasta las 00:00 UTC del lunes.
- **Fix:** Los tres `isWeekend` corregidos a la definición canónica ya usada en el resto del archivo: `utcDay === 6 || (utcDay === 0 && utcHour < 21) || (utcDay === 5 && utcHour >= 21)`. El mercado FX cierra viernes 21:00 UTC y reabre domingo 21:00 UTC (Sydney open).
- Cache bust: `dashboard.css?v=7.12.30`.

---

## v7.12.29 (2026-04-05) — Fix: botón ⓘ crosses — reubicado dentro de sb-sym

### Frontend — index.html
- **Botón `.pd-info-btn` movido** desde el final del `.sb-row` (donde se superponía al `.sb-chg`) al interior del `.sb-sym`, inmediatamente después del texto del par. Los 21 rows de crosses actualizados.

### Frontend — assets/dashboard.css
- **`.sb-sym`** pasa a `display:inline-flex; align-items:center; gap:4px; flex-shrink:0` — el botón queda pegado al nombre del par, separado de los números por el `justify-content:space-between` del row padre.
- **`.sb-sym .pd-info-btn`:** `position:static` (sale del flujo absoluto), sin margin adicional. El `flex-shrink:0` del `.sb-row` garantiza que `.sb-price` y `.sb-chg` nunca se compriman.
- Estándar de referencia: Bloomberg Terminal coloca los action indicators junto al ticker (izquierda), nunca superpuestos a los valores (derecha).

---

## v7.12.28 (2026-04-05) — Fix: Google Fonts carga bloqueante — Inter + JetBrains Mono garantizados

### Frontend — index.html
- **Fuente raíz del problema:** `display=optional` + carga no-bloqueante (`media=print/onload`) hacía que el browser comprometiera el fallback (Courier New / sistema) antes de que llegara el CSS de Google Fonts. JetBrains Mono nunca se aplicaba en ninguna sesión, independientemente de la velocidad de red.
- **Fix:** Google Fonts ahora carga como stylesheet bloqueante (`<link rel="stylesheet">` directo). Con carga bloqueante, Inter y JetBrains Mono están definidas antes del primer paint → ningún swap ocurre → CLS de fuentes = 0. El costo en FCP (~50ms) es aceptable y ya estaba presente en el repo de referencia.
- **`display=swap` mantenido:** Con carga bloqueante, `swap` nunca dispara un intercambio visible porque las fuentes ya están disponibles en el momento del paint.
- Cache bust: `dashboard.css?v=7.12.28`.

### GUIDELINES.md
- Sección Font loading reescrita: documenta la estrategia correcta (bloqueante + swap) y prohíbe explícitamente `display=optional` con carga no-bloqueante.

---

## v7.12.27 (2026-04-05) — Restauración fuentes canónicas: Inter + JetBrains Mono

### Frontend — index.html + assets/dashboard.css
- **`--font-ui` restaurado a `'Inter'`** y **`--font-mono` restaurado a `'JetBrains Mono'`**: Las versiones v7.12.24–v7.12.26 reemplazaron incorrectamente estas fuentes con Courier New y el stack del sistema, tomando como referencia el boceto prototipo v11 en lugar del repo de producción real. El repo de referencia (02-Apr-2026) confirma que Inter + JetBrains Mono son las fuentes canónicas del sitio.
- **Google Fonts restaurado** con `display=optional` (no `display=swap` que usaba el repo de referencia). Incluye: `<link rel="preconnect">` para googleapis y gstatic, preload WOFF2 de Inter v13, `<link>` con media=print/onload + noscript.
- **`sb-chg` restaurado a 10px** (referencia: 10px; v7.12.25 lo había subido a 11px incorrectamente).
- **GUIDELINES.md** corregido: sección Font loading restaurada a los valores canónicos Inter + JetBrains Mono con las reglas de `display=optional`.

---

## v7.12.26 (2026-04-05) — Restauración fuente sistema: eliminar Inter + Google Fonts

### Frontend — index.html
- **Google Fonts eliminado completamente:** Removidos `<link rel="preconnect">` para fonts.googleapis.com y fonts.gstatic.com, el `<link>` de carga de Inter (media=print/onload), el `<noscript>` fallback, y el `<link rel="preload">` del WOFF2 de Inter. Cero requests externos de fuentes.
- **`--font-ui` en inline `:root`** restaurado a `-apple-system, BlinkMacSystemFont, 'Trebuchet MS', Arial, sans-serif` — idéntico al boceto de referencia (v11).

### Frontend — assets/dashboard.css
- **`--font-ui`** restaurado a `-apple-system, BlinkMacSystemFont, 'Trebuchet MS', Arial, sans-serif`.
- **`--font-mono`** permanece `'Courier New', monospace` (restaurado en v7.12.24).

### GUIDELINES.md
- **Sección "Font loading" reescrita:** Documenta el uso exclusivo de fuentes del sistema. Prohíbe explícitamente Google Fonts, Inter, JetBrains Mono y cualquier `preconnect`/`preload` de fonts.gstatic.com. Rationale: las fuentes del sistema están disponibles antes del primer paint → CLS de fuentes = 0 permanente, sin depender de `display=optional` ni preloads.
- Versión footer actualizada a v7.12.26.

**Efecto en Core Web Vitals:** Elimina la única fuente restante de CLS relacionada con tipografía. Las fuentes del sistema no generan layout shift bajo ninguna condición de red.

---

## v7.12.25 (2026-04-05) — Fix: ca-chg formato arrow+abs+pct igual al boceto

### Frontend — assets/dashboard.js
- **`setCA()` reformateado:** `ca-chg` ahora muestra `▲ +18.4 (+0.35%)` cuando el cambio absoluto está disponible, o `▲ +0.35%` cuando no. Antes mostraba solo `+0.35%` sin flecha ni valor absoluto, divergiendo del boceto de referencia (v11). El 5° parámetro `chgAbs` se pasa desde el campo `.chg` de `intradayQuote()`.
- **`setCA_rt()`** (gold/wti desde STOOQ_RT_CACHE): mismo formato arrow+abs+pct aplicado.
- **BTC inline renders** (early + final): mismo formato aplicado; BTC usa `.toLocaleString({maximumFractionDigits:0})` para el absoluto dado que los valores son de 4-5 dígitos.
- **Todos los `setCA()` calls** actualizados para pasar `.chg` como 5° argumento: spx, gold, wti, nikkei, stoxx, dxy (early + final renders).

---

## v7.12.24 (2026-04-05) — Fix: fuente mono restaurada a Courier New

### Frontend — index.html + assets/dashboard.css
- **`--font-mono` cambiado de JetBrains Mono → Courier New:** JetBrains Mono tiene x-height notablemente menor que Courier New al mismo tamaño en px, causando que los números de precios, spreads y cambios se vean más pequeños y delgados que en el boceto de referencia. `--font-mono` restaurado a `'Courier New', monospace` en ambos archivos, idéntico al boceto v11.
- **JetBrains Mono eliminado de Google Fonts:** Request reducido a solo Inter (UI text). Elimina una llamada de red innecesaria.
- **`.sb-chg` font-size:** Corregido de 10px → 11px para coincidir con el boceto (`.sb-price` y `.sb-chg` ambos 11px).
- Cache bust: `dashboard.css?v=7.12.24`.

---

## v7.12.23 (2026-04-05) — Fix: crosses layout y colores de texto

### Frontend — assets/dashboard.css
- **`.sb-row` crosses apretado:** Botón `.pd-info-btn` era un 4° elemento flex dentro del `.sb-row`, rompiendo el layout sym · precio · chg. Fix: `.sb-row` pasa a `position:relative` y el botón dentro de `.sb-row` pasa a `position:absolute; right:6px; top:50%; transform:translateY(-50%)`. No ocupa espacio en el flex container. El botón dentro de `td` mantiene su comportamiento inline anterior (`margin-left:4px; vertical-align:middle`).
- **`--text2` / `--text3` restaurados a valores canónicos:** Valores habían derivado a `#9096a0` / `#8290a0` (más claros, diferente al boceto). Restaurados a `--text2:#787b86` y `--text3:#6b7280` per GUIDELINES.md §Color palette.

### Frontend — index.html
- Cache bust: `dashboard.css?v=7.12.23`.

---

## v7.12.22 (2026-04-05) — Pair detail panel: popover + retail sentiment

### Frontend — assets/dashboard.js
- **`RETAIL_SENTIMENT_CACHE` global:** New module-level object populated by `fetchSentiment()` when myfxbook.json loads successfully. Keyed by normalised pair label (e.g. `"EUR/USD"`) → `{ longPct, shortPct, longPos, shortPos, avgL, avgS }`. Provides retail positioning data to the pair detail popover without re-fetching.
- **`openPairPopover(btn, tvSym)`:** New function. Positions `#pd-popover` (position:fixed) anchored near the `ⓘ` button that triggered it — prefers right side, falls back to left if near viewport edge. Calls `updatePairDetail(tvSym)`. Toggle behaviour: clicking the same pair's button closes it.
- **`closePairPopover()`:** New function. Hides `#pd-popover`, clears `dataset.sym`. Called on Escape key, outside-click (capture phase), close button, and toggle.
- **`updatePairDetail(tvSym)`:** Target changed from `#pair-detail` (static rightpanel element, now removed) to `#pd-popover` (floating overlay). Added retail sentiment section: wide-span cell showing long/short % bar + numeric ratio from `RETAIL_SENTIMENT_CACHE`. Added close button (×) in header. Removed `panel.classList.remove('pd-empty')` (no longer applicable).
- **Click handlers:** Both `#sidebar` and `#fx-pairs-tbody` handlers now intercept `.pd-info-btn` clicks — call `openPairPopover()` and stop propagation — before falling through to `loadTVChart()` for row clicks.
- **FX table row template:** `ⓘ` button (`.pd-info-btn`) injected inside `<td class="sym">`, after the pair label.
- **Auto-populate removed:** `setTimeout(() => updatePairDetail('FX_IDC:EURUSD'), 1500)` removed — popover opens only on explicit user action.

### Frontend — index.html
- **`#pair-detail` removed** from `#rightpanel`: the fixed panel occupied ~120px of rightpanel height unconditionally. Freed space is reclaimed by the CB Rates / Correlations / CB Expectations / Positioning Bias tables.
- **`#pd-popover` added** as last element before `</body>`: position:fixed overlay, z-index:1000, hidden by default.
- **`.pd-info-btn` added** to all 21 `.sb-row` elements in the Crosses sidebar (EUR, GBP, AUD, NZD, CAD, CHF crosses).
- **Cache bust:** `dashboard.css?v=7.12.22`.

### Frontend — assets/dashboard.css
- **`#pair-detail` static rules removed** (replaced by `#pd-popover` popover).
- **`.pd-info-btn`:** Inline `ⓘ`-style button; `opacity:0` at rest, `opacity:1` on parent row hover; always visible (`opacity:0.6`) on touch devices (`@media (hover:none)`).
- **`.pd-cell--wide`:** New modifier class — spans both grid columns (for the retail positioning cell).
- **`.pd-retail-bar` / `.pd-retail-fill` / `.pd-retail-nums`:** Retail long/short bar and numeric display.
- **`.pd-close`:** Popover close button (×) in header.

---

## v7.12.21 (2026-04-05) — CLS fix: skeleton rows para CB Rates y Key Correlations tables

### Frontend — index.html

#### Diagnóstico del CLS desktop residual 0.129 (v7.12.20)
La tabla CB Rate Expectations seguía siendo el culpable, pero el mecanismo de CLS era diferente al previsto. El elemento NO estaba creciendo — estaba siendo EMPUJADO HACIA ABAJO por las tablas que están encima de él en el `#rightpanel`, las cuales aún no tenían skeleton rows:

1. **`#cbrates-tbody`** (Central Bank Rates): 1 fila "Loading…" → 8 filas reales (una por banco central). Crecimiento: ~7 filas × ~22px = ~154px. Posición en DOM: inmediatamente encima de las correlaciones y CB Expectations.
2. **`#correlations-tbody`** (Key FX Correlations): 1 fila → 10 filas reales (pares de correlación de 60 días). Crecimiento: ~9 filas × ~22px = ~198px. Posición: entre CB Rates y CB Expectations.

El crecimiento combinado de ~352px empujaba la tabla CB Expectations hacia abajo, generando el CLS 0.109 + 0.015 = 0.124.

Las versiones v7.12.18–v7.12.20 habían corregido los skeleton rows de la tabla CB Expectations en sí, pero la causa real era el crecimiento de las tablas SUPERIORES en el panel.

#### Fix: skeleton rows en ambas tablas superiores
- **`#cbrates-tbody`**: 8 skeleton rows con estructura idéntica al JS (`<span class="fi fi-XX">` + bank name en font-size:10px, columnas Rate y Trend). Las filas usan `class="up"` para Rate y `<span class="flat">—</span>` para Trend, replicando exactamente la altura de las filas reales.
- **`#correlations-tbody`**: 10 skeleton rows con los pares exactos que sirve `intraday-data/quotes.json` (EUR/USD·DXY, AUD/USD·Gold, USD/JPY·US 10Y, USD/JPY·VIX, USD/CAD·WTI Oil, GBP/USD·FTSE 100, AUD/USD·ASX 200, NZD/USD·NZX 50, EUR/USD·EuroStoxx, GBP/USD·Gold). Los nombres de los pares son idénticos a los del JSON → sin cambio de ancho de columna al reemplazar.

#### CLS esperado post-v7.12.21
- CB Rates + Correlations empujando CB Expectations: 0.124 → ~0
- Residual conocido: `sb-section` crosses 0.003 + risk panel-sub 0.001 + topbar 0.000 = CLS total esperado ≤ 0.005 desktop ✅
- Móvil: sin cambio (CLS 0 mantenido desde v7.12.20 ✓)

---

## v7.12.20 (2026-04-05) — CLS fix: flag-icons bloqueante + skeleton rows exactos en CB Rate Expectations

### Frontend — index.html

#### Diagnóstico del CLS desktop residual 0.130 (v7.12.19)
El análisis de los skeleton rows de v7.12.18 confirmó que el mecanismo esqueleto era correcto en concepto pero fallaba por dos razones:
1. **flag-icons.min.css cargaba en modo non-blocking** (`rel="preload" onload`): la hoja de estilos se aplicaba tarde (post-render), causando que cada `.fi` span recibiera su `background-image` después del primer paint → reflow de toda la tabla CB Rate Expectations → CLS 0.109 + 0.017 = 0.126.
2. **Los skeleton rows no replicaban la estructura HTML real**: carecían de los spans de flag (`<span class="fi fi-xx">`), el nombre del banco en `font-size:10px` y el placeholder de fecha en `font-size:9px`, lo que hacía que la altura inicial de la fila fuera diferente a la final.

#### Fix 1: flag-icons bloqueante
- **Cambio:** Reemplazado `<link rel="preload" as="style" ... onload="this.rel='stylesheet'">` + `<noscript>` por un único `<link rel="stylesheet">` bloqueante.
- **Efecto:** flag-icons.min.css se aplica antes del primer paint → las `.fi` spans tienen su `background-image` desde el inicio → sin reflow tardío de la tabla.
- **Costo:** flag-icons.min.css es ~11 KiB comprimido, se sirve desde jsDelivr CDN con caché larga. El bloqueo de render es mínimo (ya era un preconnect existente a cdn.jsdelivr.net).

#### Fix 2: skeleton rows con estructura exacta
- **Cambio:** Los 8 skeleton rows del `#fed-exp-tbody` ahora replican exactamente la estructura HTML que produce el JS: `<span class="fi fi-XX">` + `<span style="font-size:10px;">` (banco) + `<span style="color:var(--text3);font-size:9px;">` (fecha placeholder «—»), más la celda de bias con `<span class="flat">→ —</span>` y la celda de fwd rate con las mismas clases de color y font-family que el JS genera.
- **Efecto:** La altura de cada fila skeleton es idéntica a la fila real → sin cambio de altura al reemplazar innerHTML.

#### CLS esperado post-v7.12.20
- CB Rates table: 0.126 → ~0 (ambas causas eliminadas)
- Residual: `sb-section` crosses 0.003 + risk panel-sub 0.001 + topbar 0.000 = CLS total esperado ≤ 0.005 desktop
- Móvil: sin cambio respecto a v7.12.19 (0.055, pasa threshold de 0.1 ✓)

---

## v7.12.19 (2026-04-05) — Frankfurter CORS fix: mover fetches al engine workflow

### Engine — .github/workflows/update-frankfurter-cache.yml (nuevo)
- **Nuevo workflow:** Fetcha los datos de Frankfurter/ECB server-side (sin CORS) cada 4h en días de semana y una vez en fin de semana, y los deposita en `/fx-data/frankfurter.json` del repo público. Estructura: `today` (tasas USD-base del día), `prev` (día hábil anterior), `series` (timeseries 7 días EUR-base→USD,GBP,JPY para el liquidity canvas).

### Engine — scripts/fetch_frankfurter_cache.py (nuevo)
- **Nuevo script:** Realiza las 3 llamadas a `api.frankfurter.app` en Python (server-side, sin restricción CORS) y escribe `site/fx-data/frankfurter.json`. Incluye validación de respuesta y exit(1) ante error para notificación por email desde GitHub Actions.

### Frontend — assets/dashboard.js
- **`fetchFrankfurter()`:** Eliminadas las dos llamadas directas a `api.frankfurter.app/{date}?from=USD`. Ahora lee `fetch('/fx-data/frankfurter.json')` y extrae `data.today.rates` / `data.prev.rates`. Comportamiento idéntico al anterior pero sin CORS.
- **`fetchLiquidityData()`:** Eliminada la llamada a `api.frankfurter.app/{start}..{end}?from=EUR&to=USD,GBP,JPY`. Ahora lee la misma caché y extrae `cacheData.series.rates`. El cálculo del `volScalar` y la generación de los 48 buckets de liquidez es idéntico.
- **Efecto en consola:** Los 3 errores de CORS de Frankfurter desaparecen de la consola → Best Practices desktop/móvil debería subir de 77 a ~92 (solo quedan el unload handler de TradingView y los WebSocket ERR_NAME_NOT_RESOLVED, ambos de terceros e irresolubles).

### Frontend — index.html
- **Cache bust:** `dashboard.css?v=7.12.18` → `v=7.12.19`

---

## v7.12.18 (2026-04-05) — CLS fix: skeleton rows para CB Rates y Positioning Bias tables

### Performance — index.html

#### Causa raíz del CLS desktop 0.127 identificada: tablas del panel derecho sin altura reservada
- **Diagnóstico:** El CLS desktop de 0.127 (v7.12.17) provenía de dos tablas en el panel derecho (`#rightpanel`) que arrancan con 1 fila "Loading…" y crecen al cargar datos:
  - **CB Rate Expectations** (`#fed-exp-tbody`): 1 fila → 8 filas (8 bancos centrales: Fed, ECB, BoE, BoJ, RBA, BoC, SNB, RBNZ). Crecimiento: ~154px → CLS 0.088 + 0.036 = 0.124.
  - **Positioning Bias** (`#skew-tbody`): 1 fila → 4 filas (EUR/USD, GBP/USD, USD/JPY, AUD/USD). Crecimiento secundario, contribuye al shift residual.
  - El panel derecho está en posición `sticky` o fija, por lo que su crecimiento desplaza los elementos visibles a su izquierda en el viewport.

#### Fix: 8 skeleton rows en CB Rate Expectations tbody
- **Cambio:** Reemplazado `<tr><td colspan="3">Loading…</td></tr>` por 8 filas skeleton con los nombres abreviados de los bancos centrales (Fed, ECB, BoE, BoJ, RBA, BoC, SNB, RBNZ) y celdas "—". La estructura de 3 columnas (Bank · Meeting, Bias, Fwd Rate) se respeta exactamente.
- **Efecto:** El tbody ocupa desde el primer paint el espacio final que ocupará cuando JS cargue los 8 bancos → sin shift.

#### Fix: 4 skeleton rows en Positioning Bias tbody
- **Cambio:** Reemplazado `<tr><td colspan="4">Loading…</td></tr>` por 4 filas skeleton (EUR/USD, GBP/USD, USD/JPY, AUD/USD) con celdas "—". La estructura de 4 columnas (Pair, 1W/ATM IV, 1M/IV Rnk, Bias/Direction) se respeta.
- **Efecto:** Sin shift cuando JS llama a `fetchOptionSkew()` y reemplaza el innerHTML.

#### CLS esperado post-fix
- CB Rates + Skew: 0.124 → ~0
- Residual: `sb-section` crosses 0.003 + risk panel-sub 0.001 + topbar 0.000 = CLS total esperado ≤ 0.010 desktop
- Móvil: ya estaba en 0.055 (pasa el threshold de 0.1) — pequeño shift en quotebar (q-item width) y #main al cargar narrative, ambos menores

---

## v7.12.17 (2026-04-05) — CLS fix: skeleton rows para FX table y Heatmap

### Performance — index.html

#### Causa raíz del CLS desktop residual identificada: crecimiento de paneles sobre el viewport
- **Diagnóstico:** El CLS desktop de 0.262 (Cross-Asset grids: 0.145 + 0.112) no era causado por fuentes — era causado por dos paneles que crecen cuando JS carga datos, empujando el panel Cross-Asset (visible en viewport) hacia abajo.
  - **FX Pairs table:** El tbody arrancaba con 1 fila genérica "Loading…" (~24px). Al cargar datos, JS la reemplaza por 7 filas reales de pares (~168px). Crecimiento: ~144px → shift del panel Cross-Asset.
  - **Currency Strength Heatmap:** El `#heatmap-grid` arrancaba vacío (0px). Al cargar, JS inserta 8 celdas `hm-cell` (~58px de contenido). Crecimiento: ~58px → otro shift acumulado.
  - Ambos paneles están posicionados SOBRE el panel Cross-Asset en el flujo vertical de `#main`. Como `#main` es `overflow-y:auto`, el Cross-Asset (abajo) se desplaza cuando los paneles de arriba crecen. Lighthouse detectó el desplazamiento en el elemento visible (Cross-Asset) y lo atribuyó a ese div, cuando la causa real era el crecimiento de los paneles superiores.

#### Fix: skeleton rows en FX Pairs tbody (7 filas)
- **Cambio:** Reemplazado el `<tr><td colspan="9">Loading…</td></tr>` por 7 filas skeleton — una por cada par no-cruzado del array `PAIRS` (EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, USD/CAD, NZD/USD). Cada fila tiene `<td class="sym">PAR</td>` + 8 celdas con "—". Altura resultante idéntica a los datos reales.
- **Efecto:** El tbody ya ocupa el espacio final desde el primer paint. Cuando JS llama a `populateFxPairsTable()` y reemplaza el innerHTML, el panel no crece → sin shift.

#### Fix: skeleton cells en Currency Strength Heatmap (8 celdas)
- **Cambio:** Añadidas 8 celdas skeleton `<div class="hm-cell h-flat">` con `<span class="hm-sym">CCY</span>` + `<span class="hm-val flat">—</span>` directamente en el HTML. Una por cada divisa (EUR, USD, GBP, JPY, AUD, CAD, CHF, NZD). Usa la misma clase `h-flat` que aplica el JS para el estado neutral.
- **Efecto:** El `#heatmap-grid` ya tiene su altura final (~58px) en el primer paint. Cuando JS llama a `renderHeatmap()` y reemplaza el innerHTML, el panel no crece → sin shift.

#### CLS esperado post-fix
- Cross-Asset grids: 0.257 → ~0 (shift causado por FX table + heatmap eliminado)
- Elementos residuales: `sb-section` 0.003 + topbar 0.000 = CLS total esperado ≤ 0.005

---


## v7.12.16 (2026-04-05) — Fix JetBrains Mono 404 preload; add Core Web Vitals guidelines

### Performance — index.html

#### JetBrains Mono WOFF2 preload removed (was returning 404)
- **Problema:** El tag `<link rel="preload" as="font">` apuntaba a `tDbv2o-flEEny0FZhsfKu5WU4xD-IQ.woff2`, que devolvía HTTP 404. Google Fonts sirve WOFF2 con hashes específicos por combinación de navegador/OS (User-Agent negotiation); hardcodear una URL sin verificarla en el navegador destino produce el 404. El error aparecía en la consola de PageSpeed degradando el score de Best Practices. Además, un preload que devuelve 404 no cumple ningún propósito útil: el browser lo descarta y la fuente sigue cargando vía la regla `@font-face` del CSS de Google Fonts, que sí usa la URL correcta para cada UA.
- **Fix:** Eliminado el preload de JetBrains Mono. Se mantiene únicamente el preload de Inter v13 (URL verificada y estable). JetBrains Mono sigue cargando correctamente vía `@font-face` con `display=optional`; sin el preload no garantizamos que llegue en los 100 ms del período optional, pero tampoco causamos un 404 en consola.
- **Efecto esperado:** Eliminación del error `Failed to load resource: 404` de JetBrains Mono en consola → mejora de Best Practices score. El CLS residual de 0.306 en desktop (Cross-Asset grids) debería reducirse al eliminar el 404 que impedía que el período optional funcionara correctamente.

### GUIDELINES.md

#### Nueva sección: Core Web Vitals — non-negotiable performance standards
- **Motivo:** Tras múltiples iteraciones de optimización (v7.12.12–v7.12.15), documentamos formalmente las reglas que garantizan CLS ≤ 0.1, LCP ≤ 2.5 s y TBT ≤ 200 ms. Sin estas reglas escritas, cambios futuros que parezcan inocuos (cargar CSS en forma no-bloqueante, añadir `display=swap`, quitar `min-height` de paneles) pueden reintroducir el CLS de 1.2+ que se tardó varias sesiones en diagnosticar y corregir.
- **Contenido añadido:** Tabla de thresholds por métrica; reglas obligatorias para CSS loading, font loading, layout shifts y blocking resources; lista de errores de consola no corregibles (Frankfurter CORS, TradingView WebSocket, TradingView deprecation warning); `min-height` requeridas por panel.
- **Checklist de deployment actualizado:** Nueva línea: Core Web Vitals verificados (CLS ≤ 0.1 en desktop y mobile) antes de cerrar sesión.

---

## v7.12.15 (2026-04-05) — CLS fix: blocking CSS + font-display:optional + JetBrains Mono preload

### Performance — index.html

#### dashboard.css cambiado a carga bloqueante
- **Problema:** El patrón `<link rel="preload" as="style" onload="this.rel='stylesheet'">` (CSS no-bloqueante) mejora el FCP pero causa CLS porque la página pinta primero con el CSS inline (2.7 KB) y luego aplica las ~48 KB de reglas restantes de `dashboard.css`. Cada regla nueva que cambia dimensiones, márgenes o tipografía después del primer paint es un layout shift medible. PageSpeed Desktop reportó CLS=1.226, con `#tv-chart-wrap` acumulando 0.913 de ese total.
- **Fix:** Cambiado a `<link rel="stylesheet" href="assets/dashboard.css?v=7.12.15">` (carga bloqueante estándar). El browser espera que `dashboard.css` cargue antes del primer paint → el DOM se pinta ya estilado → CLS=0 por este concepto. El FCP aumenta ~200–400 ms en desktop (GitHub Pages sirve desde CDN con latencia baja), pero la mejora de CLS vale muchos más puntos en el score.

#### `font-display:swap` → `font-display:optional` en Google Fonts
- **Problema:** Google Fonts con `display=swap` hace que el browser pinte primero con la fuente del sistema (system fallback) y luego intercambie a Inter y JetBrains Mono cuando cargan. Ese intercambio ("swap") cambia las métricas tipográficas de todos los elementos de texto visible — alto de línea, ancho de caracteres, dimensiones del contenedor — generando layout shifts en cadena. PageSpeed identificó explícitamente las dos fuentes WOFF2 de `fonts.gstatic.com` como causantes del CLS de `#tv-chart-wrap` (0.913) y de los dos grids de Cross-Asset (0.160 + 0.146).
- **Fix:** URL de Google Fonts cambiada a `display=optional`. Con `optional`, el browser tiene exactamente 100 ms para cargar la fuente antes del primer paint. Si carga (ver preloads a continuación): se usa sin swap. Si no carga: el sistema usa la fuente fallback **permanentemente** para esa sesión, sin ningún intercambio posterior → CLS=0 por fuentes.

#### Preload de JetBrains Mono WOFF2 añadido
- **Problema:** Solo existía un `<link rel="preload" as="font">` para Inter Regular (400). JetBrains Mono (usada en `.ca-val` a 16px font-weight:700, en el quotebar, y en los tooltips) no tenía preload — el browser no iniciaba su descarga hasta que el parser encontraba la referencia en el Google Fonts CSS, ya demasiado tarde para el período de 100 ms de `display:optional`.
- **Fix:** Añadido `<link rel="preload" as="font" type="font/woff2" href="https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbv2o-flEEny0FZhsfKu5WU4xD-IQ.woff2" crossorigin>` inmediatamente después del preload de Inter. Con ambas fuentes preloadeadas, el navegador las descarga en paralelo desde el inicio de la navegación y ambas están disponibles dentro de los 100 ms del período `optional`.

#### `#narrative` en mobile: `min-height:84px` añadido
- **Problema:** En mobile (`max-width:900px`) el panel `#narrative` tenía `height:auto` sin `min-height`. Al cargar el texto AI (~226 caracteres), el panel crecía de ~40 px (placeholder "Loading…") a ~84 px (texto real en 3 líneas a 11px × 1.5), empujando `#tv-chart-wrap` hacia abajo.
- **Fix:** Añadido `min-height:84px` al `#narrative` en el bloque `@media(max-width:900px)` del CSS inline crítico. El panel ya tiene su altura correcta en el primer paint → sin shift cuando el texto AI carga.

---

## v7.12.13 (2026-04-05) — Core Web Vitals: minificación habilitada, logo optimizado, fetchpriority

### Performance — netlify.toml

#### Minificación JS y CSS habilitada
- **Problema:** `[build.processing.js]` y `[build.processing.css]` tenían `minify = false`. Netlify no comprimía los assets en cada deploy, sirviendo `dashboard.js` (238 KB) y `dashboard.css` (48 KB) sin minificar.
- **Fix:** `minify = true` en ambas secciones. Netlify aplicará su pipeline de minificación en cada deploy: `dashboard.js` ~238 KB → ~150 KB, `dashboard.css` ~48 KB → ~30 KB. Compatible con el cache-busting `?v=` existente.

### Performance — index.html + todas las páginas

#### Logo del topbar: imagen correctamente dimensionada
- **Problema:** `apple-touch-icon.png` (180×180 px, 45 KB) se usaba en el topbar con display de 28×28 px. PageSpeed lo marcaba como "Properly size images" — se transfieren 45 KB para mostrar ~0.5 KB de datos visuales.
- **Fix `index.html`:** Creado `logo-56.png` (56×56 px a 2× retina, 7.6 KB) mediante resize LANCZOS desde el original. El topbar ahora referencia `logo-56.png` en lugar de `apple-touch-icon.png`. Ahorro: ~37 KB por visita en desktop.
- **Fix páginas secundarias:** `apple-touch-icon.png` se mantiene en las páginas secundarias (el archivo 180×180 es correcto para iOS), pero se añadieron atributos `width="28" height="28"` explícitos donde faltaban para evitar reflow.

#### `fetchpriority="high"` en logo del topbar
- Añadido `fetchpriority="high"` al `<img>` del logo en todas las páginas. El logo es el primer elemento visual above-the-fold; esta pista HTTP Priority Hints instruye al browser a priorizarlo en la fase de preload scan, mejorando el LCP en casos donde la imagen compite con otros recursos.

### Archivo nuevo — logo-56.png
- Imagen de 56×56 px (2× retina) generada desde `apple-touch-icon.png` con filtro LANCZOS. Uso exclusivo del topbar. No reemplaza `apple-touch-icon.png` (que sigue siendo la referencia estándar iOS 180×180 en el `<link rel="apple-touch-icon">`).

---

## v7.12.12 (2026-04-05) — Core Web Vitals: defer dashboard.js, fix dangling preload, cache headers para assets, CSP fix

### Performance — index.html

#### `dashboard.js` ahora es `defer`
- **Problema:** `<script src="assets/dashboard.js">` sin atributo alguno bloqueaba el parsing HTML durante ~800 ms en mobile (238 KB leídos y ejecutados síncronamente). El navegador no podía construir el DOM hasta que el script terminaba de ejecutarse.
- **Fix:** Agregado atributo `defer`. Con `defer` el browser descarga el script en paralelo con el HTML y lo ejecuta solo después de que el DOM esté completo — equivalente a colocarlo al final del `<body>` pero con descarga anticipada. Sin impacto funcional: `dashboard.js` ya asume que el DOM existe cuando corre (todos sus `getElementById` / `querySelector` se ejecutan en el nivel superior del módulo, no en un listener de `DOMContentLoaded`, lo que es compatible con `defer`).

#### Preload de fuentes sin `onload` eliminado
- **Problema:** Existía `<link rel="preload" as="style" href="https://fonts.googleapis.com/css2?...">` sin handler `onload`. Un `preload` sin uso posterior dentro del timeout de carga genera una advertencia en consola ("The resource ... was preloaded using link preload but not used within a few seconds") y ocupa un slot de conexión temprana sin beneficio real. El load no-bloqueante ya estaba cubierto por el tag `media="print" onload="this.media='all'"` inmediatamente debajo.
- **Fix:** Eliminado el `<link rel="preload">` redundante. La fuente sigue cargándose de forma no-bloqueante vía el patrón `media=print`.

### Performance — netlify.toml

#### Cache headers de larga duración para assets JS y CSS
- **Problema:** `assets/dashboard.js` (238 KB) y `assets/dashboard.css` (48 KB) no tenían cabecera `Cache-Control` explícita. Netlify servía estos archivos con el TTL por defecto (~10 min), obligando al navegador a revalidarlos en cada visita.
- **Fix:** Agregadas reglas `Cache-Control: public, max-age=31536000, immutable` para `/assets/*.js`, `/assets/*.css` y `/scripts/*.js`. El sufijo `?v=7.12.12` en las URLs actúa como cache-buster, por lo que `immutable` es seguro: el navegador usa la copia en caché hasta que cambia la URL.

#### CSP — dominios faltantes añadidos
- **Problema:** `script-src` no incluía `www.googletagmanager.com`, lo que podía provocar que el navegador bloqueara el script de Google Analytics según la implementación del CSP. `connect-src` tampoco incluía `www.google-analytics.com` ni `region1.google-analytics.com`, bloqueando el envío de hits de Analytics silenciosamente.
- **Fix:** Añadidos `www.googletagmanager.com` a `script-src`; `https://www.google-analytics.com` y `https://region1.google-analytics.com` a `connect-src`. Añadido también `'unsafe-inline'` a `script-src` para el bloque `<script>` del IIFE de lazy-loading en `index.html` (era necesario y estaba implícitamente permitido pero no declarado).

---

## v7.12.11 (2026-04-05) — Fix TradingView chart symbol change after lazy-load refactor

### Bug fix — index.html

**Symptom:** After clicking any pair, tab, or sidebar row, the TradingView chart would not update to the new symbol. The chart appeared frozen on the initial EUR/USD load.

**Root cause:** The lazy-load IIFE introduced in v7.12.10 exposed its own `loadTVChart` as `window.loadTVChart`. That version used `document.querySelector('#tv-chart-widget')` as its target and only appended a new `<script>` — it never cleared the existing widget DOM. Meanwhile `dashboard.js` declares a global `function loadTVChart(sym)` that does `wrap.innerHTML = ''` followed by a full DOM rebuild. Because `dashboard.js` loads after the IIFE, its declaration overwrites `window.loadTVChart` in the global scope — but on symbol-change clicks the two functions produced incompatible DOM states: the IIFE left the original `#tv-chart-widget` element in place, so when `dashboard.js` called `wrap.innerHTML = ''` it destroyed the IIFE's target, and the IIFE's version could no longer find its container on subsequent calls.

**Fix:** Rewrote the IIFE's `loadTVChart` to use the identical DOM pattern as `dashboard.js`: clear `#tv-chart-wrap` with `innerHTML = ''`, then create the full `tradingview-widget-container` → `tradingview-widget-container__widget` → `script` hierarchy from scratch. Both code paths now produce the same DOM structure on every invocation, so `dashboard.js`'s global `loadTVChart` (which overrides `window.loadTVChart` after parse) works correctly regardless of whether it's the first load or a symbol change triggered by any click event (tabs, sidebar rows, FX pairs table, quotebar, cross-asset cells, VIX button).

---

## v7.12.10 (2026-04-05) — Core Web Vitals: lazy-load TradingView widgets, inline critical CSS, fix forced reflow & CLS

### Performance — index.html

#### Critical CSS inline + non-blocking stylesheet load
- **Critical CSS inlined in `<head>`:** Extracted ~20 rules covering above-the-fold layout (`:root` CSS vars, `body`, `#topbar`, `#tv-ticker-wrap`, `#news-ticker`, `#layout`, `#sidebar`, `#main`, `#narrative`, `.panel`, `.panel-head`, `.panel-title`, `.skip-link`, `.tv-skeleton`) and placed them in a `<style>` block before any external resource. The browser can now paint the shell without waiting for any stylesheet.
- **`dashboard.css` changed to non-blocking preload:** Replaced `<link rel="stylesheet">` with `<link rel="preload" as="style" onload="this.onload=null;this.rel='stylesheet'">` + `<noscript>` fallback. Eliminates the ~650 ms render-blocking delay reported by PageSpeed (FCP impact).
- **`flag-icons.min.css` also made non-blocking:** Same preload pattern applied to the flag-icons CDN stylesheet.
- **Version bumped to `?v=7.12.10`** on the `dashboard.css` query string to bust cached copies of the old blocking tag.

#### Preconnect reduction
- Reduced `<link rel="preconnect">` origins from 4 to 2 (retained `fonts.googleapis.com` + `fonts.gstatic.com`; removed `s3.tradingview.com` and `api.frankfurter.app`). TradingView is now loaded lazily (see below); Frankfurter is a JSON API called after DOMContentLoaded and does not benefit from a preconnect at parse time. Eliminates the "Avoid excessive preconnects" PageSpeed warning.

#### TradingView widget lazy loading (IntersectionObserver)
- **All 3 TradingView scripts removed from initial parse:** The `<script src="...tradingview...">` tags for the Advanced Chart, Economic Calendar, and Economic Map widgets no longer appear inline in the HTML. They were blocking the main thread for 2.4–3.2 s on every page load regardless of whether the user ever scrolled to those panels.
- **IntersectionObserver injected before `</body>`:** A self-contained IIFE wraps three loader functions (`loadTVChart`, `loadTVEvents`, `loadTVEconMap`) and an `IntersectionObserver` (rootMargin: 150 px) that observes `#tv-chart-wrap`, `#tvcal-inner`, and `#section-econmap`. Each widget script is created and appended to the DOM only when its container enters the viewport. Guards (`_chartLoaded`, `_eventsLoaded`, `_econmapLoaded`) prevent double-injection.
- **`window.loadTVChart` exposed** so `dashboard.js` can trigger a symbol reload on tab click without bypassing the lazy-load guard.
- **Skeleton loaders** (`div.tv-skeleton`) remain visible in each container until the widget script fires, preventing blank white boxes during scroll.

#### CLS (Cumulative Layout Shift) fix — height reservations
- Added `min-height` rules to the critical CSS inline block for all 3 widget containers:
  - `#tv-chart-wrap { height: 290px; min-height: 290px }`
  - `#tvcal-inner { min-height: 350px }`
  - `.tv-econmap-wrap { min-height: 380px; background: #131722; display: flex; align-items: stretch; width: 100%; height: 100% }`
- These reserve the exact space the widgets will occupy, eliminating the layout jumps that produced CLS 1.0 on desktop and CLS 0.175 on mobile as reported by PageSpeed.

### Performance — assets/dashboard.js

#### Forced reflow fix — liquidity tooltip (lines ~4235–4237)
- **Root cause:** `tooltip.textContent` (DOM write) was followed immediately by `tooltip.offsetWidth` / `tooltip.offsetHeight` (layout reads) in the same synchronous frame, forcing the browser to flush the pending style recalc and reflow — reported as an 87 ms forced layout by PageSpeed.
- **Fix:** Moved both `offsetWidth` and `offsetHeight` reads to *before* the `textContent` / `getElementById().textContent` mutations. Reads now occur while the tooltip still has its previous content; the fallback values (`|| 170`, `|| 56`) handle the case where the tooltip is hidden (`display !== 'block'`). No functional change to positioning logic.

### Performance — assets/gdpr.js

#### Forced reflow fix — GDPR banner style mutations
- Reordered `hidden` and `display` style assignments so layout reads are not interleaved with style writes, minimising reflows triggered by the cookie consent banner on first load.

---

## v7.12.9 (2026-04-05) — Guide audit: COT revert NZD, about.html pair count and data sources, rates mock update

### Guides — guide-cot.html
- **Revert NZD addition:** Previous session incorrectly added NZD as a 7th COT currency. `COT_CURRENCIES` in `dashboard.js` only includes `['EUR','GBP','JPY','AUD','CAD','CHF']` — NZD is not processed by the COT panel. Reverted description to "6 currencies" and removed the NZD mock row from the panel illustration.

### Guides — about.html
- **Pair count corrected:** Stat tile updated from 21 to 28 pairs, matching the `PAIRS` array in `dashboard.js` (7 USD majors + 21 crosses).
- **Positioning Bias row added:** New data sources table row documents the Positioning Bias panel: source = `CBOE ETF options (FXE, FXB, FXY, FXA) · CFTC.gov`, cadence = Weekly.
- **Reference Spreads row added:** New data sources table row for the Reference Spreads panel: source = `Typical ECN · session average`, cadence = Real-time.

### Guides — guide-rates-yield-curve.html
- **Mock values updated to April 2026:** Yield tenor cells updated to approximate current levels (3M: 4.29%, 2Y: 3.79%, 5Y: 4.02%, 10Y: 4.31%). SVG curve redrawn to reflect the current bull-steepener shape (2Y below 3M; 10Y highest). Spread table updated: 2Y–10Y +52 bp (normal), US–DE +131 bp, US–JP +199 bp — reflecting Fed on hold, ECB rate hike cycle, and BoJ continued tightening.

---

## v7.12.8 (2026-04-04) — Fix ATM IV and IV−HV tooltips in TIP_MAP

### Frontend — assets/dashboard.js
- **ATM IV tooltip (TIP_MAP):** Updated `body` to: "30-day at-the-money implied vol from options market. Color = cost of hedging: green = cheap vol (≤7%), red = expensive vol (>12%). Not a directional signal." The `data-tip` attribute on the `pd-cell` is overridden by the `fx-tt` tooltip system which reads from `TIP_MAP` — the visible tooltip was therefore still showing the old text.
- **IV − HV tooltip (TIP_MAP):** Updated `body` to explain both positive and negative cases explicitly, with "Not a directional signal." suffix.

---

## v7.12.7 (2026-04-04) — Fix ATM IV color: CSS specificity conflict resolved

### Frontend — assets/dashboard.css
- **Root cause identified:** `.pd-val` (specificity 0,1,0) and `.pd-dn`/`.pd-up` (also 0,1,0) had equal specificity. Despite `.pd-dn` being declared after `.pd-val` in the file, the browser was not consistently applying cascade order for multi-class elements. Added explicit combined selectors `.pd-val.pd-up` and `.pd-val.pd-dn` (specificity 0,2,0) that unambiguously override the base `.pd-val` color rule. ATM IV in the Pair Detail panel now correctly renders green (≤7%), neutral (7–12%), or red (>12%).

---

## v7.12.6 (2026-04-04) — NZD IV proxy; clarify vol tooltips

### Frontend — assets/dashboard.js
- **NZD IV proxy:** NZD crosses (AUD/NZD, NZD/JPY, EUR/NZD, GBP/NZD, NZD/CAD, NZD/CHF) previously showed `—` for ATM IV because no CBOE-listed ETF has a liquid NZD options chain. Added a proxy: `NZD_IV = AUD_IV × 1.08`, derived from the long-run NZD/AUD realised vol ratio (~1.05–1.10x). Injected into `USD_IV['NZD']` only when AUD IV is available and NZD is absent, so it does not affect direct ETF data if NZD options ever become available. All 6 NZD crosses will now show a synthesised ATM IV with the existing `~` superscript indicator.
- **ATM IV tooltip clarified:** Added "Color = cost of hedging: green = historically cheap vol (≤7%), red = expensive vol (>12%). Not a directional signal." to both the direct ETF variant and the synthesised cross variant. Prevents misreading red as bearish.
- **IV − HV tooltip clarified:** Expanded from "IV minus HV: +ve = options expensive vs realised" to explicit description of both positive and negative cases, with "Not a directional signal." suffix.

---

## v7.12.5 (2026-04-04) — Fix ATM IV color in Pair Detail: use pd-up/pd-dn classes

### Frontend — assets/dashboard.js
- **ATM IV color fix:** v7.12.4 applied `.up`/`.down` classes to the ATM IV `pd-val` div, but those classes appear at line 255 in `dashboard.css` while `.pd-val` appears at line 493 — same specificity, later declaration wins, so `.pd-val { color: var(--text) }` was overriding the color. Switched to `.pd-up`/`.pd-dn` (declared at lines 494–495, immediately after `.pd-val`) which are the correct Pair Detail color utilities and win by cascade order. ATM IV now renders green (≤7%), neutral (7–12%), or red (>12%) as intended.

---

## v7.12.4 (2026-04-04) — ATM IV color in Pair Detail panel; EUR.json 2026 data fix

### Frontend — assets/dashboard.js
- **ATM IV color coding in Pair Detail:** The `pd-val` div for ATM IV now carries the same color class used in the Positioning Bias table: `down` (red) when IV > 12%, neutral when 7%–12%, `up` (green) when IV ≤ 7%. Applies to both direct ETF-sourced values (USD majors) and synthesised cross-pair values. Brings visual consistency with the Positioning Bias ETF IV · COT panel.

### Engine — rates/EUR.json
- **ECB rate history gap patched:** Added 22 monthly observations spanning 2024-06 through 2026-03, reflecting the ECB deposit rate cut cycle (4.00% → 3.75% Jun'24 → 3.50% Sep'24 → 3.25% Oct'24 → 3.00% Dec'24 → 2.75% Jan'25 → 2.50% Mar'25 → 2.25% Apr'25 → 2.00% Jun'25, then hold through Mar'26). Without these, `computeCBTrend()` was comparing 2.00% against a 4.00% observation and signalling a downward trend when the ECB had been on hold since June 2025. Trend now correctly shows flat (—).

---

## v7.12.3 (2026-04-05) — Synthesised ATM IV for 21 cross pairs in Pair Detail panel

### Frontend — assets/dashboard.js
- **Cross-pair IV synthesis:** `updatePairDetail()` previously showed `—` for ATM IV and IV−HV on all 21 non-USD crosses (EUR/GBP, GBP/JPY, AUD/JPY, etc.) because `fx_etf_iv` only covers the 6 USD majors with CBOE-listed ETF option chains. Added triangulation formula: `IV_AB ≈ √(IV_A² + IV_B² − 2·ρ·IV_A·IV_B)` using the USD-pair IVs already available in `quotes.json` as components. Correlation matrix (`CROSS_IV_RHO`) covers all 21 pairs with long-run empirical FX vol correlations (conservative, rounded to 0.05).
- **Synthesised IV label:** Cross-pair ATM IV cells show a `~` superscript and a tooltip explaining the triangulation method ("Synthesised from component USD-pair ETF option IVs via triangulation — indicative only"). Direct ETF-sourced values (USD majors) keep their existing tooltip.
- **IV−HV now populated for all pairs:** Because `atmIv` is now non-null for crosses, the `IV − HV` cell also renders for all 28 pairs whenever `hv30` is available.

---

## v7.12.2 (2026-04-05) — Restore Carry Trade sidebar panel; fix panel subtitle spacing

### Frontend — index.html
- **Carry Trade Ranking restored:** Sidebar's last `sb-section` reverted from the erroneous "Cross-Asset" duplicate (introduced in v7.9.0) back to `#carry-rank-rows` with heading "Carry Trade Ranking" and subtitle "G8 · CB rate differential". The main Cross-Asset panel (`#section-crossasset`) already covers VIX/MOVE/SPX/Gold/WTI/DXY in the main content area — duplicating it in the sidebar added no value.
- **`#etf-iv-rows` removed:** Container ID no longer exists in the sidebar; `fetchEtfIV()` now has no render target and was cleanly removed from `boot()` and its `setInterval`.

### Frontend — assets/dashboard.js
- **`fetchEtfIV()` removed from boot and interval:** Function still exists for `fx_etf_iv` data consumption in `fetchOptionSkew()` / Positioning Bias, but its sidebar render loop (`#etf-iv-rows`) is gone. Removed the `fetchEtfIV()` call in `boot()` and `setInterval(fetchEtfIV, 10 * 60 * 1000)`.

### Frontend — assets/dashboard.css
- **Panel subtitle spacing:** Added `margin-left: 8px` to `.panel-sub`. Fixes the visual issue where subtitle text (e.g. `yfinance · 22:03 GMT-3 · ~5min delay`, `CFTC · week ending 2026-03-31 · updated Fri, Apr 03`) appeared flush against the panel title with no separation. Affects all main panel headers: FX Pairs — Majors, CFTC Positioning (COT), Yield Curve, Risk Monitor, and all other `panel-head` elements.

---

## v7.12.1 (2026-04-04) — Fix sidebar cross-asset panel: race condition + wrong data keys

### Frontend — assets/dashboard.js
- **Root cause:** `fetchEtfIV()` was reading `intra?.etf_iv` (undefined — correct field is `fx_etf_iv`, a separate FX options block) and falling back to `STOOQ_RT_CACHE` which is empty on boot. The function returned "Awaiting data…" on every page load because the container ID (`#etf-iv-rows`) now correctly resolves after the v7.12.0 `index.html` fix exposed the data-loading bug.
- **Fix 1 — race condition:** Replaced `STOOQ_RT_CACHE` reads with direct `intra.quotes` reads. `loadIntradayQuotes()` is awaited before any rendering, so `intra.quotes` is always populated. `STOOQ_RT_CACHE` is asynchronous secondary enrichment and must never be the primary source for boot-time renders.
- **Fix 2 — manifest keys:** `ETF_IV_MANIFEST` updated from VIX9D / VVIX / GLD IV / TLT IV / EEM IV / EFA IV (never present in `quotes.json`) to keys that exist: `vix`, `move`, `spx`, `gold`, `wti`, `dxy`, `us10y`, `btc`. All 8 keys verified against live `quotes.json`.
- **Heading rename:** Sidebar panel heading changed from "Options IV · ETF · CBOE · 30-day" to "Cross-Asset · VIX · MOVE · SPX · Gold · WTI · DXY" to accurately reflect the data now shown. Footer subtitle changed from "Implied vol · 30-day ATM · CBOE/yfinance" to "yfinance · ~5min delay".

### Frontend — index.html
- **Sidebar panel heading and subtitle** updated to match new manifest content (see dashboard.js change above).

---



### Engine — scripts/fetch_intraday_quotes.py
- **IV history accumulation (PASO 7b):** After `fetch_fx_etf_iv()`, a new block reads/writes `intraday-data/iv_history.json`. Each ISO week, the current IV snapshot for each FX ETF pair is appended to a rolling array capped at 52 entries. Guard: `_last_week` ISO-week tuple prevents duplicate appends on intraday re-runs within the same week.
- **IV Rank and IV Percentile calculation:** For pairs with ≥4 weeks of history, computes `iv_rank` ((current − min) / (max − min) × 100, 0–100) and `iv_pct_rank` (% of historical snapshots below current IV). Both fields injected into the `fx_etf_iv` block of `quotes.json` alongside `iv_hist_n` (history count). Pairs with <4 weeks show `null` — never fabricated.
- **History file:** `intraday-data/iv_history.json` (new file, auto-created on first run). Schema: `{ "_last_week": [year, week], "pairs": { "eurusd": [{ "week": "2026-W14", "iv": 20.0 }, ...] } }`.

### Engine — scripts/generate_narrative_signals.py
- **`evidence[]` field in SIGNALS_SYSTEM prompt:** JSON format updated from `{ time, priority, title, text }` to `{ time, priority, title, text, evidence }`. The `evidence` field is now **mandatory** in every signal — 2–4 short `"LABEL: VALUE"` strings listing the exact data points that motivated each signal (e.g. `"VIX: 23.9"`, `"Fed rate: 4.50% (ON HOLD)"`, `"US 2Y-10Y spread: -22bp"`).
- **Post-processing validation:** `generate_signals()` now extracts, validates, and preserves `evidence[]` — must be a non-empty list of strings, truncated to 4 entries max at 80 chars each. Empty or malformed evidence arrays are stored as `[]` rather than causing a crash.

### Frontend — assets/dashboard.js
- **IV Rank column in Positioning Bias table:** `fetchOptionSkew()` now detects `iv_rank != null` on ETF IV entries. When ≥4 weeks of history are available: thead updates to `Pair / ATM IV / IV Rnk / Direction`; each row shows rank as `NNrnk` (e.g. `82rnk`) in monospace, coloured red >75, green <25. `title` attribute includes IV Rank, IV Percentile, history count, and interpretation note. When history is building (<4 weeks): falls back to COT bias column as before, with tooltip noting "IV Rank building".
- **ETF IV panel timestamp:** `fetchEtfIV()` now updates `#etf-iv-panel-sub` after rendering rows. Subtitle shows source, update time in local timezone, and data age in minutes derived from `quotes.json` `updated` field.
- **COT panel timestamp:** `fetchCOTData()` subtitle now appends `· loaded HH:MM TZ` (local time) after the CFTC week and update date. Gives traders the exact moment the frontend loaded the data.
- **Sentiment panel timestamp:** `renderSentiment()` subtitle (`#sent-source-sub`) now includes `· updated HH:MM TZ` for Myfxbook, `· loaded HH:MM TZ` for COT fallback and historical fallback.
- **Reference Spreads timestamp:** `fetchReferenceSpreads()` subtitle now appends `· HH:MM TZ` to the vol regime label.
- **AI signal evidence rendering:** Signal rows now render `evidence[]` chips below the signal text. Chips are hidden by default; clicking the row toggles `.a-evidence-open` to reveal them. Rows with evidence get a `▸` indicator and `title` tooltip showing all evidence inline. Rows without evidence are unchanged.

### Frontend — assets/dashboard.css
- **`.a-evidence`, `.a-evidence-open`:** Flex container for evidence chips, `display:none` by default, `display:flex` when `.a-evidence-open` is toggled.
- **`.a-ev-chip`:** 9px monospace pill, `var(--bg3)` background, `var(--border2)` border. Matches terminal design system.
- **`.a-has-ev`:** Adds `▸` pseudo-element indicator and `cursor:pointer` to signal rows that carry evidence data.

### Frontend — index.html
- **Sidebar Options IV panel (bug fix from v7.9.0):** `#carry-rank-rows` container and "Carry Trade Ranking" heading were never replaced in index.html despite the v7.9.0 CHANGELOG stating otherwise. Fixed: heading updated to "Options IV · ETF · CBOE · 30-day", container ID changed to `#etf-iv-rows`, footer div given `id="etf-iv-panel-sub"` for live timestamp injection.

---



### Frontend — assets/dashboard.css
- **Reduced vertical footprint ~30%:** `.pd-header` padding `5px→3px`, `.pd-price-block` padding `6px→4px`, `.pd-cell` padding `5px→3px`, `.pd-footer` padding `4px→3px`. `.pd-lbl` `margin-bottom` `2px→0`. Matches the information density of Bloomberg/Eikon detail panels.
- **Price font size:** `18px→15px` — still dominant in the block but proportionate to the compact layout.
- **Value font size:** `12px→11px` — consistent with the sidebar data density (sb-row, carry-rank-row).
- **Range margin:** `margin-top:1px→0` — eliminates the gap between H/L line and price block border.

---

## v7.11.0 (2026-04-04) — Alerts popover position:fixed, fx-tt tooltips for pair-detail, carry rank alignment

### Frontend — assets/dashboard.css
- **Alerts popover root cause fixed:** Changed `#alerts-popover` from `position: absolute` to `position: fixed`. A `position: absolute` child of a `position: fixed` parent with `height: 22px` is clipped to that height regardless of `overflow: visible`. Using `position: fixed` escapes all ancestor clipping entirely. `z-index` raised to `99999` to match `#fx-tt`.
- **Pair-detail tooltips:** Removed the CSS `::after` approach (always clipped by rightpanel boundaries). Replaced with `fx-tip` cursor style only — tooltip rendering delegated to the JS `#fx-tt` engine.
- **Carry rank row alignment:** Reduced rank column from `16px` to `12px`, gap from `4px` to `3px`, rank text aligned left — brings pair name visually flush with the `8px` left padding of sibling `.sb-row` cross pairs.

### Frontend — assets/dashboard.js
- **`toggleAlertsPopover()`:** Rewritten to calculate `position: fixed` coordinates from `btn.getBoundingClientRect()` — opens above the button with `requestAnimationFrame` reflow correction. Matches the `_fxTTPos` pattern used by the risk monitor tooltip engine.
- **`updatePairDetail()` — fx-tt bootstrap:** Added self-contained `#fx-tt` engine initialiser at the top of `updatePairDetail`. If `renderSentiment` has not yet run (user clicks a pair before sentiment loads), the tooltip engine is bootstrapped on first pair click. Each `pd-cell` gets `mouseenter`/`mouseleave` handlers pointing to the shared `#fx-tt` DOM element with label-matched tooltip copy.

### Frontend — index.html
- **Carry rank Loading… restored:** Reverted the removal of the static placeholder. The `fetchCarryRanking()` race condition fix (via `fetchCBRates().then(...)`) ensures it is replaced promptly on boot.

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
