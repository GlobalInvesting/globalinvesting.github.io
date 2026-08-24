// CURRENCY STRENGTH HEATMAP MODAL  v2.6.4 — Market Commentary follow-up
//   (the client, screenshot): the 2-line clamp from v2.6.3 was wrong — he never
//   wanted the article text cut, just fewer visible at once with a scroll to
//   reach the older ones (same idea as cb-rates-modal.js's .cbr-ps-wrap).
//   Reverted -webkit-line-clamp on .hm-news-art-body (full text again, still
//   subject to the existing 400-char/sentence-boundary safeguard in
//   _hmLoadSessionNews()). Instead added max-height:210px (~2 full articles)
//   to .hm-news-wrap, on top of its existing flex:1/overflow-y:auto — box now
//   shows ~2 articles and scrolls to reveal the 3rd (older) one. Article
//   fetch cap unchanged at 3 (v2.6.0).
// CURRENCY STRENGTH HEATMAP MODAL  v2.6.3 — Currency switcher: arrows flanking
//   the chip (the client, session request, referencing the original ‹ NZD ▾ ›
//   spec): reverted from v2.6.1/v2.6.2's grouped-arrow layouts back to one
//   arrow on each side of the chip — [flag] — text ‹ [chip] ›. #hm-ccy-arrows
//   wrapper removed; #hm-ccy-prev now sits directly before #hm-ccy-switch and
//   #hm-ccy-next directly after it, both still plain flex children of
//   #hm-title-row (existing 5px gap). Pure DOM-order/CSS change — IDs
//   unchanged, hmCycleCcy/hmToggleCcyDropdown/hmPivotCcy/_hmUpdateCcySwitcher
//   all look up #hm-ccy-prev/#hm-ccy-next/#hm-ccy-chip/#hm-ccy-dd by ID.
//   Also: Session tab Market Commentary article body now clamped to 2 lines
//   (`-webkit-line-clamp:2` on .hm-news-art-body, matching the client's "dos
//   líneas máximo" request) — the existing 400-char/sentence-boundary cut in
//   _hmLoadSessionNews() is a length safeguard, not a line-count one, so long
//   paragraphs were still running 4-5 visual lines. .hm-news-wrap was already
//   flex:1/overflow-y:auto (same scroll pattern as cb-rates-modal.js's
//   .cbr-ps-wrap); with bodies now clamped to 2 lines, ~2 articles fit in the
//   .hm-cw's existing 140px min-height before the 3rd needs that scroll.
// CURRENCY STRENGTH HEATMAP MODAL  v2.6.2 — Currency switcher layout fix #2
//   (the client, screenshot): v2.6.1 moved the arrows next to the chip but kept
//   the chip immediately after the flag, ahead of the full-name text — still
//   not what was asked for. Correct order: [flag] — full name text [‹›] [chip].
//   Reordered #hm-title-row's children (title text, then arrows, then the
//   switcher) so the flag+name reads first as the primary label, with the
//   picker (arrows + chip) trailing it as a secondary control. Pure DOM-order
//   change again — no logic touched, all lookups are by ID.
// CURRENCY STRENGTH HEATMAP MODAL  v2.6.1 — Currency switcher layout fix (the client,
//   screenshot): the ‹/› arrows previously flanked the chip (‹ [USD▾] ›), so the
//   left arrow sat immediately after the flag, ahead of the currency chip itself —
//   read oddly, like it belonged to the flag rather than to the switcher. Reordered
//   to [flag] [USD▾ chip] [‹›] — text: chip stays right after the flag, both arrows
//   now grouped together to its right as a single control. Purely a DOM-order/CSS
//   change (#hm-ccy-arrows wraps the two buttons with a tight 1px gap) — no change
//   to hmCycleCcy/hmToggleCcyDropdown/hmPivotCcy logic, ID lookups unaffected.
// CURRENCY STRENGTH HEATMAP MODAL  v2.6.0 — Session tab follow-up (the client,
//   screenshots): (1) fixed the Market Commentary block flickering — it was
//   re-fetching and re-rendering (loading spinner → articles) on every
//   Finnhub RT tick via _hmRefreshIfOpen's populateSession() call, even
//   though news doesn't change tick-to-tick; now gated by a _hmNewsCcy
//   currency-tracking guard so it only re-runs on an actual currency change.
//   (2) capped Market Commentary at 3 articles (was 6). (3) added an in-modal
//   currency switcher (‹ NZD ▾ ›, dropdown + prev/next arrows + ArrowLeft/
//   ArrowRight keys) to the header, mirroring cot-modal-chart.js's existing
//   switcher — wired through the existing hmPivotCcy() pivot path so Session/
//   CSI only re-render when their tab is actually visible, same lazy pattern
//   hmTab() already uses.
// CURRENCY STRENGTH HEATMAP MODAL  v2.5.0 — Session tab: added a Market Commentary
//   fill block below Session Context, same shape/source as cb-rates-modal.js's
//   _cbrLoadPolicySummary() (news-data/news.json, filtered to the modal's active
//   currency via `cur`, up to 6 articles). Reported by the client (screenshots):
//   on tall/large screens the Session tab left visible empty space below Session
//   Context, since neither of its two .hm-cw cards grow to fill .hm-panel.on's
//   flex column. Third .hm-cw card takes flex:1 (same pattern already used by
//   Rel. Strength's first .hm-cw, and by .cbr-ps-wrap in cb-rates-modal.js) so it
//   absorbs the leftover vertical space instead of leaving it blank. Not filtered
//   by CB_KW like the CB modal's version — this tab covers the currency generally.
// CURRENCY STRENGTH HEATMAP MODAL  v2.4.0 — CSI chart: replaced the two-row Interval+Range control (added earlier this session) with a single industry-standard range selector (1D/1W/1M/3M/6M/1Y/All), each mapping internally to both a lookback and an auto-selected OHLC resolution — confirmed against TradingView's own docs and CSM-specific tools (FXSSI, MarketMilk), which all expose exactly one range control, never two. Also: added chart.timeScale().fitContent() after loading series data, which was missing entirely — without it LWC used a fixed bar-spacing default instead of stretching the loaded range to fill the chart width, so ranges with few bars left visible empty space and chart width looked inconsistent across timeframes
// CURRENCY STRENGTH HEATMAP MODAL  v2.3.4 — CSI chart: normalize each currency's daily return by ACTUAL per-date pair coverage instead of a fixed pair count, so a legitimately-missing bar for one pair (e.g. fetch_ohlc.py's flat-bar guard dropping a degenerate O=H=L=C bar) no longer systematically understates that currency's move for that one bar
// CURRENCY STRENGTH HEATMAP MODAL  v2.3.3 — CSI chart: added "Interval" / "Range" group labels above the TF and period button rows (the client's pick, Bloomberg-style) so the two rows read as distinct controls instead of duplicate-looking buttons (both rows can show "1D"/"1W" text since they answer different questions — interval vs. lookback)
// CURRENCY STRENGTH HEATMAP MODAL  v2.3.2 — CSI chart: fixed _updateCSILiveBar() still rebasing the live RT point against a bar-count cutoff (missed in v2.3.1's calendar-day migration), which snapped every currency's most recent point to a wildly different baseline than the rest of the series on every RT tick — visible as all CSI lines jumping/converging together at the chart's right edge
// CURRENCY STRENGTH HEATMAP MODAL  v2.3.1 — CSI chart: TF+period controls merged into one row with uniform button sizing; TF buttons moved off the shared .lw-tf-btn class onto their own .hm-csi-btn (was cross-contaminating with the main chart's global TF-button selector); period presets switched from an assumed bar-count to real calendar-day cutoffs (bar-count drifted for H1/H4/W1 depending on incidental weekend placement in the trailing window)
// CURRENCY STRENGTH HEATMAP MODAL  v2.3.0 — CSI chart: added H1/H4/1D/1W timeframe selector (reuses main chart's intraday ohlc-data/h1|h4 sources; 1W is a telescoping downsample of the daily series, no new data source needed); fixed tooltip showing raw unix seconds for intraday TFs
// CURRENCY STRENGTH HEATMAP MODAL  v2.2.4 — CSI chart: ResizeObserver keeps chart width in sync with container (was fixed at creation-time offsetWidth, so a browser resize while the modal was open left the chart clipped/misaligned)
// CURRENCY STRENGTH HEATMAP MODAL  v2.2 — audit fixes: tooltipEl bug, keyframes, tab a11y, labels
// CURRENCY STRENGTH HEATMAP MODAL  v1.1.0
// File: assets/heatmap-modal.js
// Loaded AFTER dashboard.js (see index.html)
//
// Public API (called from dashboard.js populateHeatmap):
//   openHeatmapModal(ccy, strengths, rtCache)
//   closeHeatmapModal()
//   hmTab(el, tabId)
//
// Pattern mirrors cb-rates-modal.js and corr-modal.js.
// All IDs prefixed hm- to avoid CSS collisions.
// ═══════════════════════════════════════════════════════════════════════════

(function () {

  // ── CSS ───────────────────────────────────────────────────────────────────
  if (document.getElementById('hm-modal2-css')) return;
  const s = document.createElement('style');
  s.id = 'hm-modal2-css';
  s.textContent = `
/* ── Heatmap Modal — cohesive with Real Carry Modal ── */

#hm-bd {
  display:block!important;
}
@keyframes hm-fadein  { from{opacity:0}                              to{opacity:1} }
@keyframes hm-slidein { from{transform:translateY(-8px);opacity:0}  to{transform:none;opacity:1} }

#hm-modal {
  width:100%!important;max-width:none!important;height:auto!important;max-height:none!important;
  border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;
  background:var(--bg)!important;position:static!important;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:var(--text);
  display:flex;flex-direction:column;
}

#hm-modal::before {
  display:none;
}

/* ── Header ── */
#hm-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 14px 9px;
  border-bottom:1px solid var(--border,#252d3d);
  flex-shrink:0;
  background:var(--bg2);
}
#hm-hd-left { display:flex;flex-direction:column;gap:2px; }


#hm-title-row { display:flex;align-items:center;gap:5px; }
#hm-title { font-size:14px;font-weight:600;color:var(--text);letter-spacing:-.01em;line-height:1.2;font-family:var(--font-ui,'Inter',-apple-system,sans-serif); }
#hm-title .fi { border-radius:2px;font-size:16px; }
#hm-sub { font-size:10px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text2);letter-spacing:.02em;margin-top:1px; }
#hm-close {
  background:none;border:none;color:var(--text3,#4e5c70);font-size:16px;
  cursor:pointer;padding:3px 6px;border-radius:3px;line-height:1;
  transition:color .1s,background .1s;font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
}
#hm-close:hover { color:var(--text);background:var(--bg3); }

/* Currency switcher (in-modal, no need to close/reopen) — same pattern as
   cot-modal-chart.js's #cot-ccy-switch, namespaced hm- for this file. */
.hm-ccy-arrow {
  background:none;border:none;color:var(--text3,#4e5c70);font-size:11px;
  cursor:pointer;padding:2px 4px;border-radius:3px;line-height:1;flex-shrink:0;
  transition:color .1s,background .1s;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}
.hm-ccy-arrow:hover { color:var(--text);background:var(--bg3); }
.hm-ccy-arrow:disabled { opacity:.3;cursor:default; }
.hm-ccy-arrow:disabled:hover { background:none;color:var(--text3,#4e5c70); }
#hm-ccy-switch { position:relative;display:inline-flex; }
#hm-ccy-chip {
  background:var(--bg3,#151b26);border:1px solid var(--border,#252d3d);border-radius:4px;
  color:var(--text);font-size:13px;font-weight:600;padding:1px 7px;cursor:pointer;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);letter-spacing:-.01em;
  display:inline-flex;align-items:center;gap:4px;transition:border-color .1s,background .1s;
}
#hm-ccy-chip:hover { border-color:var(--blue);background:var(--bg2); }
#hm-ccy-chip::after {
  content:'';width:0;height:0;margin-left:1px;
  border-left:3.5px solid transparent;border-right:3.5px solid transparent;
  border-top:4px solid var(--text3,#4e5c70);
}
#hm-ccy-dd {
  display:none;position:absolute;top:calc(100% + 4px);left:0;z-index:20;
  background:var(--bg2);border:1px solid var(--border,#252d3d);border-radius:5px;
  box-shadow:0 6px 18px rgba(0,0,0,.4);padding:4px;min-width:64px;
  grid-template-columns:repeat(2,1fr);gap:2px;
}
#hm-ccy-dd.open { display:grid; }
.hm-ccy-dd-item {
  background:none;border:none;color:var(--text2);font-size:11px;font-weight:600;
  padding:5px 6px;border-radius:3px;cursor:pointer;text-align:center;
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
  transition:color .1s,background .1s;
}
.hm-ccy-dd-item:hover { color:var(--text);background:var(--bg3); }
.hm-ccy-dd-item.on { color:var(--blue);background:var(--bg3); }

/* ── Metrics strip ── */
#hm-metrics {
  display:grid;grid-template-columns:repeat(6,1fr);
  border-bottom:1px solid var(--border2);
  flex-shrink:0;
  background:var(--bg);
}
.hm-mm {
  padding:9px 14px;
  border-right:1px solid var(--border2);
  display:flex;flex-direction:column;gap:1px;
}
.hm-mm:last-child { border-right:none; }
.hm-mm-lbl {
  font-size:9px;font-family:var(--font-mono,monospace);font-weight:600;
  color:var(--text2);text-transform:uppercase;letter-spacing:.09em;
}
.hm-mm-val {
  font-size:15px;font-weight:600;font-family:var(--font-mono,monospace);
  color:var(--text);line-height:1;margin-top:2px;
}
.hm-mm-val.sm   { font-size:12px; }
.hm-mm-val.up   { color:var(--up); }
.hm-mm-val.down { color:var(--down); }
.hm-mm-val.flat { color:var(--text2); }
.hm-mm-sub {
  font-size:9px;font-family:var(--font-mono,monospace);
  color:var(--text2);margin-top:1px;
}
.hm-mm-sub.up   { color:var(--up); }
.hm-mm-sub.down { color:var(--down); }

/* ── Tabs ── */
#hm-tabs {
  display:flex;padding:0 14px;
  border-bottom:1px solid var(--border,#252d3d);
  flex-shrink:0;background:var(--bg2);
  overflow-x:auto;scrollbar-width:none;
}
#hm-tabs::-webkit-scrollbar { display:none; }
.hm-tab {
  font-size:11px;font-weight:500;
  padding:9px 14px;cursor:pointer;
  color:var(--text2);
  border-bottom:2px solid transparent;
  transition:color .12s;white-space:nowrap;user-select:none;
  font-family:var(--font-ui,sans-serif);
  /* button reset — keeps visual identical to former div */
  background:none;border-top:none;border-left:none;border-right:none;outline:none;
}
.hm-tab:focus-visible { outline:2px solid var(--blue);outline-offset:-2px;border-radius:2px; }
.hm-tab:hover { color:var(--text2); }
.hm-tab.on { color:var(--text);border-bottom-color:var(--blue); }

/* ── Body ── */
#hm-body {
  flex:1;min-height:0;
  overflow-y:auto;
  padding:0;
  background:var(--bg);
  scrollbar-width:thin;
  scrollbar-color:var(--border2,#2e3a50) transparent;
}
#hm-body::-webkit-scrollbar { width:3px!important; }
#hm-body::-webkit-scrollbar-track { background:transparent; }
#hm-body::-webkit-scrollbar-thumb { background:var(--border2,#2e3a50);border-radius:2px; }
#hm-body::-webkit-scrollbar-thumb:hover { background:var(--text2); }
.hm-panel { display:none;padding:0; }
.hm-panel.on { display:flex;flex:1;flex-direction:column;min-height:0; }

/* ── Card wrapper ── */
.hm-cw {
  background:var(--bg);
  border:none;
  border-radius:0;
  padding:14px;
  margin-bottom:0;
  border-bottom:1px solid var(--border,#252d3d);
  overflow-x:auto;
  scrollbar-width:thin;
  scrollbar-color:var(--border2,#2e3a50) transparent;
}
.hm-cw:last-child { border-bottom:none; }
.hm-cw::-webkit-scrollbar { height:3px; }
.hm-cw::-webkit-scrollbar-thumb { background:var(--border2,#2e3a50);border-radius:2px; }

/* Session tab — Market Commentary fill block (v2.5.0). Mirrors cb-rates-modal's
   .cbr-ps-wrap fill pattern: this card takes flex:1 inside .hm-panel.on's
   column flex so it absorbs whatever vertical space the fixed-height session
   grid + AI notes above it don't use, instead of leaving it empty on tall
   viewports. Same reason correlations' first .hm-cw is flex:1 (line ~745). */
.hm-news-wrap { flex:1;min-height:0;max-height:210px;overflow-y:auto;margin-top:8px;scrollbar-width:thin;scrollbar-color:var(--border2,#2e3a50) transparent; }
.hm-news-wrap::-webkit-scrollbar { width:3px!important; }
.hm-news-wrap::-webkit-scrollbar-thumb { background:var(--border2,#2e3a50);border-radius:2px; }
.hm-news-article { padding:8px 0;border-bottom:1px solid rgba(54,60,78,.35); }
.hm-news-article:last-child { border-bottom:none; }
.hm-news-art-meta { display:flex;align-items:center;gap:6px;margin-bottom:4px; }
.hm-news-art-source { font-size:8px;font-weight:600;color:var(--blue,#4f7fff);font-family:var(--font-mono);text-transform:uppercase;letter-spacing:.05em; }
.hm-news-art-time { font-size:8px;color:var(--text2);font-family:var(--font-mono); }
.hm-news-art-title { font-size:10px;font-weight:600;color:var(--text);line-height:1.35;margin-bottom:4px;font-family:var(--font-ui,'Inter',-apple-system,sans-serif); }
.hm-news-art-title a { color:inherit;text-decoration:none; }
.hm-news-art-title a:hover { text-decoration:underline;text-decoration-color:var(--text2); }
.hm-news-art-body { font-size:10px;color:var(--text2);line-height:1.55;font-family:var(--font-ui,'Inter',-apple-system,sans-serif); }
.hm-news-loading, .hm-news-empty { padding:14px 0;font-size:10px;color:var(--text2);font-family:var(--font-mono); }

/* Section label */
.hm-ct {
  font-size:8.5px;font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:var(--text3,#4e5c70);
  letter-spacing:.07em;margin-bottom:10px;
  font-weight:600;
  text-transform:uppercase;
}

/* ── Pair breakdown table ── */
.hm-tbl {
  width:100%;border-collapse:collapse;
  font-size:11.5px;font-family:var(--font-mono,monospace);
}
.hm-tbl thead th {
  text-align:right;color:var(--text2);font-weight:500;
  font-size:9px;text-transform:uppercase;letter-spacing:.08em;
  padding:7px 10px;
  border-bottom:1px solid var(--border2);
  white-space:nowrap;
}
.hm-tbl thead th:first-child { text-align:left; }
.hm-tbl th { text-align:right;color:var(--text2);font-weight:500;font-size:9px;text-transform:uppercase;letter-spacing:.08em;padding:7px 10px;border-bottom:1px solid var(--border2);white-space:nowrap; }
.hm-tbl th:first-child { text-align:left; }
.hm-tbl tbody tr { transition:background .08s; }
.hm-tbl tbody tr:nth-child(even) td { background:rgba(255,255,255,.015); }
.hm-tbl tbody tr:hover td { background:rgba(88,166,255,.05); }
.hm-tbl td {
  text-align:right;padding:7px 10px;
  border-bottom:1px solid rgba(255,255,255,.04);
  color:var(--text);vertical-align:middle;white-space:nowrap;
}
.hm-tbl td:first-child { text-align:left; }
.hm-tbl tr:last-child td { border-bottom:none; }
.hm-tbl td.up   { color:var(--up); }
.hm-tbl td.down { color:var(--down); }
.hm-tbl td.flat { color:var(--text2); }
.hm-tbl .sym,.hm-tbl .hm-sym { font-weight:600;color:var(--text); }
.imp-wrap { display:flex;align-items:center;gap:6px;justify-content:flex-end; }
.imp-bar-bg { width:36px;height:3px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden; }
.imp-bar-fill { height:100%;border-radius:2px; }

/* ── Color classes ── */
.up   { color:var(--up); }
.down,.dn { color:var(--down); }
.flat { color:var(--text2); }

/* ── Ranking bars ── */
.hm-rank-row { display:flex;align-items:center;gap:8px;margin-bottom:5px; }
.hm-rank-ccy {
  width:34px;font-size:10px;font-weight:600;
  font-family:var(--font-mono,monospace);color:var(--text2);text-align:right;
}
.hm-rank-ccy.hl { color:var(--text); }
.hm-rank-bg { flex:1;height:14px;background:rgba(255,255,255,.04);border-radius:2px;overflow:hidden; }
.hm-rank-fill { height:100%;border-radius:2px;transition:width .35s ease; }
.hm-rank-fill.no-transition { transition:none; }
.hm-rank-fill.hl   { background:var(--blue); }
.hm-rank-fill.up   { background:rgba(38,166,154,.35); }
.hm-rank-fill.down { background:rgba(239,83,80,.30); }
.hm-rank-fill.flat { background:rgba(139,148,158,.20); }
.hm-rank-val { width:56px;text-align:right;font-size:10px;font-family:var(--font-mono,monospace);color:var(--text2); }
.hm-rank-sublbl { font-size:8.5px;font-family:var(--font-mono,monospace);color:var(--text2);letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px; }

/* ── Session tab ── */
.sess-grid { display:grid;grid-template-columns:80px 1fr 60px;align-items:center;gap:5px 8px;font-family:var(--font-mono,monospace);font-size:10px; }
.sess-lbl { color:var(--text2);text-align:right;letter-spacing:.04em; }
.sess-lbl.hl { color:var(--blue); }
.sess-track { height:10px;background:rgba(255,255,255,.04);border-radius:2px;overflow:hidden; }
.sess-fill { height:100%;border-radius:2px; }
.sess-val { text-align:right; }

/* State chips */
.state-chip {
  display:inline-flex;align-items:center;
  font-size:8px;font-family:var(--font-mono,monospace);font-weight:700;
  padding:1px 5px;border-radius:3px;letter-spacing:.06em;
  vertical-align:middle;margin-left:5px;
}
.state-live     { background:rgba(56,139,253,.15);color:var(--blue);border:1px solid rgba(56,139,253,.25); }
.state-closed   { background:transparent;color:var(--text2);border:1px solid var(--border2); }
.state-upcoming { background:rgba(210,153,34,.10);color:#d29922;border:1px solid rgba(210,153,34,.22); }

.sess-note { margin-bottom:7px; }
.sess-note-hdr { display:flex;align-items:center;gap:6px;margin-bottom:3px; }
.sess-note-name { font-size:10px;font-family:var(--font-mono,monospace);font-weight:600;letter-spacing:.04em;color:var(--text); }
.sess-note-body { font-size:10.5px;font-family:var(--font-mono,monospace);color:var(--text2);line-height:1.6;padding-left:2px; }

/* ── Rel. Strength tab (Strength Differential matrix) ── */
/* ── Strength matrix — rcm-matrix aesthetic ── */
.corr-wrap { overflow:auto;flex:1;min-height:0;scrollbar-width:thin;scrollbar-color:#444c56 transparent; }
.corr-wrap::-webkit-scrollbar { width:4px;height:4px; }
.corr-wrap::-webkit-scrollbar-track { background:transparent; }
.corr-wrap::-webkit-scrollbar-thumb { background:#444c56;border-radius:2px; }
.corr-wrap::-webkit-scrollbar-thumb:hover { background:var(--text2); }
.corr-matrix { border-collapse:collapse;font-size:10.5px;font-family:var(--font-mono,monospace);width:100%;table-layout:fixed; }
.corr-matrix th { font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;padding:6px 4px;color:var(--text2);text-align:center;white-space:nowrap;background:var(--bg2);position:sticky;top:0;z-index:2;border-bottom:1px solid var(--border2); }
.corr-matrix th.row-head { text-align:left;width:72px;padding:6px 8px; }
.corr-matrix th.focal { color:var(--blue); }
.corr-matrix td { padding:6px 4px;text-align:center;border:1px solid rgba(255,255,255,.04);font-size:10.5px;transition:filter .1s; }
.corr-matrix td:hover { filter:brightness(1.4); }
.corr-matrix td.diag { background:#2d333b;color:var(--text2);font-size:10px;font-weight:600; }
.corr-matrix td.row-head { text-align:left;color:var(--text2);font-weight:700;font-size:10.5px;background:var(--bg2);border:none;position:sticky;left:0;z-index:1; }
.corr-matrix td.row-head.focal { color:var(--blue); }
.corr-matrix td.empty { background:transparent;border:none; }
.corr-matrix td.comp-col { border-left:2px solid rgba(255,255,255,.10); }
.corr-matrix tr.comp-row td { border-top:2px solid rgba(255,255,255,.10); }
/* cell shading — terminal palette */
.corr-cell-pos-hi { background:rgba(38,166,154,.25);color:var(--up);font-weight:700; }
.corr-cell-pos    { background:rgba(38,166,154,.10);color:var(--up); }
.corr-cell-neg-hi { background:rgba(239,83,80,.25);color:var(--down);font-weight:700; }
.corr-cell-neg    { background:rgba(239,83,80,.10);color:var(--down); }
.corr-cell-flat   { color:var(--text2); }
.corr-cell-focal  { outline:1px solid rgba(56,139,253,.35); }
/* legend — mirrors rcm-matrix-legend */
.corr-legend { display:flex;gap:16px;flex-wrap:wrap;font-size:9px;font-family:var(--font-mono,monospace);color:var(--text2);margin-top:10px;padding-top:10px;border-top:1px solid var(--border2);align-items:center;flex-shrink:0; }

/* ── Top-3 drivers ── */
.driver-row { display:flex;align-items:flex-start;gap:10px;margin-bottom:8px;font-family:var(--font-mono,monospace); }
.driver-pair { font-size:11px;font-weight:600;color:var(--text);width:72px;padding-top:1px;flex-shrink:0; }
.driver-body { flex:1; }
.driver-top  { display:flex;align-items:center;gap:8px; }
.driver-pct  { font-size:11px;font-weight:600; }
.driver-vs   { font-size:11px;color:var(--text2); }
.driver-note { font-size:10px;color:var(--text2);margin-top:3px;line-height:1.5; }

/* ── CSI chart ── */
/* Single range-selector row (2026-08-07 redesign, replacing an earlier
   same-day two-row Interval+Range design). Industry-standard chart range
   bars (TradingView, Google/Yahoo Finance-style, and CSM-specific tools
   like FXSSI) expose exactly ONE user-facing control — the lookback range
   — and pick bar resolution automatically underneath; see the
   _CSI_RANGE_CONFIG comment in the script block below for the full
   rationale and the range→resolution mapping. */
#hm-csi-controls { display:flex;align-items:center;gap:2px;margin-bottom:10px;flex-wrap:wrap; }
#hm-csi-wrap,.csi-wrap {
  position:relative;height:280px;
  background:var(--bg);border-radius:4px;overflow:hidden;
  margin-bottom:10px;
}
#hm-csi-chart,.csi-canvas-placeholder { width:100%;height:100%; }
.hm-csi-btn,.hm-csi-pbtn,.csi-pbtn {
  font-size:10px;padding:3px 9px;border-radius:3px;
  border:1px solid var(--border2);
  background:none;color:var(--text2);cursor:pointer;
  font-family:var(--font-mono,monospace);
  transition:background .1s,color .1s,border-color .1s;
  white-space:nowrap;
  line-height:1.4;
}
.hm-csi-btn:hover,.hm-csi-pbtn:hover,.csi-pbtn:hover { background:rgba(255,255,255,.05);color:var(--text); }
.hm-csi-btn.on,.hm-csi-pbtn.on,.csi-pbtn.on {
  background:rgba(56,139,253,.15);
  border-color:rgba(56,139,253,.35);
  color:var(--blue);
}
#hm-csi-legend,.csi-legend {
  display:flex;flex-wrap:wrap;gap:4px 10px;margin-top:8px;
  font-size:9px;font-family:var(--font-mono,monospace);
}
.hm-csi-leg,.csi-leg {
  display:flex;align-items:center;gap:4px;cursor:pointer;
  padding:2px 4px;border-radius:2px;transition:background .1s;
}
.hm-csi-leg:hover,.csi-leg:hover { background:rgba(255,255,255,.06); }
.hm-csi-leg-dot,.csi-leg-dot { width:8px;height:2px;border-radius:1px;flex-shrink:0; }
.hm-csi-leg-lbl,.csi-leg-lbl { color:var(--text2);letter-spacing:.04em; }
.hm-csi-leg-val,.csi-leg-val { color:var(--text);font-weight:600;min-width:42px;text-align:right; }
#hm-csi-tooltip {
  position:absolute;pointer-events:none;z-index:10;
  background:rgba(13,17,23,.95);border:1px solid var(--border2);
  border-radius:4px;padding:7px 10px;font-size:9px;
  font-family:var(--font-mono,monospace);
  min-width:130px;display:none;
}
.hm-csi-tt-date { color:var(--text2);margin-bottom:5px;font-size:9px;letter-spacing:.04em; }
.hm-csi-tt-row  { display:flex;justify-content:space-between;gap:12px;margin-bottom:2px; }
.hm-csi-tt-ccy  { color:var(--text2); }
.hm-csi-tt-val  { font-weight:600; }
#hm-csi-loading {
  position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-size:10px;color:var(--text2);
  font-family:var(--font-mono,monospace);
  letter-spacing:.04em;background:var(--bg);
}

/* ── Source note ── */
.hm-src-note {
  font-size:9px;font-family:var(--font-mono,monospace);color:var(--text2);
  margin-top:10px;padding-top:9px;
  border-top:1px solid var(--border2);
  line-height:1.6;
}

/* ── Footer ── */
#hm-footer {
  padding:8px 18px;
  border-top:1px solid var(--border2);
  display:flex;align-items:center;justify-content:space-between;
  flex-shrink:0;background:var(--bg2);
}
#hm-footer-meta { font-size:9px;font-family:var(--font-mono,monospace);color:var(--text2);letter-spacing:.03em; }

/* ── Mobile ── */
@media (max-width:640px) {
  #hm-modal {
  width:100%!important;max-width:none!important;height:auto!important;max-height:none!important;
  border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;
  background:var(--bg)!important;position:static!important;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:var(--text);
  display:flex;flex-direction:column;
}
  #hm-metrics { grid-template-columns:repeat(3,1fr); }
  .hm-mm { border-bottom:1px solid var(--border2); }
  .hm-mm:nth-child(3),.hm-mm:nth-child(6) { border-right:none; }
  #hm-body { padding:10px; }
  .hm-tbl .col-rng,.hm-tbl .col-prev { display:none; }
}
@media (max-width:520px) {
  .hm-panel { padding:0; }
  .hm-cw { padding:10px; }
  .hm-tbl th,.hm-tbl td { padding:4px 5px; }
  .hm-tbl { font-size:10px; }
  .hm-tbl .col-prev-close { display:none; }
  .imp-bar-bg { width:28px; }
}
`;
  document.head.appendChild(s);

  // ── Currency metadata ────────────────────────────────────────────────────
  const CCY_META = {
    EUR: { flag: 'eu', full: 'Euro' },
    GBP: { flag: 'gb', full: 'Brit. Pound' },
    JPY: { flag: 'jp', full: 'Japanese Yen' },
    AUD: { flag: 'au', full: 'Aus. Dollar' },
    CHF: { flag: 'ch', full: 'Swiss Franc' },
    CAD: { flag: 'ca', full: 'Can. Dollar' },
    NZD: { flag: 'nz', full: 'NZ Dollar' },
    USD: { flag: 'us', full: 'US Dollar' },
    NOK: { flag: 'no', full: 'Norwegian Krone' },
    SEK: { flag: 'se', full: 'Swedish Krona' },
  };

  // All 28 10 G10 currency pair definitions (same as populateHeatmap in dashboard.js)
  const PAIR_DEFS = [
    { id:'eurusd', base:'EUR', quote:'USD', sign:1 },
    { id:'gbpusd', base:'GBP', quote:'USD', sign:1 },
    { id:'audusd', base:'AUD', quote:'USD', sign:1 },
    { id:'nzdusd', base:'NZD', quote:'USD', sign:1 },
    { id:'usdjpy', base:'USD', quote:'JPY', sign:1 },
    { id:'usdchf', base:'USD', quote:'CHF', sign:1 },
    { id:'usdcad', base:'USD', quote:'CAD', sign:1 },
    { id:'eurgbp', base:'EUR', quote:'GBP', sign:1 },
    { id:'eurjpy', base:'EUR', quote:'JPY', sign:1 },
    { id:'eurchf', base:'EUR', quote:'CHF', sign:1 },
    { id:'eurcad', base:'EUR', quote:'CAD', sign:1 },
    { id:'euraud', base:'EUR', quote:'AUD', sign:1 },
    { id:'eurnzd', base:'EUR', quote:'NZD', sign:1 },
    { id:'gbpjpy', base:'GBP', quote:'JPY', sign:1 },
    { id:'gbpchf', base:'GBP', quote:'CHF', sign:1 },
    { id:'gbpcad', base:'GBP', quote:'CAD', sign:1 },
    { id:'gbpaud', base:'GBP', quote:'AUD', sign:1 },
    { id:'gbpnzd', base:'GBP', quote:'NZD', sign:1 },
    { id:'audjpy', base:'AUD', quote:'JPY', sign:1 },
    { id:'audchf', base:'AUD', quote:'CHF', sign:1 },
    { id:'audcad', base:'AUD', quote:'CAD', sign:1 },
    { id:'audnzd', base:'AUD', quote:'NZD', sign:1 },
    { id:'nzdjpy', base:'NZD', quote:'JPY', sign:1 },
    { id:'nzdchf', base:'NZD', quote:'CHF', sign:1 },
    { id:'nzdcad', base:'NZD', quote:'CAD', sign:1 },
    { id:'cadjpy', base:'CAD', quote:'JPY', sign:1 },
    { id:'cadchf', base:'CAD', quote:'CHF', sign:1 },
    { id:'chfjpy', base:'CHF', quote:'JPY', sign:1 },
    // G10 Scandinavian
    { id:'usdnok', base:'USD', quote:'NOK', sign:1 },
    { id:'usdsek', base:'USD', quote:'SEK', sign:1 },
    { id:'eurnok', base:'EUR', quote:'NOK', sign:1 },
    { id:'eursek', base:'EUR', quote:'SEK', sign:1 },
  ];

  // Session windows (UTC hours, start inclusive)
  const SESSIONS = [
    { name:'Sydney',  utcStart:21, utcEnd:6  },
    { name:'Tokyo',   utcStart:0,  utcEnd:9  },
    { name:'London',  utcStart:7,  utcEnd:16 },
    { name:'New York',utcStart:12, utcEnd:21 },
  ];

  // ── State ────────────────────────────────────────────────────────────────
  let _ccy      = null;
  let _strengths = null;
  let _rtCache  = null;
  let _driversCache  = null;   // { generated_at, drivers: { USD: "...", EUR: "...", ... } }
  let _driversFetched = false;
  let _catalystsCache  = null; // { generated_at, currencies: { EUR: { catalyst, sources, updated }, ... } }
  let _catalystsFetched = false;
  let _sessionCtxCache = null; // { generated_at, sessions: { EUR: { Sydney: "...", ... }, ... } }
  let _sessionCtxFetched = false;
  let _sessionCtxIsWeekend = false; // true when session-context.json was generated in closed-market mode

  // CSI state
  let _csiData       = null;  // { dates: [...], series: { EUR: [...], GBP: [...], ... } } — CLOSED sessions only (ohlc-data/*.json)
  let _csiDataLive   = null;  // same shape as _csiData, with one extra live in-progress-session point per ccy appended
  let _csiChart      = null;  // LWC chart instance
  let _csiResizeObs  = null;  // ResizeObserver keeping _csiChart width in sync with #hm-csi-wrap
  let _csiTf         = 'D1';  // H1 | H4 | D1 | W1 — bar/return granularity feeding the accumulated-% line. Derived from _csiRange via _CSI_RANGE_CONFIG (2026-08-07) — not a separately user-facing control, see below.
  let _csiPeriodDays = 91;    // default 3M — literal calendar days back from the series' last date (see _csiCutoffDate). Derived from _csiRange.
  let _csiRange      = '3M';  // the ONE user-facing control (2026-08-07 redesign) — see _CSI_RANGE_CONFIG
  let _csiSeriesMap  = {};    // { EUR: LineSeries, ... } — kept for focal-line styling
  let _csiInited     = false;

  // ── CSI range selector — single control, industry-standard (2026-08-07) ───
  // Earlier same-day iterations of this panel exposed TWO controls: an
  // "Interval" row (H1/H4/1D/1W bar granularity) and a "Range" row
  // (lookback period). The client flagged that this doesn't match how range
  // selectors actually work anywhere in the industry — confirmed against
  // TradingView's own docs: "When users switch a time frame... The chart
  // resolution changes. The bars scale horizontally to cover the entire
  // requested date/time range" (Time-Scale docs) — i.e. every reference
  // implementation (TradingView, Google/Yahoo Finance-style range bars, and
  // CSM-specific tools like FXSSI) exposes exactly ONE control (the range),
  // and picks bar resolution automatically underneath so the chart stays
  // readable. Two independent controls that can both show "1D"/"1W" text
  // for two different questions (candle interval vs. lookback) is not a
  // pattern used anywhere the client or I could find.
  //
  // Fix: back to a single #hm-csi-controls row. Each range button maps to
  // BOTH a lookback (`days`, calendar days via _csiCutoffDate — unchanged
  // from the earlier v2.3.1 fix) and the OHLC resolution needed to render
  // it readably (`tf`) — chosen so every range shows on the order of
  // 25-260 bars, never a handful of dots or thousands of overlapping ones:
  //   1D/1W  → H1 (hourly)     ~24 / ~120 bars
  //   1M     → H4 (4-hourly)   ~180 bars
  //   3M/6M/1Y → D1 (daily)    ~65 / ~130 / ~260 bars
  //   All    → W1 (weekly)     full history, still readable
  // `_csiTf`/`_csiPeriodDays` remain as internal derived state (read by
  // _csiCutoffDate, _renderCSIChart, _renderCSIStats, _updateCSILiveBar —
  // none of that logic needed to change) — they're just no longer set by
  // two independent user clicks, only by csiSetRange() picking one config
  // entry as a unit.
  const _CSI_RANGE_CONFIG = [
    { key: '1D', label: '1D', tf: 'H1', days: 1   },
    { key: '1W', label: '1W', tf: 'H1', days: 7   },
    { key: '1M', label: '1M', tf: 'H4', days: 30  },
    { key: '3M', label: '3M', tf: 'D1', days: 91  },
    { key: '6M', label: '6M', tf: 'D1', days: 182 },
    { key: '1Y', label: '1Y', tf: 'D1', days: 365 },
    { key: 'All',label: 'All',tf: 'W1', days: 0   },
  ];
  const _CSI_TF_TITLE = { H1: 'H1', H4: 'H4', D1: 'DAILY', W1: 'WEEKLY' };

  // Resolve a period-button "days" value into an actual cutoff — the first
  // date/timestamp that should be INCLUDED in the visible window — anchored
  // to the series' own last date (not "today", since the series may lag by
  // one closed session). Returns null for days<=0 (means "show everything").
  // Handles both date-string series ('YYYY-MM-DD', for D1/W1) and
  // unix-second-number series (H1/H4) since _loadCSIData's `time` field
  // type depends on tf.
  function _csiCutoffDate(lastDate, days) {
    if (!days || days <= 0 || lastDate == null) return null;
    if (typeof lastDate === 'number') return lastDate - days * 86400;
    const d = new Date(lastDate + 'T00:00:00Z');
    d.setUTCDate(d.getUTCDate() - days);
    return d.toISOString().slice(0, 10);
  }

  // Fetch currency-drivers.json once per page load (lazy, on first modal open).
  // Falls back silently — the drivers note is additive, never blocking.
  function fetchDrivers() {
    if (_driversFetched) return;
    _driversFetched = true;
    fetch('./ai-analysis/currency-drivers.json?_=' + Date.now())
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && data.drivers && typeof data.drivers === 'object') {
          _driversCache = data;
        }
      })
      .catch(() => { /* silent fallback — drivers are additive */ });
  }

  // Fetch currency-catalysts.json once per page load (lazy, on first modal open).
  // Substantive per-currency catalyst paragraph with named sources (v8.32.0) — distinct
  // from currency-drivers.json (pair-level COT/carry boilerplate repeated across pairs).
  // Falls back silently — the catalyst block is additive, never blocking.
  function fetchCatalysts() {
    if (_catalystsFetched) return;
    _catalystsFetched = true;
    fetch('./ai-analysis/currency-catalysts.json?_=' + Date.now())
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && data.currencies && typeof data.currencies === 'object') {
          _catalystsCache = data;
          // Macro Drivers is now a persistent block rendered synchronously at modal
          // open — if this fetch resolves afterward (the common case), re-render it
          // in place with real data instead of leaving the "not available yet" copy.
          if (_ccy && document.getElementById('hm-bd')?.style.display !== 'none') {
            populateMacroDrivers(_ccy);
          }
        }
      })
      .catch(() => { /* silent fallback — catalyst block is additive */ });
  }

  // Fetch session-context.json once per page load (lazy, on first modal open).
  // Falls back silently — session notes are additive, never blocking.
  // On weekends, the file contains AI-generated recap notes
  // generate_session_context_closed() — same schema, market_closed:true flag added.
  function fetchSessionContext() {
    if (_sessionCtxFetched) return;
    _sessionCtxFetched = true;
    fetch('./ai-analysis/session-context.json?_=' + Date.now())
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && data.sessions && typeof data.sessions === 'object') {
          _sessionCtxCache = data;
          // market_closed flag in JSON tells us notes are weekend recaps —
          // used by populateSession() to apply correct framing regardless of
          // whether the client clock says weekend (handles edge cases at open/close).
          _sessionCtxIsWeekend = !!data.market_closed;
        }
      })
      .catch(() => { /* silent fallback */ });
  }

  // ── Helpers ──────────────────────────────────────────────────────────────
  function fmt2(v) {
    if (v == null || isNaN(v)) return '—';
    const s = v >= 0 ? '+' : '';
    return s + v.toFixed(2) + '%';
  }

  function fmtPrice(v) {
    if (v == null || isNaN(v)) return '—';
    return v >= 100 ? v.toFixed(3) : v.toFixed(5);
  }

  function pctClass(v) {
    if (v == null || isNaN(v)) return 'flat';
    return v > 0 ? 'up' : v < 0 ? 'down' : 'flat';
  }

  // Returns true when the FX market is closed for the weekend.
  // FX convention (industry standard):
  //   Closes:  Friday    21:00 UTC  (New York close)
  //   Opens:   Sunday    21:00 UTC  (Sydney open)
  // UTC day: 0=Sun, 1=Mon, …, 5=Fri, 6=Sat
  function isMarketWeekend() {
    const now  = new Date();
    const day  = now.getUTCDay();
    const hour = now.getUTCHours();
    return (
      day === 6 ||                    // All of Saturday
      (day === 5 && hour >= 21) ||    // Friday from 21:00 UTC onward
      (day === 0 && hour < 21)        // Sunday before 21:00 UTC
    );
  }

  // Returns a Set of all session names that are currently active (handles overlaps).
  // Returns an empty Set during the FX weekend (Fri 21:00 – Sun 21:00 UTC).
  function getActiveSessions() {
    if (isMarketWeekend()) return new Set();
    const h = new Date().getUTCHours();
    const active = new Set();
    // Sydney: 21:00–06:00 UTC (crosses midnight)
    if (h >= 21 || h < 6)  active.add('Sydney');
    // Tokyo: 00:00–09:00 UTC
    if (h >= 0  && h < 9)  active.add('Tokyo');
    // London: 07:00–16:00 UTC
    if (h >= 7  && h < 16) active.add('London');
    // New York: 12:00–21:00 UTC
    if (h >= 12 && h < 21) active.add('New York');
    return active;
  }

  // Legacy helper — returns the single "primary" active session for fallback text.
  function currentSessionName() {
    const active = getActiveSessions();
    for (const s of ['London', 'New York', 'Tokyo', 'Sydney']) {
      if (active.has(s)) return s;
    }
    return 'London';
  }

  // ── Build HTML ───────────────────────────────────────────────────────────
  function buildModal() {
    if (document.getElementById('hm-bd')) return;
    const el = document.createElement('div');
    el.id = 'hm-bd';
    el.setAttribute('role', 'dialog');
    el.setAttribute('aria-modal', 'true');
    el.setAttribute('aria-label', 'Currency Strength Breakdown');
    el.innerHTML = `
<div id="hm-modal">
  <div id="hm-hd">
    <div id="hm-hd-left">
      <div id="hm-title-row">
        <div id="hm-title"></div>
        <button class="hm-ccy-arrow" id="hm-ccy-prev" onclick="hmCycleCcy(-1)" aria-label="Previous currency" title="Previous (←)">‹</button>
        <div id="hm-ccy-switch">
          <button id="hm-ccy-chip" onclick="hmToggleCcyDropdown(event)" aria-haspopup="listbox" aria-expanded="false" title="Switch currency"></button>
          <div id="hm-ccy-dd" role="listbox" aria-label="Select currency"></div>
        </div>
        <button class="hm-ccy-arrow" id="hm-ccy-next" onclick="hmCycleCcy(1)" aria-label="Next currency" title="Next (→)">›</button>
      </div>
      <div id="hm-sub">G10 composite · 32 pairs · Delayed ~5min</div>
    </div>
    <button id="hm-close" aria-label="Close" title="Close">&#10005;</button>
  </div>
  <div id="hm-metrics">
    <div class="hm-mm">
      <div class="hm-mm-lbl">Composite</div>
      <div class="hm-mm-val" id="hm-m-composite">—</div>
      <div class="hm-mm-sub" id="hm-m-comp-sub">avg vs 7 pairs</div>
    </div>
    <div class="hm-mm">
      <div class="hm-mm-lbl">1W Strength</div>
      <div class="hm-mm-val" id="hm-m-1w">—</div>
      <div class="hm-mm-sub" id="hm-m-1w-sub">vs prior Fri</div>
    </div>
    <div class="hm-mm">
      <div class="hm-mm-lbl">Rank</div>
      <div class="hm-mm-val flat" id="hm-m-rank">—</div>
      <div class="hm-mm-sub">of 10 G10 currencies</div>
    </div>
    <div class="hm-mm">
      <div class="hm-mm-lbl">Pairs won</div>
      <div class="hm-mm-val flat" id="hm-m-won">—</div>
      <div class="hm-mm-sub">gaining vs</div>
    </div>
    <div class="hm-mm">
      <div class="hm-mm-lbl">Strongest vs</div>
      <div class="hm-mm-val sm flat" id="hm-m-strong">—</div>
      <div class="hm-mm-sub up" id="hm-m-strong-sub">—</div>
    </div>
    <div class="hm-mm">
      <div class="hm-mm-lbl">Weakest vs</div>
      <div class="hm-mm-val sm flat" id="hm-m-weak">—</div>
      <div class="hm-mm-sub down" id="hm-m-weak-sub">—</div>
    </div>
  </div>
  <div class="hm-cw" id="hm-macro">
    <div class="hm-ct" id="hm-catalyst-title">MACRO DRIVERS</div>
    <div id="hm-catalyst"></div>
  </div>
  <div id="hm-tabs" role="tablist" aria-label="Heatmap breakdown tabs">
    <button class="hm-tab on" role="tab" aria-selected="true"  data-tab="breakdown"    onclick="hmTab(this,'breakdown')">Pair Breakdown</button>
    <button class="hm-tab"    role="tab" aria-selected="false" data-tab="session"      onclick="hmTab(this,'session')">Session</button>
    <button class="hm-tab"    role="tab" aria-selected="false" data-tab="correlations" onclick="hmTab(this,'correlations')">Rel. Strength</button>
    <button class="hm-tab"    role="tab" aria-selected="false" data-tab="csi"          onclick="hmTab(this,'csi')">CSI</button>
  </div>
  <div id="hm-body">
    <div class="hm-panel on" id="hm-p-breakdown">
      <div class="hm-cw">
        <div class="hm-ct" id="hm-pairs-title">DIRECT PAIRS · DAY % &amp; 1W % · vs PREV CLOSE / PREV FRIDAY</div>
        <table class="hm-tbl" aria-label="Direct pairs for selected currency">
          <thead>
            <tr>
              <th scope="col">Pair</th>
              <th scope="col">Close</th>
              <th scope="col" class="col-prev-close">Prev close</th>
              <th scope="col">Day %</th>
              <th scope="col">1W %</th>
              <th scope="col" title="Relative contribution vs peers — bar width = magnitude">Contribution</th>
              <th scope="col">Session range</th>
            </tr>
          </thead>
          <tbody id="hm-pair-tbody"></tbody>
        </table>
      </div>
      <div class="hm-cw">
        <div class="hm-ct">FULL RANKING · ALL 10 G10 CURRENCIES · COMPOSITE STRENGTH</div>
        <div style="display:flex;gap:16px;">
          <div style="flex:1;">
            <div class="hm-rank-sublbl">Day %</div>
            <div id="hm-ranking-rows"></div>
          </div>
          <div style="flex:1;">
            <div class="hm-rank-sublbl">1W % · vs prior Fri</div>
            <div id="hm-ranking-1w-rows"></div>
          </div>
        </div>
      </div>
    </div>
    <div class="hm-panel" id="hm-p-session">
      <div class="hm-cw">
        <div class="hm-ct" id="hm-sess-title">COMPOSITE STRENGTH BY SESSION</div>
        <div id="hm-sess-content"></div>
      </div>
      <div class="hm-cw">
        <div class="hm-ct">SESSION CONTEXT</div>
        <div id="hm-sess-notes" style="font-size:11px;color:var(--text2,#787b86);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);line-height:1.7;"></div>
      </div>
      <div class="hm-cw" style="flex:1;overflow:hidden;display:flex;flex-direction:column;min-height:140px;">
        <div class="hm-ct" id="hm-sess-news-title">MARKET COMMENTARY</div>
        <div id="hm-sess-news" class="hm-news-wrap">
          <div class="hm-news-loading">Loading market commentary…</div>
        </div>
      </div>
    </div>
    <div class="hm-panel" id="hm-p-correlations">
      <div class="hm-cw" style="flex:1;overflow:hidden;display:flex;flex-direction:column;">
        <div class="hm-ct">RELATIVE STRENGTH DIFFERENTIAL · ALL 10 G10 · % COMPOSITE vs PREV CLOSE</div>
        <div style="font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono);margin:-2px 0 6px;">Click a currency to pivot this panel</div>
        <div id="hm-corr-matrix" style="flex:1;overflow:hidden;display:flex;flex-direction:column;min-height:0;"></div>
      </div>
      <div class="hm-cw">
        <div class="hm-ct" id="hm-drivers-title">STRENGTH DRIVERS · TOP 3 PAIRS BY CONTRIBUTION</div>
        <div id="hm-drivers"></div>
      </div>
    </div>
    <div class="hm-panel" id="hm-p-csi">
      <div class="hm-cw">
        <div class="hm-ct" id="hm-csi-title">CURRENCY STRENGTH INDEX · ACCUMULATED % RETURN · DAILY OHLC</div>
        <div id="hm-csi-controls">
          <button class="hm-csi-btn" data-range="1D"  onclick="csiSetRange(this,'1D')"  title="1 Day, hourly">1D</button>
          <button class="hm-csi-btn" data-range="1W"  onclick="csiSetRange(this,'1W')"  title="1 Week, hourly">1W</button>
          <button class="hm-csi-btn" data-range="1M"  onclick="csiSetRange(this,'1M')"  title="1 Month, 4-hourly">1M</button>
          <button class="hm-csi-btn on" data-range="3M"  onclick="csiSetRange(this,'3M')"  title="3 Months, daily">3M</button>
          <button class="hm-csi-btn" data-range="6M"  onclick="csiSetRange(this,'6M')"  title="6 Months, daily">6M</button>
          <button class="hm-csi-btn" data-range="1Y"  onclick="csiSetRange(this,'1Y')"  title="1 Year, daily">1Y</button>
          <button class="hm-csi-btn" data-range="All" onclick="csiSetRange(this,'All')" title="Full history, weekly">All</button>
        </div>
        <div id="hm-csi-wrap">
          <div id="hm-csi-loading">Loading OHLC data…</div>
          <div id="hm-csi-chart"></div>
          <div id="hm-csi-tooltip"></div>
        </div>
        <div id="hm-csi-legend"></div>
      </div>
      <div class="hm-cw">
        <div class="hm-ct" id="hm-csi-stats-title">CSI SNAPSHOT · CURRENT PERIOD</div>
        <div id="hm-csi-stats"></div>
      </div>
    </div>
  </div>
  <div id="hm-footer">
    <div id="hm-footer-meta">Delayed ~5min · G10 composite · 32 pairs</div>
  </div>
</div>`;
    document.body.appendChild(el);
    requestAnimationFrame(()=>requestAnimationFrame(()=>{ el.scrollIntoView({behavior:'smooth',block:'start'}); }));
    document.getElementById('hm-close').addEventListener('click', closeHeatmapModal);
    el.addEventListener('click', function(e) {
      if (e.target === el) { closeHeatmapModal(); return; }
      // Click outside the currency switcher closes its dropdown without closing the modal
      if (!e.target.closest('#hm-ccy-switch')) _hmCloseCcyDropdown();
    });
    document.addEventListener('keydown', _onKey);
  }

  function _onKey(e) {
    if (e.key === 'Escape') {
      const dd = document.getElementById('hm-ccy-dd');
      if (dd && dd.classList.contains('open')) { _hmCloseCcyDropdown(); return; }
      closeHeatmapModal();
      return;
    }
    if (e.key === 'ArrowLeft')  { window.hmCycleCcy(-1); return; }
    if (e.key === 'ArrowRight') { window.hmCycleCcy(1); return; }
  }

  // ── Populate ─────────────────────────────────────────────────────────────
  function populateMetrics(ccy, strengths, rtCache) {
    const sorted = [...strengths].sort((a,b) => b.pct - a.pct);
    const rank   = sorted.findIndex(s => s.ccy === ccy) + 1;
    const self   = strengths.find(s => s.ccy === ccy);
    if (!self) return;

    // Pairs for this currency
    const myPairs = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
    let won = 0;
    let bestPair = null, bestPct = -Infinity;
    let worstPair = null, worstPct = Infinity;

    myPairs.forEach(p => {
      const d = rtCache[p.id];
      if (!d || d.pct == null) return;
      const impact = d.pct * p.sign * (p.base === ccy ? 1 : -1);
      if (impact > 0) won++;
      const opp = p.base === ccy ? p.quote : p.base;
      if (impact > bestPct)  { bestPct  = impact; bestPair  = { pair: p.id.toUpperCase(), opp, pct: impact }; }
      if (impact < worstPct) { worstPct = impact; worstPair = { pair: p.id.toUpperCase(), opp, pct: impact }; }
    });

    const compositeEl    = document.getElementById('hm-m-composite');
    const compositeSubEl = document.getElementById('hm-m-comp-sub');
    const v = self.pct;
    compositeEl.textContent = fmt2(v);
    compositeEl.className   = 'hm-mm-val ' + pctClass(v);

    // Count how many pairs actually contributed to this currency's composite
    const compPairCnt = myPairs.filter(p => {
      const d = rtCache[p.id];
      return d && d.pct != null;
    }).length;
    if (compositeSubEl) {
      compositeSubEl.textContent = compPairCnt + ' pair' + (compPairCnt !== 1 ? 's' : '') + ' · intraday';
    }

    // 1W composite — same equal-weighted model but using pct1w per pair
    let w1sum = 0, w1n = 0;
    myPairs.forEach(p => {
      const d = rtCache[p.id];
      if (!d || d.pct1w == null) return;
      const impact1w = d.pct1w * p.sign * (p.base === ccy ? 1 : -1);
      w1sum += impact1w;
      w1n++;
    });
    const w1El    = document.getElementById('hm-m-1w');
    const w1SubEl = document.getElementById('hm-m-1w-sub');
    if (w1n > 0) {
      const w1avg = w1sum / w1n;
      w1El.textContent    = fmt2(w1avg);
      w1El.className      = 'hm-mm-val ' + pctClass(w1avg);
      w1SubEl.textContent = w1n + ' pairs · vs prior Fri';
    } else {
      w1El.textContent    = '—';
      w1El.className      = 'hm-mm-val flat';
      w1SubEl.textContent = 'no data';
    }

    document.getElementById('hm-m-rank').textContent = '#' + rank + ' / ' + sorted.length;
    document.getElementById('hm-m-won').textContent  = won + ' / ' + myPairs.length;

    const strongEl    = document.getElementById('hm-m-strong');
    const strongSubEl = document.getElementById('hm-m-strong-sub');
    const weakEl      = document.getElementById('hm-m-weak');
    const weakSubEl   = document.getElementById('hm-m-weak-sub');

    if (bestPair) {
      strongEl.textContent    = bestPair.opp;
      strongSubEl.textContent = fmt2(bestPair.pct);
    }
    if (worstPair) {
      weakEl.textContent    = worstPair.opp;
      weakSubEl.textContent = fmt2(worstPair.pct);
    }
  }

  // _skipAnim: true when called from _updateBreakdownRT on sort-order changes
  //            (modal already open — bars should appear at target width, not animate from 0)
  function populateBreakdown(ccy, strengths, rtCache, _skipAnim) {
    document.getElementById('hm-pairs-title').textContent =
      ccy + ' DIRECT PAIRS · INTRADAY % CHANGE · vs PREV CLOSE';

    const myPairs = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
    const impacts = [];

    myPairs.forEach(p => {
      const d = rtCache[p.id];
      const isCcyBase = p.base === ccy;
      const opp = isCcyBase ? p.quote : p.base;
      const rawPct = d?.pct ?? null;
      // impact on the selected ccy: positive = ccy gained vs opp
      const impact = rawPct != null ? rawPct * p.sign * (isCcyBase ? 1 : -1) : null;
      // 1W impact — same sign convention as intraday
      const raw1w  = d?.pct1w ?? null;
      const imp1w  = raw1w != null ? raw1w * p.sign * (isCcyBase ? 1 : -1) : null;
      const close  = isCcyBase ? (d?.close ?? null) : (d?.close != null ? 1/d.close : null);
      const open   = isCcyBase ? (d?.open  ?? null) : (d?.open  != null ? 1/d.open  : null);
      const hi     = isCcyBase ? (d?.high  ?? null) : (d?.high  != null ? 1/d.high  : null);
      const lo     = isCcyBase ? (d?.low   ?? null) : (d?.low   != null ? 1/d.low   : null);
      const label  = isCcyBase
        ? (p.base + '/' + p.quote)
        : (p.quote + '/' + p.base);   // show ccy first
      impacts.push({ label, opp, close, open, hi, lo, impact, rawPct, imp1w });
    });

    // Sort: biggest positive impact first
    impacts.sort((a,b) => (b.impact??-99) - (a.impact??-99));
    const maxImp = Math.max(...impacts.map(i => Math.abs(i.impact ?? 0)), 0.001);

    const tbody = document.getElementById('hm-pair-tbody');
    tbody.innerHTML = impacts.map((r, _i) => {
      const iCls  = pctClass(r.impact);
      const rng   = (r.hi != null && r.lo != null)
        ? fmtPrice(r.lo) + ' – ' + fmtPrice(r.hi)
        : '—';
      const barW  = r.impact != null ? Math.round(Math.abs(r.impact)/maxImp*100) : 0;
      const barClr = r.impact != null && r.impact >= 0 ? 'var(--up,#26a69a)' : 'var(--down,#ef5350)';
      return `<tr data-pair="${r.label}">
        <td><span class="sym">${r.label}</span></td>
        <td data-cell="close">${fmtPrice(r.close)}</td>
        <td class="col-prev-close" data-cell="open">${fmtPrice(r.open)}</td>
        <td class="${iCls}" data-cell="impact">${fmt2(r.impact)}</td>
        <td class="${pctClass(r.imp1w)}" data-cell="imp1w">${r.imp1w != null ? fmt2(r.imp1w) : '—'}</td>
        <td><div class="imp-wrap" title="${fmt2(r.impact)} vs peers">
          <div class="imp-bar-bg"><div class="imp-bar-fill" data-cell="bar" style="width:${barW}%;background:${barClr}"></div></div>
        </div></td>
        <td style="font-size:9px;color:var(--text3)" data-cell="rng">${rng}</td>
      </tr>`;
    }).join('');

    // Ranking
    const sorted   = [...strengths].sort((a,b) => b.pct - a.pct);
    const maxAbsPct = Math.max(...sorted.map(s => Math.abs(s.pct)), 0.001);
    const container = document.getElementById('hm-ranking-rows');
    container.innerHTML = '';
    sorted.forEach(s => {
      const isHL  = s.ccy === ccy;
      const cls   = isHL ? 'hl' : pctClass(s.pct);
      const fillW = Math.round(Math.abs(s.pct) / maxAbsPct * 100);
      const row   = document.createElement('div');
      row.className = 'hm-rank-row';
      row.dataset.rankCcy = s.ccy;
      // On RT rebuilds (_skipAnim) set final width directly — no 0→target animation that causes flash
      const initW = _skipAnim ? fillW + '%' : '0';
      row.innerHTML = `
        <div class="hm-rank-ccy${isHL?' hl':''}">${s.ccy}</div>
        <div class="hm-rank-bg">
          <div class="hm-rank-fill ${cls}" style="width:${initW}" data-w="${fillW}"></div>
        </div>
        <div class="hm-rank-val ${pctClass(s.pct)}" data-rank-val>${fmt2(s.pct)}</div>`;
      container.appendChild(row);
    });
    // Only run the entry animation on first open (not on RT sort-order rebuilds)
    if (!_skipAnim) {
      requestAnimationFrame(() => {
        container.querySelectorAll('.hm-rank-fill').forEach(el => {
          el.style.width = el.dataset.w + '%';
        });
      });
    }

    // 1W Ranking — compute G10 composite weekly strength from pct1w across all G10 pairs
    const ccys = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD','USD','NOK','SEK'];
    const w1map = {};
    ccys.forEach(c => { w1map[c] = { sum: 0, n: 0 }; });
    PAIR_DEFS.forEach(p => {
      const d = rtCache[p.id];
      if (!d || d.pct1w == null) return;
      const v = d.pct1w * p.sign;   // positive = base strengthened vs quote
      w1map[p.base].sum += v;  w1map[p.base].n++;
      w1map[p.quote].sum -= v; w1map[p.quote].n++;
    });
    const w1strengths = ccys
      .map(c => ({ ccy: c, pct: w1map[c].n > 0 ? w1map[c].sum / w1map[c].n : null }))
      .filter(s => s.pct != null)
      .sort((a, b) => b.pct - a.pct);

    const cont1w = document.getElementById('hm-ranking-1w-rows');
    cont1w.innerHTML = '';
    if (w1strengths.length > 0) {
      const maxAbs1w = Math.max(...w1strengths.map(s => Math.abs(s.pct)), 0.001);
      w1strengths.forEach(s => {
        const isHL  = s.ccy === ccy;
        const cls   = isHL ? 'hl' : pctClass(s.pct);
        const fillW = Math.round(Math.abs(s.pct) / maxAbs1w * 100);
        const row   = document.createElement('div');
        row.className = 'hm-rank-row';
        row.dataset.rankCcy = s.ccy;
        const initW = _skipAnim ? fillW + '%' : '0';
        row.innerHTML = `
          <div class="hm-rank-ccy${isHL?' hl':''}">${s.ccy}</div>
          <div class="hm-rank-bg">
            <div class="hm-rank-fill ${cls}" style="width:${initW}" data-w="${fillW}"></div>
          </div>
          <div class="hm-rank-val ${pctClass(s.pct)}" data-rank-val>${fmt2(s.pct)}</div>`;
        cont1w.appendChild(row);
      });
      if (!_skipAnim) {
        requestAnimationFrame(() => {
          cont1w.querySelectorAll('.hm-rank-fill').forEach(el => {
            el.style.width = el.dataset.w + '%';
          });
        });
      }
    } else {
      cont1w.innerHTML = '<div style="font-size:10px;color:var(--text3);padding:6px 0">No 1W data available</div>';
    }
  }

  // Convert a UTC hour to local HH:MM string (respects user's timezone)
  function utcHourToLocalStr(utcHour) {
    const d = new Date();
    d.setUTCHours(utcHour, 0, 0, 0);
    return d.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', hour12: false });
  }

  // Returns the user's timezone abbreviation (e.g. "EST", "GMT+3")
  function localTzAbbr() {
    return new Date().toLocaleTimeString('en', { timeZoneName: 'short' }).split(' ').pop() || 'LT';
  }

  // Replaces all "HH:MM UTC" patterns in a Groq-generated note with the user's
  // local equivalent. E.g. "PMI at 12:00 UTC" becomes "PMI at 09:00 GMT-3".
  function convertUtcTimesInNote(text) {
    if (!text) return text;
    const tzAbbr = localTzAbbr();
    return text.replace(/\b(\d{1,2}):(\d{2})\s*UTC\b/g, function(_, hh, mm) {
      const d = new Date();
      d.setUTCHours(parseInt(hh, 10), parseInt(mm, 10), 0, 0);
      const local = d.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', hour12: false });
      return local + ' ' + tzAbbr;
    });
  }

  // Returns the temporal state of a session at the current UTC hour:
  //   'active'   — session is currently open
  //   'past'     — session opened and closed earlier today (result is real)
  //   'upcoming' — session has not yet opened today
  // Industry convention (Bloomberg FXGO, Refinitiv Eikon):
  //   Bars and values are shown for active and past sessions only.
  //   Upcoming sessions show a placeholder track — no fabricated data.
  function getBarSessionState(sess) {
    if (isMarketWeekend()) return 'past'; // weekend: all bars show last-close (dimmed)
    const h = new Date().getUTCHours();
    const isActive = getActiveSessions().has(sess.name);
    if (isActive) return 'active';
    // Sydney crosses midnight: closed when 06:00 <= h < 21:00
    if (sess.name === 'Sydney') return (h >= 6 && h < 21) ? 'past' : 'upcoming';
    // All other sessions: past if current hour is past their close, upcoming if before their open
    return h >= sess.utcEnd ? 'past' : 'upcoming';
  }

  // Returns the fraction [0,1] of how far the CURRENT trading window has
  // progressed for an active session — minute-resolution, handles the
  // midnight-crossing Sydney window. Meaningless (but harmless) for a
  // session that isn't currently active; callers only use this for 'active'.
  function getSessionProgress(sess) {
    const now    = new Date();
    const nowMin = now.getUTCHours() * 60 + now.getUTCMinutes();
    let startMin = sess.utcStart * 60;
    let endMin   = sess.utcEnd   * 60;
    if (endMin <= startMin) endMin += 24 * 60; // crosses midnight (Sydney)
    let elapsedMin = nowMin - startMin;
    if (elapsedMin < 0) elapsedMin += 24 * 60; // wrap: now is past midnight, before startMin numerically
    const durationMin = endMin - startMin;
    return Math.max(0, Math.min(1, elapsedMin / durationMin));
  }

  // Reorders SESSIONS into a chronological-narrative sequence — past sessions
  // first (oldest to most recent), then the currently active session(s), then
  // upcoming sessions — instead of the fixed Sydney→Tokyo→London→New York
  // cycle order. Fixed order breaks down as a reading order once a session
  // has wrapped past midnight: e.g. once Sydney reopens for a new day while
  // Tokyo/London/New York show as CLOSED from the *previous* cycle, listing
  // Sydney first makes the closed sessions below it read as upcoming/future
  // rather than the history that already happened. Bloomberg/Eikon session
  // panels read left-to-right as a timeline (what already happened → what's
  // live now → what's next); this keeps the bars and the AI session notes
  // consistent with that convention. Relative order within each group is
  // preserved from the fixed cycle. Weekend: no active/past distinction
  // (all bars are last-close), so the fixed order is left untouched.
  function getOrderedSessions() {
    if (isMarketWeekend()) return SESSIONS;
    const past = [], active = [], upcoming = [];
    SESSIONS.forEach(s => {
      const state = getBarSessionState(s);
      if (state === 'active') active.push(s);
      else if (state === 'past') past.push(s);
      else upcoming.push(s);
    });
    return past.concat(active, upcoming);
  }

  // Session tab — Market Commentary fill block (v2.6.0). Same source and shape
  // as cb-rates-modal.js's _cbrLoadPolicySummary(): fetches news-data/news.json,
  // filters to this currency's articles, renders title + expand paragraph for
  // up to 3. Unlike the CB modal it doesn't filter by CB_KW — this tab is about
  // the currency generally, not central-bank policy specifically.
  //
  // v2.6.0 fix: _hmRefreshIfOpen calls populateSession() on every Finnhub RT
  // tick while the Session tab is active (intentionally — see that function's
  // own "flash-free" comment), which previously called this function on every
  // tick too, replacing the wrap's innerHTML with a loading spinner and then
  // the same articles over and over — the visible flicker the client reported.
  // News doesn't change tick-to-tick, so this now only fetches/re-renders when
  // the currency actually changes (tracked via _hmNewsCcy); repeat calls for
  // the same currency are a no-op.
  let _hmNewsCcy = null;
  async function _hmLoadSessionNews(ccy) {
    if (ccy === _hmNewsCcy) return; // same currency already loaded/loading — skip, no flicker
    _hmNewsCcy = ccy;
    const wrap = document.getElementById('hm-sess-news');
    if (!wrap) return;
    wrap.innerHTML = '<div class="hm-news-loading">Loading market commentary…</div>';
    try {
      const res = await fetch('./news-data/news.json', { cache: 'no-store' }).catch(() => null);
      if (!res || !res.ok) throw new Error('fetch failed');
      const j = await res.json();
      if (ccy !== _hmNewsCcy) return; // stale response guard — ccy moved on again while this was in flight
      const articles = (j.articles || [])
        .filter(a => {
          if (a.cur !== ccy) return false;
          const exp = (a.expand || '').replace(/<[^>]+>/g, '').trim();
          return exp.length >= 80;
        })
        .sort((a, b) => (b.ts || 0) - (a.ts || 0))
        .slice(0, 3);

      if (!articles.length) {
        wrap.innerHTML = '<div class="hm-news-empty">No ' + ccy + ' commentary available.</div>';
        return;
      }

      wrap.innerHTML = articles.map(a => {
        const timeStr = [a.date, a.time].filter(Boolean).join(' \u00b7 ');
        let body = (a.expand || '').replace(/&#\d+;/g, '').replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
        if (body.length > 400) {
          const cut = body.slice(0, 400);
          const lastPeriod = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('? '), cut.lastIndexOf('! '));
          body = (lastPeriod > 150 ? cut.slice(0, lastPeriod + 1) : cut) + '\u2026';
        }
        const titleHtml = a.link
          ? '<a href="' + a.link + '" target="_blank" rel="noopener noreferrer">' + (a.title || '') + '</a>'
          : (a.title || '');
        return '<div class="hm-news-article">' +
          '<div class="hm-news-art-meta">' +
          '<span class="hm-news-art-source">' + (a.source || '') + '</span>' +
          '<span class="hm-news-art-time">' + timeStr + '</span>' +
          '</div>' +
          '<div class="hm-news-art-title">' + titleHtml + '</div>' +
          '<div class="hm-news-art-body">' + body + '</div>' +
          '</div>';
      }).join('');
    } catch (e) {
      if (ccy !== _hmNewsCcy) return;
      wrap.innerHTML = '<div class="hm-news-empty">Market commentary unavailable.</div>';
    }
  }

  function populateSession(ccy, rtCache) {
    const tzAbbr   = localTzAbbr();
    const weekend  = isMarketWeekend();
    document.getElementById('hm-sess-title').textContent =
      ccy + ' INTRADAY COMPOSITE · SESSION WINDOW STATUS · ' + tzAbbr;
    const newsTitleEl = document.getElementById('hm-sess-news-title');
    if (newsTitleEl) newsTitleEl.textContent = 'MARKET COMMENTARY \u00b7 ' + ccy;
    _hmLoadSessionNews(ccy); // fill-space block below Session Context — non-blocking

    const myPairs      = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
    const activeSessions = getActiveSessions();   // empty Set on weekends
    const activeSess   = currentSessionName();    // legacy fallback text

    // Compute the single intraday composite once — this is the day % vs prev close
    // weighted equally across all direct pairs. Bloomberg convention: when session-specific
    // OHLC is not available, show the full-day composite alongside session window status
    // rather than fabricating per-session values from volume weights.
    let compositeSum = 0, compositeCnt = 0;
    myPairs.forEach(p => {
      const d = rtCache[p.id];
      if (!d || d.pct == null) return;
      compositeSum += d.pct * p.sign * (p.base === ccy ? 1 : -1);
      compositeCnt++;
    });
    const dayComposite = compositeCnt > 0 ? compositeSum / compositeCnt : null;

    // Volume share labels — used as context markers, not bar values
    const volShare = { 'New York': '38%', 'London': '35%', 'Tokyo': '18%', 'Sydney': '9%' };

    // Session bar data: active and past sessions show the day composite (honest label).
    // Upcoming sessions show no bar — Bloomberg/Eikon do not fabricate forward values.
    const sessionData = getOrderedSessions().map(sess => {
      const barState = getBarSessionState(sess);
      const showBar  = barState === 'active' || barState === 'past';
      // All shown sessions display the same day composite — this is transparent about
      // data availability. The session window context is conveyed by the state indicator
      // and AI notes, not by fabricated per-session performance figures.
      const pct = showBar ? dayComposite : null;
      return { ...sess, pct, barState, isActive: barState === 'active' };
    });

    const grid = document.createElement('div');
    grid.className = 'sess-grid';

    // Bar color: --blue (terminal design system) — active at full opacity, past dimmed.
    // This matches Proposal A: bars represent session window status, not directional sign.
    const compositePos = dayComposite != null && dayComposite >= 0;
    const barClr = getComputedStyle(document.documentElement).getPropertyValue('--blue').trim() || '#4f7fff';

    sessionData.forEach(s => {
      const lbl = document.createElement('div');
      lbl.className = 'sess-lbl';

      let labelText = s.name.toUpperCase();
      if (s.barState === 'active') {
        labelText += ' \u25CF';        // ● active
      } else if (s.barState === 'upcoming' && !weekend) {
        labelText += ' \u25CB';        // ○ upcoming
      }
      lbl.textContent = labelText;

      const track = document.createElement('div');
      track.className = 'sess-track';

      const val = document.createElement('div');

      if (s.barState === 'upcoming' || s.pct == null) {
        // Upcoming: empty track, no value — Bloomberg/Eikon show no forward bar
        lbl.style.cssText = 'opacity:.35;color:var(--orange,#f6941c)';
        track.style.opacity = '0.08';
        val.className = 'sess-val flat';
        val.style.cssText = 'opacity:.35;font-size:9px;color:var(--text3,#6b7280)';
        val.textContent = utcHourToLocalStr(s.utcStart);  // show open time as hint
      } else {
        const fill = document.createElement('div');
        fill.className = 'sess-fill';
        // Active: bar fills proportionally to elapsed time within the session
        // window — matches the MT5 EA's on-chart session indicator (live
        // session, result still in progress). Past: full-width dimmed bar —
        // closed session, final result (Bloomberg convention). Weekend: all
        // bars full-width dimmed (last-close convention).
        const isActive  = s.barState === 'active' && !weekend;
        const dimBar    = !isActive;
        const fillWidth = isActive ? (getSessionProgress(s) * 100).toFixed(1) + '%' : '100%';
        fill.style.cssText = 'width:' + fillWidth + ';background:' + barClr +
          (dimBar ? ';opacity:.30' : ';opacity:.70');
        track.appendChild(fill);
        val.className = 'sess-val ' + (compositePos ? 'up' : 'down');
        val.textContent = fmt2(s.pct);
        if (dimBar) {
          val.style.opacity = '0.55';
          lbl.style.opacity = '0.55';
        }
      }

      grid.appendChild(lbl);
      grid.appendChild(track);
      grid.appendChild(val);
    });

    // Data note: explain the bars represent full-day composite (institutional transparency)
    const dataNote = document.createElement('div');
    dataNote.style.cssText = 'font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono,\'JetBrains Mono\',\'Courier New\',monospace);letter-spacing:.02em;margin-top:6px;opacity:.7';
    dataNote.textContent = 'Day % vs prev close \xb7 session-specific OHLC not available';

    // Weekend: no banner, no status line — dimmed bars only (Bloomberg/Eikon convention)
    const content = document.getElementById('hm-sess-content');
    content.innerHTML = '';
    content.appendChild(grid);
    content.appendChild(dataNote);

    // Session context notes — suspended on weekends; otherwise show Groq or fallback
    const notes = document.getElementById('hm-sess-notes');
    const _now    = new Date();
    const localHH = String(_now.getHours()).padStart(2,'0');
    const localMM = String(_now.getMinutes()).padStart(2,'0');
    const localStr = localHH + ':' + localMM;

    if (weekend) {
      // Weekend: show recap notes from session-context.json (generated by
      // generate_session_context_closed() — Bloomberg weekend desk note style).
      // If Groq notes are available, display them with "WEEKLY RECAP" framing.
      // Each note maps to: Sydney=Friday close, Tokyo=weekly range,
      // London=main catalyst, New York=Monday outlook.
      const groqSessions = _sessionCtxCache && _sessionCtxCache.sessions
        ? _sessionCtxCache.sessions[ccy]
        : null;

      if (groqSessions && Object.keys(groqSessions).length >= 3) {
        const weekendLabels = {
          'Sydney':   'FRI CLOSE',
          'Tokyo':    'WEEKLY',
          'London':   'CATALYST',
          'New York': 'MON OPEN',
        };
        const sessOrder = ['Sydney', 'Tokyo', 'London', 'New York'];
        notes.innerHTML = sessOrder.map(sName => {
          const note   = convertUtcTimesInNote(groqSessions[sName] || '—');
          const wLabel = weekendLabels[sName] || sName.toUpperCase();
          return (
            '<div style="margin-bottom:5px">' +
            '<span style="color:var(--text3,#6b7280);min-width:72px;display:inline-block;' +
            'font-size:9px;letter-spacing:.04em;font-weight:600">' +
            wLabel + '</span> ' +
            '<span style="color:var(--text2,#787b86)">' + note + '</span>' +
            '</div>'
          );
        }).join('') +
        '<div style="margin-top:8px;font-size:9px;color:var(--text3,#6b7280);' +
        'font-family:var(--font-mono);letter-spacing:.03em;">' +
        'AI Analytics \xb7 Weekly recap \xb7 Resumes at Sunday 21:00 UTC</div>';
      } else {
        // Groq notes not yet available for weekend (first run after Friday close)
        notes.innerHTML =
          '<div style="font-size:10px;color:var(--text3,#6b7280);' +
          'font-family:var(--font-mono,\'JetBrains Mono\',\'Courier New\',monospace);line-height:1.6;">' +
          'Weekly recap generating\u2026 Check back shortly.' +
          '<br>Session context resumes at Sunday 21:00 UTC (Sydney open).' +
          '</div>';
      }
      return;
    }

    // Weekday: check if Groq session context is available for this currency
    const groqSessions = _sessionCtxCache && _sessionCtxCache.sessions
      ? _sessionCtxCache.sessions[ccy]
      : null;

    if (groqSessions && Object.keys(groqSessions).length >= 3) {
      // Render session notes using getBarSessionState for consistent classification
      // with the bar section above (same UTC boundary logic, single source of truth).
      // Industry convention (Bloomberg FXGO, Refinitiv Eikon):
      //   active   — blue label + ● + full-brightness AI note (live session)
      //   past     — gray label + AI note dimmed + CLOSED badge (result, historical fact)
      //   upcoming — amber label + ○ + "opens HH:MM" placeholder, no AI note
      //              AI-generated text for a future session is an outlook written at
      //              06:00 UTC; showing it at full brightness before the session opens
      //              makes a forward projection read as an accomplished result.
      // Session notes: state-first layout (Bloomberg convention)
      //   LIVE chip     — blue, session currently open
      //   CLOSED chip   — muted gray on its own line above note (result is historical fact)
      //   UPCOMING chip — amber, session not yet open; AI note suppressed to prevent
      //                   outlook text from reading as accomplished result
      // Order matches the bars above: past → active → upcoming (see
      // getOrderedSessions()), not the fixed Sydney→Tokyo→London→New York
      // cycle — keeps closed sessions from a prior cycle reading as future
      // notes once a new session (e.g. Sydney) has reopened.
      const sessOrder = getOrderedSessions().map(s => s.name);
      notes.innerHTML = sessOrder.map(sName => {
        const sess  = SESSIONS.find(s => s.name === sName);
        const state = getBarSessionState(sess);
        const aiNote = convertUtcTimesInNote(groqSessions[sName] || '\u2014');

        const labelColor = state === 'active'   ? 'var(--blue,#4f7fff)'
                         : state === 'past'      ? 'var(--text3,#6b7280)'
                         :                         'var(--orange,#f6941c)';
        const textColor  = state === 'active'   ? 'var(--text,#d1d4dc)'
                         : state === 'past'      ? 'var(--text3,#6b7280)'
                         :                         'var(--text3,#6b7280)';
        const labelDot   = state === 'active'   ? ' \u25CF'    // ●
                         : state === 'upcoming' ? ' \u25CB'    // ○
                         :                        '';

        // State chip on the header line: unambiguous before reading the note text
        const stateChip  = state === 'active'
          ? '<span style="font-size:8px;background:rgba(79,127,255,.15);color:var(--blue,#4f7fff);border-radius:2px;padding:1px 4px;letter-spacing:.07em;font-weight:700;margin-left:6px;vertical-align:middle">LIVE</span>'
          : state === 'past'
          ? '<span style="font-size:8px;color:var(--text3,#6b7280);letter-spacing:.07em;opacity:.6;margin-left:6px;vertical-align:middle">CLOSED</span>'
          : '<span style="font-size:8px;background:rgba(246,148,28,.10);color:var(--orange,#f6941c);border-radius:2px;padding:1px 4px;letter-spacing:.07em;opacity:.8;margin-left:6px;vertical-align:middle">UPCOMING</span>';

        // Upcoming: replace AI outlook with open time — AI note suppressed
        // (generated at 06:00 UTC; showing it before open reads as accomplished fact)
        const displayNote = state === 'upcoming'
          ? '<span style="color:var(--text3,#6b7280);font-style:italic">Opens ' + utcHourToLocalStr(sess.utcStart) + ' \u2014 context generated daily at 06:00 UTC</span>'
          : aiNote;

        return (
          '<div style="margin-bottom:7px">' +
          '<div style="margin-bottom:2px">' +
          '<span style="color:' + labelColor + ';font-weight:' + (state === 'active' ? '700' : '500') + ';letter-spacing:.04em;font-size:10px">' +
          sName.toUpperCase() + labelDot + '</span>' + stateChip +
          '</div>' +
          '<div style="color:' + textColor + ';padding-left:2px;font-size:11px;' + (state === 'upcoming' ? 'opacity:.6' : '') + '">' +
          displayNote +
          '</div>' +
          '</div>'
        );
      }).join('') +
      '<div style="margin-top:6px;font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono);letter-spacing:.03em;border-top:1px solid rgba(255,255,255,.05);padding-top:6px">' +
      'AI Analytics \xb7 ' + (_sessionCtxCache && _sessionCtxCache.generated_at
        ? (() => {
            const d = new Date(_sessionCtxCache.generated_at);
            const hh = String(d.getUTCHours()).padStart(2,'0');
            const mm = String(d.getUTCMinutes()).padStart(2,'0');
            return 'Updated ' + hh + ':' + mm + ' UTC';
          })()
        : '~2h refresh') +
      ' &nbsp;|&nbsp; ' + tzAbbr + ' ' + localStr + '</div>';
    } else {
      // Fallback: basic intraday stats (no Groq data yet)
      notes.innerHTML =
        `Active session: <span class="up">${activeSess}</span> &nbsp;|&nbsp; ` +
        `${tzAbbr} ${localStr}<br>` +
        `Session attribution weighted by typical volume distribution.<br>` +
        `Intraday strength: <span class="${pctClass(0)}" id="hm-sess-intra">—</span>`;

      // Update intraday note
      let sum = 0, cnt = 0;
      myPairs.forEach(p => {
        const d = rtCache[p.id];
        if (!d || d.pct == null) return;
        const impact = d.pct * p.sign * (p.base === ccy ? 1 : -1);
        sum += impact; cnt++;
      });
      const intra = cnt > 0 ? sum / cnt : null;
      const el = document.getElementById('hm-sess-intra');
      if (el && intra != null) {
        el.textContent = fmt2(intra);
        el.className   = pctClass(intra);
      }
    }
  }

  // Per-currency catalyst paragraph + named sources (v8.32.0; v8.56.1: relocated to a
  // persistent block above the tabs — see hm-macro — so it's visible regardless of
  // which tab is active, matching Bloomberg's separation of narrative (NI) from
  // quantitative ranking functions (WCRS) rather than nesting one inside the other).
  // Distinct from the pair-level driver notes in the Rel. Strength tab: this is a
  // substantive, sourced writeup of WHY the currency is moving (named officials,
  // decisions, dates), in the style of Bloomberg FXFB / Reuters wires — not a
  // repeated COT/carry boilerplate.
  function populateMacroDrivers(ccy) {
    const titleEl = document.getElementById('hm-catalyst-title');
    if (titleEl) titleEl.textContent = ccy + ' MACRO DRIVERS';

    const catalystEl = document.getElementById('hm-catalyst');
    if (!catalystEl) return;
    const ccyCatalyst = (_catalystsCache && _catalystsCache.currencies)
      ? _catalystsCache.currencies[ccy]
      : null;
    if (ccyCatalyst && ccyCatalyst.catalyst) {
      const sources = Array.isArray(ccyCatalyst.sources) ? ccyCatalyst.sources : [];
      const sourcesHtml = sources.length
        ? `<div style="margin-top:6px;font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono);line-height:1.6;">
             Sources: ${sources.slice(0,4).map(s =>
               `<a href="${s.url}" target="_blank" rel="noopener noreferrer" style="color:var(--text3,#6b7280);text-decoration:underline;">${(s.title||s.url).slice(0,40)}</a>`
             ).join(' · ')}
           </div>`
        : '';
      catalystEl.innerHTML = `
        <div style="font-size:11px;color:var(--text2,#787b86);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);line-height:1.6;">
          ${ccyCatalyst.catalyst}
        </div>
        ${sourcesHtml}
        <div style="margin-top:4px;font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono);letter-spacing:.03em;">AI Analytics · updated 1×/day</div>
      `;
    } else {
      catalystEl.innerHTML = '<div style="font-size:11px;color:var(--text3,#6b7280);font-family:var(--font-mono)">No macro driver data available yet</div>';
    }
  }

  function populateCorrelations(ccy, strengths, rtCache) {
    document.getElementById('hm-drivers-title').textContent =
      ccy + ' STRENGTH DRIVERS · TOP 3 PAIRS BY CONTRIBUTION';

    const ccys = ['EUR','GBP','JPY','AUD','CHF','CAD','NZD','USD','NOK','SEK'];

    // Build 8×8 strength differential matrix
    // cell[i][j] = strengths[i] - strengths[j]  (positive = row ccy stronger)
    const pctMap = {};
    strengths.forEach(s => { pctMap[s.ccy] = s.pct; });

    // ── Helper: classify diff into CSS class (mirrors rcm-matrix logic, terminal palette) ──
    function corrCellClass(diff) {
      if (diff == null) return 'corr-cell-flat';
      if (diff >=  0.40) return 'corr-cell-pos-hi';
      if (diff >=  0.06) return 'corr-cell-pos';
      if (diff <= -0.40) return 'corr-cell-neg-hi';
      if (diff <= -0.06) return 'corr-cell-neg';
      return 'corr-cell-flat';
    }
    function corrFmt(v) {
      if (v == null) return '—';
      if (Math.abs(v) < 0.005) return '0';
      return (v > 0 ? '+' : '') + v.toFixed(2);
    }

    // ── Build <table> identical in structure to rcm-matrix ───────────────
    const matrix = document.getElementById('hm-corr-matrix');
    const wrap   = document.createElement('div');
    wrap.className = 'corr-wrap';

    // Header row
    const headerCells = `<th class="row-head" scope="col" title="Row − Column = strength differential. Positive = row currency outperforms column currency today.">Δ Strength (row − col)</th>` +
      ccys.map(c => `<th scope="col"${c === ccy ? ' class="focal"' : ''} style="cursor:pointer" title="Click to pivot this panel to ${c}" onclick="hmPivotCcy('${c}')">${c}</th>`).join('') +
      `<th scope="col" class="focal" title="Equal-weighted composite — avg % vs all ${ccys.length - 1} major currency peers">Comp.</th>`;

    // Data rows
    const bodyRows = ccys.map(rowCcy => {
      const isFocalRow = rowCcy === ccy;
      const cells = ccys.map(colCcy => {
        if (rowCcy === colCcy) {
          // Diagonal: absolute composite strength
          const abs = pctMap[rowCcy] ?? 0;
          return `<td class="diag" data-diag="${rowCcy}" title="${rowCcy} composite: ${corrFmt(abs)}">${corrFmt(abs)}</td>`;
        }
        const diff = (pctMap[rowCcy] ?? 0) - (pctMap[colCcy] ?? 0);
        const cls  = corrCellClass(diff);
        const focalCls = (isFocalRow || colCcy === ccy) ? ' corr-cell-focal' : '';
        return `<td class="${cls}${focalCls}" data-r="${rowCcy}" data-c="${colCcy}" title="${rowCcy} vs ${colCcy}: ${corrFmt(diff)}">${corrFmt(diff)}</td>`;
      }).join('');

      // Comp. column (row composite)
      const rowComp = pctMap[rowCcy] ?? 0;
      const compCls = corrCellClass(rowComp);
      const compFocalCls = isFocalRow ? ' corr-cell-focal' : '';
      const compCell = `<td class="${compCls} comp-col${compFocalCls}" data-comp-row="${rowCcy}" style="font-weight:700" title="${rowCcy} composite vs major currency peers: ${corrFmt(rowComp)}">${corrFmt(rowComp)}</td>`;

      return `<tr><td class="row-head${isFocalRow ? ' focal' : ''}" style="cursor:pointer" title="Click to pivot this panel to ${rowCcy}" onclick="hmPivotCcy('${rowCcy}')">${rowCcy}</td>${cells}${compCell}</tr>`;
    }).join('');

    // Footer row (column composites)
    const footCells = ccys.map(colCcy => {
      const cv  = pctMap[colCcy] ?? 0;
      const cls = corrCellClass(cv);
      const focalCls = colCcy === ccy ? ' corr-cell-focal' : '';
      return `<td class="${cls}${focalCls}" data-comp-col="${colCcy}" style="font-weight:700" title="${colCcy} composite vs major currency peers: ${corrFmt(cv)}">${corrFmt(cv)}</td>`;
    }).join('');
    const footRow = `<tr class="comp-row"><td class="row-head focal" style="font-size:9px">Comp.</td>${footCells}<td class="diag" style="font-size:9px">—</td></tr>`;

    const legend = `<div class="corr-legend">
      <span><span style="display:inline-block;width:10px;height:10px;background:rgba(38,166,154,.25);border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Strong outperformance (Δ ≥ +0.40%)</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:rgba(38,166,154,.10);border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Mild outperformance (Δ ≥ +0.06%)</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:rgba(239,83,80,.10);border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Mild underperformance</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:rgba(239,83,80,.25);border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Strong underperformance</span>
      <span style="color:var(--text3,#4e5c70)">Diagonal = intraday composite · Values = equal-weighted Δ%, not Pearson correlations</span>
    </div>`;

    wrap.innerHTML = `<table class="corr-matrix" aria-label="Intraday strength differential matrix G10 currencies">
      <thead><tr>${headerCells}</tr></thead>
      <tbody>${bodyRows}${footRow}</tbody>
    </table>`;

    matrix.innerHTML = '';
    matrix.appendChild(wrap);
    matrix.insertAdjacentHTML('beforeend', legend);

    // Top 3 drivers
    const myPairs = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
    const driven  = [];
    myPairs.forEach(p => {
      const d = rtCache[p.id];
      if (!d || d.pct == null) return;
      const impact = d.pct * p.sign * (p.base === ccy ? 1 : -1);
      const opp    = p.base === ccy ? p.quote : p.base;
      const label  = p.base === ccy ? (p.base+'/'+p.quote) : (p.quote+'/'+p.base);
      const canon  = p.base + '/' + p.quote;   // canonical key matching currency-drivers.json
      driven.push({ label, opp, impact, canon });
    });
    driven.sort((a,b) => Math.abs(b.impact) - Math.abs(a.impact));
    const top3 = driven.slice(0,3);

    const driversEl = document.getElementById('hm-drivers');
    if (top3.length === 0) {
      driversEl.innerHTML = '<div style="font-size:11px;color:var(--text3,#6b7280);font-family:var(--font-mono)">No RT data available</div>';
      return;
    }

    // Per-pair AI notes from ai-analysis/currency-drivers.json (v8.34.0: free
    // institutional prose, web-search grounded — replaces the old per-field
    // COT/carry/CB-hold template). Sources are captured per CURRENCY (one grounded
    // call covers all 7 of that currency's pairs), not per individual pair.
    const ccyNotes = (_driversCache && _driversCache.drivers && _driversCache.drivers[ccy])
      ? _driversCache.drivers[ccy]
      : null;
    const ccySources = (_driversCache && _driversCache.driver_sources && _driversCache.driver_sources[ccy])
      ? _driversCache.driver_sources[ccy]
      : [];

    const sourcesLine = (ccyNotes && ccySources.length)
      ? `<div style="margin-top:8px;font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono);line-height:1.6;">
           Sources: ${ccySources.slice(0,4).map(s =>
             `<a href="${s.url}" target="_blank" rel="noopener noreferrer" style="color:var(--text3,#6b7280);text-decoration:underline;">${(s.title||s.url).slice(0,36)}</a>`
           ).join(' · ')}
         </div>`
      : '';

    driversEl.innerHTML = top3.map((d,i) => {
      const cls    = pctClass(d.impact);
      const note   = ccyNotes ? (ccyNotes[d.label] || ccyNotes[d.canon] || null) : null;
      const noteEl = note
        ? `<div style="font-size:10.5px;color:var(--text2,#787b86);font-family:var(--font-mono);margin-top:4px;line-height:1.6;">${note}</div>`
        : '';
      return `<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:${note ? 14 : 6}px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);">
        <div style="font-size:11px;font-weight:600;color:var(--text);width:70px;padding-top:1px;flex-shrink:0;">${d.label}</div>
        <div style="flex:1;min-width:0;">
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:11px;font-weight:600" class="${cls}" data-driver-idx="${i}">${fmt2(d.impact)}</span>
            <span style="font-size:11px;color:var(--text2,#787b86)">vs ${d.opp}</span>
          </div>
          ${noteEl}
        </div>
      </div>`;
    }).join('') + sourcesLine + (ccyNotes
      ? `<div style="margin-top:4px;font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono);letter-spacing:.03em;">AI Analytics</div>`
      : '');
  }

  // ── CSI (Currency Strength Index) ────────────────────────────────────────
  // Bloomberg WCRS convention: normalized cumulative log-return from period start.
  // All 8 series start at 0bp on day 0 — divergence represents relative performance.

  // Colour palette — 8 distinct, accessible colours matching terminal design language
  const CSI_COLORS = {
    EUR: '#4f7fff',  // --blue
    GBP: '#26a69a',  // --up (teal)
    JPY: '#ef5350',  // --down (red)
    AUD: '#f6941c',  // --orange
    CAD: '#a78bfa',  // purple
    CHF: '#34d399',  // emerald
    NZD: '#fb923c',  // amber
    USD: '#94a3b8',  // slate (USD neutral)
    NOK: '#0097b2',  // Norges Bank blue
    SEK: '#fecc00',  // Riksbank gold
  };

  const CCY_ORDER = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD','USD','NOK','SEK'];

  // Pairs sign convention for deriving ccy strength from OHLC:
  // +1 = pair close goes up → base strengthens; -1 = inverse.
  // v8.28.4: every entry is +1 by definition — log(close/prevClose) of any
  // base/quote pair already represents the base currency's return, regardless
  // of which currency is base. There is no pair that needs inversion. Matches
  // the EA's CSI_Score(): `sum += is_base ? ret : -ret` with no per-pair
  // special-casing. Do not reintroduce sign:-1 for USD-base pairs (usdjpy,
  // usdchf, usdcad, usdnok, usdsek) — that was the root cause of the CSI/
  // composite divergence between the web terminal and the EA fixed in v8.28.4.
  const PAIR_SIGN = {};
  PAIR_DEFS.forEach(p => { PAIR_SIGN[p.id] = p.sign; });

  // Load all 28 OHLC files in parallel, compute per-currency daily log-returns
  // and accumulate into the CSI series.
  // ── CSI TF → data source (2026-08-07) ───────────────────────────────────
  // H1/H4 reuse the exact same intraday OHLC files the main price chart
  // already fetches for those timeframes (ohlc-data/h1|h4/{pair}.json,
  // unix-second bar times) — see dashboard.js _isIntradayTf. D1 is the
  // original daily source (ohlc-data/{pair}.json, 'YYYY-MM-DD' bar times).
  // W1 has no separate source file: it reuses the D1 daily bars and the
  // resulting cumulative series is downsampled to one point per ISO week
  // in _resampleCSIWeekly() below — mathematically equivalent to computing
  // returns from weekly closes directly, since the cumulative log-return
  // sum telescopes (see that function's comment for the proof).
  function _csiBasePathForTf(tf) {
    if (tf === 'H1') return './ohlc-data/h1/';
    if (tf === 'H4') return './ohlc-data/h4/';
    return './ohlc-data/'; // D1 and W1 both source daily bars
  }

  async function _loadCSIData(tf) {
    tf = tf || 'D1';
    const pairIds = PAIR_DEFS.map(p => p.id);
    const basePath = _csiBasePathForTf(tf);
    // Cache-buster (?_=Date.now()) — matches the pattern already used by the
    // other fetches in this file (currency-drivers.json, currency-catalysts.json,
    // session-context.json, below). Without it, this was the only fetch in the
    // file relying on default browser/CDN HTTP caching, which could serve a
    // stale ohlc-data/{pair}.json — e.g. missing a same-day rally — while the
    // 1W Strength tile and heatmap (sourced from the cache-busted
    // intraday-data/quotes.json) already reflected it. Same staleness class as
    // documented in GUIDELINES.md re: GitHub Pages/CDN caching.
    const fetches = pairIds.map(id =>
      fetch(basePath + id + '.json?_=' + Date.now())
        .then(r => r.ok ? r.json() : [])
        .catch(() => [])
    );
    const allOHLC = await Promise.all(fetches);

    // Build a date-keyed map of log-returns for each pair
    // pairRet[id][date] = log(close/prevClose) * sign * (base=ccy ? +1 : -1)
    // "date" here is whatever the source bar's time field is — a
    // 'YYYY-MM-DD' string for D1/W1, a unix-second number for H1/H4.
    const pairRet = {};
    const allDates = new Set();

    pairIds.forEach((id, i) => {
      const bars = allOHLC[i];
      const p    = PAIR_DEFS[i];
      pairRet[id] = {};
      for (let j = 1; j < bars.length; j++) {
        const date = bars[j].time;
        const ret  = Math.log(bars[j].close / bars[j - 1].close);
        pairRet[id][date] = ret * p.sign;  // positive = base ccy gained vs quote
        allDates.add(date);
      }
    });

    // Sort dates — explicit comparator rather than the default .sort()
    // (which stringifies). It happens to give the right order for same-
    // length unix-second numbers too, but that's coincidence, not a
    // guarantee, so this is correct for both the D1/W1 string dates and
    // the H1/H4 numeric ones on purpose rather than by luck.
    const dates = [...allDates].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));

    // For each ccy and each date: average log-return across its participating
    // pairs (sign-corrected so positive always = this ccy strengthened).
    // Also track COVERAGE (how many of that ccy's pairs actually reported a
    // bar for this exact date) alongside the sum — see normalization fix
    // below (2026-08-07).
    const ccyDailyRet = {};
    const ccyDailyCov = {};
    CCY_ORDER.forEach(ccy => { ccyDailyRet[ccy] = {}; ccyDailyCov[ccy] = {}; });

    dates.forEach(date => {
      PAIR_DEFS.forEach(p => {
        const ret = pairRet[p.id][date];
        if (ret == null || isNaN(ret)) return;
        // base ccy gets +ret, quote ccy gets -ret
        if (ccyDailyRet[p.base]) {
          ccyDailyRet[p.base][date] = (ccyDailyRet[p.base][date] || 0) + ret;
          ccyDailyCov[p.base][date] = (ccyDailyCov[p.base][date] || 0) + 1;
        }
        if (ccyDailyRet[p.quote]) {
          ccyDailyRet[p.quote][date] = (ccyDailyRet[p.quote][date] || 0) - ret;
          ccyDailyCov[p.quote][date] = (ccyDailyCov[p.quote][date] || 0) + 1;
        }
      });
    });

    // Normalize by ACTUAL per-date coverage, not the fixed full pair count
    // (2026-08-07 fix). Previously divided by `pairsForCcy` (e.g. 9 for
    // USD) unconditionally. That's correct when all 9 pairs reported a bar
    // for a given date, but on any date where one pair's bar was legitimately
    // missing — e.g. fetch_ohlc.py's flat-bar guard (`o==h==l==c`) correctly
    // dropping a degenerate O=H=L=C bar yfinance occasionally returns for a
    // symbol's most recent in-progress hour — dividing the remaining 8
    // pairs' summed return by the full count of 9 systematically understated
    // that ccy's move for that one bar (biased toward zero, not just noisier).
    // Dividing by the pair count that ACTUALLY contributed each date is the
    // unbiased estimator: it's identical to the old behavior on every date
    // with full coverage (the overwhelming majority of history — cov ===
    // pairsForCcy there) and only changes the rare partial-coverage bar to a
    // correct per-pair average instead of a diluted one. `cov` is guaranteed
    // >=1 whenever `sum` is non-null (see the accumulation loop above), so no
    // divide-by-zero risk.
    const series = {};
    CCY_ORDER.forEach(ccy => {
      let cum = 0;
      series[ccy] = dates.map(date => {
        const sum = ccyDailyRet[ccy][date];
        const cov = ccyDailyCov[ccy][date];
        if (sum != null && cov) cum += sum / cov;
        // Convert to % (×100) for display
        return { time: date, value: parseFloat((cum * 100).toFixed(4)) };
      });
    });

    if (tf === 'W1') return _resampleCSIWeekly({ dates, series });
    return { dates, series };
  }

  // ── _resampleCSIWeekly — downsample the daily CSI series to one point per
  // ISO week (2026-08-07) ──────────────────────────────────────────────────
  // Keeps only the LAST daily point in each Mon–Sun week, labeled by that
  // week's Monday. This is mathematically identical to computing the series
  // directly from weekly closes: the CSI series is a running SUM of daily
  // log-returns, and a sum telescopes — cum(week N's last day) already
  // equals the sum of every daily return up to and including that week, the
  // same number you'd get log-ing (week N close / series-start close)
  // directly. So no separate weekly ohlc-data source is needed; this is a
  // pure downsample, not a re-derivation.
  function _resampleCSIWeekly(daily) {
    const { dates, series } = daily;
    if (!dates.length) return daily;

    function isoMonday(dateStr) {
      const d = new Date(dateStr + 'T00:00:00Z');
      const dow = d.getUTCDay() || 7; // Sun(0) -> 7, so Mon=1..Sun=7
      if (dow !== 1) d.setUTCDate(d.getUTCDate() - (dow - 1));
      return d.toISOString().slice(0, 10);
    }

    // Last index in `dates` for each ISO week, in chronological order
    // (dates is already sorted ascending, so Map insertion/iteration order
    // stays chronological as later same-week dates simply overwrite it).
    const lastIdxForWeek = new Map();
    dates.forEach((date, i) => lastIdxForWeek.set(isoMonday(date), i));

    const weekKeys = [...lastIdxForWeek.keys()];
    const wSeries  = {};
    CCY_ORDER.forEach(ccy => {
      const allPts = series[ccy];
      if (!allPts) { wSeries[ccy] = []; return; }
      wSeries[ccy] = weekKeys
        .map(wk => {
          const pt = allPts[lastIdxForWeek.get(wk)];
          return pt ? { time: wk, value: pt.value } : null;
        })
        .filter(Boolean);
    });

    return { dates: weekKeys, series: wSeries };
  }

  // ── Live in-progress-session point (mirrors dashboard.js _lwBuildTodayBar) ──
  // _loadCSIData() only ever returns CLOSED daily sessions: ohlc-data/*.json is
  // written by fetch_ohlc.py, which by design strips the in-progress today-bar
  // before writing (see that script's own header comment) so a candle with
  // truncated H/L wicks never persists. The main price chart compensates for
  // this via _lwBuildTodayBar()/_lwUpdateTodayBar() in dashboard.js, sourced
  // live from STOOQ_RT_CACHE — the CSI chart had no equivalent, so its most
  // recent day lagged behind the 1W Strength tile and heatmap (both read the
  // same STOOQ_RT_CACHE) until the OHLC workflow's next session-close run
  // (~21:00-22:30 UTC).
  //
  // This derives one extra point per currency for the session currently in
  // progress, using the exact same per-pair sign convention and log-return
  // math as _loadCSIData() above (Math.log(close/prevClose) * sign, averaged
  // across each currency's participating pairs) so the live point is
  // numerically consistent with the historical series, not just visually
  // appended to it.

  // FX session-open date convention (21:00 UTC boundary) — same rule
  // _lwBuildTodayBar()'s isFxBar branch uses in dashboard.js: a bar forming
  // at/after 21:00 UTC belongs to the session fetch_ohlc.py will date
  // tomorrow.
  function _csiLiveDateStr() {
    const now = new Date();
    if (now.getUTCHours() >= 21) {
      const tomorrow = new Date(now);
      tomorrow.setUTCDate(tomorrow.getUTCDate() + 1);
      return tomorrow.toISOString().slice(0, 10);
    }
    return now.toISOString().slice(0, 10);
  }

  // Returns { dates, series } with a live point appended to each currency's
  // series, or the plain historical _csiData when there's nothing live to
  // add (market closed, no rtCache yet, or the OHLC workflow has already
  // closed out today's session and the historical series already has it).
  function _computeCSILiveView() {
    if (!_csiData) return null;
    // Scope boundary (2026-08-07): the live in-progress-session point below
    // is built specifically from STOOQ_RT_CACHE's daily close/prev_close
    // fields and a 21:00-UTC daily-session-boundary rule — it has no
    // equivalent for H1/H4 intraday bars or W1's resampled weekly bars, so
    // it's skipped for any TF other than D1. Those TFs still show fully
    // up-to-date data as of the last completed bar; they just don't get the
    // extra "session still in progress" point D1 gets. A live intraday
    // point is a materially different feature (would need its own
    // resolution-aware boundary logic) — flagged as a possible follow-up.
    if (_csiTf !== 'D1') return _csiData;
    if (isMarketWeekend() || !_rtCache) return _csiData;

    const { dates, series } = _csiData;
    const liveDate     = _csiLiveDateStr();
    const lastHistDate = dates.length ? dates[dates.length - 1] : null;
    if (liveDate === lastHistDate) return _csiData;  // already closed out and in the JSON — nothing to append

    // Per-currency live return: average the signed log-return across each
    // currency's participating pairs, using STOOQ_RT_CACHE's close/prev_close
    // — the same fields _lwBuildTodayBar() reads for the main chart's live bar.
    const liveRet = {};
    CCY_ORDER.forEach(ccy => {
      const myPairs = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
      let sum = 0, cnt = 0;
      myPairs.forEach(p => {
        const d = _rtCache[p.id];
        if (!d || !d.close || !d.prev_close || d.close <= 0 || d.prev_close <= 0) return;
        const ret = Math.log(d.close / d.prev_close) * p.sign;
        sum += (p.base === ccy ? ret : -ret);
        cnt++;
      });
      liveRet[ccy] = cnt > 0 ? (sum / cnt) : null;
    });

    const liveSeries = {};
    let anyLive = false;
    CCY_ORDER.forEach(ccy => {
      const hist = series[ccy] || [];
      if (liveRet[ccy] == null) { liveSeries[ccy] = hist; return; }
      const lastVal = hist.length ? hist[hist.length - 1].value : 0;
      const liveVal = parseFloat((lastVal + liveRet[ccy] * 100).toFixed(4));
      liveSeries[ccy] = hist.concat([{ time: liveDate, value: liveVal }]);
      anyLive = true;
    });

    if (!anyLive) return _csiData;  // rtCache present but no usable pair data yet (e.g. right at session open)
    return { dates: dates.concat([liveDate]), series: liveSeries };
  }

  // Render or update the LWC chart with the current period
  function _renderCSIChart(ccy) {
    const LWC = window.LightweightCharts;
    if (!LWC || !_csiData) return;
    const csiView = _csiDataLive || _csiData;

    const wrap      = document.getElementById('hm-csi-wrap');
    const chartEl   = document.getElementById('hm-csi-chart');
    const tooltipEl = document.getElementById('hm-csi-tooltip');
    if (!wrap || !chartEl) return;

    // Determine date slice — real calendar-day cutoff anchored to the
    // series' own last date (see _csiCutoffDate; 2026-08-07 fix, replaces
    // the old bar-count offset that made the visible start point drift for
    // H1/H4/W1 depending on incidental weekend placement).
    const allDates  = csiView.dates;
    const lastDate  = allDates.length ? allDates[allDates.length - 1] : null;
    const cutoffDate = _csiPeriodDays > 0 ? _csiCutoffDate(lastDate, _csiPeriodDays) : (allDates.length ? allDates[0] : null);

    // Destroy old chart if it exists
    if (_csiResizeObs) {
      try { _csiResizeObs.disconnect(); } catch(e) {}
      _csiResizeObs = null;
    }
    if (_csiChart) {
      try { _csiChart.remove(); } catch(e) {}
      _csiChart = null;
      chartEl.innerHTML = '';
    }
    _csiSeriesMap = {};

    const _csiBg    = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()    || '#131722';
    const _csiText2 = getComputedStyle(document.documentElement).getPropertyValue('--text2').trim() || '#9096a0';
    const _csiBlue  = getComputedStyle(document.documentElement).getPropertyValue('--blue').trim()  || '#4f7fff';

    _csiChart = LWC.createChart(chartEl, {
      layout: {
        background: { color: _csiBg },
        textColor: _csiText2,
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,.04)' },
        horzLines: { color: 'rgba(255,255,255,.04)' },
      },
      crosshair: {
        mode: LWC.CrosshairMode?.Normal ?? 1,
        vertLine: { color: 'rgba(255,255,255,.25)', style: 2, labelVisible: true },
        horzLine: { color: 'rgba(255,255,255,.15)', style: 2, labelVisible: true },
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,.08)',
        scaleMargins: { top: 0.08, bottom: 0.08 },
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,.08)',
        timeVisible: (_csiTf === 'H1' || _csiTf === 'H4'), // intraday TFs need hour granularity on the axis, or repeated same-day bars all show an identical date label
        fixLeftEdge: true,
        fixRightEdge: true,
      },
      width: wrap.offsetWidth,
      height: 280,
    });

    // Keep chart width in sync with its container — without this, a browser
    // resize (or the panel becoming visible after a layout shift) leaves the
    // chart frozen at whatever width `wrap.offsetWidth` happened to be at
    // creation time, which can visually clip or misalign the plotted lines.
    // Mirrors the ResizeObserver pattern already used for the main chart in
    // dashboard.js (_lwResizeObs).
    if (typeof ResizeObserver !== 'undefined') {
      _csiResizeObs = new ResizeObserver(entries => {
        const cr = entries[0] && entries[0].contentRect;
        if (!cr || !_csiChart) return;
        const w = Math.floor(cr.width);
        if (w > 0) { try { _csiChart.applyOptions({ width: w }); } catch(e) {} }
      });
      _csiResizeObs.observe(wrap);
    }

    CCY_ORDER.forEach(c => {
      const isFocus = c === ccy;
      const allPts = csiView.series[c];
      const sliceIdx = allPts.findIndex(pt => pt.time >= cutoffDate);
      const baseVal  = sliceIdx >= 0 ? allPts[sliceIdx].value : 0;
      const raw = (sliceIdx >= 0 ? allPts.slice(sliceIdx) : allPts)
        .map(pt => ({ time: pt.time, value: parseFloat((pt.value - baseVal).toFixed(4)) }));
      const ls = _csiChart.addSeries(LWC.LineSeries, {
        color: CSI_COLORS[c],
        lineWidth: isFocus ? 2.5 : 1,
        lineStyle: 0,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: isFocus,
        crosshairMarkerRadius: 4,
        crosshairMarkerBorderColor: _csiBg,
        crosshairMarkerBackgroundColor: CSI_COLORS[c],
      });
      ls.setData(raw);
      if (!isFocus) ls.applyOptions({ lineWidth: 1, color: CSI_COLORS[c] + 'aa' });
      _csiSeriesMap[c] = ls;
    });

    // FIX (2026-08-07): without this, Lightweight Charts falls back to its
    // default fixed bar-spacing (~6px) instead of stretching the loaded
    // range to fill the container — so a range with few bars (e.g. "1D" on
    // H1, ~24 bars) left visible empty space instead of spanning the full
    // chart width, while the width itself looked inconsistent switching
    // between ranges with very different bar counts. fitContent() sizes bar
    // spacing so the currently-loaded data always fills the available width,
    // for every range.
    _csiChart.timeScale().fitContent();

    // Zero baseline
    const firstSeries = _csiSeriesMap[CCY_ORDER[0]];
    if (firstSeries) {
      firstSeries.createPriceLine({
        price: 0,
        color: 'rgba(255,255,255,.20)',
        lineWidth: 1,
        lineStyle: 1,
        axisLabelVisible: false,
        title: '',
      });
    }

    // Bloomberg-style multi-series crosshair tooltip
  // Bloomberg-style multi-series crosshair tooltip
  // param.time's shape depends on what Time type fed the series: a plain
  // 'YYYY-MM-DD' string for D1/W1 (LWC may normalize this to a
  // {year,month,day} BusinessDay object depending on version — handled
  // below), or a raw unix-second number for H1/H4 (2026-08-07 — previously
  // this concatenated param.time directly into the tooltip, which would
  // have shown a raw epoch number like "1786068000" once H1/H4 existed).
  function _csiFormatTooltipTime(t) {
    if (typeof t === 'number') {
      const d = new Date(t * 1000);
      return d.toISOString().slice(0, 10) + ' ' + d.toISOString().slice(11, 16) + ' UTC';
    }
    if (t && typeof t === 'object' && t.year) {
      return t.year + '-' + String(t.month).padStart(2, '0') + '-' + String(t.day).padStart(2, '0');
    }
    return String(t);
  }

  _csiChart.subscribeCrosshairMove(param => {
      if (!param || !param.time || !tooltipEl) {
        if (tooltipEl) tooltipEl.style.display = 'none';
        return;
      }
      const rows = CCY_ORDER.map(c => {
        const v = param.seriesData.get(_csiSeriesMap[c]);
        return { ccy: c, val: v ? v.value : null };
      }).filter(r => r.val != null).sort((a, b) => b.val - a.val);

      if (!rows.length) { tooltipEl.style.display = 'none'; return; }

      tooltipEl.innerHTML =
        '<div class="hm-csi-tt-date">' + _csiFormatTooltipTime(param.time) + '</div>' +
        rows.map(r => {
          const cls = r.val > 0 ? 'up' : r.val < 0 ? 'down' : 'flat';
          const dot = '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:' + CSI_COLORS[r.ccy] + ';margin-right:5px;"></span>';
          return '<div class="hm-csi-tt-row">' +
            '<span class="hm-csi-tt-ccy">' + dot + r.ccy + '</span>' +
            '<span class="hm-csi-tt-val ' + cls + '">' +
              (r.val >= 0 ? '+' : '') + r.val.toFixed(2) + '%' +
            '</span></div>';
        }).join('');
      tooltipEl.style.display = 'block';

      // Position: keep within wrap bounds
      const wrapRect = wrap.getBoundingClientRect();
      const x = param.point ? param.point.x : 0;
      const y = param.point ? param.point.y : 0;
      const ttW = 140, ttH = 20 + rows.length * 18;
      const left = (x + ttW + 20 > wrap.offsetWidth) ? (x - ttW - 10) : (x + 16);
      const top  = Math.max(0, Math.min(y - ttH / 2, wrap.offsetHeight - ttH));
      tooltipEl.style.left = left + 'px';
      tooltipEl.style.top  = top  + 'px';
    });

    // Update legend with final values
    _updateCSILegend(ccy, cutoffDate);
  }

  function _updateCSILegend(ccy, cutoffDate) {
    const legendEl = document.getElementById('hm-csi-legend');
    if (!legendEl || !_csiData) return;
    const csiView = _csiDataLive || _csiData;

    // Get final value for each ccy in the current period — rebased to 0 at period start
    const vals = CCY_ORDER.map(c => {
      const allPts   = csiView.series[c];
      const sliceIdx = allPts.findIndex(pt => pt.time >= cutoffDate);
      if (sliceIdx < 0) return { ccy: c, val: null, change: null };
      const baseVal  = allPts[sliceIdx].value;
      const filtered = allPts.slice(sliceIdx).map(pt => pt.value - baseVal);
      const last  = filtered.length ? filtered[filtered.length - 1] : null;
      const first = 0; // always 0 after rebase
      return { ccy: c, val: last != null ? parseFloat(last.toFixed(4)) : null, change: last };
    }).sort((a, b) => (b.val ?? -99) - (a.val ?? -99));

    legendEl.innerHTML = vals.map(r => {
      const isFocus = r.ccy === ccy;
      const cls = r.val > 0 ? 'up' : r.val < 0 ? 'down' : 'flat';
      const valStr = r.val != null ? (r.val >= 0 ? '+' : '') + r.val.toFixed(2) + '%' : '—';
      return '<div class="hm-csi-leg" onclick="hmPivotCcy(\'' + r.ccy + '\')" style="cursor:pointer" title="Click to view ' + r.ccy + '">' +
        '<div class="hm-csi-leg-dot" style="background:' + CSI_COLORS[r.ccy] + ';' +
          (isFocus ? 'height:3px;' : '') + '"></div>' +
        '<span class="hm-csi-leg-lbl" style="' + (isFocus ? 'color:var(--text,#d1d4dc);font-weight:600;' : '') + '">' +
          r.ccy + '</span>' +
        '<span class="hm-csi-leg-val ' + cls + '">' + valStr + '</span>' +
      '</div>';
    }).join('');
  }

  function _renderCSIStats(ccy) {
    const statsEl = document.getElementById('hm-csi-stats');
    const titleEl = document.getElementById('hm-csi-stats-title');
    if (!statsEl || !_csiData) return;
    const csiView = _csiDataLive || _csiData;

    const allDates  = csiView.dates;
    const lastDate  = allDates.length ? allDates[allDates.length - 1] : null;
    const cutoffDate = _csiPeriodDays > 0 ? _csiCutoffDate(lastDate, _csiPeriodDays) : (allDates.length ? allDates[0] : null);

    const rows = CCY_ORDER.map(c => {
      const allPts   = csiView.series[c];
      const sliceIdx = allPts.findIndex(pt => pt.time >= cutoffDate);
      if (sliceIdx < 0) return { ccy: c, val: null, min: null, max: null, range: null };
      const baseVal = allPts[sliceIdx].value;
      const vals = allPts.slice(sliceIdx).map(pt => parseFloat((pt.value - baseVal).toFixed(4)));
      return {
        ccy: c,
        val: vals[vals.length - 1],
        min: Math.min(...vals),
        max: Math.max(...vals),
        range: Math.max(...vals) - Math.min(...vals),
      };
    }).sort((a, b) => (b.val ?? -99) - (a.val ?? -99));

    const periodLabel = _csiRange;
    if (titleEl) titleEl.textContent = 'CSI SNAPSHOT · ' + periodLabel + ' · ACCUMULATED RETURN';

    statsEl.innerHTML = '<table class="hm-tbl" aria-label="CSI period statistics">' +
      '<thead><tr>' +
      '<th scope="col">Currency</th>' +
      '<th scope="col">Accum. Return</th>' +
      '<th scope="col">Drawdown (low)</th>' +
      '<th scope="col">Peak (high)</th>' +
      '<th scope="col">Peak-to-Trough</th>' +
      '</tr></thead><tbody>' +
      rows.map(r => {
        const isFocus = r.ccy === ccy;
        const cls = r.val > 0 ? 'up' : r.val < 0 ? 'down' : 'flat';
        const fmt = v => v != null ? (v >= 0 ? '+' : '') + v.toFixed(2) + '%' : '—';
        return '<tr style="' + (isFocus ? 'background:rgba(79,127,255,.07);' : '') + '">' +
          '<td><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:' +
            CSI_COLORS[r.ccy] + ';margin-right:6px;vertical-align:middle;"></span>' +
            '<span class="sym" style="' + (isFocus ? 'color:var(--blue,#4f7fff);' : '') + '">' + r.ccy + '</span></td>' +
          '<td class="' + cls + '">' + fmt(r.val) + '</td>' +
          '<td class="' + (r.min != null && r.min < 0 ? 'down' : r.min != null && r.min > 0 ? 'up' : 'flat') + '">' + fmt(r.min) + '</td>' +
          '<td class="' + (r.max != null && r.max > 0 ? 'up' : r.max != null && r.max < 0 ? 'down' : 'flat') + '">' + fmt(r.max) + '</td>' +
          '<td style="color:var(--text2,#787b86)">' + (r.range != null ? r.range.toFixed(2) + '%' : '—') + '</td>' +
        '</tr>';
      }).join('') +
      '</tbody></table>' +
      '<div style="margin-top:8px;font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono);letter-spacing:.03em;">' +
      (_CSI_TF_TITLE[_csiTf] || 'DAILY') + ' OHLC history · 32-pair G10 composite CSI · Accum. Return = total from period start · Drawdown/Peak = lowest/highest CSI value within period</div>';
  }

  async function populateCSI(ccy) {
    const loadingEl = document.getElementById('hm-csi-loading');

    // Only fetch OHLC data once per page session
    if (!_csiData) {
      if (loadingEl) loadingEl.style.display = 'flex';

      // Ensure LWC is loaded (reuse dashboard.js loadLWC pattern)
      if (!window.LightweightCharts) {
        await new Promise((res, rej) => {
          const s = document.createElement('script');
          s.src = 'https://cdn.jsdelivr.net/npm/lightweight-charts@5.0.7/dist/lightweight-charts.standalone.production.js';
          s.onload = res; s.onerror = rej;
          document.head.appendChild(s);
        });
      }

      try {
        _csiData = await _loadCSIData(_csiTf);
      } catch(e) {
        if (loadingEl) loadingEl.textContent = 'Failed to load OHLC data';
        return;
      }
    }

    if (loadingEl) loadingEl.style.display = 'none';

    _csiDataLive = _computeCSILiveView();

    _renderCSIChart(ccy);
    _renderCSIStats(ccy);
  }

  // ── Currency switcher (Session tab request, v2.6.0) — mirrors cot-modal-chart.js's
  // cotCycleCcy/cotToggleCcyDropdown/cotSwitchCcy trio, driven off _strengths (already
  // holds all 10 G10 currencies passed in from dashboard.js's window._hmStrengths, no
  // separate data store needed the way COT's per-currency lazy-fetch cache requires).
  const _G10_ORDER = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'NOK', 'SEK'];
  let _hmAvailCcys = [];

  function _hmSetTitle(ccy) {
    const meta = CCY_META[ccy] || { flag: 'un', full: ccy };
    const titleRow = document.getElementById('hm-title-row');
    const titleEl  = document.getElementById('hm-title');
    titleEl.textContent = `\u2014 ${meta.full} Strength`;
    let flagSpan = titleRow.querySelector('.fi');
    if (!flagSpan) {
      flagSpan = document.createElement('span');
      flagSpan.style.cssText = 'border-radius:2px;font-size:15px;vertical-align:middle;flex-shrink:0;';
      titleRow.insertBefore(flagSpan, titleRow.firstChild); // flag leads the row: [flag] — text ‹ [chip] ›
    }
    flagSpan.className = `fi fi-${meta.flag}`;
  }

  function _hmUpdateCcySwitcher(ccy) {
    const avail = _G10_ORDER.filter(c => (_strengths || []).some(s => s.ccy === c));
    _hmAvailCcys = avail.length ? avail : [ccy];
    const idx = _hmAvailCcys.indexOf(ccy);
    const chip = document.getElementById('hm-ccy-chip');
    const dd   = document.getElementById('hm-ccy-dd');
    const prev = document.getElementById('hm-ccy-prev');
    const next = document.getElementById('hm-ccy-next');
    if (chip) chip.textContent = ccy;
    if (dd) {
      dd.innerHTML = _hmAvailCcys.map(c =>
        `<button class="hm-ccy-dd-item ${c === ccy ? 'on' : ''}" role="option" aria-selected="${c === ccy}" onclick="hmPivotCcy('${c}')">${c}</button>`
      ).join('');
    }
    if (prev) prev.disabled = idx <= 0;
    if (next) next.disabled = idx === -1 || idx >= _hmAvailCcys.length - 1;
  }

  window.hmCycleCcy = function(dir) {
    if (!_hmAvailCcys.length || !_ccy) return;
    const idx = _hmAvailCcys.indexOf(_ccy);
    const nextIdx = idx + dir;
    if (nextIdx < 0 || nextIdx >= _hmAvailCcys.length) return;
    hmPivotCcy(_hmAvailCcys[nextIdx]);
  };

  window.hmToggleCcyDropdown = function(e) {
    e.stopPropagation();
    const dd = document.getElementById('hm-ccy-dd');
    const chip = document.getElementById('hm-ccy-chip');
    if (!dd) return;
    const open = dd.classList.toggle('open');
    if (chip) chip.setAttribute('aria-expanded', open ? 'true' : 'false');
  };

  function _hmCloseCcyDropdown() {
    const dd = document.getElementById('hm-ccy-dd');
    if (dd && dd.classList.contains('open')) {
      dd.classList.remove('open');
      document.getElementById('hm-ccy-chip')?.setAttribute('aria-expanded', 'false');
    }
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  window.openHeatmapModal = function(ccy, strengths, rtCache) {
    _ccy       = ccy;
    _strengths = strengths;
    _rtCache   = rtCache;

    buildModal();

    _hmSetTitle(ccy);
    _hmUpdateCcySwitcher(ccy);

    // Reset to first tab
    document.querySelectorAll('.hm-tab').forEach(t => {
      t.classList.toggle('on', t.dataset.tab === 'breakdown');
      t.setAttribute('aria-selected', t.dataset.tab === 'breakdown' ? 'true' : 'false');
    });
    document.querySelectorAll('.hm-panel').forEach(p => {
      p.classList.toggle('on', p.id === 'hm-p-breakdown');
    });

    populateMetrics(ccy, strengths, rtCache);
    populateBreakdown(ccy, strengths, rtCache);
    populateMacroDrivers(ccy); // persistent block — render immediately with whatever's cached so far
    fetchDrivers();        // lazy-load AI driver notes in the background
    fetchCatalysts();      // lazy-load AI per-currency catalyst notes; re-renders macro drivers on arrival
    fetchSessionContext(); // lazy-load AI session context notes in the background

    // Update source labels to reflect active data source (Finnhub live vs yfinance)
    _updateModalSourceLabels();

    const bd = document.getElementById('hm-bd');
    bd.style.display = 'flex';
    document.getElementById('hm-close').focus();
  };


  // ── csiSetRange — the ONE user-facing CSI control (2026-08-07 redesign) ────
  // Replaces the earlier same-day csiSetTf()/csiPeriod()/_renderCSIPeriodBtns()
  // trio (two independent controls) with a single range switch. Looks up both
  // the lookback (`days`) and the resolution (`tf`) from one _CSI_RANGE_CONFIG
  // entry, so picking "1D" always means exactly one thing.
  // Only re-fetches OHLC data when the resolution actually changes (e.g. 1D→1W
  // both use H1 — switching between them just re-slices/re-renders already-
  // loaded data; 1W→1M changes H1→H4 and needs a real fetch).
  window.csiSetRange = function(btn, rangeKey) {
    if (_csiRange === rangeKey) return;
    const cfg = _CSI_RANGE_CONFIG.find(r => r.key === rangeKey);
    if (!cfg) return;
    _csiRange = rangeKey;
    const tfChanged = _csiTf !== cfg.tf;
    _csiTf = cfg.tf;
    _csiPeriodDays = cfg.days;

    document.querySelectorAll('#hm-csi-controls .hm-csi-btn').forEach(b => b.classList.toggle('on', b === btn));

    const titleEl = document.getElementById('hm-csi-title');
    if (titleEl) titleEl.textContent = 'CURRENCY STRENGTH INDEX · ACCUMULATED % RETURN · ' + (_CSI_TF_TITLE[cfg.tf] || 'DAILY') + ' OHLC';

    if (!_ccy) return;

    if (!tfChanged) {
      // Same underlying resolution already loaded (e.g. 1D <-> 1W, both H1;
      // or 3M <-> 6M <-> 1Y, all D1) — just re-slice at the new cutoff and
      // re-render, no need to hit the network again.
      _csiDataLive = _computeCSILiveView();
      _renderCSIChart(_ccy);
      _renderCSIStats(_ccy);
      return;
    }

    const loadingEl = document.getElementById('hm-csi-loading');
    const chartEl   = document.getElementById('hm-csi-chart');
    if (loadingEl) { loadingEl.style.display = 'flex'; loadingEl.textContent = 'Loading OHLC data…'; }
    if (chartEl) chartEl.style.display = 'none';

    _loadCSIData(cfg.tf).then(data => {
      // Stale-response guard: if the user clicked a different range again
      // before this fetch resolved, drop this result — _csiRange has
      // already moved on and applying it now would show the wrong data.
      if (_csiRange !== rangeKey) return;
      _csiData = data;
      _csiDataLive = _computeCSILiveView();
      if (loadingEl) loadingEl.style.display = 'none';
      if (chartEl) chartEl.style.display = '';
      _renderCSIChart(_ccy);
      _renderCSIStats(_ccy);
    }).catch(() => {
      if (_csiRange !== rangeKey) return;
      if (loadingEl) loadingEl.textContent = 'Failed to load OHLC data';
    });
  };

  window.closeHeatmapModal = function() {
    const bd = document.getElementById('hm-bd');
    if (bd) bd.style.display = 'none';
    document.removeEventListener('keydown', _onKey);
    // Destroy chart so it re-renders at correct size on next open
    if (_csiResizeObs) { try { _csiResizeObs.disconnect(); } catch(e) {} _csiResizeObs = null; }
    if (_csiChart) { try { _csiChart.remove(); } catch(e) {} _csiChart = null; }
  };

  window.hmTab = function(el, tabId) {
    document.querySelectorAll('.hm-tab').forEach(t => {
      t.classList.toggle('on', t.dataset.tab === tabId);
      t.setAttribute('aria-selected', t.dataset.tab === tabId ? 'true' : 'false');
    });
    document.querySelectorAll('.hm-panel').forEach(p => {
      p.classList.toggle('on', p.id === 'hm-p-' + tabId);
    });

    // Lazy-populate on first switch
    if (tabId === 'session' && _ccy) {
      populateSession(_ccy, _rtCache);
    } else if (tabId === 'correlations' && _ccy) {
      populateCorrelations(_ccy, _strengths, _rtCache);
    } else if (tabId === 'csi' && _ccy) {
      populateCSI(_ccy);
    }
  };

  // Pivot the whole modal to a different focal currency without closing it — clicking
  // a row/column header in the Rel. Strength matrix, or a currency chip in the CSI
  // legend, calls this (Bloomberg/Eikon-style in-panel instrument pivot, instead of
  // forcing the user to close and reopen from the heatmap for every currency they
  // want to inspect).
  window.hmPivotCcy = function(newCcy) {
    if (!newCcy || newCcy === _ccy || !_strengths || !_rtCache) return;
    _ccy = newCcy;

    _hmSetTitle(newCcy);
    _hmUpdateCcySwitcher(newCcy);
    _hmCloseCcyDropdown();

    populateMetrics(newCcy, _strengths, _rtCache);
    populateBreakdown(newCcy, _strengths, _rtCache, true);
    populateMacroDrivers(newCcy); // persistent block — must follow the pivot too, not just the active tab
    populateCorrelations(newCcy, _strengths, _rtCache);
    // Session and CSI each have their own chart/canvas or fetch cost, so only
    // refresh them when that tab is actually the one on screen — matching the
    // lazy-populate pattern hmTab() already uses on first switch.
    const sessionPanel = document.getElementById('hm-p-session');
    if (sessionPanel && sessionPanel.classList.contains('on')) populateSession(newCcy, _rtCache);
    const csiPanel = document.getElementById('hm-p-csi');
    if (csiPanel && csiPanel.classList.contains('on')) populateCSI(newCcy);
  };

  // ── Live source label — updates hm-sub and hm-footer-meta to reflect active source ──
  function _updateModalSourceLabels() {
    const hasFh = window.STOOQ_RT_CACHE
      ? Object.values(window.STOOQ_RT_CACHE).some(e => e?.fromFinnhub)
      : false;
    const srcLabel = hasFh
      ? 'Live \u00b7 G10 composite \u00b7 32 pairs'
      : 'G10 composite \u00b7 32 pairs \u00b7 Delayed ~5min';
    const footerLabel = hasFh
      ? 'Live \u00b7 G10 composite \u00b7 32 pairs'
      : 'Delayed ~5min \u00b7 G10 composite \u00b7 32 pairs';
    const subEl    = document.getElementById('hm-sub');
    const footerEl = document.getElementById('hm-footer-meta');
    if (subEl)    subEl.textContent    = srcLabel;
    if (footerEl) footerEl.textContent = footerLabel;
  }

  // ── _updateBreakdownRT — flash-free in-place update for the Breakdown tab ──────────
  // Called by _hmRefreshIfOpen on every Finnhub tick instead of full populateBreakdown().
  // Updates only textContent/className/style on already-rendered DOM nodes.
  // Falls back to full populateBreakdown() if the DOM structure is stale (e.g. ccy changed).
  function _updateBreakdownRT(ccy, strengths, rtCache) {
    const tbody = document.getElementById('hm-pair-tbody');
    if (!tbody || tbody.children.length === 0) {
      populateMetrics(ccy, strengths, rtCache);
      populateBreakdown(ccy, strengths, rtCache, true); // _skipAnim: modal already open
      return;
    }

    // Re-compute impacts (same logic as populateBreakdown but no DOM rebuild)
    const myPairs = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
    const impacts = [];
    myPairs.forEach(p => {
      const d = rtCache[p.id];
      const isCcyBase = p.base === ccy;
      const opp = isCcyBase ? p.quote : p.base;
      const rawPct = d?.pct ?? null;
      const impact = rawPct != null ? rawPct * p.sign * (isCcyBase ? 1 : -1) : null;
      const raw1w  = d?.pct1w ?? null;
      const imp1w  = raw1w != null ? raw1w * p.sign * (isCcyBase ? 1 : -1) : null;
      const close  = isCcyBase ? (d?.close ?? null) : (d?.close != null ? 1/d.close : null);
      const open   = isCcyBase ? (d?.open  ?? null) : (d?.open  != null ? 1/d.open  : null);
      const hi     = isCcyBase ? (d?.high  ?? null) : (d?.high  != null ? 1/d.high  : null);
      const lo     = isCcyBase ? (d?.low   ?? null) : (d?.low   != null ? 1/d.low   : null);
      const label  = isCcyBase ? (p.base+'/'+p.quote) : (p.quote+'/'+p.base);
      impacts.push({ label, opp, close, open, hi, lo, impact, rawPct, imp1w });
    });
    impacts.sort((a,b) => (b.impact??-99) - (a.impact??-99));
    const maxImp = Math.max(...impacts.map(i => Math.abs(i.impact ?? 0)), 0.001);

    // Check if sort order changed — if so, fall back to full render (with _skipAnim to avoid flash)
    const rows = Array.from(tbody.querySelectorAll('tr[data-pair]'));
    const currentOrder = rows.map(r => r.dataset.pair);
    const newOrder = impacts.map(r => r.label);
    if (currentOrder.join(',') !== newOrder.join(',')) {
      populateMetrics(ccy, strengths, rtCache);
      populateBreakdown(ccy, strengths, rtCache, true); // _skipAnim: modal already open
      return;
    }

    // In-place update only — no innerHTML touches
    impacts.forEach(r => {
      const row = tbody.querySelector(`tr[data-pair="${r.label}"]`);
      if (!row) return;
      const iCls   = pctClass(r.impact);
      const barW   = r.impact != null ? Math.round(Math.abs(r.impact)/maxImp*100) : 0;
      const barClr = r.impact != null && r.impact >= 0 ? 'var(--up,#26a69a)' : 'var(--down,#ef5350)';
      const rng    = (r.hi != null && r.lo != null) ? fmtPrice(r.lo) + ' – ' + fmtPrice(r.hi) : '—';

      const closeCell  = row.querySelector('[data-cell="close"]');
      const openCell   = row.querySelector('[data-cell="open"]');
      const impactCell = row.querySelector('[data-cell="impact"]');
      const imp1wCell  = row.querySelector('[data-cell="imp1w"]');
      const barFill    = row.querySelector('[data-cell="bar"]');
      const rngCell    = row.querySelector('[data-cell="rng"]');

      if (closeCell)  closeCell.textContent  = fmtPrice(r.close);
      if (openCell)   openCell.textContent   = fmtPrice(r.open);
      if (impactCell) { impactCell.textContent = fmt2(r.impact); impactCell.className = iCls; }
      if (imp1wCell)  { imp1wCell.textContent  = r.imp1w != null ? fmt2(r.imp1w) : '—'; imp1wCell.className = pctClass(r.imp1w); }
      if (barFill)    { barFill.style.width = barW + '%'; barFill.style.background = barClr; }
      if (rngCell)    rngCell.textContent   = rng;
    });

    // Update metrics header (already in-place — populateMetrics uses textContent throughout)
    populateMetrics(ccy, strengths, rtCache);

    // Update day% ranking in-place
    const container = document.getElementById('hm-ranking-rows');
    if (container) {
      const sorted    = [...strengths].sort((a,b) => b.pct - a.pct);
      const maxAbsPct = Math.max(...sorted.map(s => Math.abs(s.pct)), 0.001);
      sorted.forEach(s => {
        const rankRow = container.querySelector(`[data-rank-ccy="${s.ccy}"]`);
        if (!rankRow) return;
        const fillEl = rankRow.querySelector('.hm-rank-fill');
        const valEl  = rankRow.querySelector('[data-rank-val]');
        const fillW  = Math.round(Math.abs(s.pct) / maxAbsPct * 100);
        const cls    = 'hm-rank-fill ' + ((s.ccy === ccy) ? 'hl' : pctClass(s.pct));
        const newW   = fillW + '%';
        if (fillEl) {
          if (fillEl.style.width !== newW)    fillEl.style.width = newW;
          if (fillEl.className  !== cls)      fillEl.className   = cls;
        }
        if (valEl) {
          const newTxt = fmt2(s.pct);
          const newCls = 'hm-rank-val ' + pctClass(s.pct);
          if (valEl.textContent !== newTxt) valEl.textContent = newTxt;
          if (valEl.className   !== newCls) valEl.className   = newCls;
        }
      });
    }

    // Update 1W ranking in-place
    const cont1w = document.getElementById('hm-ranking-1w-rows');
    if (cont1w && cont1w.querySelector('[data-rank-ccy]')) {
      const ccys = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD','USD','NOK','SEK'];
      const w1map = {};
      ccys.forEach(c => { w1map[c] = { sum: 0, n: 0 }; });
      PAIR_DEFS.forEach(p => {
        const d = rtCache[p.id];
        if (!d || d.pct1w == null) return;
        const v = d.pct1w * p.sign;
        w1map[p.base].sum += v; w1map[p.base].n++;
        w1map[p.quote].sum -= v; w1map[p.quote].n++;
      });
      const w1strengths = ccys.map(c => ({ ccy: c, pct: w1map[c].n > 0 ? w1map[c].sum / w1map[c].n : null })).filter(s => s.pct != null);
      const maxAbs1w = Math.max(...w1strengths.map(s => Math.abs(s.pct)), 0.001);
      w1strengths.forEach(s => {
        const rankRow = cont1w.querySelector(`[data-rank-ccy="${s.ccy}"]`);
        if (!rankRow) return;
        const fillEl = rankRow.querySelector('.hm-rank-fill');
        const valEl  = rankRow.querySelector('[data-rank-val]');
        const fillW  = Math.round(Math.abs(s.pct) / maxAbs1w * 100);
        const cls    = 'hm-rank-fill ' + ((s.ccy === ccy) ? 'hl' : pctClass(s.pct));
        const newW   = fillW + '%';
        if (fillEl) {
          if (fillEl.style.width !== newW)    fillEl.style.width = newW;
          if (fillEl.className  !== cls)      fillEl.className   = cls;
        }
        if (valEl) {
          const newTxt = fmt2(s.pct);
          const newCls = 'hm-rank-val ' + pctClass(s.pct);
          if (valEl.textContent !== newTxt) valEl.textContent = newTxt;
          if (valEl.className   !== newCls) valEl.className   = newCls;
        }
      });
    }
  }

  // ── _updateCorrelationsRT — flash-free in-place update for the Correlations tab ──
  // Updates only cell values/classes in the already-rendered corr matrix.
  // Falls back to full populateCorrelations() if the table is missing (shouldn't happen).
  function _updateCorrelationsRT(ccy, strengths, rtCache) {
    const matrix = document.getElementById('hm-corr-matrix');
    if (!matrix || !matrix.querySelector('[data-r]')) {
      populateCorrelations(ccy, strengths, rtCache);
      return;
    }

    // Re-compute pctMap
    const ccys = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD','USD','NOK','SEK'];
    const pctMap = {};
    ccys.forEach(c => { pctMap[c] = null; });
    strengths.forEach(s => { pctMap[s.ccy] = s.pct; });

    function corrFmt(v) {
      if (v == null) return '—';
      if (Math.abs(v) < 0.005) return '0';
      return (v > 0 ? '+' : '') + v.toFixed(2);
    }
    function corrCellClass(diff) {
      if (diff == null) return 'corr-cell-flat';
      if (diff >=  0.40) return 'corr-cell-pos-hi';
      if (diff >=  0.06) return 'corr-cell-pos';
      if (diff <= -0.40) return 'corr-cell-neg-hi';
      if (diff <= -0.06) return 'corr-cell-neg';
      return 'corr-cell-flat';
    }

    // Update body cells
    matrix.querySelectorAll('td[data-r][data-c]').forEach(td => {
      const r = td.dataset.r, c = td.dataset.c;
      const diff = (pctMap[r] ?? 0) - (pctMap[c] ?? 0);
      const focalCls = (r === ccy || c === ccy) ? ' corr-cell-focal' : '';
      td.className = corrCellClass(diff) + focalCls;
      td.textContent = corrFmt(diff);
      td.title = `${r} vs ${c}: ${corrFmt(diff)}`;
    });

    // Update diagonal cells
    matrix.querySelectorAll('td[data-diag]').forEach(td => {
      const r = td.dataset.diag;
      const abs = pctMap[r] ?? 0;
      td.textContent = corrFmt(abs);
      td.title = `${r} composite: ${corrFmt(abs)}`;
    });

    // Update Comp. column cells (row composites)
    matrix.querySelectorAll('td[data-comp-row]').forEach(td => {
      const r = td.dataset.compRow;
      const v = pctMap[r] ?? 0;
      const focalCls = r === ccy ? ' corr-cell-focal' : '';
      td.className = corrCellClass(v) + ' comp-col' + focalCls;
      td.textContent = corrFmt(v);
      td.title = `${r} composite vs major currency peers: ${corrFmt(v)}`;
    });

    // Update footer cells (column composites)
    matrix.querySelectorAll('td[data-comp-col]').forEach(td => {
      const c = td.dataset.compCol;
      const v = pctMap[c] ?? 0;
      const focalCls = c === ccy ? ' corr-cell-focal' : '';
      td.className = corrCellClass(v) + focalCls;
      td.textContent = corrFmt(v);
      td.title = `${c} composite vs major currency peers: ${corrFmt(v)}`;
    });

    // Update top-3 drivers in-place (just the pct values, no layout change)
    const driversEl = document.getElementById('hm-drivers');
    if (driversEl) {
      const myPairs = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
      const driven  = [];
      myPairs.forEach(p => {
        const d = rtCache[p.id];
        if (!d || d.pct == null) return;
        const impact = d.pct * p.sign * (p.base === ccy ? 1 : -1);
        const opp    = p.base === ccy ? p.quote : p.base;
        const label  = p.base === ccy ? (p.base+'/'+p.quote) : (p.quote+'/'+p.base);
        driven.push({ label, opp, impact });
      });
      driven.sort((a,b) => Math.abs(b.impact) - Math.abs(a.impact));
      driven.slice(0,3).forEach((d,i) => {
        const pctEl = driversEl.querySelector(`[data-driver-idx="${i}"]`);
        if (pctEl) {
          pctEl.textContent = fmt2(d.impact);
          pctEl.className   = pctClass(d.impact);
        }
      });
    }
  }

  // ── _updateCSILiveBar — flash-free in-place update for the CSI tab ─────────────
  // Recomputes the live in-progress-session point and pushes it onto each
  // already-rendered LWC series via .update() (adds if newer, replaces if same
  // date — LWC's standard incremental-update behavior), instead of tearing down
  // and rebuilding the whole chart on every RT tick the way _renderCSIChart()
  // does for period changes / tab switches.
  function _updateCSILiveBar() {
    if (!_csiChart || !_csiData || !_ccy) return;

    _csiDataLive = _computeCSILiveView();
    const csiView = _csiDataLive || _csiData;

    // BUG FIX (2026-08-07): this cutoff used to be computed as
    // `allDates.length - _csiPeriodDays`, i.e. treating _csiPeriodDays as a
    // BAR-COUNT offset into the array — the exact bar-count model that was
    // deliberately replaced everywhere else (_renderCSIChart, _renderCSIStats)
    // by the calendar-day _csiCutoffDate() cutoff earlier in this same
    // 2026-08-07 session. This function was missed in that migration, so on
    // every RT tick it rebased the LIVE point against a baseline only
    // _csiPeriodDays BARS back (e.g. 7 bars = 7 hours on H1) while every
    // other point already on the chart was rebased against the true
    // _csiCutoffDate() baseline (e.g. 7 CALENDAR days = ~120 H1 bars back).
    // Those two baselines differ by however much the series moved over that
    // gap, so the live point snapped to a value on a totally different
    // footing from its neighbors every time a tick came in — visible as all
    // currencies' lines jumping/converging together at the most recent bar.
    // Fix: use the exact same _csiCutoffDate()-based cutoff and baseVal
    // lookup as _renderCSIChart, so the live-updated point stays on the same
    // footing as the rest of the series.
    const allDates   = csiView.dates;
    const lastDate    = allDates.length ? allDates[allDates.length - 1] : null;
    const cutoffDate  = _csiPeriodDays > 0 ? _csiCutoffDate(lastDate, _csiPeriodDays) : (allDates.length ? allDates[0] : null);

    CCY_ORDER.forEach(c => {
      const ls = _csiSeriesMap[c];
      const allPts = csiView.series[c];
      if (!ls || !allPts || !allPts.length) return;
      const sliceIdx = allPts.findIndex(pt => pt.time >= cutoffDate);
      if (sliceIdx < 0) return;
      const baseVal = allPts[sliceIdx].value;
      const lastPt  = allPts[allPts.length - 1];
      ls.update({ time: lastPt.time, value: parseFloat((lastPt.value - baseVal).toFixed(4)) });
    });

    _updateCSILegend(_ccy, cutoffDate);
    _renderCSIStats(_ccy);
  }

    // ── _hmRefreshIfOpen — called by dashboard.js populateHeatmap() on every RT update ──
  // Refreshes whichever tab is currently active without closing/reopening the modal.
  // Only runs when the modal is actually visible — no-op otherwise.
  // This is the mechanism that makes the modal update in real time from Finnhub ticks.
  window._hmRefreshIfOpen = function(newStrengths, newRtCache) {
    const bd = document.getElementById('hm-bd');
    if (!bd || bd.style.display === 'none' || !_ccy) return;

    // Update stored references so tab switches also get fresh data
    _strengths = newStrengths;
    _rtCache   = newRtCache;

    // Update source labels in header and footer
    _updateModalSourceLabels();

    // Refresh the active tab with flash-free in-place updates
    const activeTab = document.querySelector('.hm-tab.on');
    if (!activeTab) return;
    const tabId = activeTab.dataset.tab;

    if (tabId === 'breakdown') {
      // In-place update — no innerHTML rebuild, no flash
      _updateBreakdownRT(_ccy, _strengths, _rtCache);
    } else if (tabId === 'session') {
      // Session data is from session_high/low (changes slowly) — full render acceptable
      populateSession(_ccy, _rtCache);
    } else if (tabId === 'correlations') {
      // In-place update — only cell values/classes, no table rebuild
      _updateCorrelationsRT(_ccy, _strengths, _rtCache);
    } else if (tabId === 'csi') {
      // In-place update — pushes/refreshes only the live in-progress-session
      // point on each existing LWC series (no chart teardown/rebuild, so no
      // flash), same intent as dashboard.js's _lwUpdateTodayBar().
      _updateCSILiveBar();
    }
  };

})();
