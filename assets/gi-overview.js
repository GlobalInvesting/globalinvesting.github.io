/**
 * GlobalInvesting FX Terminal — Market Overview module  v1.0.0
 * assets/gi-overview.js — include AFTER dashboard.js and gi-auth.js in index.html
 *
 * v1.0.0 (2026-08-12): New file. Populates the #gi-overview section (shown
 * by default to first-time visitors, see index.html v8.129.0 / CHANGELOG.md)
 * and wires the Overview <-> full-terminal (#gi-terminal-view) toggle.
 *
 * Deliberately has NO independent data fetch or computation of its own —
 * per GUIDELINES.md's "two independent parsers of the same feed silently
 * drift" principle, every value shown here is read directly from state
 * dashboard.js has already fetched/computed:
 *   - window._hmStrengths       — set by populateHeatmap() in dashboard.js;
 *                                  same G10 currency-strength composite the
 *                                  Currency Strength Heatmap panel uses.
 *   - #narrative-text.textContent — set by buildRichNarrative() in
 *                                  dashboard.js; the full AI Narrative panel
 *                                  text (that panel itself stays gated —
 *                                  this only mirrors a short excerpt).
 * Both are populated asynchronously after this script runs, so this module
 * polls briefly rather than assuming they're ready on first check.
 */
(function () {
  'use strict';

  const POLL_MS = 500;
  const POLL_MAX_ATTEMPTS = 30; // 15s — matches dashboard.js's own data-load budget

  // ── View toggle ────────────────────────────────────────────────────────────
  function showTerminal() {
    document.getElementById('gi-overview')?.style.setProperty('display', 'none');
    const tv = document.getElementById('gi-terminal-view');
    if (tv) tv.style.display = '';
  }

  function showOverview() {
    document.getElementById('gi-overview')?.style.removeProperty('display');
    const tv = document.getElementById('gi-terminal-view');
    if (tv) tv.style.display = 'none';
  }

  function initToggle() {
    // Returning user with a still-valid license token skips the snapshot —
    // gi-auth.js's init() (which runs earlier in script order, see its own
    // header) has already set window.GI_AUTH.isActive synchronously by now.
    if (window.GI_AUTH && window.GI_AUTH.isActive) {
      showTerminal();
      return;
    }

    document.getElementById('gi-ov-open-terminal')
      ?.addEventListener('click', showTerminal);

    // If the user activates from a locked-preview card on the Overview
    // itself (calls window.GI_AUTH.showModal() directly — see index.html),
    // switch to the terminal automatically once activation succeeds,
    // instead of leaving them looking at the snapshot post-activation.
    // Short-lived poll, not a permanent listener — stops itself once either
    // condition is met.
    let attempts = 0;
    const watchActivation = setInterval(() => {
      attempts++;
      if (window.GI_AUTH && window.GI_AUTH.isActive) {
        clearInterval(watchActivation);
        showTerminal();
      } else if (attempts > 600) { // 5 min — user may just be reading, not activating
        clearInterval(watchActivation);
      }
    }, 500);
  }

  // ── Bias cards + CSI strip (both read window._hmStrengths) ──────────────────
  const BIAS_CCYS = ['USD', 'EUR', 'GBP', 'JPY'];
  const CSI_CCYS  = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF']; // matches mock's 7-currency strip

  function tagFor(pct) {
    if (pct > 0.15) return { cls: 'up', label: 'Strong' };
    if (pct < -0.15) return { cls: 'down', label: 'Weak' };
    return { cls: '', label: 'Neutral' };
  }

  function renderBiasRow(strengths) {
    const row = document.getElementById('gi-ov-bias-row');
    if (!row) return;
    const byCcy = {};
    strengths.forEach(s => { byCcy[s.ccy] = s.pct; });
    row.innerHTML = BIAS_CCYS.map(ccy => {
      const pct = byCcy[ccy];
      if (pct === undefined) return '';
      const t = tagFor(pct);
      const sign = pct >= 0 ? '+' : '';
      return `<div class="gi-ov-bias-card" data-ccy="${ccy}">
        <div class="gi-ov-bias-top"><span class="gi-ov-bias-ccy">${ccy}</span><span class="gi-ov-bias-tag ${t.cls}">${t.label}</span></div>
        <div class="gi-ov-bias-pct">${sign}${pct.toFixed(2)}%</div>
      </div>`;
    }).join('');
  }

  function renderCsiStrip(strengths) {
    const strip = document.getElementById('gi-ov-csi-strip');
    if (!strip) return;
    const byCcy = {};
    strengths.forEach(s => { byCcy[s.ccy] = s.pct; });
    const maxAbs = Math.max(0.05, ...CSI_CCYS.map(c => Math.abs(byCcy[c] || 0)));
    strip.innerHTML = CSI_CCYS.map(ccy => {
      const pct = byCcy[ccy] || 0;
      const t = tagFor(pct);
      const color = t.cls === 'up' ? 'var(--up)' : t.cls === 'down' ? 'var(--down)' : 'var(--text2)';
      const heightPct = Math.max(8, Math.round((Math.abs(pct) / maxAbs) * 100));
      return `<div class="gi-ov-csi-col">
        <div class="gi-ov-csi-bar" style="height:${heightPct}%;background:${color};"></div>
        <div class="gi-ov-csi-lbl">${ccy}</div>
      </div>`;
    }).join('');
  }

  function pollHeatmapStrengths() {
    let attempts = 0;
    const t = setInterval(() => {
      attempts++;
      const strengths = window._hmStrengths;
      if (strengths && strengths.length) {
        renderBiasRow(strengths);
        renderCsiStrip(strengths);
        clearInterval(t);
      } else if (attempts >= POLL_MAX_ATTEMPTS) {
        clearInterval(t); // leave the "Loading…" skeleton — no fabricated fallback numbers
      }
    }, POLL_MS);
  }

  // ── Narrative excerpt (mirrors #narrative-text, the gated panel's own element) ──
  const NARRATIVE_PLACEHOLDER = 'Loading market narrative\u2026';
  const NARRATIVE_EXCERPT_MAX = 220;

  function excerpt(text) {
    if (text.length <= NARRATIVE_EXCERPT_MAX) return text;
    const cut = text.slice(0, NARRATIVE_EXCERPT_MAX);
    const lastSpace = cut.lastIndexOf(' ');
    return (lastSpace > 0 ? cut.slice(0, lastSpace) : cut) + '\u2026';
  }

  function pollNarrative() {
    let attempts = 0;
    const t = setInterval(() => {
      attempts++;
      const src = document.getElementById('narrative-text');
      const text = src ? src.textContent.trim() : '';
      if (text && text !== NARRATIVE_PLACEHOLDER) {
        const dest = document.getElementById('gi-ov-narrative-text');
        if (dest) dest.textContent = excerpt(text);
        clearInterval(t);
      } else if (attempts >= POLL_MAX_ATTEMPTS) {
        clearInterval(t); // leave the "Loading…" placeholder — never fabricate narrative text
      }
    }, POLL_MS);
  }

  // ── Init ───────────────────────────────────────────────────────────────────
  function init() {
    initToggle();
    pollHeatmapStrengths();
    pollNarrative();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
