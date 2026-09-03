/**
 * GlobalInvesting FX Terminal — Market Overview module, BETA "editorial
 * desk" redesign  v1.0.0
 * assets/gi-overview-beta.js — include AFTER dashboard.js and gi-auth.js
 * in index-beta.html, in place of assets/gi-overview.js. Preview-only fork:
 * never referenced by production index.html.
 *
 * Population target is the new #gi-overview markup (index-beta.html,
 * .gi-ovb-* classes) — headline/narrative prose, a 10-currency G10 strength
 * strip (was 8 in production, and used the old .gi-ov-* bias-card markup),
 * a live-quotes rail, and a currency-bias mini list, replacing production's
 * gi-ov-bias-row/gi-ov-csi-strip/gi-ov-narrative-box.
 *
 * Deliberately has NO independent data fetch or computation of its own —
 * per GUIDELINES.md's "two independent parsers of the same feed silently
 * drift" principle, every value shown here is read directly from state
 * dashboard.js (or, for the regime word, dashboard.js's own DOM write) has
 * already fetched/computed:
 *   - window._hmStrengths        — set by populateHeatmap() in dashboard.js;
 *                                   same G10 currency-strength composite the
 *                                   Currency Strength Heatmap panel uses.
 *   - #narrative-text.textContent — set by buildRichNarrative() in
 *                                   dashboard.js; the full AI Narrative
 *                                   panel text (that panel itself stays
 *                                   gated — this only mirrors an excerpt).
 *   - #risk-regime.textContent    — the live cross-asset stress regime
 *                                   (RISK-ON/MIXED/CAUTION/RISK-OFF),
 *                                   already the single source of truth this
 *                                   codebase's own alert engine reads
 *                                   (dashboard.js's evaluateRegimeAlerts()).
 *                                   Read here only to fold into the
 *                                   headline's opening clause (Option C —
 *                                   no separate color badge on the
 *                                   narrative; see index-beta.html's
 *                                   section comment), never re-derived from
 *                                   the underlying VIX/MOVE/credit inputs.
 *   - #q-.../#qc-... quote-bar spans — set by dashboard.js's live quote feed.
 * All are populated asynchronously after this script runs and keep
 * changing live afterward, so this module polls continuously (not once)
 * while the Overview is the visible view — same pattern as production's
 * gi-overview.js.
 */
(function () {
  'use strict';

  const POLL_MS = 500;
  const POLL_MAX_ATTEMPTS = 30; // 15s — matches dashboard.js's own data-load budget
  const SYNC_MS = 1500; // live re-render cadence once first data arrives

  // ── "Terminal actually visible" hook — same contract as production's
  // window.giOnTerminalShown, since dashboard.js/onboarding.js call it
  // unconditionally regardless of which Overview script defined it. ──
  window._giTerminalShown = false;
  window.giOnTerminalShown = function (cb) {
    if (typeof cb !== 'function') return;
    if (window._giTerminalShown) { cb(); return; }
    document.addEventListener('gi:terminal-shown', cb, { once: true });
  };

  function maybeAnnounceTerminalEntry() {
    if (window._giTerminalShown) return;
    const tv = document.getElementById('gi-terminal-view');
    if (!tv || tv.style.display === 'none') return;
    const modal = document.getElementById('gi-auth-modal');
    if (modal && modal.classList.contains('visible')) return;
    window._giTerminalShown = true;
    document.dispatchEvent(new CustomEvent('gi:terminal-shown'));
  }

  function watchTerminalEntry() {
    let attempts = 0;
    const t = setInterval(() => {
      attempts++;
      maybeAnnounceTerminalEntry();
      if (window._giTerminalShown || attempts > 1800) clearInterval(t); // ~21 min cap
    }, 700);
  }

  // ── View toggle — unchanged from production's gi-overview.js ──────────────
  function showTerminal() {
    document.getElementById('gi-overview')?.style.setProperty('display', 'none');
    const tv = document.getElementById('gi-terminal-view');
    if (tv) tv.style.display = '';
    document.getElementById('gi-footer-actions')?.style.setProperty('display', 'flex');
    stopOverviewSync();
    maybeAnnounceTerminalEntry();
  }

  function showOverview() {
    document.documentElement.removeAttribute('data-gi-preauth');
    document.getElementById('gi-overview')?.style.removeProperty('display');
    const tv = document.getElementById('gi-terminal-view');
    if (tv) tv.style.display = 'none';
    document.getElementById('gi-footer-actions')?.style.setProperty('display', 'none');
    if (!_syncTimer) pollAndRender();
  }

  window.giShowOverview = showOverview;

  function initToggle() {
    if (window.GI_AUTH && window.GI_AUTH.isActive) {
      showTerminal();
      return;
    }

    document.getElementById('gi-ovb-open-terminal')?.addEventListener('click', () => {
      window.GI_AUTH && window.GI_AUTH.showModal();
      showTerminal();
    });

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

    watchTerminalEntry();
  }

  // ── Currency strength strip — full 10-currency G10 board, BIS Triennial
  // Survey turnover order (matches CORR_MTX_CCYS/COT_BREAKDOWN_CCYS in
  // dashboard.js — this repo's one canonical fixed-list ordering). ──
  const G10_CCYS = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'SEK', 'NOK', 'NZD'];
  const CCY_FLAG = { USD: 'us', EUR: 'eu', JPY: 'jp', GBP: 'gb', AUD: 'au', CAD: 'ca', CHF: 'ch', SEK: 'se', NOK: 'no', NZD: 'nz' };
  // Bias-tag rail — mirrors production's tagFor() threshold exactly (single
  // source of truth), showing only the first 4 (space-constrained rail).
  const BIAS_CCYS = ['USD', 'EUR', 'GBP', 'JPY'];

  function tagFor(pct) {
    if (pct > 0.15) return { cls: 'strong', label: 'Strong' };
    if (pct < -0.15) return { cls: 'weak', label: 'Weak' };
    return { cls: 'neutral', label: 'Neutral' };
  }

  // Mirrors dashboard.js's own ±0.05/±0.15 heatmap-cell breakpoints (single
  // source of truth — GUIDELINES.md's "two independent parsers of one feed
  // drift" incident, v1.3.1 of production's gi-overview.js).
  function strengthBg(pct) {
    if (pct > 0.15) return 'rgba(38,166,154,.38)';
    if (pct > 0.05) return 'rgba(38,166,154,.16)';
    if (pct < -0.15) return 'rgba(239,83,80,.34)';
    if (pct < -0.05) return 'rgba(239,83,80,.16)';
    return 'rgba(141,147,163,.10)';
  }
  function strengthColor(pct) {
    if (pct > 0.05) return 'var(--up)';
    if (pct < -0.05) return 'var(--down)';
    return 'var(--text2)';
  }

  function renderStrengthRow(strengths) {
    const row = document.getElementById('gi-ovb-strength-row');
    if (!row) return;
    const byCcy = {};
    strengths.forEach(s => { byCcy[s.ccy] = s.pct; });
    row.innerHTML = G10_CCYS.map(ccy => {
      const pct = byCcy[ccy];
      if (pct === undefined) return '';
      const sign = pct >= 0 ? '+' : '';
      const flag = CCY_FLAG[ccy];
      return `<div class="gi-ovb-scell" style="background:${strengthBg(pct)}">
        <div class="cc">${flag ? `<span class="fi fi-${flag}" style="margin-right:3px;border-radius:2px;"></span>` : ''}${ccy}</div>
        <div class="vv" style="color:${strengthColor(pct)}">${sign}${pct.toFixed(2)}</div>
      </div>`;
    }).join('');
  }

  function renderBiasRail(strengths) {
    const rail = document.getElementById('gi-ovb-bias');
    if (!rail) return;
    const byCcy = {};
    strengths.forEach(s => { byCcy[s.ccy] = s.pct; });
    rail.innerHTML = BIAS_CCYS.map(ccy => {
      const pct = byCcy[ccy];
      if (pct === undefined) return '';
      const t = tagFor(pct);
      const sign = pct >= 0 ? '+' : '';
      return `<div class="gi-ovb-bias-mini">
        <span>${ccy}</span>
        <span class="gi-ovb-tag ${t.cls}">${t.label} ${sign}${pct.toFixed(2)}%</span>
      </div>`;
    }).join('');
  }

  // ── Live quotes rail — reuses the existing top quote-bar spans directly
  // (dashboard.js's live feed already writes these); this module never
  // fetches a price itself. ──
  const QUOTE_MAP = [
    ['eurusd', 'gi-ovb-q-eurusd', 'gi-ovb-qc-eurusd'],
    ['usdjpy', 'gi-ovb-q-usdjpy', 'gi-ovb-qc-usdjpy'],
    ['gbpusd', 'gi-ovb-q-gbpusd', 'gi-ovb-qc-gbpusd'],
    ['audusd', 'gi-ovb-q-audusd', 'gi-ovb-qc-audusd'],
    ['xauusd', 'gi-ovb-q-xauusd', 'gi-ovb-qc-xauusd'],
    ['us10y',  'gi-ovb-q-us10y',  'gi-ovb-qc-us10y'],
  ];

  function renderQuotes() {
    QUOTE_MAP.forEach(([srcId, dstPriceId, dstChgId]) => {
      const srcPrice = document.getElementById('q-' + srcId);
      const srcChg = document.getElementById('qc-' + srcId);
      const dstPrice = document.getElementById(dstPriceId);
      const dstChg = document.getElementById(dstChgId);
      if (srcPrice && dstPrice) dstPrice.textContent = srcPrice.textContent;
      if (srcChg && dstChg) {
        const txt = srcChg.textContent;
        dstChg.textContent = txt;
        dstChg.className = 'chg ' + (txt.trim().startsWith('+') ? 'up' : txt.trim().startsWith('-') || txt.trim().startsWith('\u2212') ? 'down' : 'flat');
      }
    });
  }

  // ── Narrative headline + body, regime folded into the opening clause
  // (Option C) instead of a separate color badge. ──
  const NARRATIVE_PLACEHOLDER = 'Loading market narrative\u2026';
  const BODY_EXCERPT_MAX = 320;

  // Reads the SAME element dashboard.js's own alert engine already treats
  // as the canonical live regime (see evaluateRegimeAlerts() reading
  // #risk-regime's textContent directly) — never re-derives the VIX/MOVE/
  // credit-spread scoring itself, per this file's own no-duplicate-parser
  // header note.
  function liveRegimePhrase() {
    const regime = document.getElementById('risk-regime')?.textContent?.trim();
    switch (regime) {
      case 'RISK-ON':  return 'reads risk-on';
      case 'RISK-OFF': return 'reads risk-off, in a broad flight to safety';
      case 'CAUTION':  return 'reads cautious';
      case 'MIXED':    return 'reads mixed';
      default: return null; // regime not yet computed — omit rather than guess
    }
  }

  function firstSentence(text) {
    const m = text.match(/^.*?[.!?](\s|$)/);
    return m ? m[0].trim() : text;
  }

  function excerpt(text) {
    if (text.length <= BODY_EXCERPT_MAX) return text;
    const cut = text.slice(0, BODY_EXCERPT_MAX);
    const lastSpace = cut.lastIndexOf(' ');
    return (lastSpace > 0 ? cut.slice(0, lastSpace) : cut) + '\u2026';
  }

  let _lastNarrText = null;

  function renderNarrative(fullText, generatedAt) {
    const headEl = document.getElementById('gi-ovb-headline');
    const bodyEl = document.getElementById('gi-ovb-narrative-body');
    const tsEl = document.getElementById('gi-ovb-narrative-ts');
    if (headEl) headEl.textContent = firstSentence(fullText);
    if (bodyEl) {
      const regimePhrase = liveRegimePhrase();
      const prefix = regimePhrase ? `Risk sentiment ${regimePhrase}. ` : '';
      bodyEl.textContent = prefix + excerpt(fullText);
    }
    if (tsEl) {
      const ts = generatedAt
        ? `Narrative \u00b7 updated ${new Date(generatedAt).toUTCString().slice(17, 22)} UTC`
        : 'Narrative';
      tsEl.textContent = ts;
    }
  }

  // ── Single polling loop driving strength/bias/quotes/narrative together
  // (all render off state dashboard.js already owns; no reason to run four
  // independent timers). Self-rescheduling setTimeout, not setInterval, so
  // cadence can relax from POLL_MS to SYNC_MS once data is flowing — same
  // "never freeze on a stale snapshot while the tab stays open" fix as
  // production's pollHeatmapStrengths()/pollNarrative(). ──
  let _syncTimer = null;

  function pollAndRender() {
    let attempts = 0;
    let gotStrength = false;
    let gotNarrative = false;
    function tick() {
      attempts++;
      const strengths = window._hmStrengths;
      const haveStrength = strengths && strengths.length;
      if (haveStrength && (gotStrength || window._hmStrengthsLive === true)) {
        renderStrengthRow(strengths);
        renderBiasRail(strengths);
        gotStrength = true;
      } else if (!gotStrength && attempts >= POLL_MAX_ATTEMPTS && haveStrength) {
        // Live composite never arrived within budget — render the fallback
        // estimate rather than leave the strip on "Loading…" forever.
        renderStrengthRow(strengths);
        renderBiasRail(strengths);
        gotStrength = true;
      }

      const narrSrc = document.getElementById('narrative-text');
      const narrText = narrSrc ? narrSrc.textContent.trim() : '';
      if (narrText && narrText !== NARRATIVE_PLACEHOLDER) {
        if (narrText !== _lastNarrText) {
          renderNarrative(narrText, window._narrativeGeneratedAt);
          _lastNarrText = narrText;
        }
        gotNarrative = true;
      }

      renderQuotes();

      if (!(gotStrength && gotNarrative) && attempts < POLL_MAX_ATTEMPTS * 2) {
        _syncTimer = setTimeout(tick, POLL_MS);
      } else {
        _syncTimer = setTimeout(tick, SYNC_MS);
      }
    }
    _syncTimer = setTimeout(tick, POLL_MS);
  }

  function stopOverviewSync() {
    if (_syncTimer) { clearTimeout(_syncTimer); _syncTimer = null; }
  }

  // ── Init ───────────────────────────────────────────────────────────────
  function init() {
    initToggle();
    if (window._giTerminalShown) return;
    pollAndRender();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
