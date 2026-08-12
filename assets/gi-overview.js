/**
 * GlobalInvesting FX Terminal — Market Overview module  v1.1.0
 * assets/gi-overview.js — include AFTER dashboard.js and gi-auth.js in index.html
 *
 * v1.1.0 (2026-08-12): Four fixes reported by Santiago against the live
 * v1.0.0 build (screenshots + reload comparisons):
 *   1. Bias cards / CSI strip looked "wrong" and changed to unrelated
 *      numbers on every reload. Root cause: pollHeatmapStrengths() stopped
 *      polling the instant it found data once — but window._hmStrengths
 *      keeps updating live via WS ticks (dashboard.js's
 *      populateHeatmapThrottled(), ~800ms cadence) for as long as the page
 *      is open. The Overview froze on whatever value existed a fraction of
 *      a second after load and never moved again, so it silently drifted
 *      from the live terminal the longer the tab stayed open — and any two
 *      reloads naturally froze on two different real snapshots. Not
 *      fabricated data, just a stale single read. Fixed: renderSync() now
 *      re-renders on a recurring timer for as long as #gi-overview is the
 *      visible view, same pattern as the narrative excerpt.
 *   2. Quick Tour (onboarding.js) and the "Configure Alert" tooltip
 *      (dashboard.js) both fired their own DOMContentLoaded+delay timers
 *      with no awareness that the Overview snapshot — not the terminal —
 *      was on screen. Fixed here by exposing window.giOnTerminalShown(cb),
 *      a one-shot hook that fires exactly when the full terminal actually
 *      becomes visible AND the activation modal isn't covering it (see
 *      maybeAnnounceTerminalEntry() below). onboarding.js and dashboard.js
 *      now gate their init on this instead of raw DOMContentLoaded.
 *   3. "Open full terminal" called showTerminal() directly, revealing
 *      #gi-terminal-view with only PREMIUM_SECTIONS individually locked
 *      (the v8.128.0 "some panels open" model) — Santiago wants this
 *      specific entry point to behave like the pre-v8.128.0 flow instead:
 *      terminal revealed *behind* the full-page activation modal (blurred),
 *      not freely browsable first. Fixed: the click handler now also calls
 *      window.GI_AUTH.showModal() when the visitor isn't yet active.
 *   4. (Fixed in gi-auth.js v1.6.0, not here) — the activation modal had no
 *      way to close except reloading the page.
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
 * Both are populated asynchronously after this script runs and keep
 * changing live afterward, so this module polls continuously (not once)
 * while the Overview is the visible view.
 */
(function () {
  'use strict';

  const POLL_MS = 500;
  const POLL_MAX_ATTEMPTS = 30; // 15s — matches dashboard.js's own data-load budget
  const SYNC_MS = 1500; // live re-render cadence once first data arrives

  // ── "Terminal actually visible" hook ─────────────────────────────────────
  // Exposed synchronously (not inside DOMContentLoaded) so onboarding.js and
  // dashboard.js — which load after this file, see index.html script order —
  // can reference it unconditionally. Fires exactly once, the first moment
  // #gi-terminal-view is on screen AND the activation modal isn't covering
  // it (a modal-covered terminal is not something a first-time tour should
  // start highlighting elements behind).
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

  // ── View toggle ────────────────────────────────────────────────────────────
  function showTerminal() {
    document.getElementById('gi-overview')?.style.setProperty('display', 'none');
    const tv = document.getElementById('gi-terminal-view');
    if (tv) tv.style.display = '';
    stopOverviewSync();
    maybeAnnounceTerminalEntry();
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

    document.getElementById('gi-ov-open-terminal')?.addEventListener('click', () => {
      // Santiago: this specific entry point should behave like the
      // pre-v8.128.0 flow — terminal revealed *behind* the full-page
      // activation modal (blurred backdrop), not freely browsable first.
      // A visitor who is already active never reaches this branch (handled
      // above), so showModal() here always means "not yet activated".
      showTerminal();
      if (!(window.GI_AUTH && window.GI_AUTH.isActive)) {
        window.GI_AUTH && window.GI_AUTH.showModal();
      }
    });

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

    // The activation modal can also be dismissed WITHOUT activating (its own
    // new close button / Escape / backdrop click, gi-auth.js v1.6.0) while
    // the terminal is already revealed underneath (per-panel gates from
    // applyGates() still in place) — that's still "entering the terminal"
    // for tour/onboarding purposes, so keep watching independently of
    // whether activation ever completes.
    watchTerminalEntry();
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

  let _hmSyncTimer = null;

  // Re-renders for as long as the Overview stays the visible view
  // (stopOverviewSync() below cancels this once showTerminal() runs) —
  // window._hmStrengths keeps changing live via WS ticks, so a one-shot
  // "poll until found, then stop" left the Overview frozen on a single
  // stale snapshot while the real terminal kept moving underneath it.
  // Self-rescheduling setTimeout (not setInterval) so the cadence can
  // slow down from POLL_MS to the lighter SYNC_MS once data is flowing.
  function pollHeatmapStrengths() {
    let attempts = 0;
    let gotFirst = false;
    function tick() {
      attempts++;
      const strengths = window._hmStrengths;
      if (strengths && strengths.length) {
        renderBiasRow(strengths);
        renderCsiStrip(strengths);
        gotFirst = true;
      } else if (!gotFirst && attempts >= POLL_MAX_ATTEMPTS) {
        _hmSyncTimer = null; // leave the "Loading…" skeleton — no fabricated fallback numbers
        return;
      }
      _hmSyncTimer = setTimeout(tick, gotFirst ? SYNC_MS : POLL_MS);
    }
    _hmSyncTimer = setTimeout(tick, POLL_MS);
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

  let _narrSyncTimer = null;
  let _narrLastText = null;

  // Same "keep syncing, don't freeze" fix as pollHeatmapStrengths() — the
  // AI narrative refreshes every 15 min (dashboard.js's setInterval on
  // buildRichNarrative), so a one-shot read could show a desk note that's
  // since gone stale for anyone who leaves the Overview tab open a while.
  function pollNarrative() {
    let attempts = 0;
    let gotFirst = false;
    function tick() {
      attempts++;
      const src = document.getElementById('narrative-text');
      const text = src ? src.textContent.trim() : '';
      if (text && text !== NARRATIVE_PLACEHOLDER) {
        if (text !== _narrLastText) {
          const dest = document.getElementById('gi-ov-narrative-text');
          if (dest) dest.textContent = excerpt(text);
          _narrLastText = text;
        }
        gotFirst = true;
      } else if (!gotFirst && attempts >= POLL_MAX_ATTEMPTS) {
        _narrSyncTimer = null; // leave the "Loading…" placeholder — never fabricate narrative text
        return;
      }
      _narrSyncTimer = setTimeout(tick, gotFirst ? SYNC_MS : POLL_MS);
    }
    _narrSyncTimer = setTimeout(tick, POLL_MS);
  }

  function stopOverviewSync() {
    if (_hmSyncTimer)   { clearTimeout(_hmSyncTimer);   _hmSyncTimer = null; }
    if (_narrSyncTimer) { clearTimeout(_narrSyncTimer); _narrSyncTimer = null; }
  }

  // ── Init ───────────────────────────────────────────────────────────────────
  function init() {
    initToggle();
    // Returning active users skip straight to showTerminal() inside
    // initToggle(), which already calls stopOverviewSync() — starting the
    // sync timers after that would just spin them forever against a
    // section that's now display:none.
    if (window._giTerminalShown) return;
    pollHeatmapStrengths();
    pollNarrative();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
