/**
 * GlobalInvesting FX Terminal — Market Overview module  v1.4.1
 * assets/gi-overview.js — include AFTER dashboard.js and gi-auth.js in index.html
 *
 * v1.4.1 (2026-08-15): Reverted renderBiasRow()'s v1.4.0 template change
 *   (three-child flex-row card, built for the v8.155.0 Desk Briefing hero)
 *   back to the original two-row card markup — the client reverted the
 *   Desk Briefing layout in index.html back to the original stacked
 *   layout, keeping only the "Read full note" link. See CHANGELOG.md
 *   v8.156.0.
 *
 * v1.4.0 (2026-08-15): Desk Briefing layout (index.html v8.155.0) —
 *   renderBiasRow()'s template updated to match the new hero-column bias
 *   card layout (ccy/tag/pct as three direct flex children instead of
 *   ccy+tag nested together above a separate pct row below). This function
 *   fully overwrites #gi-ov-bias-row's innerHTML on every populate, so the
 *   static skeleton markup in index.html and this template must be kept in
 *   sync by hand — no shared source. See CHANGELOG.md v8.155.0.
 *
 * v1.3.2 (2026-08-14): Companion fix to gi-auth.js v1.7.5's pre-auth flash
 *   guard (see that file's own changelog entry for the full root-cause
 *   writeup — index.html now sets data-gi-preauth="1" on <html> via a
 *   synchronous inline <head> script, before first paint, so a returning
 *   active user's browser never paints the Overview snapshot at all). This
 *   file's own logic (initToggle(), showTerminal()) is otherwise unchanged
 *   and remains the authoritative source of truth once gi-auth.js's real
 *   isActive check runs. Only change here: showOverview() now clears
 *   data-gi-preauth as its first action, before touching any inline
 *   display style — that attribute drives a !important CSS override that
 *   force-shows #gi-terminal-view, and if left set here it would silently
 *   fight this function's own inline-style write two lines below,
 *   reintroducing the "closing the modal without activating leaves the
 *   full terminal open" bug fixed in v1.2.0/v1.7.0.
 *
 * v1.3.1 (2026-08-13): Reported by the client — every currency in the
 *   Overview's "Currency Strength — G10 Snapshot" strip rendering flat gray
 *   since the prior day, while the Currency Strength Heatmap panel (same
 *   underlying window._hmStrengths) showed a normal green/red spread.
 *   Root cause: renderCsiStrip()'s bar color reused tagFor()'s ±0.15%
 *   "Strong/Weak" label threshold — appropriate for a text label, far too
 *   wide for a color fill — so on any day where every G10 currency stayed
 *   inside ±0.15% (i.e. most days), every bar defaulted to flat gray.
 *   dashboard.js's populateHeatmap() uses a genuinely different, tighter
 *   ±0.05%/±0.15% two-tier scheme for the heatmap's own cell shading
 *   (h-up/h-s-up, h-down/h-s-down) — a second silently-diverged methodology
 *   for the same feed, the exact "two independent parsers/methodologies of
 *   one feed drift" pattern GUIDELINES.md warns against. Fix: new
 *   csiColor(pct) mirrors dashboard.js's exact ±0.05/±0.15 breakpoints
 *   (single source of truth) instead of borrowing tagFor()'s unrelated
 *   label threshold, with the ±0.05-0.15% "mild" band getting a dimmed
 *   tint of --up/--down via color-mix() rather than going fully gray —
 *   matching the intent of the heatmap's own mild-vs-strong shading,
 *   adapted to a bar's solid fill. tagFor() itself (bias-card Strong/Weak/
 *   Neutral text labels) is untouched — that threshold is a legitimate,
 *   separate design choice, not the bug.
 *
 * v1.3.0 (2026-08-12): Three items reported by the client against the live
 *   Overview:
 *   1. Bias-card/CSI values visibly changed a few seconds after page load.
 *      Root cause: dashboard.js's populateHeatmap() only uses the real
 *      32-pair live composite once enough Finnhub ticks have cached
 *      (`rtAvailable`, ≥21/32 pairs) — before that it silently falls back
 *      to a much cruder ECB-daily-rates estimate, which this module was
 *      rendering immediately, then swapping out once the live composite
 *      arrived. Fix: dashboard.js (v8.101.0) now also exposes
 *      `window._hmStrengthsLive`; `pollHeatmapStrengths()` waits for it
 *      before its first render, only falling back to the ECB estimate if
 *      live data still hasn't shown up after the existing 15s budget —
 *      same "never show nothing forever" guarantee as before, just no
 *      longer showing a number that's about to change moments later.
 *   2. Added country flags (`.fi fi-xx`, flag-icons — the same library and
 *      class pattern already used by the CB Rates / CB Expectations panels,
 *      not emoji) to the four bias cards.
 *   3. The footer's rss / alerts / ?shortcuts buttons (and version tag) had
 *      no reason to be interactive on the Overview snapshot — they now
 *      live inside `#gi-footer-actions`, hidden by default in index.html
 *      and shown by showTerminal() / hidden by showOverview(), same
 *      lifecycle as `#gi-terminal-view` itself.
 *
 * v1.2.0 (2026-08-12): Closing the activation modal (gi-auth.js v1.6.0's new
 *   close button / Escape / backdrop click) without activating left the
 *   visitor standing inside a fully interactive terminal — reported by
 *   The client as a serious bug: "si cierro el modal queda con la terminal
 *   completa funcional. Eso no puede pasar." Root cause: "Open full
 *   terminal" was always calling showTerminal() unconditionally — the
 *   modal was never an actual gate on #gi-terminal-view, just a blurred
 *   overlay drawn on top of a terminal that was already fully revealed
 *   (with only the pre-existing PREMIUM_SECTIONS individually locked, the
 *   v8.128.0 "some panels open" model). As long as the modal had no close
 *   affordance this was invisible — v1.6.0's close button exposed it.
 *   Fix, matching the client's own suggested direction ("volver a overview
 *   si se cierra"): exposed window.giShowOverview(), which re-hides
 *   #gi-terminal-view and resumes the Overview's live bias/CSI/narrative
 *   sync. gi-auth.js's hideModal() (v1.7.0) now calls it whenever the
 *   modal closes while GI_AUTH.isActive is still false, so a non-activated
 *   visitor closing the gate always lands back on the free snapshot, never
 *   on a semi-open terminal. Active users dismissing the renewal-reminder
 *   modal are unaffected (isActive stays true throughout).
 *   Also fixed a related ordering bug in the "Open full terminal" handler:
 *   it called showTerminal() (which internally fires
 *   maybeAnnounceTerminalEntry(), the Quick-Tour/alert-tooltip trigger)
 *   BEFORE showModal() had added the 'visible' class the entry-check reads
 *   — so the tour could start firing behind a modal that wasn't visible
 *   yet, on the very same click. Reordered to showModal() then
 *   showTerminal().
 *
 * v1.1.0 (2026-08-12): Four fixes reported by the client against the live
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
 *      (the v8.128.0 "some panels open" model) — the client wants this
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
    // rss / alerts / ?shortcuts (and the version tag) are terminal features
    // with nothing to act on from the Overview snapshot — only shown once
    // the visitor is actually inside the full terminal (v8.131.0).
    document.getElementById('gi-footer-actions')?.style.setProperty('display', 'flex');
    stopOverviewSync();
    maybeAnnounceTerminalEntry();
  }

  function showOverview() {
    // v8.139.0: clear index.html's inline pre-auth flash-guard attribute
    // before touching inline display styles below. That attribute drives a
    // !important CSS override (html[data-gi-preauth="1"] #gi-terminal-view
    // { display:block !important }) meant only to survive the first paint
    // for a returning active user — if still set here, it would silently
    // fight the two inline-style writes immediately below and keep the
    // terminal visible, reintroducing the exact "closing the modal without
    // activating leaves the full terminal open" bug fixed in v1.7.0/v1.2.0
    // (see gi-auth.js's handleRevocation() for the matching defensive
    // clear on that call path).
    document.documentElement.removeAttribute('data-gi-preauth');
    document.getElementById('gi-overview')?.style.removeProperty('display');
    const tv = document.getElementById('gi-terminal-view');
    if (tv) tv.style.display = 'none';
    document.getElementById('gi-footer-actions')?.style.setProperty('display', 'none');
    // Resume the live bias/CSI/narrative sync that stopOverviewSync()
    // paused when the terminal was revealed — otherwise a visitor bounced
    // back here (gi-auth.js's hideModal(), v1.7.0) would see the Overview
    // frozen on whatever it last showed before they clicked through.
    // Guarded so re-entering an already-running poller doesn't spin a
    // second timer.
    if (!_hmSyncTimer)   pollHeatmapStrengths();
    if (!_narrSyncTimer) pollNarrative();
  }

  // Exposed synchronously (same reasoning as window.giOnTerminalShown above)
  // so gi-auth.js's hideModal() can call it regardless of script load order
  // — it only ever runs later, at user-interaction time, by which point
  // this IIFE has always already executed.
  window.giShowOverview = showOverview;

  function initToggle() {
    // Returning user with a still-valid license token skips the snapshot —
    // gi-auth.js's init() (which runs earlier in script order, see its own
    // header) has already set window.GI_AUTH.isActive synchronously by now.
    if (window.GI_AUTH && window.GI_AUTH.isActive) {
      showTerminal();
      return;
    }

    document.getElementById('gi-ov-open-terminal')?.addEventListener('click', () => {
      // the client: this specific entry point should behave like the
      // pre-v8.128.0 flow — terminal revealed *behind* the full-page
      // activation modal (blurred backdrop), not freely browsable first.
      // A visitor who is already active never reaches this branch (handled
      // above), so this always means "not yet activated".
      // v1.2.0: modal shown FIRST, terminal revealed second (reversed from
      // v1.1.0) — showTerminal() ends by calling maybeAnnounceTerminalEntry(),
      // which reads the modal's 'visible' class to decide whether the tour
      // may start; calling it before showModal() added that class let the
      // tour fire a frame early, behind a modal that wasn't shown yet.
      window.GI_AUTH && window.GI_AUTH.showModal();
      showTerminal();
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

    // Dismissing the modal WITHOUT activating (its close button / Escape /
    // backdrop click, gi-auth.js) now bounces back to the Overview instead
    // of leaving the terminal exposed underneath (v1.2.0, see header) — so
    // this poll only ever observes a genuine, unobstructed terminal entry:
    // either a successful activation, or a returning visitor who already
    // had a valid license. Kept as a poll (not a direct call from the
    // activate() success path) so it also covers that returning-visitor
    // case, which never goes through activate() at all.
    watchTerminalEntry();
  }

  // ── Bias cards + CSI strip (both read window._hmStrengths) ──────────────────
  const BIAS_CCYS = ['USD', 'EUR', 'GBP', 'JPY'];
  const CSI_CCYS  = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF']; // matches mock's 7-currency strip

  // flag-icons (already loaded globally, see index.html <link> — same CSS
  // class pattern used by the CB Rates / CB Expectations panels' .fi spans).
  const CCY_FLAG = { USD: 'us', EUR: 'eu', GBP: 'gb', JPY: 'jp', AUD: 'au', CAD: 'ca', CHF: 'ch' };

  function tagFor(pct) {
    if (pct > 0.15) return { cls: 'up', label: 'Strong' };
    if (pct < -0.15) return { cls: 'down', label: 'Weak' };
    return { cls: '', label: 'Neutral' };
  }

  // v1.3.1: CSI strip bar color — was reusing tagFor()'s ±0.15% "Strong/Weak"
  // label threshold, so on any normal/quiet day (every G10 currency inside
  // ±0.15%) every single bar rendered flat gray (var(--text2)), while the
  // Currency Strength Heatmap panel right below it — reading the exact same
  // window._hmStrengths — showed a normal green/red spread. Root cause: the
  // heatmap's populateHeatmap() (dashboard.js) uses a genuinely different,
  // tighter methodology — a 2-tier ±0.05%/±0.15% scheme (h-up/h-s-up,
  // h-down/h-s-down) with only the ±0.05% band itself rendering flat — and
  // this module had never been aligned to it, a second silently-diverged
  // "methodology" for the same feed (GUIDELINES.md: two independent
  // parsers/methodologies of one feed drift). Reported by the client:
  // Overview showing every currency gray since the prior day.
  // Fix: mirror dashboard.js's exact ±0.05/±0.15 breakpoints (single source
  // of truth, not a third invented scheme), with the ±0.05-0.15% "mild"
  // band getting a dimmed tint of --up/--down (via color-mix) instead of
  // going fully gray — matching the intent of the heatmap's own h-up/h-down
  // (mild) vs h-s-up/h-s-down (strong) two-tier shading, adapted to a bar's
  // solid-fill context rather than a cell's background tint.
  function csiColor(pct) {
    if (pct > 0.15) return 'var(--up)';
    if (pct > 0.05) return 'color-mix(in srgb, var(--up) 55%, var(--text2) 45%)';
    if (pct < -0.15) return 'var(--down)';
    if (pct < -0.05) return 'color-mix(in srgb, var(--down) 55%, var(--text2) 45%)';
    return 'var(--text2)';
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
      const flag = CCY_FLAG[ccy];
      return `<div class="gi-ov-bias-card" data-ccy="${ccy}">
        <div class="gi-ov-bias-top"><span class="gi-ov-bias-ccy">${flag ? `<span class="fi fi-${flag}" style="margin-right:6px;border-radius:2px;"></span>` : ''}${ccy}</span><span class="gi-ov-bias-tag ${t.cls}">${t.label}</span></div>
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
      const color = csiColor(pct);
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
      const haveData = strengths && strengths.length;
      // v8.131.0: dashboard.js's populateHeatmap() computes the real 32-pair
      // live composite only once enough Finnhub ticks have cached
      // (window._hmStrengthsLive) — until then it silently uses a much
      // cruder ECB-daily-rates fallback instead. Rendering that fallback
      // immediately, then swapping to the live numbers a few seconds later,
      // is exactly the "values look wrong, then change to the real ones"
      // flash the client reported. Once genuinely live (gotFirst), keep
      // re-rendering on every tick regardless — rtAvailable does not flip
      // back to false once the composite has enough pairs cached.
      if (haveData && (gotFirst || window._hmStrengthsLive === true)) {
        renderBiasRow(strengths);
        renderCsiStrip(strengths);
        gotFirst = true;
      } else if (!gotFirst && attempts >= POLL_MAX_ATTEMPTS) {
        if (haveData) {
          // Live composite never arrived within budget (thin-liquidity
          // window, WS hiccup, etc.) — render the fallback estimate rather
          // than leave the Overview on "Loading…" forever. dashboard.js's
          // own Currency Strength Heatmap panel is already showing this
          // same fallback figure at this point, so this isn't a new number,
          // just not hiding one that's already live elsewhere in the page.
          renderBiasRow(strengths);
          renderCsiStrip(strengths);
          gotFirst = true;
        } else {
          _hmSyncTimer = null; // leave the "Loading…" skeleton — no fabricated fallback numbers
          return;
        }
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
