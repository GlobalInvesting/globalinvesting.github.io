/**
 * GlobalInvesting FX Terminal — Landing gate trigger  v2.0.1
 * assets/gi-overview.js — include AFTER dashboard.js and gi-auth.js in
 * index.html.
 *
 * v2.0.1 (2026-09-02): Fixed a real bug found while auditing v2.0.0 live
 *   against index.html before delivery, not just described — #gi-footer-
 *   actions (rss/alerts/shortcuts/version-tag cluster) still ships with
 *   `style="display:none"` in the static markup, same as under the old
 *   overview→terminal toggle, but v2.0.0's rewrite dropped the old
 *   showTerminal()/showOverview() functions entirely (no toggle left to
 *   drive them) without replacing the one thing they did that still
 *   matters here: revealing that element once the visitor is genuinely
 *   past the gate. Grepped dashboard.js/onboarding.js/gi-auth.js for any
 *   other reference to `gi-footer-actions` before concluding this — none
 *   exists, so it was a real dead-forever element, not a relocated one.
 *   Fixed by having lockTerminal()/unlockTerminal() toggle it directly,
 *   keyed off the same hard-gate state as the terminal blur itself (hidden
 *   while gated, shown once unlocked) — no new event, no new state.
 *
 * v2.0.0 (2026-09-02): Complete rewrite, promoted from index-beta.html's
 *   gi-overview-beta.js v2.1.0 — removes the free #gi-overview snapshot
 *   page (v8.129.0–v8.351.0) outright, per direct instruction: with the
 *   entire terminal already blurred and gated behind a non-dismissible
 *   activation modal for any non-activated visitor, the snapshot no longer
 *   showed anything a locked terminal didn't already communicate, and its
 *   "no account required" framing was actively misleading once nothing on
 *   the page was actually usable pre-auth. index.html no longer has a
 *   #gi-overview element or an overview/terminal toggle — #gi-terminal-view
 *   is the page's only view, visible from the static markup.
 *
 *   What this file now does, in full:
 *   1. Landing gate — on load, if the visitor is not activated, adds a
 *      `gi-hard-gated` class to #gi-terminal-view (CSS in index.html: blur
 *      + pointer-events:none across the WHOLE terminal — quote bar, price
 *      chart, FX pairs table, currency-strength heatmap, market sessions,
 *      everything) and calls window.GI_AUTH.showModal(false) — the false
 *      argument (gi-auth.js v1.7.7+) makes the modal non-dismissible via
 *      ×/backdrop/Escape until a real activation succeeds. This sits
 *      alongside, not instead of, gi-auth.js's own per-panel
 *      PREMIUM_SECTIONS gate (applyGates()), which still runs unchanged and
 *      still matters once the visitor activates — PREMIUM_SECTIONS are
 *      simply redundant with the whole-terminal blur while inactive.
 *      Listens for gi-auth.js's `gi-auth:activated` / `gi-auth:revoked`
 *      window events to lift/reapply the class, rather than polling
 *      isActive — activation and revocation are the only two things that
 *      should ever change this page's gate state after load.
 *   2. "Terminal actually usable" hook — window.giOnTerminalShown(cb),
 *      exposed synchronously (not inside DOMContentLoaded) so onboarding.js
 *      and dashboard.js, which load after this file (see index.html script
 *      order), can reference it unconditionally. This API is UNCHANGED from
 *      v1.4.2's contract (same function name, same one-shot-fire behavior,
 *      same window._giTerminalShown flag) — only what counts as "shown" is
 *      redefined: previously, the moment #gi-terminal-view's inline display
 *      style flipped away from 'none' (the overview→terminal toggle);
 *      now, since #gi-terminal-view is unconditionally in the DOM and
 *      visible (just blurred+inert while gated), "shown" means the
 *      opposite of gi-hard-gated — the visitor is genuinely activated AND
 *      the activation modal isn't still covering the screen. A first-time
 *      visitor's Quick Tour (onboarding.js) and "Configure Alert" tooltip
 *      (dashboard.js) both gate their init on this exact hook, same as
 *      before — neither needed to change, only this file's own definition
 *      of the event they're waiting for.
 *
 * Deliberately has NO independent data fetch or computation of its own —
 * gi-auth.js remains the single authoritative source for isActive/modal
 * state, per GUIDELINES.md's "two independent parsers of the same feed
 * silently drift" principle applied to auth state as much as market data.
 *
 * See CHANGELOG.md v8.354.0 for the full promotion writeup and the
 * matching index.html/access.html changes shipped in the same session.
 */
(function () {
  'use strict';

  const TERMINAL_ID       = 'gi-terminal-view';
  const HARD_GATE_CLASS   = 'gi-hard-gated';
  const FOOTER_ACTIONS_ID = 'gi-footer-actions';

  function terminalEl() {
    return document.getElementById(TERMINAL_ID);
  }

  function footerActionsEl() {
    return document.getElementById(FOOTER_ACTIONS_ID);
  }

  // ── "Terminal actually usable" hook (unchanged contract from v1.4.2) ────
  window._giTerminalShown = false;
  window.giOnTerminalShown = function (cb) {
    if (typeof cb !== 'function') return;
    if (window._giTerminalShown) { cb(); return; }
    document.addEventListener('gi:terminal-shown', cb, { once: true });
  };

  function maybeAnnounceTerminalEntry() {
    if (window._giTerminalShown) return;
    const tv = terminalEl();
    if (!tv || tv.classList.contains(HARD_GATE_CLASS)) return;
    const modal = document.getElementById('gi-auth-modal');
    if (modal && modal.classList.contains('visible')) return;
    window._giTerminalShown = true;
    document.dispatchEvent(new CustomEvent('gi:terminal-shown'));
  }

  // ── Landing gate ──────────────────────────────────────────────────────
  function lockTerminal() {
    terminalEl()?.classList.add(HARD_GATE_CLASS);
    footerActionsEl()?.style.setProperty('display', 'none');
  }

  function unlockTerminal() {
    terminalEl()?.classList.remove(HARD_GATE_CLASS);
    footerActionsEl()?.style.setProperty('display', 'flex');
    maybeAnnounceTerminalEntry();
  }

  function gateIfInactive() {
    if (window.GI_AUTH && window.GI_AUTH.isActive) {
      unlockTerminal();
      return;
    }
    lockTerminal();
    if (window.GI_AUTH && typeof window.GI_AUTH.showModal === 'function') {
      // false = non-dismissible — there is deliberately no way to see the
      // real terminal on this page without activating.
      window.GI_AUTH.showModal(false);
    }
  }

  window.addEventListener('gi-auth:activated', unlockTerminal);
  window.addEventListener('gi-auth:revoked', gateIfInactive);

  function init() {
    gateIfInactive();
    // Covers the case where the visitor was already active on load (a
    // returning license holder) — unlockTerminal() above already announced
    // it, but a modal visible for an unrelated reason (e.g. a renewal
    // reminder) shouldn't block the one-shot hook from ever firing once it
    // closes. Short-lived poll, mirrors the old watchTerminalEntry() budget.
    let attempts = 0;
    const t = setInterval(() => {
      attempts++;
      maybeAnnounceTerminalEntry();
      if (window._giTerminalShown || attempts > 1800) clearInterval(t); // ~21 min cap
    }, 700);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
