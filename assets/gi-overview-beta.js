/**
 * GlobalInvesting FX Terminal — Landing gate trigger, BETA  v2.1.0
 * assets/gi-overview-beta.js — include AFTER dashboard.js and gi-auth.js
 * in index-beta.html, in place of assets/gi-overview.js. Preview-only fork:
 * never referenced by production index.html.
 *
 * v2.1.0 (2026-09-02): Two problems found live against v2.0.0's landing
 *   gate, both fixed here without touching gi-auth.js's default behavior
 *   for production (verified — see that file's own v1.7.7 header).
 *
 *   1. The modal could be dismissed (× / backdrop click / Escape),
 *      leaving a non-activated visitor standing in front of the real,
 *      unlocked terminal. Fixed by calling the new
 *      window.GI_AUTH.showModal(false) (gi-auth.js v1.7.7) instead of
 *      showModal() — the modal is now undismissible until a valid
 *      activation actually succeeds.
 *   2. Only the seven PREMIUM_SECTIONS (gi-auth.js's applyGates()) were
 *      ever gated — every other panel (quote bar, price chart, FX pairs
 *      table, currency-strength heatmap, market sessions, reference
 *      spreads) rendered fully live and interactive underneath/beside the
 *      modal for a visitor with no license at all. That is the opposite
 *      of the approved blur+modal mock, where NOTHING is usable pre-auth.
 *      Fixed by adding a `gi-hard-gated` class to #gi-terminal-view
 *      itself while inactive (CSS in index-beta.html: blur + pointer-events:
 *      none across the whole view, not just PREMIUM_SECTIONS) — this sits
 *      alongside, not instead of, gi-auth.js's own per-panel gate, which
 *      still runs unchanged and still matters once the visitor activates
 *      (PREMIUM_SECTIONS stay real gates in production; here they're
 *      simply redundant while `gi-hard-gated` is also active).
 *
 *   Listens for gi-auth.js v1.7.7's new `gi-auth:activated` /
 *   `gi-auth:revoked` window events to lift/reapply the class instead of
 *   polling isActive — activation and revocation are the only two things
 *   that should ever change this page's gate state after load.
 *
 * v2.0.0 (2026-09-02): Replaces the abandoned v1.0.0 "editorial desk"
 *   Overview redesign outright, per direct instruction to remove
 *   #gi-overview from this beta rather than continue reskinning it. That
 *   redesign, and production's own #gi-overview free-snapshot section it
 *   was meant to replace, are both gone from this file's scope now —
 *   index-beta.html no longer has a #gi-overview element at all, and
 *   #gi-terminal-view is the page's only view, visible from the static
 *   markup (no more inline display:none / no more toggle).
 */
(function () {
  'use strict';

  const TERMINAL_ID   = 'gi-terminal-view';
  const HARD_GATE_CLASS = 'gi-hard-gated';

  function terminalEl() {
    return document.getElementById(TERMINAL_ID);
  }

  function lockTerminal() {
    terminalEl()?.classList.add(HARD_GATE_CLASS);
  }

  function unlockTerminal() {
    terminalEl()?.classList.remove(HARD_GATE_CLASS);
  }

  function gateIfInactive() {
    if (window.GI_AUTH && window.GI_AUTH.isActive) {
      unlockTerminal();
      return;
    }
    lockTerminal();
    if (window.GI_AUTH && typeof window.GI_AUTH.showModal === 'function') {
      // false = non-dismissible — see header changelog. There is
      // deliberately no other way to see the real terminal on this page.
      window.GI_AUTH.showModal(false);
    }
  }

  window.addEventListener('gi-auth:activated', unlockTerminal);
  window.addEventListener('gi-auth:revoked', gateIfInactive);

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', gateIfInactive);
  } else {
    gateIfInactive();
  }
})();
