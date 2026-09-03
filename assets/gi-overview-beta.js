/**
 * GlobalInvesting FX Terminal — Landing gate trigger, BETA  v2.0.0
 * assets/gi-overview-beta.js — include AFTER dashboard.js and gi-auth.js
 * in index-beta.html, in place of assets/gi-overview.js. Preview-only fork:
 * never referenced by production index.html.
 *
 * v2.0.0 (2026-09-02): Replaces the abandoned v1.0.0 "editorial desk"
 *   Overview redesign outright, per direct instruction to remove
 *   #gi-overview from this beta rather than continue reskinning it. That
 *   redesign, and production's own #gi-overview free-snapshot section it
 *   was meant to replace, are both gone from this file's scope now —
 *   index-beta.html no longer has a #gi-overview element at all, and
 *   #gi-terminal-view is the page's only view, visible from the static
 *   markup (no more inline display:none / no more toggle).
 *
 *   What this file does instead: for a visitor with no valid license, it
 *   fires the SAME activation gate index.html already uses for its
 *   "Open full terminal" entry point and for handleRevocation() — a
 *   full-page modal drawn over the blurred, already-rendering terminal
 *   (window.GI_AUTH.showModal(), gi-auth.js) — automatically on load,
 *   instead of waiting for a click on a button that no longer exists.
 *   Nothing about the terminal itself (dashboard.js, its panels, its data)
 *   is touched by this file — this only decides whether the activation
 *   modal is shown.
 *
 *   A returning visitor with a still-valid license needs no gate at all:
 *   gi-auth.js's own init() (which runs before this file, see script order
 *   in index-beta.html) has already set window.GI_AUTH.isActive
 *   synchronously by the time this IIFE runs, and PREMIUM_SECTIONS'
 *   existing per-panel gates (applyGates(), gi-auth.js) are the only gate
 *   that still applies to them — unchanged by this file.
 */
(function () {
  'use strict';

  function gateIfInactive() {
    if (window.GI_AUTH && window.GI_AUTH.isActive) return;
    if (window.GI_AUTH && typeof window.GI_AUTH.showModal === 'function') {
      window.GI_AUTH.showModal();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', gateIfInactive);
  } else {
    gateIfInactive();
  }
})();
