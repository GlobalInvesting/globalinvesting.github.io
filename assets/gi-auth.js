/**
 * GlobalInvesting FX Terminal — License Auth Module  v1.7.5
 * assets/gi-auth.js  — include BEFORE dashboard.js in index.html
 *
 * v1.7.5 (2026-08-14): Reported by the client (screenshot) — the free Overview
 *   snapshot briefly flashed before the full terminal appeared, for every
 *   visitor including ones with a fully-valid, already-active license.
 *   Root cause: gi-overview.js's initToggle() decides overview-vs-terminal
 *   by reading window.GI_AUTH.isActive, but that's only set once this
 *   file's `defer`red init() runs — which happens after the browser has
 *   already parsed and painted the body's static markup (#gi-overview
 *   visible, #gi-terminal-view display:none are the no-JS defaults). Fixed
 *   with a new synchronous inline guard script at the very top of
 *   index.html's <head> (runs before any deferred script, before first
 *   paint) that re-checks the same JWT_KEY/isJWTValid() logic and sets
 *   data-gi-preauth="1" on <html> if still valid; a matching CSS override
 *   in index.html's critical <style> block hides #gi-overview / shows
 *   #gi-terminal-view before the browser ever paints the snapshot. This
 *   file's own isActive check and gi-overview.js's showTerminal() still run
 *   normally afterward — unchanged, and idempotent when the guard already
 *   guessed correctly.
 *   handleRevocation() now also clears data-gi-preauth the moment a token
 *   is found invalid — otherwise the attribute's !important CSS override
 *   would still be sitting on <html> the next time the visitor closes the
 *   activation modal without reactivating, fighting showOverview()'s own
 *   inline-style attempt to hide the terminal and reintroducing the exact
 *   "closing the modal leaves the full terminal open" bug fixed in
 *   v1.7.0/gi-overview.js v1.2.0. See index.html's guard-script comment and
 *   gi-overview.js's showOverview() for the matching defensive clear.
 *
 * v1.7.4 (2026-08-13): Reported by the client — on mobile the activation modal
 *   "ocupa toda la pantalla y no se ve el botón X para cerrarlo." Root cause:
 *   #gi-auth-modal (the backdrop) had no height cap or overflow handling and
 *   simply vertically-centered #gi-auth-box regardless of viewport height —
 *   on a phone, the box's natural content height (form + broker-walkthrough
 *   copy + hint block) routinely exceeds the viewport, so centering pushed
 *   the box's own top — where the close button lives, position:absolute
 *   top:10px relative to the box, not the viewport — above the visible top
 *   edge, with no scroll available to reach it. Fix: below 640px, the
 *   backdrop switches from centered to top-anchored + scrollable
 *   (align-items:flex-start; overflow-y:auto), so the box always starts
 *   flush with the viewport top (close button immediately visible) and any
 *   remaining content scrolls via the backdrop. Desktop layout (≥640px)
 *   unchanged.
 *
 * v1.7.3 (2026-08-12): Reverted per the client's explicit instruction — the
 *   "See the full walkthrough & pricing" modal link now points at plain
 *   `access.html` (no anchor), not `access.html#broker-access` as landed in
 *   v1.7.2. v1.7.2's own intent (skip past the featured EA card straight to
 *   the broker cards) is superseded by this direction; the link now lands
 *   at the top of access.html like a normal page link. No JS behavior
 *   change, plain href fix. NOTE: index.html's own "Broker access" Overview
 *   pill still points at `access.html#broker-access` (fixed in v8.133.2,
 *   untouched here) — not in scope for this change, flag to the client if the
 *   same reversion should apply there too.
 *
 * v1.7.2 (2026-08-12): Follow-up to v1.7.1's incomplete broker-access
 *   scroll-target fix. The client's original v1.4.0 fix already documented
 *   the intent — point "See the full walkthrough & pricing" at the
 *   Vantage/TMGM broker cards, not the page top — but both this link and
 *   a same-named "Broker access" pill on the Overview (index.html) still
 *   pointed at `access.html#sub-heading`, the *section* title above BOTH
 *   pricing cards (the featured EA card sits first), not the broker card
 *   itself. access.html's own hero CTA got a proper `#broker-access`
 *   anchor id in the prior session's access.html changes; these two
 *   external entry points into the same page were missed and still
 *   landed on the section top — on tall desktop viewports enough of the
 *   page was visible that the broker card was still partly in view, so it
 *   read as "working," but on mobile and smaller desktop viewports the
 *   featured EA card filled the screen instead, with the broker card
 *   scrolled out of view below the fold.
 *   Fix: repointed this modal's link to `access.html#broker-access`
 *   (index.html's Overview pill fixed in the same pass — see its own
 *   inline history). No JS behavior change, plain href fix.
 *
 * v1.7.1 (2026-08-12): GUIDELINES.md compliance fix reported by the client —
 *   the PREMIUM_SECTIONS gate overlay's lock icon (.gi-gate-icon) was
 *   rendering the literal 🔒 emoji (&#128274;), which read as an odd
 *   orange/yellow glyph rather than a neutral lock and violates
 *   GUIDELINES.md's "no emojis in any HTML" rule outright. Replaced with a
 *   monochrome inline SVG padlock (currentColor stroke, matches the
 *   existing --blue icon color at 0.65 opacity). .gi-gate-icon no longer
 *   needs font-size (was sizing a text glyph); added line-height:0 so the
 *   SVG doesn't pick up extra inline-box height. Companion fix to the same
 *   emoji issue on the Overview's locked-preview cards, made in index.html
 *   this same session — see index.html's inline <style> comment history
 *   for that half of the fix and the "Correlations, options analytics,
 *   IRM" → "risk regime" copy correction (IRM/"Institutional Risk Manager"
 *   is the MT5 EA's standalone indicator product name, not a web-terminal
 *   panel — the locked-preview card was describing the Cross-Asset Risk
 *   panel and had no business naming an unrelated EA product there).
 *
 * v1.7.0 (2026-08-12): Serious bug reported by the client the same day v1.6.0
 *   shipped its close button: "si cierro el modal queda con la terminal
 *   completa funcional. Eso no puede pasar." Root cause: the "Open full
 *   terminal" entry point (gi-overview.js) always called showTerminal()
 *   unconditionally — the modal was never an actual gate on
 *   #gi-terminal-view, just a blurred overlay drawn on top of a terminal
 *   that was already fully revealed underneath (with only the pre-existing
 *   PREMIUM_SECTIONS individually locked — the v8.128.0 "some panels open"
 *   model, intentional for OTHER entry paths but not for this one, per
 *   The client's v8.130.0-session request). As long as the modal had no way
 *   to close except reloading, this was invisible; v1.6.0's close button
 *   exposed it — closing without activating simply revealed the terminal
 *   that had already been sitting there the whole time.
 *   Fix, matching the client's own suggested direction ("volver a overview
 *   si se cierra"): hideModal() now calls window.giShowOverview() (new in
 *   gi-overview.js v1.2.0) whenever it runs while GI_AUTH.isActive is
 *   still false — re-hiding the terminal and returning to the free
 *   Overview snapshot. This covers all three close paths (×, Escape,
 *   backdrop click) plus handleRevocation()'s showModal(), since a
 *   revoked session also has isActive === false by the time its modal can
 *   be closed. Active users are unaffected — e.g. dismissing the
 *   renewal-reminder modal (isActive stays true; the license just expires
 *   soon, it hasn't yet) leaves them exactly where they were, as before.
 *   See gi-overview.js v1.2.0 header for the companion fix to a related
 *   ordering bug in the "Open full terminal" click handler.
 *
 * v1.6.0 (2026-08-12): The activation modal had no way to close except
 *   reloading the page — no close button, no backdrop click, no Escape key;
 *   hideModal() was only ever called from activate()'s success path.
 *   Reported by the client against the new Overview snapshot (v8.129.0): a
 *   visitor clicking a locked-preview card sees the modal with no way back
 *   short of a hard refresh. Added an "×" close button in the modal header,
 *   plus click-on-backdrop and Escape-key handlers, all calling the
 *   existing hideModal() — no change to the activation flow itself.
 *   Deliberately allowed in every state, including the revocation message
 *   (handleRevocation() still sets a clear "access revoked" status text
 *   first; closing the modal afterward doesn't undo the revocation, it
 *   just lets the person stop looking at it).
 *
 * v1.5.0 (2026-08-12): Removed the full-page showModal() call from the
 *   initial window 'load' handler — the terminal no longer blocks the
 *   entire viewport behind a modal + backdrop blur on first visit.
 *   applyGates() (unchanged) still runs on load and continues to gate
 *   PREMIUM_SECTIONS individually via the existing per-panel
 *   .gi-gate-overlay + "Activate Access" button, which already calls
 *   window.GI_AUTH.showModal(). Non-premium areas (quote bar, sidebar,
 *   FX Pairs, CSI, charts) were never in PREMIUM_SECTIONS and were only
 *   ever blocked by the full-page modal — they are now interactive
 *   immediately, matching the "some panels open, some locked" model
 *   agreed with the client over showing nothing until activation. The
 *   full-page showModal() is UNCHANGED for two other call sites, both
 *   intentional: (1) handleRevocation() — a previously-active session
 *   being cut off should get a clear, unmissable message, not a quiet
 *   reversion to locked panels; (2) any .gi-gate-btn's onclick, i.e. the
 *   user explicitly asking to activate. Also fixed the per-panel gate
 *   overlay's copy (was "verified TMGM account", stale since v1.3.0 added
 *   Vantage as a second broker partner) and pointed the modal's "See the
 *   full walkthrough & pricing" link at access.html#sub-heading (the
 *   Vantage/TMGM broker-access section) instead of the page top, so a
 *   user who clicks through from a locked panel lands directly on the two
 *   broker cards rather than having to scroll to find them.
 *
 * v1.4.0 (2026-08-10): pingSession() now reads the /session/ping response
 *   status instead of only using .catch() for network errors. A 403 means
 *   the license worker has revoked this (account, server) pair (new
 *   revoked_accounts table + isRevoked() in worker.js, GI_admin_dashboard's
 *   "Revoke" button) — handleRevocation() clears the stored JWT, stops the
 *   ping timer, and re-shows the activation gate so the terminal locks
 *   without waiting for a hard refresh. A 401 (malformed/expired token) is
 *   treated the same way defensively, though isJWTValid()'s own exp check
 *   should normally catch that case first. Still best-effort in the sense
 *   that a network error or a tab that's already closed can't be reached —
 *   see the "Revocation" section in worker.js's header docblock for the
 *   known propagation-delay limit this does NOT solve.
 *
 * v1.3.0 (2026-08-05): activation modal's broker copy updated from
 *   TMGM-only to "TMGM or Vantage" — GlobalInvesting became an IB of
 *   Vantage Global Limited alongside the existing TMGM partnership.
 *   No functional/JWT change: the license worker's /admin/grant and
 *   /validate flows were already broker-agnostic (account+server, where
 *   server is the MT5 broker server name, was never checked against a
 *   specific broker) — see GUIDELINES.md → "License worker — audit trail
 *   & error visibility". Also fixed a pre-existing cache-buster drift:
 *   index.html and sw.js were still pinned to ?v=1.0.2 despite this file's
 *   own header already reading v1.2.1 before this change — same drift
 *   class documented for calendar-panel.js/sw.js in CHANGELOG v8.100.7.
 *
 * Flow:
 *   1. On load, check localStorage/sessionStorage for a valid JWT
 *   2. If none, show activation modal after panels have rendered (~400ms)
 *   3. POST key+account+server to the Cloudflare Worker → receive JWT
 *   4. Store JWT; remove gate overlays; expose window.GI_AUTH
 *   5. Once active, ping the Worker's /session/ping every 3 minutes so the
 *      license backend can track which activated accounts currently have
 *      the terminal open (best-effort — a failed ping never blocks the UI).
 *      Each ping carries X-Session-Id (see getOrCreateSessionId()) so two
 *      devices sharing one token don't overwrite each other's presence row.
 *   6. On pagehide/visibilitychange (tab close, navigation away, backgrounding),
 *      fire one last /session/ping via navigator.sendBeacon() (v1.2.0) — a
 *      regular periodic ping can be up to 3 minutes stale by the time a user
 *      closes the tab; sendBeacon is designed to reliably complete even as
 *      the page is being torn down, which a normal fetch() is not guaranteed
 *      to do. sendBeacon cannot set custom headers, so the token and session
 *      id ride as ?token=&session_id= query params on this call only —
 *      see handleSessionPing()'s header-first, query-param-fallback logic
 *      in worker.js.
 *
 * Premium sections gated (real index.html IDs):
 *   section-positioning   — CFTC COT
 *   section-sentiment     — Retail Sentiment
 *   section-cb-expectations — CB Rate Expectations
 *   section-macro         — Composite / Macro
 *   narrative             — AI Narrative
 *   section-news          — News / Intel
 *   rightpanel            — Sidebar (CB Rates, ESI, Carry)
 */
(function () {
  'use strict';

  // ── Config ──────────────────────────────────────────────────────────────────
  // Update after: wrangler deploy  →  copy the workers.dev URL here
  const WORKER_URL = 'https://gi-license-worker.globalinvestingmarkets.workers.dev';
  const JWT_KEY    = 'gi_license_token';
  const SESSION_ID_KEY = 'gi_session_id'; // per-browser device id — see getOrCreateSessionId()
  const MODAL_ID   = 'gi-auth-modal';

  // Periodic presence ping — lets the license worker track which activated
  // accounts currently have the terminal open (best-effort, silent on failure).
  const SESSION_PING_INTERVAL_MS = 3 * 60 * 1000; // 3 minutes
  let sessionPingTimer = null;

  const PREMIUM_SECTIONS = [
    'section-positioning',       // CFTC COT
    'section-sentiment',         // Retail Sentiment
    'section-cb-expectations',   // CB Rate Expectations
    'section-macro',             // Composite / Macro
    'narrative',                 // AI Narrative
    'section-news',              // News / Intel full panel
    'rightpanel',                // Sidebar: CB Rates, ESI, Carry ranking
  ];

  // ── CSS ─────────────────────────────────────────────────────────────────────
  const MODAL_CSS = `
#gi-auth-modal {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 99999;
  background: rgba(0,0,0,0.82);
  backdrop-filter: blur(4px);
  align-items: center;
  justify-content: center;
  font-family: var(--font-ui, 'Consolas', 'Courier New', monospace);
}
#gi-auth-modal.visible { display: flex; }
#gi-auth-box {
  position: relative;
  background: var(--bg2, #141414);
  border: 1px solid var(--border, #323232);
  border-top: 2px solid var(--blue,#4f7fff);
  padding: 32px 36px 28px;
  width: 460px;
  max-width: 92vw;
  box-shadow: 0 24px 64px rgba(0,0,0,0.7);
}
#gi-auth-close {
  position: absolute;
  top: 10px;
  right: 12px;
  background: none;
  border: none;
  color: var(--text3, #727272);
  font-size: 22px;
  line-height: 1;
  cursor: pointer;
  padding: 4px 6px;
  transition: color 0.15s;
}
#gi-auth-close:hover, #gi-auth-close:focus { color: var(--text, #E8E4DC); }
#gi-auth-box h2 {
  margin: 0 0 4px;
  font-size: 14px;
  font-weight: 600;
  color: var(--text, #E8E4DC);
  letter-spacing: 0.08em;
}
.gi-auth-sub {
  font-size: 11px;
  color: var(--text3, #727272);
  margin: 0 0 24px;
  line-height: 1.6;
}
.gi-auth-sub a { color: var(--blue,#4f7fff); text-decoration: none; }
.gi-auth-sub a:hover { text-decoration: underline; }
#gi-auth-box label {
  display: block;
  font-size: 10px;
  color: var(--text2, #A0A0A0);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
#gi-auth-box input {
  width: 100%;
  background: var(--bg, #0D0D0D);
  border: 1px solid var(--border, #323232);
  color: var(--text, #E8E4DC);
  font-family: inherit;
  font-size: 13px;
  padding: 9px 10px;
  box-sizing: border-box;
  margin-bottom: 14px;
  outline: none;
  transition: border-color 0.15s;
}
#gi-auth-box input:focus { border-color: var(--blue,#4f7fff); }
#gi-auth-box input::placeholder { color: var(--border2, #404040); }
#gi-auth-activate {
  width: 100%;
  background: var(--blue,#4f7fff);
  border: none;
  color: #fff;
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 11px 0;
  cursor: pointer;
  transition: background 0.15s;
}
#gi-auth-activate:hover { background: var(--blue,#4f7fff); }
#gi-auth-activate:disabled { background: #444; cursor: default; }
#gi-auth-status {
  font-size: 11px;
  margin-top: 6px;
  min-height: 14px;
  text-align: center;
}
#gi-auth-status.is-err { color: var(--down,#e03030); }
#gi-auth-status.is-ok  { color: var(--up,#00b050); }
#gi-auth-hint {
  font-size: 10px;
  color: var(--text3,#666666);
  margin-top: 12px;
  line-height: 1.6;
  border-top: 1px solid var(--border, #222);
  padding-top: 12px;
}
#gi-auth-hint code {
  color: var(--text2, #A0A0A0);
  background: var(--bg, #0D0D0D);
  padding: 1px 4px;
  font-size: 10px;
}
.gi-gate-overlay {
  position: absolute;
  inset: 0;
  background: rgba(13,13,13,0.90);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 100;
  gap: 10px;
}
.gi-gate-icon { color: var(--blue,#4f7fff); opacity: 0.65; line-height: 0; }
.gi-gate-msg {
  font-family: var(--font-ui, 'Consolas', monospace);
  font-size: 11px;
  color: var(--text3, #727272);
  text-align: center;
  line-height: 1.6;
  max-width: 220px;
}
.gi-gate-btn {
  font-family: var(--font-ui, 'Consolas', monospace);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--blue,#4f7fff);
  background: transparent;
  border: 1px solid var(--blue,#4f7fff);
  padding: 7px 18px;
  cursor: pointer;
  margin-top: 4px;
  transition: background 0.15s, color 0.15s;
}
.gi-gate-btn:hover { background: var(--blue,#4f7fff); color: #fff; }
#gi-renew-banner {
  display: none;
  position: fixed;
  bottom: 16px;
  right: 20px;
  z-index: 9999;
  background: var(--bg2, #141414);
  border: 1px solid var(--blue,#4f7fff);
  padding: 10px 14px;
  font-family: var(--font-ui, 'Consolas', monospace);
  font-size: 11px;
  color: var(--text, #E8E4DC);
  align-items: center;
  gap: 14px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}
#gi-renew-banner.visible { display: flex; }
#gi-renew-banner span { color: var(--text3, #727272); }
#gi-renew-banner strong { color: var(--blue,#4f7fff); }
#gi-renew-btn {
  background: transparent;
  border: 1px solid var(--blue,#4f7fff);
  color: var(--blue,#4f7fff);
  font-family: inherit;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 5px 12px;
  cursor: pointer;
  white-space: nowrap;
  flex-shrink: 0;
}
#gi-renew-btn:hover { background: var(--blue,#4f7fff); color: #fff; }
#gi-renew-dismiss {
  background: none;
  border: none;
  color: var(--text3, #727272);
  font-size: 16px;
  cursor: pointer;
  padding: 0;
  line-height: 1;
  flex-shrink: 0;
}

/* v1.7.4: mobile — the modal previously only ever centered #gi-auth-box
   vertically with no height cap and no scroll on the outer #gi-auth-modal
   backdrop. On a phone viewport the box's natural content height (form +
   broker-walkthrough copy + hint block) regularly exceeds the visible
   viewport; since the backdrop had no overflow handling, the excess simply
   rendered off-screen above/below the fold with no way to reach it — most
   visibly, the box's own top (where #gi-auth-close lives, position:absolute
   top:10px *within the box*, not the viewport) got centered above the top
   edge of the screen, so the close button was invisible and unreachable.
   Reported by the client: "en mobile ocupa toda la pantalla y no se ve el
   botón X para cerrarlo." Fix: below 640px, switch the backdrop from
   vertically-centered to top-anchored + scrollable, so the box always
   starts flush with the top of the viewport (close button immediately
   visible, no centering overflow) and any content taller than the
   viewport scrolls via the backdrop itself rather than being clipped. */
@media (max-width: 640px) {
  #gi-auth-modal {
    align-items: flex-start;
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
    padding: 14px 0 32px;
  }
  #gi-auth-box {
    width: 100%;
    max-width: 94vw;
    margin: 0 auto;
  }
}
`;

  // ── Modal HTML ─────────────────────────────────────────────────────────────
  const MODAL_HTML = `
<div id="${MODAL_ID}" role="dialog" aria-modal="true" aria-label="Activate terminal access">
  <div id="gi-auth-box">
    <button id="gi-auth-close" type="button" aria-label="Close">&times;</button>
    <h2>ACTIVATE TERMINAL</h2>
    <p class="gi-auth-sub">
      Full access is included with the
      <a href="https://www.mql5.com/en/market/product/180326" target="_blank" rel="noopener">
        Global Investing FX Terminal EA
      </a>
      on MQL5 Market. Enter the activation key shown in your MT5 terminal to unlock all panels.
    </p>
    <p class="gi-auth-sub" style="font-size:12px;opacity:0.85;margin-top:-10px;">
      Opened a verified account with one of our partner brokers instead? As our referral partner,
      account holders get full web terminal access too &mdash; <a href="contact.html">contact us</a>
      for your access link, no MT5 key required.
    </p>
    <p class="gi-auth-sub" style="font-size:12px;opacity:0.75;margin-top:-8px;">
      New here? <a href="access.html">See the full walkthrough &amp; pricing &rarr;</a>
    </p>

    <label for="gi-inp-key">Activation Key (from MT5 terminal top bar)</label>
    <input id="gi-inp-key" type="text" placeholder="XXXX-XXXX-XXXX" maxlength="14"
           autocomplete="off" spellcheck="false" />

    <label for="gi-inp-account">MT5 Account Number</label>
    <input id="gi-inp-account" type="text" placeholder="e.g. 12345678"
           maxlength="20" autocomplete="off" />

    <label for="gi-inp-server">Broker Server Name</label>
    <input id="gi-inp-server" type="text" placeholder="e.g. Broker-Live01"
           maxlength="80" autocomplete="off" />

    <button id="gi-auth-activate">Activate</button>
    <div id="gi-auth-status" role="alert" aria-live="assertive"></div>

    <div id="gi-auth-hint">
      <strong style="color:var(--text3,#666666)">Where is my key?</strong><br>
      Open MetaTrader 5 → attach the <em>Global Investing FX Terminal EA</em> to any chart.
      The activation key appears in the terminal top bar as <code>KEY:XXXX-XXXX-XXXX</code>.
      Copy the 14-character code (dashes included) and paste it above.<br><br>
      The account number and server name must match the MT5 account the EA is running on.
    </div>
  </div>
</div>
`;

  const RENEW_HTML = `
<div id="gi-renew-banner" role="status" aria-live="polite">
  <span>License expires in <strong id="gi-renew-days">?</strong> days &mdash; re-enter your key to renew</span>
  <button id="gi-renew-btn">Renew</button>
  <button id="gi-renew-dismiss" aria-label="Dismiss">&times;</button>
</div>
`;

  // ── JWT helpers ────────────────────────────────────────────────────────────
  function parseJWT(token) {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) return null;
      return JSON.parse(atob(parts[1].replace(/-/g, '+').replace(/_/g, '/')));
    } catch { return null; }
  }

  function isJWTValid(token) {
    if (!token) return false;
    const p = parseJWT(token);
    if (!p || !p.exp) return false;
    return p.exp > Math.floor(Date.now() / 1000);
  }

  function jwtDaysRemaining(token) {
    if (!token) return 0;
    const p = parseJWT(token);
    if (!p || !p.exp) return 0;
    return Math.floor((p.exp - Math.floor(Date.now() / 1000)) / 86400);
  }

  // ── Presence ping ──────────────────────────────────────────────────────────
  // getOrCreateSessionId() — one random id per browser/device, persisted in
  // localStorage. This is what lets the admin dashboard tell apart two
  // different browsers using the *same* JWT (e.g. an admin testing a
  // founder/promo grant link, then handing it to the real recipient) —
  // without it, both pings collapse into one active_sessions row and each
  // overwrites the other's IP/location. It is NOT a fingerprint or tracking
  // id in the analytics sense — it's just a random UUID, generated locally,
  // used only to avoid two devices being mistaken for one on this admin view.
  function getOrCreateSessionId() {
    try {
      let id = localStorage.getItem(SESSION_ID_KEY);
      if (id) return id;
      id = (crypto && crypto.randomUUID) ? crypto.randomUUID() : randomIdFallback();
      localStorage.setItem(SESSION_ID_KEY, id);
      return id;
    } catch {
      // localStorage unavailable (private mode, etc.) — fall back to an
      // in-memory id that's stable for this page load only.
      if (!getOrCreateSessionId._mem) getOrCreateSessionId._mem = randomIdFallback();
      return getOrCreateSessionId._mem;
    }
  }

  function randomIdFallback() {
    return 'sess-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
  }

  function pingSession(token) {
    if (!token) return;
    fetch(`${WORKER_URL}/session/ping`, {
      method:  'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'X-Session-Id':  getOrCreateSessionId(),
      },
    }).then(res => {
      // 401/403 here means the worker rejected this token outright — most
      // commonly a revocation (see isRevoked() in worker.js), possibly an
      // expired/malformed token isJWTValid() didn't catch. Distinct from a
      // network failure (caught below), which must NOT lock the user out.
      if (res && (res.status === 401 || res.status === 403)) handleRevocation();
    }).catch(() => {}); // network errors are best-effort — must never disrupt the terminal
  }

  // Fired when the worker rejects an active session's token (see pingSession
  // above). Clears local state and re-locks the terminal — the person sees
  // the same activation modal as a first-time visitor, rather than the
  // terminal silently continuing to render with revoked access.
  function handleRevocation() {
    if (sessionPingTimer) { clearInterval(sessionPingTimer); sessionPingTimer = null; }
    try { sessionStorage.removeItem(JWT_KEY); } catch {}
    try { localStorage.removeItem(JWT_KEY); }   catch {}
    window.GI_AUTH.isActive = false;
    // v8.139.0: clear the pre-auth flash guard's attribute (index.html's
    // inline <head> script) the moment a token is known-invalid. That
    // attribute drives a !important CSS override that force-shows
    // #gi-terminal-view — if left set, it would fight showOverview()'s own
    // inline-style attempt to hide the terminal the next time the visitor
    // closes the modal without reactivating (gi-overview.js's hideModal()
    // path, v1.7.0), reintroducing the exact "closing the modal leaves the
    // full terminal open" bug v1.7.0 fixed. See gi-overview.js's
    // showOverview() for the matching defensive clear on that side.
    document.documentElement.removeAttribute('data-gi-preauth');
    document.getElementById('gi-renew-banner')?.classList.remove('visible');
    applyGates();
    showModal();
    const statusEl = document.getElementById('gi-auth-status');
    if (statusEl) setStatus(statusEl, 'Your access to this terminal has been revoked. Contact support if you believe this is an error.', 'err');
  }

  function startSessionPing(token) {
    if (!token) return;
    if (sessionPingTimer) clearInterval(sessionPingTimer);
    pingSession(token);
    sessionPingTimer = setInterval(() => pingSession(loadToken()), SESSION_PING_INTERVAL_MS);
  }

  // Fires one last presence ping as the page is being torn down (tab close,
  // navigation away, or backgrounding on mobile). A regular fetch() started
  // in a pagehide/visibilitychange handler is not guaranteed to complete —
  // the browser can abort it once the page unloads. navigator.sendBeacon()
  // exists specifically for this case: the browser queues it and completes
  // it independently of the page's lifetime. sendBeacon cannot set custom
  // request headers, so token/session id go as query params instead of the
  // usual Authorization/X-Session-Id headers — the Worker's /session/ping
  // accepts either (see handleSessionPing() in worker.js).
  function pingSessionOnUnload() {
    const token = loadToken();
    if (!token || !navigator.sendBeacon) return;
    const params = new URLSearchParams({
      token:      token,
      session_id: getOrCreateSessionId(),
    });
    try {
      navigator.sendBeacon(`${WORKER_URL}/session/ping?${params.toString()}`);
    } catch {
      // best-effort — must never disrupt the page unload
    }
  }

  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') pingSessionOnUnload();
  });
  window.addEventListener('pagehide', pingSessionOnUnload);

  function saveToken(t) {
    try { sessionStorage.setItem(JWT_KEY, t); } catch {}
    try { localStorage.setItem(JWT_KEY, t); }   catch {}
  }

  function loadToken() {
    try { const t = sessionStorage.getItem(JWT_KEY); if (t) return t; } catch {}
    try { return localStorage.getItem(JWT_KEY); }                        catch {}
    return null;
  }

  // ── Modal control ──────────────────────────────────────────────────────────
  function showModal() {
    document.getElementById(MODAL_ID)?.classList.add('visible');
  }

  function hideModal() {
    document.getElementById(MODAL_ID)?.classList.remove('visible');
    // v1.7.0: closing the gate must never leave a non-activated visitor
    // standing inside the (partially-gated) terminal — see header
    // changelog. Bounce back to the Overview snapshot in that case.
    // Active users are unaffected (e.g. dismissing the renewal-reminder
    // modal, or the brief 'Activated. Loading terminal…' pause before this
    // function's own setTimeout call in activate()'s success path, by
    // which point isActive is already true).
    if (!window.GI_AUTH.isActive && typeof window.giShowOverview === 'function') {
      window.giShowOverview();
    }
  }

  // ── Activate ───────────────────────────────────────────────────────────────
  function setStatus(el, text, kind) {
    el.textContent = text;
    el.classList.remove('is-err', 'is-ok');
    if (kind) el.classList.add(kind === 'ok' ? 'is-ok' : 'is-err');
    el.setAttribute('role', kind === 'ok' ? 'status' : 'alert');
    el.setAttribute('aria-live', kind === 'ok' ? 'polite' : 'assertive');
  }

  async function activate() {
    const key     = (document.getElementById('gi-inp-key')?.value     || '').trim();
    const account = (document.getElementById('gi-inp-account')?.value  || '').trim();
    const server  = (document.getElementById('gi-inp-server')?.value   || '').trim();
    const statusEl = document.getElementById('gi-auth-status');
    const btn     = document.getElementById('gi-auth-activate');

    setStatus(statusEl, '', null);

    if (!/^[0-9A-Za-z]{4}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}$/.test(key)) {
      setStatus(statusEl, 'Key must be in XXXX-XXXX-XXXX format.', 'err'); return;
    }
    if (!account || !/^\d+$/.test(account)) {
      setStatus(statusEl, 'Account number must be numeric.', 'err'); return;
    }
    if (!server || server.length < 2) {
      setStatus(statusEl, 'Please enter your broker server name.', 'err'); return;
    }

    btn.disabled    = true;
    btn.textContent = 'Validating\u2026';

    try {
      const res  = await fetch(`${WORKER_URL}/validate`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ key, account, server }),
      });
      const data = await res.json();

      if (res.ok && data.token) {
        saveToken(data.token);
        window.GI_AUTH.isActive = true;
        startSessionPing(data.token);
        setStatus(statusEl, 'Activated. Loading terminal\u2026', 'ok');
        setTimeout(() => { hideModal(); unlockPremiumPanels(); }, 900);
      } else {
        setStatus(statusEl, data.error ||
          'Activation failed. Check your key, account number, and server name.', 'err');
      }
    } catch {
      setStatus(statusEl, 'Could not reach activation server. Check your connection.', 'err');
    } finally {
      btn.disabled    = false;
      btn.textContent = 'Activate';
    }
  }

  // ── Gate / unlock ──────────────────────────────────────────────────────────
  function applyGates() {
    if (window.GI_AUTH.isActive) return;
    PREMIUM_SECTIONS.forEach(id => {
      const el = document.getElementById(id);
      if (!el || el.querySelector('.gi-gate-overlay')) return;
      const cs = window.getComputedStyle(el);
      if (cs.position === 'static') el.style.position = 'relative';
      const ov = document.createElement('div');
      ov.className = 'gi-gate-overlay';
      ov.innerHTML =
        '<div class="gi-gate-icon"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><rect x="3" y="11" width="18" height="11" rx="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg></div>' +
        '<div class="gi-gate-msg">Premium \u2014 included with EA rental or a verified TMGM/Vantage account</div>' +
        '<button class="gi-gate-btn" onclick="window.GI_AUTH.showModal()">Activate Access</button>';
      el.appendChild(ov);
    });
  }

  function unlockPremiumPanels() {
    document.querySelectorAll('.gi-gate-overlay').forEach(el => el.remove());
  }

  // ── Init ───────────────────────────────────────────────────────────────────
  function init() {
    const style = document.createElement('style');
    style.textContent = MODAL_CSS;
    document.head.appendChild(style);

    document.body.insertAdjacentHTML('beforeend', MODAL_HTML);
    document.body.insertAdjacentHTML('beforeend', RENEW_HTML);

    document.getElementById('gi-auth-activate')
      ?.addEventListener('click', activate);

    // Close affordances (v1.6.0) — button, backdrop click, Escape. Allowed
    // in every state, including after handleRevocation() sets its "access
    // revoked" status text; closing doesn't undo the revocation, it just
    // lets the person stop looking at the modal instead of being forced to
    // reload the page.
    document.getElementById('gi-auth-close')
      ?.addEventListener('click', hideModal);
    document.getElementById(MODAL_ID)
      ?.addEventListener('click', e => { if (e.target.id === MODAL_ID) hideModal(); });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && document.getElementById(MODAL_ID)?.classList.contains('visible')) {
        hideModal();
      }
    });

    ['gi-inp-key', 'gi-inp-account', 'gi-inp-server'].forEach(id =>
      document.getElementById(id)
        ?.addEventListener('keydown', e => { if (e.key === 'Enter') activate(); })
    );

    // Auto-format key input as XXXX-XXXX-XXXX
    const keyEl = document.getElementById('gi-inp-key');
    if (keyEl) {
      keyEl.addEventListener('input', () => {
        let v = keyEl.value.replace(/[^0-9A-Za-z]/g, '').toUpperCase();
        if (v.length > 4)  v = v.slice(0, 4)  + '-' + v.slice(4);
        if (v.length > 9)  v = v.slice(0, 9)  + '-' + v.slice(9);
        if (v.length > 14) v = v.slice(0, 14);
        keyEl.value = v;
      });
    }

    // Renewal banner wiring
    document.getElementById('gi-renew-btn')
      ?.addEventListener('click', () => {
        document.getElementById('gi-renew-banner')?.classList.remove('visible');
        showModal();
      });
    document.getElementById('gi-renew-dismiss')
      ?.addEventListener('click', () => {
        document.getElementById('gi-renew-banner')?.classList.remove('visible');
      });

    // Founder/promo grant links (?grant=<jwt>) — one-click activation for
    // pre-minted tokens issued via POST /admin/grant on the license worker.
    // Bypasses the key/account/server form entirely; used for giveaways where
    // the recipient has no MT5 account running the EA.
    try {
      const params = new URLSearchParams(window.location.search);
      const grantToken = params.get('grant');
      if (grantToken && isJWTValid(grantToken)) {
        saveToken(grantToken);
        params.delete('grant');
        const qs = params.toString();
        const cleanUrl = window.location.pathname + (qs ? '?' + qs : '') + window.location.hash;
        window.history.replaceState({}, document.title, cleanUrl);
      }
    } catch {}

    const token = loadToken();
    if (isJWTValid(token)) {
      window.GI_AUTH.isActive = true;
      startSessionPing(token);
      // Show renewal banner if fewer than 7 days remain
      const daysLeft = jwtDaysRemaining(token);
      if (daysLeft < 7) {
        const daysEl = document.getElementById('gi-renew-days');
        if (daysEl) daysEl.textContent = daysLeft;
        document.getElementById('gi-renew-banner')?.classList.add('visible');
      }
    } else {
      try { sessionStorage.removeItem(JWT_KEY); } catch {}
      try { localStorage.removeItem(JWT_KEY); }   catch {}
    }
  }

  // ── Public API ─────────────────────────────────────────────────────────────
  window.GI_AUTH = {
    isActive:   false,
    showModal:  () => showModal(),
    hideModal:  () => hideModal(),
    applyGates: () => applyGates(),
    unlock:     () => unlockPremiumPanels(),
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.addEventListener('load', () => {
    setTimeout(() => {
      // v1.5.0: showModal() intentionally removed from this path — a
      // first-time visitor should land on an interactive terminal, not a
      // full-page activation gate. applyGates() alone still locks
      // PREMIUM_SECTIONS individually; each locked panel's own
      // "Activate Access" button calls window.GI_AUTH.showModal() when the
      // user actually wants to unlock it. See header changelog (v1.5.0)
      // for the two call sites where showModal() still fires automatically.
      if (!window.GI_AUTH.isActive) {
        applyGates();
      }
    }, 400);
  });

})();
