/**
 * GlobalInvesting FX Terminal — License Auth Module  v1.0.0
 * assets/gi-auth.js  — include BEFORE dashboard.js in index.html
 *
 * Flow:
 *   1. On load, check localStorage/sessionStorage for a valid JWT
 *   2. If none, show activation modal after panels have rendered (~400ms)
 *   3. POST key+account+server to the Cloudflare Worker → receive JWT
 *   4. Store JWT; remove gate overlays; expose window.GI_AUTH
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
  const MODAL_ID   = 'gi-auth-modal';

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
  background: var(--bg2, #141414);
  border: 1px solid var(--border, #323232);
  border-top: 2px solid #D95000;
  padding: 32px 36px 28px;
  width: 460px;
  max-width: 92vw;
  box-shadow: 0 24px 64px rgba(0,0,0,0.7);
}
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
.gi-auth-sub a { color: #D95000; text-decoration: none; }
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
#gi-auth-box input:focus { border-color: #D95000; }
#gi-auth-box input::placeholder { color: var(--border2, #404040); }
#gi-auth-activate {
  width: 100%;
  background: #D95000;
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
#gi-auth-activate:hover { background: #F06010; }
#gi-auth-activate:disabled { background: #444; cursor: default; }
#gi-auth-err {
  font-size: 11px;
  color: #FF3C3C;
  margin-top: 10px;
  min-height: 16px;
  text-align: center;
}
#gi-auth-ok {
  font-size: 11px;
  color: #00D455;
  margin-top: 10px;
  min-height: 16px;
  text-align: center;
}
#gi-auth-hint {
  font-size: 10px;
  color: #505050;
  margin-top: 18px;
  line-height: 1.6;
  border-top: 1px solid var(--border, #222);
  padding-top: 14px;
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
.gi-gate-icon { font-size: 22px; color: #D95000; opacity: 0.65; }
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
  color: #D95000;
  background: transparent;
  border: 1px solid #D95000;
  padding: 7px 18px;
  cursor: pointer;
  margin-top: 4px;
  transition: background 0.15s, color 0.15s;
}
.gi-gate-btn:hover { background: #D95000; color: #fff; }
#gi-renew-banner {
  display: none;
  position: fixed;
  bottom: 16px;
  right: 20px;
  z-index: 9999;
  background: var(--bg2, #141414);
  border: 1px solid #D95000;
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
#gi-renew-banner strong { color: #D95000; }
#gi-renew-btn {
  background: transparent;
  border: 1px solid #D95000;
  color: #D95000;
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
#gi-renew-btn:hover { background: #D95000; color: #fff; }
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
`;

  // ── Modal HTML ─────────────────────────────────────────────────────────────
  const MODAL_HTML = `
<div id="${MODAL_ID}" role="dialog" aria-modal="true" aria-label="Activate terminal access">
  <div id="gi-auth-box">
    <h2>GI&gt;&nbsp; ACTIVATE TERMINAL</h2>
    <p class="gi-auth-sub">
      Full access is included with the
      <a href="https://www.mql5.com/en/market/product/180326" target="_blank" rel="noopener">
        Global Investing FX Terminal EA
      </a>
      on MQL5 Market. Enter the activation key shown in your MT5 terminal to unlock all panels.
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
    <div id="gi-auth-err" role="alert" aria-live="assertive"></div>
    <div id="gi-auth-ok"  role="status" aria-live="polite"></div>

    <div id="gi-auth-hint">
      <strong style="color:#606060">Where is my key?</strong><br>
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
  }

  // ── Activate ───────────────────────────────────────────────────────────────
  async function activate() {
    const key     = (document.getElementById('gi-inp-key')?.value     || '').trim();
    const account = (document.getElementById('gi-inp-account')?.value  || '').trim();
    const server  = (document.getElementById('gi-inp-server')?.value   || '').trim();
    const errEl   = document.getElementById('gi-auth-err');
    const okEl    = document.getElementById('gi-auth-ok');
    const btn     = document.getElementById('gi-auth-activate');

    errEl.textContent = '';
    okEl.textContent  = '';

    if (!/^[0-9A-Za-z]{4}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}$/.test(key)) {
      errEl.textContent = 'Key must be in XXXX-XXXX-XXXX format.'; return;
    }
    if (!account || !/^\d+$/.test(account)) {
      errEl.textContent = 'Account number must be numeric.'; return;
    }
    if (!server || server.length < 2) {
      errEl.textContent = 'Please enter your broker server name.'; return;
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
        okEl.textContent = 'Activated. Loading terminal\u2026';
        setTimeout(() => { hideModal(); unlockPremiumPanels(); }, 900);
      } else {
        errEl.textContent = data.error ||
          'Activation failed. Check your key, account number, and server name.';
      }
    } catch {
      errEl.textContent = 'Could not reach activation server. Check your connection.';
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
        '<div class="gi-gate-icon">&#128274;</div>' +
        '<div class="gi-gate-msg">Premium \u2014 included with EA purchase<br>Activate with your MT5 license key</div>' +
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

    const token = loadToken();
    if (isJWTValid(token)) {
      window.GI_AUTH.isActive = true;
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
      if (!window.GI_AUTH.isActive) {
        showModal();
        applyGates();
      }
    }, 400);
  });

})();
