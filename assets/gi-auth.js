(function () {
  'use strict';

  const WORKER_URL = 'https://gi-license-worker.globalinvestingmarkets.workers.dev';
  const JWT_KEY    = 'gi_license_token';
  const SESSION_ID_KEY = 'gi_session_id'; 
  const MODAL_ID   = 'gi-auth-modal';

  const SESSION_PING_INTERVAL_MS = 3 * 60 * 1000; 
  let sessionPingTimer = null;

  const PREMIUM_SECTIONS = [
    'section-positioning',       
    'section-sentiment',         
    'section-cb-expectations',   
    'section-macro',             
    'narrative',                 
    'section-news',              
    'rightpanel',                
  ];

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

#gi-auth-modal.gi-auth-modal--locked #gi-auth-close { display: none; }
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

#gi-auth-tabs {
  display: flex;
  border: 1px solid var(--border, #323232);
  border-radius: 3px;
  padding: 2px;
  margin: 0 0 16px;
}
.gi-auth-tab {
  flex: 1;
  text-align: center;
  padding: 6px 0;
  font-family: inherit;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text2, #A0A0A0);
  background: none;
  border: none;
  border-radius: 2px;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
.gi-auth-tab.active { color: #fff; background: var(--blue,#4f7fff); }
.gi-auth-panel { display: none; }
.gi-auth-panel.active { display: block; }
.gi-auth-broker-cta {
  display: block;
  text-align: center;
  background: transparent;
  border: 1px solid var(--blue,#4f7fff);
  border-radius: 3px;
  color: var(--blue,#4f7fff);
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  text-decoration: none;
  padding: 11px 0;
  margin-top: 4px;
  transition: background 0.15s, color 0.15s;
}
.gi-auth-broker-cta:hover { background: var(--blue,#4f7fff); color: #fff; }
#gi-auth-newhere {
  font-size: 11px;
  text-align: center;
  margin: 16px 0 0;
  padding-top: 14px;
  border-top: 1px solid var(--border, #222);
}
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
  margin-top: 4px;
}
#gi-auth-hint summary {
  font-size: 10.5px;
  color: var(--text3,#727272);
  cursor: pointer;
  list-style: none;
}
#gi-auth-hint summary::-webkit-details-marker { display: none; }
#gi-auth-hint p {
  font-size: 10.5px;
  color: var(--text3,#727272);
  line-height: 1.6;
  margin: 8px 0 0;
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

  const MODAL_HTML = `
<div id="${MODAL_ID}" role="dialog" aria-modal="true" aria-label="Activate terminal access">
  <div id="gi-auth-box">
    <button id="gi-auth-close" type="button" aria-label="Close">&times;</button>
    <h2>ACTIVATE TERMINAL</h2>
    <p class="gi-auth-sub">
      Included with an EA license or a partner broker account &mdash; no separate terminal subscription.
    </p>

    <div id="gi-auth-tabs" role="tablist">
      <button type="button" class="gi-auth-tab active" data-tab="ea" role="tab" aria-selected="true" aria-controls="gi-auth-panel-ea">EA license</button>
      <button type="button" class="gi-auth-tab" data-tab="broker" role="tab" aria-selected="false" aria-controls="gi-auth-panel-broker">Broker account</button>
    </div>

    <div id="gi-auth-panel-ea" class="gi-auth-panel active" role="tabpanel">
      <p class="gi-auth-sub" style="margin-bottom:16px;">
        Rent or buy the
        <a href="https://www.mql5.com/en/market/product/180326" target="_blank" rel="noopener">Global Investing FX Terminal EA</a>
        on MQL5 Market, then enter the activation key from your MT5 terminal.
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

      <details id="gi-auth-hint">
        <summary>Where is my key?</summary>
        <p>
          Open MetaTrader 5 &rarr; attach the <em>Global Investing FX Terminal EA</em> to any chart.
          The activation key appears in the terminal top bar as <code>KEY:XXXX-XXXX-XXXX</code>.
          Copy the 14-character code (dashes included) and paste it above. The account number and
          server name must match the MT5 account the EA is running on.
        </p>
      </details>
    </div>

    <div id="gi-auth-panel-broker" class="gi-auth-panel" role="tabpanel">
      <p class="gi-auth-sub" style="margin-bottom:16px;">
        Hold (or open) a live account with TMGM or Vantage through our referral link. No MT5
        account or activation key required &mdash; we issue web-only access directly.
      </p>
      <a href="contact.html" class="gi-auth-broker-cta">Contact us for access &rarr;</a>
    </div>

    <p class="gi-auth-sub" id="gi-auth-newhere">
      New here? <a href="access.html">Compare both options &amp; full pricing &rarr;</a>
    </p>
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

  function getOrCreateSessionId() {
    try {
      let id = localStorage.getItem(SESSION_ID_KEY);
      if (id) return id;
      id = (crypto && crypto.randomUUID) ? crypto.randomUUID() : randomIdFallback();
      localStorage.setItem(SESSION_ID_KEY, id);
      return id;
    } catch {
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
      if (res && (res.status === 401 || res.status === 403)) handleRevocation();
    }).catch(() => {}); 
  }

  function handleRevocation() {
    if (sessionPingTimer) { clearInterval(sessionPingTimer); sessionPingTimer = null; }
    try { sessionStorage.removeItem(JWT_KEY); } catch {}
    try { localStorage.removeItem(JWT_KEY); }   catch {}
    window.GI_AUTH.isActive = false;
    window.dispatchEvent(new CustomEvent('gi-auth:revoked'));
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

  let modalDismissible = true;

  function showModal(dismissible) {
    modalDismissible = dismissible !== false;
    const modal = document.getElementById(MODAL_ID);
    if (!modal) return;
    modal.classList.toggle('gi-auth-modal--locked', !modalDismissible);
    modal.classList.add('visible');
  }

  function hideModal() {
    if (!modalDismissible && !window.GI_AUTH.isActive) return;
    document.getElementById(MODAL_ID)?.classList.remove('visible');
    if (!window.GI_AUTH.isActive && typeof window.giShowOverview === 'function') {
      window.giShowOverview();
    }
  }

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
        window.dispatchEvent(new CustomEvent('gi-auth:activated'));
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

  function init() {
    const style = document.createElement('style');
    style.textContent = MODAL_CSS;
    document.head.appendChild(style);

    document.body.insertAdjacentHTML('beforeend', MODAL_HTML);
    document.body.insertAdjacentHTML('beforeend', RENEW_HTML);

    document.getElementById('gi-auth-activate')
      ?.addEventListener('click', activate);

    document.querySelectorAll('.gi-auth-tab').forEach(tabBtn => {
      tabBtn.addEventListener('click', () => {
        document.querySelectorAll('.gi-auth-tab').forEach(t => {
          t.classList.remove('active');
          t.setAttribute('aria-selected', 'false');
        });
        tabBtn.classList.add('active');
        tabBtn.setAttribute('aria-selected', 'true');
        document.querySelectorAll('.gi-auth-panel').forEach(p => p.classList.remove('active'));
        document.getElementById(`gi-auth-panel-${tabBtn.dataset.tab}`)?.classList.add('active');
      });
    });

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

    document.getElementById('gi-renew-btn')
      ?.addEventListener('click', () => {
        document.getElementById('gi-renew-banner')?.classList.remove('visible');
        showModal();
      });
    document.getElementById('gi-renew-dismiss')
      ?.addEventListener('click', () => {
        document.getElementById('gi-renew-banner')?.classList.remove('visible');
      });

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

  window.GI_AUTH = {
    isActive:   false,
    showModal:  (dismissible) => showModal(dismissible),
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
      if (!window.GI_AUTH.isActive) {
        applyGates();
      }
    }, 400);
  });

})();
