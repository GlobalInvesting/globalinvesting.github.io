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
      window.GI_AUTH.showModal(false);
    }
  }

  window.addEventListener('gi-auth:activated', unlockTerminal);
  window.addEventListener('gi-auth:revoked', gateIfInactive);

  function init() {
    gateIfInactive();
    let attempts = 0;
    const t = setInterval(() => {
      attempts++;
      maybeAnnounceTerminalEntry();
      if (window._giTerminalShown || attempts > 1800) clearInterval(t); 
    }, 700);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
