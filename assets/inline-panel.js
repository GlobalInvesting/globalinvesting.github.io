// ═══════════════════════════════════════════════════════════════════
// INLINE PANEL SYSTEM  v1.2.0
// File: assets/inline-panel.js
// Loaded AFTER all modal scripts (see index2.html)
//
// Intercepts modal open calls and renders content inline inside the
// split-layout center columns instead of floating overlays:
//
//   LEFT center  (#split-upper):  Carry Trade · Heatmap
//   RIGHT center (#split-lower):  Correlations · CB Rates · COT
//
// v1.2.0: Synchronous transplant strategy — pre-hide bd BEFORE calling
//         orig so display:'flex' is invisible, then steal the element
//         in the same JS tick. Zero flicker.
// ═══════════════════════════════════════════════════════════════════

(function () {
  'use strict';

  var LS_KEY  = 'gi_split_layout';
  var IP_ATTR = 'data-inline-panel';

  // ── Ensure split-layout is active, return {upper, lower} ──────────
  function _ensureSplit() {
    var main  = document.getElementById('main');
    var upper = document.getElementById('split-upper');
    var lower = document.getElementById('split-lower');
    if (!main || !upper || !lower) return null;

    if (!main.classList.contains('split-layout')) {
      var btn    = document.getElementById('split-layout-btn');
      var handle = document.getElementById('split-drag-handle');
      var alerts = document.getElementById('section-macro');

      main.classList.add('split-layout');
      if (btn)    { btn.classList.add('active'); btn.setAttribute('aria-pressed','true'); }
      if (handle) handle.style.display = '';
      upper.style.width = '55%';
      upper.style.flex  = 'none';
      if (alerts && alerts.parentNode !== upper) upper.appendChild(alerts);

      try { localStorage.setItem(LS_KEY, JSON.stringify({ active: true, leftPct: 55 })); } catch(e) {}
    }
    return { upper: upper, lower: lower };
  }

  // ── Create inline shell: header bar + scrollable body ─────────────
  function _makeShell(target, title, onClose) {
    var old = target.querySelector('[' + IP_ATTR + ']');
    if (old) old.remove();

    var wrap = document.createElement('div');
    wrap.setAttribute(IP_ATTR, '1');
    wrap.style.cssText = 'position:relative;display:flex;flex-direction:column;height:100%;min-height:0;background:var(--bg);overflow:hidden;';

    var hd = document.createElement('div');
    hd.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:5px 10px 4px;border-bottom:1px solid var(--border2);flex-shrink:0;background:var(--bg2);';

    var titleEl = document.createElement('span');
    titleEl.style.cssText = 'font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text3);font-family:var(--font-mono);';
    titleEl.textContent = title;

    var closeBtn = document.createElement('button');
    closeBtn.textContent = '✕';
    closeBtn.setAttribute('aria-label', 'Close panel');
    closeBtn.style.cssText = 'background:none;border:none;color:var(--text3);cursor:pointer;font-size:11px;line-height:1;padding:2px 5px;border-radius:3px;transition:color .1s;';
    closeBtn.onmouseenter = function() { closeBtn.style.color = 'var(--text)'; };
    closeBtn.onmouseleave = function() { closeBtn.style.color = 'var(--text3)'; };
    closeBtn.onclick = function() {
      wrap.remove();
      if (typeof onClose === 'function') onClose();
    };

    hd.appendChild(titleEl);
    hd.appendChild(closeBtn);

    var body = document.createElement('div');
    body.style.cssText = 'flex:1;min-height:0;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border2) transparent;display:flex;flex-direction:column;';

    wrap.appendChild(hd);
    wrap.appendChild(body);
    target.prepend(wrap);
    target.scrollTop = 0;

    return body;
  }

  // ── Synchronously transplant modal into inline body ────────────────
  // Strips backdrop and modal chrome so it flows naturally in the panel.
  function _transplant(body, bdId, modalId, closeId, modalExtraCSS) {
    var bd    = document.getElementById(bdId);
    var modal = document.getElementById(modalId);
    if (!bd || !modal) return false;

    // Keep bd in DOM so modal's internal JS still finds it — just make it inert
    bd.style.cssText = 'display:block!important;position:static!important;background:none!important;padding:0!important;animation:none!important;z-index:auto!important;';

    // Strip modal chrome
    var base = 'width:100%!important;max-width:none!important;height:100%!important;max-height:none!important;border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;background:var(--bg)!important;position:static!important;';
    modal.style.cssText = base + (modalExtraCSS || '');

    // Hide the modal's own close button
    if (closeId) {
      var oc = document.getElementById(closeId);
      if (oc) oc.style.display = 'none';
    }

    body.appendChild(bd);
    return true;
  }

  // ── Restore modal back to document.body on panel close ────────────
  function _restore(bdId, closeId, bdOrigCSS, modalId) {
    var bd = document.getElementById(bdId);
    if (!bd) return;
    if (bd.parentNode !== document.body) document.body.appendChild(bd);
    bd.style.cssText = bdOrigCSS || 'display:none;';
    if (closeId) {
      var oc = document.getElementById(closeId);
      if (oc) oc.style.display = '';
    }
    if (modalId) {
      var modal = document.getElementById(modalId);
      if (modal) modal.style.cssText = '';
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: Real Carry Modal → #split-upper (LEFT)
  //
  // openRealCarryModal sets bd.style.display='flex' synchronously,
  // then optionally awaits _rcmFetchData(). Strategy:
  //   1. Pre-hide bd BEFORE calling orig → flex is never visible
  //   2. Call orig (DOM builds, fetch starts in background)
  //   3. SYNCHRONOUSLY transplant — same JS tick, no paint yet
  //   4. orig's async _rcmRender() updates #rcm-body which is now
  //      inside our panel → content appears naturally, zero flicker
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenRCM = window.openRealCarryModal;

  window.openRealCarryModal = function(longCcy, shortCcy) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenRCM && _origOpenRCM(longCcy, shortCcy); return; }

    var label = (longCcy && shortCcy) ? longCcy + '/' + shortCcy : 'G8';
    var body  = _makeShell(panels.upper, 'Real Rate Carry · ' + label, function() {
      _restore('rcm-bd', 'rcm-close',
        'position:fixed;inset:0;z-index:9200;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;padding:16px;',
        'rcm-modal');
    });

    // 1. Pre-hide
    var bdPre = document.getElementById('rcm-bd');
    if (bdPre) bdPre.style.display = 'none';

    // 2. Call orig (sync part: builds DOM if needed, sets display:flex — but we pre-hid it)
    _origOpenRCM && _origOpenRCM(longCcy, shortCcy);

    // 3. Synchronous transplant
    if (!_transplant(body, 'rcm-bd', 'rcm-modal', 'rcm-close',
        'display:flex!important;flex-direction:column!important;overflow:hidden!important;')) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Carry data unavailable.</div>';
    }
    document.body.style.overflow = '';
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: Heatmap Modal → #split-upper (LEFT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenHM = window.openHeatmapModal;

  window.openHeatmapModal = function(ccy, strengths, rtCache) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenHM && _origOpenHM(ccy, strengths, rtCache); return; }

    var body = _makeShell(panels.upper, 'Currency Strength · ' + (ccy || ''), function() {
      // Destroy chart instance so it remounts correctly on next open
      var hmBd = document.getElementById('hm-bd');
      if (hmBd) hmBd.remove();
    });

    var bdPre = document.getElementById('hm-bd');
    if (bdPre) bdPre.style.display = 'none';

    _origOpenHM && _origOpenHM(ccy, strengths, rtCache);

    if (!_transplant(body, 'hm-bd', 'hm-modal', 'hm-close',
        'display:flex!important;flex-direction:column!important;overflow:hidden!important;')) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Heatmap unavailable.</div>';
    }
    document.body.style.overflow = '';
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: Correlation Modal → #split-lower (RIGHT)
  // openCorrModal appends a fresh #cm-bd to document.body each call.
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenCorr = window.openCorrModal;

  window.openCorrModal = function(corrObj) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenCorr && _origOpenCorr(corrObj); return; }

    var a = corrObj ? corrObj.a : '';
    var b = corrObj ? corrObj.b : '';
    var body = _makeShell(panels.lower, 'Correlations · ' + a + ' / ' + b, function() {
      if (typeof window.closeCorrModal === 'function') window.closeCorrModal();
    });

    var bdPre = document.getElementById('cm-bd');
    if (bdPre) bdPre.style.display = 'none';

    _origOpenCorr && _origOpenCorr(corrObj);

    var cmBd    = document.getElementById('cm-bd');
    var cmModal = document.getElementById('cm-modal');
    if (!cmBd || !cmModal) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Correlation data unavailable.</div>';
      return;
    }

    cmBd.style.cssText    = 'display:block!important;position:static!important;background:none!important;padding:0!important;animation:none!important;z-index:auto!important;';
    cmModal.style.cssText = 'width:100%!important;max-width:none!important;border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;background:var(--bg)!important;height:auto!important;max-height:none!important;position:static!important;';

    var oc = document.getElementById('cm-close');
    if (oc) oc.style.display = 'none';

    body.appendChild(cmBd);
    document.body.style.overflow = '';
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: CB Rates Modal → #split-lower (RIGHT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenCBR = window.openCBRatesModal;

  window.openCBRatesModal = function(ccy, obs, bankInfo, meetingData) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenCBR && _origOpenCBR(ccy, obs, bankInfo, meetingData); return; }

    var body = _makeShell(panels.lower, 'CB Rates · ' + (ccy || ''), function() {
      _restore('cbr-bd', 'cbr-m-close',
        'position:fixed;inset:0;z-index:9100;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;padding:12px;',
        'cbr-modal');
    });

    var bdPre = document.getElementById('cbr-bd');
    if (bdPre) bdPre.style.display = 'none';

    _origOpenCBR && _origOpenCBR(ccy, obs, bankInfo, meetingData);

    if (!_transplant(body, 'cbr-bd', 'cbr-modal', 'cbr-m-close', '')) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">CB rate data unavailable.</div>';
    }
    document.body.style.overflow = '';
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: COT Modal → #split-lower (RIGHT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenCOT = window.openCOTModal;

  window.openCOTModal = function(ccy, data) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenCOT && _origOpenCOT(ccy, data); return; }

    var body = _makeShell(panels.lower, 'COT Positioning · ' + (ccy || ''), function() {
      _restore('cot-bd', 'cot-m-close',
        'position:fixed;inset:0;z-index:9100;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;padding:12px;',
        'cot-modal');
    });

    var bdPre = document.getElementById('cot-bd');
    if (bdPre) bdPre.style.display = 'none';

    _origOpenCOT && _origOpenCOT(ccy, data);

    if (!_transplant(body, 'cot-bd', 'cot-modal', 'cot-m-close', '')) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">COT data unavailable.</div>';
    }
    document.body.style.overflow = '';
  };

  // ── Expose internals ────────────────────────────────────────────────
  window._showInlinePanel   = _makeShell;
  window._ensureInlineSplit = _ensureSplit;

})();
