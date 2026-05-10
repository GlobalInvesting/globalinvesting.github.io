// ═══════════════════════════════════════════════════════════════════
// INLINE PANEL SYSTEM  v1.0.0
// File: assets/inline-panel.js
// Loaded AFTER all modal scripts (see index.html)
//
// Intercepts modal open calls and renders content inline inside the
// split-layout center columns instead of floating overlays:
//
//   LEFT center  (#split-upper):  Carry Trade · Heatmap
//   RIGHT center (#split-lower):  Correlations · CB Rates · COT
// ═══════════════════════════════════════════════════════════════════

(function () {
  'use strict';

  const LS_KEY  = 'gi_split_layout';
  const IP_ATTR = 'data-inline-panel';

  // ── Ensure split-layout is active ─────────────────────────────────
  function _ensureSplit() {
    var main   = document.getElementById('main');
    var upper  = document.getElementById('split-upper');
    var lower  = document.getElementById('split-lower');
    if (!main || !upper || !lower) return null;

    if (!main.classList.contains('split-layout')) {
      var btn    = document.getElementById('split-layout-btn');
      var handle = document.getElementById('split-drag-handle');
      var alerts = document.getElementById('section-macro');

      main.classList.add('split-layout');
      if (btn)    { btn.classList.remove('active'); btn.setAttribute('aria-pressed','true'); }
      if (handle) handle.style.display = '';
      upper.style.width = '55%';
      upper.style.flex  = 'none';
      if (alerts && alerts.parentNode !== upper) upper.appendChild(alerts);

      try { localStorage.setItem(LS_KEY, JSON.stringify({ active: true, leftPct: 55 })); } catch(e) {}
    }
    return { upper: upper, lower: lower };
  }

  // ── Core: inject content into target panel ─────────────────────────
  function showInlinePanel(target, renderFn, title, onClose) {
    if (!target) return;

    // Remove any existing inline panel in this target
    var existing = target.querySelector('[' + IP_ATTR + ']');
    if (existing) {
      // Trigger its onClose cleanup if needed before replacing
      existing.remove();
    }

    var wrap = document.createElement('div');
    wrap.setAttribute(IP_ATTR, '1');
    wrap.style.cssText = 'position:relative;display:flex;flex-direction:column;height:100%;min-height:0;background:var(--bg);overflow:hidden;';

    // Header bar
    var hd = document.createElement('div');
    hd.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:5px 10px 4px;border-bottom:1px solid var(--border2);flex-shrink:0;background:var(--bg2);';

    var titleEl = document.createElement('span');
    titleEl.style.cssText = 'font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text3);font-family:var(--font-mono);';
    titleEl.textContent = title;

    var closeBtn = document.createElement('button');
    closeBtn.textContent = '✕';
    closeBtn.setAttribute('aria-label', 'Close panel');
    closeBtn.style.cssText = 'background:none;border:none;color:var(--text3);cursor:pointer;font-size:11px;line-height:1;padding:2px 5px;border-radius:3px;';
    closeBtn.addEventListener('mouseenter', function() { closeBtn.style.color = 'var(--text)'; });
    closeBtn.addEventListener('mouseleave', function() { closeBtn.style.color = 'var(--text3)'; });
    closeBtn.addEventListener('click', function() {
      wrap.remove();
      if (typeof onClose === 'function') onClose();
    });

    hd.appendChild(titleEl);
    hd.appendChild(closeBtn);

    // Content body
    var body = document.createElement('div');
    body.style.cssText = 'flex:1;min-height:0;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border2) transparent;';

    wrap.appendChild(hd);
    wrap.appendChild(body);

    // Prepend so it appears at top of target
    target.prepend(wrap);
    target.scrollTop = 0;

    // Render
    Promise.resolve().then(function() {
      try { renderFn(body); }
      catch(e) {
        body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Error loading panel.</div>';
        console.warn('[InlinePanel]', e);
      }
    });
  }

  // ── Helper: transplant a modal's inner element into an inline body ──
  // Resets positioning styles so it flows naturally in the panel.
  function _transplant(body, bdId, modalId, closeId, extraStyles) {
    var bd = document.getElementById(bdId);
    var modal = document.getElementById(modalId);
    if (!bd || !modal) return false;

    // Hide the fixed-overlay backdrop
    bd.style.cssText = 'position:relative;inset:auto;z-index:1;padding:0;display:block;animation:none;background:none;';

    // Strip modal chrome styles, apply inline-friendly overrides
    var base = 'width:100%;max-width:none;border-radius:0;border:none;box-shadow:none;animation:none;background:var(--bg);height:auto;max-height:none;';
    modal.style.cssText = base + (extraStyles || '');

    // Hide the modal's own close button
    if (closeId) {
      var origClose = document.getElementById(closeId);
      if (origClose) origClose.style.display = 'none';
    }

    body.style.overflow = 'hidden';
    body.appendChild(bd);
    return true;
  }

  // ── Helper: restore modal back to body after inline panel closes ────
  function _restore(bdId, modalId, closeId, originalStyles) {
    var bd = document.getElementById(bdId);
    var modal = document.getElementById(modalId);
    if (!bd) return;

    // Move back to body
    if (bd.parentNode !== document.body) document.body.appendChild(bd);

    // Restore overlay backdrop
    bd.style.cssText = originalStyles || '';

    if (modal) modal.style.cssText = '';

    if (closeId) {
      var origClose = document.getElementById(closeId);
      if (origClose) origClose.style.display = '';
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: Real Carry Modal → #split-upper (LEFT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenRCM = window.openRealCarryModal;

  window.openRealCarryModal = async function(longCcy, shortCcy) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenRCM && _origOpenRCM(longCcy, shortCcy); return; }

    var target = panels.upper;

    // Build the title
    var pairLabel = (longCcy && shortCcy) ? (longCcy + '/' + shortCcy) : 'G8';

    showInlinePanel(target, async function(body) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Loading real carry data…</div>';

      // Call original — it populates #rcm-bd
      if (_origOpenRCM) {
        var bd = document.getElementById('rcm-bd');
        if (bd) bd.style.display = 'none'; // suppress fixed overlay while loading
        await _origOpenRCM(longCcy, shortCcy);
        if (bd) bd.style.display = 'none'; // keep suppressed
      }

      body.innerHTML = '';
      var ok = _transplant(body, 'rcm-bd', 'rcm-modal', 'rcm-close',
        'display:flex;flex-direction:column;overflow:hidden;');

      if (!ok) body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Real carry data unavailable.</div>';
    }, 'Real Rate Carry · ' + pairLabel, function() {
      _restore('rcm-bd', 'rcm-modal', 'rcm-close',
        'position:fixed;inset:0;z-index:9200;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;padding:16px;');
    });
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: Heatmap Modal → #split-upper (LEFT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenHM = window.openHeatmapModal;

  window.openHeatmapModal = function(ccy, strengths, rtCache) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenHM && _origOpenHM(ccy, strengths, rtCache); return; }

    var target = panels.upper;

    showInlinePanel(target, function(body) {
      // Call original — populates #hm-bd
      if (_origOpenHM) {
        var bd = document.getElementById('hm-bd');
        if (bd) bd.style.display = 'none';
        _origOpenHM(ccy, strengths, rtCache);
        if (bd) bd.style.display = 'none';
      }

      var ok = _transplant(body, 'hm-bd', 'hm-modal', 'hm-close',
        'display:flex;flex-direction:column;overflow:hidden;');

      if (!ok) body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Heatmap data unavailable.</div>';
    }, 'Currency Strength · ' + (ccy || ''), function() {
      // Destroy and re-create on next open (heatmap has a Lightweight Charts instance that must be remounted)
      var hmBd = document.getElementById('hm-bd');
      if (hmBd) hmBd.remove();
      // The next call to openHeatmapModal will rebuild via buildModal()
    });
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: Correlation Modal → #split-lower (RIGHT)
  // openCorrModal creates a fresh #cm-bd each call (closeCorrModal removes it)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenCorr = window.openCorrModal;

  window.openCorrModal = function(corrObj) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenCorr && _origOpenCorr(corrObj); return; }

    var target = panels.lower;
    var a = corrObj ? corrObj.a : '';
    var b = corrObj ? corrObj.b : '';

    showInlinePanel(target, function(body) {
      // openCorrModal does document.body.appendChild → fresh element each time
      _origOpenCorr && _origOpenCorr(corrObj);

      var cmBd    = document.getElementById('cm-bd');
      var cmModal = document.getElementById('cm-modal');
      if (!cmBd || !cmModal) {
        body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Correlation data unavailable.</div>';
        return;
      }

      // Neutralise the backdrop
      cmBd.style.cssText = 'position:relative;inset:auto;z-index:1;padding:0;display:block;animation:none;background:none;';
      // Neutralise modal chrome
      cmModal.style.cssText = 'width:100%;max-width:none;border-radius:0;border:none;box-shadow:none;animation:none;background:var(--bg);height:auto;max-height:none;position:static;bottom:auto;left:auto;right:auto;';

      var origClose = document.getElementById('cm-close');
      if (origClose) origClose.style.display = 'none';

      body.appendChild(cmBd);
    }, 'Correlations · ' + a + ' / ' + b, function() {
      // closeCorrModal removes #cm-bd from DOM — just call it
      if (typeof window.closeCorrModal === 'function') window.closeCorrModal();
    });
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: CB Rates Modal → #split-lower (RIGHT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenCBR = window.openCBRatesModal;

  window.openCBRatesModal = async function(ccy, obs, bankInfo, meetingData) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenCBR && _origOpenCBR(ccy, obs, bankInfo, meetingData); return; }

    var target = panels.lower;

    showInlinePanel(target, async function(body) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Loading rate data…</div>';

      if (_origOpenCBR) {
        var bd = document.getElementById('cbr-bd');
        if (bd) bd.style.display = 'none';
        await _origOpenCBR(ccy, obs, bankInfo, meetingData);
        if (bd) bd.style.display = 'none';
      }

      body.innerHTML = '';
      var ok = _transplant(body, 'cbr-bd', 'cbr-modal', 'cbr-m-close', '');

      if (!ok) body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">CB rate data unavailable.</div>';
    }, 'CB Rates · ' + (ccy || ''), function() {
      _restore('cbr-bd', 'cbr-modal', 'cbr-m-close',
        'position:fixed;inset:0;z-index:9100;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;padding:12px;');
    });
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: COT Modal → #split-lower (RIGHT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenCOT = window.openCOTModal;

  window.openCOTModal = function(ccy, data) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenCOT && _origOpenCOT(ccy, data); return; }

    var target = panels.lower;

    showInlinePanel(target, function(body) {
      if (_origOpenCOT) {
        var bd = document.getElementById('cot-bd');
        if (bd) bd.style.display = 'none';
        _origOpenCOT(ccy, data);
        if (bd) bd.style.display = 'none';
      }

      var ok = _transplant(body, 'cot-bd', 'cot-modal', 'cot-m-close', '');

      if (!ok) body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">COT data unavailable.</div>';
    }, 'COT Positioning · ' + (ccy || ''), function() {
      _restore('cot-bd', 'cot-modal', 'cot-m-close',
        'position:fixed;inset:0;z-index:9100;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;padding:12px;');
    });
  };

  // ── Expose internals ────────────────────────────────────────────────
  window._showInlinePanel    = showInlinePanel;
  window._ensureInlineSplit  = _ensureSplit;

})();
