/**
 * Global Investing FX Terminal — First-Visit Welcome Tour
 * v7.73.0 — Mobile fix + robustness improvements
 *
 * Fixes vs v7.71.0:
 *   BUG-1 (CRITICAL): shouldShow() returned false when localStorage is blocked
 *          by browser Tracking Prevention -> tour never showed after hard reload.
 *          Fix: treat blocked storage as "not seen" (show tour, skip markDone).
 *
 *   BUG-2 (CRITICAL): positionPopover() setTimeout(120ms) fired before
 *          scrollIntoView({ behavior:'smooth' }) completed (~300-500ms on mobile).
 *          getBoundingClientRect() returned mid-scroll or pre-scroll coords -> all
 *          space[] values wrong -> popover fell back to bottom-center or off-screen.
 *          Fix: use 'instant' scroll (no animation) + 0ms timeout; keeps scroll
 *          context correct. On mobile the page scroll is the UX, not the indicator.
 *
 *   BUG-3 (CRITICAL): On mobile (≤900px) layout becomes flex-column.
 *          #rightpanel (contains #risk-regime) renders BELOW #main, thousands of px
 *          from top. positionPopover() tried left/right with minW=348px on a 390px
 *          viewport -> always fell through to bottom-center, correct in principle
 *          but the popover was rendered at the mid-scroll or wrong position.
 *          Fix: on mobile, skip side-positioning entirely; always use
 *          bottom-center with a safe bottom offset that clears browser chrome.
 *
 *   BUG-4: overlayEl.offsetHeight read BEFORE element paint on first renderStep()
 *          -> ph=0 -> positioning math wrong for all non-null targets.
 *          Fix: read offsetHeight inside the deferred callback.
 *
 *   BUG-5: localStorage blocked silently -> markDone() never writes -> tour
 *          re-shows on every page load when Tracking Prevention is active.
 *          Fix: in-memory flag as fallback when storage is unavailable.
 *
 * Storage key: 'gi_welcome_done' (unchanged)
 */

(function () {
  'use strict';

  var STORAGE_KEY   = 'gi_welcome_done';
  var DELAY_MS      = 2800;
  var AUTO_CLOSE_S  = 90;
  var POPOVER_W     = 320;
  var POPOVER_GAP   = 12;
  var ARROW_SIZE    = 8;
  var MOBILE_BP     = 900;   // matches dashboard.css breakpoint

  var REGIME_CONTEXT = {
    'RISK-ON':  'In this environment, AUD, NZD, and higher-beta pairs tend to attract flows as appetite for yield increases.',
    'MIXED':    'Mixed signals: some risk appetite but with offsetting stress factors. Pair selection warrants more caution than a clean RISK-ON.',
    'CAUTION':  'Elevated stress in 2-3 factors. USD, JPY, and CHF are seeing defensive demand. Avoid high-beta longs into this environment.',
    'RISK-OFF': 'In RISK-OFF conditions, JPY, CHF, and USD attract safe-haven flows. High-beta pairs (AUD, NZD) are typically under pressure.',
  };

  var STEPS = [
    {
      target:  null,
      side:    'bottom',
      title:   'Institutional-grade FX data, in one place.',
      body:    'Most retail traders operate blind to what institutions are doing. This terminal changes that: live G8 rates, CFTC positioning, central bank policy, yield curves, cross-asset risk — and an AI that reads it all and tells you what it means right now.',
      dynamic: null,
    },
    {
      target:  'narrative',
      side:    'bottom',
      title:   'AI Market Narrative',
      body:    'At each major session transition the terminal reads COT positioning, rate differentials, yield spreads, and cross-asset risk to produce a single narrative: what the macro environment looks like right now, in plain language.',
      dynamic: null,
    },
    {
      target:  'risk-regime',
      side:    'left',
      title:   'Macro Regime',
      body:    null,
      dynamic: 'regime',
    },
    {
      target:  'section-crossasset',
      side:    'right',
      title:   'Cross-Asset Risk Monitor',
      body:    'VIX, MOVE, DXY, Gold, and S&P500 — the five inputs that determine the regime. When two or more are in stress, the terminal shifts to CAUTION or RISK-OFF. This panel shows you why, updated every 15 minutes.',
      dynamic: null,
    },
    {
      target:  'section-positioning',
      side:    'top',
      title:   'CFTC COT Positioning',
      body:    'The Commitment of Traders report shows what institutional speculators — hedge funds and large traders — are actually holding. Combined with rate differentials, it produces the directional bias per pair: the core multi-factor framework behind every macro trade setup.',
      dynamic: null,
    },
    {
      target:  'sig-notif-btn',
      side:    'top',
      title:   'Stay ahead — enable signal alerts',
      body:    'The terminal publishes AI-generated signals when the regime shifts or a new high-conviction setup appears. Enable browser notifications so it reaches you even when this tab is in the background — no account, no email required.',
      dynamic: null,
      lastCta: true,
    },
  ];

  var currentStep   = 0;
  var overlayEl     = null;
  var arrowEl       = null;
  var countdownBar  = null;
  var countdownInt  = null;
  var autoTimer     = null;
  var secondsLeft   = AUTO_CLOSE_S;

  // BUG-5 fix: in-memory fallback when localStorage is blocked
  var _memDone = false;

  function storageAvailable() {
    try {
      var k = '__gi_test__';
      localStorage.setItem(k, '1');
      localStorage.removeItem(k);
      return true;
    } catch (e) {
      return false;
    }
  }

  function shouldShow() {
    if (_memDone) return false;
    try {
      return !localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      // Tracking Prevention or private mode — treat as unseen so tour shows
      return true;
    }
  }

  function markDone() {
    _memDone = true;
    try { localStorage.setItem(STORAGE_KEY, '1'); } catch (e) { /* blocked — in-memory flag covers this */ }
  }

  // BUG-3 fix: helper to detect mobile layout
  function isMobile() {
    return window.innerWidth <= MOBILE_BP;
  }

  function getRegimeValue() {
    var el = document.getElementById('risk-regime');
    return el ? (el.textContent || '').trim().toUpperCase() : '';
  }

  function buildRegimeBody() {
    var regime  = getRegimeValue();
    var context = REGIME_CONTEXT[regime];
    if (!regime || regime === '--' || !context) {
      return 'The Regime score synthesises VIX, MOVE, yield spreads, and gold to classify the market as RISK-ON, CAUTION, or RISK-OFF — driving which currency pairs the current environment favours.';
    }
    return 'Right now the terminal is reading <strong>' + regime + '</strong>. ' + context;
  }

  function getStepBody(step) {
    if (step.dynamic === 'regime') return buildRegimeBody();
    return step.body;
  }

  function highlight(targetId) {
    document.querySelectorAll('.gi-tour-highlight').forEach(function (el) {
      el.classList.remove('gi-tour-highlight');
    });
    if (!targetId) return null;
    var el = document.getElementById(targetId);
    if (el) el.classList.add('gi-tour-highlight');
    return el || null;
  }

  function applyBottomCenter() {
    if (!overlayEl) return;
    // BUG-3 fix: on mobile, use a larger bottom offset to clear browser chrome (address bar ~56px)
    var bottomOffset = isMobile() ? '72px' : '24px';
    overlayEl.style.position  = 'fixed';
    overlayEl.style.bottom    = bottomOffset;
    overlayEl.style.left      = '50%';
    overlayEl.style.transform = 'translateX(-50%)';
    overlayEl.style.top       = '';
    overlayEl.style.right     = '';
    if (arrowEl) arrowEl.style.display = 'none';
  }

  function setArrow(pointingFrom, x, y) {
    if (!arrowEl) return;
    arrowEl.style.display = 'block';
    arrowEl.style.left    = x + 'px';
    arrowEl.style.top     = y + 'px';
    arrowEl.style.border  = ARROW_SIZE + 'px solid transparent';
    arrowEl.style.borderRightColor  = 'transparent';
    arrowEl.style.borderLeftColor   = 'transparent';
    arrowEl.style.borderTopColor    = 'transparent';
    arrowEl.style.borderBottomColor = 'transparent';
    var col = 'var(--border)';
    if (pointingFrom === 'left')   arrowEl.style.borderRightColor  = col;
    if (pointingFrom === 'right')  arrowEl.style.borderLeftColor   = col;
    if (pointingFrom === 'top')    arrowEl.style.borderBottomColor = col;
    if (pointingFrom === 'bottom') arrowEl.style.borderTopColor    = col;
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function positionPopover(targetEl, preferredSide) {
    if (!overlayEl) return;

    if (!targetEl) {
      applyBottomCenter();
      return;
    }

    // BUG-3 fix: on mobile, always bottom-center — no room for side positioning
    if (isMobile()) {
      // BUG-2 fix: use 'instant' so getBoundingClientRect is accurate immediately
      try { targetEl.scrollIntoView({ behavior: 'instant', block: 'center', inline: 'nearest' }); } catch (e) {}
      applyBottomCenter();
      return;
    }

    // Desktop: scroll then position adjacent to element
    // BUG-2 fix: 'instant' scroll + 0ms deferred read (next task, not next microtask)
    try { targetEl.scrollIntoView({ behavior: 'instant', block: 'nearest', inline: 'nearest' }); } catch (e) {}

    // BUG-4 fix: read offsetHeight inside the deferred callback (after paint)
    setTimeout(function () {
      if (!overlayEl) return;

      var r   = targetEl.getBoundingClientRect();
      var vw  = window.innerWidth;
      var vh  = window.innerHeight;
      var ph  = overlayEl.offsetHeight || 220;   // BUG-4: was read before paint
      var pw  = POPOVER_W;
      var g   = POPOVER_GAP;
      var a   = ARROW_SIZE;

      // Skip if element is completely off-screen (shouldn't happen with instant scroll)
      if (r.bottom < 0 || r.top > vh || r.right < 0 || r.left > vw) {
        applyBottomCenter();
        return;
      }

      var space = {
        right:  vw - r.right,
        left:   r.left,
        bottom: vh - r.bottom,
        top:    r.top,
      };
      var order = [preferredSide, 'right', 'left', 'bottom', 'top'];
      var minW  = pw + g + a + 8;
      var minH  = ph + g + a + 8;
      var side  = 'bottom-center';

      for (var i = 0; i < order.length; i++) {
        var s = order[i];
        if ((s === 'right' || s === 'left')  && space[s] >= minW) { side = s; break; }
        if ((s === 'top'   || s === 'bottom') && space[s] >= minH) { side = s; break; }
      }

      if (side === 'bottom-center') {
        applyBottomCenter();
        return;
      }

      overlayEl.style.position  = 'fixed';
      overlayEl.style.transform = 'none';
      overlayEl.style.bottom    = '';
      overlayEl.style.left      = '';
      overlayEl.style.right     = '';
      overlayEl.style.top       = '';
      if (arrowEl) arrowEl.style.display = 'block';

      var cx = r.left + r.width  / 2;
      var cy = r.top  + r.height / 2;

      if (side === 'right') {
        var top = clamp(cy - ph / 2, 8, vh - ph - 8);
        overlayEl.style.left = (r.right + g + a) + 'px';
        overlayEl.style.top  = top + 'px';
        setArrow('left', r.right + g, clamp(cy - a, top + 8, top + ph - a * 2 - 8));
      } else if (side === 'left') {
        var top = clamp(cy - ph / 2, 8, vh - ph - 8);
        overlayEl.style.left = (r.left - pw - g - a) + 'px';
        overlayEl.style.top  = top + 'px';
        setArrow('right', r.left - g - a, clamp(cy - a, top + 8, top + ph - a * 2 - 8));
      } else if (side === 'bottom') {
        var left = clamp(cx - pw / 2, 8, vw - pw - 8);
        overlayEl.style.top  = (r.bottom + g + a) + 'px';
        overlayEl.style.left = left + 'px';
        setArrow('top', clamp(cx - a, left + 8, left + pw - a * 2 - 8), r.bottom + g);
      } else { // top
        var left = clamp(cx - pw / 2, 8, vw - pw - 8);
        overlayEl.style.top  = (r.top - ph - g - a) + 'px';
        overlayEl.style.left = left + 'px';
        setArrow('bottom', clamp(cx - a, left + 8, left + pw - a * 2 - 8), r.top - g - a);
      }
    }, 0);
  }

  function startCountdown() {
    secondsLeft = AUTO_CLOSE_S;
    updateCountdownBar();
    countdownInt = setInterval(function () {
      secondsLeft--;
      updateCountdownBar();
      if (secondsLeft <= 0) stopCountdown();
    }, 1000);
  }

  function stopCountdown() {
    if (countdownInt) { clearInterval(countdownInt); countdownInt = null; }
  }

  function resetCountdown() {
    stopCountdown();
    secondsLeft = AUTO_CLOSE_S;
    if (autoTimer) clearTimeout(autoTimer);
    autoTimer = setTimeout(dismiss, AUTO_CLOSE_S * 1000);
    startCountdown();
  }

  function updateCountdownBar() {
    if (!countdownBar) return;
    countdownBar.style.width = ((secondsLeft / AUTO_CLOSE_S) * 100) + '%';
    var textEl = overlayEl && overlayEl.querySelector('#gi-tour-countdown-text');
    if (textEl) textEl.textContent = secondsLeft + 's';
  }

  function activateFirstPair() {
    var tbody = document.getElementById('fx-pairs-tbody');
    if (!tbody) return;
    var firstRow = tbody.querySelector('tr[data-sym]');
    if (!firstRow) return;
    firstRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
    setTimeout(function () {
      if (typeof toggleInlineDetail === 'function') toggleInlineDetail(firstRow);
    }, 400);
  }

  function dismiss() {
    markDone();
    stopCountdown();
    if (autoTimer) { clearTimeout(autoTimer); autoTimer = null; }
    [overlayEl, arrowEl].forEach(function (el) {
      if (!el) return;
      el.style.opacity    = '0';
      el.style.transition = 'opacity .25s ease';
      setTimeout(function () { if (el && el.parentNode) el.parentNode.removeChild(el); }, 270);
    });
    overlayEl = null;
    arrowEl   = null;
    document.querySelectorAll('.gi-tour-highlight').forEach(function (el) {
      el.classList.remove('gi-tour-highlight');
    });
    document.removeEventListener('keydown', keyHandler);
  }

  function goToStep(idx) {
    currentStep = idx;
    renderStep();
    resetCountdown();
  }

  function next() {
    if (currentStep < STEPS.length - 1) {
      goToStep(currentStep + 1);
    } else {
      dismiss();
      activateFirstPair();
    }
  }

  function back() {
    if (currentStep > 0) goToStep(currentStep - 1);
  }

  function renderStep() {
    if (!overlayEl) return;
    var step   = STEPS[currentStep];
    var isLast = currentStep === STEPS.length - 1;
    var total  = STEPS.length;

    var dotsHTML = STEPS.map(function (_, i) {
      return '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;' +
        'background:' + (i === currentStep ? 'var(--blue)' : 'var(--border2)') +
        ';transition:background .2s;cursor:pointer;" data-tour-dot="' + i + '"></span>';
    }).join('');

    overlayEl.querySelector('#gi-tour-title').textContent        = step.title;
    overlayEl.querySelector('#gi-tour-body').innerHTML           = getStepBody(step);
    overlayEl.querySelector('#gi-tour-dots').innerHTML           = dotsHTML;
    overlayEl.querySelector('#gi-tour-step-counter').textContent = (currentStep + 1) + ' / ' + total;
    overlayEl.querySelector('#gi-tour-back').style.display       = currentStep === 0 ? 'none' : '';
    overlayEl.querySelector('#gi-tour-next').textContent         = isLast ? 'Enable alerts' : 'Next \u2192';

    overlayEl.querySelectorAll('[data-tour-dot]').forEach(function (dot) {
      dot.addEventListener('click', function () { goToStep(+dot.dataset.tourDot); });
    });

    var targetEl = highlight(step.target);
    positionPopover(targetEl, step.side);
  }

  function keyHandler(e) {
    if (!overlayEl) return;
    if (e.key === 'Escape')     dismiss();
    if (e.key === 'ArrowRight') next();
    if (e.key === 'ArrowLeft')  back();
  }

  function buildOverlay() {
    arrowEl = document.createElement('div');
    arrowEl.id = 'gi-tour-arrow';
    arrowEl.style.cssText = 'position:fixed;z-index:2999;display:none;pointer-events:none;width:0;height:0;';
    document.body.appendChild(arrowEl);

    var div = document.createElement('div');
    div.id = 'gi-welcome-tour';
    div.setAttribute('role', 'dialog');
    div.setAttribute('aria-modal', 'true');
    div.setAttribute('aria-label', 'Welcome to the FX Terminal');
    div.style.cssText = [
      'position:fixed',
      'z-index:3000',
      'width:' + POPOVER_W + 'px',
      'max-width:calc(100vw - 32px)',
      'background:var(--bg2)',
      'border:1px solid var(--border)',
      'border-radius:8px',
      'box-shadow:0 8px 32px rgba(0,0,0,0.55)',
      'font-family:var(--font-ui)',
      'padding:18px 18px 0',
      'opacity:0',
      'overflow:hidden',
      // Mobile: add max-height so long body text doesn't push buttons off screen
      'max-height:calc(100dvh - 120px)',
    ].join(';');

    div.innerHTML =
      '<button id="gi-tour-close" aria-label="Skip tour"' +
      ' style="position:absolute;top:10px;right:12px;background:none;border:none;color:var(--text2);font-size:14px;cursor:pointer;line-height:1;padding:2px 4px;">&#x2715;</button>' +

      '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">' +
        '<div style="font-size:10px;font-weight:700;letter-spacing:.08em;color:var(--text2);text-transform:uppercase;">Quick Tour</div>' +
        '<div id="gi-tour-step-counter" style="font-size:10px;color:var(--text3);"></div>' +
      '</div>' +

      '<div id="gi-tour-title" style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:6px;padding-right:20px;"></div>' +
      '<div id="gi-tour-body" style="font-size:12px;color:var(--text2);line-height:1.65;margin-bottom:14px;overflow-y:auto;max-height:120px;"></div>' +

      '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">' +
        '<div id="gi-tour-dots" style="display:flex;gap:5px;align-items:center;"></div>' +
        '<div style="display:flex;gap:8px;">' +
          '<button id="gi-tour-back" style="padding:5px 14px;background:none;color:var(--text2);border:1px solid var(--border2);border-radius:4px;font-size:11px;cursor:pointer;font-family:var(--font-ui);">\u2190 Back</button>' +
          '<button id="gi-tour-next" style="padding:5px 16px;background:var(--blue);color:#fff;border:none;border-radius:4px;font-size:11px;font-weight:700;cursor:pointer;font-family:var(--font-ui);letter-spacing:.04em;">Next \u2192</button>' +
        '</div>' +
      '</div>' +

      '<div style="display:flex;align-items:center;gap:6px;padding:6px 18px;margin:0 -18px;border-top:1px solid var(--border2);background:var(--bg3);">' +
        '<div style="flex:1;height:2px;background:var(--border2);border-radius:1px;overflow:hidden;">' +
          '<div id="gi-tour-countdown-bar" style="height:100%;background:var(--blue);width:100%;transition:width 1s linear;border-radius:1px;"></div>' +
        '</div>' +
        '<div id="gi-tour-countdown-text" style="font-size:9px;color:var(--text3);min-width:22px;text-align:right;">' + AUTO_CLOSE_S + 's</div>' +
      '</div>';

    document.body.appendChild(div);
    overlayEl    = div;
    countdownBar = div.querySelector('#gi-tour-countdown-bar');

    div.querySelector('#gi-tour-close').addEventListener('click', dismiss);
    div.querySelector('#gi-tour-next').addEventListener('click', function () {
      var step = STEPS[currentStep];
      if (step.lastCta) {
        var notifBtn = document.getElementById('sig-notif-btn');
        if (notifBtn) notifBtn.click();
      }
      next();
    });
    div.querySelector('#gi-tour-back').addEventListener('click', back);
    document.addEventListener('keydown', keyHandler);

    renderStep();

    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        div.style.transition = 'opacity .35s ease';
        div.style.opacity    = '1';
      });
    });

    autoTimer = setTimeout(dismiss, AUTO_CLOSE_S * 1000);
    startCountdown();
    setTimeout(function () {
      var closeBtn = div.querySelector('#gi-tour-close');
      if (closeBtn) closeBtn.focus();
    }, 400);
  }

  function injectStyles() {
    if (document.getElementById('gi-tour-styles')) return;
    var style = document.createElement('style');
    style.id = 'gi-tour-styles';
    style.textContent =
      '.gi-tour-highlight{' +
        'outline:2px solid var(--blue)!important;' +
        'outline-offset:3px!important;' +
        'border-radius:3px!important;' +
        'transition:outline .2s ease;' +
        'position:relative;' +
        'z-index:2998;' +
      '}';
    document.head.appendChild(style);
  }

  function init() {
    if (!shouldShow()) return;
    injectStyles();
    setTimeout(function () {
      if (!shouldShow()) return;
      buildOverlay();
    }, DELAY_MS);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.giReplayTour = function () {
    _memDone = false;
    try { localStorage.removeItem(STORAGE_KEY); } catch (e) { }
    if (overlayEl) dismiss();
    currentStep = 0;
    injectStyles();
    buildOverlay();
  };

})();
