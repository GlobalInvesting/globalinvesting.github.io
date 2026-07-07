/**
 * Global Investing FX Terminal — First-Visit Welcome Tour
 * v7.89.10 — production build
 *
 * Changes vs v7.89.9:
 *   - BUG FIX: Economic Calendar step targeted 'section-tvcalendar-top', an ID that
 *     does not exist in index.html (actual element is 'section-tvcalendar'). Since
 *     getElementById returned null, positionPopover() fell back to applyBottomCenter(),
 *     pinning the popover to the bottom of the viewport instead of anchoring it below
 *     the calendar panel. Fixed target to 'section-tvcalendar'.
 *
 * Changes vs v7.81.5 (prior production):
 *   - 15-step tour: FX Pairs, Economic Calendar, ESI (opens modal), Macro Regime,
 *     Cross-Asset, COT Positioning (opens modal), Rates, Sessions, Heatmap, Derivatives,
 *     MT5 companion (new), Signal Alerts
 *   - ESI step opens openEconSurprisesModal('USD') — same pattern as COT step
 *   - dismiss() now also closes ESI modal via closeESModal()
 *   - Removed Install/RSS step (replaced by higher-value panels)
 *   - [prior] 10-step tour: AI Narrative, Macro Regime, Cross-Asset, COT Positioning,
 *     Rates, Heatmap, Derivatives, Install & Subscribe, Signal Alerts
 *   - Step 8 (Install & Subscribe): surfaces PWA install + RSS feed — retention
 *   - Derivatives step clicks nav link and waits 400ms before positioning
 *   - COT modal opened non-blocking (600ms delay) so popover renders first
 *   - All BUG-1..5 fixes from v7.73.0 preserved
 *   - Wider popover (340px), improved body max-height (140px), step fade animation
 *   - Icon badges on step titles (institutional look)
 *   - Progress bar replaces countdown bar — cleaner visual hierarchy
 */

(function () {
  'use strict';

  /* ─── config ─────────────────────────────────────────────────────────── */
  var STORAGE_KEY  = 'gi_welcome_done';
  var DELAY_MS     = 2400;
  var AUTO_CLOSE_S = 100;
  var POPOVER_W    = 340;
  var POPOVER_GAP  = 14;
  var ARROW_SIZE   = 8;
  var MOBILE_BP    = 900;

  /* ─── regime copy ────────────────────────────────────────────────────── */
  var REGIME_CONTEXT = {
    'RISK-ON':  'AUD, NZD, and higher-beta pairs tend to attract flows as yield appetite increases. The terminal is biased long carry.',
    'MIXED':    'Mixed signals: some risk appetite but with offsetting stress factors. Pair selection requires more discrimination than a clean RISK-ON.',
    'CAUTION':  'Elevated stress in 2-3 factors. USD, JPY, and CHF are seeing defensive demand. Avoid high-beta longs into this environment.',
    'RISK-OFF': 'JPY, CHF, and USD attract safe-haven flows. High-beta pairs (AUD, NZD) are under structural pressure until the regime clears.',
  };

  /* ─── step icons (unicode, no external dep) ─────────────────────────── */
  var STEP_ICONS = ['⬡', '◈', '◎', '◈', '◆', '▪', '▣', '◈', '◉'];

  /* ─── steps ──────────────────────────────────────────────────────────── */
  var STEPS = [
    /* 0 — Welcome */
    {
      target:  null,
      side:    'bottom',
      title:   'Institutional-grade FX data, in one place.',
      badge:   'Welcome',
      body:    'Most retail traders operate blind to what institutions are doing. This terminal changes that: live major currency rates, CFTC positioning, central bank policy, yield curves, economic surprise scoring, derivatives market structure, cross-asset risk — and an AI that synthesises it all into a single macro narrative, updated at every major session open.',
      action:  null,
    },

    /* 1 — AI Narrative */
    {
      target:  'narrative',
      side:    'bottom',
      title:   'AI Market Narrative',
      badge:   'Overview',
      body:    'At each major session open the terminal reads COT positioning, rate differentials, yield spreads, and cross-asset risk to produce a single narrative: what the macro environment looks like right now, in plain language. No interpretation required — scroll down for the full breakdown.',
      action:  null,
    },

    /* 2 — FX Pairs & Price Chart */
    {
      target:  'section-fxpairs',
      side:    'right',
      title:   'Live Price Chart & FX Pairs Table',
      badge:   'Price Chart',
      body:    'Live charts for all 28 major pairs — EUR/USD, USD/JPY, GBP/USD and more — plus a full pairs table with intraday change, spread, and carry differential. Switch pair with one click; the chart, heatmap, and carry rank all update together.',
      action:  null,
    },

    /* 3 — Economic Calendar */
    {
      target:  'section-tvcalendar',
      side:    'bottom',
      title:   'Economic Calendar',
      badge:   'Calendar',
      body:    'Upcoming major-economy macro releases filtered to medium and high impact only — the events that actually move FX. Each event shows local time across all major currency timezones, the consensus forecast, and the previous print. Once the actual is released the terminal scores it as a beat or miss and feeds the result directly into the Economic Surprise Index.',
      action:  null,
    },

    /* 4 — Economic Surprise Index (opens modal) */
    {
      target:    'split-lower',
      highlight: 'section-econ-surprise',
      side:    'left',
      title:   'Economic Surprise Index · 8 major currencies · 90d rolling',
      badge:   'ESI',
      body:    'The ESI is a decay-weighted score that measures whether major-economy economic data is consistently beating or missing consensus — decay-weighted so recent releases count more than older ones. A rising ESI signals that the economy is outperforming expectations, which is typically bullish for the currency. The chart is opening now so you can explore the 90-day rolling window.',
      action:  function () {
        try {
          // Scroll rightpanel so "Economic Surprises" section is visible
          var rp = document.getElementById('rightpanel');
          var es = document.getElementById('section-econ-surprise');
          if (rp && es) {
            rp.scrollTo({ top: es.offsetTop - rp.offsetTop - 4, behavior: 'smooth' });
          }
          if (typeof window.openEconSurprisesModal === 'function') {
            setTimeout(function () { window.openEconSurprisesModal('USD'); }, 700);
          }
        } catch (e) {}
      },
    },

    /* 5 — Macro Risk Regime */
    {
      target:    'split-lower',
      highlight: 'section-risk',
      side:    'left',
      title:   'Macro Risk Regime',
      badge:   'Risk',
      body:    null,
      dynamic: 'regime',
      action:  function () {
        // Close the ESI inline panel completely.
        // closeESModal() removes #esm-bd but the inline-panel wrap stays alive
        // with overflowY:hidden and no content → black screen.
        // Clicking the wrap's own close button triggers onClose (closeESModal)
        // AND runs restoreChildren() + removes the wrap correctly.
        try {
          var lwr = document.getElementById('split-lower');
          if (lwr) {
            var wrap = lwr.querySelector('[data-inline-panel]');
            if (wrap) {
              var closeBtn = wrap.querySelector('button[aria-label="Close panel"]');
              if (closeBtn) { closeBtn.click(); return; }
              wrap.remove();
            }
          }
          if (typeof window.closeESModal === 'function') window.closeESModal();
        } catch (e) {}
      },
    },

    /* 6 — Cross-Asset */
    {
      target:  'section-crossasset',
      side:    'right',
      title:   'Cross-Asset Risk Monitor',
      badge:   'Cross-Asset',
      body:    'VIX, MOVE, DXY, Gold, and S&amp;P 500 — the five inputs that determine the regime. When two or more enter stress territory the terminal shifts to CAUTION or RISK-OFF. This panel shows you exactly which factors are driving the read, refreshed every 5 minutes.',
      action:  null,
    },

    /* 7 — COT Positioning (opens modal) */
    {
      target:    'split-lower',
      highlight: 'section-positioning',
      side:    'left',
      title:   'CFTC COT Positioning',
      badge:   'Positioning',
      body:    'The Commitment of Traders report reveals what institutional speculators — hedge funds and large money managers — are actually holding. Click any currency row to open a detailed modal: net positioning history, z-score, crowding indicator, and the COT-based directional bias. The modal is opening now so you can see it in action.',
      action:  function () {
        try {
          var store = window.COT_DATA_STORE;
          if (!store) return;
          var cotRows = document.querySelectorAll('#cot-rows .cot-row, #cot-rows [data-ccy]');
          var ccy = cotRows.length > 0 ? (cotRows[0].dataset.ccy || 'EUR') : 'EUR';
          if (store[ccy] && typeof window.openCOTModal === 'function') {
            setTimeout(function () { window.openCOTModal(ccy, store[ccy]); }, 700);
          }
        } catch (e) {}
      },
    },

    /* 8 — Rates & Yield Curve */
    {
      target:  'section-rates',
      side:    'top',
      title:   'Rates & Yield Curve',
      badge:   'Rates',
      body:    'major sovereign yields across the full term structure — 3M through 30Y — plotted against the prior close. Switch tabs for DE, GB, JP, AU, CA, NZ or Sovereign Spreads. Key spread signals (2Y–10Y slope, US–DE, US–JP) flag curve regime shifts in real time. Click the curve to open the detailed yield modal.',
      action:  function () {
        // Close the COT inline panel correctly via the shell close button.
        try {
          var lwr = document.getElementById('split-lower');
          if (lwr) {
            var wrap = lwr.querySelector('[data-inline-panel]');
            if (wrap) {
              var closeBtn = wrap.querySelector('button[aria-label="Close panel"]');
              if (closeBtn) { closeBtn.click(); return; }
              wrap.remove();
            }
          }
          if (typeof window.closeCOTModal === 'function') window.closeCOTModal();
        } catch (e) {}
      },
    },

    /* 9 — CB Rate Expectations & OIS */
    {
      target:    'section-rates',
      highlight: 'section-cb-expectations',
      side:      'top',
      title:     'CB Rate Expectations & OIS Forwards',
      badge:     'OIS',
      body:      'The market\'s best estimate of where each major central bank will be at its next meeting — derived from OIS and futures markets (SOFR, €STR, SONIA, TONA, CORRA, SARON). Each row shows the current policy rate, meeting date, directional bias, and the probability-weighted implied rate. Click any central bank flag to open the full decision history: 24 months of decisions plotted on the yield curve, plus the OIS-implied forward rate.',
      action:    function () {
        try {
          var rp = document.getElementById('rightpanel');
          var el = document.getElementById('section-cb-expectations');
          if (rp && el) rp.scrollTo({ top: el.offsetTop - rp.offsetTop - 4, behavior: 'smooth' });
        } catch (e) {}
      },
    },

    /* 10 — Market Sessions */
    {
      target:  'section-sessions',
      side:    'top',
      title:   'Market Sessions',
      badge:   'Sessions',
      body:    'Live session clocks for Tokyo, London, and New York with overlap windows highlighted — the hours where liquidity and volatility peak. Each session shows which major pairs are most active in that window, so you know exactly when your pair is trading at its tightest spread and deepest order book.',
      action:  function () {
        try { if (typeof window.closeCBRatesModal === 'function') window.closeCBRatesModal(); } catch (e) {}
      },
    },

    /* 11 — Heatmap */
    {
      target:  'heatmap-grid',
      side:    'top',
      title:   'Currency Strength Heatmap',
      badge:   'Heatmap',
      body:    'G10 currencies ranked by intraday strength across all direct major pairs. Click any cell to open the currency detail modal: direct pairs, live carry, COT bias, realized vol, and pair correlations — the complete single-currency picture without switching views.',
      action:  null,
    },

    /* 12 — Derivatives */
    {
      target:  'section-derivatives',
      side:    'top',
      title:   'Derivatives — Forwards, Vol Surface & OTC Flow',
      badge:   'Derivatives',
      body:    'Four data streams in one section:<br>' +
               '&nbsp;&nbsp;<b>CIP Forwards</b> — implied 1M–1Y forward prices from covered interest parity.<br>' +
               '&nbsp;&nbsp;<b>RR Term Structure</b> — 25-delta risk reversal skew across tenors, updated daily.<br>' +
               '&nbsp;&nbsp;<b>ECB Fixings</b> — official daily reference rates vs live spot.<br>' +
               '&nbsp;&nbsp;<b>DTCC GTR</b> — actual OTC FX notional reported under Dodd-Frank: Swap, Forward &amp; NDF breakdown per pair.',
      action:  function () {
        try { if (typeof window.closeHeatmapModal === 'function') window.closeHeatmapModal(); } catch (e) {}
        try {
          var derivLink = document.querySelector('.top-nav a[data-target="section-derivatives"]');
          if (derivLink) {
            derivLink.click();
          } else {
            var sec = document.getElementById('section-derivatives');
            if (sec) {
              sec.style.display = '';
              if (typeof renderDerivativesSection === 'function') renderDerivativesSection();
            }
          }
        } catch (e) {}
      },
    },

    /* 13 — Web terminal as analytical layer */
    {
      target:  null,
      side:    'bottom',
      title:   'This terminal is the analytical layer of the platform.',
      badge:   'Platform',
      body:    'The MT5 EA covers real-time prices, CB bias, COT, carry, sessions, and alerts — the live trading layer. This web terminal extends that with the full macro suite: yield curve, derivatives (25d risk reversals, implied forwards, realized vol), cross-asset correlations, AI-generated market narrative, and complete chart history. Both are included under the same EA rental.',
      action:  function () {
        try { if (typeof window._derivNavHide === 'function') window._derivNavHide(); } catch (e) {}
      },
    },

    /* 14 — Signal alerts (last CTA) */
    {
      target:  'sig-notif-btn',
      side:    'top',
      title:   'Stay ahead — enable signal alerts',
      badge:   'Alerts',
      body:    'The terminal publishes AI-generated signals when the regime shifts or a high-conviction setup appears. Enable browser notifications to catch the signal at session open, even when this tab is in the background — no account or email required.',
      action:  null,
      lastCta: true,
    },
  ];
  /* ─── state ──────────────────────────────────────────────────────────── */
  var currentStep  = 0;
  var overlayEl    = null;
  var arrowEl      = null;
  var countdownBar = null;
  var progressBar  = null;
  var countdownInt = null;
  var autoTimer    = null;
  var secondsLeft  = AUTO_CLOSE_S;
  var _memDone     = false;

  /* ─── storage ────────────────────────────────────────────────────────── */
  // Multi-layer persistence: localStorage → sessionStorage → cookie.
  // Edge Tracking Prevention can silently block localStorage.setItem without
  // throwing, so we verify the write succeeded and fall back to sessionStorage,
  // then to a session cookie. _memDone guards against showing the tour twice
  // within the same page load even if all storage layers fail.
  function _storageGet() {
    try { if (localStorage.getItem(STORAGE_KEY)) return true; } catch (e) {}
    try { if (sessionStorage.getItem(STORAGE_KEY)) return true; } catch (e) {}
    try { if (document.cookie.indexOf(STORAGE_KEY + '=1') !== -1) return true; } catch (e) {}
    return false;
  }

  function _storageSet() {
    var ok = false;
    try {
      localStorage.setItem(STORAGE_KEY, '1');
      // Verify the write actually persisted (silent-fail defence)
      if (localStorage.getItem(STORAGE_KEY) === '1') ok = true;
    } catch (e) {}
    if (!ok) {
      try { sessionStorage.setItem(STORAGE_KEY, '1'); ok = true; } catch (e) {}
    }
    if (!ok) {
      try { document.cookie = STORAGE_KEY + '=1;path=/;max-age=31536000'; } catch (e) {}
    }
  }

  function shouldShow() {
    if (_memDone) return false;
    return !_storageGet();
  }

  function markDone() {
    _memDone = true;
    _storageSet();
  }

  /* ─── utils ──────────────────────────────────────────────────────────── */
  function isMobile() { return window.innerWidth <= MOBILE_BP; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function getRegimeValue() {
    var el = document.getElementById('risk-regime');
    return el ? (el.textContent || '').trim().toUpperCase() : '';
  }

  function buildRegimeBody() {
    var regime  = getRegimeValue();
    var context = REGIME_CONTEXT[regime];
    if (!regime || regime === '--' || !context) {
      return 'The Regime score synthesises VIX, MOVE, yield spreads, and gold to classify the macro environment as RISK-ON, CAUTION, or RISK-OFF — determining which currency pairs the current environment structurally favours.';
    }
    return 'Right now the terminal reads <strong>' + regime + '</strong>. ' + context;
  }

  function getStepBody(step) {
    if (step.dynamic === 'regime') return buildRegimeBody();
    return step.body;
  }

  /* ─── highlight ──────────────────────────────────────────────────────── */
  function highlight(targetId) {
    document.querySelectorAll('.gi-tour2-highlight').forEach(function (el) {
      el.classList.remove('gi-tour2-highlight');
    });
    if (!targetId) return null;
    var el = document.getElementById(targetId);
    if (el) el.classList.add('gi-tour2-highlight');
    return el || null;
  }

  /* ─── positioning ────────────────────────────────────────────────────── */
  function applyBottomCenter() {
    if (!overlayEl) return;
    overlayEl.style.cssText += [
      ';position:fixed',
      'bottom:' + (isMobile() ? '72px' : '24px'),
      'left:50%',
      'transform:translateX(-50%)',
      'top:auto',
      'right:auto',
    ].join(';');
    if (arrowEl) arrowEl.style.display = 'none';
  }

  function setArrow(from, x, y) {
    if (!arrowEl) return;
    var s = arrowEl.style;
    s.display = 'block';
    s.left    = x + 'px';
    s.top     = y + 'px';
    s.border  = ARROW_SIZE + 'px solid transparent';
    var col = 'var(--border)';
    s.borderRightColor  = from === 'left'   ? col : 'transparent';
    s.borderLeftColor   = from === 'right'  ? col : 'transparent';
    s.borderTopColor    = from === 'bottom' ? col : 'transparent';
    s.borderBottomColor = from === 'top'    ? col : 'transparent';
  }

  function positionPopover(targetEl, preferredSide) {
    if (!overlayEl) return;
    if (!targetEl)  { applyBottomCenter(); return; }
    if (isMobile()) {
      try { targetEl.scrollIntoView({ behavior: 'instant', block: 'center', inline: 'nearest' }); } catch (e) {}
      applyBottomCenter();
      return;
    }
    try { targetEl.scrollIntoView({ behavior: 'instant', block: 'nearest', inline: 'nearest' }); } catch (e) {}

    setTimeout(function () {
      if (!overlayEl) return;
      var r  = targetEl.getBoundingClientRect();
      var vw = window.innerWidth, vh = window.innerHeight;
      var ph = overlayEl.offsetHeight || 280;
      var pw = POPOVER_W, g = POPOVER_GAP, a = ARROW_SIZE;

      if (r.bottom < 0 || r.top > vh || r.right < 0 || r.left > vw) {
        applyBottomCenter(); return;
      }

      var space = { right: vw - r.right, left: r.left, bottom: vh - r.bottom, top: r.top };
      var order = [preferredSide, 'right', 'left', 'bottom', 'top'];
      var side  = 'bottom-center';

      for (var i = 0; i < order.length; i++) {
        var s = order[i];
        if ((s === 'right' || s === 'left')  && space[s] >= pw + g + a + 8) { side = s; break; }
        if ((s === 'top'   || s === 'bottom') && space[s] >= ph + g + a + 8) { side = s; break; }
      }
      if (side === 'bottom-center') { applyBottomCenter(); return; }

      overlayEl.style.position  = 'fixed';
      overlayEl.style.transform = 'none';
      overlayEl.style.bottom    = '';

      var cx = r.left + r.width / 2, cy = r.top + r.height / 2;

      if (side === 'right') {
        var top = clamp(cy - ph / 2, 8, vh - ph - 8);
        overlayEl.style.left = (r.right + g + a) + 'px';
        overlayEl.style.top  = top + 'px';
        overlayEl.style.right = '';
        setArrow('left', r.right + g, clamp(cy - a, top + 8, top + ph - a * 2 - 8));
      } else if (side === 'left') {
        var top = clamp(cy - ph / 2, 8, vh - ph - 8);
        overlayEl.style.left = (r.left - pw - g - a) + 'px';
        overlayEl.style.top  = top + 'px';
        overlayEl.style.right = '';
        setArrow('right', r.left - g - a, clamp(cy - a, top + 8, top + ph - a * 2 - 8));
      } else if (side === 'bottom') {
        var left = clamp(cx - pw / 2, 8, vw - pw - 8);
        overlayEl.style.top  = (r.bottom + g + a) + 'px';
        overlayEl.style.left = left + 'px';
        overlayEl.style.right = '';
        setArrow('top', clamp(cx - a, left + 8, left + pw - a * 2 - 8), r.bottom + g);
      } else {
        var left = clamp(cx - pw / 2, 8, vw - pw - 8);
        overlayEl.style.top  = (r.top - ph - g - a) + 'px';
        overlayEl.style.left = left + 'px';
        overlayEl.style.right = '';
        setArrow('bottom', clamp(cx - a, left + 8, left + pw - a * 2 - 8), r.top - g - a);
      }
    }, 0);
  }

  /* ─── countdown ──────────────────────────────────────────────────────── */
  function startCountdown() {
    secondsLeft = AUTO_CLOSE_S;
    updateCountdown();
    countdownInt = setInterval(function () {
      secondsLeft--;
      updateCountdown();
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

  function updateCountdown() {
    if (countdownBar) countdownBar.style.width = ((secondsLeft / AUTO_CLOSE_S) * 100) + '%';
    var textEl = overlayEl && overlayEl.querySelector('#gi-tour2-countdown-text');
    if (textEl) textEl.textContent = secondsLeft + 's';
    /* also update step progress bar width */
    if (progressBar) {
      progressBar.style.width = (((currentStep + 1) / STEPS.length) * 100) + '%';
    }
  }

  /* ─── dismiss ────────────────────────────────────────────────────────── */
  function dismiss() {
    markDone();
    stopCountdown();
    if (autoTimer) { clearTimeout(autoTimer); autoTimer = null; }
    try { if (typeof window.closeCOTModal     === 'function') window.closeCOTModal();     } catch (e) {}
    try { if (typeof window.closeCBRatesModal === 'function') window.closeCBRatesModal(); } catch (e) {}
    try { if (typeof window.closeHeatmapModal === 'function') window.closeHeatmapModal(); } catch (e) {}
    try { if (typeof window._derivNavHide     === 'function') window._derivNavHide();     } catch (e) {}
    try {
      var _lwr = document.getElementById('split-lower');
      if (_lwr) {
        var _wrap = _lwr.querySelector('[data-inline-panel]');
        if (_wrap) {
          var _cb = _wrap.querySelector('button[aria-label="Close panel"]');
          if (_cb) { _cb.click(); }
          else { _wrap.remove(); }
        }
      }
      if (typeof window.closeESModal === 'function') window.closeESModal();
    } catch (e) {}
    [overlayEl, arrowEl].forEach(function (el) {
      if (!el) return;
      el.style.opacity    = '0';
      el.style.transition = 'opacity .25s ease';
      setTimeout(function () { if (el && el.parentNode) el.parentNode.removeChild(el); }, 280);
    });
    overlayEl = null; arrowEl = null;
    document.querySelectorAll('.gi-tour2-highlight').forEach(function (el) {
      el.classList.remove('gi-tour2-highlight');
    });
    document.removeEventListener('keydown', keyHandler);
  }

  function activateFirstPair() {
    try {
      var tbody    = document.getElementById('fx-pairs-tbody');
      if (!tbody) return;
      var firstRow = tbody.querySelector('tr[data-sym]');
      if (!firstRow) return;
      firstRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
      setTimeout(function () {
        if (typeof toggleInlineDetail === 'function') toggleInlineDetail(firstRow);
      }, 400);
    } catch (e) {}
  }

  /* ─── navigation ─────────────────────────────────────────────────────── */
  function goToStep(idx) { currentStep = idx; renderStep(); resetCountdown(); }
  function next() {
    if (currentStep < STEPS.length - 1) goToStep(currentStep + 1);
    else { dismiss(); activateFirstPair(); }
  }
  function back() { if (currentStep > 0) goToStep(currentStep - 1); }

  /* ─── render ─────────────────────────────────────────────────────────── */
  function renderStep() {
    if (!overlayEl) return;
    var step   = STEPS[currentStep];
    var isLast = currentStep === STEPS.length - 1;
    var total  = STEPS.length;

    /* dots */
    var dotsHTML = STEPS.map(function (_, i) {
      var active = i === currentStep;
      return '<span style="display:inline-block;width:' + (active ? '18' : '6') + 'px;height:6px;border-radius:3px;' +
        'background:' + (active ? 'var(--blue)' : 'var(--border2)') +
        ';transition:all .25s;cursor:pointer;" data-tour-dot="' + i + '"></span>';
    }).join('');

    /* badge */
    var badge = step.badge
      ? '<span style="display:inline-block;font-size:9px;font-weight:700;letter-spacing:.08em;' +
        'text-transform:uppercase;color:var(--blue);background:color-mix(in srgb,var(--blue) 12%,transparent);' +
        'border:1px solid color-mix(in srgb,var(--blue) 30%,transparent);' +
        'border-radius:3px;padding:1px 5px;margin-bottom:6px;">' + step.badge + '</span><br>'
      : '';

    overlayEl.querySelector('#gi-tour2-badge-title').innerHTML  = badge + step.title;
    overlayEl.querySelector('#gi-tour2-body').innerHTML         = getStepBody(step);
    overlayEl.querySelector('#gi-tour2-dots').innerHTML         = dotsHTML;
    overlayEl.querySelector('#gi-tour2-step-counter').textContent = (currentStep + 1) + '\u202f/\u202f' + total;
    overlayEl.querySelector('#gi-tour2-back').style.display     = currentStep === 0 ? 'none' : '';
    overlayEl.querySelector('#gi-tour2-next').textContent       = isLast ? 'Enable alerts \u2192' : 'Next \u2192';

    /* progress bar */
    if (progressBar) progressBar.style.width = (((currentStep + 1) / total) * 100) + '%';

    /* dot click */
    overlayEl.querySelectorAll('[data-tour-dot]').forEach(function (dot) {
      dot.addEventListener('click', function () { goToStep(+dot.dataset.tourDot); });
    });

    /* run step action */
    if (step.action) {
      try { step.action(); } catch (e) { console.warn('[gi-tour2] step action error:', e); }
    }

    /* position popover — extra delay for derivatives nav and modal steps */
    /* step.highlight (optional) separates the highlighted element from the positioning target */
    highlight(step.highlight !== undefined ? step.highlight : step.target);
    var targetEl = step.target ? (document.getElementById(step.target) || null) : null;
    var posDelay = (step.target === 'section-derivatives' || step.target === 'section-econ-surprise' || step.target === 'split-lower') ? 800 : 0;
    setTimeout(function () { positionPopover(targetEl, step.side); }, posDelay);
  }

  /* ─── keyboard ───────────────────────────────────────────────────────── */
  function keyHandler(e) {
    if (!overlayEl) return;
    if (e.key === 'Escape')     dismiss();
    if (e.key === 'ArrowRight') next();
    if (e.key === 'ArrowLeft')  back();
  }

  /* ─── build DOM ──────────────────────────────────────────────────────── */
  function buildOverlay() {
    /* arrow tip */
    arrowEl = document.createElement('div');
    arrowEl.id = 'gi-tour2-arrow';
    arrowEl.style.cssText = 'position:fixed;z-index:2999;display:none;pointer-events:none;width:0;height:0;';
    document.body.appendChild(arrowEl);

    /* popover */
    var div = document.createElement('div');
    div.id = 'gi-welcome-tour2';
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
      'border-radius:10px',
      'box-shadow:0 12px 40px rgba(0,0,0,.65),0 0 0 1px rgba(255,255,255,.04) inset',
      'font-family:var(--font-ui)',
      'padding:16px 16px 0',
      'opacity:0',
      'overflow:hidden',
      'max-height:calc(100dvh - 120px)',
    ].join(';');

    div.innerHTML =
      /* header row — label left, step-counter + close button right (no absolute overlap) */
      '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">' +
        '<div style="font-size:9.5px;font-weight:700;letter-spacing:.09em;color:var(--text3);text-transform:uppercase;' +
        'display:flex;align-items:center;gap:5px;">' +
          '<span style="width:5px;height:5px;border-radius:50%;background:var(--blue);display:inline-block;' +
          'box-shadow:0 0 5px var(--blue);"></span>Quick Tour' +
        '</div>' +
        '<div style="display:flex;align-items:center;gap:6px;">' +
          '<div id="gi-tour2-step-counter" style="font-size:9.5px;color:var(--text3);font-variant-numeric:tabular-nums;"></div>' +
          '<button id="gi-tour2-close" aria-label="Skip tour" style="' +
          'background:none;border:none;color:var(--text3);font-size:13px;cursor:pointer;line-height:1;' +
          'padding:3px 5px;border-radius:3px;transition:color .15s;flex-shrink:0;" ' +
          'onmouseover="this.style.color=\'var(--text)\'" onmouseout="this.style.color=\'var(--text3)\'">&#x2715;</button>' +
        '</div>' +
      '</div>' +

      /* step progress bar */
      '<div style="height:2px;background:var(--border2);border-radius:1px;margin:0 0 12px;overflow:hidden;">' +
        '<div id="gi-tour2-progress" style="height:100%;background:var(--blue);width:0;' +
        'transition:width .35s cubic-bezier(.4,0,.2,1);border-radius:1px;"></div>' +
      '</div>' +

      /* badge + title */
      '<div id="gi-tour2-badge-title" style="font-size:13px;font-weight:700;color:var(--text);' +
      'margin-bottom:8px;line-height:1.4;"></div>' +

      /* body */
      '<div id="gi-tour2-body" style="font-size:11.5px;color:var(--text2);line-height:1.7;' +
      'margin-bottom:14px;overflow-y:auto;max-height:140px;"></div>' +

      /* footer row — dots + buttons */
      '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">' +
        '<div id="gi-tour2-dots" style="display:flex;gap:4px;align-items:center;"></div>' +
        '<div style="display:flex;gap:7px;">' +
          '<button id="gi-tour2-back" style="padding:5px 13px;background:none;color:var(--text2);' +
          'border:1px solid var(--border2);border-radius:5px;font-size:11px;cursor:pointer;' +
          'font-family:var(--font-ui);transition:border-color .15s,color .15s;"' +
          'onmouseover="this.style.borderColor=\'var(--text3)\';this.style.color=\'var(--text)\'"' +
          'onmouseout="this.style.borderColor=\'var(--border2)\';this.style.color=\'var(--text2)\'">' +
          '\u2190 Back</button>' +
          '<button id="gi-tour2-next" style="padding:5px 15px;background:var(--blue);color:#fff;' +
          'border:none;border-radius:5px;font-size:11px;font-weight:700;cursor:pointer;' +
          'font-family:var(--font-ui);letter-spacing:.04em;transition:filter .15s;"' +
          'onmouseover="this.style.filter=\'brightness(1.15)\'"' +
          'onmouseout="this.style.filter=\'none\'">Next \u2192</button>' +
        '</div>' +
      '</div>' +

      /* countdown strip */
      '<div style="display:flex;align-items:center;gap:7px;padding:5px 16px;margin:0 -16px;' +
      'border-top:1px solid var(--border2);background:var(--bg3);">' +
        '<div style="flex:1;height:2px;background:var(--border2);border-radius:1px;overflow:hidden;">' +
          '<div id="gi-tour2-countdown-bar" style="height:100%;background:color-mix(in srgb,var(--blue) 50%,var(--border2));' +
          'width:100%;transition:width 1s linear;border-radius:1px;"></div>' +
        '</div>' +
        '<div id="gi-tour2-countdown-text" style="font-size:9px;color:var(--text3);' +
        'min-width:24px;text-align:right;font-variant-numeric:tabular-nums;">' + AUTO_CLOSE_S + 's</div>' +
      '</div>';

    document.body.appendChild(div);
    overlayEl    = div;
    countdownBar = div.querySelector('#gi-tour2-countdown-bar');
    progressBar  = div.querySelector('#gi-tour2-progress');

    /* events */
    div.querySelector('#gi-tour2-close').addEventListener('click', dismiss);

    div.querySelector('#gi-tour2-next').addEventListener('click', function () {
      var step = STEPS[currentStep];
      if (step.lastCta) {
        var notifBtn = document.getElementById('sig-notif-btn');
        if (notifBtn) notifBtn.click();
      }
      next();
    });

    div.querySelector('#gi-tour2-back').addEventListener('click', back);
    document.addEventListener('keydown', keyHandler);

    renderStep();

    /* fade in */
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        div.style.transition = 'opacity .4s ease, transform .4s ease';
        div.style.opacity    = '1';
      });
    });

    autoTimer = setTimeout(dismiss, AUTO_CLOSE_S * 1000);
    startCountdown();

    setTimeout(function () {
      var closeBtn = div.querySelector('#gi-tour2-close');
      if (closeBtn) closeBtn.focus();
    }, 450);
  }

  /* ─── styles ─────────────────────────────────────────────────────────── */
  function injectStyles() {
    if (document.getElementById('gi-tour2-styles')) return;
    var style = document.createElement('style');
    style.id = 'gi-tour2-styles';
    style.textContent =
      '.gi-tour2-highlight{' +
        'outline:2px solid var(--blue)!important;' +
        'outline-offset:4px!important;' +
        'border-radius:4px!important;' +
        'transition:outline .2s ease;' +
        'position:relative;' +
        'z-index:2998;' +
      '}' +
      '#gi-welcome-tour2 #gi-tour2-body::-webkit-scrollbar{width:3px;}' +
      '#gi-welcome-tour2 #gi-tour2-body::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}';
    document.head.appendChild(style);
  }

  /* ─── init ───────────────────────────────────────────────────────────── */
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

  /* ─── public replay API ──────────────────────────────────────────────── */
  window.giReplayTour = function () {
    _memDone = false;
    try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
    try { sessionStorage.removeItem(STORAGE_KEY); } catch (e) {}
    try { document.cookie = STORAGE_KEY + '=;path=/;max-age=0'; } catch (e) {}
    if (overlayEl) dismiss();
    currentStep = 0;
    injectStyles();
    buildOverlay();
  };

})();
