/**
 * Global Investing FX Terminal — First-Visit Welcome Tour
 * Shows a 3-step overlay to new users to surface the terminal's core value
 * within the first 60 seconds.
 *
 * Storage key: 'gi_welcome_done' — set to '1' on completion or dismissal.
 * Does not run if the user has already completed it (subsequent visits).
 * Auto-dismissed after 60s if the user ignores it.
 */

(function () {
  'use strict';

  const STORAGE_KEY = 'gi_welcome_done';
  const DELAY_MS    = 2800; // Wait for data panels to visually populate
  const AUTO_CLOSE  = 60000;

  const STEPS = [
    {
      title: 'AI Market Narrative',
      body:  'Every 15 minutes, the terminal reads COT positioning, central bank rates, yield spreads, and cross-asset risk to produce a single market narrative — what the macro environment looks like right now.',
      target: 'narrative',
    },
    {
      title: 'Macro Regime',
      body:  'The Regime score synthesises VIX, MOVE, yield spreads, and gold to classify the market as RISK-ON, NEUTRAL, or RISK-OFF. Regime drives which currency pairs the current environment favours.',
      target: 'risk-regime',
    },
    {
      title: 'COT Positioning & Pair Bias',
      body:  'Scroll to the COT section to see institutional positioning in each G8 currency. Combined with rate differentials, it produces the directional bias shown per pair — the same multi-factor framework used in the GBP/JPY guide.',
      target: null,
    }
  ];

  let currentStep = 0;
  let autoTimer   = null;
  let overlayEl   = null;

  function shouldShow() {
    try { return !localStorage.getItem(STORAGE_KEY); } catch { return false; }
  }

  function markDone() {
    try { localStorage.setItem(STORAGE_KEY, '1'); } catch { /* ignore */ }
  }

  function dismiss() {
    markDone();
    if (autoTimer) clearTimeout(autoTimer);
    if (overlayEl) {
      overlayEl.style.opacity = '0';
      overlayEl.style.transition = 'opacity .25s ease';
      setTimeout(() => { if (overlayEl && overlayEl.parentNode) overlayEl.parentNode.removeChild(overlayEl); overlayEl = null; }, 270);
    }
    // Reset gi_ob_done so the alerts onboarding tooltip can run after the tour
    try { localStorage.removeItem('gi_ob_done'); } catch { /* ignore */ }
  }

  function highlight(targetId) {
    // Remove previous highlight
    document.querySelectorAll('.gi-tour-highlight').forEach(el => el.classList.remove('gi-tour-highlight'));
    if (!targetId) return;
    const el = document.getElementById(targetId);
    if (el) {
      el.classList.add('gi-tour-highlight');
      // Scroll element into view smoothly
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
    }
  }

  function renderStep() {
    const step = STEPS[currentStep];
    const isLast = currentStep === STEPS.length - 1;

    const dotsHTML = STEPS.map((_, i) =>
      `<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:${i === currentStep ? 'var(--blue)' : 'var(--border2)'};transition:background .2s;"></span>`
    ).join('');

    document.getElementById('gi-tour-title').textContent   = step.title;
    document.getElementById('gi-tour-body').textContent    = step.body;
    document.getElementById('gi-tour-dots').innerHTML      = dotsHTML;
    document.getElementById('gi-tour-back').style.display  = currentStep === 0 ? 'none' : '';
    document.getElementById('gi-tour-next').textContent    = isLast ? 'Get started' : 'Next';

    highlight(step.target);
  }

  function next() {
    if (currentStep < STEPS.length - 1) {
      currentStep++;
      renderStep();
    } else {
      dismiss();
      document.querySelectorAll('.gi-tour-highlight').forEach(el => el.classList.remove('gi-tour-highlight'));
    }
  }

  function back() {
    if (currentStep > 0) {
      currentStep--;
      renderStep();
    }
  }

  function buildOverlay() {
    const div = document.createElement('div');
    div.id = 'gi-welcome-tour';
    div.setAttribute('role', 'dialog');
    div.setAttribute('aria-modal', 'true');
    div.setAttribute('aria-label', 'Welcome to the FX Terminal');
    div.style.cssText = [
      'position:fixed',
      'bottom:24px',
      'left:50%',
      'transform:translateX(-50%)',
      'z-index:3000',
      'width:320px',
      'max-width:calc(100vw - 32px)',
      'background:var(--bg2)',
      'border:1px solid var(--border)',
      'border-radius:8px',
      'box-shadow:0 8px 32px rgba(0,0,0,0.55)',
      'font-family:var(--font-ui)',
      'padding:18px 18px 14px',
      'opacity:0',
    ].join(';');

    div.innerHTML = `
      <button id="gi-tour-close" aria-label="Skip tour"
        style="position:absolute;top:10px;right:12px;background:none;border:none;color:var(--text2);font-size:14px;cursor:pointer;line-height:1;padding:2px 4px;">&#x2715;</button>
      <div style="font-size:10px;font-weight:700;letter-spacing:.08em;color:var(--text2);margin-bottom:8px;text-transform:uppercase;">
        QUICK TOUR &nbsp;<span id="gi-tour-step-num" style="color:var(--blue);"></span>
      </div>
      <div id="gi-tour-title" style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:6px;"></div>
      <div id="gi-tour-body" style="font-size:12px;color:var(--text2);line-height:1.65;margin-bottom:14px;"></div>
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div id="gi-tour-dots" style="display:flex;gap:5px;align-items:center;"></div>
        <div style="display:flex;gap:8px;">
          <button id="gi-tour-back"
            style="padding:5px 14px;background:none;color:var(--text2);border:1px solid var(--border2);border-radius:4px;font-size:11px;cursor:pointer;font-family:var(--font-ui);">Back</button>
          <button id="gi-tour-next"
            style="padding:5px 16px;background:var(--blue);color:#fff;border:none;border-radius:4px;font-size:11px;font-weight:700;cursor:pointer;font-family:var(--font-ui);letter-spacing:.04em;">Next</button>
        </div>
      </div>
    `;

    document.body.appendChild(div);
    overlayEl = div;

    div.querySelector('#gi-tour-close').addEventListener('click', () => {
      document.querySelectorAll('.gi-tour-highlight').forEach(el => el.classList.remove('gi-tour-highlight'));
      dismiss();
    });
    div.querySelector('#gi-tour-next').addEventListener('click', next);
    div.querySelector('#gi-tour-back').addEventListener('click', back);

    // Keyboard: Escape closes, arrow keys navigate
    div.addEventListener('keydown', e => {
      if (e.key === 'Escape') { document.querySelectorAll('.gi-tour-highlight').forEach(el => el.classList.remove('gi-tour-highlight')); dismiss(); }
      if (e.key === 'ArrowRight') next();
      if (e.key === 'ArrowLeft') back();
    });

    renderStep();

    // Fade in
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        div.style.transition = 'opacity .35s ease';
        div.style.opacity = '1';
      });
    });

    autoTimer = setTimeout(() => {
      document.querySelectorAll('.gi-tour-highlight').forEach(el => el.classList.remove('gi-tour-highlight'));
      dismiss();
    }, AUTO_CLOSE);

    // Focus close button for accessibility
    setTimeout(() => { const cl = div.querySelector('#gi-tour-close'); if (cl) cl.focus(); }, 400);
  }

  function injectStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .gi-tour-highlight {
        outline: 2px solid var(--blue) !important;
        outline-offset: 3px !important;
        border-radius: 3px !important;
        transition: outline .2s ease;
      }
    `;
    document.head.appendChild(style);
  }

  function init() {
    if (!shouldShow()) return;
    injectStyles();
    setTimeout(() => {
      if (!shouldShow()) return;
      buildOverlay();
    }, DELAY_MS);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
