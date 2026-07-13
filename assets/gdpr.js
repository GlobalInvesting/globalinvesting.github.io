(function() {
  'use strict';
  var CONSENT_KEY = 'gdpr_consent_v2';

  function hideBanner() {
    var banner = document.getElementById('gdpr-banner');
    // Single style mutation avoids forced reflow from toggling hidden + display separately
    if (banner) { banner.style.display = 'none'; banner.hidden = true; }
  }

  function applyConsent(choice) {
    localStorage.setItem(CONSENT_KEY, JSON.stringify({ v: choice, ts: Date.now() }));
    hideBanner();
    // Update GA4 consent state — must be called before any gtag event fires.
    // 'denied' keeps GA4 in consent-mode cookieless mode (no cookies, no PII sent).
    // The Terminal does not use Google AdSense or any ad-serving network, so
    // only analytics_storage is relevant here (ad_storage/ad_user_data/
    // ad_personalization were removed along with AdSense — see CHANGELOG.md).
    if (typeof gtag === 'function') {
      gtag('consent', 'update', {
        analytics_storage: choice === 'accepted' ? 'granted' : 'denied',
      });
    }
  }

  document.addEventListener('click', function(e) {
    if (e.target && e.target.id === 'gdpr-accept') { e.preventDefault(); e.stopPropagation(); applyConsent('accepted'); }
    else if (e.target && e.target.id === 'gdpr-reject') { e.preventDefault(); e.stopPropagation(); applyConsent('rejected'); }
  }, true);

  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      var banner = document.getElementById('gdpr-banner');
      if (banner && !banner.hidden) applyConsent('rejected');
    }
  });

  function init() {
    try {
      var stored = localStorage.getItem(CONSENT_KEY);
      if (stored) {
        var parsed = JSON.parse(stored);
        if (parsed && parsed.ts && (Date.now() - parsed.ts) < 13 * 30 * 24 * 3600 * 1000) {
          // Restore GA4 consent state from prior session before any events fire.
          if (typeof gtag === 'function') {
            gtag('consent', 'update', {
              analytics_storage: parsed.v === 'accepted' ? 'granted' : 'denied',
            });
          }
          return;
        }
      }
    } catch(e) {}
    var banner = document.getElementById('gdpr-banner');
    if (banner) {
      // Batch style mutations: set display first (visible), then remove hidden in same task
      banner.style.display = 'flex';
      banner.hidden = false;
      setTimeout(function() { var btn = document.getElementById('gdpr-accept'); if (btn) btn.focus(); }, 300);
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
