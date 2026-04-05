(function() {
  'use strict';
  var CONSENT_KEY = 'gdpr_consent_v2';
  var ADSENSE_CLIENT = 'ca-pub-7977942731354565';

  function loadAdSense() {
    if (document.getElementById('adsense-script')) return;
    var s = document.createElement('script');
    s.id = 'adsense-script';
    s.async = true;
    s.crossOrigin = 'anonymous';
    s.src = 'https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=' + ADSENSE_CLIENT;
    document.head.appendChild(s);
  }

  function loadAdSenseNonPersonalized() {
    if (document.getElementById('adsense-script')) return;
    var s = document.createElement('script');
    s.id = 'adsense-script';
    s.async = true;
    s.crossOrigin = 'anonymous';
    s.src = 'https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=' + ADSENSE_CLIENT;
    s.setAttribute('data-ad-channel', 'non-personalized');
    document.head.appendChild(s);
    (window.adsbygoogle = window.adsbygoogle || []).push({ params: { google_npa: '1' } });
  }

  function hideBanner() {
    var banner = document.getElementById('gdpr-banner');
    // Single style mutation avoids forced reflow from toggling hidden + display separately
    if (banner) { banner.style.display = 'none'; banner.hidden = true; }
  }

  function applyConsent(choice) {
    localStorage.setItem(CONSENT_KEY, JSON.stringify({ v: choice, ts: Date.now() }));
    hideBanner();
    if (choice === 'accepted') loadAdSense();
    else loadAdSenseNonPersonalized();
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
          if (parsed.v === 'accepted') loadAdSense();
          else loadAdSenseNonPersonalized();
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
