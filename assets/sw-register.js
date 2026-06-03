// ═══════════════════════════════════════════════════════════════════
// sw-register.js — Service Worker registration + Push subscription
// ═══════════════════════════════════════════════════════════════════
// Push flow:
//  1. SW is registered on page load.
//  2. After first COT data load, the terminal calls window.subscribeCoTPush()
//     if the user has not yet been prompted (persisted in localStorage).
//  3. subscribeCoTPush() requests Notification permission, then calls
//     pushManager.subscribe() with the VAPID public key.
//  4. The subscription endpoint + keys are stored in localStorage for
//     reference. A real backend would POST this to a push server.
//
// Graceful degradation:
//  - If Push API is unavailable (older browser, iOS < 16.4) the module
//    silently no-ops. No error is surfaced to the user.
//  - If permission is denied, the prompt flag is still set so we never
//    ask again in the same browser.
// ═══════════════════════════════════════════════════════════════════

(function () {
  'use strict';

  var PROMPT_KEY    = 'gi_push_prompted';   // localStorage flag — never re-prompt
  var SUB_KEY       = 'gi_push_sub';        // localStorage — serialised subscription
  var VAPID_PUBLIC  = '';                   // Provide your VAPID public key here

  // ── 0. Capture PWA install prompt ──────────────────────────────────
  // Stores the deferred prompt so the tour or a button can trigger it.
  window._giDeferredInstallPrompt = null;
  window.addEventListener('beforeinstallprompt', function (e) {
    e.preventDefault();
    window._giDeferredInstallPrompt = e;
    var btn = document.getElementById('gi-pwa-install-btn');
    if (btn) btn.style.display = 'inline-block';
  });
  window.addEventListener('appinstalled', function () {
    window._giDeferredInstallPrompt = null;
    var btn = document.getElementById('gi-pwa-install-btn');
    if (btn) btn.style.display = 'none';
  });

  // ── 1. Register Service Worker ─────────────────────────────────────
  if (!('serviceWorker' in navigator)) return;

  var swReg = null;

  window.addEventListener('load', function () {
    navigator.serviceWorker.register('/sw.js')
      .then(function (reg) {
        swReg = reg;
        // Expose subscription helper — called only from an explicit UI button or tour step.
        // Never auto-prompts: browsers require a user gesture, and UX best practice requires
        // the user to understand what they're signing up for before seeing the permission dialog.
        window.subscribeCoTPush = function () { _requestPushSubscription(reg); };
      })
      .catch(function (err) {
        console.warn('[SW] Registration failed:', err);
      });
  });

  // ── 2. Request push subscription ──────────────────────────────────
  function _requestPushSubscription(reg) {
    if (!('PushManager' in window)) return;
    if (localStorage.getItem(PROMPT_KEY)) return;
    localStorage.setItem(PROMPT_KEY, '1');

    Notification.requestPermission().then(function (permission) {
      if (permission !== 'granted') return;
      _subscribeToPush(reg);
    }).catch(function () { /* ignore */ });
  }

  function _subscribeToPush(reg) {
    var options = { userVisibleOnly: true };
    if (VAPID_PUBLIC) {
      options.applicationServerKey = _urlB64ToUint8Array(VAPID_PUBLIC);
    }

    reg.pushManager.subscribe(options)
      .then(function (sub) {
        try { localStorage.setItem(SUB_KEY, JSON.stringify(sub)); } catch (e) {}
        // TODO: POST sub to push server when VAPID is configured
        // fetch('/api/push/subscribe', {
        //   method: 'POST',
        //   headers: { 'Content-Type': 'application/json' },
        //   body: JSON.stringify(sub)
        // });
      })
      .catch(function (err) {
        console.warn('[SW] Push subscription failed:', err);
      });
  }

  // ── Utility: convert VAPID base64 key to Uint8Array ───────────────
  function _urlB64ToUint8Array(base64String) {
    var padding = '='.repeat((4 - base64String.length % 4) % 4);
    var base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
    var rawData = atob(base64);
    var outputArray = new Uint8Array(rawData.length);
    for (var i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
  }
})();
