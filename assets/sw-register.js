
(function () {
  'use strict';

  var PROMPT_KEY    = 'gi_push_prompted';   
  var SUB_KEY       = 'gi_push_sub';        
  var VAPID_PUBLIC  = '';                   

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

  if (!('serviceWorker' in navigator)) return;

  var swReg = null;

  window.addEventListener('load', function () {
    navigator.serviceWorker.register('/sw.js')
      .then(function (reg) {
        swReg = reg;
        window.subscribeCoTPush = function () { _requestPushSubscription(reg); };
      })
      .catch(function (err) {
        console.warn('[SW] Registration failed:', err);
      });
  });

  function _requestPushSubscription(reg) {
    if (!('PushManager' in window)) return;
    if (localStorage.getItem(PROMPT_KEY)) return;
    localStorage.setItem(PROMPT_KEY, '1');

    Notification.requestPermission().then(function (permission) {
      if (permission !== 'granted') return;
      _subscribeToPush(reg);
    }).catch(function () {  });
  }

  function _subscribeToPush(reg) {
    var options = { userVisibleOnly: true };
    if (VAPID_PUBLIC) {
      options.applicationServerKey = _urlB64ToUint8Array(VAPID_PUBLIC);
    }

    reg.pushManager.subscribe(options)
      .then(function (sub) {
        try { localStorage.setItem(SUB_KEY, JSON.stringify(sub)); } catch (e) {}
      })
      .catch(function (err) {
        console.warn('[SW] Push subscription failed:', err);
      });
  }

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
