// ═══════════════════════════════════════════════════════════════════
// sw.js — Global Investing FX Terminal Service Worker
// Strategy:
//   • Static shell (HTML, CSS, JS, icons) → Cache-first, update in bg
//   • JSON data endpoints  → Network-first, cache as fallback
//   • Everything else      → Network only
// ═══════════════════════════════════════════════════════════════════

const CACHE_VERSION = 'gi-v1';
const CACHE_STATIC  = `${CACHE_VERSION}-static`;
const CACHE_DATA    = `${CACHE_VERSION}-data`;

// Core shell files cached on install
const STATIC_PRECACHE = [
  '/',
  '/index.html',
  '/assets/dashboard.css?v=7.26.3',
  '/assets/dashboard.js?v=7.26.3',
  '/assets/gdpr.js',
  '/assets/sw-register.js',
  '/favicon.ico',
  '/favicon-32x32.png',
  '/favicon-192x192.png',
  '/apple-touch-icon.png',
  '/manifest.json',
];

// Paths treated as data (network-first)
const DATA_PATH_PREFIXES = [
  '/ai-analysis/',
  '/cot-data/',
  '/economic-data/',
  '/extended-data/',
  '/fx-data/',
  '/intraday-data/',
  '/meetings-data/',
  '/news-data/',
  '/rates/',
  '/rr-data/',
  '/sentiment-data/',
];

// ── Install: precache static shell ──────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_STATIC).then(cache => cache.addAll(STATIC_PRECACHE))
  );
  self.skipWaiting();
});

// ── Activate: delete old caches ─────────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(k => k !== CACHE_STATIC && k !== CACHE_DATA)
          .map(k => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// ── Fetch ────────────────────────────────────────────────────────────
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Only handle same-origin GET requests
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;

  const isData = DATA_PATH_PREFIXES.some(p => url.pathname.startsWith(p));

  if (isData) {
    // Network-first: fresh data preferred, cache as offline fallback
    event.respondWith(
      fetch(request)
        .then(response => {
          const clone = response.clone();
          caches.open(CACHE_DATA).then(cache => cache.put(request, clone));
          return response;
        })
        .catch(() => caches.match(request))
    );
  } else {
    // Cache-first: shell assets served instantly; stale-while-revalidate in bg
    event.respondWith(
      caches.match(request).then(cached => {
        const networkFetch = fetch(request).then(response => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_STATIC).then(cache => cache.put(request, clone));
          }
          return response;
        });
        return cached || networkFetch;
      })
    );
  }
});

// ── Notification click — open/focus the terminal ──────────────────
self.addEventListener('notificationclick', event => {
  event.notification.close();
  var targetUrl = (event.notification.data && event.notification.data.url)
    ? event.notification.data.url
    : '/';

  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(list => {
      // Focus existing tab if already open
      for (var i = 0; i < list.length; i++) {
        var c = list[i];
        if (c.url.includes('globalinvesting.github.io') && 'focus' in c) {
          return c.focus();
        }
      }
      // Otherwise open a new tab
      if (clients.openWindow) return clients.openWindow(targetUrl);
    })
  );
});
