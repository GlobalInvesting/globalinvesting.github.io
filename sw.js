// ═══════════════════════════════════════════════════════════════════
// sw.js — Global Investing FX Terminal Service Worker
// Strategy:
//   • index.html            → Network-first (always fresh entry point)
//   • Static shell (CSS, JS, icons) → Cache-first, update in bg
//   • JSON data endpoints   → Network-first, cache as fallback
//   • Everything else       → Network only
//
// VERSIONING: bump CACHE_VERSION on every deploy that changes static
// assets. The activate handler deletes all old-versioned caches so
// users always get fresh files after the next page load.
// ═══════════════════════════════════════════════════════════════════

const CACHE_VERSION = 'gi-v8.261.2';
const CACHE_STATIC  = `${CACHE_VERSION}-static`;
const CACHE_DATA    = `${CACHE_VERSION}-data`;

// Core shell files cached on install.
// NOTE: index.html is intentionally excluded — it is handled via
// network-first so the browser always gets the latest entry point
// (and therefore the latest asset query-string versions).
//
// KEEP IN SYNC WITH index.html ON EVERY DEPLOY. This list was stuck at
// v8.21.0 for many releases, including a filename (`dashboard-v2.css`)
// that had since been renamed to `dashboard.css` — cache.addAll() is
// all-or-nothing, so that one 404 silently failed the ENTIRE install()
// every time, and because CACHE_VERSION never changed either, the
// activate handler never had a version bump to trigger deleting
// whatever static cache HAD successfully installed the last time this
// worker's install() actually succeeded — i.e. any returning client
// could still be served that old cached shell indefinitely, however
// many versions ago it was. Bumping CACHE_VERSION here forces every
// client to drop old caches on next activation regardless of the exact
// prior failure mode.
const STATIC_PRECACHE = [
  '/assets/dashboard.css?v=8.191.0',
  '/assets/dashboard.js?v=8.261.2',
  '/assets/gi-auth.js?v=1.7.5',
  '/assets/gi-overview.js?v=1.4.1',
  '/assets/fx-websocket.js?v=1.0.0',
  '/assets/cot-modal-chart.js?v=7.92.0',
  '/assets/cb-rates-modal.js?v=8.0.6',
  '/assets/real-carry-modal.js?v=2.7.9',
  '/assets/corr-modal.js?v=2.6.0',
  '/assets/yc-modal.js?v=8.8.5',
  '/assets/heatmap-modal.js?v=2.6.4',
  '/assets/econ-surprises-modal.js?v=1.3.10',
  '/assets/onboarding.js?v=7.89.11',
  '/assets/layout-resizer.js?v=1.0.0',
  '/assets/feed.js?v=1.0.0',
  '/assets/share.js?v=1.0.0',
  '/assets/inline-panel.js?v=1.4.2',
  '/assets/calendar-panel.js?v=1.19.20',
  '/assets/econ-matrix.js?v=2.5.2',
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
  '/calendar-data/',
  '/cot-data/',
  '/dtcc-data/',
  '/economic-data/',
  '/extended-data/',
  '/fx-data/',
  '/growth-differential-data/',
  '/intraday-data/',
  '/meetings-data/',
  '/news-data/',
  '/ohlc-data/',
  '/rates/',
  '/research-data/',
  '/rr-data/',
  '/seasonality-data/',
  '/sentiment-data/',
];

// ── Install: precache static shell ──────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_STATIC).then(cache => cache.addAll(STATIC_PRECACHE))
  );
  self.skipWaiting();
});

// ── Activate: delete all caches from previous versions ──────────────
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

  // index.html: always network-first so deploys are picked up immediately
  const isEntryPoint = url.pathname === '/' || url.pathname === '/index.html';

  if (isEntryPoint || isData) {
    // Network-first: fresh content preferred, cache as offline fallback
    event.respondWith(
      fetch(request)
        .then(response => {
          if (response.ok) {
            const clone = response.clone();
            const cacheName = isData ? CACHE_DATA : CACHE_STATIC;
            caches.open(cacheName).then(cache => cache.put(request, clone));
          }
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
        }).catch(() => {});
        return cached || networkFetch;
      })
    );
  }
});

// ── Push — COT Friday notifications ──────────────────────────────
self.addEventListener('push', event => {
  var data = {};
  try { data = event.data ? event.data.json() : {}; } catch (e) { /* ignore */ }

  var title   = data.title   || 'COT Report Updated';
  var body    = data.body    || 'CFTC data for GBP, EUR, JPY & AUD is now live.';
  var url     = data.url     || '/';
  var icon    = data.icon    || '/favicon-192x192.png';
  var badge   = data.badge   || '/favicon-32x32.png';

  event.waitUntil(
    self.registration.showNotification(title, {
      body:  body,
      icon:  icon,
      badge: badge,
      tag:   'cot-update',
      renotify: false,
      data:  { url: url }
    })
  );
});

// ── Notification click — open/focus the terminal ──────────────────
self.addEventListener('notificationclick', event => {
  event.notification.close();
  var targetUrl = (event.notification.data && event.notification.data.url)
    ? event.notification.data.url
    : '/';

  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(list => {
      for (var i = 0; i < list.length; i++) {
        var c = list[i];
        if (c.url.includes('globalinvesting.github.io') && 'focus' in c) {
          return c.focus();
        }
      }
      if (clients.openWindow) return clients.openWindow(targetUrl);
    })
  );
});
