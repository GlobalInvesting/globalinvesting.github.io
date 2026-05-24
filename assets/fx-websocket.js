/**
 * fx-websocket.js — Real-time FX tick feed via Cloudflare Worker proxy v1.0
 *
 * WHAT THIS DOES
 *   Connects to the globalinvesting-fx-ws-proxy Cloudflare Worker, which holds
 *   a single upstream WebSocket to Finnhub and fans ticks out to all browsers.
 *
 *   On each tick received:
 *   1. Updates STOOQ_RT_CACHE[pairId].close with the live mid-price
 *   2. Recalculates chg and pct using the prev_close already in the cache
 *   3. Calls updateFxPairsTableRT() to push the change to the DOM
 *   4. Updates the source label from "yfinance · HH:MM" to "Finnhub · live"
 *
 *   Everything else in the cache (hv30, session H/L, pct1w, prev_close, open)
 *   continues to come from quotes.json via the existing GitHub Actions pipeline.
 *   The WebSocket only updates close, chg, and pct — the fast-changing fields.
 *
 * FALLBACK
 *   If the Worker URL is not configured, the connection is refused, or the
 *   WebSocket drops and cannot reconnect after MAX_RECONNECT_ATTEMPTS, the
 *   module silently stops. The existing fetchQuoteBarRT() polling (every 60s)
 *   continues to work exactly as before — no user-visible degradation.
 *
 * CONFIGURATION
 *   Set FX_PROXY_WS_URL to the /ws endpoint of your deployed Worker.
 *   Example: "wss://globalinvesting-fx-ws-proxy.example.workers.dev/ws"
 *
 *   If this constant is empty or null, the module exits immediately.
 *
 * MARKET HOURS
 *   The proxy Worker is always available, but Finnhub stops sending FX ticks
 *   over the weekend. During those periods the WebSocket stays connected but
 *   silent — the source label remains "Finnhub · live" while no new prices
 *   arrive. The existing yfinance polling handles stale-price detection.
 *
 * HOW TO ENABLE
 *   1. Deploy cf-worker/fx-ws-proxy.js per the instructions in wrangler-fx-proxy.toml
 *   2. Set the FX_PROXY_WS_URL constant below to the Worker's /ws URL
 *   3. Uncomment the <script> tag in index.html (see comment near dashboard.js)
 *   4. No other changes needed — the module patches STOOQ_RT_CACHE in place
 */

// ── CONFIGURATION ─────────────────────────────────────────────────────────────
//
// Paste your Worker /ws URL here after deploying.
// Leave empty to disable the module entirely (falls back to yfinance polling).
//
const FX_PROXY_WS_URL = "wss://globalinvesting-fx-ws-proxy.globalinvestingmarkets.workers.dev/ws";   // e.g. "wss://globalinvesting-fx-ws-proxy.example.workers.dev/ws"

// Reconnection settings
const RECONNECT_DELAY_MS_BASE = 2_000;
const RECONNECT_DELAY_MS_MAX  = 60_000;
const MAX_RECONNECT_ATTEMPTS  = 10;     // then give up — yfinance polling takes over

// ── MODULE STATE ──────────────────────────────────────────────────────────────

let _ws                = null;
let _reconnectAttempts = 0;
let _reconnectTimer    = null;
let _lastTickTs        = 0;   // timestamp of last tick received (for label logic)
let _active            = false;

// ── ENTRY POINT ───────────────────────────────────────────────────────────────

/**
 * Call once from boot() after fetchQuoteBarRT() has populated STOOQ_RT_CACHE.
 * Safe to call multiple times — subsequent calls are no-ops if already connected.
 */
function initFxWebSocket() {
  if (_active) return;
  if (!FX_PROXY_WS_URL) {
    // Not configured — stay with yfinance polling
    return;
  }
  _active = true;
  _connect();
}

// ── WEBSOCKET LIFECYCLE ───────────────────────────────────────────────────────

function _connect() {
  if (_ws) return;  // already connecting or connected

  try {
    _ws = new WebSocket(FX_PROXY_WS_URL);
  } catch (err) {
    console.warn("[fx-ws] WebSocket constructor failed:", err.message);
    _scheduleReconnect();
    return;
  }

  _ws.addEventListener("open", () => {
    console.log("[fx-ws] Connected to FX proxy");
    _reconnectAttempts = 0;
  });

  _ws.addEventListener("message", event => {
    _handleMessage(event.data);
  });

  _ws.addEventListener("close", event => {
    console.warn(`[fx-ws] Connection closed (code ${event.code}) — will reconnect`);
    _ws = null;
    _scheduleReconnect();
  });

  _ws.addEventListener("error", () => {
    // error event always precedes close — let close handler retry
    _ws = null;
  });
}

function _scheduleReconnect() {
  if (!_active) return;
  if (_reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
    console.warn("[fx-ws] Max reconnect attempts reached — falling back to yfinance polling");
    _active = false;
    _updateSourceLabel(null);  // revert label to yfinance
    return;
  }

  const delay = Math.min(
    RECONNECT_DELAY_MS_BASE * Math.pow(2, _reconnectAttempts),
    RECONNECT_DELAY_MS_MAX
  );
  _reconnectAttempts++;

  clearTimeout(_reconnectTimer);
  _reconnectTimer = setTimeout(() => {
    if (_active && !_ws) _connect();
  }, delay);
}

// ── MESSAGE HANDLER ───────────────────────────────────────────────────────────

function _handleMessage(raw) {
  let msg;
  try { msg = JSON.parse(raw); } catch { return; }

  switch (msg.type) {
    case "tick":
      _applyTick(msg);
      break;

    case "subscribed":
      console.log(`[fx-ws] Subscribed — ${msg.pairs} FX pairs streaming`);
      break;

    case "connected":
      console.log(`[fx-ws] Proxy connected — ${msg.clients} client(s) total`);
      break;

    case "ping":
      // Heartbeat from the proxy — connection is alive, no action needed
      break;

    case "error":
      console.warn("[fx-ws] Proxy error:", msg.msg);
      break;
  }
}

// ── APPLY TICK TO CACHE AND DOM ───────────────────────────────────────────────

function _applyTick(msg) {
  // msg: { type, s: "eurusd", p: 1.08234, t: 1716500000000 }
  const pairId = msg.s;
  const price  = msg.p;

  if (!pairId || typeof price !== "number" || isNaN(price) || price <= 0) return;

  // STOOQ_RT_CACHE must already be populated by fetchQuoteBarRT()
  // If not yet ready, skip this tick — the cache will be populated shortly
  if (!window.STOOQ_RT_CACHE) return;

  const cached = window.STOOQ_RT_CACHE[pairId];

  if (!cached) {
    // Pair not yet in cache — create a minimal entry so the tick is visible
    window.STOOQ_RT_CACHE[pairId] = {
      close:     price,
      open:      price,
      prev_close: null,
      chg:       null,
      pct:       null,
      high:      null,
      low:       null,
      session_high: null,
      session_low:  null,
      hv30:      null,
      pct1w:     null,
      fromFinnhub: true,
    };
  } else {
    // Update only the price-sensitive fields — preserve everything else
    const prevClose = cached.prev_close;
    const chg  = prevClose != null ? (price - prevClose)           : null;
    const pct  = prevClose != null ? ((price / prevClose) - 1) * 100 : null;

    cached.close = price;
    if (chg !== null) cached.chg = chg;
    if (pct !== null) cached.pct = pct;
    cached.fromFinnhub = true;
  }

  _lastTickTs = Date.now();
  _updateSourceLabel(pairId);

  // Push the change to the DOM — reuse the exact same function that yfinance uses
  if (typeof updateFxPairsTableRT === "function") {
    updateFxPairsTableRT();
  }

  // Also update the quote bar price elements directly for immediate feedback
  _updateQuoteBarPriceElement(pairId, price);

  // Push to any open LW chart
  if (typeof _lwUpdateTodayBar === "function") {
    _lwUpdateTodayBar();
  }
}

// ── DOM UPDATES ───────────────────────────────────────────────────────────────

/**
 * Update the individual price/chg elements in the quote bar (header strip).
 * These are the same elements that fetchQuoteBarRT() updates — we just do it
 * immediately on each tick rather than waiting for the 60s polling cycle.
 */
function _updateQuoteBarPriceElement(pairId, price) {
  const pair = QB_STOOQ_PAIRS?.find(p => p.id === pairId);
  if (!pair) return;

  const priceEl = document.getElementById("q-" + pairId);
  const chgEl   = document.getElementById("qc-" + pairId);
  if (!priceEl && !chgEl) return;

  const cached = window.STOOQ_RT_CACHE?.[pairId];
  if (!cached) return;

  if (priceEl) {
    priceEl.textContent = price.toFixed(pair.dec);
    priceEl.className   = "q-price " + (typeof clsDir === "function" ? clsDir(cached.chg) : "");
  }
  if (chgEl) {
    chgEl.textContent = typeof pctStr === "function" ? pctStr(cached.pct) : "";
    chgEl.className   = "q-chg " + (typeof clsDir === "function" ? clsDir(cached.chg) : "");
  }
}

/**
 * Update the source label in the quote bar from "yfinance · HH:MM" to
 * "Finnhub · live" on first tick, and back to null (reverts to yfinance label)
 * when the WebSocket is no longer active.
 */
function _updateSourceLabel(pairId) {
  const qbLabel = document.getElementById("qb-source-label");
  if (!qbLabel) return;

  if (pairId === null) {
    // WebSocket gave up — let the next fetchQuoteBarRT() cycle rewrite the label
    return;
  }

  // Show "Finnhub · live" while WebSocket is delivering ticks
  qbLabel.textContent = "Finnhub · live";
}

// ── PUBLIC API ────────────────────────────────────────────────────────────────

// Expose initFxWebSocket globally so boot() in dashboard.js can call it
window.initFxWebSocket = initFxWebSocket;
