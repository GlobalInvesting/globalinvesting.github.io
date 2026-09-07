
const FX_PROXY_WS_URL = "wss://globalinvesting-fx-ws-proxy.globalinvestingmarkets.workers.dev/ws";   

const RECONNECT_DELAY_MS_BASE = 2_000;
const RECONNECT_DELAY_MS_MAX  = 60_000;
const MAX_RECONNECT_ATTEMPTS  = 10;     


let _ws                = null;
let _reconnectAttempts = 0;
let _reconnectTimer    = null;
let _lastTickTs        = 0;   
let _active            = false;
const TICK_STALE_MS = 120_000; 
let _staleCheckTimer   = null;
let _connectedAt       = 0;    


function initFxWebSocket() {
  if (_active) return;
  if (!FX_PROXY_WS_URL) {
    return;
  }
  _active = true;
  _connect();
}


function _connect() {
  if (_ws) return;  

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
    _connectedAt = Date.now();
    _startStaleWatchdog();
  });

  _ws.addEventListener("message", event => {
    _handleMessage(event.data);
  });

  _ws.addEventListener("close", event => {
    console.warn(`[fx-ws] Connection closed (code ${event.code}) — will reconnect`);
    _ws = null;
    _stopStaleWatchdog();
    _scheduleReconnect();
  });

  _ws.addEventListener("error", () => {
    _ws = null;
  });
}

function _scheduleReconnect() {
  if (!_active) return;
  if (_reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
    console.warn("[fx-ws] Max reconnect attempts reached — falling back to yfinance polling");
    _active = false;
    _updateSourceLabel(null);  
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

function _startStaleWatchdog() {
  _stopStaleWatchdog();
  _staleCheckTimer = setInterval(() => {
    const referenceMs = Math.max(_lastTickTs, _connectedAt);
    const staleMs = referenceMs ? Date.now() - referenceMs : Infinity;
    if (staleMs > TICK_STALE_MS) {
      console.warn(`[fx-ws] No tick received in ${Math.round(staleMs / 1000)}s — reverting label; yfinance polling remains authoritative`);
      _updateSourceLabel(null);
    }
  }, 30_000);
}

function _stopStaleWatchdog() {
  if (_staleCheckTimer) {
    clearInterval(_staleCheckTimer);
    _staleCheckTimer = null;
  }
}


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
      break;

    case "error":
      console.warn("[fx-ws] Proxy error:", msg.msg);
      break;
  }
}


function _applyTick(msg) {
  const pairId = msg.s;
  const price  = msg.p;

  if (!pairId || typeof price !== "number" || isNaN(price) || price <= 0) return;

  if (!window.STOOQ_RT_CACHE) return;

  const cached = window.STOOQ_RT_CACHE[pairId];

  if (!cached) {
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

  if (typeof updateFxPairsTableRT === "function") {
    updateFxPairsTableRT();
  }

  _updateQuoteBarPriceElement(pairId, price);

  if (typeof _lwUpdateTodayBar === "function") {
    _lwUpdateTodayBar();
  }
}


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

function _updateSourceLabel(pairId) {
  const qbLabel   = document.getElementById("qb-source-label");
  const delayChip = document.getElementById("footer-delay-label");

  if (pairId === null) {
    if (delayChip) {
      delayChip.textContent = "~5 MIN";
      delayChip.style.color = "var(--orange)";
    }
    return;
  }

  if (qbLabel)   qbLabel.textContent = "Live";
  if (delayChip) {
    delayChip.textContent = "LIVE";
    delayChip.style.color = "var(--up)";
  }
}


window.initFxWebSocket = initFxWebSocket;
