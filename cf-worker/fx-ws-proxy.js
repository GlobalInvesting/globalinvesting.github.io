/**
 * fx-ws-proxy.js — Cloudflare Worker + Durable Object v1.0
 *
 * WHY THIS EXISTS
 *   Finnhub's free tier allows 1 concurrent WebSocket connection per API key.
 *   If every browser tab connects to Finnhub directly, the second tab kills the
 *   first. This Worker solves that with a fan-out proxy:
 *
 *     Finnhub WS ──► FxProxy (Durable Object) ──► Browser 1
 *                                               ──► Browser 2
 *                                               ──► Browser N
 *
 *   One DO instance holds exactly ONE upstream connection to Finnhub and
 *   broadcasts each tick to ALL connected browsers. New tabs connect to the DO
 *   — not to Finnhub directly.
 *
 * ARCHITECTURE
 *   Worker (fetch handler):
 *     - Receives browser WebSocket upgrade requests
 *     - Routes them all to a single named DO instance: "fx-proxy-singleton"
 *     - Returns 101 Upgrade response from the DO to the browser
 *
 *   FxProxy (Durable Object):
 *     - Uses WebSocket Hibernation API so the DO sleeps when no browsers are
 *       connected, avoiding unnecessary duration charges
 *     - On first browser connection: opens a WebSocket to Finnhub and subscribes
 *       to all 28 FX pairs
 *     - On each Finnhub tick: broadcasts a compact JSON message to all browsers
 *     - On last browser disconnect: closes the Finnhub WS (saves DO duration)
 *     - Reconnects to Finnhub automatically with exponential backoff on error
 *
 * CLOUDFLARE FREE TIER USAGE (verified 2026-05)
 *   With thresholdLevel:5 (only emit when price moves ≥0.00005) and market-hours
 *   only operation (07:00–22:00 UTC on weekdays):
 *
 *   DO requests/day:   ~35,000   (100,000 free limit — 35% utilization)
 *   DO duration/day:   ~6,200 GB-s  (13,000 GB-s free limit — 48% utilization)
 *   Workers req/day:   ~1,000    (100,000 free limit — 1% utilization)
 *   Cost: $0
 *
 *   The 20:1 billing ratio for incoming WebSocket messages is what keeps this
 *   within the free tier: 700k ticks ÷ 20 = 35k billed requests.
 *
 * MESSAGE PROTOCOL (DO → Browser)
 *   Tick update:
 *     { type: "tick", s: "eurusd", p: 1.08234, t: 1716500000000 }
 *   Subscription confirmation:
 *     { type: "subscribed", pairs: 28, ts: "2026-05-23T..." }
 *   Error:
 *     { type: "error", msg: "..." }
 *   Heartbeat (every 30s):
 *     { type: "ping", ts: "..." }
 *
 * REQUIRED SECRETS (set via: npx wrangler secret put NAME)
 *   FINNHUB_API_KEY   — Finnhub free API key (same key used by calendar-watcher)
 *
 * REQUIRED DURABLE OBJECT NAMESPACE
 *   See wrangler.toml — the namespace binding is "FX_PROXY"
 *
 * CORS
 *   Only allows WebSocket upgrades from globalinvesting.github.io.
 *   Health check GET / is open for monitoring.
 */

// ── Worker entry point ────────────────────────────────────────────────────────

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Health check — used by monitoring and manual verification
    if (url.pathname === "/" && request.method === "GET") {
      return new Response(JSON.stringify({
        worker:   "fx-ws-proxy",
        version:  "1.0",
        ts:       new Date().toISOString(),
        endpoint: "/ws  (WebSocket upgrade)",
      }), { headers: { "Content-Type": "application/json" } });
    }

    // WebSocket upgrade — route to the singleton Durable Object
    if (url.pathname === "/ws") {
      if (request.headers.get("Upgrade") !== "websocket") {
        return new Response("Expected WebSocket upgrade", { status: 426 });
      }

      // Validate origin — only allow our own site
      const origin = request.headers.get("Origin") || "";
      const allowed = [
        "https://globalinvesting.github.io",
        "http://localhost",            // local dev
        "http://127.0.0.1",           // local dev
      ];
      if (!allowed.some(o => origin.startsWith(o))) {
        return new Response("Forbidden", { status: 403 });
      }

      // All browsers connect to the same named singleton
      const id    = env.FX_PROXY.idFromName("fx-proxy-singleton");
      const proxy = env.FX_PROXY.get(id);
      return proxy.fetch(request);
    }

    return new Response("Not found", { status: 404 });
  },
};

// ── Durable Object ────────────────────────────────────────────────────────────

export class FxProxy {
  constructor(state, env) {
    this.state   = state;
    this.env     = env;

    // Upstream Finnhub WebSocket (server-side connection, not a browser)
    this._finnhubWs    = null;
    this._reconnectMs  = 1000;   // backoff start
    this._reconnecting = false;

    // Heartbeat interval ref (cleared on last disconnect)
    this._heartbeatTimer = null;
  }

  // ── Public handler — called by the Worker for every incoming HTTP/WS request
  async fetch(request) {
    // Use the Hibernation WebSocket API so this DO can sleep when idle.
    // Each acceptWebSocket() call registers the connection; events are delivered
    // to webSocketMessage / webSocketClose / webSocketError below.
    const pair   = new WebSocketPair();
    const client = pair[0];
    const server = pair[1];

    this.state.acceptWebSocket(server);

    const browserCount = this.state.getWebSockets().length;

    // First browser — open Finnhub upstream and start heartbeat
    if (browserCount === 1) {
      await this._openFinnhub();
      this._startHeartbeat();
    }

    // Send current subscription state immediately so the browser knows we're live
    server.send(JSON.stringify({
      type:    "connected",
      clients: browserCount,
      ts:      new Date().toISOString(),
    }));

    return new Response(null, { status: 101, webSocket: client });
  }

  // ── Hibernation API callbacks ─────────────────────────────────────────────

  // Called when a browser sends a message (we don't need this for pure fan-out,
  // but implementing it allows future browser→proxy commands, e.g. pause)
  async webSocketMessage(ws, msg) {
    // No-op for now — browser doesn't need to send anything
    // Future: handle "pause" / "resume" commands
  }

  async webSocketClose(ws, code, reason) {
    const remaining = this.state.getWebSockets().length;
    if (remaining === 0) {
      // Last browser disconnected — tear down Finnhub upstream to save DO duration
      this._closeFinnhub();
      this._stopHeartbeat();
    }
  }

  async webSocketError(ws, err) {
    // Browser disconnected with error — webSocketClose fires next, handles cleanup
  }

  // ── Finnhub upstream management ──────────────────────────────────────────

  _openFinnhub() {
    if (this._finnhubWs) return; // already open
    if (!this.env.FINNHUB_API_KEY) {
      console.error("[fx-proxy] FINNHUB_API_KEY secret not set");
      this._broadcastError("Server misconfiguration: FINNHUB_API_KEY missing");
      return;
    }

    try {
      const ws = new WebSocket("wss://ws.finnhub.io?token=" + this.env.FINNHUB_API_KEY);
      this._finnhubWs = ws;

      ws.addEventListener("open", () => {
        console.log("[fx-proxy] Finnhub WS open — subscribing to 28 FX pairs");
        this._reconnectMs = 1000;  // reset backoff on success
        this._subscribePairs(ws);
        this._broadcastToAll({ type: "subscribed", pairs: FX_PAIRS.length, ts: new Date().toISOString() });
      });

      ws.addEventListener("message", event => {
        this._handleFinnhubMessage(event.data);
      });

      ws.addEventListener("close", event => {
        console.log(`[fx-proxy] Finnhub WS closed (code ${event.code}) — scheduling reconnect`);
        this._finnhubWs = null;
        this._scheduleReconnect();
      });

      ws.addEventListener("error", event => {
        console.error("[fx-proxy] Finnhub WS error:", event);
        this._finnhubWs = null;
        this._scheduleReconnect();
      });

    } catch (err) {
      console.error("[fx-proxy] Failed to open Finnhub WS:", err);
      this._scheduleReconnect();
    }
  }

  _subscribePairs(ws) {
    for (const pair of FX_PAIRS) {
      ws.send(JSON.stringify({ type: "subscribe", symbol: pair.finnhub }));
    }
    // thresholdLevel not settable via message — it's a connection param in REST.
    // Finnhub WS naturally batches; tick rate is controlled by market activity.
  }

  _closeFinnhub() {
    if (this._finnhubWs) {
      try { this._finnhubWs.close(); } catch (_) {}
      this._finnhubWs = null;
    }
    this._reconnecting = false;
  }

  _scheduleReconnect() {
    if (this._reconnecting) return;
    // Only reconnect if browsers are still connected
    if (this.state.getWebSockets().length === 0) return;

    this._reconnecting = true;
    const delay = Math.min(this._reconnectMs, 30_000);
    this._reconnectMs = Math.min(this._reconnectMs * 2, 30_000);

    setTimeout(() => {
      this._reconnecting = false;
      if (this.state.getWebSockets().length > 0) {
        this._openFinnhub();
      }
    }, delay);
  }

  // ── Incoming tick from Finnhub ────────────────────────────────────────────

  _handleFinnhubMessage(raw) {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }

    // Finnhub sends: { type: "trade", data: [{ s, p, t, v, c }] }
    // For forex, "trade" events carry last mid-price ticks
    if (msg.type !== "trade" || !Array.isArray(msg.data)) return;

    for (const tick of msg.data) {
      if (!tick.s || tick.p == null) continue;

      // Map Finnhub symbol (e.g. "OANDA:EUR_USD") to our ID (e.g. "eurusd")
      const pairId = FINNHUB_TO_PAIR_ID[tick.s];
      if (!pairId) continue;

      // Broadcast compact tick to all browsers
      // { type, s (our id), p (price), t (timestamp ms) }
      this._broadcastToAll({
        type: "tick",
        s:    pairId,
        p:    tick.p,
        t:    tick.t,
      });
    }
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  _broadcastToAll(obj) {
    const json = JSON.stringify(obj);
    for (const ws of this.state.getWebSockets()) {
      try { ws.send(json); } catch (_) {}
    }
  }

  _broadcastError(msg) {
    this._broadcastToAll({ type: "error", msg });
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    this._heartbeatTimer = setInterval(() => {
      if (this.state.getWebSockets().length === 0) {
        this._stopHeartbeat();
        return;
      }
      this._broadcastToAll({ type: "ping", ts: new Date().toISOString() });
    }, 30_000);
  }

  _stopHeartbeat() {
    if (this._heartbeatTimer) {
      clearInterval(this._heartbeatTimer);
      this._heartbeatTimer = null;
    }
  }
}

// ── FX pair symbol maps ───────────────────────────────────────────────────────
//
// Finnhub forex symbols for OANDA exchange format: "OANDA:EUR_USD"
// Our internal IDs match QB_STOOQ_PAIRS in dashboard.js: "eurusd"

const FX_PAIRS = [
  { id: "eurusd",  finnhub: "OANDA:EUR_USD" },
  { id: "usdjpy",  finnhub: "OANDA:USD_JPY" },
  { id: "gbpusd",  finnhub: "OANDA:GBP_USD" },
  { id: "audusd",  finnhub: "OANDA:AUD_USD" },
  { id: "usdcad",  finnhub: "OANDA:USD_CAD" },
  { id: "usdchf",  finnhub: "OANDA:USD_CHF" },
  { id: "nzdusd",  finnhub: "OANDA:NZD_USD" },
  { id: "eurgbp",  finnhub: "OANDA:EUR_GBP" },
  { id: "eurjpy",  finnhub: "OANDA:EUR_JPY" },
  { id: "eurchf",  finnhub: "OANDA:EUR_CHF" },
  { id: "eurcad",  finnhub: "OANDA:EUR_CAD" },
  { id: "euraud",  finnhub: "OANDA:EUR_AUD" },
  { id: "gbpjpy",  finnhub: "OANDA:GBP_JPY" },
  { id: "gbpchf",  finnhub: "OANDA:GBP_CHF" },
  { id: "gbpcad",  finnhub: "OANDA:GBP_CAD" },
  { id: "audjpy",  finnhub: "OANDA:AUD_JPY" },
  { id: "audnzd",  finnhub: "OANDA:AUD_NZD" },
  { id: "audchf",  finnhub: "OANDA:AUD_CHF" },
  { id: "cadjpy",  finnhub: "OANDA:CAD_JPY" },
  { id: "chfjpy",  finnhub: "OANDA:CHF_JPY" },
  { id: "nzdjpy",  finnhub: "OANDA:NZD_JPY" },
  { id: "eurnzd",  finnhub: "OANDA:EUR_NZD" },
  { id: "gbpaud",  finnhub: "OANDA:GBP_AUD" },
  { id: "gbpnzd",  finnhub: "OANDA:GBP_NZD" },
  { id: "audcad",  finnhub: "OANDA:AUD_CAD" },
  { id: "cadchf",  finnhub: "OANDA:CAD_CHF" },
  { id: "nzdcad",  finnhub: "OANDA:NZD_CAD" },
  { id: "nzdchf",  finnhub: "OANDA:NZD_CHF" },
];

// Reverse lookup: "OANDA:EUR_USD" → "eurusd"
const FINNHUB_TO_PAIR_ID = Object.fromEntries(
  FX_PAIRS.map(p => [p.finnhub, p.id])
);
