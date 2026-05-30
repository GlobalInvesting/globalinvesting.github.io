/**
 * calendar-watcher.js — Cloudflare Worker v1.0
 *
 * Polls Finnhub every 1 minute for major-economy economic calendar actuals/forecasts.
 * When new data is detected, fires a repository_dispatch event to GitHub Actions,
 * which triggers the full fetch_ff_calendar.py pipeline.
 *
 * WHY THIS EXISTS
 *   GitHub Actions minimum cron interval is 5 minutes, with 0–15 min runner
 *   queue delay. During active economic release windows (NFP, CPI, etc.) this
 *   means a 5–20 minute lag between Finnhub publishing an actual and the
 *   terminal showing it.
 *
 *   Cloudflare Workers support 1-minute cron triggers on the free tier.
 *   By polling Finnhub every minute and only triggering GitHub when real data
 *   changes, we achieve ~2–3 min end-to-end latency:
 *     Finnhub publishes actual
 *     → CF Worker detects on next 1-min poll  (~0–60 sec)
 *     → repository_dispatch fires             (~1 sec)
 *     → GitHub runner starts                  (~15–30 sec)
 *     → fetch_ff_calendar.py runs             (~45 sec)
 *     → git push + Pages rebuild              (~30 sec)
 *     TOTAL:  ~2–3 minutes
 *
 *   The GitHub Actions cron (*/5) remains as a fallback: if the Worker fails
 *   or CF has an outage, the pipeline still runs within 5–15 minutes.
 *
 * CLOUDFLARE FREE TIER USAGE
 *   Worker invocations:  1,440/day   (<<< 100,000/day limit)
 *   KV reads:            1,440/day   (<<< 100,000/day limit)
 *   KV writes:           ~10–50/day  (<<< 1,000/day limit)
 *   repository_dispatch: ~10–50/day  (no GitHub documented limit)
 *   Cost: $0
 *
 * REQUIRED SECRETS (set via wrangler CLI or CF dashboard):
 *   FINNHUB_API_KEY   — Finnhub free API key (finnhub.io)
 *   GITHUB_PAT        — GitHub Personal Access Token (scope: public_repo)
 *   GITHUB_OWNER      — e.g. "globalinvesting"
 *   GITHUB_REPO       — e.g. "globalinvesting.github.io"
 *
 * REQUIRED KV NAMESPACE:
 *   CALENDAR_KV       — bound in wrangler.toml as [[kv_namespaces]]
 *
 * CRON SCHEDULE:
 *   "* * * * *"       — every 1 minute (CF Workers minimum)
 *
 * SETUP INSTRUCTIONS: see cf-worker/README.md
 */

// ── Constants ─────────────────────────────────────────────────────────────────

const G8_COUNTRIES = new Set(["US", "EU", "GB", "JP", "AU", "CA", "CH", "NZ"]);
const COUNTRY_TO_CCY = {
  US: "USD", EU: "EUR", GB: "GBP", JP: "JPY",
  AU: "AUD", CA: "CAD", CH: "CHF", NZ: "NZD",
};

// Look ahead 2 days and back 3 days to catch today's releases
const FETCH_PAST_DAYS  = 3;
const FETCH_FUTURE_DAYS = 2;

// KV key for the last known fingerprint
const KV_FINGERPRINT_KEY = "calendar:fingerprint:v1";

// ── Entry point ───────────────────────────────────────────────────────────────

export default {
  /**
   * Scheduled handler — fires every 1 minute via cron trigger.
   */
  async scheduled(event, env, ctx) {
    ctx.waitUntil(runCalendarWatch(env));
  },

  /**
   * HTTP handler — allows manual trigger via GET /trigger and health checks via GET /.
   * Useful for testing without waiting for the cron.
   */
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (url.pathname === "/trigger") {
      ctx.waitUntil(runCalendarWatch(env));
      return new Response(JSON.stringify({ status: "triggered", ts: new Date().toISOString() }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/fingerprint") {
      const fp = await env.CALENDAR_KV.get(KV_FINGERPRINT_KEY);
      return new Response(JSON.stringify({ fingerprint: fp, ts: new Date().toISOString() }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/reset") {
      await env.CALENDAR_KV.delete(KV_FINGERPRINT_KEY);
      return new Response(JSON.stringify({ status: "fingerprint reset", ts: new Date().toISOString() }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(JSON.stringify({
      worker: "calendar-watcher",
      version: "1.0",
      endpoints: ["/trigger", "/fingerprint", "/reset"],
      ts: new Date().toISOString(),
    }), { headers: { "Content-Type": "application/json" } });
  },
};

// ── Core logic ────────────────────────────────────────────────────────────────

async function runCalendarWatch(env) {
  const now = new Date();
  const label = `[${now.toISOString().slice(0, 16)}Z]`;
  console.log(`${label} calendar-watcher: starting poll`);

  // Step 1: Build date range (narrow window — just enough to catch today's releases)
  const dateFrom = fmtDate(addDays(now, -FETCH_PAST_DAYS));
  const dateTo   = fmtDate(addDays(now, FETCH_FUTURE_DAYS));

  // Step 2: Fetch from Finnhub
  let events;
  try {
    events = await fetchFinnhub(dateFrom, dateTo, env.FINNHUB_API_KEY);
  } catch (err) {
    console.error(`${label} Finnhub fetch failed: ${err.message}`);
    return; // do not trigger GitHub — wait for next poll
  }

  if (!events || events.length === 0) {
    console.log(`${label} No 8 major currencies events returned from Finnhub — skipping.`);
    return;
  }

  // Step 3: Compute fingerprint of actuals + forecasts
  const newFingerprint = buildFingerprint(events);

  // Step 4: Compare with stored fingerprint
  const prevFingerprint = await env.CALENDAR_KV.get(KV_FINGERPRINT_KEY) ?? "";

  if (newFingerprint === prevFingerprint) {
    console.log(`${label} No change detected — skipping repository_dispatch.`);
    return;
  }

  // Step 5: Data changed — store new fingerprint
  const diff = describeDiff(prevFingerprint, newFingerprint, events);
  console.log(`${label} CHANGE DETECTED: ${diff}`);

  await env.CALENDAR_KV.put(KV_FINGERPRINT_KEY, newFingerprint, {
    expirationTtl: 60 * 60 * 24 * 7, // expire after 7 days (auto-cleanup)
  });

  // Step 6: Fire repository_dispatch to GitHub Actions
  try {
    await triggerGitHubWorkflow(env, diff);
    console.log(`${label} repository_dispatch sent — GitHub Actions pipeline triggered.`);
  } catch (err) {
    console.error(`${label} repository_dispatch failed: ${err.message}`);
    // Roll back fingerprint so next poll retries the dispatch
    await env.CALENDAR_KV.put(KV_FINGERPRINT_KEY, prevFingerprint, {
      expirationTtl: 60 * 60 * 24 * 7,
    });
  }
}

// ── Finnhub fetch ─────────────────────────────────────────────────────────────

async function fetchFinnhub(dateFrom, dateTo, apiKey) {
  if (!apiKey) throw new Error("FINNHUB_API_KEY secret not set");

  const url = `https://finnhub.io/api/v1/calendar/economic?from=${dateFrom}&to=${dateTo}&token=${apiKey}`;
  const resp = await fetch(url, {
    headers: { "User-Agent": "globalinvesting-calendar-watcher/1.0" },
    signal: AbortSignal.timeout(8000), // 8s timeout
  });

  if (resp.status === 429) {
    throw new Error("Finnhub rate limit (429) — will retry next minute");
  }
  if (resp.status === 401) {
    throw new Error("Finnhub auth failed (401) — check FINNHUB_API_KEY secret");
  }
  if (!resp.ok) {
    throw new Error(`Finnhub HTTP ${resp.status}`);
  }

  const data = await resp.json();
  const raw = Array.isArray(data?.economicCalendar) ? data.economicCalendar : [];

  // Filter to major-economy medium+high impact only, with actual or forecast
  const events = [];
  for (const ev of raw) {
    const iso2 = (ev.country || "").toUpperCase();
    if (!G8_COUNTRIES.has(iso2)) continue;

    const impact = (ev.impact || "low").toLowerCase();
    if (impact === "low") continue;

    const actual   = cleanVal(ev.actual);
    const forecast = cleanVal(ev.estimate);

    // Only include events with actual OR forecast (future events may only have forecast)
    if (actual === null && forecast === null) continue;

    const dt = parseISODate(ev.time || "");
    if (!dt) continue;

    events.push({
      title:    (ev.event || "").trim(),
      currency: COUNTRY_TO_CCY[iso2],
      dateISO:  fmtDate(dt),
      actual,
      forecast,
      released: actual !== null,
    });
  }

  return events;
}

// ── Fingerprint ───────────────────────────────────────────────────────────────

/**
 * Build a compact, deterministic fingerprint of all actuals and forecasts.
 * Sorted for stability regardless of Finnhub response order.
 * Format: "YYYY-MM-DD|CCY|title|actual|forecast" per line, joined with \n.
 */
function buildFingerprint(events) {
  const lines = events
    .map(ev => `${ev.dateISO}|${ev.currency}|${ev.title}|${ev.actual ?? ""}|${ev.forecast ?? ""}`)
    .sort();
  return lines.join("\n");
}

/**
 * Produce a human-readable diff summary for the commit message / log.
 */
function describeDiff(prevFP, newFP, events) {
  const prevLines = new Set(prevFP ? prevFP.split("\n") : []);
  const newLines  = new Set(newFP.split("\n"));

  const added = [...newLines].filter(l => !prevLines.has(l));
  const newActuals   = added.filter(l => l.split("|")[3] !== "");
  const newForecasts = added.filter(l => l.split("|")[3] === "" && l.split("|")[4] !== "");

  const parts = [];
  if (newActuals.length > 0) {
    const samples = newActuals.slice(0, 2)
      .map(l => { const p = l.split("|"); return `${p[1]} ${p[2].slice(0, 30)}=${p[3]}`; });
    parts.push(`${newActuals.length} new actual(s): ${samples.join(", ")}`);
  }
  if (newForecasts.length > 0) {
    parts.push(`${newForecasts.length} new forecast(s)`);
  }
  return parts.join(" | ") || `${added.length} data point(s) changed`;
}

// ── GitHub repository_dispatch ────────────────────────────────────────────────

async function triggerGitHubWorkflow(env, description) {
  const { GITHUB_PAT, GITHUB_OWNER, GITHUB_REPO } = env;

  if (!GITHUB_PAT)    throw new Error("GITHUB_PAT secret not set");
  if (!GITHUB_OWNER)  throw new Error("GITHUB_OWNER secret not set");
  if (!GITHUB_REPO)   throw new Error("GITHUB_REPO secret not set");

  const url = `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/dispatches`;
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${GITHUB_PAT}`,
      "Accept":        "application/vnd.github+json",
      "Content-Type":  "application/json",
      "User-Agent":    "globalinvesting-calendar-watcher/1.0",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    body: JSON.stringify({
      event_type: "calendar-data-changed",
      client_payload: {
        description,
        triggered_at: new Date().toISOString(),
        source: "cloudflare-worker",
      },
    }),
    signal: AbortSignal.timeout(10000),
  });

  // GitHub returns 204 No Content on success
  if (resp.status !== 204) {
    const body = await resp.text().catch(() => "");
    throw new Error(`GitHub API returned ${resp.status}: ${body}`);
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function cleanVal(v) {
  if (v == null) return null;
  const s = String(v).trim();
  return s === "" || s === "None" || s === "null" || s === "N/A" || s === "—" ? null : s;
}

function fmtDate(d) {
  return d.toISOString().slice(0, 10);
}

function addDays(d, n) {
  const r = new Date(d);
  r.setUTCDate(r.getUTCDate() + n);
  return r;
}

function parseISODate(s) {
  try {
    const d = new Date(s.replace("Z", "+00:00"));
    return isNaN(d.getTime()) ? null : d;
  } catch {
    return null;
  }
}
