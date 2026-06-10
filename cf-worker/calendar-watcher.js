/**
 * calendar-watcher.js — Cloudflare Worker v2.0
 *
 * Polls ForexFactory public JSON every 6 minutes for G8 economic calendar
 * actuals/forecasts. When new data is detected, fires a repository_dispatch
 * event to GitHub Actions, which triggers the full fetch_ff_calendar.py pipeline.
 *
 * MIGRATION FROM v1.0 (Finnhub → ForexFactory public JSON)
 *   v1.0 used Finnhub /api/v1/calendar/economic, which requires a paid plan.
 *   The free tier returns HTTP 403 for that endpoint. This version migrates to
 *   https://nfs.faireconomy.media/ff_calendar_thisweek.json — the same public
 *   ForexFactory JSON feed already used by fetch_ff_calendar.py for holidays.
 *   This feed includes actual, forecast, and previous fields for all events.
 *
 * WHY 6 MINUTES (not 1 minute)
 *   ForexFactory rate-limits the weekly JSON exports to 2 requests per 5 minutes
 *   per IP (enforced since August 2024). At 1-minute polling we would exceed this
 *   immediately and receive "Request Denied" HTML instead of JSON. At 6 minutes
 *   we make exactly 1 request per cycle — well within the limit.
 *   CF Workers minimum cron is 1 minute; cron expression "* /6" (no space)
 *   fires at :00, :06, :12, :18, :24, :30, :36, :42, :48, :54 each hour.
 *
 * END-TO-END LATENCY
 *   ForexFactory publishes actual
 *   → CF Worker detects on next 6-min poll     (~0–6 min)
 *   → repository_dispatch fires                (~1 sec)
 *   → GitHub runner starts                     (~15–30 sec)
 *   → fetch_ff_calendar.py runs                (~45 sec)
 *   → git push + Pages rebuild                 (~30 sec)
 *   TOTAL:  ~2–8 minutes
 *
 *   The GitHub Actions cron (every 5 min) remains as a fallback: if the Worker fails
 *   or CF has an outage, the pipeline still runs within 5–15 minutes.
 *
 * FOREXFACTORY JSON FIELDS
 *   title     — event name
 *   country   — G8 currency code (USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD)
 *   date      — ISO datetime string (ET timezone)
 *   impact    — "High" | "Medium" | "Low" | "Holiday"
 *   actual    — string or "" (populated in real-time after release)
 *   forecast  — string or "" (consensus estimate)
 *   previous  — string or "" (prior period value)
 *
 * CLOUDFLARE FREE TIER USAGE
 *   Worker invocations:  240/day    (<<< 100,000/day limit)
 *   KV reads:            240/day    (<<< 100,000/day limit)
 *   KV writes:           ~5–20/day  (<<< 1,000/day limit)
 *   repository_dispatch: ~5–20/day  (no GitHub documented limit)
 *
 * REQUIRED SECRETS (FINNHUB_API_KEY no longer needed):
 *   GITHUB_PAT        — GitHub Personal Access Token (scope: public_repo)
 *   GITHUB_OWNER      — e.g. "globalinvesting"
 *   GITHUB_REPO       — e.g. "globalinvesting.github.io"
 *
 * REQUIRED KV NAMESPACE:
 *   CALENDAR_KV       — bound in wrangler.toml as [[kv_namespaces]]
 *
 * CRON SCHEDULE:
 *   Cron expression: every-6-minutes ("star-slash-6 star star star star")
 */

// ── Constants ─────────────────────────────────────────────────────────────────

const G8_CURRENCIES = new Set(["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]);

// ForexFactory public JSON — no API key required
const FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json";

// KV key for the last known fingerprint
const KV_FINGERPRINT_KEY = "calendar:fingerprint:v2";

// ── Entry point ───────────────────────────────────────────────────────────────

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(runCalendarWatch(env));
  },

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
      version: "2.0",
      source: "ForexFactory public JSON (nfs.faireconomy.media)",
      schedule: "every 6 minutes",
      endpoints: ["/trigger", "/fingerprint", "/reset"],
      ts: new Date().toISOString(),
    }), { headers: { "Content-Type": "application/json" } });
  },
};

// ── Core logic ────────────────────────────────────────────────────────────────

async function runCalendarWatch(env) {
  const now = new Date();
  const label = `[${now.toISOString().slice(0, 16)}Z]`;
  console.log(`${label} calendar-watcher v2.0: starting poll (source: ForexFactory)`);

  // Step 1: Fetch from ForexFactory public JSON
  let events;
  try {
    events = await fetchForexFactory();
  } catch (err) {
    console.error(`${label} ForexFactory fetch failed: ${err.message}`);
    return;
  }

  if (!events || events.length === 0) {
    console.log(`${label} No G8 medium/high events returned — skipping.`);
    return;
  }

  console.log(`${label} Fetched ${events.length} G8 medium/high events`);

  // Step 2: Compute fingerprint of actuals + forecasts
  const newFingerprint = buildFingerprint(events);

  // Step 3: Compare with stored fingerprint
  const prevFingerprint = await env.CALENDAR_KV.get(KV_FINGERPRINT_KEY) ?? "";

  if (newFingerprint === prevFingerprint) {
    console.log(`${label} No change detected — skipping repository_dispatch.`);
    return;
  }

  // Step 4: Data changed — store new fingerprint
  const diff = describeDiff(prevFingerprint, newFingerprint);
  console.log(`${label} CHANGE DETECTED: ${diff}`);

  await env.CALENDAR_KV.put(KV_FINGERPRINT_KEY, newFingerprint, {
    expirationTtl: 60 * 60 * 24 * 7,
  });

  // Step 5: Fire repository_dispatch to GitHub Actions
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

// ── ForexFactory fetch ────────────────────────────────────────────────────────

async function fetchForexFactory() {
  const resp = await fetch(FF_URL, {
    headers: {
      "User-Agent": "globalinvesting-calendar-watcher/2.0 (https://globalinvesting.github.io)",
      "Accept": "application/json",
    },
    signal: AbortSignal.timeout(10000),
  });

  // FF returns "Request Denied" HTML when rate limit is exceeded
  const contentType = resp.headers.get("content-type") || "";
  if (!resp.ok || !contentType.includes("application/json")) {
    const body = await resp.text().catch(() => "");
    if (body.includes("Request Denied") || body.includes("DOCTYPE")) {
      throw new Error(`ForexFactory rate limit hit (${resp.status}) — will retry next poll`);
    }
    throw new Error(`ForexFactory HTTP ${resp.status}`);
  }

  const raw = await resp.json();
  if (!Array.isArray(raw)) {
    throw new Error("ForexFactory response is not an array");
  }

  // Filter to G8 medium+high impact only
  const events = [];
  for (const ev of raw) {
    const country = (ev.country || "").trim().toUpperCase();
    if (!G8_CURRENCIES.has(country)) continue;

    const impact = (ev.impact || "").trim().toLowerCase();
    if (impact !== "high" && impact !== "medium") continue;

    const actual   = cleanVal(ev.actual);
    const forecast = cleanVal(ev.forecast);
    const previous = cleanVal(ev.previous);

    // Only include events with at least one data point
    if (actual === null && forecast === null && previous === null) continue;

    events.push({
      title:    (ev.title || "").trim(),
      country,
      date:     (ev.date  || "").trim(),
      actual,
      forecast,
      previous,
    });
  }

  return events;
}

// ── Fingerprint ───────────────────────────────────────────────────────────────

/**
 * Fingerprint covers actual + forecast fields only.
 * previous rarely changes and would cause spurious dispatches if included.
 * Sorted for stability regardless of FF response order.
 */
function buildFingerprint(events) {
  const lines = events
    .map(ev => `${ev.date}|${ev.country}|${ev.title}|${ev.actual ?? ""}|${ev.forecast ?? ""}`)
    .sort();
  return lines.join("\n");
}

function describeDiff(prevFP, newFP) {
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

  if (!GITHUB_PAT)   throw new Error("GITHUB_PAT secret not set");
  if (!GITHUB_OWNER) throw new Error("GITHUB_OWNER secret not set");
  if (!GITHUB_REPO)  throw new Error("GITHUB_REPO secret not set");

  const url = `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/dispatches`;
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${GITHUB_PAT}`,
      "Accept":        "application/vnd.github+json",
      "Content-Type":  "application/json",
      "User-Agent":    "globalinvesting-calendar-watcher/2.0",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    body: JSON.stringify({
      event_type: "calendar-data-changed",
      client_payload: {
        description,
        triggered_at: new Date().toISOString(),
        source: "cloudflare-worker-ff",
      },
    }),
    signal: AbortSignal.timeout(10000),
  });

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
