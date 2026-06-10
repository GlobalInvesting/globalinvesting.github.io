/**
 * calendar-watcher.js — Cloudflare Worker v3.0
 *
 * PRIMARY SOURCE MIGRATED: ForexFactory public JSON → ForexFactory HTML calendar
 *
 * WHY HTML INSTEAD OF JSON
 *   The nfs.faireconomy.media JSON feed has a significant lag (observed: 10+ hours)
 *   before publishing actuals after an event releases. The FF HTML calendar
 *   (forexfactory.com/calendar) shows actuals within seconds of release.
 *   Cloudflare Worker IPs (CF edge network) are NOT datacenter IPs and consistently
 *   receive the full HTML without a JS challenge — the calendar table is server-rendered
 *   and present in the raw HTML response before any client-side JS runs.
 *   GitHub Actions runners (Azure datacenter IPs) would be blocked, but the Worker
 *   acts as the detection layer and only dispatches to GitHub when data changes.
 *
 * ARCHITECTURE
 *   CF Worker polls FF HTML every 2 minutes (was 6 min for JSON due to rate limit;
 *   HTML has no documented rate limit at this cadence — 1 req/2 min per IP).
 *   Parses the calendar table with regex (no DOM parser needed in Workers).
 *   Computes fingerprint of actuals+forecasts for G8 medium+high events.
 *   On change → stores actuals payload in KV → fires repository_dispatch.
 *   GitHub Actions fetch_ff_calendar.py reads the KV payload via /payload endpoint
 *   (Step 1c) to inject actuals without re-fetching FF (avoids the lag entirely).
 *
 * END-TO-END LATENCY (v3.0)
 *   ForexFactory publishes actual on HTML
 *   → CF Worker detects on next 2-min poll     (~0–2 min)  ← was 6 min
 *   → KV payload written + repository_dispatch (~1 sec)
 *   → GitHub runner starts                     (~15–30 sec)
 *   → fetch_ff_calendar.py reads KV /payload   (~1 sec)    ← was 45 sec FF fetch
 *   → git push + Pages rebuild                 (~30 sec)
 *   TOTAL: ~1–3 minutes  (was 2–8 min; actual lag eliminated)
 *
 * REQUIRED SECRETS (unchanged):
 *   GITHUB_PAT, GITHUB_OWNER, GITHUB_REPO
 *
 * REQUIRED KV NAMESPACE:
 *   CALENDAR_KV  — stores fingerprint (calendar:fingerprint:v3) and
 *                  actuals payload (calendar:payload:v3)
 *
 * CRON: every 2 minutes (star-slash-2 * * * *)
 *   wrangler.toml: crons = ["* /2 * * * *"]
 *
 * ENDPOINTS:
 *   /trigger     — manual poll trigger
 *   /fingerprint — view current fingerprint
 *   /payload     — read the latest parsed events (JSON array) — used by fetch_ff_calendar.py
 *   /reset       — clear fingerprint + payload
 *
 * v3.0 changes (2026-06-10):
 * - Primary source: FF HTML calendar (forexfactory.com/calendar) instead of nfs.faireconomy.media JSON
 * - HTML parser: regex-based, handles new-day/no-grid rows correctly
 * - KV payload: full event array written to KV so fetch_ff_calendar.py can read without re-fetching FF
 * - Cron: 2 min (was 6 min — HTML has no per-IP rate limit at this cadence)
 * - Fallback: JSON feed attempted if HTML parse fails (CF block, structure change)
 * - /payload endpoint added for GitHub Actions Step 1c integration
 */

// ── Constants ─────────────────────────────────────────────────────────────────

const G8_CURRENCIES = new Set(["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]);

const FF_HTML_URL    = "https://www.forexfactory.com/calendar";
const FF_JSON_URL    = "https://nfs.faireconomy.media/ff_calendar_thisweek.json";
const FF_JSON_NW_URL = "https://nfs.faireconomy.media/ff_calendar_nextweek.json";

const KV_FINGERPRINT = "calendar:fingerprint:v3";
const KV_PAYLOAD     = "calendar:payload:v3";   // JSON array of parsed events for GH Actions

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
      const fp = await env.CALENDAR_KV.get(KV_FINGERPRINT);
      return new Response(JSON.stringify({ fingerprint: fp, ts: new Date().toISOString() }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/payload") {
      const payload = await env.CALENDAR_KV.get(KV_PAYLOAD);
      return new Response(payload || "null", { headers: { "Content-Type": "application/json" } });
    }

    if (url.pathname === "/reset") {
      await env.CALENDAR_KV.delete(KV_FINGERPRINT);
      await env.CALENDAR_KV.delete(KV_PAYLOAD);
      return new Response(JSON.stringify({ status: "reset", ts: new Date().toISOString() }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(JSON.stringify({
      worker:    "calendar-watcher",
      version:   "3.0",
      source:    "ForexFactory HTML (forexfactory.com/calendar)",
      fallback:  "ForexFactory JSON (nfs.faireconomy.media)",
      schedule:  "every 2 minutes",
      endpoints: ["/trigger", "/fingerprint", "/payload", "/reset"],
      ts:        new Date().toISOString(),
    }), { headers: { "Content-Type": "application/json" } });
  },
};

// ── Core logic ────────────────────────────────────────────────────────────────

async function runCalendarWatch(env) {
  const now   = new Date();
  const label = `[${now.toISOString().slice(0, 16)}Z]`;
  console.log(`${label} calendar-watcher v3.0: starting poll`);

  // Step 1: Fetch and parse FF HTML (primary) — fall back to JSON if blocked
  let events = [];
  let source  = "unknown";

  try {
    events = await fetchAndParseHTML();
    source  = "html";
    console.log(`${label} HTML parse: ${events.length} G8 med/high events`);
  } catch (htmlErr) {
    console.warn(`${label} HTML fetch failed (${htmlErr.message}) — trying JSON fallback`);
    try {
      events = await fetchJSON();
      source  = "json";
      console.log(`${label} JSON fallback: ${events.length} G8 med/high events`);
    } catch (jsonErr) {
      console.error(`${label} Both sources failed — JSON: ${jsonErr.message}`);
      return;
    }
  }

  if (!events.length) {
    console.log(`${label} No events parsed — skipping.`);
    return;
  }

  // Step 2: Fingerprint (actuals + forecasts only — previous excluded to avoid spurious fires)
  const newFP  = buildFingerprint(events);
  const prevFP = (await env.CALENDAR_KV.get(KV_FINGERPRINT)) ?? "";

  if (newFP === prevFP) {
    console.log(`${label} No change detected.`);
    return;
  }

  const diff = describeDiff(prevFP, newFP);
  console.log(`${label} CHANGE: ${diff} [source: ${source}]`);

  // Step 3: Write new fingerprint + full payload to KV
  await env.CALENDAR_KV.put(KV_FINGERPRINT, newFP,               { expirationTtl: 60 * 60 * 24 * 7 });
  await env.CALENDAR_KV.put(KV_PAYLOAD, JSON.stringify(events),  { expirationTtl: 60 * 60 * 24 * 7 });

  // Step 4: Fire repository_dispatch
  try {
    await triggerGitHubWorkflow(env, diff, source);
    console.log(`${label} repository_dispatch sent.`);
  } catch (err) {
    console.error(`${label} repository_dispatch failed: ${err.message}`);
    // Roll back fingerprint so next poll retries
    await env.CALENDAR_KV.put(KV_FINGERPRINT, prevFP, { expirationTtl: 60 * 60 * 24 * 7 });
  }
}

// ── FF HTML parser ────────────────────────────────────────────────────────────

async function fetchAndParseHTML() {
  const resp = await fetch(FF_HTML_URL, {
    headers: {
      "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
      "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      "Accept-Language": "en-US,en;q=0.9",
      "Referer":         "https://www.forexfactory.com/",
      "Cache-Control":   "no-cache",
    },
    signal: AbortSignal.timeout(15000),
  });

  if (!resp.ok) {
    throw new Error(`FF HTML HTTP ${resp.status}`);
  }

  const html = await resp.text();

  // Detect Cloudflare challenge page (no calendar data)
  if (html.includes("Just a moment") || html.includes("cf-browser-verification") ||
      html.includes("Checking your browser") || !html.includes("calendar__table")) {
    throw new Error("CF challenge page returned — no calendar data");
  }

  return parseFFHTML(html);
}

/**
 * Parse ForexFactory HTML calendar table.
 *
 * Row types:
 *   calendar__row--new-day : contains the date cell (rowspan); first event of a day
 *   calendar__row--no-grid : shares time with preceding row (time cell is empty: <!-->)
 *   calendar__row--day-breaker: visual separator, no data — skip
 *
 * Returns: Array of { date, timeET, currency, impact, title, actual, forecast, previous }
 *   date   — "Jun 10" (no year; inferred by fetch_ff_calendar.py from context)
 *   timeET — "9:30am" ET (ForexFactory native timezone display)
 */
function parseFFHTML(html) {
  const rowRe = /(<tr[^>]+class="[^"]*calendar__row[^"]*"[^>]*>)([\s\S]*?)<\/tr>/g;

  const events   = [];
  let currentDate = null;
  let lastTime    = null;
  let match;

  while ((match = rowRe.exec(html)) !== null) {
    const trTag  = match[1];
    const trBody = match[2];

    if (trTag.includes("day-breaker")) continue;

    // Track date from new-day rows
    const dateM = trBody.match(/calendar__date[\s\S]*?<span[^>]*>(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+<span>([\s\S]*?)<\/span>/);
    if (dateM) currentDate = dateM[1].trim();  // e.g. "Jun 10"

    // Skip rows without event-id
    if (!trTag.includes("data-event-id")) continue;

    // Currency
    const curM = trBody.match(/translate="no"[^>]*>([\s\S]*?)<\/span>/);
    if (!curM) continue;
    const currency = curM[1].trim();
    if (!G8_CURRENCIES.has(currency)) continue;

    // Impact
    const impM   = trBody.match(/title="(High|Medium|Low) Impact/);
    const impact = impM ? impM[1].toLowerCase() : "low";
    if (impact === "low") continue;

    // Title
    const titleM = trBody.match(/calendar__event-title">([\s\S]*?)<\/span>/);
    if (!titleM) continue;
    const title = titleM[1].trim();

    // Time — no-grid rows have empty time cell (<!-->); inherit lastTime
    const timeCellM = trBody.match(/calendar__time[^>]*>([\s\S]*?)<\/td>/);
    if (timeCellM) {
      const timeSpanM = timeCellM[1].match(/<span>([^<]+)<\/span>/);
      if (timeSpanM) {
        const t = timeSpanM[1].trim();
        if (t) lastTime = t;
      }
    }
    const timeET = lastTime || "";

    // Actual, forecast, previous — strip all HTML tags
    const actualCellM   = trBody.match(/calendar__actual[^>]*>([\s\S]*?)<\/td>/);
    const forecastCellM = trBody.match(/calendar__forecast[^>]*>([\s\S]*?)<\/td>/);
    const prevCellM     = trBody.match(/calendar__previous[^>]*>([\s\S]*?)<\/td>/);

    const actual   = actualCellM   ? stripTags(actualCellM[1])   : "";
    const forecast = forecastCellM ? stripTags(forecastCellM[1]) : "";
    const previous = prevCellM     ? stripTags(prevCellM[1])     : "";

    events.push({
      date:     currentDate || "",
      timeET,
      currency,
      impact,
      title,
      actual:   actual   || null,
      forecast: forecast || null,
      previous: previous || null,
    });
  }

  if (events.length === 0) {
    throw new Error("HTML parsed but 0 G8 med/high events found — page structure may have changed");
  }

  return events;
}

function stripTags(html) {
  return html.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
}

// ── FF JSON fallback ──────────────────────────────────────────────────────────

async function fetchJSON() {
  const HEADERS = {
    "User-Agent": "globalinvesting-calendar-watcher/3.0 (https://globalinvesting.github.io)",
    "Accept":     "application/json",
  };

  const resp = await fetch(FF_JSON_URL, { headers: HEADERS, signal: AbortSignal.timeout(10000) });
  const ct   = resp.headers.get("content-type") || "";
  if (!resp.ok || !ct.includes("application/json")) {
    const body = await resp.text().catch(() => "");
    throw new Error(`JSON HTTP ${resp.status}: ${body.slice(0, 60)}`);
  }
  const raw = await resp.json();
  if (!Array.isArray(raw)) throw new Error("JSON response is not an array");

  // Best-effort nextweek fetch
  let rawNw = [];
  try {
    const r2 = await fetch(FF_JSON_NW_URL, { headers: HEADERS, signal: AbortSignal.timeout(8000) });
    const ct2 = r2.headers.get("content-type") || "";
    if (r2.ok && ct2.includes("application/json")) {
      const p = await r2.json();
      if (Array.isArray(p)) rawNw = p;
    }
  } catch (_) {}

  const events = [];
  for (const ev of [...raw, ...rawNw]) {
    const currency = (ev.country || "").trim().toUpperCase();
    if (!G8_CURRENCIES.has(currency)) continue;
    const impact = (ev.impact || "").trim().toLowerCase();
    if (impact !== "high" && impact !== "medium") continue;
    events.push({
      date:     (ev.date || "").trim(),
      timeET:   "",
      currency,
      impact,
      title:    (ev.title || "").trim(),
      actual:   cleanVal(ev.actual),
      forecast: cleanVal(ev.forecast),
      previous: cleanVal(ev.previous),
    });
  }
  return events;
}

// ── Fingerprint ───────────────────────────────────────────────────────────────

function buildFingerprint(events) {
  return events
    .map(ev => `${ev.date}|${ev.currency}|${ev.title}|${ev.actual ?? ""}|${ev.forecast ?? ""}`)
    .sort()
    .join("\n");
}

function describeDiff(prevFP, newFP) {
  const prev  = new Set(prevFP ? prevFP.split("\n") : []);
  const next  = new Set(newFP.split("\n"));
  const added = [...next].filter(l => !prev.has(l));

  const newActuals   = added.filter(l => l.split("|")[3] !== "");
  const newForecasts = added.filter(l => l.split("|")[3] === "" && l.split("|")[4] !== "");

  const parts = [];
  if (newActuals.length) {
    const samples = newActuals.slice(0, 2)
      .map(l => { const p = l.split("|"); return `${p[1]} ${p[2].slice(0, 25)}=${p[3]}`; });
    parts.push(`${newActuals.length} actual(s): ${samples.join(", ")}`);
  }
  if (newForecasts.length) parts.push(`${newForecasts.length} forecast(s)`);
  return parts.join(" | ") || `${added.length} change(s)`;
}

// ── GitHub repository_dispatch ────────────────────────────────────────────────

async function triggerGitHubWorkflow(env, description, source) {
  const { GITHUB_PAT, GITHUB_OWNER, GITHUB_REPO } = env;
  if (!GITHUB_PAT)   throw new Error("GITHUB_PAT not set");
  if (!GITHUB_OWNER) throw new Error("GITHUB_OWNER not set");
  if (!GITHUB_REPO)  throw new Error("GITHUB_REPO not set");

  const resp = await fetch(
    `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/dispatches`,
    {
      method: "POST",
      headers: {
        "Authorization":        `Bearer ${GITHUB_PAT}`,
        "Accept":               "application/vnd.github+json",
        "Content-Type":         "application/json",
        "User-Agent":           "globalinvesting-calendar-watcher/3.0",
        "X-GitHub-Api-Version": "2022-11-28",
      },
      body: JSON.stringify({
        event_type: "calendar-data-changed",
        client_payload: {
          description,
          triggered_at: new Date().toISOString(),
          source: `cloudflare-worker-${source}`,
        },
      }),
      signal: AbortSignal.timeout(10000),
    }
  );

  if (resp.status !== 204) {
    const body = await resp.text().catch(() => "");
    throw new Error(`GitHub API ${resp.status}: ${body}`);
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function cleanVal(v) {
  if (v == null) return null;
  const s = String(v).trim();
  return s === "" || s === "None" || s === "null" || s === "N/A" || s === "—" ? null : s;
}
