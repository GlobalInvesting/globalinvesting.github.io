/**
 * calendar-watcher.js — Cloudflare Worker v3.1
 *
 * PRIMARY SOURCE: FCS API economy_cal endpoint
 *   https://api-v4.fcsapi.com/forex/economy_cal?access_key=KEY
 *
 * WHY FCS API INSTEAD OF FF HTML
 *   ForexFactory HTML returns HTTP 403 to Cloudflare Worker IPs — FF protects its
 *   HTML with Cloudflare Bot Management (__cf_bm cookie requiring JS execution).
 *   CF Worker edge IPs are identified and blocked by FF's WAF rules.
 *   FCS API is a REST API that explicitly supports programmatic access from any IP.
 *   It returns actual, forecast, previous for G8 med/high events with near-zero lag.
 *
 * ARCHITECTURE
 *   CF Worker polls FCS API every 30 minutes (free tier: 500 req/month limit).
 *   On-demand polls via /trigger endpoint — called by fetch_ff_calendar.py after
 *   detecting new upcoming events, ensuring fresh actuals when they release.
 *   Filters to G8 currencies, medium+high importance.
 *   Computes fingerprint of actuals+forecasts.
 *   On change: writes event array to KV (/payload) + fires repository_dispatch.
 *   fetch_ff_calendar.py Step 1c reads /payload and injects actuals into fresh events.
 *
 * FCS API free plan: 500 credits/month, 1 credit/request.
 *   At */30 cron: 48 req/day = 1,440 req/month — WITHIN free tier (500/month).
 *   NOTE: 500 req/month cap. At 48/day that's ~10 days before hitting the limit.
 *   If the limit becomes an issue, set cron to */60 (720 req/month still too high)
 *   or rely on /trigger only (on-demand from GH Actions).
 *
 * CRON: every 30 minutes.
 *   wrangler.toml: crons = ["*/30 * * * *"]
 *
 * REQUIRED SECRETS:
 *   GITHUB_PAT, GITHUB_OWNER, GITHUB_REPO  — unchanged
 *   FCS_API_KEY                             — NEW: add via `npx wrangler secret put FCS_API_KEY`
 *
 * REQUIRED KV NAMESPACE:
 *   CALENDAR_KV  — calendar:fingerprint:v3, calendar:payload:v3
 *
 * ENDPOINTS:
 *   /trigger     — manual/on-demand poll (called by fetch_ff_calendar.py)
 *   /fingerprint — view current fingerprint
 *   /payload     — read the latest parsed events (JSON array) — used by fetch_ff_calendar.py
 *   /reset       — clear fingerprint + payload
 *
 * FCS API economy_cal response fields (per event):
 *   id, event, title, country (ISO-2), currency, importance (0=low,1=med,2=high),
 *   period, actual, forecast, previous, source, date ("YYYY-MM-DD HH:MM:SS" UTC)
 *
 * v3.1 changes (2026-06-10):
 * - Primary source: FCS API (accessible from CF edge IPs, not blocked by WAF)
 * - Removed FF HTML scraping (CF edge IPs receive HTTP 403 from FF WAF — confirmed)
 * - FF JSON retained as in-process fallback (best-effort, no extra request cost)
 * - Cron: */30 (FCS free tier: 500 req/month)
 * - /trigger endpoint: called by fetch_ff_calendar.py Step 1c for on-demand FCS poll
 * - FCS_API_KEY secret required (add separately: npx wrangler secret put FCS_API_KEY)
 * - KV payload date format: YYYY-MM-DD HH:MM:SS UTC (Step 1c updated accordingly)
 *
 * v3.0 changes (2026-06-10):
 * - Primary source migrated from FF JSON to FF HTML (later found to be blocked — see v3.1)
 * - Added /payload KV endpoint for Step 1c injection
 * - Cron: star-slash-2 (reverted to star-slash-30 in v3.1 due to FCS rate limit)
 */

// ── Constants ─────────────────────────────────────────────────────────────────

const G8 = new Set(["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]);

// FCS API economy calendar — G8 country filter
const FCS_URL = "https://api-v4.fcsapi.com/forex/economy_cal";

// FF JSON fallback (no IP block from CF edge, but has lag — used only if FCS fails)
const FF_JSON_URL    = "https://nfs.faireconomy.media/ff_calendar_thisweek.json";
const FF_JSON_NW_URL = "https://nfs.faireconomy.media/ff_calendar_nextweek.json";

const KV_FINGERPRINT = "calendar:fingerprint:v3";
const KV_PAYLOAD     = "calendar:payload:v3";

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
      version:   "3.1",
      source:    "FCS API (api-v4.fcsapi.com/forex/economy_cal)",
      fallback:  "ForexFactory JSON (nfs.faireconomy.media)",
      schedule:  "every 30 minutes (FCS free tier limit)",
      endpoints: ["/trigger", "/fingerprint", "/payload", "/reset"],
      ts:        new Date().toISOString(),
    }), { headers: { "Content-Type": "application/json" } });
  },
};

// ── Core logic ────────────────────────────────────────────────────────────────

async function runCalendarWatch(env) {
  const now   = new Date();
  const label = `[${now.toISOString().slice(0, 16)}Z]`;
  console.log(`${label} calendar-watcher v3.1: starting poll`);

  let events = [];
  let source  = "unknown";

  // Step 1: Try FCS API (primary — REST API, not blocked from CF edge IPs)
  if (env.FCS_API_KEY) {
    try {
      events = await fetchFCSAPI(env.FCS_API_KEY);
      source  = "fcs";
      console.log(`${label} FCS API: ${events.length} G8 med/high events`);
    } catch (err) {
      console.warn(`${label} FCS API failed (${err.message}) — trying FF JSON fallback`);
    }
  } else {
    console.warn(`${label} FCS_API_KEY not set — skipping FCS, trying FF JSON fallback`);
  }

  // Step 2: FF JSON fallback if FCS failed or key missing
  if (!events.length) {
    try {
      events = await fetchFFJSON();
      source  = "ff-json";
      console.log(`${label} FF JSON fallback: ${events.length} G8 med/high events`);
    } catch (err) {
      console.error(`${label} FF JSON also failed: ${err.message}`);
      return;
    }
  }

  if (!events.length) {
    console.log(`${label} No events — skipping.`);
    return;
  }

  // Step 3: Fingerprint
  const newFP  = buildFingerprint(events);
  const prevFP = (await env.CALENDAR_KV.get(KV_FINGERPRINT)) ?? "";

  if (newFP === prevFP) {
    console.log(`${label} No change detected [source: ${source}].`);
    return;
  }

  const diff = describeDiff(prevFP, newFP);
  console.log(`${label} CHANGE: ${diff} [source: ${source}]`);

  // Step 4: Write KV
  await env.CALENDAR_KV.put(KV_FINGERPRINT, newFP,               { expirationTtl: 60 * 60 * 24 * 7 });
  await env.CALENDAR_KV.put(KV_PAYLOAD, JSON.stringify(events),  { expirationTtl: 60 * 60 * 24 * 7 });

  // Step 5: Fire repository_dispatch
  try {
    await triggerGitHubWorkflow(env, diff, source);
    console.log(`${label} repository_dispatch sent.`);
  } catch (err) {
    console.error(`${label} repository_dispatch failed: ${err.message}`);
    // Roll back fingerprint so the next poll retries the dispatch
    await env.CALENDAR_KV.put(KV_FINGERPRINT, prevFP, { expirationTtl: 60 * 60 * 24 * 7 });
  }
}

// ── FCS API fetch ─────────────────────────────────────────────────────────────

async function fetchFCSAPI(apiKey) {
  // G8 country codes for FCS API country filter (ISO-2)
  const g8Countries = "US,EU,GB,JP,AU,CA,CH,NZ";
  const url = `${FCS_URL}?country=${g8Countries}&access_key=${apiKey}`;

  const resp = await fetch(url, {
    headers: {
      "Accept":     "application/json",
      "User-Agent": "globalinvesting-calendar-watcher/3.1",
    },
    signal: AbortSignal.timeout(12000),
  });

  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(`FCS API HTTP ${resp.status}: ${body.slice(0, 80)}`);
  }

  const data = await resp.json();

  // FCS response: { status: true, response: [...], info: {...} }
  if (!data.status || !Array.isArray(data.response)) {
    throw new Error(`FCS API unexpected response: ${JSON.stringify(data).slice(0, 120)}`);
  }

  const events = [];
  for (const ev of data.response) {
    const currency = (ev.currency || "").trim().toUpperCase();
    if (!G8.has(currency)) continue;

    // importance: 0=low, 1=medium, 2=high — skip low
    const imp = Number(ev.importance ?? -1);
    if (imp < 1) continue;

    events.push({
      date:     (ev.date || "").trim(),    // "YYYY-MM-DD HH:MM:SS" UTC
      timeET:   "",                         // FCS provides UTC dates, not ET times
      currency,
      impact:   imp === 2 ? "high" : "medium",
      title:    (ev.title || "").trim(),
      actual:   cleanVal(ev.actual),
      forecast: cleanVal(ev.forecast),
      previous: cleanVal(ev.previous),
    });
  }

  if (!events.length) {
    throw new Error("FCS API returned 0 G8 med/high events (key invalid or no events this week)");
  }

  return events;
}

// ── FF JSON fallback ──────────────────────────────────────────────────────────

async function fetchFFJSON() {
  const HEADERS = {
    "User-Agent": "globalinvesting-calendar-watcher/3.1 (https://globalinvesting.github.io)",
    "Accept":     "application/json",
  };

  const resp = await fetch(FF_JSON_URL, { headers: HEADERS, signal: AbortSignal.timeout(10000) });
  const ct   = resp.headers.get("content-type") || "";
  if (!resp.ok || !ct.includes("application/json")) {
    const body = await resp.text().catch(() => "");
    throw new Error(`FF JSON HTTP ${resp.status}: ${body.slice(0, 60)}`);
  }
  const raw = await resp.json();
  if (!Array.isArray(raw)) throw new Error("FF JSON not an array");

  // Best-effort nextweek
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
    if (!G8.has(currency)) continue;
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
        "User-Agent":           "globalinvesting-calendar-watcher/3.1",
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
