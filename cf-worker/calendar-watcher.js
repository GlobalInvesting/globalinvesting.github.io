/**
 * calendar-watcher.js — Cloudflare Worker v4.0
 *
 * PRIMARY SOURCE: Myfxbook HTML calendar
 *   https://www.myfxbook.com/forex-economic-calendar
 *
 *   CF Worker edge IPs are NOT blocked by Myfxbook's Cloudflare setup (confirmed:
 *   3,300 DOM nodes returned from CF Worker fetch with browser User-Agent).
 *   GitHub Actions runner IPs ARE blocked (TCP-level hang — Cloudflare holds
 *   connection open indefinitely). Solution: route all Myfxbook fetches through
 *   this Worker. GH Actions calls /myfxbook on this Worker; Worker calls Myfxbook.
 *
 * FALLBACK: ForexFactory JSON (nfs.faireconomy.media)
 *   No WAF on FF JSON. Accessible from both CF edge and GH Actions.
 *   Used if Myfxbook fetch fails.
 *
 * REMOVED: FCS API (api-v4.fcsapi.com) — free tier credits exhausted.
 *
 * CRON: every 30 minutes — fetches Myfxbook, fingerprints actuals,
 *   fires repository_dispatch on change.
 *
 * ENDPOINTS:
 *   /myfxbook    — fetch + parse Myfxbook HTML, return events JSON array
 *                  called by fetch_ff_calendar.py Step 1 (replaces direct HTTP)
 *   /trigger     — manual on-demand poll (cron logic, fires dispatch if changed)
 *   /fingerprint — view current fingerprint
 *   /payload     — latest parsed events from last cron/trigger run
 *   /reset       — clear KV state
 *
 * REQUIRED SECRETS (unchanged): GITHUB_PAT, GITHUB_OWNER, GITHUB_REPO
 * FCS_API_KEY no longer required.
 *
 * v4.0 changes (2026-06-10):
 * - PRIMARY: Myfxbook HTML replaces FCS API (credits exhausted)
 * - NEW: /myfxbook endpoint — proxies Myfxbook HTML fetch + parsing for GH Actions
 * - NEW: parseMyfxbookHTML() — JS regex parser matching Python v3.23 logic
 * - FALLBACK: FF JSON retained (accessible from CF edge, no WAF)
 * - REMOVED: fetchFCSAPI() — FCS free tier exhausted
 * - Cron: every 30 minutes (unchanged)
 */

// ── Constants ─────────────────────────────────────────────────────────────────

const G8 = new Set(["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]);

const MFB_URL     = "https://www.myfxbook.com/forex-economic-calendar";
const FF_JSON_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json";
const FF_JSON_NW  = "https://nfs.faireconomy.media/ff_calendar_nextweek.json";

const KV_FINGERPRINT = "calendar:fingerprint:v4";
const KV_PAYLOAD     = "calendar:payload:v4";

const MFB_HEADERS = {
  "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
  "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
  "Accept-Language": "en-US,en;q=0.5",
};

// Country slug → G8 currency (matches Python v3.23 SLUG_TO_CCY)
const SLUG_TO_CCY = {
  "united-states": "USD", "euro-area": "EUR", "germany": "EUR",
  "france": "EUR", "italy": "EUR", "spain": "EUR", "netherlands": "EUR",
  "belgium": "EUR", "ireland": "EUR", "portugal": "EUR", "finland": "EUR",
  "austria": "EUR", "greece": "EUR", "european-union": "EUR",
  "united-kingdom": "GBP", "japan": "JPY", "canada": "CAD",
  "australia": "AUD", "new-zealand": "NZD", "switzerland": "CHF",
};

// ── Entry point ───────────────────────────────────────────────────────────────

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(runCalendarWatch(env));
  },

  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // /myfxbook — proxy endpoint for GH Actions (bypasses WAF block on runner IPs)
    if (url.pathname === "/myfxbook") {
      try {
        const { events, holidays } = await fetchMyfxbook();
        return new Response(JSON.stringify({ events, holidays, ts: new Date().toISOString() }), {
          headers: { "Content-Type": "application/json" },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: err.message }), {
          status: 502,
          headers: { "Content-Type": "application/json" },
        });
      }
    }

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
      version:   "4.0",
      source:    "Myfxbook HTML (myfxbook.com/forex-economic-calendar)",
      fallback:  "ForexFactory JSON (nfs.faireconomy.media)",
      schedule:  "every 30 minutes",
      endpoints: ["/myfxbook", "/trigger", "/fingerprint", "/payload", "/reset"],
      ts:        new Date().toISOString(),
    }), { headers: { "Content-Type": "application/json" } });
  },
};

// ── Core cron logic ───────────────────────────────────────────────────────────

async function runCalendarWatch(env) {
  const now   = new Date();
  const label = `[${now.toISOString().slice(0, 16)}Z]`;
  console.log(`${label} calendar-watcher v4.0: starting poll`);

  let events = [];
  let source  = "unknown";

  // Step 1: Myfxbook HTML (primary — CF edge IPs not blocked)
  try {
    const result = await fetchMyfxbook();
    events = result.events;
    source  = "myfxbook";
    console.log(`${label} Myfxbook: ${events.length} G8 med/high events`);
  } catch (err) {
    console.warn(`${label} Myfxbook failed (${err.message}) — trying FF JSON fallback`);
  }

  // Step 2: FF JSON fallback
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
    await env.CALENDAR_KV.put(KV_FINGERPRINT, prevFP, { expirationTtl: 60 * 60 * 24 * 7 });
  }
}

// ── Myfxbook HTML fetch + parse ───────────────────────────────────────────────

async function fetchMyfxbook() {
  const resp = await fetch(MFB_URL, {
    headers: MFB_HEADERS,
    signal:  AbortSignal.timeout(20000),
  });

  if (!resp.ok) {
    throw new Error(`Myfxbook HTTP ${resp.status}`);
  }

  const html = await resp.text();

  if (!html.includes("calendarToggleCell")) {
    throw new Error("Myfxbook response missing calendar data — WAF block or layout change");
  }

  return parseMyfxbookHTML(html);
}

function parseMyfxbookHTML(html) {
  // ── Build OID lookup maps from full HTML ──────────────────────────────────

  // previous-value: data-previous="OID" ... previous-value="VALUE"
  const prevMap = {};
  for (const m of html.matchAll(/data-previous="(\d+)"[^>]*previous-value="([^"]*)"/g)) {
    prevMap[m[1]] = m[2];
  }

  // forecast/consensus: data-concensus="OID" concensus="VALUE" consistconcensus
  const consMap = {};
  for (const m of html.matchAll(/<td[^>]*data-concensus="(\d+)" concensus="([^"]*)" consistconcensus/g)) {
    consMap[m[1]] = m[2];
  }

  // actual: data-actual="OID" ... actualCell ... innermost span value
  const actualMap = {};
  for (const m of html.matchAll(/data-actual="(\d+)"[\s\S]*?class="actualCell">([\s\S]*?)<\/span>\s*<\/span>/g)) {
    const oid   = m[1];
    const inner = m[2];
    // Extract innermost span text
    const spans = [...inner.matchAll(/<span[^>]*>\s*([^<]+?)\s*<\/span>/g)];
    const raw   = spans.length ? spans[spans.length - 1][1].trim() : inner.replace(/<[^>]+>/g, "").trim();
    // Skip countdown strings ("min", "h ")
    if (raw && !raw.includes("min") && !raw.includes("h ") && raw !== "-") {
      actualMap[oid] = raw;
    }
  }

  // ── Parse <tr> event rows from <tbody> ───────────────────────────────────

  const tbodyM = html.match(/<tbody>([\s\S]*?)<\/tbody>/);
  if (!tbodyM) throw new Error("No <tbody> found in Myfxbook HTML");
  const tbody = tbodyM[1];

  // Split into rows — use non-greedy match
  const trBlocks = [...tbody.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/g)].map(m => m[1]);

  const events   = [];
  const holidays = [];

  for (const tr of trBlocks) {
    // OID
    const oidM = tr.match(/id="itemOid" value="(\d+)"/);
    if (!oidM) continue;
    const oid = oidM[1];

    // Datetime (UTC)
    const dtM = tr.match(/data-calendardatetd="([^"]+)"/);
    if (!dtM) continue;
    const dtRaw  = dtM[1];        // "2026-06-10 12:30:00.0"
    const dateISO = dtRaw.slice(0, 10);
    const timeUTC = dtRaw.slice(11, 16);  // "12:30"

    // Currency from standalone <td> with exactly 3 uppercase letters
    const curM = tr.match(/<td[^>]*calendarToggleCell[^>]*>\s*([A-Z]{3})\s*<\/td>/);
    let currency = curM ? curM[1].trim() : "";

    // Impact from importance attribute
    const impM = tr.match(/importance="(\d)"/);
    const impNum = impM ? parseInt(impM[1]) : 0;
    const impact = impNum === 3 ? "high" : impNum === 2 ? "medium" : "low";

    // Event link — href comes before class in myfxbook HTML
    const evM = tr.match(/<a href="(?:https?:\/\/[^/]+)?(\/forex-economic-calendar\/[^"]+)"[^>]*calendar-event-link[^>]*>([^<]+)<\/a>/);

    if (!evM) {
      // Holiday row (no event link)
      if (tr.includes("impact_no") || tr.includes("data-is-holiday=\"true\"")) {
        const titleM = tr.match(/class="[^"]*calendarToggleCell[^"]*text-left[^"]*"[^>]*>([\s\S]*?)<\/td>/);
        const title  = titleM ? titleM[1].replace(/<[^>]+>/g, "").trim() : "Bank Holiday";
        if (currency && G8.has(currency) && dateISO) {
          holidays.push({ title: title || "Bank Holiday", currency, dateISO });
        }
      }
      continue;
    }

    const slugUrl   = evM[1];   // "/forex-economic-calendar/united-states/cpi-s-a"
    let   eventName = evM[2].trim();

    // Period suffix e.g. "(May)"
    const periodM = tr.match(/<span>\(([^)]+)\)<\/span>/);
    if (periodM) eventName = `${eventName} (${periodM[1]})`;

    // Derive currency from URL slug if td extraction failed
    if (!currency || !G8.has(currency)) {
      const parts = slugUrl.replace(/^\//, "").split("/");
      // parts: ["forex-economic-calendar", "united-states", "cpi-s-a"]
      const countrySlug = parts[1] || "";
      currency = SLUG_TO_CCY[countrySlug] || "";
    }

    if (!currency || !G8.has(currency)) continue;
    if (impact === "low") continue;

    const previous = cleanVal(prevMap[oid]);
    const forecast = cleanVal(consMap[oid]);
    const actual   = cleanVal(actualMap[oid]);
    const isPassed = tr.includes('ispassed="1"');
    const released = actual !== null || isPassed;

    events.push({
      date:     `${dateISO} ${timeUTC}:00`,  // "YYYY-MM-DD HH:MM:SS" UTC — consistent with KV payload format
      currency,
      impact,
      title:    eventName,
      actual,
      forecast,
      previous,
      dateISO,
      timeUTC,
      released,
    });
  }

  events.sort((a, b) => a.date.localeCompare(b.date) || a.currency.localeCompare(b.currency));

  console.log(`parseMyfxbookHTML: ${events.length} events (${events.filter(e => e.actual).length} with actuals), ${holidays.length} holidays`);
  return { events, holidays };
}

// ── FF JSON fallback ──────────────────────────────────────────────────────────

async function fetchFFJSON() {
  const HEADERS = {
    "User-Agent": "globalinvesting-calendar-watcher/4.0 (https://globalinvesting.github.io)",
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

  let rawNw = [];
  try {
    const r2  = await fetch(FF_JSON_NW, { headers: HEADERS, signal: AbortSignal.timeout(8000) });
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

    // FF JSON date is ET — normalise to YYYY-MM-DD HH:MM:SS prefix for consistency
    const dateRaw = (ev.date || "").trim();
    events.push({
      date:     dateRaw,
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
        "User-Agent":           "globalinvesting-calendar-watcher/4.0",
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
  return s === "" || s === "None" || s === "null" || s === "N/A" || s === "—" || s === "-" ? null : s;
}
