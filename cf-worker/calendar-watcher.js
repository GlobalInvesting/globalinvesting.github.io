/**
 * calendar-watcher.js — Cloudflare Worker v5.0
 *
 * PRIMARY SOURCE: Myfxbook RSS feed
 *   https://www.myfxbook.com/rss/forex-economic-calendar-events
 *
 *   RSS is accessible from both GH Actions and CF edge IPs (no WAF).
 *   Covers a rolling ~24h window. Actuals appear within minutes of release.
 *   Single source — FF JSON removed to avoid title-key mismatches that cause
 *   duplicate events or erased carry-forward actuals.
 *
 * CRON: every 30 minutes — fetches RSS, fingerprints actuals+forecasts,
 *   fires repository_dispatch if any change detected.
 *
 * ENDPOINTS:
 *   /trigger     — manual on-demand poll (fires dispatch if changed)
 *   /fingerprint — view current KV fingerprint
 *   /payload     — latest parsed events from last cron/trigger run
 *   /reset       — clear KV state
 *
 * REQUIRED SECRETS (unchanged): GITHUB_PAT, GITHUB_OWNER, GITHUB_REPO
 *
 * v5.0 changes (2026-06-11):
 * - PRIMARY SOURCE REPLACED: Myfxbook HTML → Myfxbook RSS feed.
 *   HTML fetch blocked (CF edge IPs in same Cloudflare-blocked datacenter
 *   ranges as GH Actions — error 1102). RSS has no WAF — accessible from
 *   both CF edge and GH Actions (confirmed: 127KB, instant).
 * - NEW: fetchMyfxbookRSS() — RSS fetch with browser User-Agent.
 * - NEW: parseMyfxbookRSS() — parses RSS XML: iterates <item> blocks,
 *   derives currency from country slug in <link>, decodes HTML entities in
 *   <description>, extracts impact from sprite class, reads previous/forecast/
 *   actual from <td> positions. Released: actual present OR time_left negative.
 * - REMOVED: fetchMyfxbook(), parseMyfxbookHTML() — HTML approach abandoned.
 * - REMOVED: fetchFFJSON(), FF_JSON_URL, FF_JSON_NW constants — single source
 *   only to prevent title-key mismatches in (currency+date+time+title) dedup.
 * - REMOVED: /myfxbook endpoint — no longer proxying HTML fetch.
 * - KV keys bumped: calendar:fingerprint:v5, calendar:payload:v5.
 *
 * v4.0 changes (2026-06-10):
 * - PRIMARY: Myfxbook HTML (CF edge IPs not blocked — since disproven).
 * - REMOVED: FCS API (free tier exhausted).
 */

// ── Constants ─────────────────────────────────────────────────────────────────

const G8 = new Set(["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]);

const MFB_RSS_URL = "https://www.myfxbook.com/rss/forex-economic-calendar-events";

const KV_FINGERPRINT = "calendar:fingerprint:v5";
const KV_PAYLOAD     = "calendar:payload:v5";

// Country slug → G8 currency
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
      version:   "5.0",
      source:    "Myfxbook RSS (myfxbook.com/rss/forex-economic-calendar-events)",
      schedule:  "every 30 minutes",
      endpoints: ["/trigger", "/fingerprint", "/payload", "/reset"],
      ts:        new Date().toISOString(),
    }), { headers: { "Content-Type": "application/json" } });
  },
};

// ── Core cron logic ───────────────────────────────────────────────────────────

async function runCalendarWatch(env) {
  const now   = new Date();
  const label = `[${now.toISOString().slice(0, 16)}Z]`;
  console.log(`${label} calendar-watcher v5.0: starting poll`);

  let events = [];

  try {
    events = await fetchMyfxbookRSS();
    console.log(`${label} Myfxbook RSS: ${events.length} G8 med/high events (${events.filter(e => e.actual).length} with actuals)`);
  } catch (err) {
    console.error(`${label} Myfxbook RSS failed: ${err.message} — skipping.`);
    return;
  }

  if (!events.length) {
    console.log(`${label} No G8 med/high events parsed — skipping.`);
    return;
  }

  // Fingerprint actuals + forecasts
  const newFP  = buildFingerprint(events);
  const prevFP = (await env.CALENDAR_KV.get(KV_FINGERPRINT)) ?? "";

  if (newFP === prevFP) {
    console.log(`${label} No change detected.`);
    return;
  }

  const diff = describeDiff(prevFP, newFP);
  console.log(`${label} CHANGE: ${diff}`);

  // Write KV
  await env.CALENDAR_KV.put(KV_FINGERPRINT, newFP,              { expirationTtl: 60 * 60 * 24 * 7 });
  await env.CALENDAR_KV.put(KV_PAYLOAD, JSON.stringify(events), { expirationTtl: 60 * 60 * 24 * 7 });

  // Fire repository_dispatch
  try {
    await triggerGitHubWorkflow(env, diff, "myfxbook-rss");
    console.log(`${label} repository_dispatch sent.`);
  } catch (err) {
    console.error(`${label} repository_dispatch failed: ${err.message}`);
    // Roll back fingerprint so next cron retries
    await env.CALENDAR_KV.put(KV_FINGERPRINT, prevFP, { expirationTtl: 60 * 60 * 24 * 7 });
  }
}

// ── Myfxbook RSS fetch + parse ────────────────────────────────────────────────

async function fetchMyfxbookRSS() {
  const resp = await fetch(MFB_RSS_URL, {
    headers: {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
      "Accept":     "application/rss+xml, application/xml, text/xml, */*",
    },
    signal: AbortSignal.timeout(20000),
  });

  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

  const xml = await resp.text();
  if (!xml.includes("<item>")) throw new Error("RSS response missing <item> elements");

  return parseMyfxbookRSS(xml);
}

function parseMyfxbookRSS(xml) {
  const events = [];
  let skippedCcy = 0, skippedImpact = 0;

  // Iterate <item> blocks
  for (const itemM of xml.matchAll(/<item>([\s\S]*?)<\/item>/g)) {
    const block = itemM[1];

    // Currency from country slug in <link>
    const linkM = block.match(/<link>https?:\/\/[^/]+\/forex-economic-calendar\/([^/]+)\/[^<]+<\/link>/);
    if (!linkM) continue;
    const currency = SLUG_TO_CCY[linkM[1]] || "";
    if (!currency) { skippedCcy++; continue; }

    // Event title
    const titleM = block.match(/<title>([^<]+)<\/title>/);
    const title  = titleM ? titleM[1].trim() : "";
    if (!title) continue;

    // Date/time from <pubDate> (RFC 2822 UTC)
    const pubM = block.match(/<pubDate>([^<]+)<\/pubDate>/);
    let dateISO = "", timeUTC = "00:00";
    if (pubM) {
      const d = new Date(pubM[1].trim());
      if (!isNaN(d)) {
        dateISO = d.toISOString().slice(0, 10);
        timeUTC = d.toISOString().slice(11, 16);
      }
    }
    if (!dateISO) continue;

    // Decode HTML entities in <description>
    const descM = block.match(/<description>([\s\S]*?)<\/description>/);
    if (!descM) continue;
    const desc = descM[1]
      .replace(/&#60;/g, "<").replace(/&#62;/g, ">")
      .replace(/&#39;/g, "'").replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">").replace(/&amp;/g, "&");

    // Impact from sprite class
    const impM  = desc.match(/sprite-(high|medium|low)-impact/);
    const impact = impM ? impM[1] : "low";
    if (impact === "low") { skippedImpact++; continue; }

    // Values from <td> positions in data row (skip <th> header row)
    const tds = [...desc.matchAll(/<td>([\s\S]*?)<\/td>/g)].map(m =>
      m[1].replace(/<[^>]+>/g, "").trim()
    );

    const timeLeftRaw = tds[0] || "";
    const previous    = cleanVal(tds[2]);
    const forecast    = cleanVal(tds[3]);
    const actual      = cleanVal(tds[4]);

    // Released: actual present OR time_left is negative seconds (event passed)
    const isPassed = timeLeftRaw.startsWith("-") && timeLeftRaw.includes("second");
    const released = actual !== null || isPassed;

    events.push({
      date:     `${dateISO} ${timeUTC}:00`,
      currency,
      impact,
      title,
      actual,
      forecast,
      previous,
      dateISO,
      timeUTC,
      released,
    });
  }

  events.sort((a, b) => a.date.localeCompare(b.date) || a.currency.localeCompare(b.currency));

  console.log(`parseMyfxbookRSS: ${events.length} G8 med/high events (${events.filter(e => e.actual).length} with actuals; skipped ${skippedCcy} non-G8, ${skippedImpact} low-impact)`);
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
        "User-Agent":           "globalinvesting-calendar-watcher/5.0",
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
