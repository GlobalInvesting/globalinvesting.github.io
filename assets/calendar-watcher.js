/**
 * calendar-watcher.js — Cloudflare Worker v5.6
 *
 * PRIMARY SOURCE: Myfxbook RSS feed
 *   https://www.myfxbook.com/rss/forex-economic-calendar-events
 *   — Provides full event list: currencies, dates, times, impact, forecasts,
 *     and actuals (actuals appear ~1h post-release based on observed behaviour).
 *
 * SECONDARY SOURCE: Myfxbook HTML calendar (anonymous, no cookie required)
 *   https://www.myfxbook.com/forex-economic-calendar
 *   — Confirmed accessible from CF Workers edge IPs (200 OK, 1.9MB, 332 rows,
 *     31 actuals found — tested 2026-06-11). No WAF block on HTML endpoint.
 *   — Actuals appear in HTML minutes after release vs ~1h in RSS.
 *   — Runs on every cron tick, unconditionally. Failure is non-fatal: if HTML
 *     fetch fails, RSS actuals (delayed) are used as natural fallback.
 *   — MFB_SESSION_COOKIE is no longer required. Cookie path kept for
 *     potential future use but HTML is fetched anonymously first.
 *
 * FALLBACK CHAIN (fully automatic, zero manual intervention):
 *   1. RSS → full event structure + actuals (~1h delay post-release)
 *   2. HTML anonymous → overlay actuals minutes post-release (non-fatal)
 *   3. If HTML fails → RSS actuals fill in when available (~1h)
 *   4. POST /inject-actuals → emergency manual override (auth required)
 *
 * v5.6 changes (2026-06-19):
 * - G10 extension: NOK/SEK added. G8 aliased to G10 for backward compat.
 *   SLUG_TO_CCY extended: norway→NOK, sweden→SEK.
 *
 * v5.5 changes (2026-06-11):
 * - HTML secondary source now runs unconditionally on every cron tick.
 *   Previously gated on env.MFB_SESSION_COOKIE — now always attempted
 *   anonymously. Confirmed working from CF edge: 200 OK, actuals present.
 * - fetchMyfxbookHTMLActuals() no longer requires env param — fetches
 *   anonymously. Cookie path removed from this function.
 * - /test-html endpoint retained for ongoing diagnostics.
 * - Cron sequence: RSS fetch → HTML merge (non-fatal) → fingerprint → dispatch.
 *
 * v5.4 changes (2026-06-11):
 * - GET /test-html diagnostic endpoint added.
 *
 * v5.3 changes (2026-06-11):
 * - HTML secondary source (cookie-gated), POST /inject-actuals, /payload public.
 *
 * v5.2 changes (2026-06-11):
 * - CRON every 30 min → every 1 min.
 *
 * v5.0 changes (2026-06-11):
 * - PRIMARY replaced: Myfxbook HTML → RSS (HTML blocked from CF edge at that time).
 *
 * REQUIRED SECRETS: GITHUB_PAT, GITHUB_OWNER, GITHUB_REPO
 * OPTIONAL SECRETS: PAYLOAD_TOKEN (for /inject-actuals auth)
 *                   MFB_SESSION_COOKIE (no longer needed — kept for future use)
 */

// ── Constants ─────────────────────────────────────────────────────────────────

const G10 = new Set(["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "NOK", "SEK"]);
const G8 = G10; // alias — G10 active universe since v5.6

const MFB_RSS_URL  = "https://www.myfxbook.com/rss/forex-economic-calendar-events";
const MFB_HTML_URL = "https://www.myfxbook.com/forex-economic-calendar";

const KV_FINGERPRINT = "calendar:fingerprint:v5";
const KV_PAYLOAD     = "calendar:payload:v5";

const SLUG_TO_CCY = {
  "united-states": "USD", "euro-area": "EUR", "germany": "EUR",
  "france": "EUR", "italy": "EUR", "spain": "EUR", "netherlands": "EUR",
  "belgium": "EUR", "ireland": "EUR", "portugal": "EUR", "finland": "EUR",
  "austria": "EUR", "greece": "EUR", "european-union": "EUR",
  "united-kingdom": "GBP", "japan": "JPY", "canada": "CAD",
  "australia": "AUD", "new-zealand": "NZD", "switzerland": "CHF",
  "norway": "NOK", "sweden": "SEK",
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
      return new Response(payload || "[]", { headers: { "Content-Type": "application/json" } });
    }

    if (url.pathname === "/inject-actuals" && request.method === "POST") {
      return handleInjectActuals(request, env);
    }

    if (url.pathname === "/reset") {
      await env.CALENDAR_KV.delete(KV_FINGERPRINT);
      await env.CALENDAR_KV.delete(KV_PAYLOAD);
      return new Response(JSON.stringify({ status: "reset", ts: new Date().toISOString() }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/test-html") {
      return handleTestHTML(env);
    }

    return new Response(JSON.stringify({
      worker:    "calendar-watcher",
      version:   "5.5",
      sources:   ["Myfxbook RSS (primary — full events)", "Myfxbook HTML anonymous (secondary — fast actuals)"],
      schedule:  "every 1 minute",
      endpoints: ["/trigger", "/fingerprint", "/payload", "/inject-actuals (POST)", "/reset", "/test-html"],
      ts:        new Date().toISOString(),
    }), { headers: { "Content-Type": "application/json" } });
  },
};

// ── Core cron logic ───────────────────────────────────────────────────────────

async function runCalendarWatch(env) {
  const now   = new Date();
  const label = `[${now.toISOString().slice(0, 16)}Z]`;
  console.log(`${label} calendar-watcher v5.5: starting poll`);

  // ── Step 1: RSS — full event structure ───────────────────────────────────
  let events = [];
  try {
    events = await fetchMyfxbookRSS();
    console.log(`${label} RSS: ${events.length} G10 med/high events (${events.filter(e => e.actual).length} with actuals)`);
  } catch (err) {
    console.error(`${label} RSS failed: ${err.message} — aborting poll.`);
    return;
  }

  if (!events.length) {
    console.log(`${label} No G10 med/high events — skipping.`);
    return;
  }

  // ── Step 2: HTML — fast actuals overlay (non-fatal) ──────────────────────
  try {
    const htmlActuals = await fetchMyfxbookHTMLActuals();
    const merged = mergeHTMLActuals(events, htmlActuals);
    const total  = Object.keys(htmlActuals).length;
    if (merged > 0) {
      console.log(`${label} HTML: merged ${merged} new actual(s) (${total} found in page)`);
    } else {
      console.log(`${label} HTML: ${total} actuals in page, 0 new merges (RSS already had them or no match)`);
    }
  } catch (err) {
    console.warn(`${label} HTML fetch failed (non-fatal, RSS fallback active): ${err.message}`);
  }

  // ── Step 3: Fingerprint + dispatch if changed ─────────────────────────────
  const newFP  = buildFingerprint(events);
  const prevFP = (await env.CALENDAR_KV.get(KV_FINGERPRINT)) ?? "";

  if (newFP === prevFP) {
    console.log(`${label} No change detected.`);
    return;
  }

  const diff = describeDiff(prevFP, newFP);
  console.log(`${label} CHANGE: ${diff}`);

  await env.CALENDAR_KV.put(KV_FINGERPRINT, newFP,              { expirationTtl: 60 * 60 * 24 * 7 });
  await env.CALENDAR_KV.put(KV_PAYLOAD, JSON.stringify(events), { expirationTtl: 60 * 60 * 24 * 7 });

  try {
    await triggerGitHubWorkflow(env, diff, "myfxbook-rss+html");
    console.log(`${label} repository_dispatch sent.`);
  } catch (err) {
    console.error(`${label} repository_dispatch failed: ${err.message}`);
    // Roll back fingerprint so next cron retries the dispatch
    await env.CALENDAR_KV.put(KV_FINGERPRINT, prevFP, { expirationTtl: 60 * 60 * 24 * 7 });
  }
}

// ── Myfxbook RSS ──────────────────────────────────────────────────────────────

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

  for (const itemM of xml.matchAll(/<item>([\s\S]*?)<\/item>/g)) {
    const block = itemM[1];

    const linkM = block.match(/<link>https?:\/\/[^/]+\/forex-economic-calendar\/([^/]+)\/[^<]+<\/link>/);
    if (!linkM) continue;
    const currency = SLUG_TO_CCY[linkM[1]] || "";
    if (!currency) { skippedCcy++; continue; }

    const titleM = block.match(/<title>([^<]+)<\/title>/);
    const title  = titleM ? titleM[1].trim() : "";
    if (!title) continue;

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

    const descM = block.match(/<description>([\s\S]*?)<\/description>/);
    if (!descM) continue;
    const desc = descM[1]
      .replace(/&#60;/g, "<").replace(/&#62;/g, ">")
      .replace(/&#39;/g, "'").replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">").replace(/&amp;/g, "&");

    const impM   = desc.match(/sprite-(high|medium|low)-impact/);
    const impact = impM ? impM[1] : "low";
    if (impact === "low") { skippedImpact++; continue; }

    const tds = [...desc.matchAll(/<td>([\s\S]*?)<\/td>/g)].map(m =>
      m[1].replace(/<[^>]+>/g, "").trim()
    );

    const timeLeftRaw = tds[0] || "";
    const previous    = cleanVal(tds[2]);
    const forecast    = cleanVal(tds[3]);
    const actual      = cleanVal(tds[4]);
    const isPassed    = timeLeftRaw.startsWith("-") && timeLeftRaw.includes("second");
    const released    = actual !== null || isPassed;

    events.push({ date: `${dateISO} ${timeUTC}:00`, currency, impact, title,
                  actual, forecast, previous, dateISO, timeUTC, released });
  }

  events.sort((a, b) => a.date.localeCompare(b.date) || a.currency.localeCompare(b.currency));
  console.log(`parseMyfxbookRSS: ${events.length} events (${events.filter(e=>e.actual).length} with actuals; skipped ${skippedCcy} non-G10, ${skippedImpact} low-impact)`);
  return events;
}

// ── Myfxbook HTML (anonymous) ─────────────────────────────────────────────────

async function fetchMyfxbookHTMLActuals() {
  const resp = await fetch(MFB_HTML_URL, {
    headers: {
      "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
      "Accept":          "text/html,application/xhtml+xml,*/*",
      "Accept-Language": "en-US,en;q=0.9",
      "Referer":         "https://www.myfxbook.com/",
    },
    signal: AbortSignal.timeout(25000),
  });

  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const html = await resp.text();
  if (!html.includes("economicCalendarTable")) throw new Error("Calendar table not found in HTML response");
  return parseMyfxbookHTMLActuals(html);
}

function parseMyfxbookHTMLActuals(html) {
  const actuals = {};
  let parsed = 0, found = 0;

  for (const rowM of html.matchAll(/<tr[^>]+data-row-id="(\d+)"[^>]*>([\s\S]*?)<\/tr>/g)) {
    const rowBody = rowM[2];
    parsed++;

    const dateM = rowBody.match(/data-calendarDateTd="(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})/);
    if (!dateM) continue;
    const dateISO = dateM[1];
    const timeUTC = dateM[2];

    const ccyM = rowBody.match(/<td[^>]*>\s*([A-Z]{3})\s*<\/td>/);
    if (!ccyM || !G8.has(ccyM[1])) continue;
    const currency = ccyM[1];

    const titleM = rowBody.match(/class="calendar-event-link"[^>]*>([^<]+)<\/a>/);
    if (!titleM) continue;
    const title = titleM[1].trim();

    const actualM = rowBody.match(/class="actualCell"[^>]*>([\s\S]*?)<\/span>/);
    if (!actualM) continue;
    const actualRaw = actualM[1].replace(/<[^>]+>/g, "").trim();
    const actual = cleanVal(actualRaw);
    if (!actual) continue;

    actuals[`${currency}|${dateISO}|${timeUTC}|${normTitle(title)}`] = actual;
    found++;
  }

  console.log(`parseMyfxbookHTMLActuals: parsed ${parsed} rows, found ${found} G8 actuals`);
  return actuals;
}

function mergeHTMLActuals(events, htmlActuals) {
  let merged = 0;
  for (const ev of events) {
    if (ev.actual) continue; // RSS already has it
    const key = `${ev.currency}|${ev.dateISO}|${ev.timeUTC}|${normTitle(ev.title)}`;
    if (htmlActuals[key]) {
      ev.actual   = htmlActuals[key];
      ev.released = true;
      merged++;
    }
  }
  return merged;
}

function normTitle(t) {
  return t.toLowerCase().replace(/\s+/g, " ").trim();
}

// ── /test-html diagnostic ─────────────────────────────────────────────────────

async function handleTestHTML(env) {
  const baseResult = {
    ts:             new Date().toISOString(),
    url:            MFB_HTML_URL,
    cookie_set:     !!env.MFB_SESSION_COOKIE,
    http_status:    null,
    http_error:     null,
    bytes_received: null,
    has_table:      false,
    rows_parsed:    0,
    actuals_found:  0,
    examples:       [],
  };

  // Always test anon; also test with cookie if set
  const attempts = [{ label: "anon", cookie: null }];
  if (env.MFB_SESSION_COOKIE) {
    attempts.push({ label: "cookie", cookie: env.MFB_SESSION_COOKIE });
  }

  const allResults = [];

  for (const attempt of attempts) {
    const r = { ...baseResult, attempt: attempt.label, examples: [] };

    try {
      const headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept":          "text/html,application/xhtml+xml,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.myfxbook.com/",
      };
      if (attempt.cookie) headers["Cookie"] = attempt.cookie;

      const resp = await fetch(MFB_HTML_URL, { headers, signal: AbortSignal.timeout(25000) });
      r.http_status = resp.status;

      if (!resp.ok) {
        r.http_error = `HTTP ${resp.status} ${resp.statusText}`;
        allResults.push(r);
        continue;
      }

      const html = await resp.text();
      r.bytes_received = html.length;
      r.has_table = html.includes("economicCalendarTable");

      if (!r.has_table) {
        r.http_error = "Response OK but calendar table not found";
        allResults.push(r);
        continue;
      }

      for (const rowM of html.matchAll(/<tr[^>]+data-row-id="(\d+)"[^>]*>([\s\S]*?)<\/tr>/g)) {
        const rowBody = rowM[2];
        r.rows_parsed++;

        const dateM = rowBody.match(/data-calendarDateTd="(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})/);
        if (!dateM) continue;
        const ccyM = rowBody.match(/<td[^>]*>\s*([A-Z]{3})\s*<\/td>/);
        if (!ccyM || !G8.has(ccyM[1])) continue;
        const titleM = rowBody.match(/class="calendar-event-link"[^>]*>([^<]+)<\/a>/);
        if (!titleM) continue;
        const actualM = rowBody.match(/class="actualCell"[^>]*>([\s\S]*?)<\/span>/);
        if (!actualM) continue;
        const actual = cleanVal(actualM[1].replace(/<[^>]+>/g, "").trim());
        if (!actual) continue;

        r.actuals_found++;
        if (r.examples.length < 5) {
          r.examples.push({
            currency: ccyM[1], dateISO: dateM[1], timeUTC: dateM[2],
            title: titleM[1].trim(), actual,
          });
        }
      }
    } catch (err) {
      r.http_error = err.message;
    }

    allResults.push(r);
  }

  return new Response(JSON.stringify(allResults, null, 2), {
    headers: { "Content-Type": "application/json" },
  });
}

// ── /inject-actuals ───────────────────────────────────────────────────────────

async function handleInjectActuals(request, env) {
  const authHeader = request.headers.get("Authorization") || "";
  const expected   = `Bearer ${env.PAYLOAD_TOKEN || ""}`;
  if (!env.PAYLOAD_TOKEN || authHeader !== expected) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401, headers: { "Content-Type": "application/json" },
    });
  }

  let injected;
  try {
    injected = await request.json();
    if (!Array.isArray(injected)) throw new Error("Expected array");
  } catch (e) {
    return new Response(JSON.stringify({ error: `Invalid JSON: ${e.message}` }), {
      status: 400, headers: { "Content-Type": "application/json" },
    });
  }

  const raw    = await env.CALENDAR_KV.get(KV_PAYLOAD);
  const events = raw ? JSON.parse(raw) : [];
  let merged = 0;

  for (const inj of injected) {
    const { currency, dateISO, timeUTC, title, actual } = inj;
    if (!currency || !dateISO || !title || !actual) continue;
    const ev = events.find(e =>
      e.currency === currency && e.dateISO === dateISO &&
      normTitle(e.title) === normTitle(title)
    );
    if (ev && !ev.actual) {
      ev.actual = actual; ev.released = true;
      if (timeUTC) ev.timeUTC = timeUTC;
      merged++;
    }
  }

  if (!merged) {
    return new Response(JSON.stringify({ status: "no_match", merged: 0 }), {
      headers: { "Content-Type": "application/json" },
    });
  }

  const newFP = buildFingerprint(events);
  await env.CALENDAR_KV.put(KV_FINGERPRINT, newFP,              { expirationTtl: 60 * 60 * 24 * 7 });
  await env.CALENDAR_KV.put(KV_PAYLOAD, JSON.stringify(events), { expirationTtl: 60 * 60 * 24 * 7 });

  try {
    await triggerGitHubWorkflow(env, `manual inject: ${merged} actual(s)`, "manual-inject");
  } catch (err) {
    console.error(`inject-actuals: dispatch failed: ${err.message}`);
  }

  return new Response(JSON.stringify({ status: "ok", merged }), {
    headers: { "Content-Type": "application/json" },
  });
}

// ── Fingerprint ───────────────────────────────────────────────────────────────

function buildFingerprint(events) {
  return events
    .map(ev => `${ev.date}|${ev.currency}|${ev.title}|${ev.actual ?? ""}|${ev.forecast ?? ""}`)
    .sort().join("\n");
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
      .map(l => { const p = l.split("|"); return `${p[1]} ${p[2].slice(0,25)}=${p[3]}`; });
    parts.push(`${newActuals.length} actual(s): ${samples.join(", ")}`);
  }
  if (newForecasts.length) parts.push(`${newForecasts.length} forecast(s)`);
  return parts.join(" | ") || `${added.length} change(s)`;
}

// ── GitHub dispatch ───────────────────────────────────────────────────────────

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
        "User-Agent":           "globalinvesting-calendar-watcher/5.5",
        "X-GitHub-Api-Version": "2022-11-28",
      },
      body: JSON.stringify({
        event_type: "calendar-data-changed",
        client_payload: { description, triggered_at: new Date().toISOString(),
                          source: `cloudflare-worker-${source}` },
      }),
      signal: AbortSignal.timeout(10000),
    }
  );

  if (resp.status !== 204) {
    const body = await resp.text().catch(() => "(unreadable)");
    console.error(`GitHub dispatch URL: https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/dispatches`);
    console.error(`GitHub API ${resp.status} body: ${body}`);
    throw new Error(`GitHub API ${resp.status}: ${body}`);
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function cleanVal(v) {
  if (v == null) return null;
  const s = String(v).trim();
  return s === "" || s === "None" || s === "null" || s === "N/A" || s === "—" || s === "-" ? null : s;
}
