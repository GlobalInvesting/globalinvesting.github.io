// ═══════════════════════════════════════════════════════════════════
// check-consistency.js — Automated doc↔code↔UI consistency checks
//
// Codifies three drift patterns that have each caused real production
// bugs in this project's history (see CHANGELOG.md for the incidents
// referenced inline below) into checks that fail loudly instead of
// silently. This does NOT replace dashboard.test.js (behavior tests) —
// it checks that files which must be kept in sync with each other
// actually are, which is a different failure mode.
//
// Run with: node scripts/check-consistency.js
// Exits 0 on all-pass, 1 on any failure (safe for CI / pre-deploy gating).
// Run from the repo root.
// ═══════════════════════════════════════════════════════════════════

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const read = (rel) => fs.readFileSync(path.join(ROOT, rel), 'utf8');
const listHtmlFiles = () =>
  fs.readdirSync(ROOT).filter((f) => f.endsWith('.html'));

let failures = 0;
const fail = (msg) => { failures++; console.error(`✗ ${msg}`); };
const pass = (msg) => console.log(`✓ ${msg}`);

// ── Check 1: index.html versioned assets ↔ sw.js STATIC_PRECACHE ──────────
// Real incident: STATIC_PRECACHE was stuck at v8.21.0 for many releases,
// including a filename (dashboard-v2.css) already renamed in index.html.
// cache.addAll() is all-or-nothing, so that one 404 silently failed the
// ENTIRE service-worker install() every time (see sw.js header comment).
function checkCacheBusterSync() {
  const indexHtml = read('index.html');
  const swJs = read('sw.js');

  const assetRe = /(?:src|href)="((?:assets\/)?[a-zA-Z0-9_-]+\.(?:js|css))(\?v=[^"]+)?"/g;
  const indexAssets = new Map(); // filename -> versioned ref (or null if unversioned)
  let m;
  while ((m = assetRe.exec(indexHtml))) {
    const file = m[1].replace(/^assets\//, '');
    if (m[2]) indexAssets.set(file, `${file}${m[2]}`);
  }

  const precacheMatch = swJs.match(/const STATIC_PRECACHE\s*=\s*\[([\s\S]*?)\];/);
  if (!precacheMatch) {
    fail('sw.js: could not locate STATIC_PRECACHE array');
    return;
  }
  const precacheEntries = [...precacheMatch[1].matchAll(/'\/(?:assets\/)?([a-zA-Z0-9_-]+\.(?:js|css))(\?v=[^']+)?'/g)];
  const precacheAssets = new Map();
  for (const entry of precacheEntries) {
    if (entry[2]) precacheAssets.set(entry[1], `${entry[1]}${entry[2]}`);
  }

  let ok = true;
  for (const [file, ref] of indexAssets) {
    if (!precacheAssets.has(file)) {
      fail(`cache-buster: index.html references ${ref} but it's missing from sw.js STATIC_PRECACHE entirely`);
      ok = false;
    } else if (precacheAssets.get(file) !== ref) {
      fail(`cache-buster: version mismatch for ${file} — index.html has "${ref}", sw.js STATIC_PRECACHE has "${precacheAssets.get(file)}"`);
      ok = false;
    }
  }
  for (const [file, ref] of precacheAssets) {
    if (!indexAssets.has(file)) {
      fail(`cache-buster: sw.js STATIC_PRECACHE has "${ref}" but index.html no longer references ${file} — stale/orphaned entry`);
      ok = false;
    }
  }
  if (ok) pass(`cache-buster sync: all ${indexAssets.size} versioned assets match between index.html and sw.js`);
}

// ── Check 2: fetch('*-data/...') prefixes ↔ sw.js DATA_PATH_PREFIXES ──────
// Same failure family as Check 1: a new data folder wired into a fetch()
// call that never gets added to DATA_PATH_PREFIXES silently falls through
// to whatever caching strategy applies to "everything else", instead of
// the network-first behavior data endpoints require.
function checkDataPathCoverage() {
  const swJs = read('sw.js');
  const prefixMatch = swJs.match(/const DATA_PATH_PREFIXES\s*=\s*\[([\s\S]*?)\];/);
  if (!prefixMatch) {
    fail('sw.js: could not locate DATA_PATH_PREFIXES array');
    return;
  }
  const prefixes = [...prefixMatch[1].matchAll(/'([^']+)'/g)].map((x) => x[1]);

  const assetsDir = path.join(ROOT, 'assets');
  const jsFiles = fs.readdirSync(assetsDir).filter((f) => f.endsWith('.js'));
  const referenced = new Set();
  const fetchRe = /fetch\(\s*[`'"]\.?\/?([a-zA-Z0-9_-]+-data)\//g;
  for (const f of jsFiles) {
    const src = fs.readFileSync(path.join(assetsDir, f), 'utf8');
    let m;
    while ((m = fetchRe.exec(src))) referenced.add(m[1]);
  }

  let ok = true;
  for (const dir of referenced) {
    const covered = prefixes.some((p) => p === `/${dir}/`);
    if (!covered) {
      fail(`data-path coverage: assets/*.js fetches from "${dir}/" but sw.js DATA_PATH_PREFIXES has no "/${dir}/" entry — it will get the wrong caching strategy`);
      ok = false;
    }
  }
  if (ok) pass(`data-path coverage: all ${referenced.size} "-data/" fetch prefixes are covered in sw.js DATA_PATH_PREFIXES`);
}

// ── Check 3: FAQPage JSON-LD text ↔ visible qa-a text ──────────────────────
// Two failure modes here, both real:
//   (a) Real incident (v8.295.0): the broker-compensation disclosure was
//       added to the visible FAQ answer but initially left out of the
//       matching JSON-LD "text" field — the two are independent copies of
//       the same sentence with no shared source, so nothing forces them
//       to agree.
//   (b) A FAQPage entity with NO matching visible text anywhere on the
//       page at all. This isn't just a doc/UI drift risk — since Google's
//       August 2023 structured-data policy change, FAQPage rich results
//       are restricted to well-known, authoritative sites UNLESS the
//       Q&A content is genuinely visible to the user on the page; content
//       that exists only in JSON-LD is against Google's own guidance and
//       can make the markup ineligible for rich results (or worse, read
//       as manipulative structured data). See
//       https://developers.google.com/search/blog/2023/08/howsearchworks-structured-data
function checkFaqJsonLdSync() {
  let ok = true;
  let checkedFiles = 0;
  let checkedPairs = 0;
  const invisibleByFile = new Map();

  for (const file of listHtmlFiles()) {
    const html = read(file);
    if (!html.includes('"@type": "FAQPage"') && !html.includes('"@type":"FAQPage"')) continue;

    const ldBlocks = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)];
    let faqEntities = [];
    for (const block of ldBlocks) {
      let json;
      try { json = JSON.parse(block[1]); } catch { continue; }
      if (json['@type'] === 'FAQPage' && Array.isArray(json.mainEntity)) {
        faqEntities = faqEntities.concat(json.mainEntity);
      }
    }
    if (faqEntities.length === 0) continue;
    checkedFiles++;

    // Visible pairs: two markup conventions are valid in this codebase and
    // BOTH are checked — a page can use either (or, in principle, both).
    // (1) access.html's static <div class="qa-q">/<div class="qa-a"> pair.
    // (2) The <details class="faq-item"><summary>/<p> accordion used on
    //     index.html and the guide-*.html pages (v8.297.0) — native HTML5,
    //     no JS required, and collapsible, which the static qa-box pattern
    //     isn't — a better fit once a page has more than a handful of FAQs.
    // Do not treat either pattern as "the" canonical one going forward:
    // whichever fits a given page's FAQ count/design is fine, as long as
    // ITS OWN visible text matches its own JSON-LD entity.
    // Visible HTML must legally escape &, <, >, " as entities; the JSON-LD
    // string doesn't (it's JSON, not HTML) — decode before comparing, or
    // every visible answer containing a literal "&" false-fails here.
    const decodeEntities = (s) => s
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'");
    const qaRe = /<div class="qa-q">([\s\S]*?)<\/div>\s*<div class="qa-a">([\s\S]*?)<\/div>/g;
    const detailsRe = /<details class="faq-item"[^>]*>\s*<summary>([\s\S]*?)<\/summary>\s*<p>([\s\S]*?)<\/p>\s*<\/details>/g;
    const visiblePairs = new Map();
    let m;
    while ((m = qaRe.exec(html))) {
      const q = decodeEntities(m[1].replace(/<[^>]+>/g, '').trim());
      const a = decodeEntities(m[2].replace(/<[^>]+>/g, '').trim());
      visiblePairs.set(q, a);
    }
    while ((m = detailsRe.exec(html))) {
      const q = decodeEntities(m[1].replace(/<[^>]+>/g, '').trim());
      const a = decodeEntities(m[2].replace(/<[^>]+>/g, '').trim());
      visiblePairs.set(q, a);
    }

    for (const entity of faqEntities) {
      const q = (entity.name || '').trim();
      const ldAnswer = (entity.acceptedAnswer && entity.acceptedAnswer.text || '').trim();

      if (visiblePairs.has(q)) {
        checkedPairs++;
        const visibleAnswer = visiblePairs.get(q);
        if (visibleAnswer !== ldAnswer) {
          fail(`${file}: FAQ JSON-LD text differs from visible text for "${q}"\n    visible: ${visibleAnswer}\n    JSON-LD: ${ldAnswer}`);
          ok = false;
        }
        continue;
      }

      // No qa-q/qa-a match. Last resort: does the question text appear
      // ANYWHERE else in the raw HTML (some other markup pattern)? If it
      // appears nowhere but the JSON-LD itself, the content is invisible.
      const occurrences = html.split(q).length - 1;
      if (occurrences <= 1) {
        if (!invisibleByFile.has(file)) invisibleByFile.set(file, []);
        invisibleByFile.get(file).push(q);
        ok = false;
      }
    }
  }

  if (invisibleByFile.size > 0) {
    let total = 0;
    for (const [file, qs] of invisibleByFile) {
      total += qs.length;
      fail(`${file}: ${qs.length} FAQPage question(s) exist ONLY in JSON-LD — no matching visible text on the page (Google structured-data policy violation, not just a sync issue)`);
    }
    fail(`FAQ visibility: ${total} FAQ entries across ${invisibleByFile.size} file(s) are invisible to users — see GUIDELINES.md for remediation plan`);
  }
  if (ok) pass(`FAQ JSON-LD sync: ${checkedPairs} question/answer pairs match across ${checkedFiles} files, none invisible`);
}

checkCacheBusterSync();
checkDataPathCoverage();
checkFaqJsonLdSync();

console.log('');
if (failures > 0) {
  console.error(`${failures} consistency check(s) failed.`);
  process.exit(1);
} else {
  console.log('All consistency checks passed.');
  process.exit(0);
}
