const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const read = (rel) => fs.readFileSync(path.join(ROOT, rel), 'utf8');
const listHtmlFiles = () =>
  fs.readdirSync(ROOT).filter((f) => f.endsWith('.html'));

let failures = 0;
const fail = (msg) => { failures++; console.error(`✗ ${msg}`); };
const pass = (msg) => console.log(`✓ ${msg}`);

function checkCacheBusterSync() {
  const indexHtml = read('index.html');
  const swJs = read('sw.js');

  const assetRe = /(?:src|href)="((?:assets\/)?[a-zA-Z0-9_-]+\.(?:js|css))(\?v=[^"]+)?"/g;
  const indexAssets = new Map();
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
