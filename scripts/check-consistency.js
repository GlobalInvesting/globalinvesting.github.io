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

function checkNoScriptsInWorkflowsDir() {
  const workflowsDir = path.join(ROOT, '.github', 'workflows');
  if (!fs.existsSync(workflowsDir)) {
    pass('workflows dir hygiene: .github/workflows/ not present, nothing to check');
    return;
  }
  const stray = fs.readdirSync(workflowsDir).filter((f) => f.endsWith('.py') || f.endsWith('.js'));
  if (stray.length > 0) {
    fail(`workflows dir hygiene: ${stray.length} script file(s) directly under .github/workflows/ (${stray.join(', ')}) — GitHub Actions never executes scripts from this location; this is always a stale orphaned copy left over from a prior session, not a working file. Delete it.`);
  } else {
    pass('workflows dir hygiene: no stray .py/.js files directly under .github/workflows/');
  }
}

function scanJsComments(src) {
  const comments = [];
  let i = 0;
  const n = src.length;
  let lastSignificant = '';
  while (i < n) {
    const c = src[i];
    const c2 = src[i + 1];
    if (c === '/' && c2 === '/') {
      const start = i;
      let j = i + 2;
      while (j < n && src[j] !== '\n') j++;
      comments.push({ start, line: src.slice(0, i).split('\n').length });
      i = j;
      continue;
    }
    if (c === '/' && c2 === '*') {
      const start = i;
      let j = src.indexOf('*/', i + 2);
      j = j === -1 ? n : j + 2;
      comments.push({ start, line: src.slice(0, i).split('\n').length });
      i = j;
      continue;
    }
    if (c === '"' || c === "'") {
      const quote = c;
      let j = i + 1;
      while (j < n) {
        if (src[j] === '\\') { j += 2; continue; }
        if (src[j] === quote) { j++; break; }
        if (src[j] === '\n') break;
        j++;
      }
      i = j;
      lastSignificant = quote;
      continue;
    }
    if (c === '`') {
      let j = i + 1;
      let depth = 0;
      while (j < n) {
        if (src[j] === '\\') { j += 2; continue; }
        if (src[j] === '`' && depth === 0) { j++; break; }
        if (src[j] === '$' && src[j + 1] === '{') { depth++; j += 2; continue; }
        if (src[j] === '}' && depth > 0) { depth--; j++; continue; }
        j++;
      }
      i = j;
      lastSignificant = '`';
      continue;
    }
    if (c === '/') {
      const kw = ['return', 'typeof', 'case', 'in', 'of', 'instanceof', 'new', 'delete', 'void', 'throw', 'yield', 'do', 'else'];
      const regexAllowed = !/[\w)\]]/.test(lastSignificant) || kw.includes(lastSignificant);
      if (regexAllowed) {
        let j = i + 1;
        let inClass = false;
        let ok = false;
        while (j < n) {
          if (src[j] === '\\') { j += 2; continue; }
          if (src[j] === '[') { inClass = true; j++; continue; }
          if (src[j] === ']') { inClass = false; j++; continue; }
          if (src[j] === '/' && !inClass) { ok = true; j++; break; }
          if (src[j] === '\n') break;
          j++;
        }
        if (ok) {
          while (j < n && /[a-z]/i.test(src[j])) j++;
          i = j;
          lastSignificant = '/';
          continue;
        }
      }
    }
    if (!/\s/.test(c)) lastSignificant = c;
    i++;
  }
  return comments;
}

function scanCssComments(src) {
  const comments = [];
  let i = 0;
  const n = src.length;
  while (i < n) {
    if (src[i] === '/' && src[i + 1] === '*') {
      const start = i;
      let j = src.indexOf('*/', i + 2);
      j = j === -1 ? n : j + 2;
      comments.push({ start, line: src.slice(0, i).split('\n').length });
      i = j;
      continue;
    }
    if (src[i] === '"' || src[i] === "'") {
      const quote = src[i];
      let j = i + 1;
      while (j < n) {
        if (src[j] === '\\') { j += 2; continue; }
        if (src[j] === quote) { j++; break; }
        j++;
      }
      i = j;
      continue;
    }
    i++;
  }
  return comments;
}

function scanPyComments(src) {
  const comments = [];
  let i = 0;
  const n = src.length;
  while (i < n) {
    if (src[i] === '#') {
      const start = i;
      let j = i + 1;
      while (j < n && src[j] !== '\n') j++;
      comments.push({ start, line: src.slice(0, i).split('\n').length });
      i = j;
      continue;
    }
    if (src[i] === '"' || src[i] === "'") {
      const triple = src.substr(i, 3) === src[i].repeat(3);
      const quote = triple ? src[i].repeat(3) : src[i];
      let j = i + quote.length;
      while (j < n) {
        if (src[j] === '\\') { j += 2; continue; }
        if (src.substr(j, quote.length) === quote) { j += quote.length; break; }
        if (!triple && src[j] === '\n') break;
        j++;
      }
      i = j;
      continue;
    }
    i++;
  }
  return comments;
}

function scanYamlCommentCandidates(src) {
  const comments = [];
  let i = 0;
  const n = src.length;
  let inSingle = false, inDouble = false;
  while (i < n) {
    const c = src[i];
    if (inSingle) {
      if (c === "'" && src[i + 1] === "'") { i += 2; continue; }
      if (c === "'") { inSingle = false; i++; continue; }
      i++; continue;
    }
    if (inDouble) {
      if (c === '\\') { i += 2; continue; }
      if (c === '"') { inDouble = false; i++; continue; }
      i++; continue;
    }
    if (c === "'") { inSingle = true; i++; continue; }
    if (c === '"') { inDouble = true; i++; continue; }
    if (c === '#') {
      const prev = i === 0 ? '\n' : src[i - 1];
      if (/\s/.test(prev)) {
        comments.push({ start: i, line: src.slice(0, i).split('\n').length });
        let j = i + 1;
        while (j < n && src[j] !== '\n') j++;
        i = j;
        continue;
      }
    }
    i++;
  }
  return comments;
}

function checkNoCommentsInPublicRepo() {
  let ok = true;
  let filesChecked = 0;

  const jsDirs = ['assets', 'scripts'];
  for (const dir of jsDirs) {
    const full = path.join(ROOT, dir);
    if (!fs.existsSync(full)) continue;
    for (const f of fs.readdirSync(full).filter((x) => x.endsWith('.js'))) {
      filesChecked++;
      const src = fs.readFileSync(path.join(full, f), 'utf8');
      const hits = scanJsComments(src);
      if (hits.length > 0) {
        fail(`zero-comments (public repo, v8.412.0): ${dir}/${f} has ${hits.length} real comment(s) — first at line ${hits[0].line}. Public-repo code files may carry zero comments of any kind.`);
        ok = false;
      }
    }
  }

  const assetsDir = path.join(ROOT, 'assets');
  if (fs.existsSync(assetsDir)) {
    for (const f of fs.readdirSync(assetsDir).filter((x) => x.endsWith('.css'))) {
      filesChecked++;
      const src = fs.readFileSync(path.join(assetsDir, f), 'utf8');
      const hits = scanCssComments(src);
      if (hits.length > 0) {
        fail(`zero-comments (public repo, v8.412.0): assets/${f} has ${hits.length} real comment(s) — first at line ${hits[0].line}.`);
        ok = false;
      }
    }
  }

  const scriptsDir = path.join(ROOT, 'scripts');
  if (fs.existsSync(scriptsDir)) {
    for (const f of fs.readdirSync(scriptsDir).filter((x) => x.endsWith('.py'))) {
      filesChecked++;
      const src = fs.readFileSync(path.join(scriptsDir, f), 'utf8');
      const hits = scanPyComments(src);
      if (hits.length > 0) {
        fail(`zero-comments (public repo, v8.412.0): scripts/${f} has ${hits.length} real comment(s) — first at line ${hits[0].line}.`);
        ok = false;
      }
    }
  }

  const workflowsDir = path.join(ROOT, '.github', 'workflows');
  if (fs.existsSync(workflowsDir)) {
    for (const f of fs.readdirSync(workflowsDir).filter((x) => x.endsWith('.yml') || x.endsWith('.yaml'))) {
      filesChecked++;
      const src = fs.readFileSync(path.join(workflowsDir, f), 'utf8');
      const hits = scanYamlCommentCandidates(src);
      if (hits.length > 0) {
        fail(`zero-comments (public repo, v8.412.0): .github/workflows/${f} has ${hits.length} likely comment(s) — first at line ${hits[0].line}. YAML detection is heuristic (whitespace-preceded '#'); confirm by eye before dismissing as a false positive (e.g. bash '\${VAR#pattern}' or '10#$N' base-conversion are correctly excluded, but a rare quoted-string edge case could still slip through).`);
        ok = false;
      }
    }
  }

  if (ok) pass(`zero-comments (public repo, v8.412.0): 0 comments found across ${filesChecked} checked files (assets/*.js, assets/*.css, scripts/*.js, scripts/*.py, .github/workflows/*.yml)`);
}

checkCacheBusterSync();
checkDataPathCoverage();
checkFaqJsonLdSync();
checkNoScriptsInWorkflowsDir();
checkNoCommentsInPublicRepo();

console.log('');
if (failures > 0) {
  console.error(`${failures} consistency check(s) failed.`);
  process.exit(1);
} else {
  console.log('All consistency checks passed.');
  process.exit(0);
}
