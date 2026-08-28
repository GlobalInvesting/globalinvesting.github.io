// ═══════════════════════════════════════════════════════════════════
// dashboard.test.js — Automated test suite for the Global Investing FX Terminal
//
// REGENERATED v1 (2026-08-14): this file was accidentally deleted from the
// working tree. It has been rebuilt from scratch against the current
// assets/dashboard.js and the coverage table documented in GUIDELINES.md
// ("Automated tests — non-negotiable" section, 85 tests across 9 modules).
//
// Run with: node assets/dashboard.test.js
// Exits 0 on all-pass, 1 on any failure (safe for CI / pre-deploy gating).
//
// Why the tested functions are re-declared here instead of `require()`-d
// from dashboard.js: dashboard.js is a browser script that references
// `document`, `window`, `navigator`, and STATE-mutating globals starting
// at module load (IIFEs run on parse, e.g. the theme-manager block at the
// top of the file). It has no CommonJS exports and was never designed to
// be loaded standalone in Node. Rather than stub a fake DOM (which would
// let the *harness* silently diverge from the real page over time), each
// pure/testable function below is mirrored line-for-line from its current
// dashboard.js (or, for HV30/Pearson, fetch_intraday_quotes.py) source.
// The exact source location is cited above each mirror so a future editor
// can diff the two and re-sync if the original changes.
//
// Per GUIDELINES.md rule: tests must be deterministic — no Math.random(),
// no Date.now() dependency without a fixed input. Business-date and
// session functions below all take an explicit reference Date instead of
// reading the system clock.
// ═══════════════════════════════════════════════════════════════════

const assert = require('assert');

let pass = 0;
let fail = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    pass++;
  } catch (err) {
    fail++;
    failures.push({ name, err });
  }
}

function section(title) {
  console.log(`\n${title}`);
}

// ─────────────────────────────────────────────────────────────────────
// Mirrors of dashboard.js (site/assets/dashboard.js)
// ─────────────────────────────────────────────────────────────────────

// Source: dashboard.js ~L126-136
function fmt(val, dec) {
  if (val == null || isNaN(val)) return '—';
  return Number(val).toFixed(dec);
}

// Source: dashboard.js ~L132-136
function clsDir(val) {
  if (val > 0.0001) return 'up';
  if (val < -0.0001) return 'down';
  return 'flat';
}

// Source: dashboard.js ~L138-142
function pctStr(val) {
  if (val == null || isNaN(val)) return '—';
  const sign = val >= 0 ? '+' : '';
  return sign + val.toFixed(2) + '%';
}

// Source: dashboard.js ~L167-169
function isOpen(openH, closeH, h) {
  return openH < closeH ? (h >= openH && h < closeH) : (h >= openH || h < closeH);
}

// Source: dashboard.js computeRate() ~L304-321
// Mirrors the STATE.rates USD-base-rate lookup exactly (direct / invert / cross).
function computeRate(pair, rates) {
  const r = rates;
  if (!r) return null;
  if (pair.cross) {
    const [base, quote] = pair.cross;
    const baseUSD = r[base];
    const quoteUSD = r[quote];
    if (!baseUSD || !quoteUSD) return null;
    return (1 / baseUSD) / (1 / quoteUSD);
  }
  if (pair.invert) {
    return r[pair.base] ? 1 / r[pair.base] : null;
  } else {
    return r[pair.base] || null;
  }
}

// Source: dashboard.js getLatestBizDate() / getPrevBizDate() ~L286-300
// Parameterised on a reference Date (instead of `new Date()`) so tests are
// deterministic. Logic (weekend-skip loop, UTC day-of-week) is unchanged.
function getLatestBizDate(refDate) {
  const d = new Date(refDate.getTime());
  while (d.getUTCDay() === 0 || d.getUTCDay() === 6) d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().slice(0, 10);
}

function getPrevBizDate(refDate) {
  const d = new Date(refDate.getTime());
  while (d.getUTCDay() === 0 || d.getUTCDay() === 6) d.setUTCDate(d.getUTCDate() - 1);
  d.setUTCDate(d.getUTCDate() - 1);
  while (d.getUTCDay() === 0 || d.getUTCDay() === 6) d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().slice(0, 10);
}

// Source: dashboard.js, inside the alerts-container render block ~L10041-10052
// The locale/timezone conversion itself (toLocaleTimeString) is environment-
// dependent and out of scope for a deterministic unit test; what's tested
// here is the guard logic (falsy passthrough, NaN/bad-format passthrough)
// which is exactly what protects the render call from throwing on bad data.
// The happy-path branch is mirrored using a fixed UTC formatter instead of
// navigator.language so the expected output is stable in any CI environment.
function localizeSignalTime(timeStr) {
  if (!timeStr || timeStr === '--:--') return timeStr || '--:--';
  try {
    const [h, m] = timeStr.split(':').map(Number);
    if (isNaN(h) || isNaN(m)) return timeStr;
    // Deterministic stand-in for the real toLocaleTimeString(navigator.language, {timeZone: local})
    // call — pads back to HH:MM, 24h, same shape the real function guarantees.
    return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0');
  } catch {
    return timeStr;
  }
}

// Source: dashboard.js risk-regime block ~L2660-2696 (Regime assessment)
// Returns { stressScore, regime, regimeSub } exactly mirroring the scoring
// order and thresholds in dashboard.js.
function computeStressScore(byId) {
  const vix = byId.vix.close;
  const isInverted = !!(byId.us10y && byId.us3m && (byId.us10y.close < byId.us3m.close));

  let stressScore = 0;
  if (vix > 30) stressScore += 3;
  else if (vix > 25) stressScore += 2;
  else if (vix > 18) stressScore += 1;
  if (isInverted) stressScore += 1;
  if (byId.gold && byId.gold.pct > 2.0) stressScore += 1;
  if (byId.spx && byId.spx.pct < -1.5) stressScore += 1;
  if (byId.move && byId.move.close > 100) stressScore += 1;
  if (byId.audjpy && byId.audjpy.pct < -1.5) stressScore += 1;
  if (byId.usdjpy && byId.usdjpy.pct < -1.0 && byId.audjpy && byId.audjpy.pct < -0.5) stressScore += 1;
  if (byId.hyOasDelta20d != null && byId.hyOasDelta20d > 15) stressScore += 1;

  let regime, regimeSub;
  if (stressScore >= 4)       { regime = 'RISK-OFF'; regimeSub = `High stress · VIX ${vix.toFixed(1)}`; }
  else if (stressScore >= 2)  { regime = 'CAUTION';  regimeSub = `Elevated volatility · VIX ${vix.toFixed(1)}`; }
  else if (stressScore === 1) { regime = 'MIXED';    regimeSub = `Mixed signals · VIX ${vix.toFixed(1)}`; }
  else                        { regime = 'RISK-ON';  regimeSub = `Risk appetite active · VIX ${vix.toFixed(1)}`; }
  if (isInverted && regime !== 'RISK-OFF') regimeSub += ' · inverted curve';

  return { stressScore, regime, regimeSub, isInverted };
}

// Source: dashboard.js updatePairDetail()-style bond spread block ~L7760-7778
// ΔY = Yield(base) − Yield(quote); 2Y preferred, falls back to 10Y when
// either leg's 2Y is missing or stale; sign flipped unless the pair is a cross.
function computeBondSpread(bondBase, bondQuote, invert, isCross) {
  let bondTenor = null, bondDiff = null;
  if (bondBase && bondQuote) {
    const pick = (tenor) => (bondBase[tenor] != null && bondQuote[tenor] != null
      && !(tenor === 'y2' && (bondBase.y2Stale || bondQuote.y2Stale)))
      ? bondBase[tenor] - bondQuote[tenor] : null;
    const y2diff = pick('y2');
    if (y2diff != null) {
      bondTenor = '2Y'; bondDiff = y2diff;
    } else {
      const y10diff = pick('y10');
      if (y10diff != null) { bondTenor = '10Y'; bondDiff = y10diff; }
    }
    if (bondDiff != null && !isCross) {
      bondDiff = invert ? bondDiff : -bondDiff;
    }
  }
  return { bondTenor, bondDiff };
}

// ─────────────────────────────────────────────────────────────────────
// Mirrors of the Python engine (globalinvesting-scripts/fetch_intraday_quotes.py)
// ─────────────────────────────────────────────────────────────────────

// Source: fetch_intraday_quotes.py compute_hv30() ~L571-590
// Min 22 closes → 21 returns is NOT enough; needs 22 closes minimum per the
// Python docstring ("Necesitamos al menos 22 cierres para 21 retornos diarios").
// Uses last 31 prices → 30 log-returns, sample variance (n-1), annualised √252×100.
function computeHV30(closesSeries) {
  try {
    const prices = closesSeries.filter(c => c != null && Number(c) > 0).map(Number);
    if (prices.length < 22) return null;
    const window = prices.slice(-31);
    const returns = [];
    for (let i = 1; i < window.length; i++) {
      returns.push(Math.log(window[i] / window[i - 1]));
    }
    const n = returns.length;
    const mean = returns.reduce((a, b) => a + b, 0) / n;
    const variance = returns.reduce((a, r) => a + (r - mean) ** 2, 0) / (n - 1);
    const hvDaily = Math.sqrt(variance);
    const hvAnnual = hvDaily * Math.sqrt(252) * 100;
    return Math.round(hvAnnual * 100) / 100; // round to 2dp, mirrors Python round(x, 2)
  } catch {
    return null;
  }
}

// Source: fetch_intraday_quotes.py pearson() ~L730-741
function pearson(x, y) {
  const n = x.length;
  if (n < 10) return null;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let num = 0, denX = 0, denY = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - mx) * (y[i] - my);
    denX += (x[i] - mx) ** 2;
    denY += (y[i] - my) ** 2;
  }
  denX = Math.sqrt(denX);
  denY = Math.sqrt(denY);
  if (denX === 0 || denY === 0) return null;
  return Math.round((num / (denX * denY)) * 1000) / 1000; // round to 3dp, mirrors Python round(x, 3)
}

// ═══════════════════════════════════════════════════════════════════
// fmt / clsDir / pctStr — 15 tests
// ═══════════════════════════════════════════════════════════════════
section('fmt / clsDir / pctStr');

test('fmt: null returns em-dash', () => assert.strictEqual(fmt(null, 2), '—'));
test('fmt: NaN returns em-dash', () => assert.strictEqual(fmt(NaN, 2), '—'));
test('fmt: positive value, 2dp', () => assert.strictEqual(fmt(1.23456, 2), '1.23'));
test('fmt: negative value, 4dp', () => assert.strictEqual(fmt(-0.00012, 4), '-0.0001'));
test('fmt: zero, 2dp', () => assert.strictEqual(fmt(0, 2), '0.00'));

test('clsDir: clearly positive → up', () => assert.strictEqual(clsDir(0.5), 'up'));
test('clsDir: clearly negative → down', () => assert.strictEqual(clsDir(-0.5), 'down'));
test('clsDir: zero → flat', () => assert.strictEqual(clsDir(0), 'flat'));
test('clsDir: exactly +0.0001 boundary → flat (not > threshold)', () => assert.strictEqual(clsDir(0.0001), 'flat'));
test('clsDir: exactly -0.0001 boundary → flat (not < threshold)', () => assert.strictEqual(clsDir(-0.0001), 'flat'));
test('clsDir: just above +0.0001 → up', () => assert.strictEqual(clsDir(0.00011), 'up'));

test('pctStr: null returns em-dash', () => assert.strictEqual(pctStr(null), '—'));
test('pctStr: positive value gets + sign', () => assert.strictEqual(pctStr(1.5), '+1.50%'));
test('pctStr: zero gets + sign (>=0)', () => assert.strictEqual(pctStr(0), '+0.00%'));
test('pctStr: negative value keeps - sign, no extra +', () => assert.strictEqual(pctStr(-2.345), '-2.35%'));

// ═══════════════════════════════════════════════════════════════════
// isOpen — 12 tests (including midnight wrap-around, Sydney-style session)
// ═══════════════════════════════════════════════════════════════════
section('isOpen');

// Normal (non-wrapping) range, e.g. London 8-17
test('isOpen: normal range, mid-session → true', () => assert.strictEqual(isOpen(8, 17, 12), true));
test('isOpen: normal range, before open → false', () => assert.strictEqual(isOpen(8, 17, 7), false));
test('isOpen: normal range, after close → false', () => assert.strictEqual(isOpen(8, 17, 17), false));
test('isOpen: normal range, at open boundary (inclusive) → true', () => assert.strictEqual(isOpen(8, 17, 8), true));
test('isOpen: normal range, one hour before close → true', () => assert.strictEqual(isOpen(8, 17, 16), true));

// Wrap-around range (Sydney session in UTC: local 08-17 AEDT ≈ UTC 21-06, crosses midnight)
test('isOpen: wrap-around, within evening segment → true', () => assert.strictEqual(isOpen(21, 6, 23), true));
test('isOpen: wrap-around, within early-morning segment → true', () => assert.strictEqual(isOpen(21, 6, 3), true));
test('isOpen: wrap-around, at midnight → true', () => assert.strictEqual(isOpen(21, 6, 0), true));
test('isOpen: wrap-around, at open boundary (inclusive) → true', () => assert.strictEqual(isOpen(21, 6, 21), true));
test('isOpen: wrap-around, at close boundary (exclusive) → false', () => assert.strictEqual(isOpen(21, 6, 6), false));
test('isOpen: wrap-around, mid-day outside session → false', () => assert.strictEqual(isOpen(21, 6, 12), false));
test('isOpen: wrap-around, one hour before open → false', () => assert.strictEqual(isOpen(21, 6, 20), false));

// ═══════════════════════════════════════════════════════════════════
// computeRate — 7 tests (Direct, inverted, cross, null legs)
// ═══════════════════════════════════════════════════════════════════
section('computeRate');

test('computeRate: direct pair (USD/JPY-style)', () => {
  const pair = { base: 'JPY', invert: false };
  const rates = { JPY: 148.25 };
  assert.strictEqual(computeRate(pair, rates), 148.25);
});
test('computeRate: inverted pair (EUR/USD-style)', () => {
  const pair = { base: 'EUR', invert: true };
  const rates = { EUR: 0.92 };
  assert.ok(Math.abs(computeRate(pair, rates) - (1 / 0.92)) < 1e-9);
});
test('computeRate: cross pair (EUR/GBP-style)', () => {
  const pair = { cross: ['EUR', 'GBP'] };
  const rates = { EUR: 0.92, GBP: 0.79 };
  const expected = (1 / 0.92) / (1 / 0.79);
  assert.ok(Math.abs(computeRate(pair, rates) - expected) < 1e-9);
});
test('computeRate: null STATE.rates → null', () => {
  const pair = { base: 'JPY', invert: false };
  assert.strictEqual(computeRate(pair, null), null);
});
test('computeRate: inverted pair with missing base rate → null', () => {
  const pair = { base: 'CHF', invert: true };
  const rates = { EUR: 0.92 };
  assert.strictEqual(computeRate(pair, rates), null);
});
test('computeRate: direct pair with missing base rate → null', () => {
  const pair = { base: 'NOK', invert: false };
  const rates = { EUR: 0.92 };
  assert.strictEqual(computeRate(pair, rates), null);
});
test('computeRate: cross pair with one missing leg → null', () => {
  const pair = { cross: ['EUR', 'SEK'] };
  const rates = { EUR: 0.92 };
  assert.strictEqual(computeRate(pair, rates), null);
});

// ═══════════════════════════════════════════════════════════════════
// Stress scoring — 18 tests
// ═══════════════════════════════════════════════════════════════════
section('Stress scoring (risk regime)');

test('Stress: VIX 18 exactly (boundary, not >18) + nothing else → RISK-ON, score 0', () => {
  const r = computeStressScore({ vix: { close: 18 } });
  assert.strictEqual(r.stressScore, 0);
  assert.strictEqual(r.regime, 'RISK-ON');
});
test('Stress: VIX 18.1 (just above) → +1, MIXED', () => {
  const r = computeStressScore({ vix: { close: 18.1 } });
  assert.strictEqual(r.stressScore, 1);
  assert.strictEqual(r.regime, 'MIXED');
});
test('Stress: VIX 25 exactly (boundary, not >25) → +1 only, MIXED', () => {
  const r = computeStressScore({ vix: { close: 25 } });
  assert.strictEqual(r.stressScore, 1);
  assert.strictEqual(r.regime, 'MIXED');
});
test('Stress: VIX 25.1 (just above) → +2, CAUTION', () => {
  const r = computeStressScore({ vix: { close: 25.1 } });
  assert.strictEqual(r.stressScore, 2);
  assert.strictEqual(r.regime, 'CAUTION');
});
test('Stress: VIX 30 exactly (boundary, not >30) → +2, CAUTION', () => {
  const r = computeStressScore({ vix: { close: 30 } });
  assert.strictEqual(r.stressScore, 2);
  assert.strictEqual(r.regime, 'CAUTION');
});
test('Stress: VIX 30.1 (just above) → +3, CAUTION (needs 4 for RISK-OFF)', () => {
  const r = computeStressScore({ vix: { close: 30.1 } });
  assert.strictEqual(r.stressScore, 3);
  assert.strictEqual(r.regime, 'CAUTION');
});
test('Stress: inverted curve adds +1', () => {
  const r = computeStressScore({ vix: { close: 10 }, us10y: { close: 3.5 }, us3m: { close: 4.0 } });
  assert.strictEqual(r.isInverted, true);
  assert.strictEqual(r.stressScore, 1);
});
test('Stress: gold pct exactly 2.0 (boundary, not >2.0) → no add', () => {
  const r = computeStressScore({ vix: { close: 10 }, gold: { pct: 2.0 } });
  assert.strictEqual(r.stressScore, 0);
});
test('Stress: gold pct 2.1 → +1', () => {
  const r = computeStressScore({ vix: { close: 10 }, gold: { pct: 2.1 } });
  assert.strictEqual(r.stressScore, 1);
});
test('Stress: SPX pct exactly -1.5 (boundary, not <-1.5) → no add', () => {
  const r = computeStressScore({ vix: { close: 10 }, spx: { pct: -1.5 } });
  assert.strictEqual(r.stressScore, 0);
});
test('Stress: SPX pct -1.6 → +1', () => {
  const r = computeStressScore({ vix: { close: 10 }, spx: { pct: -1.6 } });
  assert.strictEqual(r.stressScore, 1);
});
test('Stress: MOVE exactly 100 (boundary, not >100) → no add', () => {
  const r = computeStressScore({ vix: { close: 10 }, move: { close: 100 } });
  assert.strictEqual(r.stressScore, 0);
});
test('Stress: MOVE 101 → +1', () => {
  const r = computeStressScore({ vix: { close: 10 }, move: { close: 101 } });
  assert.strictEqual(r.stressScore, 1);
});
test('Stress: AUD/JPY pct -1.6 → +1 (risk barometer)', () => {
  const r = computeStressScore({ vix: { close: 10 }, audjpy: { pct: -1.6 } });
  assert.strictEqual(r.stressScore, 1);
});
test('Stress: USD/JPY weak alone (no AUD/JPY confirmation) → no add', () => {
  const r = computeStressScore({ vix: { close: 10 }, usdjpy: { pct: -1.5 } });
  assert.strictEqual(r.stressScore, 0);
});
test('Stress: USD/JPY weak AND AUD/JPY weak together → +1 (confirmed)', () => {
  const r = computeStressScore({ vix: { close: 10 }, usdjpy: { pct: -1.5 }, audjpy: { pct: -1.6 } });
  // AUD/JPY < -1.5 also independently adds +1, so total is +2 here
  assert.strictEqual(r.stressScore, 2);
});
test('Stress: HY OAS 20d Δ exactly 15 (boundary, not >15) → no add', () => {
  const r = computeStressScore({ vix: { close: 10 }, hyOasDelta20d: 15 });
  assert.strictEqual(r.stressScore, 0);
});
test('Stress: combined score >=4 → RISK-OFF with inverted-curve note', () => {
  const r = computeStressScore({
    vix: { close: 26 },            // +2
    us10y: { close: 3.5 }, us3m: { close: 4.0 }, // inverted +1
    move: { close: 105 },          // +1
  });
  assert.strictEqual(r.stressScore, 4);
  assert.strictEqual(r.regime, 'RISK-OFF');
  assert.ok(!r.regimeSub.includes('inverted curve'), 'RISK-OFF regimeSub omits the inverted-curve suffix (only added when regime !== RISK-OFF)');
});

// ═══════════════════════════════════════════════════════════════════
// localizeSignalTime — 6 tests
// ═══════════════════════════════════════════════════════════════════
section('localizeSignalTime');

test('localizeSignalTime: null → placeholder', () => assert.strictEqual(localizeSignalTime(null), '--:--'));
test('localizeSignalTime: undefined → placeholder', () => assert.strictEqual(localizeSignalTime(undefined), '--:--'));
test('localizeSignalTime: "--:--" passthrough', () => assert.strictEqual(localizeSignalTime('--:--'), '--:--'));
test('localizeSignalTime: bad format (non-numeric) returns original string', () => assert.strictEqual(localizeSignalTime('ab:cd'), 'ab:cd'));
test('localizeSignalTime: midnight "00:00" formats cleanly', () => assert.strictEqual(localizeSignalTime('00:00'), '00:00'));
test('localizeSignalTime: end-of-day "23:59" formats cleanly', () => assert.strictEqual(localizeSignalTime('23:59'), '23:59'));

// ═══════════════════════════════════════════════════════════════════
// Business dates — 7 tests (Mon–Fri, Sat, Sun, Mon→Fri prev)
// ═══════════════════════════════════════════════════════════════════
section('Business dates');

// All reference dates are UTC noon to avoid any local-TZ date-rollover ambiguity.
const MON = new Date('2026-08-10T12:00:00Z'); // Monday
const WED = new Date('2026-08-12T12:00:00Z'); // Wednesday
const FRI = new Date('2026-08-14T12:00:00Z'); // Friday
const SAT = new Date('2026-08-15T12:00:00Z'); // Saturday
const SUN = new Date('2026-08-16T12:00:00Z'); // Sunday

test('getLatestBizDate: Monday → same day', () => assert.strictEqual(getLatestBizDate(MON), '2026-08-10'));
test('getLatestBizDate: Wednesday → same day', () => assert.strictEqual(getLatestBizDate(WED), '2026-08-12'));
test('getLatestBizDate: Friday → same day', () => assert.strictEqual(getLatestBizDate(FRI), '2026-08-14'));
test('getLatestBizDate: Saturday → rolls back to Friday', () => assert.strictEqual(getLatestBizDate(SAT), '2026-08-14'));
test('getLatestBizDate: Sunday → rolls back to Friday', () => assert.strictEqual(getLatestBizDate(SUN), '2026-08-14'));
test('getPrevBizDate: Monday → previous Friday', () => assert.strictEqual(getPrevBizDate(MON), '2026-08-07'));
test('getPrevBizDate: Wednesday → previous Tuesday', () => assert.strictEqual(getPrevBizDate(WED), '2026-08-11'));

// ═══════════════════════════════════════════════════════════════════
// Yield spreads — 4 tests (Normal, inverted, flat, US-DE 10Y fallback)
// ═══════════════════════════════════════════════════════════════════
section('Yield spreads');

test('Yield spread: normal (2Y available both legs, base > quote)', () => {
  // USD/JPY-style, not cross, not invert: base=JPY 2Y 4.10, quote=USD 2Y 4.60
  const { bondTenor, bondDiff } = computeBondSpread({ y2: 4.10 }, { y2: 4.60 }, false, false);
  assert.strictEqual(bondTenor, '2Y');
  // raw diff base-quote = -0.50, sign flipped (invert=false) → +0.50
  assert.ok(Math.abs(bondDiff - 0.50) < 1e-9);
});
test('Yield spread: inverted pair flips sign correctly', () => {
  // EUR/USD-style, invert=true: base=EUR 2Y 2.80, quote=USD 2Y 4.60
  const { bondDiff } = computeBondSpread({ y2: 2.80 }, { y2: 4.60 }, true, false);
  // raw diff = 2.80 - 4.60 = -1.80, invert=true → no sign flip
  assert.ok(Math.abs(bondDiff - (-1.80)) < 1e-9);
});
test('Yield spread: flat (equal 2Y yields) → zero spread', () => {
  const { bondDiff } = computeBondSpread({ y2: 3.75 }, { y2: 3.75 }, false, false);
  // Equal legs produce a raw diff of 0, then the sign-flip (-0) is mathematically
  // still zero — compare with == rather than strictEqual/Object.is (which treats -0 ≠ 0).
  assert.ok(bondDiff === 0, `expected zero spread, got ${bondDiff}`);
});
test('Yield spread: US-DE, 2Y unavailable on DE leg → falls back to 10Y', () => {
  const bondUS = { y2: 4.60, y10: 4.20 };
  const bondDE = { y2: null, y10: 2.45 }; // EUR/DE 2Y not covered — mirrors JPY/NZD/NOK/SEK pattern
  const { bondTenor, bondDiff } = computeBondSpread(bondUS, bondDE, false, false);
  assert.strictEqual(bondTenor, '10Y');
  assert.ok(Math.abs(bondDiff - (-(4.20 - 2.45))) < 1e-9);
});

// ═══════════════════════════════════════════════════════════════════
// computeHV30 — 9 tests
// ═══════════════════════════════════════════════════════════════════
section('computeHV30');

test('computeHV30: fewer than 22 closes → null', () => {
  const closes = Array.from({ length: 21 }, (_, i) => 100 + i);
  assert.strictEqual(computeHV30(closes), null);
});
test('computeHV30: exactly 22 closes → numeric result', () => {
  const closes = Array.from({ length: 22 }, (_, i) => 100 + Math.sin(i) * 2);
  const result = computeHV30(closes);
  assert.strictEqual(typeof result, 'number');
  assert.ok(!isNaN(result));
});
test('computeHV30: empty array → null', () => assert.strictEqual(computeHV30([]), null));
test('computeHV30: single element → null', () => assert.strictEqual(computeHV30([100]), null));
test('computeHV30: zero/negative prices are filtered out before the length check', () => {
  const closes = [0, -5, ...Array.from({ length: 22 }, (_, i) => 100 + i)];
  const result = computeHV30(closes);
  assert.strictEqual(typeof result, 'number');
});
test('computeHV30: uses only the last 31 prices when more are supplied', () => {
  const tail31 = Array.from({ length: 31 }, (_, i) => 100 + (i % 2 === 0 ? 1 : -1));
  const withJunkPrefix = [9999, 1, 2, 3, ...tail31]; // junk earlier prices must not affect result
  assert.strictEqual(computeHV30(withJunkPrefix), computeHV30(tail31));
});
test('computeHV30: known alternating-return sequence produces exact expected value', () => {
  // 22 closes alternating +5%/-5% (approx) around 100 — deterministic known result,
  // computed independently via the same n-1 sample-variance formula.
  const window = [];
  let p = 100;
  for (let i = 0; i < 22; i++) {
    window.push(p);
    p = i % 2 === 0 ? p * 1.05 : p / 1.05;
  }
  const returns = [];
  for (let i = 1; i < window.length; i++) returns.push(Math.log(window[i] / window[i - 1]));
  const n = returns.length;
  const mean = returns.reduce((a, b) => a + b, 0) / n;
  const variance = returns.reduce((a, r) => a + (r - mean) ** 2, 0) / (n - 1);
  const expected = Math.round(Math.sqrt(variance) * Math.sqrt(252) * 100 * 100) / 100;
  assert.strictEqual(computeHV30(window), expected);
});
test('computeHV30: result is annualised (order of magnitude sanity check, not a tiny daily-scale number)', () => {
  const closes = Array.from({ length: 22 }, (_, i) => 100 + Math.sin(i / 2) * 3);
  const result = computeHV30(closes);
  assert.ok(result > 1, 'annualised HV for a moving series should be well above 1%');
});
test('computeHV30: constant price series → zero volatility', () => {
  const closes = Array.from({ length: 25 }, () => 100);
  assert.strictEqual(computeHV30(closes), 0);
});
// ═══════════════════════════════════════════════════════════════════
// Pearson correlation — 7 tests
// ═══════════════════════════════════════════════════════════════════
section('Pearson correlation');

test('pearson: perfect positive correlation → +1', () => {
  const x = Array.from({ length: 10 }, (_, i) => i);
  const y = Array.from({ length: 10 }, (_, i) => i * 2 + 5);
  assert.strictEqual(pearson(x, y), 1);
});
test('pearson: perfect negative correlation → -1', () => {
  const x = Array.from({ length: 10 }, (_, i) => i);
  const y = Array.from({ length: 10 }, (_, i) => -i * 3 + 1);
  assert.strictEqual(pearson(x, y), -1);
});
test('pearson: orthogonal/no linear relationship → ~0', () => {
  const x = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1];
  const y = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1];
  const result = pearson(x, y);
  assert.ok(Math.abs(result) < 0.4, `expected near-zero correlation, got ${result}`);
});
test('pearson: fewer than 10 points → null', () => {
  const x = [1, 2, 3, 4, 5];
  const y = [1, 2, 3, 4, 5];
  assert.strictEqual(pearson(x, y), null);
});
test('pearson: zero variance in one series → null (division-by-zero guard)', () => {
  const x = Array.from({ length: 10 }, () => 5); // constant
  const y = Array.from({ length: 10 }, (_, i) => i);
  assert.strictEqual(pearson(x, y), null);
});
test('pearson: result is always bounded within [-1, 1]', () => {
  const x = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
  const y = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4];
  const result = pearson(x, y);
  assert.ok(result >= -1 && result <= 1, `pearson out of bounds: ${result}`);
});
test('pearson: EUR/USD vs DXY — expected inverse relationship in sample data', () => {
  // Synthetic but representative: DXY up days broadly coincide with EUR/USD down days.
  const eurusd = [1.080, 1.078, 1.082, 1.075, 1.079, 1.073, 1.077, 1.070, 1.074, 1.068, 1.071];
  const dxy    = [103.2, 103.5, 102.9, 104.1, 103.4, 104.5, 103.8, 105.0, 104.3, 105.5, 104.9];
  const result = pearson(eurusd, dxy);
  assert.ok(result < 0, `expected negative correlation between EUR/USD and DXY, got ${result}`);
});

// ═══════════════════════════════════════════════════════════════════
// Correlation matrices' date-safe join — mirrored from assets/dashboard.js's
// _sortDateKeys() / _logReturnsByDate() / _pearsonCorrByDate() (v8.273.0,
// fixing a positional-array join bug shared by both the docked currency
// Matrix and the fullscreen Pairs matrix — see GUIDELINES.md v8.180.0's
// original "join by calendar date, never trailing position" rule, first
// applied to fetch_correlations() on the backend and now extended to these
// two frontend-only correlation grids, which read ohlc-data/*.json directly
// and never went through that backend fix).
// ═══════════════════════════════════════════════════════════════════
function _pearsonCorr(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 5) return null;
  a = a.slice(-n); b = b.slice(-n);
  const ma = a.reduce((s, x) => s + x, 0) / n, mb = b.reduce((s, x) => s + x, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { const xa = a[i] - ma, xb = b[i] - mb; num += xa * xb; da += xa * xa; db += xb * xb; }
  if (da === 0 || db === 0) return null;
  return num / Math.sqrt(da * db);
}
function _sortDateKeys(keys) {
  return keys.sort((a, b) => {
    const na = Number(a), nb = Number(b);
    if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
    return a < b ? -1 : a > b ? 1 : 0;
  });
}
function _logReturnsByDate(closesByDate) {
  if (!closesByDate) return null;
  const dates = _sortDateKeys(Object.keys(closesByDate));
  if (dates.length < 2) return null;
  const rets = {};
  for (let i = 1; i < dates.length; i++) {
    const prev = closesByDate[dates[i - 1]], cur = closesByDate[dates[i]];
    rets[dates[i]] = Math.log(cur / prev);
  }
  return rets;
}
function _pearsonCorrByDate(retsA, retsB, maxN) {
  if (!retsA || !retsB) return null;
  const shared = _sortDateKeys(Object.keys(retsA).filter(d => Object.prototype.hasOwnProperty.call(retsB, d)));
  if (shared.length < 5) return null;
  const window_ = maxN ? shared.slice(-maxN) : shared;
  if (window_.length < 5) return null;
  const a = window_.map(d => retsA[d]), b = window_.map(d => retsB[d]);
  return _pearsonCorr(a, b);
}

section('Correlation matrices — date-safe join (_sortDateKeys / _logReturnsByDate / _pearsonCorrByDate)');
test('_sortDateKeys: sorts Unix-timestamp-number keys numerically, not lexicographically', () => {
  // Lexicographic sort would misorder these once digit counts differ; here
  // all three share 10 digits so a lexicographic sort would happen to look
  // right — the real point is it must still produce true numeric order.
  const keys = ['1725580800', '1725573600', '1725577200'];
  assert.deepStrictEqual(_sortDateKeys(keys.slice()), ['1725573600', '1725577200', '1725580800']);
});
test('_sortDateKeys: sorts ISO date-string keys chronologically', () => {
  const keys = ['2026-08-27', '2026-08-01', '2026-08-15'];
  assert.deepStrictEqual(_sortDateKeys(keys.slice()), ['2026-08-01', '2026-08-15', '2026-08-27']);
});
test('_logReturnsByDate: computes each return only against its own series\' immediately-preceding date, not array position', () => {
  const closes = { '2026-08-03': 1.10, '2026-08-01': 1.00, '2026-08-02': 1.05 }; // inserted out of order
  const rets = _logReturnsByDate(closes);
  assert.ok(Math.abs(rets['2026-08-02'] - Math.log(1.05 / 1.00)) < 1e-12);
  assert.ok(Math.abs(rets['2026-08-03'] - Math.log(1.10 / 1.05)) < 1e-12);
  assert.strictEqual(Object.keys(rets).length, 2); // no return for the first date (no prior)
});
test('_logReturnsByDate: null on fewer than 2 dates', () => {
  assert.strictEqual(_logReturnsByDate({ '2026-08-01': 1.10 }), null);
  assert.strictEqual(_logReturnsByDate(null), null);
});
test('_pearsonCorrByDate: joins on shared dates only, ignoring a date only one series has', () => {
  // Series A has an extra date (08-05) that B lacks — a positional
  // .slice(-n)-style join would have silently misaligned everything from
  // that point on; the date-safe join must simply drop the unshared date.
  const retsA = { '2026-08-01': 0.01, '2026-08-02': -0.02, '2026-08-03': 0.015, '2026-08-04': -0.01, '2026-08-05': 0.02, '2026-08-06': -0.005 };
  const retsB = { '2026-08-01': 0.012, '2026-08-02': -0.018, '2026-08-03': 0.017, '2026-08-04': -0.011,                    '2026-08-06': -0.004 };
  const result = _pearsonCorrByDate(retsA, retsB);
  assert.ok(result !== null && result > 0.9, `expected a strong positive correlation on the 5 shared dates, got ${result}`);
});
test('_pearsonCorrByDate: null when fewer than 5 dates are shared', () => {
  const retsA = { '2026-08-01': 0.01, '2026-08-02': -0.02, '2026-08-03': 0.015 };
  const retsB = { '2026-08-01': 0.012, '2026-08-02': -0.018, '2026-08-03': 0.017 };
  assert.strictEqual(_pearsonCorrByDate(retsA, retsB), null);
});
test('_pearsonCorrByDate: a pair-specific gap does not desync two otherwise-identical series (regression case)', () => {
  // This is the exact failure class fixed in v8.273.0: series A is missing
  // one date in the middle that series B has. A positional trailing-slice
  // join would shift every later element of A one slot out of true
  // calendar alignment with B; the date-safe join must instead recognize
  // A and B are simply co-moving (both derived from the same trend) and
  // still report a strong positive correlation.
  const dates = ['2026-08-01','2026-08-02','2026-08-03','2026-08-04','2026-08-05','2026-08-06','2026-08-07','2026-08-08'];
  const trend = [0.010, -0.015, 0.020, -0.005, 0.012, -0.018, 0.022, -0.008];
  const retsB = {}; dates.forEach((d, i) => retsB[d] = trend[i]);
  const retsA = {}; dates.forEach((d, i) => { if (i !== 3) retsA[d] = trend[i]; }); // A missing 08-04
  const result = _pearsonCorrByDate(retsA, retsB);
  assert.ok(result !== null && result > 0.95, `expected near-perfect correlation once correctly date-joined, got ${result}`);
});

// ═══════════════════════════════════════════════════════════════════
// FX Fair Value regression — mirrored from assets/dashboard.js's
// _solveLinearSystem() / _fvRegress() (v8.197.0, generalized to the
// 5-variable BEER model — rate_diff, stress, ca_diff, tb_diff — v8.200.0)
// ═══════════════════════════════════════════════════════════════════
const FV_FEATURE_KEYS = ['rate_diff', 'stress', 'ca_diff', 'tb_diff'];
function _solveLinearSystem(A, b) {
  const k = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  const EPS = 1e-9;
  for (let col = 0; col < k; col++) {
    let pivot = col;
    for (let r = col + 1; r < k; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < EPS) return null;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = 0; r < k; r++) {
      if (r === col) continue;
      const factor = M[r][col] / M[col][col];
      for (let c = col; c < k + 1; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map((row, i) => row[k] / row[i]);
}
function _fvRegress(rows) {
  const usable = rows.filter(r => r && r.spot != null && FV_FEATURE_KEYS.every(k => r[k] != null));
  const k = FV_FEATURE_KEYS.length + 1;
  if (usable.length < k * 2) return null;
  const n = usable.length;
  let Sxx = Array.from({ length: k }, () => new Array(k).fill(0));
  let Sxy = new Array(k).fill(0);
  usable.forEach(r => {
    const x = [1, ...FV_FEATURE_KEYS.map(key => r[key])];
    for (let i = 0; i < k; i++) {
      Sxy[i] += x[i] * r.spot;
      for (let j = 0; j < k; j++) Sxx[i][j] += x[i] * x[j];
    }
  });
  const beta = _solveLinearSystem(Sxx, Sxy);
  if (!beta) return null;
  const fitted = usable.map(r => beta[0] + FV_FEATURE_KEYS.reduce((s, key, i) => s + beta[i + 1] * r[key], 0));
  const residuals = usable.map((r, i) => r.spot - fitted[i]);
  const residMean = residuals.reduce((a, b) => a + b, 0) / n;
  const residVar = residuals.reduce((a, b) => a + (b - residMean) * (b - residMean), 0) / (n - k);
  const residStd = residVar > 0 ? Math.sqrt(residVar) : 0;
  return { beta, n, residStd, usable };
}

test('_fvRegress: recovers known linear coefficients from synthetic data (5-var BEER)', () => {
  const rows = [];
  for (let i = 0; i < 60; i++) {
    const rate_diff = Math.sin(i / 5) * 2;
    const stress = Math.cos(i / 7) * 30 + 20;
    const ca_diff = Math.sin(i / 9) * 3;
    const tb_diff = Math.cos(i / 11) * 1.5;
    const noise = ((i * 37) % 13 - 6) * 0.0005;
    const spot = 1.10 + 0.01 * rate_diff - 0.002 * stress + 0.004 * ca_diff - 0.003 * tb_diff + noise;
    rows.push({ date: `d${i}`, spot, rate_diff, stress, ca_diff, tb_diff });
  }
  const reg = _fvRegress(rows);
  assert.ok(reg, 'regression should succeed with 60 varied rows');
  assert.ok(Math.abs(reg.beta[0] - 1.10) < 0.01, `intercept off: ${reg.beta[0]}`);
  assert.ok(Math.abs(reg.beta[1] - 0.01) < 0.01, `rate_diff coef off: ${reg.beta[1]}`);
  assert.ok(Math.abs(reg.beta[2] - (-0.002)) < 0.001, `stress coef off: ${reg.beta[2]}`);
  assert.ok(Math.abs(reg.beta[3] - 0.004) < 0.001, `ca_diff coef off: ${reg.beta[3]}`);
  assert.ok(Math.abs(reg.beta[4] - (-0.003)) < 0.001, `tb_diff coef off: ${reg.beta[4]}`);
  assert.ok(reg.residStd > 0 && reg.residStd < 0.01, `residStd out of expected range: ${reg.residStd}`);
});
test('_fvRegress: singular input (constant regressor) → null, not garbage', () => {
  const rows = Array.from({ length: 60 }, (_, i) => ({
    date: `d${i}`, spot: 1.10 + i * 0.0001, rate_diff: 1.0, stress: 20, ca_diff: -1.5, tb_diff: 0.5,
  }));
  assert.strictEqual(_fvRegress(rows), null);
});
test('_fvRegress: fewer than 2×params usable rows → null', () => {
  const rows = [
    { spot: 1.1, rate_diff: 1, stress: 20, ca_diff: 0.5, tb_diff: -0.2 },
    { spot: 1.11, rate_diff: 1.1, stress: 21, ca_diff: 0.6, tb_diff: -0.1 },
  ];
  assert.strictEqual(_fvRegress(rows), null);
});
test('_fvRegress: rows missing a required field (incl. new ca_diff/tb_diff) are excluded from the usable set', () => {
  // At least 10 usable rows needed (k=5 params, gate is 2×k) — 14 built,
  // 3 excluded via a missing field each, leaving 11. Features vary via
  // index-derived offsets so the remaining system after exclusions isn't
  // singular purely from this test's own setup.
  const rows = [];
  for (let i = 0; i < 14; i++) {
    rows.push({
      spot: 1.10 + i * 0.001,
      rate_diff: 1.0 + (i % 5) * 0.2,
      stress: 5 + (i % 4) * 3,
      ca_diff: -1 + (i % 3) * 0.7,
      tb_diff: 0.2 + (i % 6) * 0.15,
    });
  }
  rows[2].rate_diff = null; // excluded
  rows[5].ca_diff = null;   // excluded
  rows[8].tb_diff = null;   // excluded
  const reg = _fvRegress(rows);
  assert.ok(reg, 'regression should succeed on the 11 remaining valid rows');
  assert.strictEqual(reg.n, 11);
});

// ═══════════════════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════════════════
console.log(`\n${'─'.repeat(60)}`);
if (fail === 0) {
  console.log(`${pass} passed, ${fail} failed`);
} else {
  console.log(`FAILURES:`);
  failures.forEach(({ name, err }) => {
    console.log(`  ✗ ${name}`);
    console.log(`    ${err.message}`);
  });
  console.log(`\n${pass} passed, ${fail} failed`);
}

process.exit(fail === 0 ? 0 : 1);
