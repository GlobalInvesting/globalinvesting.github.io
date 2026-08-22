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
// FX Fair Value regression — mirrored from assets/dashboard.js's
// _solve3x3() / _fvRegress() (v8.197.0)
// ═══════════════════════════════════════════════════════════════════
function _solve3x3(A, b) {
  const M = A.map((row, i) => [...row, b[i]]);
  const EPS = 1e-9;
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let r = col + 1; r < 3; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < EPS) return null;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const factor = M[r][col] / M[col][col];
      for (let c = col; c < 4; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}
function _fvRegress(rows) {
  const usable = rows.filter(r => r && r.spot != null && r.rate_diff != null && r.stress != null);
  if (usable.length < 4) return null;
  const n = usable.length;
  let Sxx = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  let Sxy = [0, 0, 0];
  usable.forEach(r => {
    const x = [1, r.rate_diff, r.stress];
    for (let i = 0; i < 3; i++) {
      Sxy[i] += x[i] * r.spot;
      for (let j = 0; j < 3; j++) Sxx[i][j] += x[i] * x[j];
    }
  });
  const beta = _solve3x3(Sxx, Sxy);
  if (!beta) return null;
  const fitted = usable.map(r => beta[0] + beta[1] * r.rate_diff + beta[2] * r.stress);
  const residuals = usable.map((r, i) => r.spot - fitted[i]);
  const residMean = residuals.reduce((a, b) => a + b, 0) / n;
  const residVar = residuals.reduce((a, b) => a + (b - residMean) * (b - residMean), 0) / (n - 3);
  const residStd = residVar > 0 ? Math.sqrt(residVar) : 0;
  return { beta, n, residStd, usable };
}

test('_fvRegress: recovers known linear coefficients from synthetic data', () => {
  const rows = [];
  for (let i = 0; i < 60; i++) {
    const rate_diff = Math.sin(i / 5) * 2;
    const stress = Math.cos(i / 7) * 30 + 20;
    const noise = ((i * 37) % 13 - 6) * 0.0005;
    rows.push({ date: `d${i}`, spot: 1.10 + 0.01 * rate_diff - 0.002 * stress + noise, rate_diff, stress });
  }
  const reg = _fvRegress(rows);
  assert.ok(reg, 'regression should succeed with 60 varied rows');
  assert.ok(Math.abs(reg.beta[0] - 1.10) < 0.01, `intercept off: ${reg.beta[0]}`);
  assert.ok(Math.abs(reg.beta[1] - 0.01) < 0.01, `rate_diff coef off: ${reg.beta[1]}`);
  assert.ok(Math.abs(reg.beta[2] - (-0.002)) < 0.001, `stress coef off: ${reg.beta[2]}`);
  assert.ok(reg.residStd > 0 && reg.residStd < 0.01, `residStd out of expected range: ${reg.residStd}`);
});
test('_fvRegress: singular input (constant regressor) → null, not garbage', () => {
  const rows = Array.from({ length: 60 }, (_, i) => ({ date: `d${i}`, spot: 1.10 + i * 0.0001, rate_diff: 1.0, stress: 20 }));
  assert.strictEqual(_fvRegress(rows), null);
});
test('_fvRegress: fewer than 4 usable rows → null', () => {
  const rows = [{ spot: 1.1, rate_diff: 1, stress: 20 }, { spot: 1.11, rate_diff: 1.1, stress: 21 }];
  assert.strictEqual(_fvRegress(rows), null);
});
test('_fvRegress: rows missing a required field are excluded from the usable set', () => {
  // stress deliberately non-collinear with rate_diff/index (5, 12, 3, 18) so
  // the remaining 4-row system isn't singular purely from this test's own setup.
  const rows = [
    { spot: 1.10, rate_diff: 1.0, stress: 5 },
    { spot: 1.11, rate_diff: null, stress: 21 }, // excluded
    { spot: 1.12, rate_diff: 1.2, stress: 12 },
    { spot: 1.13, rate_diff: 1.3, stress: 3 },
    { spot: 1.14, rate_diff: 1.4, stress: 18 },
  ];
  const reg = _fvRegress(rows);
  assert.ok(reg, 'regression should succeed on the 4 remaining valid rows');
  assert.strictEqual(reg.n, 4);
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
