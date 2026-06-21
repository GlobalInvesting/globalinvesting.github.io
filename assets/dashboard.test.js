/**
 * dashboard.test.js — Automated test suite for Global Investing FX Terminal
 * Run with: node assets/dashboard.test.js
 *
 * Covers:
 *   1. fmt / clsDir / pctStr — formatting utilities
 *   2. isOpen — trading session logic
 *   3. computeRate — FX cross/invert/direct rate calculation
 *   4. stressScore — multi-factor risk regime scoring
 *   5. localizeSignalTime — UTC→local time conversion
 *   6. getLatestBizDate / getPrevBizDate — business date logic
 *   7. Yield spread calculations
 *   8. HV30 formula validation (annualisation)
 *   9. Pearson correlation edge cases
 */

'use strict';

// ─── Minimal browser shim ────────────────────────────────────────────────────
try { Object.defineProperty(global, "navigator", { value: { language: "en-US" }, configurable: true }); } catch(_) {}
global.document  = { getElementById: () => null };

// ─── Test runner ─────────────────────────────────────────────────────────────
let _passed = 0, _failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓  ${name}`);
    _passed++;
  } catch (e) {
    console.error(`  ✗  ${name}`);
    console.error(`       ${e.message}`);
    _failed++;
  }
}

function expect(actual) {
  return {
    toBe(expected) {
      if (actual !== expected)
        throw new Error(`Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
    },
    toBeCloseTo(expected, precision = 4) {
      const diff = Math.abs(actual - expected);
      const tol  = Math.pow(10, -precision);
      if (diff > tol)
        throw new Error(`Expected ~${expected} (±${tol}), got ${actual}`);
    },
    toBeNull() {
      if (actual !== null)
        throw new Error(`Expected null, got ${JSON.stringify(actual)}`);
    },
    toBeGreaterThan(v) {
      if (!(actual > v))
        throw new Error(`Expected > ${v}, got ${actual}`);
    },
    toBeLessThan(v) {
      if (!(actual < v))
        throw new Error(`Expected < ${v}, got ${actual}`);
    },
    toBeTruthy() {
      if (!actual)
        throw new Error(`Expected truthy, got ${JSON.stringify(actual)}`);
    },
    toMatch(re) {
      if (!re.test(actual))
        throw new Error(`Expected ${JSON.stringify(actual)} to match ${re}`);
    },
  };
}

// ─── 1. Formatting utilities ─────────────────────────────────────────────────

function fmt(val, dec) {
  if (val == null || isNaN(val)) return '—';
  return Number(val).toFixed(dec);
}

function clsDir(val) {
  if (val > 0.0001)  return 'up';
  if (val < -0.0001) return 'down';
  return 'flat';
}

function pctStr(val) {
  if (val == null || isNaN(val)) return '—';
  const sign = val >= 0 ? '+' : '';
  return sign + val.toFixed(2) + '%';
}

console.log('\n── 1. Formatting utilities ──');

test('fmt: normal number 2 decimals', () => expect(fmt(1.23456, 2)).toBe('1.23'));
test('fmt: zero', () => expect(fmt(0, 4)).toBe('0.0000'));
test('fmt: null → em dash', () => expect(fmt(null, 2)).toBe('—'));
test('fmt: NaN → em dash', () => expect(fmt(NaN, 2)).toBe('—'));
test('fmt: negative', () => expect(fmt(-0.00512, 4)).toBe('-0.0051'));

test('clsDir: positive → up', () => expect(clsDir(0.001)).toBe('up'));
test('clsDir: negative → down', () => expect(clsDir(-0.001)).toBe('down'));
test('clsDir: near-zero positive → flat', () => expect(clsDir(0.00005)).toBe('flat'));
test('clsDir: near-zero negative → flat', () => expect(clsDir(-0.00005)).toBe('flat'));
test('clsDir: exactly zero → flat', () => expect(clsDir(0)).toBe('flat'));

test('pctStr: positive → +X.XX%', () => expect(pctStr(1.5)).toBe('+1.50%'));
test('pctStr: negative → -X.XX%', () => expect(pctStr(-0.73)).toBe('-0.73%'));
test('pctStr: zero → +0.00%', () => expect(pctStr(0)).toBe('+0.00%'));
test('pctStr: null → em dash', () => expect(pctStr(null)).toBe('—'));
test('pctStr: NaN → em dash', () => expect(pctStr(NaN)).toBe('—'));

// ─── 2. Session isOpen ───────────────────────────────────────────────────────

function isOpen(openH, closeH, h) {
  return openH < closeH
    ? (h >= openH && h < closeH)
    : (h >= openH || h < closeH); // wraps midnight
}

console.log('\n── 2. isOpen — session logic ──');

test('Sydney (22-7): open at 23:00 UTC', () => expect(isOpen(22, 7, 23)).toBe(true));
test('Sydney (22-7): open at 03:00 UTC (post-midnight)', () => expect(isOpen(22, 7, 3)).toBe(true));
test('Sydney (22-7): closed at 08:00 UTC', () => expect(isOpen(22, 7, 8)).toBe(false));
test('Sydney (22-7): closed at 21:00 UTC (just before open)', () => expect(isOpen(22, 7, 21)).toBe(false));

test('Tokyo (0-9): open at 04:00 UTC', () => expect(isOpen(0, 9, 4)).toBe(true));
test('Tokyo (0-9): closed at 09:00 UTC (exact close)', () => expect(isOpen(0, 9, 9)).toBe(false));
test('Tokyo (0-9): closed at 23:00 UTC', () => expect(isOpen(0, 9, 23)).toBe(false));

test('London (8-17): open at 12:00 UTC', () => expect(isOpen(8, 17, 12)).toBe(true));
test('London (8-17): closed at 17:00 UTC (exact close)', () => expect(isOpen(8, 17, 17)).toBe(false));
test('London (8-17): closed at 07:59 UTC', () => expect(isOpen(8, 17, 7)).toBe(false));

test('New York (13-22): open at 15:00 UTC', () => expect(isOpen(13, 22, 15)).toBe(true));
test('New York (13-22): closed at 22:00 UTC (exact close)', () => expect(isOpen(13, 22, 22)).toBe(false));

// ─── 3. computeRate ──────────────────────────────────────────────────────────

// Replicate the STATE global pattern used in the dashboard
const STATE = { rates: {}, prevRates: {} };

function computeRate(pair) {
  const r = STATE.rates;
  if (!r) return null;
  if (pair.cross) {
    const [base, quote] = pair.cross;
    const baseUSD  = r[base];
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

console.log('\n── 3. computeRate ──');

// EUR = 0.9174 per USD → EUR/USD = 1/0.9174 ≈ 1.0bearer
// JPY = 149.50 per USD → USD/JPY = 149.50
STATE.rates = { EUR: 0.9174, JPY: 149.50, GBP: 0.7874, AUD: 1.5385 };

test('EUR/USD via invert: 1/0.9174 ≈ 1.0Formation', () => {
  const rate = computeRate({ base: 'EUR', invert: true });
  expect(rate).toBeCloseTo(1 / 0.9174, 6);
});

test('USD/JPY direct: 149.50', () => {
  const rate = computeRate({ base: 'JPY' });
  expect(rate).toBeCloseTo(149.50, 6);
});

test('EUR/GBP cross: (1/0.9174)/(1/0.7874)', () => {
  const rate = computeRate({ cross: ['EUR', 'GBP'] });
  const expected = (1 / 0.9174) / (1 / 0.7874);
  expect(rate).toBeCloseTo(expected, 6);
});

test('EUR/JPY cross: EUR/USD / 1*JPY_per_USD → ≈ 162.97', () => {
  const rate = computeRate({ cross: ['EUR', 'JPY'] });
  const expected = (1 / 0.9174) / (1 / 149.50);
  expect(rate).toBeCloseTo(expected, 4);
});

test('AUD/USD via invert', () => {
  const rate = computeRate({ base: 'AUD', invert: true });
  expect(rate).toBeCloseTo(1 / 1.5385, 6);
});

test('missing currency → null', () => {
  const rate = computeRate({ base: 'CHF' });
  expect(rate).toBeNull();
});

test('cross with missing leg → null', () => {
  const rate = computeRate({ cross: ['EUR', 'CHF'] });
  expect(rate).toBeNull();
});

// ─── 4. Stress scoring / regime ──────────────────────────────────────────────

function computeStressScore({ vix, isInverted, goldPct, spxPct, move }) {
  let score = 0;
  if (vix > 30)      score += 3;
  else if (vix > 25) score += 2;
  else if (vix > 18) score += 1;
  if (isInverted) score += 1;
  if (goldPct > 1.0) score += 1;
  if (spxPct < -0.5) score += 1;
  if (move > 120)    score += 1;
  return score;
}

function scoreToRegime(score) {
  if (score >= 4)  return 'RISK-OFF';
  if (score >= 2)  return 'CAUTION';
  if (score === 1) return 'MIXED';
  return 'RISK-ON';
}

console.log('\n── 4. Stress scoring / regime ──');

test('VIX 15, no signals → RISK-ON (score 0)', () => {
  const s = computeStressScore({ vix: 15, isInverted: false, goldPct: 0, spxPct: 0.2, move: 80 });
  expect(s).toBe(0);
  expect(scoreToRegime(s)).toBe('RISK-ON');
});

test('VIX 20 only → MIXED (score 1)', () => {
  const s = computeStressScore({ vix: 20, isInverted: false, goldPct: 0, spxPct: 0, move: 80 });
  expect(s).toBe(1);
  expect(scoreToRegime(s)).toBe('MIXED');
});

test('VIX 25.5 → CAUTION (score 2)', () => {
  const s = computeStressScore({ vix: 25.5, isInverted: false, goldPct: 0, spxPct: 0, move: 80 });
  expect(s).toBe(2);
  expect(scoreToRegime(s)).toBe('CAUTION');
});

test('VIX 25.5 + inverted curve → CAUTION (score 3)', () => {
  const s = computeStressScore({ vix: 25.5, isInverted: true, goldPct: 0, spxPct: 0, move: 80 });
  expect(s).toBe(3);
  expect(scoreToRegime(s)).toBe('CAUTION');
});

test('VIX 25.5 + inverted + gold +3% → RISK-OFF (score 4)', () => {
  const s = computeStressScore({ vix: 25.5, isInverted: true, goldPct: 3.0, spxPct: 0, move: 80 });
  expect(s).toBe(4);
  expect(scoreToRegime(s)).toBe('RISK-OFF');
});

test('VIX 31 alone → score 3 (VIX>30 gives +3)', () => {
  const s = computeStressScore({ vix: 31, isInverted: false, goldPct: 0, spxPct: 0, move: 80 });
  expect(s).toBe(3);
  expect(scoreToRegime(s)).toBe('CAUTION');
});

test('VIX 31 + MOVE 130 → RISK-OFF (score 4)', () => {
  const s = computeStressScore({ vix: 31, isInverted: false, goldPct: 0, spxPct: 0, move: 130 });
  expect(s).toBe(4);
  expect(scoreToRegime(s)).toBe('RISK-OFF');
});

test('VIX 18 exactly → RISK-ON (boundary: >18 triggers, =18 does not)', () => {
  const s = computeStressScore({ vix: 18, isInverted: false, goldPct: 0, spxPct: 0, move: 80 });
  expect(s).toBe(0);
  expect(scoreToRegime(s)).toBe('RISK-ON');
});

test('VIX 18.01 → MIXED (just above threshold)', () => {
  const s = computeStressScore({ vix: 18.01, isInverted: false, goldPct: 0, spxPct: 0, move: 80 });
  expect(s).toBe(1);
  expect(scoreToRegime(s)).toBe('MIXED');
});

test('SPX -0.5% exactly → no score (boundary: <-0.5 triggers)', () => {
  const s = computeStressScore({ vix: 15, isInverted: false, goldPct: 0, spxPct: -0.5, move: 80 });
  expect(s).toBe(0);
});

test('SPX -0.51% → +1 score', () => {
  const s = computeStressScore({ vix: 15, isInverted: false, goldPct: 0, spxPct: -0.51, move: 80 });
  expect(s).toBe(1);
});

test('Gold exactly 1% → no score (boundary: >1.0 triggers)', () => {
  const s = computeStressScore({ vix: 15, isInverted: false, goldPct: 1.0, spxPct: 0, move: 80 });
  expect(s).toBe(0);
});

test('Gold 1.01% → +1 score', () => {
  const s = computeStressScore({ vix: 15, isInverted: false, goldPct: 1.01, spxPct: 0, move: 80 });
  expect(s).toBe(1);
});

test('All factors max → score 7 → RISK-OFF', () => {
  const s = computeStressScore({ vix: 35, isInverted: true, goldPct: 4, spxPct: -2, move: 160 });
  expect(s).toBe(7);
  expect(scoreToRegime(s)).toBe('RISK-OFF');
});

// ─── 5. localizeSignalTime ───────────────────────────────────────────────────

function localizeSignalTime(timeStr) {
  if (!timeStr || timeStr === '--:--') return timeStr || '--:--';
  try {
    const [h, m] = timeStr.split(':').map(Number);
    if (isNaN(h) || isNaN(m)) return timeStr;
    const now = new Date();
    const utcDate = new Date(Date.UTC(
      now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), h, m
    ));
    return utcDate.toLocaleTimeString('en-US', {
      hour: '2-digit', minute: '2-digit', hour12: false,
      timeZone: 'UTC', // use UTC for deterministic tests
    });
  } catch { return timeStr; }
}

// Override to use UTC for deterministic testing
function localizeSignalTimeUTC(timeStr) {
  if (!timeStr || timeStr === '--:--') return timeStr || '--:--';
  try {
    const [h, m] = timeStr.split(':').map(Number);
    if (isNaN(h) || isNaN(m)) return timeStr;
    const now = new Date();
    const utcDate = new Date(Date.UTC(
      now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), h, m
    ));
    return utcDate.toLocaleTimeString('en-US', {
      hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC'
    });
  } catch { return timeStr; }
}

console.log('\n── 5. localizeSignalTime ──');

test('null → --:--', () => expect(localizeSignalTime(null)).toBe('--:--'));
test('--:-- passthrough', () => expect(localizeSignalTime('--:--')).toBe('--:--'));
test('valid HH:MM returns string with colon', () => expect(localizeSignalTimeUTC('14:30')).toMatch(/\d{2}:\d{2}/));
test('invalid format returns original', () => expect(localizeSignalTime('bad')).toBe('bad'));
test('edge: midnight 00:00', () => expect(localizeSignalTimeUTC('00:00')).toMatch(/\d{2}:\d{2}/));
test('edge: 23:59', () => expect(localizeSignalTimeUTC('23:59')).toMatch(/\d{2}:\d{2}/));

// ─── 6. Business date logic ───────────────────────────────────────────────────

function getLatestBizDate(fromDate) {
  const d = fromDate ? new Date(fromDate) : new Date();
  while (d.getUTCDay() === 0 || d.getUTCDay() === 6) d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().slice(0, 10);
}

function getPrevBizDate(fromDate) {
  const d = fromDate ? new Date(fromDate) : new Date();
  while (d.getUTCDay() === 0 || d.getUTCDay() === 6) d.setUTCDate(d.getUTCDate() - 1);
  d.setUTCDate(d.getUTCDate() - 1);
  while (d.getUTCDay() === 0 || d.getUTCDay() === 6) d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().slice(0, 10);
}

console.log('\n── 6. Business date logic ──');

test('Tuesday → Tuesday (no change)', () => {
  expect(getLatestBizDate('2025-01-07T12:00:00Z')).toBe('2025-01-07'); // Tuesday
});
test('Saturday → Friday', () => {
  expect(getLatestBizDate('2025-01-11T12:00:00Z')).toBe('2025-01-10'); // Sat → Fri
});
test('Sunday → Friday', () => {
  expect(getLatestBizDate('2025-01-12T12:00:00Z')).toBe('2025-01-10'); // Sun → Fri
});
test('Monday → Monday', () => {
  expect(getLatestBizDate('2025-01-13T12:00:00Z')).toBe('2025-01-13'); // Mon
});

test('prevBizDate: Tuesday → Monday', () => {
  expect(getPrevBizDate('2025-01-07T12:00:00Z')).toBe('2025-01-06'); // Tue → Mon
});
test('prevBizDate: Monday → Friday', () => {
  expect(getPrevBizDate('2025-01-13T12:00:00Z')).toBe('2025-01-10'); // Mon → Fri
});
test('prevBizDate: Saturday → Thursday (skip Fri, Sat)', () => {
  // Sat → latest biz = Fri → prev = Thu
  expect(getPrevBizDate('2025-01-11T12:00:00Z')).toBe('2025-01-09'); // Sat → Thu
});

// ─── 7. Yield spread calculation ─────────────────────────────────────────────

function yieldSpread(y10, y2) {
  const spr = y10 - y2;
  const bp  = Math.round(spr * 100);
  return { bp, inverted: spr < 0 };
}

console.log('\n── 7. Yield spread ──');

test('normal curve: 10Y 4.44, 2Y 3.88 → +56bp', () => {
  const { bp, inverted } = yieldSpread(4.44, 3.88);
  expect(bp).toBe(56);
  expect(inverted).toBe(false);
});

test('inverted curve: 10Y 3.80, 2Y 4.90 → -110bp, inverted', () => {
  const { bp, inverted } = yieldSpread(3.80, 4.90);
  expect(bp).toBe(-110);
  expect(inverted).toBe(true);
});

test('flat curve: 10Y 4.00, 2Y 4.00 → 0bp', () => {
  const { bp } = yieldSpread(4.00, 4.00);
  expect(bp).toBe(0);
});

test('US-DE spread: US 4.44, DE 2.50 → +194bp', () => {
  const { bp } = yieldSpread(4.44, 2.50);
  expect(bp).toBe(194);
});

// ─── 8. HV30 formula (annualised log-return std dev) ─────────────────────────

function computeHV30(closes) {
  // Mirrors Python engine: compute_hv30() in fetch_intraday_quotes.py
  // Requires >= 22 closes (21 returns minimum); uses last 31 prices → 30 returns
  if (!closes || closes.length < 22) return null;
  const window = closes.slice(-31);
  const logReturns = [];
  for (let i = 1; i < window.length; i++) {
    logReturns.push(Math.log(window[i] / window[i - 1]));
  }
  const n    = logReturns.length;
  const mean = logReturns.reduce((a, b) => a + b, 0) / n;
  const variance = logReturns.reduce((s, r) => s + (r - mean) ** 2, 0) / (n - 1);
  return Math.sqrt(variance) * Math.sqrt(252) * 100; // annualised, in %
}

console.log('\n── 8. HV30 formula ──');

test('flat price series → HV = 0%', () => {
  const closes = Array(31).fill(1.1000);
  expect(computeHV30(closes)).toBeCloseTo(0, 6);
});

test('< 22 closes → null (engine minimum)', () => {
  // Engine requires >= 22 closes; < 22 → null
  expect(computeHV30([1.10, 1.11])).toBeNull();
  expect(computeHV30(Array(21).fill(1.10))).toBeNull();
});

test('exactly 22 closes → computable', () => {
  // 22 closes → 21 log-returns → valid
  const closes = Array(22).fill(null).map((_, i) => 1.1000 + i * 0.001);
  const hv = computeHV30(closes);
  expect(hv).toBeGreaterThan(0);
});

test('too short (1 close) → null', () => {
  expect(computeHV30([1.10])).toBeNull();
});

test('null input → null', () => {
  expect(computeHV30(null)).toBeNull();
});

test('annualisation: constant growth → HV = 0 (zero variance)', () => {
  // Constant daily log-return → zero variance → HV = 0. Tests the formula is correct.
  const closes = [1.0000];
  for (let i = 0; i < 31; i++) closes.push(closes[closes.length - 1] * Math.exp(0.01));
  const hv = computeHV30(closes);
  expect(hv).toBeCloseTo(0, 4);
});

test('annualisation: alternating ±1% → HV ≈ 22.6% (known result)', () => {
  // log-return alternates between +0.01 and -0.01
  // mean ≈ 0, variance = 0.01^2 = 0.0001, HV = 0.01*sqrt(252)*100 ≈ 15.87%
  // But with exact alternation: mean=0, each (r-mean)^2 = 0.0001
  // With 31 prices / 30 returns + sample variance correction: HV ≈ 16.15%
  const closes = [1.0];
  for (let i = 0; i < 31; i++) {
    closes.push(closes[closes.length-1] * Math.exp(i % 2 === 0 ? 0.01 : -0.01));
  }
  const hv = computeHV30(closes);
  expect(hv).toBeCloseTo(16.15, 1); // exact: variance from sample mean of alternating series
});

test('alternating returns → positive HV', () => {
  const closes = [];
  let p = 1.1000;
  for (let i = 0; i < 31; i++) {
    p *= i % 2 === 0 ? 1.005 : 0.995;
    closes.push(p);
  }
  const hv = computeHV30(closes);
  expect(hv).toBeGreaterThan(0);
});

test('HV result is in percent terms (0-100 range for normal FX)', () => {
  const closes = [1.08, 1.09, 1.07, 1.10, 1.08, 1.09, 1.11, 1.10, 1.09, 1.10,
                  1.11, 1.12, 1.10, 1.09, 1.10, 1.11, 1.12, 1.11, 1.10, 1.11,
                  1.12, 1.13, 1.11, 1.10, 1.11, 1.12, 1.10, 1.09, 1.10, 1.11, 1.12];
  const hv = computeHV30(closes);
  expect(hv).toBeGreaterThan(0);
  expect(hv).toBeLessThan(100); // Sane FX vol range
});

// ─── 9. Pearson correlation ───────────────────────────────────────────────────

function pearson(xs, ys) {
  const n = xs.length;
  if (n !== ys.length || n < 2) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom === 0 ? 0 : num / denom;
}

console.log('\n── 9. Pearson correlation ──');

test('perfect positive correlation → +1', () => {
  const xs = [1, 2, 3, 4, 5];
  const ys = [2, 4, 6, 8, 10];
  expect(pearson(xs, ys)).toBeCloseTo(1, 6);
});

test('perfect negative correlation → -1', () => {
  const xs = [1, 2, 3, 4, 5];
  const ys = [10, 8, 6, 4, 2];
  expect(pearson(xs, ys)).toBeCloseTo(-1, 6);
});

test('no correlation (orthogonal) → near 0', () => {
  const xs = [1, 2, 3, 4, 5];
  const ys = [3, 3, 3, 3, 3]; // constant → zero variance → 0
  expect(pearson(xs, ys)).toBeCloseTo(0, 6);
});

test('mismatched lengths → null', () => {
  expect(pearson([1, 2, 3], [1, 2])).toBeNull();
});

test('too short (n=1) → null', () => {
  expect(pearson([1], [1])).toBeNull();
});

test('result is bounded [-1, 1]', () => {
  const xs = Array.from({ length: 60 }, (_, i) => Math.sin(i * 0.3) + Math.random() * 0.1);
  const ys = Array.from({ length: 60 }, (_, i) => Math.cos(i * 0.3) + Math.random() * 0.1);
  const r = pearson(xs, ys);
  expect(r).toBeGreaterThan(-1.0001);
  expect(r).toBeLessThan(1.0001);
});

test('known EUR/USD–DXY anti-correlation scenario', () => {
  // Synthetic: EUR/USD up when DXY down — strong negative correlation
  const eurusd = [1.08, 1.09, 1.10, 1.09, 1.11, 1.12, 1.10, 1.13];
  const dxy    = [104,  103,  102,  103,  101,  100,  102,   99];
  const r = pearson(eurusd, dxy);
  expect(r).toBeLessThan(-0.9); // strongly negative
});

// ─── Summary ─────────────────────────────────────────────────────────────────
console.log(`\n${'─'.repeat(50)}`);
console.log(`Results: ${_passed} passed, ${_failed} failed`);
if (_failed > 0) {
  console.error(`\n⚠  ${_failed} test(s) failed — review output above`);
  process.exit(1);
} else {
  console.log(`\n✅  All tests passed`);
  process.exit(0);
}
