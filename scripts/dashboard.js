var _excluded = ["code"],
  _excluded2 = ["code"];
function _classCallCheck(a, n) { if (!(a instanceof n)) throw new TypeError("Cannot call a class as a function"); }
function _defineProperties(e, r) { for (var t = 0; t < r.length; t++) { var o = r[t]; o.enumerable = o.enumerable || !1, o.configurable = !0, "value" in o && (o.writable = !0), Object.defineProperty(e, _toPropertyKey(o.key), o); } }
function _createClass(e, r, t) { return r && _defineProperties(e.prototype, r), t && _defineProperties(e, t), Object.defineProperty(e, "prototype", { writable: !1 }), e; }
function _callSuper(t, o, e) { return o = _getPrototypeOf(o), _possibleConstructorReturn(t, _isNativeReflectConstruct() ? Reflect.construct(o, e || [], _getPrototypeOf(t).constructor) : o.apply(t, e)); }
function _possibleConstructorReturn(t, e) { if (e && ("object" == _typeof(e) || "function" == typeof e)) return e; if (void 0 !== e) throw new TypeError("Derived constructors may only return object or undefined"); return _assertThisInitialized(t); }
function _assertThisInitialized(e) { if (void 0 === e) throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); return e; }
function _isNativeReflectConstruct() { try { var t = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); } catch (t) {} return (_isNativeReflectConstruct = function _isNativeReflectConstruct() { return !!t; })(); }
function _getPrototypeOf(t) { return _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function (t) { return t.__proto__ || Object.getPrototypeOf(t); }, _getPrototypeOf(t); }
function _inherits(t, e) { if ("function" != typeof e && null !== e) throw new TypeError("Super expression must either be null or a function"); t.prototype = Object.create(e && e.prototype, { constructor: { value: t, writable: !0, configurable: !0 } }), Object.defineProperty(t, "prototype", { writable: !1 }), e && _setPrototypeOf(t, e); }
function _setPrototypeOf(t, e) { return _setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function (t, e) { return t.__proto__ = e, t; }, _setPrototypeOf(t, e); }
function _objectWithoutProperties(e, t) { if (null == e) return {}; var o, r, i = _objectWithoutPropertiesLoose(e, t); if (Object.getOwnPropertySymbols) { var n = Object.getOwnPropertySymbols(e); for (r = 0; r < n.length; r++) o = n[r], -1 === t.indexOf(o) && {}.propertyIsEnumerable.call(e, o) && (i[o] = e[o]); } return i; }
function _objectWithoutPropertiesLoose(r, e) { if (null == r) return {}; var t = {}; for (var n in r) if ({}.hasOwnProperty.call(r, n)) { if (-1 !== e.indexOf(n)) continue; t[n] = r[n]; } return t; }
function _createForOfIteratorHelper(r, e) { var t = "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"]; if (!t) { if (Array.isArray(r) || (t = _unsupportedIterableToArray(r)) || e && r && "number" == typeof r.length) { t && (r = t); var _n = 0, F = function F() {}; return { s: F, n: function n() { return _n >= r.length ? { done: !0 } : { done: !1, value: r[_n++] }; }, e: function e(r) { throw r; }, f: F }; } throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); } var o, a = !0, u = !1; return { s: function s() { t = t.call(r); }, n: function n() { var r = t.next(); return a = r.done, r; }, e: function e(r) { u = !0, o = r; }, f: function f() { try { a || null == t.return || t.return(); } finally { if (u) throw o; } } }; }
function _slicedToArray(r, e) { return _arrayWithHoles(r) || _iterableToArrayLimit(r, e) || _unsupportedIterableToArray(r, e) || _nonIterableRest(); }
function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function _iterableToArrayLimit(r, l) { var t = null == r ? null : "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"]; if (null != t) { var e, n, i, u, a = [], f = !0, o = !1; try { if (i = (t = t.call(r)).next, 0 === l) { if (Object(t) !== t) return; f = !1; } else for (; !(f = (e = i.call(t)).done) && (a.push(e.value), a.length !== l); f = !0); } catch (r) { o = !0, n = r; } finally { try { if (!f && null != t.return && (u = t.return(), Object(u) !== u)) return; } finally { if (o) throw n; } } return a; } }
function _arrayWithHoles(r) { if (Array.isArray(r)) return r; }
function _typeof(o) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (o) { return typeof o; } : function (o) { return o && "function" == typeof Symbol && o.constructor === Symbol && o !== Symbol.prototype ? "symbol" : typeof o; }, _typeof(o); }
function _toConsumableArray(r) { return _arrayWithoutHoles(r) || _iterableToArray(r) || _unsupportedIterableToArray(r) || _nonIterableSpread(); }
function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function _unsupportedIterableToArray(r, a) { if (r) { if ("string" == typeof r) return _arrayLikeToArray(r, a); var t = {}.toString.call(r).slice(8, -1); return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray(r, a) : void 0; } }
function _iterableToArray(r) { if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r); }
function _arrayWithoutHoles(r) { if (Array.isArray(r)) return _arrayLikeToArray(r); }
function _arrayLikeToArray(r, a) { (null == a || a > r.length) && (a = r.length); for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e]; return n; }
function _regenerator() { /*! regenerator-runtime -- Copyright (c) 2014-present, Facebook, Inc. -- license (MIT): https://github.com/babel/babel/blob/main/packages/babel-helpers/LICENSE */ var e, t, r = "function" == typeof Symbol ? Symbol : {}, n = r.iterator || "@@iterator", o = r.toStringTag || "@@toStringTag"; function i(r, n, o, i) { var c = n && n.prototype instanceof Generator ? n : Generator, u = Object.create(c.prototype); return _regeneratorDefine2(u, "_invoke", function (r, n, o) { var i, c, u, f = 0, p = o || [], y = !1, G = { p: 0, n: 0, v: e, a: d, f: d.bind(e, 4), d: function d(t, r) { return i = t, c = 0, u = e, G.n = r, a; } }; function d(r, n) { for (c = r, u = n, t = 0; !y && f && !o && t < p.length; t++) { var o, i = p[t], d = G.p, l = i[2]; r > 3 ? (o = l === n) && (u = i[(c = i[4]) ? 5 : (c = 3, 3)], i[4] = i[5] = e) : i[0] <= d && ((o = r < 2 && d < i[1]) ? (c = 0, G.v = n, G.n = i[1]) : d < l && (o = r < 3 || i[0] > n || n > l) && (i[4] = r, i[5] = n, G.n = l, c = 0)); } if (o || r > 1) return a; throw y = !0, n; } return function (o, p, l) { if (f > 1) throw TypeError("Generator is already running"); for (y && 1 === p && d(p, l), c = p, u = l; (t = c < 2 ? e : u) || !y;) { i || (c ? c < 3 ? (c > 1 && (G.n = -1), d(c, u)) : G.n = u : G.v = u); try { if (f = 2, i) { if (c || (o = "next"), t = i[o]) { if (!(t = t.call(i, u))) throw TypeError("iterator result is not an object"); if (!t.done) return t; u = t.value, c < 2 && (c = 0); } else 1 === c && (t = i.return) && t.call(i), c < 2 && (u = TypeError("The iterator does not provide a '" + o + "' method"), c = 1); i = e; } else if ((t = (y = G.n < 0) ? u : r.call(n, G)) !== a) break; } catch (t) { i = e, c = 1, u = t; } finally { f = 1; } } return { value: t, done: y }; }; }(r, o, i), !0), u; } var a = {}; function Generator() {} function GeneratorFunction() {} function GeneratorFunctionPrototype() {} t = Object.getPrototypeOf; var c = [][n] ? t(t([][n]())) : (_regeneratorDefine2(t = {}, n, function () { return this; }), t), u = GeneratorFunctionPrototype.prototype = Generator.prototype = Object.create(c); function f(e) { return Object.setPrototypeOf ? Object.setPrototypeOf(e, GeneratorFunctionPrototype) : (e.__proto__ = GeneratorFunctionPrototype, _regeneratorDefine2(e, o, "GeneratorFunction")), e.prototype = Object.create(u), e; } return GeneratorFunction.prototype = GeneratorFunctionPrototype, _regeneratorDefine2(u, "constructor", GeneratorFunctionPrototype), _regeneratorDefine2(GeneratorFunctionPrototype, "constructor", GeneratorFunction), GeneratorFunction.displayName = "GeneratorFunction", _regeneratorDefine2(GeneratorFunctionPrototype, o, "GeneratorFunction"), _regeneratorDefine2(u), _regeneratorDefine2(u, o, "Generator"), _regeneratorDefine2(u, n, function () { return this; }), _regeneratorDefine2(u, "toString", function () { return "[object Generator]"; }), (_regenerator = function _regenerator() { return { w: i, m: f }; })(); }
function _regeneratorDefine2(e, r, n, t) { var i = Object.defineProperty; try { i({}, "", {}); } catch (e) { i = 0; } _regeneratorDefine2 = function _regeneratorDefine(e, r, n, t) { function o(r, n) { _regeneratorDefine2(e, r, function (e) { return this._invoke(r, n, e); }); } r ? i ? i(e, r, { value: n, enumerable: !t, configurable: !t, writable: !t }) : e[r] = n : (o("next", 0), o("throw", 1), o("return", 2)); }, _regeneratorDefine2(e, r, n, t); }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == _typeof(i) ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != _typeof(t) || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != _typeof(i)) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
function asyncGeneratorStep(n, t, e, r, o, a, c) { try { var i = n[a](c), u = i.value; } catch (n) { return void e(n); } i.done ? t(u) : Promise.resolve(u).then(r, o); }
function _asyncToGenerator(n) { return function () { var t = this, e = arguments; return new Promise(function (r, o) { var a = n.apply(t, e); function _next(n) { asyncGeneratorStep(a, r, o, _next, _throw, "next", n); } function _throw(n) { asyncGeneratorStep(a, r, o, _next, _throw, "throw", n); } _next(void 0); }); }; }
console.log('🚀 Dashboard Forex v4 - Web Scraping Edition - Cargando...');
var _React = React,
  useState = _React.useState,
  useEffect = _React.useEffect,
  useRef = _React.useRef;

// ========== FIX 1: EDGE COMPATIBILITY - Polyfill mejorado ==========
if (!AbortSignal.timeout) {
  AbortSignal.timeout = function (ms) {
    var controller = new AbortController();
    setTimeout(function () {
      return controller.abort(new DOMException('Timeout', 'TimeoutError'));
    }, ms);
    return controller.signal;
  };
}

// Helper para fetch con timeout y retry logic
var fetchWithTimeout = /*#__PURE__*/function () {
  var _ref = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee(url) {
    var options,
      timeoutMs,
      controller,
      timeoutId,
      response,
      _args = arguments,
      _t;
    return _regenerator().w(function (_context) {
      while (1) switch (_context.p = _context.n) {
        case 0:
          options = _args.length > 1 && _args[1] !== undefined ? _args[1] : {};
          timeoutMs = _args.length > 2 && _args[2] !== undefined ? _args[2] : 8000;
          controller = new AbortController();
          timeoutId = setTimeout(function () {
            return controller.abort();
          }, timeoutMs);
          _context.p = 1;
          _context.n = 2;
          return fetch(url, _objectSpread(_objectSpread({}, options), {}, {
            signal: controller.signal
          }));
        case 2:
          response = _context.v;
          clearTimeout(timeoutId);
          return _context.a(2, response);
        case 3:
          _context.p = 3;
          _t = _context.v;
          clearTimeout(timeoutId);
          throw _t;
        case 4:
          return _context.a(2);
      }
    }, _callee, null, [[1, 3]]);
  }));
  return function fetchWithTimeout(_x) {
    return _ref.apply(this, arguments);
  };
}();

// Fetch con retry automático
var fetchWithRetry = /*#__PURE__*/function () {
  var _ref2 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee2(url) {
    var options,
      maxRetries,
      timeoutMs,
      attempt,
      isLastAttempt,
      delayMs,
      _args2 = arguments,
      _t2;
    return _regenerator().w(function (_context2) {
      while (1) switch (_context2.p = _context2.n) {
        case 0:
          options = _args2.length > 1 && _args2[1] !== undefined ? _args2[1] : {};
          maxRetries = _args2.length > 2 && _args2[2] !== undefined ? _args2[2] : 3;
          timeoutMs = _args2.length > 3 && _args2[3] !== undefined ? _args2[3] : 8000; // FIX AUDIT: timeout configurable por caller
          attempt = 1;
        case 1:
          if (!(attempt <= maxRetries)) {
            _context2.n = 7;
            break;
          }
          _context2.p = 2;
          _context2.n = 3;
          return fetchWithTimeout(url, options, timeoutMs);
        case 3:
          return _context2.a(2, _context2.v);
        case 4:
          _context2.p = 4;
          _t2 = _context2.v;
          isLastAttempt = attempt === maxRetries;
          if (!isLastAttempt) {
            _context2.n = 5;
            break;
          }
          console.error("\u274C Failed after ".concat(maxRetries, " attempts: ").concat(url));
          throw _t2;
        case 5:
          // Exponential backoff: 1s, 2s, 4s
          delayMs = 1000 * Math.pow(2, attempt - 1);
          console.warn("\u26A0\uFE0F Attempt ".concat(attempt, " failed, retrying in ").concat(delayMs, "ms..."));
          _context2.n = 6;
          return delay(delayMs);
        case 6:
          attempt++;
          _context2.n = 1;
          break;
        case 7:
          return _context2.a(2);
      }
    }, _callee2, null, [[2, 4]]);
  }));
  return function fetchWithRetry(_x2) {
    return _ref2.apply(this, arguments);
  };
}();

// ========== SISTEMA DE CACHÉ ==========

var CACHE_CONFIG = {
  FOREX_RATES: 60000,
  ECONOMIC_DATA: 86400000,
  HISTORICAL_DATA: 604800000,
  CALENDAR: 120000,
  // 2 min — el workflow corre c/10 min, no tiene sentido cachear 1h
  CENTRAL_BANK_OUTLOOK: 604800000,
  ALERTS: 300000 // 5 min — alineado con TTL de strength-scores/latest.json para evitar recomendaciones stale
};
var delay = function delay(ms) {
  return new Promise(function (resolve) {
    return setTimeout(resolve, ms);
  });
};
var CacheManager = {
  set: function set(key, data, expirationMs) {
    try {
      var item = {
        data: data,
        timestamp: Date.now(),
        expiration: expirationMs
      };
      localStorage.setItem("forex_dashboard_".concat(key), JSON.stringify(item));
      console.log("\u2705 Cached: ".concat(key));
    } catch (error) {
      console.warn('Cache storage failed:', error);
    }
  },
  get: function get(key) {
    try {
      var itemStr = localStorage.getItem("forex_dashboard_".concat(key));
      if (!itemStr) return null;
      var item = JSON.parse(itemStr);
      if (Date.now() - item.timestamp > item.expiration) {
        localStorage.removeItem("forex_dashboard_".concat(key));
        return null;
      }
      return item.data;
    } catch (error) {
      return null;
    }
  },
  clear: function clear(key) {
    return localStorage.removeItem("forex_dashboard_".concat(key));
  },
  clearAll: function clearAll() {
    Object.keys(localStorage).forEach(function (key) {
      if (key.startsWith('forex_dashboard_')) localStorage.removeItem(key);
    });
  }
};

// ========== CONFIGURACIÓN ==========

var API_CONFIG = {
  frankfurter: {
    baseUrl: 'https://api.frankfurter.app'
  }
};
var WORLD_BANK_INDICATORS = {
  gdp: 'NY.GDP.MKTP.CD',
  gdpGrowth: 'NY.GDP.MKTP.KD.ZG',
  inflation: 'FP.CPI.TOTL.ZG',
  unemployment: 'SL.UEM.TOTL.ZS',
  currentAccount: 'BN.CAB.XOKA.GD.ZS',
  debt: 'GC.DOD.TOTL.GD.ZS',
  tradeBalance: 'NE.RSB.GNFS.CD'
};

// ========== FALLBACK FUNCTIONS REMOVED ==========
// All data now comes from GitHub Pages scraped JSON files
// These functions are no longer needed (legacy scraping removed)

var fetchWorldBankData = /*#__PURE__*/function () {
  var _ref3 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee3(countryCode, indicator) {
    var cacheKey, cached, githubUrl, response, scrapedData, indicatorMap, scrapedKey, value, finalValue, _t3;
    return _regenerator().w(function (_context3) {
      while (1) switch (_context3.p = _context3.n) {
        case 0:
          cacheKey = "scraped_".concat(countryCode, "_").concat(indicator);
          cached = CacheManager.get(cacheKey);
          if (!(cached !== null)) {
            _context3.n = 1;
            break;
          }
          return _context3.a(2, cached);
        case 1:
          _context3.p = 1;
          githubUrl = "https://globalinvesting.github.io/economic-data/".concat(countryCode, ".json");
          _context3.n = 2;
          return fetchWithRetry(githubUrl, {
            cache: 'no-cache',
            mode: 'cors'
          }, 3);
        case 2:
          response = _context3.v;
          if (!response.ok) {
            _context3.n = 4;
            break;
          }
          _context3.n = 3;
          return response.json();
        case 3:
          scrapedData = _context3.v;
          if (!scrapedData.data) {
            _context3.n = 4;
            break;
          }
          indicatorMap = _defineProperty(_defineProperty(_defineProperty(_defineProperty(_defineProperty(_defineProperty(_defineProperty({}, WORLD_BANK_INDICATORS.gdp, 'gdp'), WORLD_BANK_INDICATORS.gdpGrowth, 'gdpGrowth'), WORLD_BANK_INDICATORS.inflation, 'inflation'), WORLD_BANK_INDICATORS.unemployment, 'unemployment'), WORLD_BANK_INDICATORS.currentAccount, 'currentAccount'), WORLD_BANK_INDICATORS.debt, 'debt'), WORLD_BANK_INDICATORS.tradeBalance, 'tradeBalance');
          scrapedKey = indicatorMap[indicator];
          if (!(scrapedKey && scrapedData.data[scrapedKey] !== undefined)) {
            _context3.n = 4;
            break;
          }
          value = scrapedData.data[scrapedKey];
          finalValue = value;
          if (scrapedKey === 'gdp') {
            finalValue = value / 1000;
          } else if (scrapedKey === 'tradeBalance') {
            finalValue = value;
          }
          console.log("\u2705 Scraped data for ".concat(countryCode, " ").concat(scrapedKey, ": ").concat(finalValue));
          CacheManager.set(cacheKey, finalValue, CACHE_CONFIG.ECONOMIC_DATA);
          return _context3.a(2, finalValue);
        case 4:
          _context3.n = 6;
          break;
        case 5:
          _context3.p = 5;
          _t3 = _context3.v;
          console.warn("Scraped data fetch failed for ".concat(countryCode, ":"), _t3.message);
        case 6:
          console.warn("\u274C No scraped data available for ".concat(countryCode, " ").concat(indicator));
          return _context3.a(2, null);
      }
    }, _callee3, null, [[1, 5]]);
  }));
  return function fetchWorldBankData(_x3, _x4) {
    return _ref3.apply(this, arguments);
  };
}();
var fetchInterestRate = /*#__PURE__*/function () {
  var _ref4 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee4(country) {
    var cacheKey, cached, githubUrl, response, data, latestObs, value, result, _t4;
    return _regenerator().w(function (_context4) {
      while (1) switch (_context4.p = _context4.n) {
        case 0:
          cacheKey = "interest_rate_".concat(country.code);
          cached = CacheManager.get(cacheKey);
          if (!(cached !== null)) {
            _context4.n = 1;
            break;
          }
          return _context4.a(2, cached);
        case 1:
          console.log("\uD83D\uDD0D Fetching interest rate for ".concat(country.code, "..."));

          // Estrategia principal: GitHub Pages JSON
          _context4.p = 2;
          githubUrl = "https://globalinvesting.github.io/rates/".concat(country.code, ".json");
          _context4.n = 3;
          return fetchWithRetry(githubUrl, {
            cache: 'no-cache',
            mode: 'cors'
          }, 3);
        case 3:
          response = _context4.v;
          if (!response.ok) {
            _context4.n = 5;
            break;
          }
          _context4.n = 4;
          return response.json();
        case 4:
          data = _context4.v;
          if (!(data.observations && data.observations.length > 0)) {
            _context4.n = 5;
            break;
          }
          latestObs = data.observations[0];
          if (!(latestObs.value && latestObs.value !== '.')) {
            _context4.n = 5;
            break;
          }
          value = parseFloat(latestObs.value);
          if (!(!isNaN(value) && value >= -1 && value < 25)) {
            _context4.n = 5;
            break;
          }
          console.log("\u2705 GitHub Pages rate for ".concat(country.code, ": ").concat(value, "% (period: ").concat(latestObs.date || 'N/A', ")"));
          result = {
            rate: value,
            date: latestObs.date || null
          };
          CacheManager.set(cacheKey, result, CACHE_CONFIG.ECONOMIC_DATA);
          return _context4.a(2, result);
        case 5:
          _context4.n = 7;
          break;
        case 6:
          _context4.p = 6;
          _t4 = _context4.v;
          if (_t4.name !== 'AbortError') {
            console.warn("GitHub Pages failed for ".concat(country.code, ":"), _t4.message);
          }
        case 7:
          // Alpha Vantage eliminado — rates/USD.json se genera diariamente
          // desde el workflow de GitHub Actions.
          // Si llegamos aquí, el archivo rates/USD.json no estaba disponible.
          if (country.code === 'USD') {
            console.warn('rates/USD.json no disponible — sin fallback para USD');
          }
          console.error("\u274C All sources exhausted for ".concat(country.code, " - NO DATA AVAILABLE"));
          return _context4.a(2, null);
      }
    }, _callee4, null, [[2, 6]]);
  }));
  return function fetchInterestRate(_x5) {
    return _ref4.apply(this, arguments);
  };
}();

// ========== FETCH EXTENDED DATA (bonds, sentiment, flows) ==========
var fetchExtendedData = /*#__PURE__*/function () {
  var _ref5 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee5(countryCode) {
    var cacheKey, cached, url, response, json, _t5;
    return _regenerator().w(function (_context5) {
      while (1) switch (_context5.p = _context5.n) {
        case 0:
          cacheKey = "extended_".concat(countryCode);
          cached = CacheManager.get(cacheKey);
          if (!(cached !== null)) {
            _context5.n = 1;
            break;
          }
          return _context5.a(2, cached);
        case 1:
          _context5.p = 1;
          url = "https://globalinvesting.github.io/extended-data/".concat(countryCode, ".json");
          _context5.n = 2;
          return fetchWithRetry(url, {
            cache: 'no-cache',
            mode: 'cors'
          }, 5, 13000); // FIX AUDIT: extended-data crítico para scores — 5 reintentos, 13s timeout
        case 2:
          response = _context5.v;
          if (!response.ok) {
            _context5.n = 4;
            break;
          }
          _context5.n = 3;
          return response.json();
        case 3:
          json = _context5.v;
          if (!json.data) {
            _context5.n = 4;
            break;
          }
          console.log("\u2705 Extended data for ".concat(countryCode, ":"), Object.keys(json.data).filter(function (k) {
            return json.data[k] !== null;
          }));
          CacheManager.set(cacheKey, json, CACHE_CONFIG.ECONOMIC_DATA);
          return _context5.a(2, json);
        case 4:
          _context5.n = 6;
          break;
        case 5:
          _context5.p = 5;
          _t5 = _context5.v;
          console.warn("Extended data fetch failed for ".concat(countryCode, ":"), _t5.message);
        case 6:
          return _context5.a(2, null);
      }
    }, _callee5, null, [[1, 5]]);
  }));
  return function fetchExtendedData(_x6) {
    return _ref5.apply(this, arguments);
  };
}();

// ✅ OUTLOOK DINÁMICO con validación de última decisión del banco central
var determineCentralBankOutlook = /*#__PURE__*/function () {
  var _ref6 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee6(countryCode, economicData) {
    var _economicData$rateMom, _rm;
    var cacheKey, cached, rm12, rm24, rm, outlook, lrd, _lrd;
    return _regenerator().w(function (_context6) {
      while (1) switch (_context6.n) {
        case 0:
          // v5.10 — tres mejoras data-driven derivadas de rates/ (fuente autoritativa):
          //
          // FIX-A: rateMomentum efectivo = max(rm_12M, rm_24M × 0.6)
          //   Captura ciclos graduales de subidas (ej. BoJ: 12M=+0.25 pero 24M=+0.85).
          //   El factor 0.6 evita que el histórico largo domine sobre la señal reciente.
          //   Solo aplica en dirección hawkish para no exagerar dovishness.
          //
          // FIX-B: threshold Dovish -0.75 → -0.80
          //   Evita que BCs en pausa queden clasificados como Dovish por estar exactamente
          //   en el límite (ej. USD rm=-0.75 con Fed pausada → correctamente Neutral).
          //
          // FIX-C: M-01 se activa con rm < -0.20 (antes solo cuando outlook=Neutral)
          //   Captura BCs que iniciaron ciclo de recortes pero rm_12M no llega a -0.80
          //   (ej. RBA con 1 corte: rm=-0.25, lastDecision=CUT → correctamente Dovish).
          cacheKey = "outlook_".concat(countryCode);
          cached = CacheManager.get(cacheKey);
          if (!(cached !== null)) {
            _context6.n = 1;
            break;
          }
          return _context6.a(2, cached);
        case 1:
          rm12 = economicData.rateMomentum;
          rm24 = (_economicData$rateMom = economicData.rateMomentum24M) !== null && _economicData$rateMom !== void 0 ? _economicData$rateMom : null; // FIX-A: composite rm — usa 24M solo para amplificar señal hawkish
          // Si rm12 es positivo y hay rm24 disponible, toma el mayor (con descuento 0.6 en 24M)
          rm = rm12;
          if (rm12 !== null && rm24 !== null && rm12 >= 0 && rm24 > 0) {
            rm = Math.max(rm12, rm24 * 0.6);
          }
          if (rm === undefined || rm === null || isNaN(rm)) {
            outlook = 'Neutral';
          } else if (rm > 0.20) {
            // v6.2: 0.50 → 0.20 (data-driven)
            // +0.25pp en 12M = mínimo 1 hike ejecutado = ciclo hawkish.
            // En el G8 actual solo JPY tiene rm > 0 — el ajuste es quirúrgico.
            outlook = 'Hawkish';
          } else if (rm <= -0.80) {
            // FIX-B: -0.75 → -0.80
            outlook = 'Dovish';
          } else {
            outlook = 'Neutral';
          }

          // FIX-C: M-01 — si rm < -0.20 Y última decisión fue un corte → Dovish
          // (antes solo disparaba cuando outlook base era Neutral)
          if (outlook !== 'Hawkish' && rm12 !== null && rm12 < -0.20 && economicData._aiLastRateDecision) {
            lrd = economicData._aiLastRateDecision;
            if (lrd.direction === 'BAJÓ' && lrd.delta !== undefined && lrd.delta <= -0.25) {
              outlook = 'Dovish';
              console.log("\u2713 M-01 outlook\u2192Dovish for ".concat(countryCode, ": rm=").concat(rm12, ", last CUT ").concat(lrd.delta, "pp"));
            }
          }

          // M-01 hawkish: si rm entre 0 y +0.50 Y última decisión fue subida → Hawkish
          if (outlook === 'Neutral' && economicData._aiLastRateDecision) {
            _lrd = economicData._aiLastRateDecision;
            if (_lrd.direction === 'SUBIÓ' && _lrd.delta !== undefined && _lrd.delta >= 0.25) {
              outlook = 'Hawkish';
              console.log("\u2713 M-01 outlook\u2192Hawkish for ".concat(countryCode, ": last HIKE ").concat(_lrd.delta, "pp"));
            }
          }
          console.log("\u2713 v5.10 outlook for ".concat(countryCode, ": ").concat(outlook, " (rm12=").concat(rm12, ", rm_eff=").concat((_rm = rm) === null || _rm === void 0 ? void 0 : _rm.toFixed(2), ")"));
          CacheManager.set(cacheKey, outlook, CACHE_CONFIG.CENTRAL_BANK_OUTLOOK);
          return _context6.a(2, outlook);
      }
    }, _callee6);
  }));
  return function determineCentralBankOutlook(_x7, _x8) {
    return _ref6.apply(this, arguments);
  };
}();
var fetchForexRates = /*#__PURE__*/function () {
  var _ref7 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee8() {
    var cacheKey, cached, fetchWithIndividualTimeout, endpoints, rates, _t7;
    return _regenerator().w(function (_context8) {
      while (1) switch (_context8.p = _context8.n) {
        case 0:
          cacheKey = 'forex_rates';
          cached = CacheManager.get(cacheKey);
          if (!(cached !== null)) {
            _context8.n = 1;
            break;
          }
          return _context8.a(2, cached);
        case 1:
          // FIX: lanzar todos los endpoints en paralelo con timeout individual de 5s
          // El primero que responda válido gana — sin esperar a los que fallen
          fetchWithIndividualTimeout = /*#__PURE__*/function () {
            var _ref8 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee7(url, name) {
              var timeoutMs,
                controller,
                timer,
                response,
                data,
                usdRate,
                rates,
                _args7 = arguments,
                _t6;
              return _regenerator().w(function (_context7) {
                while (1) switch (_context7.p = _context7.n) {
                  case 0:
                    timeoutMs = _args7.length > 2 && _args7[2] !== undefined ? _args7[2] : 5000;
                    controller = new AbortController();
                    timer = setTimeout(function () {
                      return controller.abort();
                    }, timeoutMs);
                    _context7.p = 1;
                    _context7.n = 2;
                    return fetch(url, {
                      signal: controller.signal
                    });
                  case 2:
                    response = _context7.v;
                    clearTimeout(timer);
                    if (response.ok) {
                      _context7.n = 3;
                      break;
                    }
                    throw new Error("HTTP ".concat(response.status));
                  case 3:
                    _context7.n = 4;
                    return response.json();
                  case 4:
                    data = _context7.v;
                    if (!data.rates) {
                      _context7.n = 5;
                      break;
                    }
                    console.log("\u2713 Forex rates from ".concat(name));
                    return _context7.a(2, data.rates);
                  case 5:
                    if (!(data.base === 'EUR' && data.rates && data.rates.USD)) {
                      _context7.n = 6;
                      break;
                    }
                    usdRate = 1 / data.rates.USD;
                    rates = {};
                    Object.keys(data.rates).forEach(function (c) {
                      if (c !== 'USD') rates[c] = data.rates[c] * usdRate;
                    });
                    console.log("\u2713 Forex rates from ".concat(name, " (EUR\u2192USD converted)"));
                    return _context7.a(2, rates);
                  case 6:
                    throw new Error('No rates field');
                  case 7:
                    _context7.p = 7;
                    _t6 = _context7.v;
                    clearTimeout(timer);
                    throw _t6;
                  case 8:
                    return _context7.a(2);
                }
              }, _callee7, null, [[1, 7]]);
            }));
            return function fetchWithIndividualTimeout(_x9, _x0) {
              return _ref8.apply(this, arguments);
            };
          }();
          endpoints = [{
            url: "".concat(API_CONFIG.frankfurter.baseUrl, "/latest?from=USD"),
            name: 'Frankfurter USD'
          }, {
            url: 'https://api.frankfurter.app/latest?from=EUR',
            name: 'Frankfurter EUR'
          }, {
            url: 'https://api.exchangerate-api.com/v4/latest/USD',
            name: 'ExchangeRate-API'
          }, {
            url: 'https://open.er-api.com/v6/latest/USD',
            name: 'Open Exchange'
          }];
          _context8.p = 2;
          _context8.n = 3;
          return Promise.any(endpoints.map(function (ep) {
            return fetchWithIndividualTimeout(ep.url, ep.name, 5000);
          }));
        case 3:
          rates = _context8.v;
          CacheManager.set(cacheKey, rates, CACHE_CONFIG.FOREX_RATES);
          return _context8.a(2, rates);
        case 4:
          _context8.p = 4;
          _t7 = _context8.v;
          console.warn('❌ All forex rate sources failed:', _t7);
          return _context8.a(2, null);
      }
    }, _callee8, null, [[2, 4]]);
  }));
  return function fetchForexRates() {
    return _ref7.apply(this, arguments);
  };
}();

// Singleton: el archivo meetings.json es el mismo para las 8 divisas.
// Se descarga UNA sola vez y se reutiliza en todas las llamadas.
var _meetingsJsonPromise = null;
var _getMeetingsJson = function _getMeetingsJson() {
  if (!_meetingsJsonPromise) {
    _meetingsJsonPromise = fetchWithRetry('https://globalinvesting.github.io/meetings-data/meetings.json', {
      cache: 'no-cache',
      mode: 'cors'
    }, 3).then(function (r) {
      return r.ok ? r.json() : null;
    }).catch(function () {
      return null;
    });
  }
  return _meetingsJsonPromise;
};
var fetchCentralBankMeetings = /*#__PURE__*/function () {
  var _ref9 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee9(countryCode) {
    var cacheKey, cached, data, meetingInfo, _t8;
    return _regenerator().w(function (_context9) {
      while (1) switch (_context9.p = _context9.n) {
        case 0:
          cacheKey = "cb_meeting_".concat(countryCode);
          cached = CacheManager.get(cacheKey);
          if (!(cached !== null)) {
            _context9.n = 1;
            break;
          }
          return _context9.a(2, cached);
        case 1:
          _context9.p = 1;
          _context9.n = 2;
          return _getMeetingsJson();
        case 2:
          data = _context9.v;
          if (!data) {
            _context9.n = 3;
            break;
          }
          meetingInfo = data.meetings && data.meetings[countryCode];
          if (!(meetingInfo && meetingInfo.nextMeeting)) {
            _context9.n = 3;
            break;
          }
          console.log("\u2705 Meeting date for ".concat(countryCode, ": ").concat(meetingInfo.nextMeeting, " (source: cbrates.com)"));
          CacheManager.set(cacheKey, meetingInfo.nextMeeting, CACHE_CONFIG.CENTRAL_BANK_OUTLOOK);
          return _context9.a(2, meetingInfo.nextMeeting);
        case 3:
          _context9.n = 5;
          break;
        case 4:
          _context9.p = 4;
          _t8 = _context9.v;
          console.warn("Meetings fetch failed for ".concat(countryCode, ":"), _t8.message);
        case 5:
          console.warn("\u274C No meeting data available for ".concat(countryCode));
          return _context9.a(2, 'Por confirmar');
      }
    }, _callee9, null, [[1, 4]]);
  }));
  return function fetchCentralBankMeetings(_x1) {
    return _ref9.apply(this, arguments);
  };
}();
var formatEventTime = function formatEventTime(timeUTC, dateISO) {
  if (!timeUTC || !dateISO) return '';
  try {
    var utcDatetime = new Date("".concat(dateISO, "T").concat(timeUTC, ":00Z"));
    if (isNaN(utcDatetime.getTime())) return timeUTC;
    return utcDatetime.toLocaleTimeString('es-ES', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    });
  } catch (_unused) {
    return timeUTC;
  }
};
var generateDynamicEconomicCalendar = function generateDynamicEconomicCalendar() {
  var today = new Date();
  var events = [];
  var economicEvents = [{
    country: 'USD',
    event: 'IPC (Inflación)',
    impact: 'high',
    dayOfMonth: 12,
    time: '14:30',
    flag: '🇺🇸'
  }, {
    country: 'EUR',
    event: 'PIB Preliminar',
    impact: 'high',
    dayOfMonth: 30,
    time: '11:00',
    flag: '🇪🇺',
    quarterly: true
  }, {
    country: 'GBP',
    event: 'IPC UK',
    impact: 'high',
    dayOfMonth: 18,
    time: '08:00',
    flag: '🇬🇧'
  }, {
    country: 'JPY',
    event: 'IPC Japón',
    impact: 'high',
    dayOfMonth: 22,
    time: '00:30',
    flag: '🇯🇵'
  }, {
    country: 'AUD',
    event: 'Decisión Tasa RBA',
    impact: 'high',
    dayOfMonth: 4,
    time: '04:30',
    flag: '🇦🇺',
    everyMonth: true
  }, {
    country: 'CAD',
    event: 'IPC Canadá',
    impact: 'high',
    dayOfMonth: 20,
    time: '14:30',
    flag: '🇨🇦'
  }, {
    country: 'CHF',
    event: 'IPC Suiza',
    impact: 'medium',
    dayOfMonth: 6,
    time: '09:30',
    flag: '🇨🇭'
  }, {
    country: 'NZD',
    event: 'PIB Trimestral',
    impact: 'high',
    dayOfMonth: 19,
    time: '21:45',
    flag: '🇳🇿',
    quarterly: true
  }];
  var _loop = function _loop() {
    var targetMonth = (today.getMonth() + monthOffset) % 12;
    var targetYear = today.getMonth() + monthOffset >= 12 ? today.getFullYear() + 1 : today.getFullYear();
    economicEvents.forEach(function (eventTemplate) {
      var shouldInclude = true;
      if (eventTemplate.quarterly && targetMonth % 3 !== 0) shouldInclude = false;
      if (eventTemplate.everyMonth) shouldInclude = true;
      if (shouldInclude) {
        var eventDate = new Date(targetYear, targetMonth, eventTemplate.dayOfMonth);
        if (eventDate >= today) {
          var monthNames = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'];
          events.push({
            date: "".concat(eventTemplate.dayOfMonth, " ").concat(monthNames[targetMonth]),
            time: eventTemplate.time,
            country: eventTemplate.country,
            event: eventTemplate.event,
            impact: eventTemplate.impact,
            flag: eventTemplate.flag,
            sortDate: eventDate
          });
        }
      }
    });
  };
  for (var monthOffset = 0; monthOffset <= 1; monthOffset++) {
    _loop();
  }
  events.sort(function (a, b) {
    return a.sortDate - b.sortDate;
  });
  return events.slice(0, 15).map(function (e) {
    delete e.sortDate;
    return e;
  });
};

// ── STRENGTH SCORES — engine loader ──────────────────────────────────────────
// Source of truth: strength-scores/latest.json — generated daily at 08:30 UTC
// by calculate_scores.py in the private engine repo.
// Fallback: strength-scores/all.json (last available snapshot) with stale badge.
// No scoring logic lives in this file.
var _precomputedScores = null; // active scores payload
var _scoresDataDate = null; // ISO date string of the loaded snapshot

var fetchStrengthScores = /*#__PURE__*/function () {
  var _ref0 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee0() {
    var cacheKey, cached, CURRENCIES, validate, url, resp, data, payload, _url, _resp, allData, snapshots, last, _payload, _t9, _t0;
    return _regenerator().w(function (_context0) {
      while (1) switch (_context0.p = _context0.n) {
        case 0:
          cacheKey = 'strength_scores_latest';
          cached = CacheManager.get(cacheKey);
          if (!(cached !== null)) {
            _context0.n = 1;
            break;
          }
          _precomputedScores = cached.scores;
          _scoresDataDate = cached.date;
          return _context0.a(2, cached);
        case 1:
          CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'];
          validate = function validate(data) {
            return data && data.scores && CURRENCIES.every(function (c) {
              return data.scores[c] !== undefined && data.scores[c] !== null && typeof data.scores[c].score === 'number';
            });
          }; // ── Primary: latest.json ──────────────────────────────────────────────────
          _context0.p = 2;
          url = 'https://globalinvesting.github.io/strength-scores/latest.json';
          _context0.n = 3;
          return fetchWithTimeout(url, {
            cache: 'no-cache',
            mode: 'cors'
          }, 5000);
        case 3:
          resp = _context0.v;
          if (resp.ok) {
            _context0.n = 4;
            break;
          }
          throw new Error('latest.json not available');
        case 4:
          _context0.n = 5;
          return resp.json();
        case 5:
          data = _context0.v;
          if (validate(data)) {
            _context0.n = 6;
            break;
          }
          throw new Error('Incomplete scores payload');
        case 6:
          payload = {
            scores: data.scores,
            date: data.lastUpdate,
            stale: false,
            version: data.modelVersion
          };
          CacheManager.set(cacheKey, payload, 3600);
          _precomputedScores = payload.scores;
          _scoresDataDate = payload.date;
          CacheManager.clear('forex_pair_recommendations'); // FIX AUDIT: invalidar recomendaciones al cargar scores frescos
          console.log("[scores] Loaded v".concat(data.modelVersion, " (").concat(data.lastUpdate, ")"));
          return _context0.a(2, payload);
        case 7:
          _context0.p = 7;
          _t9 = _context0.v;
          console.warn('[scores] latest.json failed, trying all.json:', _t9.message);
          _context0.p = 8;
          _url = 'https://globalinvesting.github.io/strength-scores/all.json';
          _context0.n = 9;
          return fetchWithTimeout(_url, {
            cache: 'no-cache',
            mode: 'cors'
          }, 5000);
        case 9:
          _resp = _context0.v;
          if (_resp.ok) {
            _context0.n = 10;
            break;
          }
          throw new Error('all.json not available');
        case 10:
          _context0.n = 11;
          return _resp.json();
        case 11:
          allData = _context0.v;
          // all.json is an array of snapshots sorted ascending — take the last valid one
          snapshots = Array.isArray(allData) ? allData : allData.snapshots || [];
          last = _toConsumableArray(snapshots).reverse().find(function (s) {
            return validate(s);
          });
          if (last) {
            _context0.n = 12;
            break;
          }
          throw new Error('No valid snapshot in all.json');
        case 12:
          _payload = {
            scores: last.scores,
            date: last.date || last.lastUpdate,
            stale: true,
            version: last.modelVersion
          };
          CacheManager.set(cacheKey, _payload, 1800); // shorter cache when stale
          _precomputedScores = _payload.scores;
          _scoresDataDate = _payload.date;
          CacheManager.clear('forex_pair_recommendations'); // FIX AUDIT: invalidar recomendaciones al cargar scores (aunque sean stale)
          console.warn("[scores] Using stale snapshot from ".concat(_payload.date));
          return _context0.a(2, _payload);
        case 13:
          _context0.p = 13;
          _t0 = _context0.v;
          console.error('[scores] All sources failed:', _t0.message);
          _precomputedScores = null;
          _scoresDataDate = null;
          return _context0.a(2, null);
      }
    }, _callee0, null, [[8, 13], [2, 7]]);
  }));
  return function fetchStrengthScores() {
    return _ref0.apply(this, arguments);
  };
}();

// Returns score object from engine. If no data available, returns a neutral placeholder.
var getStrength = function getStrength(currency) {
  // Guard: s must be a non-null object with a numeric score — rejects null entries
  // and stale cache payloads from before v6.4 that had a different structure.
  var s = _precomputedScores && _precomputedScores[currency];
  if (s && _typeof(s) === 'object' && typeof s.score === 'number') {
    var _s$dataQuality;
    return {
      score: s.score,
      confidence: s.confidence || {
        lower: s.score - 5,
        upper: s.score + 5
      },
      contributions: s.contributions || {},
      dataQuality: (_s$dataQuality = s.dataQuality) !== null && _s$dataQuality !== void 0 ? _s$dataQuality : 1,
      scoringData: s.scoringData || {},
      bcOutlook: s.bcOutlook || null,
      contextualAdj: s.contextualAdj || {},
      _stale: !!(_scoresDataDate && Date.now() - new Date(_scoresDataDate) > 48 * 3600 * 1000),
      _dataDate: _scoresDataDate
    };
  }
  // Stale/corrupt cache entry — invalidate and trigger background refresh
  if (_precomputedScores && _precomputedScores[currency] !== undefined) {
    console.warn("[scores] Invalid entry for ".concat(currency, " \u2014 clearing scores cache"));
    _precomputedScores = null;
    try {
      localStorage.removeItem('forex_dashboard_strength_scores_latest');
    } catch (e) {}
    fetchStrengthScores().catch(function (e) {
      return console.warn('[scores] Background reload failed:', e);
    });
  }
  // No scores available — return placeholder so UI never crashes
  return {
    score: null,
    confidence: {
      lower: null,
      upper: null
    },
    contributions: {},
    dataQuality: 0,
    scoringData: {},
    bcOutlook: null,
    contextualAdj: {},
    _unavailable: true
  };
};
var fetchAIAnalysis = /*#__PURE__*/function () {
  var _ref1 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee10() {
    var cacheKey, cached, indexUrl, indexResponse, index, analyses, promises, _t10;
    return _regenerator().w(function (_context10) {
      while (1) switch (_context10.p = _context10.n) {
        case 0:
          cacheKey = 'ai_analysis_Groq';
          cached = CacheManager.get(cacheKey);
          if (!(cached !== null)) {
            _context10.n = 1;
            break;
          }
          return _context10.a(2, cached);
        case 1:
          _context10.p = 1;
          indexUrl = 'https://globalinvesting.github.io/ai-analysis/index.json';
          _context10.n = 2;
          return fetchWithRetry(indexUrl, {
            cache: 'no-cache',
            mode: 'cors'
          }, 2);
        case 2:
          indexResponse = _context10.v;
          if (indexResponse.ok) {
            _context10.n = 3;
            break;
          }
          throw new Error('Index no disponible');
        case 3:
          _context10.n = 4;
          return indexResponse.json();
        case 4:
          index = _context10.v;
          analyses = {}; // Cargar todas las divisas en paralelo
          promises = (index.currencies || []).map(/*#__PURE__*/function () {
            var _ref10 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee1(currency) {
              var url, response, data, _data$dataSnapshot, _t1;
              return _regenerator().w(function (_context1) {
                while (1) switch (_context1.p = _context1.n) {
                  case 0:
                    _context1.p = 0;
                    url = "https://globalinvesting.github.io/ai-analysis/".concat(currency, ".json");
                    _context1.n = 1;
                    return fetchWithTimeout(url, {
                      mode: 'cors'
                    }, 8000);
                  case 1:
                    response = _context1.v;
                    if (!response.ok) {
                      _context1.n = 3;
                      break;
                    }
                    _context1.n = 2;
                    return response.json();
                  case 2:
                    data = _context1.v;
                    if (data.analysis) {
                      analyses[currency] = {
                        text: data.analysis,
                        generatedAt: data.generatedAt,
                        model: data.model || 'llama-3.3-70b-versatile',
                        // M-01 FIX: extraer lastRateDecision para enriquecer outlookScore
                        // Permite distinguir BC que pausó tras ciclo agresivo vs BC inactivo
                        lastRateDecision: ((_data$dataSnapshot = data.dataSnapshot) === null || _data$dataSnapshot === void 0 ? void 0 : _data$dataSnapshot.lastRateDecision) || null
                      };
                    }
                  case 3:
                    _context1.n = 5;
                    break;
                  case 4:
                    _context1.p = 4;
                    _t1 = _context1.v;
                    console.warn("No an\xE1lisis AI para ".concat(currency, ":"), _t1.message);
                  case 5:
                    return _context1.a(2);
                }
              }, _callee1, null, [[0, 4]]);
            }));
            return function (_x10) {
              return _ref10.apply(this, arguments);
            };
          }());
          _context10.n = 5;
          return Promise.all(promises);
        case 5:
          if (!(Object.keys(analyses).length > 0)) {
            _context10.n = 6;
            break;
          }
          console.log("\u2705 An\xE1lisis Groq cargados: ".concat(Object.keys(analyses).join(', ')));
          // Cache de 6 horas — Groq genera una vez al día
          CacheManager.set(cacheKey, analyses, 6 * 60 * 60 * 1000);
          return _context10.a(2, analyses);
        case 6:
          _context10.n = 8;
          break;
        case 7:
          _context10.p = 7;
          _t10 = _context10.v;
          console.warn('Análisis AI no disponibles, usando algorítmico:', _t10.message);
        case 8:
          return _context10.a(2, null);
      }
    }, _callee10, null, [[1, 7]]);
  }));
  return function fetchAIAnalysis() {
    return _ref1.apply(this, arguments);
  };
}();
var fetchEconomicCalendar = /*#__PURE__*/function () {
  var _ref11 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee11() {
    var cacheKey, cached, url, response, data, result, errorResult, _t11;
    return _regenerator().w(function (_context11) {
      while (1) switch (_context11.p = _context11.n) {
        case 0:
          cacheKey = 'economic_calendar';
          cached = CacheManager.get(cacheKey);
          if (!(cached !== null)) {
            _context11.n = 1;
            break;
          }
          return _context11.a(2, cached);
        case 1:
          _context11.p = 1;
          url = 'https://globalinvesting.github.io/calendar-data/calendar.json';
          _context11.n = 2;
          return fetchWithRetry(url, {
            cache: 'no-cache',
            mode: 'cors'
          }, 3);
        case 2:
          response = _context11.v;
          if (!response.ok) {
            _context11.n = 4;
            break;
          }
          _context11.n = 3;
          return response.json();
        case 3:
          data = _context11.v;
          if (!(data.events && data.events.length > 0)) {
            _context11.n = 4;
            break;
          }
          console.log("\u2705 Real calendar loaded: ".concat(data.events.length, " events (source: ").concat(data.source, ")"));
          // Store both events array and metadata
          result = {
            events: data.events,
            lastUpdate: data.lastUpdate,
            generatedAt: data.generatedAt || null,
            source: data.source,
            impactCounts: data.impactCounts || {},
            currencyCounts: data.currencyCounts || {}
          };
          CacheManager.set(cacheKey, result, CACHE_CONFIG.CALENDAR);
          return _context11.a(2, result);
        case 4:
          _context11.n = 6;
          break;
        case 5:
          _context11.p = 5;
          _t11 = _context11.v;
          console.warn('Real calendar fetch failed:', _t11.message);
        case 6:
          // ❌ NO FALLBACK — si el scraping falla, devolver estado de error explícito.
          // Nunca mostrar datos sintéticos en una herramienta financiera profesional.
          console.error('⛔ Calendar scraping failed — returning empty error state');
          errorResult = {
            events: [],
            lastUpdate: null,
            source: null,
            status: 'error',
            errorMessage: 'No se pudieron obtener datos del calendario. Consulte directamente forexfactory.com o investing.com/economic-calendar',
            impactCounts: {},
            currencyCounts: {}
          }; // No cachear el error — reintentar en la próxima llamada
          return _context11.a(2, errorResult);
      }
    }, _callee11, null, [[1, 5]]);
  }));
  return function fetchEconomicCalendar() {
    return _ref11.apply(this, arguments);
  };
}();
// ← cierre de fetchEconomicCalendar

var fetchHistoricalFXData = /*#__PURE__*/function () {
  var _ref12 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee12(currencyCode) {
    return _regenerator().w(function (_context12) {
      while (1) switch (_context12.n) {
        case 0:
          return _context12.a(2, null);
      }
    }, _callee12);
  }));
  return function fetchHistoricalFXData(_x11) {
    return _ref12.apply(this, arguments);
  };
}();
var fetchRealHistoricalRates = /*#__PURE__*/function () {
  var _ref13 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee13(currencyCode) {
    var cacheKey, cached, url, response, data, observations, recent, MONTH_NAMES_ES, labels, rates, result, _t12;
    return _regenerator().w(function (_context13) {
      while (1) switch (_context13.p = _context13.n) {
        case 0:
          cacheKey = "real_historical_rates_".concat(currencyCode);
          cached = CacheManager.get(cacheKey);
          if (!(cached !== null)) {
            _context13.n = 1;
            break;
          }
          return _context13.a(2, cached);
        case 1:
          _context13.p = 1;
          url = "https://globalinvesting.github.io/rates/".concat(currencyCode, ".json");
          _context13.n = 2;
          return fetchWithRetry(url, {
            cache: 'no-cache',
            mode: 'cors'
          }, 3);
        case 2:
          response = _context13.v;
          if (response.ok) {
            _context13.n = 3;
            break;
          }
          throw new Error("HTTP ".concat(response.status));
        case 3:
          _context13.n = 4;
          return response.json();
        case 4:
          data = _context13.v;
          observations = data.observations || [];
          if (!(observations.length < 2)) {
            _context13.n = 5;
            break;
          }
          console.warn("Insuficientes observaciones hist\xF3ricas para ".concat(currencyCode, ": ").concat(observations.length));
          return _context13.a(2, null);
        case 5:
          // Tomar hasta 8 observaciones (más que 6 para tener margen)
          // observations ya viene ordenado más reciente primero
          recent = observations.slice(0, 14).reverse(); // más antigua primero para el gráfico
          MONTH_NAMES_ES = {
            '01': 'Ene',
            '02': 'Feb',
            '03': 'Mar',
            '04': 'Abr',
            '05': 'May',
            '06': 'Jun',
            '07': 'Jul',
            '08': 'Ago',
            '09': 'Sep',
            '10': 'Oct',
            '11': 'Nov',
            '12': 'Dic'
          };
          labels = recent.map(function (o) {
            // o.date formato: "2025-09-15" o "2025-09-01"
            var parts = o.date.split('-');
            if (parts.length >= 2) {
              var month = MONTH_NAMES_ES[parts[1]] || parts[1];
              var year = parts[0].slice(2); // últimos 2 dígitos del año
              return "".concat(month, "/").concat(year);
            }
            return o.date;
          });
          rates = recent.map(function (o) {
            return parseFloat(o.value);
          });
          result = {
            labels: labels,
            rates: rates,
            source: 'Bancos Centrales Oficiales + BIS'
          };
          CacheManager.set(cacheKey, result, CACHE_CONFIG.HISTORICAL_DATA);
          console.log("\u2705 Hist\xF3rico real para ".concat(currencyCode, ": ").concat(labels.join(', ')));
          return _context13.a(2, result);
        case 6:
          _context13.p = 6;
          _t12 = _context13.v;
          console.warn("No se pudo cargar hist\xF3rico real para ".concat(currencyCode, ":"), _t12.message);
          return _context13.a(2, null);
      }
    }, _callee13, null, [[1, 6]]);
  }));
  return function fetchRealHistoricalRates(_x12) {
    return _ref13.apply(this, arguments);
  };
}();
var generateHistoricalData = /*#__PURE__*/function () {
  var _ref14 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee14(currentData, setDataLoadingStatus) {
    var cacheKey, cached, historical, i, country, data, realHistory;
    return _regenerator().w(function (_context14) {
      while (1) switch (_context14.n) {
        case 0:
          cacheKey = 'historical_data';
          cached = CacheManager.get(cacheKey);
          if (!(cached !== null)) {
            _context14.n = 1;
            break;
          }
          return _context14.a(2, cached);
        case 1:
          historical = {};
          setDataLoadingStatus({
            status: 'Cargando tasas históricas reales...',
            progress: 90
          });
          i = 0;
        case 2:
          if (!(i < countries.length)) {
            _context14.n = 5;
            break;
          }
          country = countries[i];
          data = currentData[country.code] || {};
          setDataLoadingStatus({
            status: "".concat(country.name, ": hist\xF3rico de tasas..."),
            progress: Math.round(90 + i / countries.length * 8)
          });

          // Intentar obtener histórico REAL desde rates/XX.json (FRED + scraping)
          _context14.n = 3;
          return fetchRealHistoricalRates(country.code);
        case 3:
          realHistory = _context14.v;
          if (realHistory && realHistory.rates && realHistory.rates.length >= 2) {
            // ✅ DATOS REALES DISPONIBLES
            // Solo graficamos la tasa de interés real.
            // El índice de fortaleza histórico se elimina (era simulado).
            historical[country.code] = {
              labels: realHistory.labels,
              rates: realHistory.rates,
              source: 'real'
              // strength se omite intencionalmente — no hay datos históricos reales
              // del índice compuesto. Honestidad sobre lo que tenemos.
            };
            console.log("\u2705 ".concat(country.code, ": hist\xF3rico real (").concat(realHistory.labels.length, " puntos)"));
          } else {
            // ❌ Sin histórico real: marcar como no disponible
            // No generamos datos simulados. El gráfico mostrará mensaje informativo.
            historical[country.code] = {
              labels: [],
              rates: [],
              source: 'unavailable'
            };
            console.warn("\u26A0\uFE0F ".concat(country.code, ": sin hist\xF3rico real disponible"));
          }
        case 4:
          i++;
          _context14.n = 2;
          break;
        case 5:
          setDataLoadingStatus({
            status: 'Datos históricos cargados',
            progress: 98
          });
          CacheManager.set(cacheKey, historical, CACHE_CONFIG.HISTORICAL_DATA);
          return _context14.a(2, historical);
      }
    }, _callee14);
  }));
  return function generateHistoricalData(_x13, _x14) {
    return _ref14.apply(this, arguments);
  };
}();
var generateForexPairRecommendations = function generateForexPairRecommendations(economicData, forexRates) {
  var calendarEvents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : null;
  var cacheKey = 'forex_pair_recommendations';
  var cached = CacheManager.get(cacheKey);
  if (cached !== null) return cached;

  // Calcular fortaleza de todas las divisas
  var strengthScores = {};
  countries.forEach(function (country) {
    var data = economicData[country.code] || {};
    var strengthObj = getStrength(country.code);
    strengthScores[country.code] = {
      score: strengthObj.score,
      dataQuality: strengthObj.dataQuality,
      contributions: strengthObj.contributions
    };
  });

  // Función auxiliar para extraer top contribuciones
  var getTopContributions = function getTopContributions(contributions, isPositive) {
    var labels = {
      'interestRate': 'Tasa de Interés',
      'inflation': 'Inflación',
      'outlookScore': 'Perspectiva BC',
      'rateMomentum': 'Momentum de Tasas',
      'currentAccount': 'Cuenta Corriente',
      'tradeBalance': 'Balanza Comercial',
      'debt': 'Deuda Pública',
      'termsOfTrade': 'Términos de Intercambio',
      'gdpGrowth': 'Crecimiento PIB',
      'unemployment': 'Desempleo',
      'production': 'Producción Industrial',
      'retailSales': 'Ventas al Retail',
      'wageGrowth': 'Crecimiento Salarial',
      'manufacturingPMI': 'PMI Manufactura',
      'servicesPMI': 'PMI Servicios',
      'cotPositioning': 'Posicionamiento COT',
      'consumerConfidence': 'Confianza Consumidor',
      'businessConfidence': 'Confianza Empresarial',
      'economicSurprise': 'Sorpresa Económica',
      'capitalFlows': 'Flujos de Capital',
      'fxPerformance1M': 'Performance FX 1M',
      'bond10y': 'Bono 10 años',
      'safe_haven_score': 'Safe Haven',
      'carry_premium': 'Carry Premium',
      'commodity_exposure': 'Exposición Commodities',
      'debt_sustainability': 'Sostenibilidad Deuda',
      'reserve_currency': 'Prima Divisa Reserva'
    };

    // Para fortalezas: top 3 contribuidores más altos
    // Para debilidades: bottom 3 contribuidores más bajos (los más rezagados vs sus pares)
    var nonZeroEntries = Object.entries(contributions).filter(function (_ref15) {
      var _ref16 = _slicedToArray(_ref15, 2),
        val = _ref16[1];
      return val > 0;
    });
    var entries;
    if (isPositive) {
      entries = nonZeroEntries.sort(function (a, b) {
        return b[1] - a[1];
      }).slice(0, 3);
    } else {
      entries = nonZeroEntries.sort(function (a, b) {
        return a[1] - b[1];
      }).slice(0, 3);
    }
    return entries.map(function (_ref17) {
      var _ref18 = _slicedToArray(_ref17, 2),
        key = _ref18[0],
        val = _ref18[1];
      return {
        factor: labels[key] || key,
        impact: isPositive ? "+".concat(val.toFixed(1)) : val.toFixed(1)
      };
    });
  };

  // Ordenar divisas por fortaleza (guard: null scores → 50 como placeholder)
  var sortedCurrencies = Object.entries(strengthScores).filter(function (_ref19) {
    var _ref20 = _slicedToArray(_ref19, 2),
      data = _ref20[1];
    return data.score !== null && data.score !== undefined;
  }).sort(function (a, b) {
    return b[1].score - a[1].score;
  });
  var recommendations = [];

  // Top 3 más fuertes y 3 más débiles
  var strongCurrencies = sortedCurrencies.slice(0, 3);
  var weakCurrencies = sortedCurrencies.slice(-3).reverse();

  // Generar combinaciones LONG (Comprar fuerte vs débil)
  strongCurrencies.forEach(function (_ref21) {
    var _ref22 = _slicedToArray(_ref21, 2),
      strongCurr = _ref22[0],
      strongData = _ref22[1];
    weakCurrencies.forEach(function (_ref23) {
      var _fxPerformance1M, _fxPerformance1M2;
      var _ref24 = _slicedToArray(_ref23, 2),
        weakCurr = _ref24[0],
        weakData = _ref24[1];
      if (strongCurr === weakCurr) return;
      var spreadStrength = strongData.score - weakData.score;
      var pairName = "".concat(strongCurr, "/").concat(weakCurr);
      var strongReasons = getTopContributions(strongData.contributions, true);
      var weakReasons = getTopContributions(weakData.contributions, false);
      var avgQuality = (strongData.dataQuality + weakData.dataQuality) / 2;
      var confidence = spreadStrength >= 20 ? 'Alta' : 'Baja'; // v6.5 backtest: diff≥20→HR 49.8% Sharpe +0.46, diff<20→HR 41-44% Sharpe negativo

      // Momentum confirmation filter (backtest-derived):
      // If the pair already moved >2% AGAINST the fundamental signal in the last month,
      // flag it — backtest shows these have HR ~29% (worse than random).
      var strongFx = (_fxPerformance1M = (economicData[strongCurr] || {}).fxPerformance1M) !== null && _fxPerformance1M !== void 0 ? _fxPerformance1M : null;
      var weakFx = (_fxPerformance1M2 = (economicData[weakCurr] || {}).fxPerformance1M) !== null && _fxPerformance1M2 !== void 0 ? _fxPerformance1M2 : null;
      var pairMom1M = strongFx !== null && weakFx !== null ? strongFx - weakFx : null;
      var momentumOpposing = pairMom1M !== null && pairMom1M < -2.0;
      // v6.4: JPY como corto tiene HR histórico ~20-40% — se mueve por risk sentiment, no por macro
      var jpyAsShort = weakCurr === 'JPY';
      // MOMENTUM INSTITUCIONAL: par neto 7d basket-adjusted
      // Evaluar el momentum del PAR como unidad (Bloomberg/DailyFX standard):
      // pairMom7d = strongMom7d - weakMom7d
      //   > 0 = precio a favor del LONG | < 0 = divergencia
      // Umbrales: 1 sigma semanal G8 = ~0.6%, umbral conservador = 0.40%
      var strongFx1W = (economicData[strongCurr] || {}).fxPerformance1W;
      var weakFx1W = (economicData[weakCurr] || {}).fxPerformance1W;
      var strongMom7d = strongFx1W != null ? strongFx1W : null;
      var weakMom7d = weakFx1W != null ? weakFx1W : null;
      var pairMom7d = strongMom7d !== null && weakMom7d !== null ? strongMom7d - weakMom7d : null;
      var momConfirmed7d = pairMom7d !== null && pairMom7d > 0.40;
      var momDivergence7d = pairMom7d !== null && pairMom7d < -0.40;
      var momDivergenceStrong = pairMom7d !== null && pairMom7d < -1.0;
      var momAlignment = momConfirmed7d ? 1 : momDivergence7d ? -1 : 0;
      // Para display individual en panel UI
      var strongCorrection7d = strongMom7d !== null && strongMom7d < -0.40;
      var weakRebounding7d = weakMom7d !== null && weakMom7d > 0.40;
      var momPriorityFactor = momDivergenceStrong ? 0.45 : momDivergence7d ? 0.70 : momConfirmed7d ? 1.20 : 1.0;
      recommendations.push({
        type: 'long',
        pair: pairName,
        direction: 'LONG',
        actionText: 'Comprar',
        strength: strongData.score,
        weakness: weakData.score,
        spread: spreadStrength,
        confidence: confidence,
        dataQuality: avgQuality,
        strongCurrency: strongCurr,
        weakCurrency: weakCurr,
        strongReasons: strongReasons,
        weakReasons: weakReasons,
        pairMom1M: pairMom1M,
        momentumOpposing: momentumOpposing,
        jpyAsShort: jpyAsShort,
        strongMom7d: strongMom7d,
        weakMom7d: weakMom7d,
        strongCorrection7d: strongCorrection7d,
        weakRebounding7d: weakRebounding7d,
        momConfirmed7d: momConfirmed7d,
        momDivergence7d: momDivergence7d,
        pairMom7d: pairMom7d,
        momAlignment: momAlignment,
        priority: spreadStrength * avgQuality * (momentumOpposing ? 0.5 : 1.0) * (jpyAsShort ? 0.6 : 1.0) * momPriorityFactor
      });
    });
  });

  // Generar combinaciones SHORT (Vender débil vs fuerte)
  weakCurrencies.forEach(function (_ref25) {
    var _ref26 = _slicedToArray(_ref25, 2),
      weakCurr = _ref26[0],
      weakData = _ref26[1];
    strongCurrencies.forEach(function (_ref27) {
      var _fxPerformance1M3, _fxPerformance1M4;
      var _ref28 = _slicedToArray(_ref27, 2),
        strongCurr = _ref28[0],
        strongData = _ref28[1];
      if (weakCurr === strongCurr) return;
      var spreadStrength = strongData.score - weakData.score;
      var pairName = "".concat(weakCurr, "/").concat(strongCurr);
      var weakReasons = getTopContributions(weakData.contributions, false);
      var strongReasons = getTopContributions(strongData.contributions, true);
      var avgQuality = (strongData.dataQuality + weakData.dataQuality) / 2;
      var confidence = spreadStrength >= 20 ? 'Alta' : 'Baja'; // v6.5 backtest: diff≥20→HR 49.8% Sharpe +0.46, diff<20→HR 41-44% Sharpe negativo

      // Momentum confirmation filter for SHORT (par = WEAK/STRONG, ej: GBP/AUD).
      // El par cae cuando el strong sube más que el weak → CONFIRMA el SHORT.
      // Adverso = el par SUBE (strong cayó vs weak), es decir strongFx - weakFx < -2%.
      // FIX v6.5: fórmula anterior (weakFx - strongFx) estaba invertida y generaba
      // falsos positivos en señales confirmadas por precio (ej: GBP/AUD SHORT).
      var strongFxS = (_fxPerformance1M3 = (economicData[strongCurr] || {}).fxPerformance1M) !== null && _fxPerformance1M3 !== void 0 ? _fxPerformance1M3 : null;
      var weakFxS = (_fxPerformance1M4 = (economicData[weakCurr] || {}).fxPerformance1M) !== null && _fxPerformance1M4 !== void 0 ? _fxPerformance1M4 : null;
      var pairMom1MS = strongFxS !== null && weakFxS !== null ? strongFxS - weakFxS : null;
      var momentumOpposingS = pairMom1MS !== null && pairMom1MS < -2.0;
      // v6.4: JPY como corto tiene HR histórico ~20-40% — se mueve por risk sentiment, no por macro
      var jpyAsShortS = weakCurr === 'JPY';
      // MOMENTUM INSTITUCIONAL SHORT: par neto 7d basket-adjusted
      // Para SHORT (par = WEAK/STRONG), el par CAE cuando strong supera a weak.
      // pairMom7dS = strongMom7dS - weakMom7dS
      //   > 0 = precio confirma el SHORT (strong sube vs weak)
      //   < 0 = divergencia (par subió, se mueve contra el SHORT)
      var strongFx1WS = (economicData[strongCurr] || {}).fxPerformance1W;
      var weakFx1WS = (economicData[weakCurr] || {}).fxPerformance1W;
      var strongMom7dS = strongFx1WS != null ? strongFx1WS : null;
      var weakMom7dS = weakFx1WS != null ? weakFx1WS : null;
      var pairMom7dS = strongMom7dS !== null && weakMom7dS !== null ? strongMom7dS - weakMom7dS : null;
      var momConfirmed7dS = pairMom7dS !== null && pairMom7dS > 0.40;
      var momDivergence7dS = pairMom7dS !== null && pairMom7dS < -0.40;
      var momDivergenceStrongS = pairMom7dS !== null && pairMom7dS < -1.0;
      var momAlignmentS = momConfirmed7dS ? 1 : momDivergence7dS ? -1 : 0;
      var strongCorrection7dS = strongMom7dS !== null && strongMom7dS < -0.40;
      var weakRebounding7dS = weakMom7dS !== null && weakMom7dS > 0.40;
      var momPriorityFactorS = momDivergenceStrongS ? 0.45 : momDivergence7dS ? 0.70 : momConfirmed7dS ? 1.20 : 1.0;
      recommendations.push({
        type: 'short',
        pair: pairName,
        direction: 'SHORT',
        actionText: 'Vender',
        strength: strongData.score,
        weakness: weakData.score,
        spread: spreadStrength,
        confidence: confidence,
        dataQuality: avgQuality,
        strongCurrency: strongCurr,
        weakCurrency: weakCurr,
        weakReasons: weakReasons,
        strongReasons: strongReasons,
        pairMom1M: pairMom1MS,
        momentumOpposing: momentumOpposingS,
        jpyAsShort: jpyAsShortS,
        strongMom7d: strongMom7dS,
        weakMom7d: weakMom7dS,
        strongCorrection7d: strongCorrection7dS,
        weakRebounding7d: weakRebounding7dS,
        momConfirmed7d: momConfirmed7dS,
        momDivergence7d: momDivergence7dS,
        pairMom7d: pairMom7dS,
        momAlignment: momAlignmentS,
        priority: spreadStrength * avgQuality * (momentumOpposingS ? 0.5 : 1.0) * (jpyAsShortS ? 0.6 : 1.0) * momPriorityFactorS
      });
    });
  });

  // Ordenar por prioridad y tomar top 8
  recommendations.sort(function (a, b) {
    return b.priority - a.priority;
  });

  // v6.4: solo mostrar Media y Alta — Baja tiene Sharpe ~0.35 y HR ~47% según backtest
  var longRecs = recommendations.filter(function (r) {
    return r.type === 'long' && r.confidence === 'Alta';
  }).slice(0, 4); // v6.5: solo Alta (diff≥20), backtest HR 49.8% Sharpe +0.46
  var shortRecs = recommendations.filter(function (r) {
    return r.type === 'short' && r.confidence === 'Alta';
  }).slice(0, 4); // v6.5: solo Alta (diff≥20)

  var finalRecommendations = [].concat(_toConsumableArray(longRecs), _toConsumableArray(shortRecs));
  CacheManager.set(cacheKey, finalRecommendations, CACHE_CONFIG.ALERTS);
  return finalRecommendations;
};
var loadAllEconomicData = /*#__PURE__*/function () {
  var _ref29 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee18(setEconomicData, setDataLoadingStatus) {
    var _economicDataToCache$;
    var cacheKey, DASHBOARD_VERSION, cachedVersion, cached, sampleCurrencies, hasFxPerf, economicData, loadCountryData, countryResults, completedCount, _iterator, _step, result, _code, data, outlookResults, _iterator2, _step2, _step2$value, country, outlook, nextMeeting, fxRawData, fxResults, _iterator3, _step3, entry, _code2, rest, raw1MValues, totalRaw1M, n, fx1WValues, total1W, n1W, fx3MValues, total3M, n3M, _i, _Object$entries, _d$fx3M, _d$raw1M, _basket1W, _Object$entries$_i, code, d, basket1M, basket1W, basket3M, composite, economicDataToCache;
    return _regenerator().w(function (_context18) {
      while (1) switch (_context18.n) {
        case 0:
          cacheKey = 'all_economic_data'; // ⚠️ IMPORTANTE: Actualizar DASHBOARD_VERSION cada vez que modifiques el código
          // El formato es 'vX.Y.Z-YYYY-MM-DD' — la fecha garantiza invalidación automática del caché
          DASHBOARD_VERSION = '6.7.0-2026-03-27'; // bump: ALERTS TTL 1h→5min, fetchExtendedData 5 reintentos/13s, invalidación caché scores, CSP fix
          // ✅ Verificar versión del caché
          cachedVersion = localStorage.getItem('forex_dashboard_version');
          if (cachedVersion !== DASHBOARD_VERSION) {
            console.log('🔄 Nueva versión detectada, limpiando caché...');
            CacheManager.clearAll();
            localStorage.setItem('forex_dashboard_version', DASHBOARD_VERSION);
          }
          cached = CacheManager.get(cacheKey);
          if (!(cached !== null)) {
            _context18.n = 2;
            break;
          }
          // v6.6 cache validation: reject cache missing fxPerformance1M or fxPerformance1W
          sampleCurrencies = ['AUD', 'CHF', 'USD', 'EUR'];
          hasFxPerf = sampleCurrencies.some(function (c) {
            return cached[c] && cached[c].fxPerformance1M !== null && cached[c].fxPerformance1M !== undefined && cached[c].fxPerformance1W !== undefined;
          });
          if (hasFxPerf) {
            _context18.n = 1;
            break;
          }
          console.log('\u26A0\uFE0F Cach\xE9 sin fxPerformance1W \u2014 invalidando para cargar datos completos...');
          CacheManager.clear(cacheKey);
          _context18.n = 2;
          break;
        case 1:
          setEconomicData(cached);
          setDataLoadingStatus({
            status: 'Datos cargados desde caché',
            progress: 100
          });
          return _context18.a(2, cached);
        case 2:
          economicData = {}; // ── FIX #2: Fetches paralelos ────────────────────────────────────────────
          // Antes: for-loop con await por cada país → 8 países × 3 fetches = ~24 req en serie
          // Ahora: Promise.all → todos los países se cargan simultáneamente
          // Reducción estimada de carga: ~15s → ~2s en conexión fría
          // ────────────────────────────────────────────────────────────────────────
          setDataLoadingStatus({
            status: 'Iniciando carga paralela de datos...',
            progress: 5
          });

          // Helper: carga los 3 archivos de un país en paralelo entre sí
          loadCountryData = /*#__PURE__*/function () {
            var _ref30 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee15(country) {
              var result, _yield$Promise$all, _yield$Promise$all2, econResponse, cotData, extData, interestRateResult, data, dates, lastUpd, _d$bond10y, _d$consumerConfidence, _d$businessConfidence, _d$capitalFlows, _d$fdi, _d$inflationExpectati, _d$rateMomentum, _d$rateMomentum24M, d, dt, _lastUpd;
              return _regenerator().w(function (_context15) {
                while (1) switch (_context15.n) {
                  case 0:
                    result = {
                      code: country.code
                    }; // Los 3 fetches del país corren simultáneamente
                    _context15.n = 1;
                    return Promise.all([
                    // 1) economic-data
                    fetchWithRetry("https://globalinvesting.github.io/economic-data/".concat(country.code, ".json"), {
                      cache: 'no-cache',
                      mode: 'cors'
                    }, 3).then(function (r) {
                      return r.ok ? r.json() : null;
                    }).catch(function () {
                      return null;
                    }),
                    // 2) cot-data
                    fetchWithRetry("https://globalinvesting.github.io/cot-data/".concat(country.code, ".json"), {
                      cache: 'no-cache',
                      mode: 'cors'
                    }, 2).then(function (r) {
                      return r.ok ? r.json() : null;
                    }).catch(function () {
                      return null;
                    }),
                    // 3) extended-data
                    fetchExtendedData(country.code).catch(function () {
                      return null;
                    }),
                    // 4) interest rate (usa rates/ endpoint separado)
                    fetchInterestRate(country).catch(function () {
                      return null;
                    })]);
                  case 1:
                    _yield$Promise$all = _context15.v;
                    _yield$Promise$all2 = _slicedToArray(_yield$Promise$all, 4);
                    econResponse = _yield$Promise$all2[0];
                    cotData = _yield$Promise$all2[1];
                    extData = _yield$Promise$all2[2];
                    interestRateResult = _yield$Promise$all2[3];
                    // — economic-data —
                    if (econResponse && econResponse.data) {
                      data = econResponse.data;
                      dates = econResponse.dates || {};
                      lastUpd = econResponse.lastUpdate || new Date().toISOString();
                      result.gdp = data.gdp ? data.gdp / 1e12 : null;
                      result.gdpDate = dates.gdp || lastUpd;
                      result.gdpGrowth = data.gdpGrowth || null;
                      result.gdpGrowthDate = dates.gdpGrowth || lastUpd;
                      result.inflation = data.inflation || null;
                      result.inflationDate = dates.inflation || lastUpd;
                      result.unemployment = data.unemployment || null;
                      result.unemploymentDate = dates.unemployment || lastUpd;
                      result.currentAccount = data.currentAccount || null;
                      result.currentAccountDate = dates.currentAccount || lastUpd;
                      result.debt = data.debt || null;
                      result.debtDate = dates.debt || lastUpd;
                      result.tradeBalance = data.tradeBalance || null;
                      result.tradeBalanceDate = dates.tradeBalance || lastUpd;
                      result.production = data.production || null;
                      result.productionDate = dates.production || lastUpd;
                      result.lastUpdate = lastUpd;
                      result.retailSales = data.retailSales !== undefined ? data.retailSales : null;
                      result.retailSalesDate = dates.retailSales || lastUpd;
                      result.wageGrowth = data.wageGrowth !== undefined ? data.wageGrowth : null;
                      result.wageGrowthDate = dates.wageGrowth || lastUpd;
                      result.manufacturingPMI = data.manufacturingPMI !== undefined ? data.manufacturingPMI : null;
                      result.manufacturingPMIDate = dates.manufacturingPMI || lastUpd;
                      result.servicesPMI = data.servicesPMI !== undefined ? data.servicesPMI : null;
                      result.servicesPMIDate = dates.servicesPMI || lastUpd;
                      result.termsOfTrade = data.termsOfTrade !== undefined ? data.termsOfTrade : null;
                      result.termsOfTradeDate = dates.termsOfTrade || lastUpd;
                      result.cotPositioning = data.cotPositioning !== undefined ? data.cotPositioning : null;
                      console.log("\u2705 Loaded economic-data for ".concat(country.code));
                    } else {
                      console.error("\u274C No economic-data for ".concat(country.code));
                    }

                    // — interest rate —
                    if (interestRateResult && _typeof(interestRateResult) === 'object') {
                      result.interestRate = interestRateResult.rate;
                      result.interestRateDate = interestRateResult.date || result.lastUpdate;
                    } else if (typeof interestRateResult === 'number') {
                      result.interestRate = interestRateResult;
                      result.interestRateDate = result.lastUpdate;
                    } else {
                      result.interestRate = null;
                      result.interestRateDate = result.lastUpdate;
                    }

                    // — cot-data —
                    if (cotData && cotData.netPosition !== null && cotData.netPosition !== undefined) {
                      result.cotPositioning = cotData.netPosition;
                      result.cotPositioningDate = cotData.reportDate || cotData.lastUpdate || result.lastUpdate || new Date().toISOString();
                      // No sobreescribir lastUpdate con fecha COT — el archivo económico es la referencia
                      console.log("\u2705 Loaded COT for ".concat(country.code, ": ").concat(cotData.positioning));
                    }

                    // — extended-data (bonds, sentiment, flows) —
                    if (extData && extData.data) {
                      d = extData.data;
                      dt = extData.dates || {};
                      _lastUpd = extData.lastUpdate;
                      result.bond10y = (_d$bond10y = d.bond10y) !== null && _d$bond10y !== void 0 ? _d$bond10y : null;
                      result.bond10yDate = dt.bond10y || _lastUpd;
                      result.consumerConfidence = (_d$consumerConfidence = d.consumerConfidence) !== null && _d$consumerConfidence !== void 0 ? _d$consumerConfidence : null;
                      result.consumerConfidenceDate = dt.consumerConfidence || _lastUpd;
                      result.businessConfidence = (_d$businessConfidence = d.businessConfidence) !== null && _d$businessConfidence !== void 0 ? _d$businessConfidence : null;
                      result.businessConfidenceDate = dt.businessConfidence || _lastUpd;
                      result.capitalFlows = (_d$capitalFlows = d.capitalFlows) !== null && _d$capitalFlows !== void 0 ? _d$capitalFlows : null;
                      result.capitalFlowsDate = dt.capitalFlows || _lastUpd;
                      result.fdi = (_d$fdi = d.fdi) !== null && _d$fdi !== void 0 ? _d$fdi : null;
                      result.fdiDate = dt.fdi || _lastUpd;
                      result.inflationExpectations = (_d$inflationExpectati = d.inflationExpectations) !== null && _d$inflationExpectati !== void 0 ? _d$inflationExpectati : null;
                      result.inflationExpectationsDate = dt.inflationExpectations || _lastUpd;
                      result.rateMomentum = (_d$rateMomentum = d.rateMomentum) !== null && _d$rateMomentum !== void 0 ? _d$rateMomentum : null;
                      result.rateMomentum24M = (_d$rateMomentum24M = d.rateMomentum24M) !== null && _d$rateMomentum24M !== void 0 ? _d$rateMomentum24M : null;
                      result.rateMomentumDate = dt.rateMomentum || _lastUpd;
                      console.log("\u2705 Extended data loaded for ".concat(country.code));
                    }
                    return _context15.a(2, result);
                }
              }, _callee15);
            }));
            return function loadCountryData(_x17) {
              return _ref30.apply(this, arguments);
            };
          }(); // Lanzar todos los países en paralelo simultáneamente
          setDataLoadingStatus({
            status: 'Cargando datos de todos los países en paralelo...',
            progress: 10
          });
          _context18.n = 3;
          return Promise.all(countries.map(function (country) {
            return loadCountryData(country);
          }));
        case 3:
          countryResults = _context18.v;
          // Consolidar resultados en economicData
          completedCount = 0;
          _iterator = _createForOfIteratorHelper(countryResults);
          try {
            for (_iterator.s(); !(_step = _iterator.n()).done;) {
              result = _step.value;
              _code = result.code, data = _objectWithoutProperties(result, _excluded);
              economicData[_code] = data;
              completedCount++;
              setDataLoadingStatus({
                status: "Procesando ".concat(_code, "... (").concat(completedCount, "/").concat(countries.length, ")"),
                progress: Math.round(10 + completedCount / countries.length * 75)
              });
            }

            // ── Outlook + meetings: también en paralelo ──────────────────────────────
          } catch (err) {
            _iterator.e(err);
          } finally {
            _iterator.f();
          }
          setDataLoadingStatus({
            status: 'Procesando outlook de bancos centrales...',
            progress: 88
          });
          _context18.n = 4;
          return Promise.all(countries.map(/*#__PURE__*/function () {
            var _ref31 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee16(country) {
              var _yield$Promise$all3, _yield$Promise$all4, outlook, nextMeeting;
              return _regenerator().w(function (_context16) {
                while (1) switch (_context16.n) {
                  case 0:
                    _context16.n = 1;
                    return Promise.all([determineCentralBankOutlook(country.code, economicData[country.code]), fetchCentralBankMeetings(country.code)]);
                  case 1:
                    _yield$Promise$all3 = _context16.v;
                    _yield$Promise$all4 = _slicedToArray(_yield$Promise$all3, 2);
                    outlook = _yield$Promise$all4[0];
                    nextMeeting = _yield$Promise$all4[1];
                    return _context16.a(2, {
                      country: country,
                      outlook: outlook,
                      nextMeeting: nextMeeting
                    });
                }
              }, _callee16);
            }));
            return function (_x18) {
              return _ref31.apply(this, arguments);
            };
          }()));
        case 4:
          outlookResults = _context18.v;
          _iterator2 = _createForOfIteratorHelper(outlookResults);
          try {
            for (_iterator2.s(); !(_step2 = _iterator2.n()).done;) {
              _step2$value = _step2.value, country = _step2$value.country, outlook = _step2$value.outlook, nextMeeting = _step2$value.nextMeeting;
              // Fuente de verdad: bcOutlook del backend (latest.json).
              // El backend tiene acceso a rm_trough (rebote desde mínimo 6M), a toda
              // la historia de rates/*.json y a la lógica v6.6.1 completa.
              // determineCentralBankOutlook() es solo fallback cuando scores no están disponibles.
              var _scoredOutlook = getStrength(country.code).bcOutlook;
              var _finalOutlook = _scoredOutlook || outlook;
              country.outlook = _finalOutlook;
              country.nextMeeting = nextMeeting;
              economicData[country.code].outlook = _finalOutlook;
              economicData[country.code].nextMeeting = nextMeeting;
              if (_scoredOutlook && _scoredOutlook !== outlook) {
                console.log("\u2705 outlook override for " + country.code + ": " + outlook + " \u2192 " + _scoredOutlook + " (backend v6.6.1)");
              }
            }

            // ✅ FX Performance ANTES de guardar caché para que quede persistido
          } catch (err) {
            _iterator2.e(err);
          } finally {
            _iterator2.f();
          }
          setDataLoadingStatus({
            status: 'Cargando FX Performance 1M...',
            progress: 94
          });

          // Paso 1: cargar todos los datos raw de FX en paralelo
          fxRawData = {};
          _context18.n = 5;
          return Promise.all(countries.map(/*#__PURE__*/function () {
            var _ref32 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee17(country) {
              var fxPerfUrl, fxPerfResponse, d, _d$fxPerformance1M_ra, _d$fxPerformance1W, _d$fxPerformance3M, _t13;
              return _regenerator().w(function (_context17) {
                while (1) switch (_context17.p = _context17.n) {
                  case 0:
                    _context17.p = 0;
                    fxPerfUrl = "https://globalinvesting.github.io/fx-performance/".concat(country.code, ".json");
                    _context17.n = 1;
                    return fetchWithRetry(fxPerfUrl, {
                      cache: 'no-cache',
                      mode: 'cors'
                    }, 2);
                  case 1:
                    fxPerfResponse = _context17.v;
                    if (!fxPerfResponse.ok) {
                      _context17.n = 3;
                      break;
                    }
                    _context17.n = 2;
                    return fxPerfResponse.json();
                  case 2:
                    d = _context17.v;
                    if (!(d.fxPerformance1M !== null && d.fxPerformance1M !== undefined)) {
                      _context17.n = 3;
                      break;
                    }
                    return _context17.a(2, {
                      code: country.code,
                      raw1M: (_d$fxPerformance1M_ra = d.fxPerformance1M_raw) !== null && _d$fxPerformance1M_ra !== void 0 ? _d$fxPerformance1M_ra : d.fxPerformance1M,
                      fx1W: (_d$fxPerformance1W = d.fxPerformance1W) !== null && _d$fxPerformance1W !== void 0 ? _d$fxPerformance1W : null,
                      fx3M: (_d$fxPerformance3M = d.fxPerformance3M) !== null && _d$fxPerformance3M !== void 0 ? _d$fxPerformance3M : null,
                      date: d.date || new Date().toISOString()
                    });
                  case 3:
                    _context17.n = 5;
                    break;
                  case 4:
                    _context17.p = 4;
                    _t13 = _context17.v;
                    console.warn("No FX performance data for ".concat(country.code));
                  case 5:
                    return _context17.a(2, null);
                }
              }, _callee17, null, [[0, 4]]);
            }));
            return function (_x19) {
              return _ref32.apply(this, arguments);
            };
          }()));
        case 5:
          fxResults = _context18.v;
          _iterator3 = _createForOfIteratorHelper(fxResults);
          try {
            for (_iterator3.s(); !(_step3 = _iterator3.n()).done;) {
              entry = _step3.value;
              if (entry) {
                _code2 = entry.code, rest = _objectWithoutProperties(entry, _excluded2);
                fxRawData[_code2] = rest;
              }
            }

            // Paso 2: corrección de sesgo USD — convertir rendimientos vs-USD a rendimientos vs-basket G8
            // Método: basket_perf_X = raw1M_X − mean(raw1M_Y para Y ≠ X)
            // Equivale a lo que hacen los índices de divisas (DXY, CXY, AXY, etc.)
            // Sin esta corrección, cuando USD es fuerte todo lo demás parece débil vs USD
            // aunque en realidad sólo está débil RELATIVO al USD, no al G8 completo.
            // v5.9: corrección aplicada a 1M_raw, 1W y 3M antes de componer el composite.
          } catch (err) {
            _iterator3.e(err);
          } finally {
            _iterator3.f();
          }
          raw1MValues = Object.values(fxRawData).map(function (d) {
            return d.raw1M;
          }).filter(function (v) {
            return v !== null;
          });
          totalRaw1M = raw1MValues.reduce(function (a, b) {
            return a + b;
          }, 0);
          n = raw1MValues.length;
          fx1WValues = Object.values(fxRawData).map(function (d) {
            return d.fx1W;
          }).filter(function (v) {
            return v !== null;
          });
          total1W = fx1WValues.reduce(function (a, b) {
            return a + b;
          }, 0);
          n1W = fx1WValues.length;
          fx3MValues = Object.values(fxRawData).map(function (d) {
            return d.fx3M;
          }).filter(function (v) {
            return v !== null;
          });
          total3M = fx3MValues.reduce(function (a, b) {
            return a + b;
          }, 0);
          n3M = fx3MValues.length;
          for (_i = 0, _Object$entries = Object.entries(fxRawData); _i < _Object$entries.length; _i++) {
            _Object$entries$_i = _slicedToArray(_Object$entries[_i], 2), code = _Object$entries$_i[0], d = _Object$entries$_i[1];
            // Basket-corrected values (subtract mean of others)
            basket1M = n > 1 ? d.raw1M - (totalRaw1M - d.raw1M) / (n - 1) : d.raw1M;
            basket1W = null;
            if (d.fx1W !== null && n1W > 1) {
              basket1W = d.fx1W - (total1W - d.fx1W) / (n1W - 1);
            }
            basket3M = n3M > 1 && d.fx3M !== null ? d.fx3M - (total3M - d.fx3M) / (n3M - 1) : (_d$fx3M = d.fx3M) !== null && _d$fx3M !== void 0 ? _d$fx3M : basket1M; // Composite: 70% basket_1M + 20% basket_1W + 10% basket_3M
            composite = void 0;
            if (basket1W !== null) {
              composite = 0.70 * basket1M + 0.20 * basket1W + 0.10 * basket3M;
            } else {
              composite = 0.90 * basket1M + 0.10 * basket3M;
            }
            economicData[code].fxPerformance1M = parseFloat(composite.toFixed(4));
            economicData[code].fxPerformance1MDate = d.date;
            economicData[code].fxPerformance1W = basket1W !== null ? parseFloat(basket1W.toFixed(4)) : null;
            console.log("\u2705 FX perf basket-adj for ".concat(code, ": ").concat(composite.toFixed(4), "% (raw=").concat((_d$raw1M = d.raw1M) === null || _d$raw1M === void 0 ? void 0 : _d$raw1M.toFixed(3), ", basket1M=").concat(basket1M.toFixed(3), ", 1W=").concat((_basket1W = basket1W) === null || _basket1W === void 0 ? void 0 : _basket1W.toFixed(3), ", 3M=").concat(basket3M === null || basket3M === void 0 ? void 0 : basket3M.toFixed(3), ")"));
          }
          setDataLoadingStatus({
            status: 'Datos cargados exitosamente',
            progress: 100
          });
          // Deep copy para garantizar que fxPerformance1M esté incluido en el caché
          economicDataToCache = JSON.parse(JSON.stringify(economicData));
          console.log('🔍 Pre-cache check AUD fxPerf1M:', (_economicDataToCache$ = economicDataToCache.AUD) === null || _economicDataToCache$ === void 0 ? void 0 : _economicDataToCache$.fxPerformance1M);
          CacheManager.set(cacheKey, economicDataToCache, CACHE_CONFIG.ECONOMIC_DATA);
          setEconomicData(economicData);
          return _context18.a(2, economicData);
      }
    }, _callee18);
  }));
  return function loadAllEconomicData(_x15, _x16) {
    return _ref29.apply(this, arguments);
  };
}();
var clearDashboardCache = function clearDashboardCache() {
  CacheManager.clearAll();
  alert('Caché limpiado. Recarga la página para obtener datos actualizados.');
};

// ========== DATA HEALTH CHECK SYSTEM ==========
var checkDataHealth = function checkDataHealth(economicData) {
  var issues = [];
  var warnings = [];
  Object.keys(economicData).forEach(function (curr) {
    var data = economicData[curr];

    // Check 1: Data age
    if (data.lastUpdate) {
      var age = Date.now() - new Date(data.lastUpdate).getTime();
      var daysOld = Math.floor(age / (24 * 60 * 60 * 1000));
      if (daysOld > 7) {
        issues.push("".concat(curr, ": Datos con ").concat(daysOld, " d\xEDas de antig\xFCedad"));
      } else if (daysOld > 3) {
        warnings.push("".concat(curr, ": Datos con ").concat(daysOld, " d\xEDas (considerar actualizaci\xF3n)"));
      }
    } else {
      issues.push("".concat(curr, ": Sin timestamp de actualizaci\xF3n"));
    }

    // Check 2: Missing critical indicators
    var critical = ['interestRate', 'inflation', 'gdpGrowth', 'unemployment'];
    var missing = critical.filter(function (ind) {
      return data[ind] === null || data[ind] === undefined;
    });
    if (missing.length > 0) {
      issues.push("".concat(curr, ": Faltan indicadores cr\xEDticos: ").concat(missing.join(', ')));
    }

    // Check 3: Out of range values
    if (data.inflation !== null && data.inflation !== undefined) {
      if (data.inflation < -5 || data.inflation > 25) {
        issues.push("".concat(curr, ": Inflaci\xF3n fuera de rango (").concat(data.inflation, "%)"));
      }
    }
    if (data.unemployment !== null && data.unemployment !== undefined) {
      if (data.unemployment < 0 || data.unemployment > 30) {
        issues.push("".concat(curr, ": Desempleo fuera de rango (").concat(data.unemployment, "%)"));
      }
    }
    if (data.interestRate !== null && data.interestRate !== undefined) {
      if (data.interestRate < -2 || data.interestRate > 20) {
        issues.push("".concat(curr, ": Tasa de inter\xE9s fuera de rango (").concat(data.interestRate, "%)"));
      }
    }

    // Check 4: Data completeness score — umbrales diferenciados por criticidad
    var criticalIndicators = ['gdpGrowth', 'interestRate', 'inflation', 'unemployment'];
    var importantIndicators = ['currentAccount', 'debt', 'tradeBalance', 'production', 'retailSales', 'wageGrowth', 'manufacturingPMI', 'servicesPMI', 'economicSurprise', 'cotPositioning'];
    var supplementaryIndicators = ['bond10y', 'consumerConfidence', 'businessConfidence', 'inflationExpectations', 'termsOfTrade'];
    var allIndicators = [].concat(criticalIndicators, importantIndicators, supplementaryIndicators);
    var missingCritical = criticalIndicators.filter(function (ind) {
      return data[ind] === null || data[ind] === undefined;
    });
    var availableAll = allIndicators.filter(function (ind) {
      return data[ind] !== null && data[ind] !== undefined;
    }).length;
    var completeness = Math.round(availableAll / allIndicators.length * 100);

    // Cualquier indicador crítico faltante es un issue, no solo un warning
    if (missingCritical.length > 0) {
      issues.push("".concat(curr, ": Indicadores cr\xEDticos sin dato: ").concat(missingCritical.join(', ')));
    }
    if (completeness < 45) {
      issues.push("".concat(curr, ": Completitud de datos muy baja (").concat(completeness, "%) \u2014 scoring poco fiable"));
    } else if (completeness < 65) {
      warnings.push("".concat(curr, ": Completitud de datos moderada (").concat(completeness, "%) \u2014 intervalo de confianza ampliado"));
    }

    // Check 4b: Mostrar completitud en card cuando es baja (para todos los casos)
    if (completeness < 75) {
      warnings.push("".concat(curr, ": ").concat(availableAll, "/").concat(allIndicators.length, " indicadores disponibles"));
    }
  });
  return {
    healthy: issues.length === 0,
    issues: issues,
    warnings: warnings,
    timestamp: new Date().toISOString()
  };
};
var getOldestDataTimestamp = function getOldestDataTimestamp(economicData) {
  var oldest = null;
  Object.values(economicData).forEach(function (data) {
    if (data.lastUpdate) {
      var date = new Date(data.lastUpdate);
      // Ignorar timestamps muy antiguos (>3 días) que corresponden
      // a datos semanales/mensuales como COT o meetings
      var ageDays = (Date.now() - date.getTime()) / (1000 * 60 * 60 * 24);
      if (ageDays <= 3) {
        if (!oldest || date < oldest) {
          oldest = date;
        }
      }
    }
  });
  return oldest;
};
var getCacheAge = function getCacheAge(economicData) {
  var oldest = getOldestDataTimestamp(economicData);
  if (!oldest) return 'Desconocido';
  var ageMs = Date.now() - oldest.getTime();
  var hours = Math.floor(ageMs / (60 * 60 * 1000));
  var days = Math.floor(hours / 24);
  if (days > 0) return "".concat(days, "d ").concat(hours % 24, "h");
  if (hours > 0) return "".concat(hours, "h");
  return 'Reciente';
};
var isCacheStale = function isCacheStale(economicData) {
  var oldest = getOldestDataTimestamp(economicData);
  if (!oldest) return true;
  var ageMs = Date.now() - oldest.getTime();
  var hours = Math.floor(ageMs / (60 * 60 * 1000));
  return hours > 48; // Stale if older than 48 hours
};
if (typeof window !== 'undefined') {
  window.clearForexCache = clearDashboardCache;
}
var countries = [{
  code: 'USD',
  name: 'Estados Unidos',
  currency: 'Dólar Estadounidense',
  flag: '🇺🇸',
  centralBank: 'Reserva Federal',
  nextMeeting: 'Por definir',
  outlook: 'Calculando...'
}, {
  code: 'EUR',
  name: 'Eurozona',
  currency: 'Euro',
  flag: '🇪🇺',
  centralBank: 'Banco Central Europeo',
  nextMeeting: 'Por definir',
  outlook: 'Calculando...'
}, {
  code: 'GBP',
  name: 'Reino Unido',
  currency: 'Libra Esterlina',
  flag: '🇬🇧',
  centralBank: 'Banco de Inglaterra',
  nextMeeting: 'Por definir',
  outlook: 'Calculando...'
}, {
  code: 'JPY',
  name: 'Japón',
  currency: 'Yen Japonés',
  flag: '🇯🇵',
  centralBank: 'Banco de Japón',
  nextMeeting: 'Por definir',
  outlook: 'Calculando...'
}, {
  code: 'AUD',
  name: 'Australia',
  currency: 'Dólar Australiano',
  flag: '🇦🇺',
  centralBank: 'Banco de la Reserva de Australia',
  nextMeeting: 'Por definir',
  outlook: 'Calculando...'
}, {
  code: 'CAD',
  name: 'Canadá',
  currency: 'Dólar Canadiense',
  flag: '🇨🇦',
  centralBank: 'Banco de Canadá',
  nextMeeting: 'Por definir',
  outlook: 'Calculando...'
}, {
  code: 'CHF',
  name: 'Suiza',
  currency: 'Franco Suizo',
  flag: '🇨🇭',
  centralBank: 'Banco Nacional Suizo',
  nextMeeting: 'Por definir',
  outlook: 'Calculando...'
}, {
  code: 'NZD',
  name: 'Nueva Zelanda',
  currency: 'Dólar Neozelandés',
  flag: '🇳🇿',
  centralBank: 'Banco de la Reserva de Nueva Zelanda',
  nextMeeting: 'Por definir',
  outlook: 'Calculando...'
}];
var indicatorTooltips = {
  gdp: "Producto Interno Bruto total del país expresado en trillones de dólares estadounidenses. Representa el valor de todos los bienes y servicios producidos en un año. Es una medida del tamaño absoluto de la economía — no de su dinamismo. Una economía grande (USA ~28T) tiene mayor influencia global en flujos de capital y confianza en su moneda. Fuente: FRED / Banco Mundial.",
  gdpGrowth: "Tasa de variación anual del PIB real, ajustada por inflación. Mide la velocidad a la que crece (o se contrae) la economía. >2.5% = expansión robusta, señal positiva para la divisa ya que atrae inversión y presiona al banco central a mantener o subir tasas. 1–2.5% = crecimiento moderado, neutral. <1% = estancamiento, presión dovish sobre el BC. Negativo = recesión técnica, señal muy negativa para la divisa. Fuente: FRED / OECD.",
  interestRate: "Tasa de política monetaria fijada por el banco central del país. Es el indicador más importante para el forex: determina el 'carry' de la divisa, es decir, el rendimiento que obtiene un inversor por mantener activos en esa moneda. Tasas altas atraen capital extranjero → demanda de la divisa → apreciación. Tasas bajas reducen el atractivo relativo → presión bajista. El diferencial de tasas entre dos países es el principal driver de los pares de divisas en el corto-medio plazo. Fuente: bancos centrales oficiales (NY Fed, BCE, BoE, RBA, BoC, BoJ, SNB, RBNZ) vía APIs directas + BIS Data Portal.",
  inflation: "Variación anual del Índice de Precios al Consumidor (IPC). Mide cuánto suben los precios para los hogares. El objetivo universal de la mayoría de bancos centrales es ~2%. Por encima del objetivo → BC sube tasas (hawkish) → positivo para la divisa a corto plazo, pero erosiona poder adquisitivo. Muy por encima (>4%) → señal de economía recalentada o crisis de oferta → negativo estructural. Por debajo (deflación) → BC baja tasas (dovish) → negativo para la divisa. La inflación es el mandato principal de los bancos centrales modernos. Fuente: FRED / OECD.",
  unemployment: "Porcentaje de la fuerza laboral activa que busca empleo y no lo encuentra. Es el segundo mandato de muchos bancos centrales (junto a inflación). <3.5% = pleno empleo, economía sobrecalentada, BC bajo presión para subir tasas → hawkish. 3.5–6% = zona de equilibrio, dependiendo del país. >6% = mercado laboral débil, BC bajo presión dovish → negativo para la divisa. Un desempleo bajo suele combinarse con crecimiento salarial, que a su vez alimenta inflación. Fuente: FRED / OECD.",
  currentAccount: "Saldo de la balanza por cuenta corriente como porcentaje del PIB. Incluye comercio de bienes, servicios, rentas e inversión. Superávit (positivo) = el país exporta más de lo que importa y recibe más rentas de las que paga → demanda estructural de la divisa para pagar esas exportaciones → positivo a largo plazo. Déficit (negativo) = el país consume más de lo que produce, necesita financiación externa → presión vendedora estructural sobre la divisa. Especialmente relevante para divisas como AUD, NZD y CAD donde el comercio de commodities domina. Fuente: FRED / OECD.",
  debt: "Deuda pública acumulada del gobierno como porcentaje del PIB. <60% = finanzas públicas sanas, margen fiscal para estímulos → positivo. 60–90% = zona de vigilancia, sostenible si el crecimiento es estable. >90% = carga elevada que limita la política fiscal y puede presionar al BC a monetizar deuda → negativo estructural. >150% = zona de riesgo soberano, aunque países con superávit en cuenta corriente y deuda mayormente doméstica (como Japón) tienen menor riesgo real de default. La relación deuda/crecimiento es clave: deuda alta con crecimiento fuerte es más manejable. Fuente: IMF / World Bank.",
  tradeBalance: "Diferencia mensual entre exportaciones e importaciones de bienes y servicios, expresada en miles de millones USD. Superávit (positivo) = el país vende más al exterior de lo que compra → entrada neta de divisas → demanda de la moneda local → positivo. Déficit (negativo) = el país compra más del exterior → salida neta de divisas → presión bajista sobre la moneda. EE.UU. históricamente tiene un déficit comercial masivo, pero el dólar se sostiene por su estatus de moneda de reserva global. Para AUD, CAD y NZD, la balanza comercial fluctúa fuertemente con los precios de commodities. Fuente: FRED / OECD.",
  production: "Variación porcentual en el volumen de producción del sector industrial (manufactura, minería, energía). Para la mayoría de divisas el dato es mensual (MoM — mes sobre mes); para NZD es trimestral (QoQ) ya que Statistics New Zealand publica con esa frecuencia. >1.5% = expansión industrial fuerte, señal de demanda robusta y empleo sostenido → positivo. >0% = expansión moderada, neutral-positivo. Negativo = contracción industrial → anticipa debilidad económica y posible presión dovish. Especialmente relevante para divisas industriales como EUR (Alemania) y JPY (manufactura de precisión). Fuente: FRED / OECD.",
  retailSales: "Variación mensual (MoM) en el valor de las ventas del sector minorista. Es el principal indicador del consumo privado, que representa el 60–70% del PIB en economías desarrolladas. >0.5% = consumo robusto → señal de confianza del consumidor y economía activa → positivo para la divisa. 0–0.5% = crecimiento moderado, neutral. Negativo = contracción del gasto privado → anticipa desaceleración económica → negativo. Las ventas minoristas son un indicador adelantado: cuando los consumidores recortan gastos, la recesión suele seguir en 2–3 trimestres. Fuente: FRED / OECD.",
  wageGrowth: "Tasa de crecimiento anual de los salarios nominales. Es un indicador clave de presiones inflacionarias de segunda ronda: salarios altos → mayor poder adquisitivo → mayor gasto → inflación de demanda. 2.5–4.5% = rango saludable, compatible con el objetivo de inflación del 2% y productividad normal → positivo. >5% = presiones salariales que pueden desanclar expectativas de inflación → BC bajo presión hawkish. <2% = deflación salarial, poder adquisitivo estancado → negativo para consumo y crecimiento. Los bancos centrales monitorizan el crecimiento salarial con especial atención como señal de inflación persistente. Fuente: FRED / OECD.",
  manufacturingPMI: "Purchasing Managers Index del sector manufacturero. Encuesta mensual a directores de compras sobre nuevos pedidos, producción, empleo, inventarios y plazos de entrega. Escala: >50 = expansión (más empresas reportan mejora que deterioro), <50 = contracción. >55 = expansión fuerte, señal positiva para la divisa. 50–55 = expansión moderada. 45–50 = contracción leve, precaución. <45 = contracción severa, señal negativa clara. Es un indicador adelantado muy seguido por los mercados porque se publica muy rápidamente (primer día hábil del mes siguiente) y anticipa actividad económica real. Fuente: S&P Global / JMFA via FRED.",
  servicesPMI: "Purchasing Managers Index del sector servicios (bancos, seguros, turismo, salud, tecnología). Peso: 5% en el score — mayor que manufacturero (3%) porque servicios representa >75% del PIB en USD, GBP y EUR. La misma escala que el PMI Manufacturero: >50 = expansión, <50 = contracción. En economías post-industriales, este indicador anticipa inflación de servicios (componente más pegajoso del IPC) y empleo, los dos factores que más pesan en las decisiones de política monetaria de la Fed, el BCE y el BOE. Fuente: S&P Global via FRED.",
  economicSurprise: "Economic Surprise Index (ESI) — mide si los datos macroeconómicos recientes superaron o decepcionaron las expectativas del consenso. Score >50 = los datos publicados estuvieron por encima de lo esperado (sorpresa positiva, bullish para la divisa). Score <50 = datos peores de lo esperado (sorpresa negativa, bearish). Calculado con arquitectura híbrida: (A) cuando hay ≥3 eventos con actual y forecast en el calendario de los últimos 30 días, usa sorpresas reales normalizadas por z-score (blend progresivo: 3 eventos→40%, 8→65%, ≥15→80% real); (B) como complemento, mide la posición relativa de la divisa en 7 indicadores clave dentro del G8. Peso: 4%. Es el factor que más falta en los modelos fundamentales que solo leen niveles — el mercado ya tiene el nivel descontado, lo que mueve el precio es la desviación vs expectativas. Fuente: calendar-data/calendar.json (Investing.com) + cálculo interno G8.",
  cotPositioning: "Posicionamiento neto de la categoría Leveraged Funds (hedge funds y CTAs) según el reporte Commitment of Traders (COT) publicado semanalmente por la CFTC (Commodity Futures Trading Commission) de EE.UU. Calcula: contratos largos (bullish) menos contratos cortos (bearish) en los mercados de futuros de divisas de Chicago (CME). Fuente: reporte Futures Only (financial_lf.htm) — excluye opciones para reflejar únicamente presión directa sobre el precio. Valores positivos = el smart money especulativo tiene posición neta larga → sesgo alcista institucional. Valores negativos = posición corta neta → sesgo bajista. >80K = posicionamiento alcista extremo, posible saturación y riesgo de reversal. <-80K = bajista extremo, posible rebote contrarian. Los Leveraged Funds son el único grupo cuya posición refleja apuestas especulativas puras sobre dirección de divisas, sin mandatos de cobertura ni posiciones pasivas. Fuente: CFTC oficial (cftc.gov/dea/futures/financial_lf.htm), actualización viernes.",
  consumerConfidence: "Índice que mide el optimismo o pesimismo de los hogares sobre su situación económica actual y futura. Varía por país: USA usa el índice de Michigan Consumer Sentiment (escala ~50–100); Eurozona, UK, AUD, CAD usan índices propios base 100. >100 (o equivalente) = los consumidores se sienten optimistas → mayor disposición al gasto, inversión, endeudamiento → positivo para crecimiento y divisa. <95 = pesimismo → contracción del consumo privado anticipada → negativo. Es un indicador adelantado del ciclo económico: la confianza cae antes de que los datos duros (PIB, ventas) lo reflejen. Fuente: FRED / OECD.",
  bond10y: "Rendimiento (yield) del bono soberano del gobierno a 10 años. Refleja las expectativas de mercado sobre tasas de interés futuras e inflación a largo plazo. Para EUR se usa el yield oficial de la Zona Euro (agregado IRLTLT01EZM156N via FRED), que refleja el costo de financiamiento soberano del conjunto de la eurozona. Yield alto (>4%) = el mercado espera tasas altas sostenidas → carry atractivo → entrada de capital extranjero → positivo para la divisa. Yield bajo (<1%) = expectativas de tasas bajas o deflación → poco atractivo de carry. El 'yield spread' entre dos países (ej. USA 10Y vs DEU 10Y) es uno de los drivers más potentes de los grandes flujos de capital institucional en forex. Una curva invertida (10Y < tasa de política) señala que el mercado anticipa recortes futuros → bearish implícito. Fuente: FRED / bancos centrales.",
  businessConfidence: "Índice que mide el sentimiento de los directivos y empresarios sobre las condiciones actuales y perspectivas futuras del negocio. Varía por país: Eurozona usa el ESI de la Comisión Europea (base 100); NZD usa el ANZ Business Outlook (balance neto, centrado en 0); USA y CAD usan variantes del ISM o Ivey PMI. >100 (o equivalente positivo) = las empresas esperan expandir producción, contratar personal e invertir → positivo para crecimiento económico y divisa. <100 = perspectivas de contracción. Es un predictor adelantado de inversión empresarial y empleo, que a su vez impacta en consumo, producción e inflación. Los mercados lo usan para anticipar el ciclo económico con 2–4 trimestres de antelación. Fuente: FRED / OECD / organismos estadísticos nacionales.",
  inflationExpectations: "Expectativas de inflación a 1–2 años derivadas de encuestas a consumidores, empresas o del mercado de bonos (breakeven inflation). Son distintas de la inflación actual y tienen un rol especial en la política monetaria: si las expectativas se 'desanclan' del objetivo del 2%, el BC se ve forzado a actuar agresivamente para preservar su credibilidad. 1.8–2.5% = bien ancladas, zona ideal → banco central no necesita actuar → positivo para estabilidad de la divisa. >3% = expectativas desancladas al alza → presión para subir tasas → hawkish implícito. <1.5% = riesgo deflacionario → presión para bajar tasas → dovish. Los mercados de bonos (TIPS en USA, OATi en Francia) ofrecen una medida de mercado de las expectativas de inflación en tiempo real. Fuente: FRED / World Bank.",
  termsOfTrade: "Índice de términos de intercambio: ratio entre el precio de las exportaciones y el precio de las importaciones (base 100). Un valor >100 significa que el país recibe más valor por sus exportaciones de lo que paga por sus importaciones → mejora la cuenta corriente → positivo para la divisa. Un valor <100 implica lo contrario: el país paga más por lo que importa de lo que recibe por lo que exporta → deterioro del balance externo → negativo. Especialmente relevante para AUD, CAD y NZD (economías exportadoras de commodities como mineral de hierro, petróleo y productos lácteos): cuando sube el precio del hierro o el petróleo, sus términos de intercambio mejoran y sus divisas se aprecian. Para EUR se calcula como ratio de los índices de precios de exportación e importación extrazona (ambos de Eurostat vía FRED, base 2015=100). Fuente: FRED / Eurostat."
};

// ========== COMPONENTE PRINCIPAL ==========

var ForexDashboard = function ForexDashboard() {
  var _useState = useState(function () {
      var params = new URLSearchParams(window.location.search);
      var tab = params.get('tab');
      return ['overview', 'heatmap', 'trends', 'calendar', 'alerts'].includes(tab) ? tab : 'overview';
    }),
    _useState2 = _slicedToArray(_useState, 2),
    activeTab = _useState2[0],
    setActiveTab = _useState2[1];
  // ── Cache-primed initial state ──────────────────────────────────────
  // Lee el caché de localStorage antes del primer render para que los datos
  // aparezcan inmediatamente al volver desde news.html u otras páginas.
  var _cachedEcon = CacheManager.get('all_economic_data');
  var _cachedCal = CacheManager.get('economic_calendar');
  var _cachedHist = CacheManager.get('historical_data');
  var _cachedPairsRaw = CacheManager.get('forex_pair_recommendations');
  // Validate cached pairs — reject stale entries where strength/weakness are null
  // (generated before fix v6.4.1 when scores could be null)
  var _cachedPairs = Array.isArray(_cachedPairsRaw) && _cachedPairsRaw.every(function (r) {
    return typeof r.strength === 'number' && typeof r.weakness === 'number';
  }) ? _cachedPairsRaw : null;
  var _cachedRates = CacheManager.get('forex_rates');
  var _cachedAI = CacheManager.get('ai_analysis_Groq');
  // Si hay caché válido en alguna de las secciones principales, arrancamos
  // sin pantalla de carga y refrescamos en background lo que haga falta.
  var _hasAnyCache = !!(_cachedEcon || _cachedCal || _cachedRates);
  var _useState3 = useState(_cachedRates || {}),
    _useState4 = _slicedToArray(_useState3, 2),
    forexRates = _useState4[0],
    setForexRates = _useState4[1];
  var _useState5 = useState(_cachedEcon || {}),
    _useState6 = _slicedToArray(_useState5, 2),
    economicData = _useState6[0],
    setEconomicData = _useState6[1];
  var _useState7 = useState(_cachedPairs || []),
    _useState8 = _slicedToArray(_useState7, 2),
    dynamicAlerts = _useState8[0],
    setDynamicAlerts = _useState8[1];
  var _useState9 = useState(_cachedCal || {
      events: [],
      lastUpdate: null,
      source: null,
      impactCounts: {},
      currencyCounts: {}
    }),
    _useState0 = _slicedToArray(_useState9, 2),
    economicCalendar = _useState0[0],
    setEconomicCalendar = _useState0[1];
  var _useState1 = useState('all'),
    _useState10 = _slicedToArray(_useState1, 2),
    calFilter = _useState10[0],
    setCalFilter = _useState10[1];
  var _useState11 = useState(null),
    _useState12 = _slicedToArray(_useState11, 2),
    calSelectedDate = _useState12[0],
    setCalSelectedDate = _useState12[1];
  var calendarContainerRef = React.useRef(null);
  var nextEventRef = React.useRef(null);
  // Auto-scroll to next event when active date, filter, or tab changes
  React.useEffect(function () {
    if (activeTab !== 'calendar') return;
    // setTimeout lets the calendar DOM render before measuring
    var timer = setTimeout(function () {
      if (nextEventRef.current && calendarContainerRef.current) {
        var container = calendarContainerRef.current;
        var el = nextEventRef.current;
        var elTop = el.offsetTop - container.offsetTop;
        container.scrollTop = Math.max(0, elTop - 16);
      }
    }, 50);
    return function () {
      return clearTimeout(timer);
    };
  }, [calSelectedDate, calFilter, activeTab]);
  // Scroll active nav-tab into view on tab change
  React.useEffect(function () {
    var activeBtn = document.querySelector('.nav-tab.active');
    if (activeBtn) {
      activeBtn.scrollIntoView({
        inline: 'center',
        block: 'nearest',
        behavior: 'smooth'
      });
    }
  }, [activeTab]);

  // ── Live polling del calendario ──────────────────────────────────
  // Zonas de frecuencia según proximidad al próximo evento sin actual:
  //   HOT  (evento entre -2 min y +15 min) → cada 30 s
  //   WARM (evento entre +15 min y +60 min) → cada 60 s
  //   COOL (sin evento próximo)             → cada 5 min
  // Cuando llega un nuevo actual se hace flash visual en la fila.
  var _React$useState = React.useState(false),
    _React$useState2 = _slicedToArray(_React$useState, 2),
    calendarUpdating = _React$useState2[0],
    setCalendarUpdating = _React$useState2[1];
  React.useEffect(function () {
    var pollTimer = null;
    var getNextPollDelay = function getNextPollDelay() {
      var events = economicCalendar && economicCalendar.events || [];
      var nowMs = Date.now();
      var hotZone = false,
        warmZone = false;
      var _iterator4 = _createForOfIteratorHelper(events),
        _step4;
      try {
        for (_iterator4.s(); !(_step4 = _iterator4.n()).done;) {
          var ev = _step4.value;
          if (ev.actual && ev.actual !== '') continue;
          var t = ev.timeUTC || ev.time;
          if (!t || !ev.dateISO) continue;
          var d = new Date("".concat(ev.dateISO, "T").concat(t, ":00Z"));
          if (isNaN(d.getTime())) continue;
          var diff = d.getTime() - nowMs; // ms hasta el evento
          if (diff > -120000 && diff < 900000) {
            hotZone = true;
            break;
          } // ±2 min / +15 min
          if (diff >= 900000 && diff < 3600000) warmZone = true; // +15–60 min
        }
      } catch (err) {
        _iterator4.e(err);
      } finally {
        _iterator4.f();
      }
      if (hotZone) return 30000; // 30 s
      if (warmZone) return 60000; // 60 s
      return 300000; // 5 min
    };
    var _schedulePoll = function schedulePoll() {
      var delay = getNextPollDelay();
      pollTimer = setTimeout(/*#__PURE__*/_asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee19() {
        var fresh, prevEvents, prevActuals, newActuals, _t14;
        return _regenerator().w(function (_context19) {
          while (1) switch (_context19.p = _context19.n) {
            case 0:
              _context19.p = 0;
              setCalendarUpdating(true);
              CacheManager.clear('economic_calendar');
              _context19.n = 1;
              return fetchEconomicCalendar();
            case 1:
              fresh = _context19.v;
              if (fresh && fresh.events && fresh.events.length > 0) {
                // Detectar filas que recibieron un actual nuevo para flash visual
                prevEvents = economicCalendar && economicCalendar.events || [];
                prevActuals = new Set(prevEvents.filter(function (e) {
                  return e.actual;
                }).map(function (e) {
                  return "".concat(e.dateISO, "|").concat(e.currency, "|").concat(e.event);
                }));
                newActuals = fresh.events.filter(function (e) {
                  return e.actual && !prevActuals.has("".concat(e.dateISO, "|").concat(e.currency, "|").concat(e.event));
                }).map(function (e) {
                  return "".concat(e.dateISO, "|").concat(e.currency, "|").concat(e.event);
                });
                setEconomicCalendar(fresh);
                // Flash en filas nuevas ~100 ms después del render
                if (newActuals.length > 0) {
                  setTimeout(function () {
                    newActuals.forEach(function (key) {
                      var el = document.querySelector("[data-event-key=\"".concat(CSS.escape(key), "\"]"));
                      if (el) {
                        el.classList.add('calendar-event--new-actual');
                        setTimeout(function () {
                          return el.classList.remove('calendar-event--new-actual');
                        }, 3000);
                      }
                    });
                  }, 120);
                }
              }
              _context19.n = 3;
              break;
            case 2:
              _context19.p = 2;
              _t14 = _context19.v;
            case 3:
              _context19.p = 3;
              setCalendarUpdating(false);
              return _context19.f(3);
            case 4:
              _schedulePoll();
            case 5:
              return _context19.a(2);
          }
        }, _callee19, null, [[0, 2, 3, 4]]);
      })), delay);
    };
    _schedulePoll();
    return function () {
      if (pollTimer) clearTimeout(pollTimer);
    };
  }, [economicCalendar]);
  var _useState13 = useState(_cachedRates ? new Date() : null),
    _useState14 = _slicedToArray(_useState13, 2),
    lastUpdate = _useState14[0],
    setLastUpdate = _useState14[1];
  var _useState15 = useState(_cachedHist || {}),
    _useState16 = _slicedToArray(_useState15, 2),
    historicalData = _useState16[0],
    setHistoricalData = _useState16[1];
  var _useState17 = useState(_cachedAI || null),
    _useState18 = _slicedToArray(_useState17, 2),
    aiAnalyses = _useState18[0],
    setAiAnalyses = _useState18[1];
  var _useState19 = useState(!!_cachedAI),
    _useState20 = _slicedToArray(_useState19, 2),
    aiAnalysisReady = _useState20[0],
    setAiAnalysisReady = _useState20[1];
  var _useState21 = useState(_hasAnyCache ? {
      status: 'Actualizando datos en segundo plano...',
      progress: 100
    } : {
      status: 'Inicializando...',
      progress: 0
    }),
    _useState22 = _slicedToArray(_useState21, 2),
    dataLoadingStatus = _useState22[0],
    setDataLoadingStatus = _useState22[1];
  var _useState23 = useState(!_hasAnyCache),
    _useState24 = _slicedToArray(_useState23, 2),
    isLoading = _useState24[0],
    setIsLoading = _useState24[1];
  // scoresVersion: incrementado cada vez que _precomputedScores se carga/actualiza,
  // forzando un re-render para que getSortedCountries() y getStrength() usen los nuevos datos.
  var _useState25 = useState(0),
    _useState26 = _slicedToArray(_useState25, 2),
    scoresVersion = _useState26[0],
    setScoresVersion = _useState26[1];
  var chartRefs = useRef({});
  var createCharts = function createCharts() {
    Object.values(chartRefs.current).forEach(function (chart) {
      if (chart) {
        try {
          chart.destroy();
        } catch (e) {
          console.warn('Error destroying chart:', e);
        }
      }
    });
    chartRefs.current = {};
    setTimeout(function () {
      countries.forEach(function (country) {
        var canvas = document.getElementById("chart-".concat(country.code));
        if (!canvas) {
          console.warn("Canvas not found for ".concat(country.code));
          return;
        }
        // Accesibilidad: describir gráfico para screen readers
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label', "Gr\xE1fico de tasas de inter\xE9s hist\xF3ricas para ".concat(country.code, ". Use la tabla de datos debajo para acceso accesible."));
        var ctx = canvas.getContext('2d');
        var data = historicalData[country.code];
        if (!(data !== null && data !== void 0 && data.labels) || !(data !== null && data !== void 0 && data.rates) || data.source === 'unavailable') {
          // Mostrar mensaje en lugar de gráfico vacío
          ctx.font = '14px Inter, sans-serif';
          ctx.fillStyle = '#6e7681';
          ctx.textAlign = 'center';
          ctx.fillText('Histórico en construcción', canvas.width / 2, canvas.height / 2 - 10);
          ctx.font = '12px Inter, sans-serif';
          ctx.fillText('Ejecuta el workflow de backfill para obtener datos reales', canvas.width / 2, canvas.height / 2 + 15);
          return;
        }
        try {
          chartRefs.current[country.code] = new Chart(ctx, {
            type: 'line',
            data: {
              labels: data.labels,
              datasets: [{
                label: 'Tasa de Interés % (bancos centrales oficiales / BIS)',
                data: data.rates,
                borderColor: '#1e88e5',
                backgroundColor: 'rgba(30, 136, 229, 0.1)',
                yAxisID: 'y',
                tension: 0,
                borderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6
              }
              // Índice de Fortaleza eliminado — no hay datos históricos reales del índice compuesto
              ]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              interaction: {
                mode: 'index',
                intersect: false
              },
              plugins: {
                legend: {
                  onClick: null,
                  labels: {
                    color: '#e1e4e8',
                    font: {
                      size: 11
                    },
                    padding: 10,
                    boxWidth: 24,
                    boxHeight: 2,
                    usePointStyle: false
                  }
                },
                title: {
                  display: false
                },
                tooltip: {
                  backgroundColor: 'rgba(26, 31, 46, 0.95)',
                  titleColor: '#e1e4e8',
                  bodyColor: '#8b949e',
                  borderColor: '#30363d',
                  borderWidth: 1
                }
              },
              scales: {
                y: {
                  type: 'linear',
                  position: 'left',
                  grid: {
                    color: '#30363d'
                  },
                  ticks: {
                    color: '#8b949e'
                  },
                  title: {
                    display: true,
                    text: 'Tasa %',
                    color: '#8b949e'
                  },
                  min: function () {
                    var minVal = Math.min.apply(Math, _toConsumableArray(data.rates || [0]));
                    // Floor to nearest 0.25 step, with 0.25 padding below
                    var floored = Math.floor(minVal * 4) / 4;
                    return Math.max(0, floored - 0.25);
                  }()
                },
                x: {
                  grid: {
                    color: '#30363d'
                  },
                  ticks: {
                    color: '#8b949e'
                  }
                }
              }
            }
          });
          console.log("\u2705 Chart created for ".concat(country.code));
        } catch (error) {
          console.error("Error creating chart for ".concat(country.code, ":"), error);
        }
      });
    }, 250);
  };
  var HeatmapCell = function HeatmapCell(_ref34) {
    var value = _ref34.value,
      type = _ref34.type,
      currency = _ref34.currency,
      indicator = _ref34.indicator,
      lastUpdate = _ref34.lastUpdate;
    var cellRef = React.useRef(null);
    var _React$useState3 = React.useState(false),
      _React$useState4 = _slicedToArray(_React$useState3, 2),
      showTooltip = _React$useState4[0],
      setShowTooltip = _React$useState4[1];
    var _React$useState5 = React.useState({
        x: 0,
        y: 0
      }),
      _React$useState6 = _slicedToArray(_React$useState5, 2),
      tooltipPosition = _React$useState6[0],
      setTooltipPosition = _React$useState6[1];
    var updateCellTooltipPos = function updateCellTooltipPos(e) {
      var tooltipWidth = 220;
      var tooltipHeight = 55;
      var offset = 14;
      var margin = 10;
      var x = e.clientX;
      var y = e.clientY + offset;
      if (x + tooltipWidth > window.innerWidth - margin) x = window.innerWidth - tooltipWidth - margin;
      if (x < margin) x = margin;
      if (y + tooltipHeight > window.innerHeight - margin) y = e.clientY - tooltipHeight - offset;
      setTooltipPosition({
        x: x,
        y: y
      });
    };
    var handleMouseEnter = function handleMouseEnter(e) {
      if (!lastUpdate) return;
      updateCellTooltipPos(e);
      setShowTooltip(true);
    };
    var handleMouseMove = function handleMouseMove(e) {
      if (showTooltip) updateCellTooltipPos(e);
    };
    var handleMouseLeave = function handleMouseLeave() {
      setShowTooltip(false);
    };
    var formatDate = function formatDate(dateString) {
      if (!dateString) return 'N/D';
      try {
        var date = new Date(dateString);
        var today = new Date();
        var diffTime = Math.abs(today - date);
        var diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        if (diffDays === 0) return 'Hoy';
        if (diffDays === 1) return 'Ayer';
        if (diffDays <= 7) return "".concat(diffDays, "d");
        if (diffDays <= 30) return "".concat(Math.floor(diffDays / 7), "sem");
        return date.toLocaleDateString('es-ES', {
          day: '2-digit',
          month: 'short'
        });
      } catch (_unused2) {
        return dateString;
      }
    };
    var indicatorNames = {
      'gdp': 'PIB',
      'gdpGrowth': 'Crecimiento PIB',
      'interestRate': 'Tasa de Interés',
      'inflation': 'Inflación',
      'unemployment': 'Desempleo',
      'currentAccount': 'Cuenta Corriente',
      'debt': 'Deuda Pública',
      'tradeBalance': 'Balanza Comercial',
      'production': 'Producción Industrial',
      'retailSales': 'Ventas Minoristas',
      'wageGrowth': 'Crecimiento Salarial',
      'manufacturingPMI': 'PMI Manufacturero',
      'servicesPMI': 'PMI Servicios',
      'economicSurprise': 'ESI',
      'cotPositioning': 'COT Positioning',
      'bond10y': 'Bono 10Y',
      'consumerConfidence': 'Conf. Consumidor',
      'businessConfidence': 'Conf. Empresarial',
      'capitalFlows': 'Flujos de Capital',
      'inflationExpectations': 'Exp. Inflación'
    };

    // Crear el contenido del tooltip
    React.useEffect(function () {
      if (showTooltip && lastUpdate) {
        // Crear tooltip en el body
        var tooltip = document.getElementById('global-heatmap-tooltip');
        if (!tooltip) {
          tooltip = document.createElement('div');
          tooltip.id = 'global-heatmap-tooltip';
          tooltip.className = 'heatmap-cell-tooltip';
          document.body.appendChild(tooltip);
        }
        tooltip.innerHTML = "\n                <span class=\"tooltip-date\">".concat(formatDate(lastUpdate), "</span>\n                <span class=\"tooltip-indicator\">").concat(currency, " - ").concat(indicatorNames[indicator] || indicator, "</span>\n            ");
        tooltip.style.left = "".concat(tooltipPosition.x, "px");
        tooltip.style.top = "".concat(tooltipPosition.y, "px");
        tooltip.style.transform = 'none';
        tooltip.classList.add('visible');
      } else {
        var _tooltip = document.getElementById('global-heatmap-tooltip');
        if (_tooltip) {
          _tooltip.classList.remove('visible');
        }
      }
      return function () {
        var tooltip = document.getElementById('global-heatmap-tooltip');
        if (tooltip) {
          tooltip.classList.remove('visible');
        }
      };
    }, [showTooltip, tooltipPosition, lastUpdate, currency, indicator]);
    return /*#__PURE__*/React.createElement("td", {
      ref: cellRef,
      style: {
        background: getHeatmapColor(value, type)
      },
      "data-has-date": lastUpdate ? "true" : "false",
      onMouseEnter: handleMouseEnter,
      onMouseMove: handleMouseMove,
      onMouseLeave: handleMouseLeave
    }, type === 'gdp' ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("span", {
      className: "cell-value"
    }, formatValue(value, 2)), /*#__PURE__*/React.createElement("span", {
      className: "cell-suffix"
    }, "T USD")) : type === 'gdpGrowth' || type === 'interestRate' || type === 'inflation' || type === 'unemployment' || type === 'production' || type === 'retailSales' || type === 'wageGrowth' ? /*#__PURE__*/React.createElement("span", {
      className: "cell-value"
    }, value > 0 && (type === 'production' || type === 'retailSales') ? '+' : '', formatValue(value, type === 'interestRate' ? 2 : 1), "%") : type === 'currentAccount' || type === 'debt' ? /*#__PURE__*/React.createElement("span", {
      className: "cell-value"
    }, value > 0 && type === 'currentAccount' ? '+' : '', formatValue(value), "%") : type === 'tradeBalance' ? /*#__PURE__*/React.createElement("span", {
      className: "cell-value"
    }, value > 0 ? '+' : '', (value / 1000).toFixed(1), "B") : type === 'manufacturingPMI' || type === 'servicesPMI' ? /*#__PURE__*/React.createElement("span", {
      className: "cell-value"
    }, formatValue(value, 1)) : type === 'economicSurprise' ? /*#__PURE__*/React.createElement("span", {
      className: "cell-value"
    }, formatValue(value, 1)) : type === 'cotPositioning' ? value !== null && value !== undefined ? /*#__PURE__*/React.createElement("span", {
      className: "cell-value"
    }, value > 0 ? '+' : '', (value / 1000).toFixed(1), "K") : /*#__PURE__*/React.createElement("span", {
      className: "cell-value",
      style: {
        color: 'var(--text-tertiary)'
      }
    }, "N/A") : type === 'bond10y' || type === 'inflationExpectations' ? /*#__PURE__*/React.createElement("span", {
      className: "cell-value"
    }, formatValue(value, 2), "%") : type === 'termsOfTrade' ? /*#__PURE__*/React.createElement("span", {
      className: "cell-value"
    }, formatValue(value, 1)) : type === 'consumerConfidence' || type === 'businessConfidence' ? /*#__PURE__*/React.createElement("span", {
      className: "cell-value"
    }, formatValue(value, 1)) : /*#__PURE__*/React.createElement("span", {
      className: "cell-value"
    }, formatValue(value)));
  };
  var TooltipCell = function TooltipCell(_ref35) {
    var children = _ref35.children,
      tooltip = _ref35.tooltip,
      title = _ref35.title;
    var _React$useState7 = React.useState(false),
      _React$useState8 = _slicedToArray(_React$useState7, 2),
      showTooltip = _React$useState8[0],
      setShowTooltip = _React$useState8[1];
    var _React$useState9 = React.useState({
        x: 0,
        y: 0
      }),
      _React$useState0 = _slicedToArray(_React$useState9, 2),
      position = _React$useState0[0],
      setPosition = _React$useState0[1];
    var updatePosition = function updatePosition(e) {
      var tooltipWidth = 340;
      var tooltipHeight = 200;
      var offset = 14;
      var margin = 12;
      var x = e.clientX;
      var y = e.clientY + offset;
      if (x + tooltipWidth > window.innerWidth - margin) x = window.innerWidth - tooltipWidth - margin;
      if (x < margin) x = margin;
      if (y + tooltipHeight > window.innerHeight - margin) y = e.clientY - tooltipHeight - offset;
      setPosition({
        x: x,
        y: y
      });
    };
    var handleMouseEnter = function handleMouseEnter(e) {
      updatePosition(e);
      setShowTooltip(true);
    };
    var handleMouseMove = function handleMouseMove(e) {
      if (showTooltip) updatePosition(e);
    };
    return /*#__PURE__*/React.createElement("th", {
      scope: "col",
      className: "tooltip-header"
    }, children, /*#__PURE__*/React.createElement("span", {
      className: "tooltip-icon",
      onMouseEnter: handleMouseEnter,
      onMouseMove: handleMouseMove,
      onMouseLeave: function onMouseLeave() {
        return setShowTooltip(false);
      }
    }, "?"), showTooltip && /*#__PURE__*/React.createElement("div", {
      className: "tooltip-box visible",
      style: {
        left: "".concat(position.x, "px"),
        top: "".concat(position.y, "px"),
        transform: 'none'
      }
    }, /*#__PURE__*/React.createElement("div", {
      className: "tooltip-header-title"
    }, title), /*#__PURE__*/React.createElement("div", {
      className: "tooltip-body"
    }, tooltip)));
  };
  var getHeatmapColor = function getHeatmapColor(value, type) {
    if (value === null || value === undefined) return 'var(--bg-card)';
    var intensity = 0;
    var isPositive = true;
    switch (type) {
      case 'gdp':
        return 'var(--bg-card)';
      case 'gdpGrowth':
        if (value >= 2.5) {
          intensity = 1.0;
          isPositive = true;
        } else if (value >= 1.5) {
          intensity = 0.6;
          isPositive = true;
        } else if (value >= 0.5) {
          intensity = 0.4;
          isPositive = false;
        } else {
          intensity = Math.min(Math.abs(value - 0.5) / 2, 1);
          isPositive = false;
        }
        break;
      case 'interestRate':
        if (value >= 4.0) {
          intensity = Math.min((value - 4.0) / 2, 1);
          isPositive = true;
        } else if (value >= 2.0) {
          intensity = 0.3;
          isPositive = true;
        } else if (value >= 0.5) {
          intensity = 0.5;
          isPositive = false;
        } else {
          intensity = 1.0;
          isPositive = false;
        }
        break;
      case 'inflation':
      case 'inflationExpectations':
        if (value >= 1.8 && value <= 2.5) {
          intensity = 0.8;
          isPositive = true;
        } else if (value > 2.5 && value <= 3.5) {
          intensity = 0.3;
          isPositive = true;
        } else if (value > 3.5) {
          intensity = Math.min((value - 3.5) / 2, 1);
          isPositive = false;
        } else {
          intensity = Math.min((1.8 - value) / 1.5, 1);
          isPositive = false;
        }
        break;
      case 'unemployment':
        if (value < 4.0) {
          intensity = 0.8;
          isPositive = true;
        } else if (value < 6.0) {
          intensity = 0.3;
          isPositive = true;
        } else {
          intensity = Math.min((value - 6.0) / 4, 1);
          isPositive = false;
        }
        break;
      case 'currentAccount':
        intensity = Math.min(Math.abs(value) / 15, 1); // value = % PIB, máx ~15%
        isPositive = value > 0;
        break;
      case 'tradeBalance':
        intensity = Math.min(Math.abs(value) / 15000, 1); // value = millones USD, máx ~$15B
        isPositive = value > 0;
        break;
      case 'debt':
        if (value < 60) {
          intensity = 0.7;
          isPositive = true;
        } else if (value < 90) {
          intensity = 0.3;
          isPositive = true;
        } else {
          intensity = Math.min((value - 90) / 100, 1);
          isPositive = false;
        }
        break;
      case 'production':
        if (value > 1.5) {
          intensity = Math.min(value / 3, 1);
          isPositive = true;
        } else if (value > 0) {
          intensity = 0.3;
          isPositive = true;
        } else {
          intensity = Math.min(Math.abs(value) / 3, 1);
          isPositive = false;
        }
        break;
      case 'retailSales':
        if (value > 0.5) {
          intensity = Math.min(value / 1.5, 1);
          isPositive = true;
        } else if (value > 0) {
          intensity = 0.3;
          isPositive = true;
        } else {
          intensity = Math.min(Math.abs(value) / 1.5, 1);
          isPositive = false;
        }
        break;
      case 'wageGrowth':
        if (value >= 2.5 && value <= 4.5) {
          intensity = 0.8;
          isPositive = true;
        } else if (value > 4.5 && value <= 6.0) {
          intensity = 0.4;
          isPositive = true;
        } else if (value > 6.0) {
          intensity = Math.min((value - 6.0) / 3, 1);
          isPositive = false;
        } else {
          intensity = Math.min((2.5 - value) / 2, 1);
          isPositive = false;
        }
        break;
      case 'manufacturingPMI':
      case 'servicesPMI':
        if (value > 52) {
          intensity = Math.min((value - 50) / 10, 1);
          isPositive = true;
        } else if (value >= 50) {
          intensity = 0.3;
          isPositive = true;
        } else if (value >= 48) {
          intensity = 0.4;
          isPositive = false;
        } else {
          intensity = Math.min((50 - value) / 10, 1);
          isPositive = false;
        }
        break;
      case 'economicSurprise':
        // 50 = neutral (G8 average), >50 = positive surprise, <50 = negative
        if (value > 65) {
          intensity = Math.min((value - 50) / 30, 1);
          isPositive = true;
        } else if (value > 52) {
          intensity = 0.35;
          isPositive = true;
        } else if (value >= 48) {
          intensity = 0.2;
          isPositive = true;
        } // near neutral
        else if (value >= 35) {
          intensity = 0.35;
          isPositive = false;
        } else {
          intensity = Math.min((50 - value) / 30, 1);
          isPositive = false;
        }
        break;
      case 'cotPositioning':
        // Umbrales calibrados para futures-only (financial_lf.htm).
        if (value === 0) return 'var(--bg-card)';
        if (value > 50000) {
          intensity = 0.8;
          isPositive = true;
        } else if (value > 25000) {
          intensity = 0.7;
          isPositive = true;
        } else if (value > 10000) {
          intensity = 0.5;
          isPositive = true;
        } else if (value > 0) {
          intensity = 0.3;
          isPositive = true;
        } else if (value > -10000) {
          intensity = 0.3;
          isPositive = false;
        } else if (value > -25000) {
          intensity = 0.5;
          isPositive = false;
        } else if (value > -50000) {
          intensity = 0.7;
          isPositive = false;
        } else {
          intensity = 0.8;
          isPositive = false;
        }
        break;
      case 'bond10y':
        if (value >= 4.0) {
          intensity = Math.min((value - 4.0) / 2, 1);
          isPositive = true;
        } else if (value >= 2.5) {
          intensity = 0.35;
          isPositive = true;
        } else if (value >= 1.0) {
          intensity = 0.4;
          isPositive = false;
        } else {
          intensity = Math.min((1.0 - value) / 1.5, 1);
          isPositive = false;
        }
        break;
      case 'consumerConfidence':
      case 'businessConfidence':
        if (value > 105) {
          intensity = Math.min((value - 105) / 15, 1);
          isPositive = true;
        } else if (value >= 100) {
          intensity = 0.3;
          isPositive = true;
        } else if (value >= 95) {
          intensity = 0.3;
          isPositive = false;
        } else {
          intensity = Math.min((95 - value) / 20, 1);
          isPositive = false;
        }
        break;
      case 'termsOfTrade':
        // Index base 100: >103 = favorable, <97 = unfavorable
        if (value >= 110) {
          intensity = Math.min((value - 110) / 20, 1);
          isPositive = true;
        } else if (value >= 103) {
          intensity = 0.6;
          isPositive = true;
        } else if (value >= 100) {
          intensity = 0.3;
          isPositive = true;
        } else if (value >= 95) {
          intensity = 0.4;
          isPositive = false;
        } else {
          intensity = Math.min((95 - value) / 15, 1);
          isPositive = false;
        }
        break;
      default:
        return 'var(--bg-card)';
    }

    // Apply colour tiers
    if (isPositive) {
      if (intensity > 0.7) return '#26a69a';
      if (intensity > 0.4) return '#00897b';
      return '#00695c';
    } else {
      if (intensity > 0.7) return '#ef5350';
      if (intensity > 0.4) return '#e53935';
      return '#c62828';
    }
  };
  var getSentiment = function getSentiment(strength) {
    // FIX: typeof null === 'object' in JS, so we must guard against null explicitly
    var numStrength = strength !== null && _typeof(strength) === 'object' && strength.score !== undefined ? strength.score : typeof strength === 'string' ? parseFloat(strength) : strength;
    // If numStrength is null/undefined/NaN, return neutral instead of crashing
    if (numStrength === null || numStrength === undefined || isNaN(numStrength)) return 'neutral';

    // Bandas fijas escala 0–100 — calibradas al estándar de la industria (DailyFX/Reuters/prop desk)
    // < 45 = Bajista · 45–65 = Neutral (zona sin convicción — no operar) · > 65 = Alcista
    // Umbral alcista: 65 (top ~25% del universo G8 — convicción alcista real)
    // Umbral bajista: 45 (convicción bajista real — 50 es neutro matemático, no señal de venta)
    // Zona 45–65: leve sesgo pero sin divergencia suficiente para señal operativa
    if (numStrength > 65) return 'alcista';
    if (numStrength < 45) return 'bajista';
    return 'neutral';
  };
  var generateAnalysis = function generateAnalysis(currency, forexRates, economicData) {
    var _economicData$currenc;
    if (!economicData[currency]) return 'Cargando datos...';
    var data = economicData[currency];
    var country = countries.find(function (c) {
      return c.code === currency;
    });

    // Usar bcOutlook del backend como fuente de verdad (v6.6.1: incluye rm_trough).
    // Fallback: outlook calculado localmente por determineCentralBankOutlook().
    var strengthObj = getStrength(currency);
    var _bcOutlook = strengthObj.bcOutlook;
    if (_bcOutlook && country) {
      country.outlook = _bcOutlook;
      if (economicData[currency]) economicData[currency].outlook = _bcOutlook;
    } else if ((_economicData$currenc = economicData[currency]) !== null && _economicData$currenc !== void 0 && _economicData$currenc.outlook && country) {
      country.outlook = economicData[currency].outlook;
    }
    var strength = strengthObj.score;
    var sentiment = getSentiment(strength);

    // ✅ Helper function para formatear valores de forma segura
    var fmt = function fmt(val) {
      var decimals = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 1;
      if (val === null || val === undefined) return null;
      return val.toFixed(decimals);
    };
    var analysis = [];

    // 1. INTRODUCCIÓN BASADA EN FORTALEZA Y TENDENCIA
    var strengthStr = fmt(strength);
    if (sentiment === 'alcista') {
      analysis.push("".concat(currency, " registra fortaleza fundamental s\xF3lida (\xEDndice ").concat(strengthStr, ") respaldada por condiciones macroecon\xF3micas favorables."));
    } else if (sentiment === 'bajista') {
      analysis.push("".concat(currency, " muestra debilidad estructural (\xEDndice ").concat(strengthStr, ") con m\xFAltiples indicadores bajo presi\xF3n."));
    } else {
      analysis.push("".concat(currency, " mantiene posici\xF3n neutral (\xEDndice ").concat(strengthStr, ") con factores mixtos que se compensan."));
    }

    // 2. ANÁLISIS DE POLÍTICA MONETARIA
    if (country && data.interestRate !== null && data.interestRate !== undefined) {
      var rateStr = fmt(data.interestRate, 2);
      if (country.outlook === 'Hawkish') {
        if (data.inflation !== null && data.inflation > 2.5) {
          var inflStr = fmt(data.inflation);
          analysis.push("".concat(country.centralBank, " mantiene postura hawkish con tasa en ").concat(rateStr, "% para controlar inflaci\xF3n de ").concat(inflStr, "%."));
        } else if (data.inflation !== null) {
          var _inflStr = fmt(data.inflation);
          analysis.push("".concat(country.centralBank, " en modo restrictivo (").concat(rateStr, "%) a pesar de inflaci\xF3n moderada de ").concat(_inflStr, "%."));
        } else {
          analysis.push("".concat(country.centralBank, " mantiene postura hawkish con tasa en ").concat(rateStr, "%."));
        }
      } else if (country.outlook === 'Dovish') {
        if (data.gdpGrowth !== null && data.gdpGrowth < 1.0) {
          var gdpStr = fmt(data.gdpGrowth);
          analysis.push("".concat(country.centralBank, " adopta postura dovish (tasa ").concat(rateStr, "%) ante desaceleraci\xF3n econ\xF3mica de ").concat(gdpStr, "%."));
        } else {
          analysis.push("".concat(country.centralBank, " mantiene pol\xEDtica acomodaticia con tasa en ").concat(rateStr, "% buscando estimular crecimiento."));
        }
      } else {
        analysis.push("".concat(country.centralBank, " en pausa estrat\xE9gica (").concat(rateStr, "%) evaluando datos antes de pr\xF3ximo movimiento."));
      }
    }

    // 3. ANÁLISIS DE BALANZA COMERCIAL Y CUENTA CORRIENTE
    if (data.tradeBalance !== null && data.tradeBalance !== undefined) {
      if (data.tradeBalance > 1000) {
        var tbStr = (data.tradeBalance / 1000).toFixed(1);
        analysis.push("Super\xE1vit comercial robusto de $".concat(tbStr, "B mensuales sostiene demanda estructural de ").concat(currency, "."));
      } else if (data.tradeBalance < -10000) {
        var _tbStr = Math.abs(data.tradeBalance / 1000).toFixed(1);
        analysis.push("D\xE9ficit comercial significativo de $".concat(_tbStr, "B mensuales presiona sobre la divisa."));
      }
    }
    if (data.currentAccount !== null && data.currentAccount !== undefined) {
      var caStr = fmt(data.currentAccount);
      if (data.currentAccount > 3) {
        analysis.push("Cuenta corriente superavitaria (+".concat(caStr, "% PIB) refuerza posici\xF3n externa."));
      } else if (data.currentAccount < -3) {
        analysis.push("D\xE9ficit en cuenta corriente (".concat(caStr, "% PIB) genera vulnerabilidad externa."));
      }
    }

    // 4. ANÁLISIS ESPECÍFICO POR DIVISA
    if (currency === 'USD') {
      if (data.gdpGrowth !== null && data.gdpGrowth !== undefined) {
        var _gdpStr = fmt(data.gdpGrowth);
        if (data.gdpGrowth > 2.5) {
          analysis.push("Econom\xEDa estadounidense en expansi\xF3n s\xF3lida (".concat(_gdpStr, "%) mantiene atractivo del d\xF3lar como activo refugio."));
        } else {
          analysis.push("Desaceleraci\xF3n econ\xF3mica dom\xE9stica (".concat(_gdpStr, "%) limita potencial alcista del d\xF3lar."));
        }
      }
    } else if (currency === 'EUR') {
      if (data.debt !== null && data.debt > 90) {
        var debtStr = fmt(data.debt, 0);
        analysis.push("Carga de deuda elevada (".concat(debtStr, "% PIB) en contexto de crecimiento d\xE9bil limita margen del BCE."));
      }
    } else if (currency === 'JPY') {
      if (data.debt !== null && data.debt > 200) {
        var _debtStr = fmt(data.debt, 0);
        analysis.push("Deuda p\xFAblica extrema (".concat(_debtStr, "% PIB) pero status de safe haven mantiene demanda en aversi\xF3n al riesgo."));
      }
      if (data.interestRate !== null && data.interestRate < 0.5) {
        analysis.push("Pol\xEDtica ultra-acomodaticia del BoJ mantiene diferencial negativo frente a otras divisas.");
      }
    } else if (currency === 'GBP') {
      if (data.inflation !== null && data.inflation > 3.0) {
        var _inflStr2 = fmt(data.inflation);
        analysis.push("Inflaci\xF3n persistente (".concat(_inflStr2, "%) fuerza al BoE a mantener pol\xEDtica restrictiva."));
      }
    } else if (currency === 'CHF') {
      if (data.inflation !== null && data.inflation < 1.5 && data.debt !== null && data.debt < 50) {
        var _inflStr3 = fmt(data.inflation);
        var _debtStr2 = fmt(data.debt, 0);
        analysis.push("Estabilidad de precios (".concat(_inflStr3, "%) y deuda baja (").concat(_debtStr2, "% PIB) sustentan fortaleza del franco."));
      }
    } else if (currency === 'AUD' || currency === 'NZD' || currency === 'CAD') {
      if (data.production !== null && data.production < 0) {
        var prodStr = fmt(data.production);
        analysis.push("Debilidad en sector industrial (".concat(prodStr, "%) refleja desaf\xEDos en econom\xEDa dependiente de commodities."));
      } else if (data.tradeBalance !== null && data.tradeBalance > 1000) {
        analysis.push("Exportaciones de materias primas sostienen super\xE1vit comercial y demanda de ".concat(currency, "."));
      }
    }

    // 5. ANÁLISIS DE CRECIMIENTO Y DESEMPLEO
    if (data.gdpGrowth !== null && data.gdpGrowth !== undefined && data.unemployment !== null && data.unemployment !== undefined) {
      var _gdpStr2 = fmt(data.gdpGrowth);
      var unempStr = fmt(data.unemployment);
      if (data.gdpGrowth > 2.5 && data.unemployment < 4.0) {
        analysis.push("Combinaci\xF3n de crecimiento robusto (".concat(_gdpStr2, "%) y pleno empleo (").concat(unempStr, "%) caracteriza entorno s\xF3lido."));
      } else if (data.gdpGrowth < 1.0 && data.unemployment > 6.0) {
        analysis.push("Estancamiento con crecimiento d\xE9bil (".concat(_gdpStr2, "%) y desempleo elevado (").concat(unempStr, "%) complica panorama."));
      } else if (data.gdpGrowth < 0) {
        analysis.push("Contracci\xF3n econ\xF3mica (".concat(_gdpStr2, "%) eleva riesgos de recesi\xF3n y presi\xF3n sobre la divisa."));
      }
    }

    // 6. ANÁLISIS DE CONSUMO Y MOMENTUM (NUEVOS INDICADORES)
    if (data.retailSales !== null && data.retailSales !== undefined && data.wageGrowth !== null && data.wageGrowth !== undefined) {
      var rsStr = fmt(data.retailSales);
      var wgStr = fmt(data.wageGrowth);
      if (data.retailSales > 0.5 && data.wageGrowth > 3.0) {
        analysis.push("Consumo dom\xE9stico robusto con ventas minoristas +".concat(rsStr, "% y salarios creciendo ").concat(wgStr, "% anual sostiene demanda agregada."));
      } else if (data.retailSales < -0.3) {
        analysis.push("Debilidad en consumo con ventas minoristas ".concat(rsStr, "% se\xF1ala desaceleraci\xF3n en demanda dom\xE9stica."));
      }
    }

    // 7. ANÁLISIS DE ACTIVIDAD MANUFACTURERA
    if (data.manufacturingPMI !== null && data.manufacturingPMI !== undefined) {
      var pmiStr = fmt(data.manufacturingPMI, 1);
      if (data.manufacturingPMI > 55) {
        analysis.push("Sector manufacturero en fuerte expansi\xF3n (PMI ".concat(pmiStr, ") impulsa crecimiento econ\xF3mico."));
      } else if (data.manufacturingPMI > 50) {
        analysis.push("Manufactura en expansi\xF3n moderada (PMI ".concat(pmiStr, ") mantiene momentum positivo."));
      } else if (data.manufacturingPMI < 45) {
        analysis.push("Sector manufacturero en contracci\xF3n aguda (PMI ".concat(pmiStr, ") presagia debilidad econ\xF3mica."));
      } else {
        analysis.push("Manufactura en contracci\xF3n (PMI ".concat(pmiStr, ") reduce perspectivas de crecimiento."));
      }
    }

    // 7b. ANÁLISIS DE ACTIVIDAD DE SERVICIOS (v6.2)
    if (data.servicesPMI !== null && data.servicesPMI !== undefined) {
      var svcStr = fmt(data.servicesPMI, 1);
      if (data.servicesPMI > 55) {
        analysis.push("Sector servicios en fuerte expansi\xF3n (PMI ".concat(svcStr, ") \u2014 principal driver de empleo e inflaci\xF3n subyacente."));
      } else if (data.servicesPMI > 50) {
        analysis.push("Servicios en expansi\xF3n (PMI ".concat(svcStr, ") sostiene el ciclo econ\xF3mico y la inflaci\xF3n subyacente."));
      } else if (data.servicesPMI < 45) {
        analysis.push("Contracci\xF3n severa en servicios (PMI ".concat(svcStr, ") se\xF1ala deterioro en empleo y consumo."));
      } else {
        analysis.push("Servicios en zona de contracci\xF3n (PMI ".concat(svcStr, "), presi\xF3n sobre el crecimiento del sector terciario."));
      }
    }

    // 8. ANÁLISIS DE POSICIONAMIENTO ESPECULATIVO (COT)
    if (data.cotPositioning !== null && data.cotPositioning !== undefined) {
      var cotK = (data.cotPositioning / 1000).toFixed(1);

      // Umbrales calibrados para futures-only (financial_lf.htm) — ~40-50% menores que combined
      if (data.cotPositioning > 80000) {
        analysis.push("Posicionamiento especulativo extremadamente alcista (+".concat(cotK, "K contratos netos) sugiere saturaci\xF3n del mercado y riesgo elevado de toma de ganancias."));
      } else if (data.cotPositioning > 40000) {
        analysis.push("Fuerte posicionamiento alcista de traders especulativos (+".concat(cotK, "K contratos) respalda momentum alcista, aunque con posible resistencia cerca."));
      } else if (data.cotPositioning > 15000) {
        analysis.push("Posicionamiento moderadamente alcista (+".concat(cotK, "K contratos) refleja confianza del mercado y sostiene tendencia alcista."));
      } else if (data.cotPositioning > 0) {
        analysis.push("Leve sesgo alcista en posicionamiento especulativo (+".concat(cotK, "K contratos) indica cautela pero con tendencia positiva."));
      } else if (data.cotPositioning > -15000) {
        analysis.push("Posicionamiento ligeramente bajista (".concat(cotK, "K contratos) refleja cautela y presi\xF3n vendedora moderada."));
      } else if (data.cotPositioning > -40000) {
        analysis.push("Sentimiento bajista entre especuladores (".concat(cotK, "K contratos netos) amplifica presi\xF3n vendedora y debilita soporte t\xE9cnico."));
      } else if (data.cotPositioning > -80000) {
        analysis.push("Fuerte posicionamiento bajista (".concat(cotK, "K contratos) se\xF1ala convicci\xF3n vendedora y riesgo de continuidad bajista."));
      } else {
        analysis.push("Posicionamiento bajista extremo (".concat(cotK, "K contratos) sugiere posible capitulaci\xF3n; podr\xEDa catalizar rebote t\xE9cnico si fundamentos cambian."));
      }
    }

    // 8B. ANÁLISIS DE BONOS Y FLUJOS (NUEVOS INDICADORES)
    if (data.bond10y !== null && data.bond10y !== undefined) {
      var b10y = fmt(data.bond10y, 2);
      if (data.bond10y >= 4.0) {
        analysis.push("Yield del bono soberano a 10 a\xF1os en ".concat(b10y, "% atrae flujos de carry trade internacionales."));
      } else if (data.bond10y < 1.0) {
        analysis.push("Yield a 10 a\xF1os en niveles bajos (".concat(b10y, "%) reduce atractivo para inversores de renta fija extranjeros."));
      }
    }

    // 8C. ANÁLISIS DE CONFIANZA Y EXPECTATIVAS
    if (data.consumerConfidence !== null && data.consumerConfidence !== undefined) {
      var ccStr = fmt(data.consumerConfidence, 1);
      if (data.consumerConfidence > 108) {
        analysis.push("Confianza del consumidor en zona de optimismo (\xEDndice ".concat(ccStr, ") anticipa expansi\xF3n del gasto privado y sostiene perspectivas de crecimiento."));
      } else if (data.consumerConfidence > 100) {
        analysis.push("Confianza del consumidor positiva (\xEDndice ".concat(ccStr, ") se\xF1ala disposici\xF3n al gasto en el corto plazo."));
      } else if (data.consumerConfidence < 92) {
        analysis.push("Confianza del consumidor deteriorada (\xEDndice ".concat(ccStr, ") anticipa contracci\xF3n del consumo privado y presiona el crecimiento a la baja."));
      } else if (data.consumerConfidence < 100) {
        analysis.push("Confianza del consumidor en territorio pesimista (\xEDndice ".concat(ccStr, ") refleja cautela en el gasto dom\xE9stico."));
      }
    }
    if (data.businessConfidence !== null && data.businessConfidence !== undefined) {
      var bcStr = fmt(data.businessConfidence, 1);
      var isNetBalance = currency === 'NZD';
      var bcHigh = isNetBalance ? 30 : 105;
      var bcLow = isNetBalance ? -10 : 95;
      var bcLabel = isNetBalance ? 'balance neto' : 'índice';
      if (data.businessConfidence > bcHigh) {
        analysis.push("Confianza empresarial elevada (".concat(bcLabel, " ").concat(bcStr, ") anticipa expansi\xF3n de inversi\xF3n y contrataciones en los pr\xF3ximos trimestres."));
      } else if (data.businessConfidence < bcLow) {
        analysis.push("Confianza empresarial deteriorada (".concat(bcLabel, " ").concat(bcStr, ") se\xF1ala reducci\xF3n esperada de inversi\xF3n y posibles recortes de empleo."));
      }
    }
    if (data.inflationExpectations !== null && data.inflationExpectations !== undefined) {
      var ieStr = fmt(data.inflationExpectations, 1);
      if (data.inflationExpectations > 3.0) {
        analysis.push("Expectativas de inflaci\xF3n desancladas (".concat(ieStr, "%) presionan al banco central hacia una postura m\xE1s restrictiva para preservar credibilidad."));
      } else if (data.inflationExpectations >= 1.8 && data.inflationExpectations <= 2.5) {
        analysis.push("Expectativas de inflaci\xF3n bien ancladas (".concat(ieStr, "%) refuerzan credibilidad del banco central y reducen necesidad de ajustes adicionales."));
      } else if (data.inflationExpectations < 1.5) {
        analysis.push("Expectativas de inflaci\xF3n por debajo del objetivo (".concat(ieStr, "%) aumentan presi\xF3n deflacionaria y limitan margen de maniobra del banco central."));
      }
    }

    // 9. CONCLUSIÓN PROSPECTIVA
    if (sentiment === 'alcista') {
      analysis.push("Perspectivas favorables mientras fundamentos econ\xF3micos se mantengan s\xF3lidos y pol\xEDtica monetaria evite giros abruptos.");
    } else if (sentiment === 'bajista') {
      analysis.push("Riesgos a la baja prevalecen en ausencia de catalizadores positivos que reviertan deterioro fundamental.");
    } else {
      analysis.push("Evoluci\xF3n depender\xE1 de pr\xF3ximos datos econ\xF3micos y se\xF1ales de bancos centrales sobre direcci\xF3n de pol\xEDtica monetaria.");
    }
    return analysis.join(' ');
  };
  var getSortedCountries = function getSortedCountries() {
    return [].concat(countries).sort(function (a, b) {
      var dataA = economicData[a.code] || {};
      var dataB = economicData[b.code] || {};

      // Primero: Ordenar por strength score
      var strengthObjA = getStrength(a.code);
      var strengthObjB = getStrength(b.code);
      var scoreA = strengthObjA.score || 50;
      var scoreB = strengthObjB.score || 50;

      // ✅ CRÍTICO: Mayor score primero (descendente)
      if (Math.abs(scoreA - scoreB) > 0.1) {
        return scoreB - scoreA;
      }

      // Desempate: Contar celdas verdes
      var indicators = ['gdpGrowth', 'interestRate', 'inflation', 'unemployment', 'currentAccount', 'debt', 'tradeBalance', 'production', 'retailSales', 'wageGrowth', 'manufacturingPMI', 'servicesPMI', 'economicSurprise', 'cotPositioning', 'bond10y', 'consumerConfidence', 'businessConfidence', 'inflationExpectations'];
      var greenCellsA = 0;
      var greenCellsB = 0;
      indicators.forEach(function (indicator) {
        var valueA = dataA[indicator];
        var valueB = dataB[indicator];
        if (valueA !== null && valueA !== undefined && !isNaN(valueA)) {
          var colorA = getHeatmapColor(valueA, indicator);
          if (colorA.includes('#26a69a') || colorA.includes('#00897b') || colorA.includes('#00695c')) {
            greenCellsA++;
          }
        }
        if (valueB !== null && valueB !== undefined && !isNaN(valueB)) {
          var colorB = getHeatmapColor(valueB, indicator);
          if (colorB.includes('#26a69a') || colorB.includes('#00897b') || colorB.includes('#00695c')) {
            greenCellsB++;
          }
        }
      });

      // ✅ Más celdas verdes primero
      return greenCellsB - greenCellsA;
    });
  };
  var hasCOTData = function hasCOTData() {
    var currencies = ['EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'];
    var cotAvailable = currencies.filter(function (curr) {
      var data = economicData[curr];
      return data && data.cotPositioning !== null && data.cotPositioning !== undefined;
    });
    return cotAvailable.length >= 4; // At least 4 currencies have COT data
  };
  var formatValue = function formatValue(value) {
    var decimals = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 1;
    if (value === null || value === undefined) return 'N/D';
    return value.toFixed(decimals);
  };
  useEffect(function () {
    var loadInitialData = /*#__PURE__*/function () {
      var _ref36 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee20() {
        var hasEconCache, hasCalCache, hasHistCache, hasPairsCache, hasAICache, hasRatesCache, hasFullCache, globalTimeout, _yield$Promise$all5, _yield$Promise$all6, _, rates, loadedData, calendarResult, historical, aiData, pairRecommendations, _t15;
        return _regenerator().w(function (_context20) {
          while (1) switch (_context20.p = _context20.n) {
            case 0:
              // ── Estrategia: caché primero, red en segundo plano ─────────────
              // Si ya tenemos datos en caché (p.ej. al volver desde news.html),
              // mostramos el dashboard inmediatamente y refrescamos en background
              // solo los datos que realmente necesiten actualización.
              hasEconCache = !!CacheManager.get('all_economic_data');
              hasCalCache = !!CacheManager.get('economic_calendar');
              hasHistCache = !!CacheManager.get('historical_data');
              hasPairsCache = !!CacheManager.get('forex_pair_recommendations');
              hasAICache = !!CacheManager.get('ai_analysis_Groq');
              hasRatesCache = !!CacheManager.get('forex_rates');
              hasFullCache = hasEconCache && hasCalCache && hasRatesCache;
              if (!hasFullCache) {
                _context20.n = 1;
                break;
              }
              // ✅ Caché completo: ya mostramos los datos (estado inicializado arriba).
              // Solo refrescamos tasas de cambio (expiran cada 60s) y lo que falte.
              console.log('⚡ Caché completo — dashboard listo al instante');
              setIsLoading(false);

              // Refresco silencioso de scores precalculados
              fetchStrengthScores().then(function () {
                // Aplicar bcOutlook del backend sobre el outlook calculado localmente.
                // Necesario en el path de caché: el outlook ya fue asignado desde caché
                // antes de que los scores se cargaran en background.
                setEconomicData(function (prev) {
                  var updated = _objectSpread({}, prev);
                  ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'].forEach(function(code) {
                    var bcOut = getStrength(code).bcOutlook;
                    if (bcOut && updated[code] && updated[code].outlook !== bcOut) {
                      console.log("\u2705 outlook override (bg) for " + code + ": " + updated[code].outlook + " \u2192 " + bcOut + " (backend v6.6.1)");
                      updated[code] = _objectSpread(_objectSpread({}, updated[code]), {}, { outlook: bcOut });
                      // Actualizar también el objeto country para el render de tarjetas
                      var c = countries.find(function(x) { return x.code === code; });
                      if (c) c.outlook = bcOut;
                    }
                  });
                  return updated;
                });
                return setScoresVersion(function (v) {
                  return v + 1;
                });
              }).catch(function (e) {
                return console.warn('Background scores refresh failed:', e);
              });

              // Refresco silencioso de tasas (siempre, expiran en 60s)
              fetchForexRates().then(function (rates) {
                setForexRates(rates);
                setLastUpdate(new Date());
              }).catch(function (e) {
                return console.warn('Background rates refresh failed:', e);
              });

              // Refresco silencioso del calendario si el caché está próximo a expirar
              if (!hasCalCache) {
                fetchEconomicCalendar().then(function (cal) {
                  return setEconomicCalendar(cal);
                }).catch(function (e) {
                  return console.warn('Background calendar refresh failed:', e);
                });
              }

              // Refresco silencioso de AI si no hay caché
              if (!hasAICache) {
                fetchAIAnalysis().then(function (aiData) {
                  if (aiData && Object.keys(aiData).length > 0) {
                    setAiAnalyses(aiData);
                    setAiAnalysisReady(true);
                    setEconomicData(function (prev) {
                      var updated = _objectSpread({}, prev);
                      Object.entries(aiData).forEach(function (_ref37) {
                        var _ref38 = _slicedToArray(_ref37, 2),
                          currency = _ref38[0],
                          analysis = _ref38[1];
                        if (updated[currency] && analysis.lastRateDecision) {
                          updated[currency] = _objectSpread(_objectSpread({}, updated[currency]), {}, {
                            _aiLastRateDecision: analysis.lastRateDecision
                          });
                        }
                      });
                      return updated;
                    });
                  }
                }).catch(function (e) {
                  return console.warn('Background AI refresh failed:', e);
                });
              }
              return _context20.a(2);
            case 1:
              // ── Sin caché completo: carga normal con pantalla de progreso ──
              setIsLoading(true);

              // Timeout máximo global: si en 60s no termina, mostrar error
              globalTimeout = setTimeout(function () {
                setIsLoading(false);
                setDataLoadingStatus({
                  status: 'Tiempo de carga agotado. Algunos datos pueden no estar disponibles.',
                  progress: 100
                });
              }, 60000);
              _context20.p = 2;
              // FIX: scores y tipos de cambio en paralelo — no bloquean la carga principal
              setDataLoadingStatus({
                status: 'Iniciando carga de datos...',
                progress: 5
              });
              _context20.n = 3;
              return Promise.all([fetchStrengthScores().then(function (v) {
                setScoresVersion(function (v2) {
                  return v2 + 1;
                });
                return v;
              }), fetchForexRates()]);
            case 3:
              _yield$Promise$all5 = _context20.v;
              _yield$Promise$all6 = _slicedToArray(_yield$Promise$all5, 2);
              _ = _yield$Promise$all6[0];
              rates = _yield$Promise$all6[1];
              setForexRates(rates);
              setLastUpdate(new Date());
              _context20.n = 4;
              return loadAllEconomicData(setEconomicData, setDataLoadingStatus);
            case 4:
              loadedData = _context20.v;
              setDataLoadingStatus({
                status: 'Cargando calendario económico...',
                progress: 92
              });
              _context20.n = 5;
              return fetchEconomicCalendar();
            case 5:
              calendarResult = _context20.v;
              setEconomicCalendar(calendarResult);
              _context20.n = 6;
              return generateHistoricalData(loadedData, setDataLoadingStatus);
            case 6:
              historical = _context20.v;
              setHistoricalData(historical);
              setDataLoadingStatus({
                status: 'Cargando análisis AI (Groq)...',
                progress: 96
              });
              _context20.n = 7;
              return fetchAIAnalysis();
            case 7:
              aiData = _context20.v;
              if (aiData && Object.keys(aiData).length > 0) {
                setAiAnalyses(aiData);
                setAiAnalysisReady(true);
                // M-01 FIX: inyectar lastRateDecision en economicData para que
                // determineCentralBankOutlook pueda usarlo en el cálculo del outlookScore
                setEconomicData(function (prev) {
                  var updated = _objectSpread({}, prev);
                  Object.entries(aiData).forEach(function (_ref39) {
                    var _ref40 = _slicedToArray(_ref39, 2),
                      currency = _ref40[0],
                      analysis = _ref40[1];
                    if (updated[currency] && analysis.lastRateDecision) {
                      updated[currency] = _objectSpread(_objectSpread({}, updated[currency]), {}, {
                        _aiLastRateDecision: analysis.lastRateDecision
                      });
                    }
                  });
                  return updated;
                });
              }
              setDataLoadingStatus({
                status: 'Generando recomendaciones de pares...',
                progress: 98
              });
              // FIX Bug 2: Guard de condición de carrera — solo generar recomendaciones
              // si _precomputedScores tiene datos de las 8 divisas. Si fetchStrengthScores()
              // aún no terminó o devolvió payload incompleto, sortedCurrencies queda vacío
              // y las tarjetas de Divergencias/Spreads no se renderizan.
              if (_precomputedScores && Object.keys(_precomputedScores).length >= 8) {
                pairRecommendations = generateForexPairRecommendations(loadedData, rates, (calendarResult === null || calendarResult === void 0 ? void 0 : calendarResult.events) || null);
                setDynamicAlerts(pairRecommendations);
              } else {
                // Scores aún incompletos: esperar a que fetchStrengthScores termine
                // y volver a intentar con un pequeño delay para no bloquear el render.
                console.warn('[pairRec] _precomputedScores incompleto (' + (_precomputedScores ? Object.keys(_precomputedScores).length : 0) + '/8) — reintentando en 2s');
                setTimeout(function () {
                  if (_precomputedScores && Object.keys(_precomputedScores).length >= 8) {
                    var retryRec = generateForexPairRecommendations(loadedData, rates, (calendarResult === null || calendarResult === void 0 ? void 0 : calendarResult.events) || null);
                    setDynamicAlerts(retryRec);
                  } else {
                    // Último intento con los scores disponibles (aunque sean parciales)
                    var fallbackRec = generateForexPairRecommendations(loadedData, rates, (calendarResult === null || calendarResult === void 0 ? void 0 : calendarResult.events) || null);
                    if (fallbackRec && fallbackRec.length > 0) setDynamicAlerts(fallbackRec);
                  }
                }, 2000);
              }
              setDataLoadingStatus({
                status: 'Completado',
                progress: 100
              });
              clearTimeout(globalTimeout);
              setIsLoading(false);
              _context20.n = 9;
              break;
            case 8:
              _context20.p = 8;
              _t15 = _context20.v;
              console.error('Error loading initial data:', _t15);
              clearTimeout(globalTimeout);
              setDataLoadingStatus({
                status: "Error al cargar datos: ".concat(_t15.message, ". Intenta limpiar el cach\xE9."),
                progress: 100
              });
              setIsLoading(false);
            case 9:
              return _context20.a(2);
          }
        }, _callee20, null, [[2, 8]]);
      }));
      return function loadInitialData() {
        return _ref36.apply(this, arguments);
      };
    }();
    loadInitialData();
    var ratesInterval = setInterval(/*#__PURE__*/_asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee21() {
      var rates;
      return _regenerator().w(function (_context21) {
        while (1) switch (_context21.n) {
          case 0:
            _context21.n = 1;
            return fetchForexRates();
          case 1:
            rates = _context21.v;
            setForexRates(rates);
            setLastUpdate(new Date());
          case 2:
            return _context21.a(2);
        }
      }, _callee21);
    })), 60000);
    return function () {
      return clearInterval(ratesInterval);
    };
  }, []);
  useEffect(function () {
    if (activeTab === 'trends' && Object.keys(historicalData).length > 0) {
      var timer = setTimeout(function () {
        createCharts();
      }, 200);
      return function () {
        return clearTimeout(timer);
      };
    }
    return function () {
      Object.values(chartRefs.current).forEach(function (chart) {
        if (chart) chart.destroy();
      });
      chartRefs.current = {};
    };
  }, [activeTab, historicalData]);
  var DataHealthCheck = function DataHealthCheck(_ref42) {
    var _health$issues, _health$warnings;
    var economicData = _ref42.economicData;
    var _useState27 = useState(null),
      _useState28 = _slicedToArray(_useState27, 2),
      health = _useState28[0],
      setHealth = _useState28[1];
    // D-01 FIX: estado del health check de tasas de interés
    var _useState29 = useState(null),
      _useState30 = _slicedToArray(_useState29, 2),
      ratesHealth = _useState30[0],
      setRatesHealth = _useState30[1];
    useEffect(function () {
      if (Object.keys(economicData).length > 0) {
        var healthStatus = checkDataHealth(economicData);
        setHealth(healthStatus);
      }
    }, [economicData]);

    // D-01 FIX: cargar rates/health.json para detectar problemas de scraping
    useEffect(function () {
      var fetchRatesHealth = /*#__PURE__*/function () {
        var _ref43 = _asyncToGenerator(/*#__PURE__*/_regenerator().m(function _callee22() {
          var url, r, data, silentStatuses, problematicCurrencies, _t16;
          return _regenerator().w(function (_context22) {
            while (1) switch (_context22.p = _context22.n) {
              case 0:
                _context22.p = 0;
                url = 'https://globalinvesting.github.io/rates/health.json';
                _context22.n = 1;
                return fetch(url, {
                  cache: 'no-cache',
                  mode: 'cors'
                });
              case 1:
                r = _context22.v;
                if (r.ok) {
                  _context22.n = 2;
                  break;
                }
                return _context22.a(2);
              case 2:
                _context22.n = 3;
                return r.json();
              case 3:
                data = _context22.v;
                // 'ok' y 'degraded_cached' no requieren alerta — datos disponibles y válidos
                silentStatuses = ['ok', 'degraded_cached'];
                if (!(silentStatuses.includes(data.overallStatus) && !(data.issues && data.issues.length > 0))) {
                  _context22.n = 4;
                  break;
                }
                return _context22.a(2);
              case 4:
                // Alertar solo si hay divisas verdaderamente sin datos (missing) o con issues
                problematicCurrencies = Object.entries(data.currencies || {}).filter(function (_ref44) {
                  var _ref45 = _slicedToArray(_ref44, 2),
                    info = _ref45[1];
                  return info.status === 'fallback' || info.status === 'missing';
                }).map(function (_ref46) {
                  var _ref47 = _slicedToArray(_ref46, 1),
                    ccy = _ref47[0];
                  return ccy;
                });
                if (problematicCurrencies.length > 0 || data.issues && data.issues.length > 0) {
                  setRatesHealth(data);
                }
                _context22.n = 6;
                break;
              case 5:
                _context22.p = 5;
                _t16 = _context22.v;
              case 6:
                return _context22.a(2);
            }
          }, _callee22, null, [[0, 5]]);
        }));
        return function fetchRatesHealth() {
          return _ref43.apply(this, arguments);
        };
      }();
      fetchRatesHealth();
    }, []);
    if (!health && !ratesHealth) return null;
    if (health !== null && health !== void 0 && health.healthy && health.warnings.length === 0 && !ratesHealth) return null;

    // Preparar alertas de tasas de interés
    var ratesIssues = (ratesHealth === null || ratesHealth === void 0 ? void 0 : ratesHealth.issues) || [];
    var fallbackCcys = Object.entries((ratesHealth === null || ratesHealth === void 0 ? void 0 : ratesHealth.currencies) || {}).filter(function (_ref48) {
      var _ref49 = _slicedToArray(_ref48, 2),
        info = _ref49[1];
      return info.status === 'fallback';
    }).map(function (_ref50) {
      var _ref51 = _slicedToArray(_ref50, 2),
        ccy = _ref51[0],
        info = _ref51[1];
      return "".concat(ccy, " (fuente alternativa: ").concat(info.source, ")");
    });
    var missingCcys = Object.entries((ratesHealth === null || ratesHealth === void 0 ? void 0 : ratesHealth.currencies) || {}).filter(function (_ref52) {
      var _ref53 = _slicedToArray(_ref52, 2),
        info = _ref53[1];
      return info.status === 'missing';
    }).map(function (_ref54) {
      var _ref55 = _slicedToArray(_ref54, 1),
        ccy = _ref55[0];
      return ccy;
    });
    return /*#__PURE__*/React.createElement("div", {
      style: {
        marginBottom: '1.5rem'
      }
    }, ratesHealth && (ratesIssues.length > 0 || fallbackCcys.length > 0 || missingCcys.length > 0) && /*#__PURE__*/React.createElement("div", {
      className: "alert-card alert-warning",
      style: {
        marginBottom: '1rem'
      }
    }, /*#__PURE__*/React.createElement("div", {
      className: "alert-content"
    }, /*#__PURE__*/React.createElement("h3", null, "Alerta: Fiabilidad de Tasas de Inter\xE9s"), /*#__PURE__*/React.createElement("ul", {
      style: {
        marginTop: '0.75rem',
        paddingLeft: '1.5rem',
        fontSize: '0.875rem',
        lineHeight: '1.8'
      }
    }, ratesIssues.map(function (issue, i) {
      return /*#__PURE__*/React.createElement("li", {
        key: "ri-".concat(i)
      }, issue);
    }), fallbackCcys.length > 0 && /*#__PURE__*/React.createElement("li", null, "Las siguientes divisas usan la fuente secundaria de tasas (la fuente primaria no estaba disponible):", ' ', /*#__PURE__*/React.createElement("strong", null, fallbackCcys.join(', ')), ". Los valores pueden tener mayor latencia."), missingCcys.length > 0 && /*#__PURE__*/React.createElement("li", null, "Sin datos de tasas para: ", /*#__PURE__*/React.createElement("strong", null, missingCcys.join(', ')), ".", ' ', "El score de estas divisas puede no ser fiable.")), /*#__PURE__*/React.createElement("p", {
      style: {
        marginTop: '0.75rem',
        fontSize: '0.8125rem',
        color: 'var(--text-secondary)'
      }
    }, "\xDAltima actualizaci\xF3n del scraper: ", ratesHealth.runDate || 'desconocida', ".", ' ', "El sistema intentar\xE1 actualizarse autom\xE1ticamente en el pr\xF3ximo ciclo."))), (health === null || health === void 0 || (_health$issues = health.issues) === null || _health$issues === void 0 ? void 0 : _health$issues.length) > 0 && /*#__PURE__*/React.createElement("div", {
      className: "alert-card alert-warning",
      style: {
        marginBottom: '1rem'
      }
    }, /*#__PURE__*/React.createElement("div", {
      className: "alert-content"
    }, /*#__PURE__*/React.createElement("h3", null, "Problemas de Calidad de Datos Detectados"), /*#__PURE__*/React.createElement("ul", {
      style: {
        marginTop: '0.75rem',
        paddingLeft: '1.5rem',
        fontSize: '0.875rem',
        lineHeight: '1.8'
      }
    }, health.issues.map(function (issue, i) {
      return /*#__PURE__*/React.createElement("li", {
        key: i
      }, issue);
    })), /*#__PURE__*/React.createElement("p", {
      style: {
        marginTop: '0.75rem',
        fontSize: '0.8125rem',
        color: 'var(--text-secondary)'
      }
    }, "Los datos pueden estar desactualizados o incompletos. Considera refrescar el cach\xE9 o esperar a la pr\xF3xima actualizaci\xF3n autom\xE1tica."))), (health === null || health === void 0 || (_health$warnings = health.warnings) === null || _health$warnings === void 0 ? void 0 : _health$warnings.length) > 0 && /*#__PURE__*/React.createElement("div", {
      className: "alert-card alert-info",
      style: {
        marginBottom: '1rem'
      }
    }, /*#__PURE__*/React.createElement("div", {
      className: "alert-content"
    }, /*#__PURE__*/React.createElement("h3", null, "Avisos de Calidad de Datos"), /*#__PURE__*/React.createElement("ul", {
      style: {
        marginTop: '0.75rem',
        paddingLeft: '1.5rem',
        fontSize: '0.875rem',
        lineHeight: '1.8'
      }
    }, health.warnings.map(function (warning, i) {
      return /*#__PURE__*/React.createElement("li", {
        key: i
      }, warning);
    })))));
  };
  if (isLoading) {
    return /*#__PURE__*/React.createElement("div", {
      className: "app-container"
    }, /*#__PURE__*/React.createElement("div", {
      className: "header"
    }, /*#__PURE__*/React.createElement("div", {
      className: "header-content"
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        display: "flex",
        flexDirection: "row",
        alignItems: "center"
      }
    }, /*#__PURE__*/React.createElement("img", {
      src: "apple-touch-icon.png",
      alt: "Global Investing",
      style: {
        width: "40px",
        height: "40px",
        borderRadius: "8px",
        marginRight: "0.75rem",
        flexShrink: 0,
        display: "block"
      }
    }), /*#__PURE__*/React.createElement("div", {
      style: {
        display: "flex",
        flexDirection: "column",
        justifyContent: "center"
      }
    }, /*#__PURE__*/React.createElement("div", {
      className: "brand-title"
    }, "Global Investing"), /*#__PURE__*/React.createElement("div", {
      className: "brand-subtitle"
    }, "Forex Fundamental Dashboard"))), /*#__PURE__*/React.createElement("div", {
      className: "header-info"
    }, /*#__PURE__*/React.createElement("div", {
      className: "info-item"
    }, /*#__PURE__*/React.createElement("div", {
      className: "status-loading"
    }), /*#__PURE__*/React.createElement("span", null, "Cargando Datos"))))), /*#__PURE__*/React.createElement("div", {
      className: "content"
    }, /*#__PURE__*/React.createElement("div", {
      className: "loading-container"
    }, /*#__PURE__*/React.createElement("div", {
      className: "loading-spinner"
    }), /*#__PURE__*/React.createElement("div", {
      className: "loading-text"
    }, dataLoadingStatus.status), /*#__PURE__*/React.createElement("div", {
      className: "loading-progress"
    }, /*#__PURE__*/React.createElement("div", {
      className: "loading-progress-bar",
      style: {
        width: "".concat(dataLoadingStatus.progress, "%")
      }
    })))));
  }
  return /*#__PURE__*/React.createElement("div", {
    className: "app-container"
  }, /*#__PURE__*/React.createElement("div", {
    className: "header"
  }, /*#__PURE__*/React.createElement("div", {
    className: "header-content"
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      flexDirection: "row",
      alignItems: "center"
    }
  }, /*#__PURE__*/React.createElement("img", {
    src: "apple-touch-icon.png",
    alt: "Global Investing",
    style: {
      width: "40px",
      height: "40px",
      borderRadius: "8px",
      marginRight: "0.75rem",
      flexShrink: 0,
      display: "block"
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      flexDirection: "column",
      justifyContent: "center"
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "brand-title"
  }, "Global Investing"), /*#__PURE__*/React.createElement("div", {
    className: "brand-subtitle"
  }, "Forex Fundamental Dashboard"))), /*#__PURE__*/React.createElement("div", {
    className: "page-nav-links"
  }, /*#__PURE__*/React.createElement("a", {
    href: "about.html"
  }, "Acerca de"), /*#__PURE__*/React.createElement("a", {
    href: "publicidad.html"
  }, "Publicidad"), /*#__PURE__*/React.createElement("div", {
    className: "nav-dropdown"
  }, /*#__PURE__*/React.createElement("a", {
    href: "guia-score-fortaleza.html"
  }, "Gu\xEDas"), /*#__PURE__*/React.createElement("div", {
    className: "nav-dropdown-menu"
  }, /*#__PURE__*/React.createElement("a", {
    href: "guia-score-fortaleza.html"
  }, "Score de Fortaleza"), /*#__PURE__*/React.createElement("a", {
    href: "guia-carry-trade.html"
  }, "Carry Trade"), /*#__PURE__*/React.createElement("a", {
    href: "guia-cot.html"
  }, "Reporte COT"), /*#__PURE__*/React.createElement("a", {
    href: "guia-bancos-centrales.html"
  }, "Bancos Centrales"), /*#__PURE__*/React.createElement("a", {
    href: "guia-calendario-economico.html"
  }, "Calendario Econ\xF3mico"), /*#__PURE__*/React.createElement("a", {
    href: "guia-pips.html"
  }, "Pips y Lotaje"), /*#__PURE__*/React.createElement("a", {
    href: "tecnico-vs-fundamental.html"
  }, "T\xE9cnico vs Fundamental"), /*#__PURE__*/React.createElement("a", {
    href: "glosario-forex.html"
  }, "Glosario Forex"))), /*#__PURE__*/React.createElement("a", {
    href: "contact.html"
  }, "Contacto"), /*#__PURE__*/React.createElement("a", {
    href: "en.html",
    className: "lang-switch"
  }, "EN")), /*#__PURE__*/React.createElement("div", {
    className: "header-info"
  }, lastUpdate && /*#__PURE__*/React.createElement("div", {
    className: "info-item"
  }, /*#__PURE__*/React.createElement("div", {
    className: "status-live"
  }), /*#__PURE__*/React.createElement("span", null, "Actualizado ", /*#__PURE__*/React.createElement("strong", null, lastUpdate.toLocaleTimeString('es-ES', {
    hour: '2-digit',
    minute: '2-digit'
  })), " (hora local)")), _scoresDataDate && function () {
    var d = new Date(_scoresDataDate);
    // Compare by calendar day to avoid false positives: date-only strings like
    // "2026-03-12" are parsed as midnight UTC, which appears >20h old for UTC-3 users.
    var today = new Date();
    var dataDay = new Date(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
    var todayDay = new Date(today.getFullYear(), today.getMonth(), today.getDate());
    var isStale = todayDay - dataDay >= 2 * 24 * 3600 * 1000; // 2+ calendar days old
    if (!isStale) return null;
    var label = d.toLocaleDateString('es-ES', {
      day: '2-digit',
      month: 'short'
    });
    return /*#__PURE__*/React.createElement("div", {
      className: "info-item",
      title: "El workflow de scores no ha corrido hoy. Mostrando el \xFAltimo snapshot disponible."
    }, /*#__PURE__*/React.createElement("span", {
      style: {
        background: '#f59e0b',
        color: '#000',
        borderRadius: '4px',
        padding: '2px 7px',
        fontSize: '11px',
        fontWeight: 600
      }
    }, "Scores: datos del ", label));
  }()))), /*#__PURE__*/React.createElement("div", {
    className: "nav-tabs-wrap"
  }, /*#__PURE__*/React.createElement("div", {
    className: "nav-tabs"
  }, /*#__PURE__*/React.createElement("button", {
    className: "nav-tab ".concat(activeTab === 'overview' ? 'active' : ''),
    onClick: function onClick() {
      return setActiveTab('overview');
    }
  }, "Resumen de Divisas"), /*#__PURE__*/React.createElement("button", {
    className: "nav-tab ".concat(activeTab === 'heatmap' ? 'active' : ''),
    onClick: function onClick() {
      return setActiveTab('heatmap');
    }
  }, "Mapa de Calor"), /*#__PURE__*/React.createElement("button", {
    className: "nav-tab ".concat(activeTab === 'trends' ? 'active' : ''),
    onClick: function onClick() {
      return setActiveTab('trends');
    }
  }, "Tendencias Hist\xF3ricas"), /*#__PURE__*/React.createElement("button", {
    className: "nav-tab ".concat(activeTab === 'calendar' ? 'active' : ''),
    onClick: function onClick() {
      return setActiveTab('calendar');
    }
  }, "Calendario Econ\xF3mico"), /*#__PURE__*/React.createElement("button", {
    className: "nav-tab ".concat(activeTab === 'alerts' ? 'active' : ''),
    onClick: function onClick() {
      return setActiveTab('alerts');
    }
  }, "Mejores Pares"), /*#__PURE__*/React.createElement("a", {
    href: "news.html",
    className: "nav-tab",
    style: {
      textDecoration: 'none'
    }
  }, "Noticias"), /*#__PURE__*/React.createElement("a", {
    href: "carry-trade.html",
    className: "nav-tab",
    style: {
      textDecoration: 'none'
    }
  }, "Carry Trade"))), /*#__PURE__*/React.createElement("div", {
    className: "content"
  }, activeTab === 'overview' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(DataHealthCheck, {
    economicData: economicData
  }), /*#__PURE__*/React.createElement("div", {
    className: "section-header"
  }, /*#__PURE__*/React.createElement("div", {
    className: "section-title"
  }, "Resumen de Divisas y An\xE1lisis Fundamental"), /*#__PURE__*/React.createElement("div", {
    className: "section-meta"
  }, "Ordenadas por fortaleza fundamental \xB7 Horizonte 2-6 semanas \xB7 Actualizado ", lastUpdate ? lastUpdate.toLocaleDateString('es-ES') : '')), /*#__PURE__*/React.createElement("div", {
    "aria-live": "polite",
    "aria-atomic": "false",
    "aria-relevant": "text",
    style: {
      position: 'absolute',
      width: '1px',
      height: '1px',
      padding: 0,
      margin: '-1px',
      overflow: 'hidden',
      clip: 'rect(0,0,0,0)',
      whiteSpace: 'nowrap',
      border: 0
    }
  }, !isLoading && Object.keys(economicData).length > 0 ? "Scores actualizados. Ranking actual: ".concat(getSortedCountries().map(function (c) {
    return "".concat(c.code, " ").concat(Math.round(getStrength(c.code).score));
  }).join(', '), ".") : isLoading ? 'Cargando datos de fortaleza de divisas...' : ''), Object.keys(economicData).length === 0 ? /*#__PURE__*/React.createElement("div", {
    style: {
      textAlign: 'center',
      padding: '4rem',
      color: 'var(--text-secondary)'
    }
  }, "Cargando datos econ\xF3micos...") : /*#__PURE__*/React.createElement("div", {
    className: "currency-grid"
  }, getSortedCountries().map(function (country) {
    var _data$interestRate2;
    var data = economicData[country.code] || {};
    var strengthObj = getStrength(country.code);
    var strength = strengthObj.score;
    var sentiment = getSentiment(strength);
    var rate = country.code === 'USD' ? 1 : forexRates[country.code] || 1;
    // calDates: fechas de los overrides del calendario. Cuando el backend actualizó
    // un dato con valores del calendario, la fecha real del dato es calDates[campo],
    // no la fecha de economic-data. Se usa en los tooltips del heatmap.
    var calDates = (strengthObj.scoringData || {}).calendarDates || {};

    // FIX-5 v5.11: Score context badge — 100% data-driven, sin hardcoding de divisas.
    // Activa badge "Score estructural" cuando el score es moderado/alto PERO las
    // señales de momentum son débiles (tasa baja + ciclo bajista).
    // Esto avisa al usuario que el score refleja solidez de balance sheet,
    // no momentum operativo a 2-6 semanas.
    var scoreContextBadge = function (_data$interestRate, _data$rateMomentum, _data$fxPerformance1M, _data$cotPositioning) {
      var ir = (_data$interestRate = data.interestRate) !== null && _data$interestRate !== void 0 ? _data$interestRate : null;
      var rm = (_data$rateMomentum = data.rateMomentum) !== null && _data$rateMomentum !== void 0 ? _data$rateMomentum : null;
      var fx = (_data$fxPerformance1M = data.fxPerformance1M) !== null && _data$fxPerformance1M !== void 0 ? _data$fxPerformance1M : null;
      var cot = (_data$cotPositioning = data.cotPositioning) !== null && _data$cotPositioning !== void 0 ? _data$cotPositioning : null;
      // Criterio 1: score relevante (no es un outlier bajista claro)
      var scoreRelevant = strength >= 55; // v6.0: badge solo activa en scores medio-altos (>=55), no en outliers bajistas
      // Criterio 2: señales de momentum monetario débiles
      // Tasa ≤ 0.5% (zero-bound) O ciclo de recortes ≤ −1.0pp
      var weakMonetary = ir !== null && ir <= 0.5 || rm !== null && rm <= -1.0;
      // Criterio 3: momento de precio también débil (FX y/o COT no confirman)
      var weakMomentum = fx !== null && fx < 0.5 && (cot === null || cot < 20000);
      if (scoreRelevant && weakMonetary && weakMomentum) {
        return {
          type: 'structural',
          label: 'Estructural',
          title: 'Score elevado por indicadores macroeconómicos estructurales (CA, deuda/PIB) — señal de momentum operativo débil: tasa baja o en ciclo de recortes. El score refleja solidez de balance sheet, no divergencia de momentum a 2–6 semanas.'
        };
      }
      // Badge momentum: confirmación positiva cruzada (FX + COT alineados)
      var strongMomentum = fx !== null && fx > 1.5 && cot !== null && cot > 15000;
      if (scoreRelevant && strongMomentum && !weakMonetary) {
        var fxStr = fx !== null ? "FX +".concat(fx.toFixed(1), "% (1M)") : 'FX positivo';
        var cotStr = cot !== null ? "COT largo neto ".concat(cot > 0 ? '+' : '').concat(Math.round(cot / 1000), "K") : 'COT confirmado';
        return {
          type: 'momentum',
          label: 'Momentum',
          title: "".concat(fxStr, " \xB7 ").concat(cotStr, " \u2014 se\xF1al de momentum operativo confirmada.")
        };
      }
      return null;
    }();

    // Compact date formatter — same logic as heatmap tooltips
    var fmtD = function fmtD(ds) {
      if (!ds) return null;
      try {
        var d = new Date(ds);
        var diff = Math.ceil((new Date() - d) / 86400000);
        if (diff <= 0) return 'Hoy';
        if (diff === 1) return 'Ayer';
        if (diff <= 7) return "".concat(diff, "d");
        if (diff <= 60) return "".concat(Math.floor(diff / 7), "sem");
        return d.toLocaleDateString('es-ES', {
          month: 'short',
          year: '2-digit'
        });
      } catch (_unused3) {
        return null;
      }
    };
    // titleWithDate: appends date to native title tooltip if available
    var td = function td(label, dateStr) {
      var d = fmtD(dateStr);
      return d ? "".concat(label, " \xB7 Dato: ").concat(d) : label;
    };
    return /*#__PURE__*/React.createElement("div", {
      key: country.code,
      className: "currency-card"
    }, /*#__PURE__*/React.createElement("div", {
      className: "card-header"
    }, /*#__PURE__*/React.createElement("div", {
      className: "card-title-section"
    }, /*#__PURE__*/React.createElement("span", {
      className: "card-flag flag-emoji"
    }, country.flag), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
      className: "card-currency-code"
    }, country.code), /*#__PURE__*/React.createElement("div", {
      className: "card-currency-name"
    }, country.currency))), /*#__PURE__*/React.createElement("div", {
      className: "card-strength"
    }, /*#__PURE__*/React.createElement("div", {
      className: "strength-score",
      style: {
        color: sentiment === 'alcista' ? 'var(--green-strong)' : sentiment === 'bajista' ? 'var(--red-strong)' : 'var(--neutral)'
      }
    }, strength), /*#__PURE__*/React.createElement("div", {
      className: "strength-label"
    }, "Fortaleza"), scoreContextBadge && /*#__PURE__*/React.createElement("div", {
      className: "score-context-badge ".concat(scoreContextBadge.type),
      title: scoreContextBadge.title
    }, scoreContextBadge.label))), /*#__PURE__*/React.createElement("div", {
      className: "card-body"
    }, /*#__PURE__*/React.createElement("div", {
      className: "central-bank"
    }, /*#__PURE__*/React.createElement("div", {
      className: "bank-label"
    }, "Banco Central"), /*#__PURE__*/React.createElement("div", {
      className: "bank-name"
    }, country.centralBank), /*#__PURE__*/React.createElement("div", {
      className: "bank-info"
    }, /*#__PURE__*/React.createElement("div", {
      className: "bank-info-item"
    }, /*#__PURE__*/React.createElement("div", {
      className: "bank-info-label"
    }, "Tasa"), /*#__PURE__*/React.createElement("div", {
      className: "bank-info-value"
    }, (_data$interestRate2 = data.interestRate) === null || _data$interestRate2 === void 0 ? void 0 : _data$interestRate2.toFixed(2), "%")), /*#__PURE__*/React.createElement("div", {
      className: "bank-info-item"
    }, /*#__PURE__*/React.createElement("div", {
      className: "bank-info-label"
    }, "Pr\xF3xima Reuni\xF3n"), /*#__PURE__*/React.createElement("div", {
      className: "bank-info-value"
    }, data.nextMeeting || country.nextMeeting)), /*#__PURE__*/React.createElement("div", {
      className: "bank-info-item"
    }, /*#__PURE__*/React.createElement("div", {
      className: "bank-info-label"
    }, "Perspectiva"), /*#__PURE__*/React.createElement("div", {
      className: "bank-info-value"
    }, data.outlook || country.outlook)))), /*#__PURE__*/React.createElement("div", {
      className: "metrics-grid"
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('Crecimiento PIB anual', data.gdpGrowthDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "Crecimiento PIB"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value",
      style: {
        color: data.gdpGrowth !== null && data.gdpGrowth > 1.8 ? 'var(--green-strong)' : data.gdpGrowth !== null && data.gdpGrowth < 1.0 ? 'var(--red-strong)' : 'var(--text-primary)'
      }
    }, formatValue(data.gdpGrowth), "%")), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('Inflación IPC', data.inflationDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "Inflaci\xF3n"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value",
      style: {
        color: data.inflation !== null && data.inflation >= 1.8 && data.inflation <= 2.5 ? 'var(--green-strong)' : data.inflation !== null && data.inflation > 3.5 ? 'var(--red-strong)' : 'var(--text-primary)'
      }
    }, formatValue(data.inflation), "%")), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('Tasa de desempleo', data.unemploymentDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "Desempleo"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value",
      style: {
        color: data.unemployment !== null && data.unemployment < 3.5 ? 'var(--green-strong)' : data.unemployment !== null && data.unemployment > 6 ? 'var(--red-strong)' : 'var(--text-primary)'
      }
    }, formatValue(data.unemployment), "%")), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('Cuenta Corriente % del PIB', data.currentAccountDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "Cuenta Corriente"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value metric-value-small",
      style: {
        color: data.currentAccount !== null && data.currentAccount > 2 ? 'var(--green-strong)' : data.currentAccount !== null && data.currentAccount < -3 ? 'var(--red-strong)' : 'var(--text-primary)'
      }
    }, data.currentAccount !== null && data.currentAccount !== undefined ? "".concat(data.currentAccount > 0 ? '+' : '').concat(formatValue(data.currentAccount), "%") : 'N/D')), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('Balanza Comercial en miles de millones USD', data.tradeBalanceDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "Balanza Comercial"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value metric-value-small",
      style: {
        color: data.tradeBalance !== null && data.tradeBalance > 1000 ? 'var(--green-strong)' : data.tradeBalance !== null && data.tradeBalance < -10000 ? 'var(--red-strong)' : 'var(--text-primary)'
      }
    }, data.tradeBalance !== null && data.tradeBalance !== undefined ? "".concat(data.tradeBalance > 0 ? '+' : '').concat((data.tradeBalance / 1000).toFixed(1), "B") : 'N/D')), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('Ventas Minoristas MoM', data.retailSalesDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "Ventas Minoristas"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value metric-value-small",
      style: {
        color: data.retailSales !== null && data.retailSales > 0.5 ? 'var(--green-strong)' : data.retailSales !== null && data.retailSales < -0.5 ? 'var(--red-strong)' : 'var(--text-primary)'
      }
    }, data.retailSales !== null && data.retailSales !== undefined ? "".concat(data.retailSales > 0 ? '+' : '').concat(formatValue(data.retailSales), "%") : 'N/D')), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('Crecimiento Salarial YoY', data.wageGrowthDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "Crecimiento Salarial"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value metric-value-small",
      style: {
        color: data.wageGrowth !== null && data.wageGrowth >= 2.5 && data.wageGrowth <= 4.5 ? 'var(--green-strong)' : data.wageGrowth !== null && data.wageGrowth > 5.5 ? 'var(--orange)' : 'var(--text-primary)'
      }
    }, formatValue(data.wageGrowth), "%")), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('PMI Manufacturero — >50 expansión', data.manufacturingPMIDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "PMI Manufacturero"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value metric-value-small",
      style: {
        color: data.manufacturingPMI !== null && data.manufacturingPMI > 50 ? 'var(--green-strong)' : data.manufacturingPMI !== null && data.manufacturingPMI < 50 ? 'var(--red-strong)' : 'var(--text-primary)'
      }
    }, formatValue(data.manufacturingPMI, 1))), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('COT Positioning — contratos netos institucionales (CFTC)', data.cotDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "COT Positioning"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value metric-value-small",
      style: {
        color: data.cotPositioning !== null && data.cotPositioning > 10000 ? 'var(--green-strong)' : data.cotPositioning !== null && data.cotPositioning < -10000 ? 'var(--red-strong)' : data.cotPositioning !== null ? 'var(--text-primary)' : 'var(--text-tertiary)'
      }
    }, data.cotPositioning !== null && data.cotPositioning !== undefined ? "".concat(data.cotPositioning > 0 ? '+' : '').concat((data.cotPositioning / 1000).toFixed(1), "K") : 'N/A')), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('Yield Bono Soberano 10 años', data.bond10yDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "Bono 10Y"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value metric-value-small",
      style: {
        color: data.bond10y !== null && data.bond10y > 4.0 ? 'var(--green-strong)' : data.bond10y !== null && data.bond10y < 1.5 ? 'var(--red-strong)' : 'var(--text-primary)'
      }
    }, data.bond10y !== null && data.bond10y !== undefined ? "".concat(formatValue(data.bond10y, 2), "%") : 'N/D')), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('Confianza del Consumidor', data.consumerConfidenceDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "Conf. Consumidor"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value metric-value-small",
      style: {
        color: data.consumerConfidence !== null && data.consumerConfidence > 100 ? 'var(--green-strong)' : data.consumerConfidence !== null && data.consumerConfidence < 95 ? 'var(--red-strong)' : 'var(--text-primary)'
      }
    }, formatValue(data.consumerConfidence, 1))), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('Confianza Empresarial', data.businessConfidenceDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "Conf. Empresarial"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value metric-value-small",
      style: {
        color: data.businessConfidence !== null && data.businessConfidence > 100 ? 'var(--green-strong)' : data.businessConfidence !== null && data.businessConfidence < 95 ? 'var(--red-strong)' : 'var(--text-primary)'
      }
    }, formatValue(data.businessConfidence, 1))), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('Expectativas de Inflación a 5 años', data.inflationExpectationsDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "Exp. Inflaci\xF3n"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value metric-value-small",
      style: {
        color: data.inflationExpectations !== null && data.inflationExpectations >= 1.8 && data.inflationExpectations <= 2.5 ? 'var(--green-strong)' : data.inflationExpectations !== null && data.inflationExpectations > 3.5 ? 'var(--red-strong)' : 'var(--text-primary)'
      }
    }, data.inflationExpectations !== null && data.inflationExpectations !== undefined ? "".concat(formatValue(data.inflationExpectations, 1), "%") : 'N/D')), /*#__PURE__*/React.createElement("div", {
      className: "metric",
      title: td('Términos de Intercambio (índice base 100)', data.termsOfTradeDate)
    }, /*#__PURE__*/React.createElement("div", {
      className: "metric-label"
    }, "T\xE9rminos Intercambio"), /*#__PURE__*/React.createElement("div", {
      className: "metric-value metric-value-small",
      style: {
        color: data.termsOfTrade !== null && data.termsOfTrade >= 103 ? 'var(--green-strong)' : data.termsOfTrade !== null && data.termsOfTrade < 97 ? 'var(--red-strong)' : 'var(--text-primary)'
      }
    }, data.termsOfTrade !== null && data.termsOfTrade !== undefined ? formatValue(data.termsOfTrade, 1) : 'N/D'))), /*#__PURE__*/React.createElement("div", {
      className: "analysis-section"
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '0.75rem'
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem'
      }
    }, /*#__PURE__*/React.createElement("div", {
      className: "analysis-title",
      style: {
        marginBottom: 0
      }
    }, "An\xE1lisis Fundamental")), /*#__PURE__*/React.createElement("div", {
      className: "sentiment-badge sentiment-".concat(sentiment)
    }, /*#__PURE__*/React.createElement("div", {
      className: "sentiment-indicator",
      style: {
        background: sentiment === 'alcista' ? 'var(--green-strong)' : sentiment === 'bajista' ? 'var(--red-strong)' : 'var(--neutral)'
      }
    }), sentiment)), /*#__PURE__*/React.createElement("div", {
      className: "analysis-text"
    }, aiAnalysisReady && aiAnalyses && aiAnalyses[country.code] ? aiAnalyses[country.code].text : generateAnalysis(country.code, forexRates, economicData)), aiAnalysisReady && aiAnalyses && aiAnalyses[country.code] && /*#__PURE__*/React.createElement("div", {
      style: {
        marginTop: '0.6rem',
        paddingTop: '0.5rem',
        borderTop: '1px solid var(--border)',
        fontSize: '0.62rem',
        color: 'var(--text-tertiary)'
      }
    }, "Generado por Groq AI \xB7", ' ', function () {
      var ts = aiAnalyses[country.code].generatedAt;
      if (!ts) return '';
      return new Date(ts).toLocaleDateString('es-ES', {
        day: '2-digit',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit'
      });
    }()))));
  }))), activeTab === 'heatmap' && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement(DataHealthCheck, {
    economicData: economicData
  }), /*#__PURE__*/React.createElement("div", {
    className: "section-header"
  }, /*#__PURE__*/React.createElement("div", {
    className: "section-title"
  }, "Mapa de Calor de Indicadores Econ\xF3micos")), /*#__PURE__*/React.createElement("div", {
    className: "heatmap-container"
  }, /*#__PURE__*/React.createElement("table", {
    className: "heatmap-table",
    role: "grid",
    "aria-label": "Mapa de calor de indicadores econ\xF3micos \u2014 8 divisas principales, 21 indicadores"
  }, /*#__PURE__*/React.createElement("caption", {
    style: {
      position: 'absolute',
      width: '1px',
      height: '1px',
      padding: 0,
      margin: '-1px',
      overflow: 'hidden',
      clip: 'rect(0,0,0,0)',
      whiteSpace: 'nowrap',
      border: 0
    }
  }, "Indicadores macroecon\xF3micos para USD, EUR, GBP, JPY, AUD, CAD, CHF y NZD. Cada celda muestra el valor actual del indicador; el color indica fortaleza relativa (verde = favorable, rojo = desfavorable)."), /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", {
    scope: "col"
  }, "Divisa"), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "S\xEDntesis General",
    tooltip: "\xCDndice de fortaleza fundamental (0-100) basado en 21 indicadores ponderados. Horizonte de validez: 2-6 semanas. FX momentum 1M basket-adjusted act\xFAa como confirmador (8%)."
  }, "General", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "S\xEDntesis")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "PIB Total",
    tooltip: indicatorTooltips.gdp
  }, "PIB", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "Trillones USD")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Crecimiento PIB",
    tooltip: indicatorTooltips.gdpGrowth
  }, "Crecimiento PIB", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "% Anual")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Tasa de Inter\xE9s",
    tooltip: indicatorTooltips.interestRate
  }, "Tasa de Inter\xE9s", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "%")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Inflaci\xF3n",
    tooltip: indicatorTooltips.inflation
  }, "Inflaci\xF3n", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "IPC %")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Desempleo",
    tooltip: indicatorTooltips.unemployment
  }, "Desempleo", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "%")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Cuenta Corriente",
    tooltip: indicatorTooltips.currentAccount
  }, "Cuenta Corriente", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "% del PIB")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Deuda P\xFAblica",
    tooltip: indicatorTooltips.debt
  }, "Deuda P\xFAblica", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "% del PIB")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Balanza Comercial",
    tooltip: indicatorTooltips.tradeBalance
  }, "Balanza Comercial", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "Miles Millones USD")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Producci\xF3n Industrial",
    tooltip: indicatorTooltips.production
  }, "Prod. Industrial", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "% MoM / QoQ")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Ventas Minoristas",
    tooltip: indicatorTooltips.retailSales
  }, "Ventas Minoristas", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "% MoM")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Crecimiento Salarial",
    tooltip: indicatorTooltips.wageGrowth
  }, "Crecimiento Salarial", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "% YoY")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "PMI Manufacturero",
    tooltip: indicatorTooltips.manufacturingPMI
  }, "PMI Manufacturero", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "\xCDndice")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "PMI Servicios",
    tooltip: indicatorTooltips.servicesPMI
  }, "PMI Servicios", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "\xCDndice")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Posicionamiento COT",
    tooltip: indicatorTooltips.cotPositioning
  }, "COT Positioning", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "Net Contratos")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Bono 10Y",
    tooltip: indicatorTooltips.bond10y
  }, "Bono 10Y", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "Yield %")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Conf. Consumidor",
    tooltip: indicatorTooltips.consumerConfidence
  }, "Conf. Consumidor", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "\xCDndice")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Conf. Empresarial",
    tooltip: indicatorTooltips.businessConfidence
  }, "Conf. Empresarial", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "\xCDndice")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "Expectativas Inflaci\xF3n",
    tooltip: indicatorTooltips.inflationExpectations
  }, "Exp. Inflaci\xF3n", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "%")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "T\xE9rminos de Intercambio",
    tooltip: indicatorTooltips.termsOfTrade
  }, "T\xE9rminos de Intercambio", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "\xCDndice base 100")), /*#__PURE__*/React.createElement(TooltipCell, {
    title: "ESI Proxy",
    tooltip: indicatorTooltips.economicSurprise
  }, "ESI Proxy", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: '0.6rem',
      fontWeight: 400,
      color: 'var(--text-tertiary)'
    }
  }, "0-100")))), /*#__PURE__*/React.createElement("tbody", null, getSortedCountries().map(function (country) {
    var _economicSurprise;
    var data = economicData[country.code] || {};
    var strengthObj = getStrength(country.code); // ✅
    var strength = strengthObj.score; // ✅
    var sentiment = getSentiment(strength);
    var overallColor = sentiment === 'alcista' ? 'var(--green-strong)' : sentiment === 'bajista' ? 'var(--red-strong)' : 'var(--neutral)';
    var calDates = (strengthObj.scoringData || {}).calendarDates || {};
    return /*#__PURE__*/React.createElement("tr", {
      key: country.code
    }, /*#__PURE__*/React.createElement("td", {
      scope: "row"
    }, /*#__PURE__*/React.createElement("div", {
      className: "country-cell"
    }, /*#__PURE__*/React.createElement("span", {
      className: "country-flag flag-emoji"
    }, country.flag), /*#__PURE__*/React.createElement("div", {
      className: "country-info"
    }, /*#__PURE__*/React.createElement("div", {
      className: "country-name"
    }, country.name), /*#__PURE__*/React.createElement("div", {
      className: "country-currency"
    }, country.code)))), /*#__PURE__*/React.createElement("td", {
      style: {
        background: 'var(--bg-card)'
      }
    }, /*#__PURE__*/React.createElement("div", {
      className: "overall-cell"
    }, /*#__PURE__*/React.createElement("div", {
      className: "overall-indicator",
      style: {
        background: overallColor
      }
    }), /*#__PURE__*/React.createElement("span", {
      className: "overall-label",
      style: {
        color: overallColor
      }
    }, sentiment))), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.gdp,
      type: "gdp",
      currency: country.code,
      indicator: "gdp",
      lastUpdate: data.gdpDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.gdpGrowth,
      type: "gdpGrowth",
      currency: country.code,
      indicator: "gdpGrowth",
      lastUpdate: calDates.gdpGrowth || data.gdpGrowthDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.interestRate,
      type: "interestRate",
      currency: country.code,
      indicator: "interestRate",
      lastUpdate: data.interestRateDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.inflation,
      type: "inflation",
      currency: country.code,
      indicator: "inflation",
      lastUpdate: calDates.inflation || data.inflationDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.unemployment,
      type: "unemployment",
      currency: country.code,
      indicator: "unemployment",
      lastUpdate: calDates.unemployment || data.unemploymentDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.currentAccount,
      type: "currentAccount",
      currency: country.code,
      indicator: "currentAccount",
      lastUpdate: data.currentAccountDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.debt,
      type: "debt",
      currency: country.code,
      indicator: "debt",
      lastUpdate: data.debtDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.tradeBalance,
      type: "tradeBalance",
      currency: country.code,
      indicator: "tradeBalance",
      lastUpdate: data.tradeBalanceDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.production,
      type: "production",
      currency: country.code,
      indicator: "production",
      lastUpdate: calDates.production || data.productionDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.retailSales,
      type: "retailSales",
      currency: country.code,
      indicator: "retailSales",
      lastUpdate: calDates.retailSales || data.retailSalesDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.wageGrowth,
      type: "wageGrowth",
      currency: country.code,
      indicator: "wageGrowth",
      lastUpdate: calDates.wageGrowth || data.wageGrowthDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.manufacturingPMI,
      type: "manufacturingPMI",
      currency: country.code,
      indicator: "manufacturingPMI",
      lastUpdate: calDates.manufacturingPMI || data.manufacturingPMIDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.servicesPMI,
      type: "servicesPMI",
      currency: country.code,
      indicator: "servicesPMI",
      lastUpdate: calDates.servicesPMI || data.servicesPMIDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.cotPositioning,
      type: "cotPositioning",
      currency: country.code,
      indicator: "cotPositioning",
      lastUpdate: calDates.cotPositioning || data.cotPositioningDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.bond10y,
      type: "bond10y",
      currency: country.code,
      indicator: "bond10y",
      lastUpdate: data.bond10yDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.consumerConfidence,
      type: "consumerConfidence",
      currency: country.code,
      indicator: "consumerConfidence",
      lastUpdate: calDates.consumerConfidence || data.consumerConfidenceDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.businessConfidence,
      type: "businessConfidence",
      currency: country.code,
      indicator: "businessConfidence",
      lastUpdate: calDates.businessConfidence || data.businessConfidenceDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.inflationExpectations,
      type: "inflationExpectations",
      currency: country.code,
      indicator: "inflationExpectations",
      lastUpdate: data.inflationExpectationsDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: data.termsOfTrade,
      type: "termsOfTrade",
      currency: country.code,
      indicator: "termsOfTrade",
      lastUpdate: data.termsOfTradeDate
    }), /*#__PURE__*/React.createElement(HeatmapCell, {
      value: (_economicSurprise = (strengthObj.scoringData || {}).economicSurprise) !== null && _economicSurprise !== void 0 ? _economicSurprise : null,
      type: "economicSurprise",
      currency: country.code,
      indicator: "economicSurprise",
      lastUpdate: data.lastUpdate
    }));
  }))))), activeTab === 'trends' && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement(DataHealthCheck, {
    economicData: economicData
  }), /*#__PURE__*/React.createElement("div", {
    className: "section-header"
  }, /*#__PURE__*/React.createElement("div", {
    className: "section-title"
  }, "Tendencias Hist\xF3ricas y An\xE1lisis de Datos")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(min(500px, 100%), 1fr))',
      gap: '1.5rem'
    }
  }, function () {
    var FLAG_CODES = {
      USD: 'us',
      EUR: 'eu',
      GBP: 'gb',
      JPY: 'jp',
      AUD: 'au',
      CAD: 'ca',
      CHF: 'ch',
      NZD: 'nz'
    };
    return countries.map(function (country) {
      var fc = FLAG_CODES[country.code] || 'us';
      return /*#__PURE__*/React.createElement("div", {
        key: country.code,
        className: "chart-container"
      }, /*#__PURE__*/React.createElement("div", {
        style: {
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '0.5rem',
          marginBottom: '0.75rem',
          fontSize: '0.875rem',
          fontWeight: 600,
          color: '#e1e4e8'
        }
      }, /*#__PURE__*/React.createElement("img", {
        src: "https://flagcdn.com/24x18/".concat(fc, ".png"),
        srcSet: "https://flagcdn.com/48x36/".concat(fc, ".png 2x"),
        width: "24",
        height: "18",
        alt: country.code,
        style: {
          borderRadius: '2px',
          flexShrink: 0
        },
        loading: "lazy"
      }), country.code, " - Tendencias 12 Meses"), /*#__PURE__*/React.createElement("div", {
        className: "chart-wrapper"
      }, /*#__PURE__*/React.createElement("canvas", {
        id: "chart-".concat(country.code),
        role: "img",
        "aria-label": "Gr\xE1fico de tasas de inter\xE9s hist\xF3ricas para ".concat(country.name, " (").concat(country.code, "). Datos de los \xFAltimos 12 meses.")
      })), function () {
        var hd = historicalData[country.code];
        if (!hd || !hd.labels || !hd.rates) return null;
        return /*#__PURE__*/React.createElement("table", {
          style: {
            position: 'absolute',
            width: '1px',
            height: '1px',
            padding: 0,
            margin: '-1px',
            overflow: 'hidden',
            clip: 'rect(0,0,0,0)',
            whiteSpace: 'nowrap',
            border: 0
          },
          "aria-label": "Datos tabulares: tasas de inter\xE9s hist\xF3ricas ".concat(country.code)
        }, /*#__PURE__*/React.createElement("caption", {
          style: {
            position: 'absolute',
            width: '1px',
            height: '1px',
            overflow: 'hidden'
          }
        }, "Tasas de inter\xE9s hist\xF3ricas \u2014 ".concat(country.name, " (").concat(country.code, "), \xFAltimos 12 meses")), /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", {
          scope: "col"
        }, "Fecha"), /*#__PURE__*/React.createElement("th", {
          scope: "col"
        }, "Tasa de inter\xE9s (%)"))), /*#__PURE__*/React.createElement("tbody", null, hd.labels.map(function (label, i) {
          return /*#__PURE__*/React.createElement("tr", {
            key: i
          }, /*#__PURE__*/React.createElement("td", null, label), /*#__PURE__*/React.createElement("td", null, hd.rates[i] != null ? Number(hd.rates[i]).toFixed(2) : '—'));
        })));
      }());
    });
  }())), activeTab === 'calendar' && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "section-header"
  }, /*#__PURE__*/React.createElement("div", {
    className: "section-title"
  }, "Calendario Econ\xF3mico", calendarUpdating && /*#__PURE__*/React.createElement("span", {
    className: "calendar-live-dot",
    title: "Actualizando..."
  })), /*#__PURE__*/React.createElement("div", {
    className: "section-meta"
  }, economicCalendar.impactCounts && economicCalendar.impactCounts.high > 0 && /*#__PURE__*/React.createElement("span", null, economicCalendar.impactCounts.high, " alto", economicCalendar.impactCounts.medium > 0 && " \xB7 ".concat(economicCalendar.impactCounts.medium, " medio")), economicCalendar.generatedAt && /*#__PURE__*/React.createElement("span", {
    style: {
      marginLeft: '0.5rem',
      fontSize: '0.72rem',
      color: 'var(--text-tertiary)'
    }
  }, "\xB7 act. ", new Date(economicCalendar.generatedAt).toLocaleTimeString('es', {
    hour: '2-digit',
    minute: '2-digit'
  }), " UTC"))), function () {
    var allEvents = economicCalendar.events || [];

    // ── Agrupar eventos por fecha LOCAL del usuario ───────────
    // Los eventos tienen dateISO+timeUTC en UTC.
    // Convertimos cada evento a fecha local antes de agrupar,
    // para que un evento 2026-03-10T00:01Z aparezca el 9 Mar
    // en UTC-3 (Montevideo), no el 10 Mar.
    var MONTH_ES_CAL = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'];
    var getLocalDateKey = function getLocalDateKey(e) {
      var t = e.timeUTC || e.time;
      if (t && e.dateISO && /^\d{2}:\d{2}$/.test(t)) {
        var d = new Date("".concat(e.dateISO, "T").concat(t, ":00Z"));
        if (!isNaN(d.getTime())) {
          var yy = d.getFullYear();
          var mm = String(d.getMonth() + 1).padStart(2, '0');
          var dd = String(d.getDate()).padStart(2, '0');
          return "".concat(yy, "-").concat(mm, "-").concat(dd);
        }
      }
      return e.dateISO || e.date || 'Sin fecha';
    };
    var dayMap = {};
    allEvents.forEach(function (e) {
      var key = getLocalDateKey(e);
      if (!dayMap[key]) {
        var d = new Date(key + 'T12:00:00');
        var label = "".concat(d.getDate(), " ").concat(MONTH_ES_CAL[d.getMonth()]);
        dayMap[key] = {
          label: label,
          dateISO: key,
          events: []
        };
      }
      dayMap[key].events.push(e);
    });
    // Ordenar eventos dentro de cada día por hora local
    Object.values(dayMap).forEach(function (day) {
      day.events.sort(function (a, b) {
        var ta = a.timeUTC || a.time || '';
        var tb = b.timeUTC || b.time || '';
        if (!ta) return 1;
        if (!tb) return -1;
        var da = new Date("".concat(a.dateISO, "T").concat(ta, ":00Z"));
        var db = new Date("".concat(b.dateISO, "T").concat(tb, ":00Z"));
        return da - db;
      });
    });
    var days = Object.values(dayMap).sort(function (a, b) {
      return (a.dateISO || '').localeCompare(b.dateISO || '');
    });

    // Use LOCAL date (not UTC) to avoid off-by-one day across timezones
    var _now = new Date();
    var todayISO = "".concat(_now.getFullYear(), "-").concat(String(_now.getMonth() + 1).padStart(2, '0'), "-").concat(String(_now.getDate()).padStart(2, '0'));
    // defaultDay: prefer today's local date if it exists in calendar, else first future day
    var defaultDay = days.find(function (d) {
      return d.dateISO === todayISO;
    }) || days.find(function (d) {
      return d.dateISO >= todayISO;
    }) || days[days.length - 1];
    var activeDateISO = calSelectedDate || defaultDay && defaultDay.dateISO || null;
    var activeDayIdx = days.findIndex(function (d) {
      return d.dateISO === activeDateISO;
    });
    var activeDay = days[activeDayIdx] || null;

    // Week grouping: group days by ISO week (Mon-Sun) for the week selector
    var getWeekKey = function getWeekKey(iso) {
      var d = new Date(iso + 'T12:00:00Z');
      var day = d.getUTCDay() || 7;
      var monday = new Date(d);
      monday.setUTCDate(d.getUTCDate() - (day - 1));
      return monday.toISOString().split('T')[0];
    };
    var weekMap = {};
    days.forEach(function (day) {
      var wk = getWeekKey(day.dateISO);
      if (!weekMap[wk]) weekMap[wk] = [];
      weekMap[wk].push(day);
    });
    var weeks = Object.keys(weekMap).sort();
    var activeWeekKey = activeDay ? getWeekKey(activeDay.dateISO) : weeks[weeks.length - 1];
    var daysInActiveWeek = weekMap[activeWeekKey] || [];

    // Week label: "3 Mar – 7 Mar"
    var fmtWeekLabel = function fmtWeekLabel(wk) {
      var wDays = weekMap[wk];
      if (!wDays || !wDays.length) return wk;
      var first = wDays[0].label,
        last = wDays[wDays.length - 1].label;
      return first === last ? first : "".concat(first, " \u2013 ").concat(last);
    };
    var goToPrev = function goToPrev() {
      if (activeDayIdx > 0) setCalSelectedDate(days[activeDayIdx - 1].dateISO);
    };
    var goToNext = function goToNext() {
      if (activeDayIdx < days.length - 1) setCalSelectedDate(days[activeDayIdx + 1].dateISO);
    };
    var filteredEvents = activeDay ? calFilter === 'all' ? activeDay.events : activeDay.events.filter(function (e) {
      return e.impact === calFilter;
    }) : [];
    if (allEvents.length === 0) {
      return /*#__PURE__*/React.createElement("div", {
        style: {
          textAlign: 'center',
          padding: '3rem',
          color: 'var(--text-secondary)',
          background: 'var(--bg-secondary)',
          borderRadius: '6px',
          border: '1px solid var(--border)'
        }
      }, /*#__PURE__*/React.createElement("div", {
        style: {
          fontWeight: 600,
          color: 'var(--orange)',
          marginBottom: '0.75rem'
        }
      }, "Datos del calendario no disponibles"), /*#__PURE__*/React.createElement("div", {
        style: {
          fontSize: '0.875rem',
          lineHeight: '1.7',
          maxWidth: '480px',
          margin: '0 auto'
        }
      }, economicCalendar.errorMessage || 'El scraping automático no ha podido obtener datos.'), /*#__PURE__*/React.createElement("div", {
        style: {
          marginTop: '1.25rem',
          fontSize: '0.8125rem',
          color: 'var(--text-tertiary)'
        }
      }, "Consulte directamente:", ' ', /*#__PURE__*/React.createElement("a", {
        href: "https://www.forexfactory.com/calendar",
        target: "_blank",
        rel: "noopener",
        style: {
          color: 'var(--blue)'
        }
      }, "ForexFactory"), ' · ', /*#__PURE__*/React.createElement("a", {
        href: "https://www.investing.com/economic-calendar/",
        target: "_blank",
        rel: "noopener",
        style: {
          color: 'var(--blue)'
        }
      }, "Investing.com"), ' · ', /*#__PURE__*/React.createElement("a", {
        href: "https://www.investing.com/economic-calendar/",
        target: "_blank",
        rel: "noopener",
        style: {
          color: 'var(--blue)'
        }
      }, "Investing.com")));
    }
    return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
      style: {
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        marginBottom: '0.5rem',
        overflowX: 'auto',
        scrollbarWidth: 'none',
        paddingBottom: '2px'
      }
    }, weeks.map(function (wk) {
      var isActiveWk = wk === activeWeekKey;
      return /*#__PURE__*/React.createElement("button", {
        key: wk,
        onClick: function onClick() {
          var wkDays = weekMap[wk];
          if (!wkDays || !wkDays.length) return;
          // Select today's day if it's in this week, otherwise first day
          var todayInWk = wkDays.find(function (d) {
            return d.dateISO === todayISO;
          });
          setCalSelectedDate((todayInWk || wkDays[0]).dateISO);
        },
        style: {
          padding: '0.35rem 0.75rem',
          background: isActiveWk ? 'var(--blue)' : 'var(--bg-secondary)',
          border: "1px solid ".concat(isActiveWk ? 'var(--blue)' : 'var(--border)'),
          borderRadius: '20px',
          color: isActiveWk ? '#fff' : 'var(--text-secondary)',
          cursor: 'pointer',
          fontSize: '0.75rem',
          fontWeight: isActiveWk ? 600 : 400,
          whiteSpace: 'nowrap',
          flexShrink: 0,
          transition: 'all 0.15s'
        }
      }, fmtWeekLabel(wk));
    })), /*#__PURE__*/React.createElement("div", {
      style: {
        display: 'flex',
        alignItems: 'center',
        gap: '0',
        marginBottom: '1rem',
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border)',
        borderRadius: '6px',
        overflow: 'hidden'
      }
    }, /*#__PURE__*/React.createElement("button", {
      onClick: goToPrev,
      disabled: activeDayIdx <= 0,
      style: {
        padding: '0.75rem 1rem',
        background: 'transparent',
        border: 'none',
        borderRight: '1px solid var(--border)',
        color: activeDayIdx <= 0 ? 'var(--text-tertiary)' : 'var(--text-primary)',
        cursor: activeDayIdx <= 0 ? 'default' : 'pointer',
        fontSize: '1rem',
        transition: 'background 0.15s',
        flexShrink: 0
      },
      onMouseEnter: function onMouseEnter(e) {
        if (activeDayIdx > 0) e.currentTarget.style.background = 'var(--bg-elevated)';
      },
      onMouseLeave: function onMouseLeave(e) {
        e.currentTarget.style.background = 'transparent';
      }
    }, "\u2039"), /*#__PURE__*/React.createElement("div", {
      style: {
        display: 'flex',
        overflowX: 'auto',
        flex: 1,
        scrollbarWidth: 'none'
      }
    }, daysInActiveWeek.map(function (day) {
      var isActive = day.dateISO === activeDateISO;
      var isToday = day.dateISO === todayISO;
      var highCount = day.events.filter(function (e) {
        return e.impact === 'high';
      }).length;
      return /*#__PURE__*/React.createElement("button", {
        key: day.dateISO,
        onClick: function onClick() {
          return setCalSelectedDate(day.dateISO);
        },
        style: {
          padding: '0.625rem 1.125rem',
          background: isActive ? 'var(--bg-card)' : 'transparent',
          border: 'none',
          borderBottom: isActive ? '2px solid var(--blue)' : '2px solid transparent',
          color: isActive ? 'var(--text-primary)' : isToday ? 'var(--blue)' : 'var(--text-secondary)',
          cursor: 'pointer',
          fontSize: '0.8125rem',
          fontWeight: isActive ? 600 : 400,
          whiteSpace: 'nowrap',
          transition: 'all 0.15s',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '0.2rem',
          position: 'relative',
          flexShrink: 0
        }
      }, /*#__PURE__*/React.createElement("span", null, day.label), highCount > 0 && /*#__PURE__*/React.createElement("span", {
        style: {
          fontSize: '0.6rem',
          color: 'var(--red)',
          fontWeight: 700,
          lineHeight: 1
        }
      }, ' ●'.repeat(Math.min(highCount, 3)).trim()));
    })), /*#__PURE__*/React.createElement("button", {
      onClick: goToNext,
      disabled: activeDayIdx >= days.length - 1,
      style: {
        padding: '0.75rem 1rem',
        background: 'transparent',
        border: 'none',
        borderLeft: '1px solid var(--border)',
        color: activeDayIdx >= days.length - 1 ? 'var(--text-tertiary)' : 'var(--text-primary)',
        cursor: activeDayIdx >= days.length - 1 ? 'default' : 'pointer',
        fontSize: '1rem',
        transition: 'background 0.15s',
        flexShrink: 0
      },
      onMouseEnter: function onMouseEnter(e) {
        if (activeDayIdx < days.length - 1) e.currentTarget.style.background = 'var(--bg-elevated)';
      },
      onMouseLeave: function onMouseLeave(e) {
        e.currentTarget.style.background = 'transparent';
      }
    }, "\u203A")), /*#__PURE__*/React.createElement("div", {
      style: {
        display: 'flex',
        gap: '0.5rem',
        marginBottom: '1rem',
        flexWrap: 'wrap',
        alignItems: 'center'
      }
    }, ['all', 'high', 'medium', 'low'].map(function (f) {
      return /*#__PURE__*/React.createElement("button", {
        key: f,
        onClick: function onClick() {
          return setCalFilter(f);
        },
        style: {
          padding: '0.375rem 0.875rem',
          borderRadius: '4px',
          border: '1px solid var(--border)',
          background: calFilter === f ? 'var(--blue)' : 'var(--bg-card)',
          color: calFilter === f ? 'white' : 'var(--text-secondary)',
          cursor: 'pointer',
          fontSize: '0.8125rem',
          fontWeight: 500,
          transition: 'all 0.15s'
        }
      }, f === 'all' ? 'Todos' : f === 'high' ? 'Alto' : f === 'medium' ? 'Medio' : 'Bajo');
    }), /*#__PURE__*/React.createElement("span", {
      style: {
        marginLeft: 'auto',
        fontSize: '0.75rem',
        color: 'var(--text-tertiary)'
      }
    }, filteredEvents.length, " eventos", activeDay && /*#__PURE__*/React.createElement("span", {
      style: {
        marginLeft: '0.5rem',
        color: 'var(--text-tertiary)'
      }
    }, "\u2014 ", activeDay.label))), filteredEvents.length === 0 ? /*#__PURE__*/React.createElement("div", {
      style: {
        textAlign: 'center',
        padding: '2.5rem',
        color: 'var(--text-secondary)',
        background: 'var(--bg-secondary)',
        borderRadius: '6px',
        border: '1px solid var(--border)'
      }
    }, "No hay eventos con ese filtro para este d\xEDa.") : /*#__PURE__*/React.createElement("div", {
      className: "calendar-container",
      ref: calendarContainerRef
    },
    /*#__PURE__*/React.createElement("div", { className: "calendar-header-row" },
      /*#__PURE__*/React.createElement("div", null, "Hora"),
      /*#__PURE__*/React.createElement("div", { style: { textAlign: 'center' } }, "Impacto"),
      /*#__PURE__*/React.createElement("div", null, "Divisa"),
      /*#__PURE__*/React.createElement("div", null, "Evento"),
      /*#__PURE__*/React.createElement("div", { style: { textAlign: 'right' } }, "Real"),
      /*#__PURE__*/React.createElement("div", { style: { textAlign: 'right' } }, "Est"),
      /*#__PURE__*/React.createElement("div", { style: { textAlign: 'right' } }, "Ant")
    ),
    filteredEvents.map(function (event, index) {
      var isToday = event.dateISO === todayISO;
      var hasActual = !!(event.actual && event.actual !== '');
      // Compute event timestamp in UTC for past/next logic
      var nowMs = Date.now();
      var eventMs = null;
      if (event.dateISO && (event.timeUTC || event.time)) {
        var t = event.timeUTC || event.time;
        // dateISO is always the real UTC date.
        // No offset correction needed (TE scraper converts UTC-3 → UTC correctly).
        var d = new Date("".concat(event.dateISO, "T").concat(t, ":00Z"));
        if (!isNaN(d.getTime())) eventMs = d.getTime();
      }
      // isPast: has actual OR event time already passed
      var isPast = hasActual || eventMs !== null && eventMs < nowMs;
      // isNext: first upcoming event with known time
      // dateISO+timeUTC already correct UTC — no offset hack needed
      var isNext = !isPast && !hasActual && eventMs !== null && filteredEvents.slice(0, index).every(function (e) {
        var ta = e.timeUTC || e.time;
        if (!ta || !e.dateISO) return true;
        var d2 = new Date("".concat(e.dateISO, "T").concat(ta, ":00Z"));
        return isNaN(d2.getTime()) || d2.getTime() < nowMs || !!(e.actual && e.actual !== '');
      });
      var eventClass = "calendar-event".concat(isPast ? ' is-past' : '').concat(isNext ? ' is-next' : '');

      // Determine actual color: green = beat, red = miss, neutral = no comparison
      var parseNumeric = function parseNumeric(s) {
        if (!s) return null;
        var n = parseFloat(String(s).replace(/[%TBMKk,]/g, ''));
        return isNaN(n) ? null : n;
      };
      var actualColor = 'var(--green-strong)';
      if (hasActual) {
        var aNum = parseNumeric(event.actual);
        var fNum = parseNumeric(event.forecast || event.previous);
        if (aNum !== null && fNum !== null) {
          // For most indicators: higher actual = green (beat). Exception: unemployment, CPI miss.
          var evLower = (event.event || '').toLowerCase();
          var lowerIsBetter = /unemploy|claims|inflation expectation|cpi|ppi|deficit/i.test(evLower);
          var beat = lowerIsBetter ? aNum < fNum : aNum > fNum;
          var miss = lowerIsBetter ? aNum > fNum : aNum < fNum;
          if (beat) actualColor = 'var(--green-strong)';else if (miss) actualColor = 'var(--red-strong)';else actualColor = 'var(--text-secondary)';
        }
      }
      return /*#__PURE__*/React.createElement("div", {
        key: index,
        className: eventClass,
        ref: isNext ? nextEventRef : null,
        "data-event-key": "".concat(event.dateISO, "|").concat(event.currency, "|").concat(event.event),
        style: {
          borderLeft: isNext ? '3px solid var(--green)' : !isPast && isToday ? '3px solid var(--blue)' : '3px solid transparent'
        }
      },
      /*#__PURE__*/React.createElement("div", { className: "event-time", style: {
        color: isNext ? 'var(--green)' : isToday && !isPast ? 'var(--blue)' : 'var(--text-tertiary)'
      }}, formatEventTime(event.timeUTC || event.time, event.dateISO)),
      /*#__PURE__*/React.createElement("div", { className: "event-impact-badge impact-".concat(event.impact) },
        event.impact === 'high' ? 'Alto' : event.impact === 'medium' ? 'Med' : 'Bajo'
      ),
      /*#__PURE__*/React.createElement("div", { className: "event-currency-cell" },
        /*#__PURE__*/React.createElement("img", {
          className: "event-flag-img",
          src: "https://flagcdn.com/w20/".concat(({'USD':'us','EUR':'eu','GBP':'gb','JPY':'jp','AUD':'au','CAD':'ca','CHF':'ch','NZD':'nz'}[event.currency] || 'un'), ".png"),
          srcSet: "https://flagcdn.com/w40/".concat(({'USD':'us','EUR':'eu','GBP':'gb','JPY':'jp','AUD':'au','CAD':'ca','CHF':'ch','NZD':'nz'}[event.currency] || 'un'), ".png 2x"),
          alt: event.currency,
          loading: "lazy"
        }),
        " ", event.currency
      ),
      /*#__PURE__*/React.createElement("div", { className: "event-title", title: event.event }, event.event),
      /*#__PURE__*/React.createElement("div", { className: "event-data-cell", style: { color: hasActual ? actualColor : 'var(--text-tertiary)' }},
        event.actual || '—'
      ),
      /*#__PURE__*/React.createElement("div", { className: "event-data-cell", style: { color: 'var(--text-secondary)' }},
        event.forecast || '—'
      ),
      /*#__PURE__*/React.createElement("div", { className: "event-data-cell", style: { color: 'var(--text-tertiary)' }},
        event.previous || '—'
      ));
    })));
  }()), activeTab === 'alerts' && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "section-header"
  }, /*#__PURE__*/React.createElement("div", {
    className: "section-title"
  }, "Divergencias Macro - Spreads Fundamentales"), /*#__PURE__*/React.createElement("div", {
    className: "section-meta"
  }, "Spreads fundamentales entre divisas G8 \xB7 Contexto macro, no se\xF1al de trading directo \xB7 Diferencial \u226520 pts")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(min(400px, 100%), 1fr))',
      gap: '1.5rem'
    }
  }, dynamicAlerts.map(function (rec, index) {
    var _rec$pairMom1M, _rec$pairMom1M2;
    return /*#__PURE__*/React.createElement("div", {
      key: index,
      className: "currency-card",
      style: {
        borderLeft: "4px solid ".concat(
          rec.momAlignment === -1
            ? 'var(--red-strong)'
            : rec.momAlignment === 1
              ? (rec.type === 'long' ? 'var(--green-strong)' : 'var(--red-strong)')
              : (rec.type === 'long' ? 'var(--green-strong)' : 'var(--red-strong)')
        )
      }
    }, /*#__PURE__*/React.createElement("div", {
      className: "card-header",
      style: {
        background: rec.momAlignment === -1
          ? (rec.pairMom7d !== null && rec.pairMom7d < -1.0 ? 'rgba(239,83,80,0.18)' : 'rgba(239,83,80,0.11)')
          : rec.momAlignment === 1
            ? (rec.type === 'long' ? 'rgba(38,166,154,0.13)' : 'rgba(239,83,80,0.13)')
            : (rec.type === 'long' ? 'rgba(38,166,154,0.08)' : 'rgba(239,83,80,0.08)')
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        flex: 1
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: '1.5rem',
        fontWeight: 700,
        marginBottom: '0.5rem',
        color: 'var(--text-primary)'
      }
    }, rec.pair), /*#__PURE__*/React.createElement("div", {
      style: {
        display: 'flex',
        alignItems: 'center',
        gap: '1rem'
      }
    }, /*#__PURE__*/React.createElement("span", {
      className: "sentiment-badge sentiment-".concat(rec.type === 'long' ? 'alcista' : 'bajista')
    }, rec.direction), rec.pairMom7d !== null && /*#__PURE__*/React.createElement("span", {
      title: rec.momAlignment === 1
        ? "Precio confirma la señal fundamental: par ".concat(rec.pairMom7d > 0 ? '+' : '', rec.pairMom7d.toFixed(2), "% esta semana (umbral >+0.40%)")
        : rec.momAlignment === -1
          ? "Divergencia de precio: par ".concat(rec.pairMom7d.toFixed(2), "% esta semana, en contra de la señal").concat(rec.pairMom7d < -1.0 ? " — corrección significativa" : "")
          : "Momentum neutro esta semana (".concat(rec.pairMom7d > 0 ? '+' : '', rec.pairMom7d.toFixed(2), "% par 7d, dentro del rango ±0.40%)"),
      style: {
        fontSize: '0.72rem',
        fontWeight: 700,
        background: rec.momAlignment === 1
          ? 'rgba(38,166,154,0.18)'
          : rec.momAlignment === -1
            ? 'rgba(239,83,80,0.18)'
            : 'rgba(120,120,120,0.12)',
        color: rec.momAlignment === 1
          ? 'var(--green-strong)'
          : rec.momAlignment === -1
            ? 'var(--red-strong)'
            : 'var(--text-tertiary)',
        border: rec.momAlignment === 1
          ? '1px solid rgba(38,166,154,0.35)'
          : rec.momAlignment === -1
            ? '1px solid rgba(239,83,80,0.35)'
            : '1px solid rgba(120,120,120,0.20)',
        borderRadius: '4px',
        padding: '2px 7px',
        cursor: 'help',
        whiteSpace: 'nowrap'
      }
    }, rec.momAlignment === 1 ? 'Confirmado ' : rec.momAlignment === -1 ? 'Divergencia ' : 'Neutral ',
       (rec.pairMom7d > 0 ? '+' : ''), rec.pairMom7d.toFixed(2), '%'
    ), rec.momentumOpposing && /*#__PURE__*/React.createElement("span", {
      title: rec.type === 'short' ? "El par subi\xF3 +".concat(Math.abs((_rec$pairMom1M = rec.pairMom1M) !== null && _rec$pairMom1M !== void 0 ? _rec$pairMom1M : 0).toFixed(1), "% el \xFAltimo mes, divergiendo de la se\xF1al SHORT \u2014 el strong cay\xF3 frente al d\xE9bil. La se\xF1al FX contradice el diferencial fundamental. El precio se ha movido en contra de la se\xF1al en el \xFAltimo mes.") : "El par cay\xF3 -".concat(Math.abs((_rec$pairMom1M2 = rec.pairMom1M) !== null && _rec$pairMom1M2 !== void 0 ? _rec$pairMom1M2 : 0).toFixed(1), "% el \xFAltimo mes, divergiendo de la se\xF1al LONG \u2014 el strong cay\xF3 frente al d\xE9bil. La se\xF1al FX contradice el diferencial fundamental. El precio se ha movido en contra de la se\xF1al en el \xFAltimo mes."),
      style: {
        fontSize: '0.75rem',
        background: 'rgba(245, 158, 11, 0.15)',
        color: '#f59e0b',
        border: '1px solid rgba(245, 158, 11, 0.3)',
        borderRadius: '4px',
        padding: '2px 7px',
        cursor: 'help'
      }
    }, "Div. precio"), rec.jpyAsShort && /*#__PURE__*/React.createElement("span", {
      title: "El JPY se mueve principalmente por apetito de riesgo global (risk-on/off), no por fundamentales macro. En entornos de risk-off hist\xF3ricamente se aprecia independientemente de sus fundamentos.",
      style: {
        fontSize: '0.75rem',
        background: 'rgba(100, 160, 255, 0.12)',
        color: '#7eb3ff',
        border: '1px solid rgba(100, 160, 255, 0.25)',
        borderRadius: '4px',
        padding: '2px 7px',
        cursor: 'help'
      }
    }, "JPY risk-off"))), /*#__PURE__*/React.createElement("div", {
      style: {
        textAlign: 'right'
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: '2rem',
        fontWeight: 700,
        color: rec.momAlignment === -1
          ? 'var(--red-strong)'
          : (rec.type === 'long' ? 'var(--green-strong)' : 'var(--red-strong)')
      }
    }, rec.spread != null ? rec.spread.toFixed(1) : '\u2014'), /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: '0.75rem',
        color: 'var(--text-tertiary)'
      }
    }, "Diferencial"))), /*#__PURE__*/React.createElement("div", {
      className: "card-body"
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        marginBottom: '1.5rem'
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '0.75rem'
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: '0.75rem',
        fontWeight: 600,
        color: 'var(--text-tertiary)',
        textTransform: 'uppercase',
        letterSpacing: '0.05em'
      }
    }, "Principales impulsores: ", rec.strongCurrency), /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: '1rem',
        fontWeight: 700,
        color: 'var(--green-strong)'
      }
    }, rec.strength != null ? rec.strength.toFixed(1) : '—')), /*#__PURE__*/React.createElement("div", {
      style: {
        background: 'var(--bg-elevated)',
        padding: '0.75rem',
        borderRadius: '4px',
        border: '1px solid var(--border)'
      }
    }, rec.strongReasons.length > 0 ? rec.strongReasons.map(function (reason, i) {
      return /*#__PURE__*/React.createElement("div", {
        key: i,
        style: {
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '0.8125rem',
          padding: '0.25rem 0',
          color: 'var(--text-secondary)'
        }
      }, /*#__PURE__*/React.createElement("span", null, "\u2022 ", reason.factor), /*#__PURE__*/React.createElement("span", {
        style: {
          color: 'var(--green-strong)',
          fontWeight: 600
        }
      }, reason.impact));
    }) : /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: '0.8125rem',
        color: 'var(--text-tertiary)',
        fontStyle: 'italic'
      }
    }, "Datos insuficientes"))), /*#__PURE__*/React.createElement("div", {
      style: {
        marginBottom: '1.5rem'
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '0.75rem'
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: '0.75rem',
        fontWeight: 600,
        color: 'var(--text-tertiary)',
        textTransform: 'uppercase',
        letterSpacing: '0.05em'
      }
    }, "Indicadores rezagados: ", rec.weakCurrency), /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: '1rem',
        fontWeight: 700,
        color: 'var(--red-strong)'
      }
    }, rec.weakness != null ? rec.weakness.toFixed(1) : '—')), /*#__PURE__*/React.createElement("div", {
      style: {
        background: 'var(--bg-elevated)',
        padding: '0.75rem',
        borderRadius: '4px',
        border: '1px solid var(--border)'
      }
    }, rec.weakReasons.length > 0 ? rec.weakReasons.map(function (reason, i) {
      return /*#__PURE__*/React.createElement("div", {
        key: i,
        style: {
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '0.8125rem',
          padding: '0.25rem 0',
          color: 'var(--text-secondary)'
        }
      }, /*#__PURE__*/React.createElement("span", null, "\u2022 ", reason.factor), /*#__PURE__*/React.createElement("span", {
        style: {
          color: 'var(--red-strong)',
          fontWeight: 600
        }
      }, reason.impact));
    }) : /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: '0.8125rem',
        color: 'var(--text-tertiary)',
        fontStyle: 'italic'
      }
    }, "Datos insuficientes"))),
    /*#__PURE__*/React.createElement("div", { style: { marginBottom: '1.5rem' } },
      /*#__PURE__*/React.createElement("div", {
        style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }
      },
        /*#__PURE__*/React.createElement("div", {
          style: { fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em' }
        }, "Momentum precio 7d: ", rec.strongCurrency, "/", rec.weakCurrency)
      ),
      /*#__PURE__*/React.createElement("div", {
        style: { background: 'var(--bg-elevated)', padding: '0.75rem', borderRadius: '4px', border: '1px solid var(--border)' }
      },
        [
          { label: rec.strongCurrency, val: rec.strongMom7d },
          { label: rec.weakCurrency,   val: rec.weakMom7d   },
          { label: 'Par neto',         val: rec.pairMom7d   }
        ].map(function(item) {
          var v = item.val;
          var clr = v === null ? 'var(--text-tertiary)' : v > 0 ? 'var(--green-strong)' : 'var(--red-strong)';
          return /*#__PURE__*/React.createElement("div", {
            key: item.label,
            style: { display: 'flex', justifyContent: 'space-between', fontSize: '0.8125rem', padding: '0.25rem 0', color: 'var(--text-secondary)' }
          },
            /*#__PURE__*/React.createElement("span", null, "\u2022 ", item.label),
            /*#__PURE__*/React.createElement("span", { style: { color: clr, fontWeight: 600 } },
              v !== null ? (v > 0 ? '+' : '') + v.toFixed(2) + '%' : '\u2014'
            )
          );
        })
      )
    ),
    /*#__PURE__*/React.createElement("div", {
      style: {
        background: 'var(--bg-card)',
        padding: '1rem',
        borderRadius: '4px',
        borderLeft: "3px solid ".concat(rec.type === 'long' ? 'var(--green-strong)' : 'var(--red-strong)'),
        fontSize: '0.875rem',
        lineHeight: '1.6',
        color: 'var(--text-secondary)'
      }
    }, /*#__PURE__*/React.createElement("strong", {
      style: { color: 'var(--text-primary)' }
    }, "An\xE1lisis:"), /*#__PURE__*/React.createElement("span", null,
      rec.type === 'long'
        ? (function() {
            var pDir = rec.pairMom7d !== null ? (rec.pairMom7d > 0 ? '+' : '') + rec.pairMom7d.toFixed(2) + '% par 7d' : null;
            return [
              ' ', rec.strongCurrency, ' supera fundamentalmente a ', rec.weakCurrency,
              ' (', rec.strength != null ? rec.strength.toFixed(1) : '\u2014', ' vs ',
              rec.weakness != null ? rec.weakness.toFixed(1) : '\u2014', ' pts, diferencial ',
              rec.spread != null ? rec.spread.toFixed(1) : '\u2014', ' pts).',
              rec.momAlignment === 1
                ? [' Precio confirma (par +', rec.pairMom7d.toFixed(2), '% esta semana): momentum alineado con el fundamental. Se\xF1al de alta convicción.']
                : rec.momAlignment === -1
                  ? [' Divergencia de precio: par ', rec.pairMom7d.toFixed(2), '% esta semana, en contra del LONG.',
                     rec.pairMom7d < -1.0 ? ' Corrección significativa — esperar estabilización antes de entrar.' : ' Monitorear; puede ser pullback hacia soporte.']
                  : [' Precio neutral (par ', pDir ? rec.pairMom7d.toFixed(2) + '% esta semana' : 'sin datos', '). Entrada basada en fundamentos.'],
              ' Par: ', rec.pair, '.'
            ];
          })()
        : (function() {
            var pDir = rec.pairMom7d !== null ? (rec.pairMom7d > 0 ? '+' : '') + rec.pairMom7d.toFixed(2) + '% par 7d' : null;
            return [
              ' ', rec.weakCurrency, ' registra debilidad fundamental vs ', rec.strongCurrency,
              ' (', rec.weakness != null ? rec.weakness.toFixed(1) : '\u2014', ' vs ',
              rec.strength != null ? rec.strength.toFixed(1) : '\u2014', ' pts, diferencial ',
              rec.spread != null ? rec.spread.toFixed(1) : '\u2014', ' pts).',
              rec.momAlignment === 1
                ? [' Precio confirma (par -', rec.pairMom7d.toFixed(2), '% esta semana, strong supera al weak): momentum alineado. Se\xF1al de alta convicción.']
                : rec.momAlignment === -1
                  ? [' Divergencia de precio: par ', rec.pairMom7d.toFixed(2), '% en contra del SHORT.',
                     rec.pairMom7d < -1.0 ? ' El weak rebota con fuerza — esperar agotamiento del rebote.' : ' Posible rebote temporal; mantener vigilancia.']
                  : [' Precio neutral (', pDir ? rec.pairMom7d.toFixed(2) + '% esta semana' : 'sin datos', '). Se\xF1al basada en fundamentos.'],
              ' Par: ', rec.pair, '.'
            ];
          })()
    )), /*#__PURE__*/React.createElement("div", {
      style: { marginTop: '0.5rem', fontSize: '0.7rem', color: 'var(--text-tertiary)' }
    }, "Calidad de datos: ", (rec.dataQuality * 100).toFixed(0), "% \xB7 Umbral confirmaci\xF3n: \xB10.40%"
    )));
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: '2rem',
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderRadius: '6px',
      padding: '1.25rem',
      fontSize: '0.8125rem',
      lineHeight: '1.6',
      color: 'var(--text-secondary)'
    }
  }, /*#__PURE__*/React.createElement("strong", {
    style: {
      color: 'var(--text-primary)'
    }
  }, "Aviso Importante:"), " Estos an\xE1lisis combinan metodolog\xEDa fundamental macroecon\xF3mica con momentum de precio (7d basket-adjusted) con fines informativos. La secci\xF3n de an\xE1lisis describe el contexto del par seg\xFAn los datos disponibles \u2014 no constituye una recomendaci\xF3n de operar ni asesoramiento financiero. El trading de divisas conlleva riesgos significativos de p\xE9rdida de capital. Confirme siempre con an\xE1lisis t\xE9cnico propio, aplique gesti\xF3n de riesgo y consulte con un asesor financiero certificado antes de tomar decisiones de inversi\xF3n.")), /*#__PURE__*/React.createElement("div", {
    className: "footer"
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontWeight: 600,
      marginBottom: '0.5rem'
    }
  }, "DASHBOARD DE AN\xC1LISIS FUNDAMENTAL FOREX \u2014 DATA-DRIVEN"), /*#__PURE__*/React.createElement("div", {
    style: {
      marginBottom: '1rem'
    }
  }, "Metodolog\xEDa: Modelo cuantitativo basado en datos econ\xF3micos reales \u2022 21 indicadores ponderados en 6 tiers", /*#__PURE__*/React.createElement("br", null), "Pol\xEDtica Monetaria (29%) \u2022 Balance Externo (19%) \u2022 Crecimiento & Empleo (16%) \u2022 Sentimiento de Mercado (21%) \u2022 FX Confirmador (11%) \u2022 Consumo & Salarios (4%)", /*#__PURE__*/React.createElement("br", null), "Datos actualizados diariamente desde FRED (St. Louis Fed), Frankfurter/ECB y CFTC"), /*#__PURE__*/React.createElement("div", {
    style: {
      background: 'var(--bg-card)',
      padding: '1rem',
      borderRadius: '6px',
      border: '1px solid var(--border)',
      marginTop: '1rem',
      fontSize: '0.8125rem',
      lineHeight: '1.6'
    }
  }, /*#__PURE__*/React.createElement("strong", null, "Aviso Legal:"), /*#__PURE__*/React.createElement("br", null), "Este dashboard es una herramienta de an\xE1lisis cuantitativo con fines informativos y educativos. Los scores de fortaleza, rankings y spreads fundamentales entre pares son c\xE1lculos basados en datos macroecon\xF3micos p\xFAblicos y representan divergencias macro — no constituyen asesoramiento financiero ni se\xF1ales de trading. Los fundamentos macroecon\xF3micos son hist\xF3ricamente descontados por el mercado antes o durante su publicaci\xF3n. El trading de divisas (forex) conlleva riesgos significativos de p\xE9rdida de capital. Consulte con un asesor financiero certificado antes de tomar decisiones de inversi\xF3n."), /*#__PURE__*/React.createElement("div", {
    style: {
      textAlign: 'center',
      marginTop: '1.5rem',
      paddingTop: '1.5rem',
      borderTop: '1px solid var(--border)'
    }
  }, /*#__PURE__*/React.createElement("button", {
    onClick: function onClick() {
      return window.clearForexCache && window.clearForexCache();
    },
    style: {
      padding: '10px 20px',
      background: 'var(--blue)',
      color: 'white',
      border: 'none',
      borderRadius: '6px',
      cursor: 'pointer',
      fontSize: '0.875rem',
      fontWeight: 600
    }
  }, "Limpiar Cach\xE9 y Recargar Datos"), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: '1rem',
      fontSize: '0.75rem',
      color: 'var(--text-tertiary)'
    }
  }, "\xA9 ", new Date().getFullYear(), " Santiago Pl\xE1 Casuriaga \xB7 Global Investing. Todos los derechos reservados."), "Dashboard v6.5.3"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexWrap: 'wrap',
      justifyContent: 'center',
      gap: '0.4rem 1rem',
      marginTop: '0.75rem',
      fontSize: '0.75rem',
      color: 'var(--text-secondary)'
    }
  }, /*#__PURE__*/React.createElement("a", {
    href: "about.html",
    style: {
      color: 'var(--text-secondary)',
      textDecoration: 'none',
      whiteSpace: 'nowrap'
    }
  }, "Acerca de"), /*#__PURE__*/React.createElement("a", {
    href: "publicidad.html",
    style: {
      color: 'var(--text-secondary)',
      textDecoration: 'none',
      whiteSpace: 'nowrap'
    }
  }, "Publicidad"), /*#__PURE__*/React.createElement("a", {
    href: "guia-score-fortaleza.html",
    style: {
      color: 'var(--text-secondary)',
      textDecoration: 'none',
      whiteSpace: 'nowrap'
    }
  }, "Gu\xEDas"), /*#__PURE__*/React.createElement("a", {
    href: "terms.html",
    style: {
      color: 'var(--text-secondary)',
      textDecoration: 'none',
      whiteSpace: 'nowrap'
    }
  }, "T\xE9rminos"), /*#__PURE__*/React.createElement("a", {
    href: "privacy.html",
    style: {
      color: 'var(--text-secondary)',
      textDecoration: 'none',
      whiteSpace: 'nowrap'
    }
  }, "Privacidad"), /*#__PURE__*/React.createElement("a", {
    href: "contact.html",
    style: {
      color: 'var(--text-secondary)',
      textDecoration: 'none',
      whiteSpace: 'nowrap'
    }
  }, "Contacto"), /*#__PURE__*/React.createElement("a", {
    href: "en.html",
    style: {
      color: 'var(--text-secondary)',
      textDecoration: 'none',
      padding: '0.2rem 0.5rem',
      border: '1px solid var(--border)',
      borderRadius: '4px',
      fontSize: '0.75rem',
      whiteSpace: 'nowrap'
    }
  }, "EN")))));
};
// Error Boundary para capturar crashes
var ErrorBoundary = /*#__PURE__*/function (_React$Component) {
  function ErrorBoundary(props) {
    var _this;
    _classCallCheck(this, ErrorBoundary);
    _this = _callSuper(this, ErrorBoundary, [props]);
    _this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
    return _this;
  }
  _inherits(ErrorBoundary, _React$Component);
  return _createClass(ErrorBoundary, [{
    key: "componentDidCatch",
    value: function componentDidCatch(error, errorInfo) {
      console.error('💥 Dashboard crashed:', error, errorInfo);
      this.setState({
        error: error,
        errorInfo: errorInfo
      });

      // Intentar limpiar caché corrupto
      try {
        CacheManager.clearAll();
        console.log('🧹 Cache cleared after crash');
      } catch (e) {
        console.error('Failed to clear cache:', e);
      }
    }
  }, {
    key: "render",
    value: function render() {
      if (this.state.hasError) {
        return /*#__PURE__*/React.createElement("div", {
          className: "app-container"
        }, /*#__PURE__*/React.createElement("div", {
          className: "header"
        }, /*#__PURE__*/React.createElement("div", {
          className: "header-content"
        }, /*#__PURE__*/React.createElement("div", {
          style: {
            display: "flex",
            flexDirection: "row",
            alignItems: "center"
          }
        }, /*#__PURE__*/React.createElement("img", {
          src: "apple-touch-icon.png",
          alt: "Global Investing",
          style: {
            width: "40px",
            height: "40px",
            borderRadius: "8px",
            marginRight: "0.75rem",
            flexShrink: 0,
            display: "block"
          }
        }), /*#__PURE__*/React.createElement("div", {
          style: {
            display: "flex",
            flexDirection: "column",
            justifyContent: "center"
          }
        }, /*#__PURE__*/React.createElement("div", {
          className: "brand-title"
        }, "Global Investing"), /*#__PURE__*/React.createElement("div", {
          className: "brand-subtitle"
        }, "Error del Sistema"))))), /*#__PURE__*/React.createElement("div", {
          className: "content"
        }, /*#__PURE__*/React.createElement("div", {
          className: "alert-card alert-warning",
          style: {
            maxWidth: '800px',
            margin: '2rem auto'
          }
        }, /*#__PURE__*/React.createElement("div", {
          className: "alert-content"
        }, /*#__PURE__*/React.createElement("h3", null, "El Dashboard ha Encontrado un Error"), /*#__PURE__*/React.createElement("p", {
          style: {
            marginTop: '1rem',
            lineHeight: '1.7'
          }
        }, "El sistema ha detectado un error cr\xEDtico. Se ha limpiado el cach\xE9 autom\xE1ticamente."), /*#__PURE__*/React.createElement("p", {
          style: {
            marginTop: '0.5rem',
            fontSize: '0.875rem',
            color: 'var(--text-secondary)'
          }
        }, /*#__PURE__*/React.createElement("strong", null, "Error:"), " ", this.state.error && this.state.error.toString()), /*#__PURE__*/React.createElement("button", {
          onClick: function onClick() {
            return window.location.reload();
          },
          style: {
            marginTop: '1.5rem',
            padding: '12px 24px',
            background: 'var(--blue)',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '0.875rem',
            fontWeight: 600
          }
        }, "Recargar Dashboard")))));
      }
      return this.props.children;
    }
  }], [{
    key: "getDerivedStateFromError",
    value: function getDerivedStateFromError(error) {
      return {
        hasError: true
      };
    }
  }]);
}(React.Component);
ReactDOM.render(/*#__PURE__*/React.createElement(ErrorBoundary, null, /*#__PURE__*/React.createElement(ForexDashboard, null)), document.getElementById('root'), function () {
  var f = document.getElementById('risk-footer');
  if (f) f.style.display = '';
});
