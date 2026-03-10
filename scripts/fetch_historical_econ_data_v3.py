#!/usr/bin/env python3
"""
fetch_historical_econ_data.py  —  v3.0
───────────────────────────────────────
Scraper de datos históricos económicos desde Trading Economics.
Playwright + Chromium simulando navegador real.

CAMBIOS v3.0 (correcciones críticas):
  1. gdpGrowth/inflation fallan → TE los sirve con JS retardado + paywall overlay.
     Fix: esperamos hasta 15s, forzamos cierre de modales/paywalls, probamos
     también el endpoint embed de TE (/chart/{country}/{slug}).
  2. EUR timeout → Después de ~13 requests seguidos el browser acumula cookies
     de tracking y TE eleva la sospecha. Fix: recreamos browser context cada
     N requests para limpiar estado.
  3. Retry más inteligente: si Highcharts está vacío después de 8s esperamos
     otros 5s antes de intentar la tabla, en lugar de fallar rápido.
  4. Anti-detección mejorada: rotate user-agents + añadir headers de referrer.
  5. Fallback a embed URL: https://tradingeconomics.com/{country}/{slug}#calendar
     tiene menos protección que la página principal.

USO LOCAL:
  pip install playwright && playwright install chromium
  python3 scripts/fetch_historical_econ_data.py
  python3 scripts/fetch_historical_econ_data.py --currency USD --indicator gdpGrowth
  python3 scripts/fetch_historical_econ_data.py --summary
  python3 scripts/fetch_historical_econ_data.py --currency USD --skip-existing
"""

import json, os, re, sys, time, random, argparse, csv
from datetime import date, datetime
from io import StringIO

# ─── Configuración ─────────────────────────────────────────────────────────

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

COUNTRY_SLUGS = {
    "USD": "united-states",
    "EUR": "euro-area",
    "GBP": "united-kingdom",
    "JPY": "japan",
    "AUD": "australia",
    "CAD": "canada",
    "CHF": "switzerland",
    "NZD": "new-zealand",
}

INDICATORS = {
    "gdpGrowth":          "gdp-growth-rate",
    "inflation":          "inflation-rate",
    "unemployment":       "unemployment-rate",
    "currentAccount":     "current-account-to-gdp",
    "production":         "industrial-production",
    "tradeBalance":       "balance-of-trade",
    "retailSales":        "retail-sales",
    "wageGrowth":         "wage-growth",
    "manufacturingPMI":   "manufacturing-pmi",
    "servicesPMI":        "services-pmi",
    "bond10y":            "government-bond-yield",
    "consumerConfidence": "consumer-confidence",
    "businessConfidence": "business-confidence",
}

INDICATOR_FREQ = {
    "gdpGrowth":          "quarterly",
    "inflation":          "monthly",
    "unemployment":       "monthly",
    "currentAccount":     "quarterly",
    "production":         "monthly",
    "tradeBalance":       "monthly",
    "retailSales":        "monthly",
    "wageGrowth":         "quarterly",
    "manufacturingPMI":   "monthly",
    "servicesPMI":        "monthly",
    "bond10y":            "monthly",
    "consumerConfidence": "monthly",
    "businessConfidence": "monthly",
}

# Indicadores que TE protege más agresivamente — necesitan tratamiento especial
PROTECTED_INDICATORS = {"gdpGrowth", "inflation"}

COT_SLUGS = {
    "EUR": "euro-fx-cftc-net-speculative-positions",
    "GBP": "british-pound-cftc-net-speculative-positions",
    "JPY": "japanese-yen-cftc-net-speculative-positions",
    "AUD": "australian-dollar-cftc-net-speculative-positions",
    "CAD": "canadian-dollar-cftc-net-speculative-positions",
    "CHF": "swiss-franc-cftc-net-speculative-positions",
    "NZD": "new-zealand-dollar-cftc-net-speculative-positions",
}

OUTPUT_DIR = "economic-data-history"
TE_BASE    = "https://tradingeconomics.com"

# Rotar user-agents para reducir fingerprinting
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

# Recrear contexto del browser cada N requests para limpiar estado/cookies
CONTEXT_REFRESH_EVERY = 15

# ─── Helpers ───────────────────────────────────────────────────────────────

def ensure_dirs():
    for cur in CURRENCIES:
        os.makedirs(f"{OUTPUT_DIR}/{cur}", exist_ok=True)

def load_existing(currency, indicator):
    path = f"{OUTPUT_DIR}/{currency}/{indicator}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_observations(currency, indicator, observations):
    path = f"{OUTPUT_DIR}/{currency}/{indicator}.json"
    pkg  = {
        "currency":     currency,
        "indicator":    indicator,
        "source":       "TradingEconomics via Playwright/Chromium v3.0",
        "fetched":      date.today().isoformat(),
        "observations": sorted(observations, key=lambda o: o["date"]),
    }
    with open(path, "w") as f:
        json.dump(pkg, f, indent=2)
    print(f"    ✓ Saved {len(observations)} obs → {path}")

def merge_observations(existing_obs, new_obs):
    merged = {o["date"]: o["value"] for o in existing_obs}
    for o in new_obs:
        if o.get("date") and o.get("value") is not None:
            merged[o["date"]] = o["value"]
    return [{"date": d, "value": v} for d, v in sorted(merged.items())]

def clean_num(text):
    if not text:
        return None
    s = str(text).strip().replace(",", "").replace("%", "").replace(" ", "")
    m = re.search(r"(-?\d+\.?\d*)", s)
    return float(m.group(1)) if m else None

def parse_te_date(date_text, freq="monthly"):
    if not date_text:
        return None
    t = str(date_text).strip()

    # ISO: '2024-03' or '2024-03-15'
    m = re.match(r"^(\d{4})-(\d{2})(?:-\d{2})?$", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}-15"

    # Timestamp ms (Highcharts): 1704067200000
    if re.match(r"^\d{10,13}$", t):
        ts = int(t)
        if ts > 9999999999:
            ts //= 1000
        dt = datetime.utcfromtimestamp(ts)
        return dt.strftime("%Y-%m-15")

    # 'Mar/25' or 'Mar/2025' or 'Mar 2025'
    m = re.match(r"^([A-Za-z]{3})[/\s](\d{2,4})$", t)
    if m:
        yr = m.group(2)
        if len(yr) == 2:
            yr = "20" + yr
        try:
            dt = datetime.strptime(f"{m.group(1)} {yr}", "%b %Y")
            return dt.strftime("%Y-%m-15")
        except ValueError:
            pass

    # 'Q4/24' or 'Q4 2024'
    m = re.match(r"Q(\d)[/\s](\d{2,4})", t)
    if m:
        q  = int(m.group(1))
        yr = m.group(2)
        if len(yr) == 2:
            yr = "20" + yr
        month = (q - 1) * 3 + 2
        return f"{yr}-{month:02d}-15"

    # Year only: '2024'
    if re.match(r"^\d{4}$", t):
        return f"{t}-06-15"

    # 'March 2025'
    m = re.match(r"^([A-Za-z]+)[/\s](\d{4})$", t)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%B %Y")
            return dt.strftime("%Y-%m-15")
        except ValueError:
            pass

    return None

def random_delay(mn=2.5, mx=5.5):
    time.sleep(random.uniform(mn, mx))

# ─── Browser setup ─────────────────────────────────────────────────────────

def create_context(playwright, ua=None):
    """Crea un nuevo browser context limpio. Llamar periódicamente para evitar tracking."""
    if ua is None:
        ua = random.choice(USER_AGENTS)
    
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--window-size=1440,900",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    ctx = browser.new_context(
        viewport={"width": 1440, "height": 900},
        user_agent=ua,
        locale="en-US",
        timezone_id="America/New_York",
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        },
    )
    ctx.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'plugins',   {get: () => [1,2,3,4,5]});
        Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
        window.chrome = {runtime: {}};
        // Spoof screen size
        Object.defineProperty(screen, 'width',  {get: () => 1920});
        Object.defineProperty(screen, 'height', {get: () => 1080});
    """)
    return browser, ctx

# ─── Cerrar modales / paywalls ─────────────────────────────────────────────

def dismiss_overlays(page):
    """
    Intenta cerrar popups, modales de suscripción y overlays de cookie consent
    que bloquean el gráfico en TE — especialmente en GDP e inflation.
    """
    # Cookie consent buttons
    cookie_sels = [
        "button:has-text('Accept')",
        "button:has-text('Accept All')",
        "button:has-text('I Accept')",
        "button:has-text('Got it')",
        "button:has-text('OK')",
        "#onetrust-accept-btn-handler",
        ".cookie-accept",
        "[aria-label='Accept cookies']",
    ]
    for sel in cookie_sels:
        try:
            btn = page.locator(sel).first
            if btn.count() > 0 and btn.is_visible():
                btn.click()
                page.wait_for_timeout(500)
                break
        except Exception:
            pass

    # Subscription / paywall modal close buttons
    modal_sels = [
        "button.close",
        ".modal .close",
        "[data-dismiss='modal']",
        "button:has-text('×')",
        "button:has-text('Close')",
        ".modal-backdrop",
        ".overlay-close",
        "#subscribe-modal .close",
    ]
    for sel in modal_sels:
        try:
            btn = page.locator(sel).first
            if btn.count() > 0 and btn.is_visible():
                btn.click()
                page.wait_for_timeout(400)
        except Exception:
            pass

    # Force-hide known overlay elements via JS
    page.evaluate("""
        const overlaySelectors = [
            '.modal', '.modal-backdrop', '.overlay', '.paywall',
            '#subscribe-modal', '#login-modal', '.subscription-wall',
            '[class*="paywall"]', '[class*="subscribe"]', '[id*="modal"]',
            '.cookie-banner', '#cookie-law-info-bar', '.gdpr-banner',
        ];
        overlaySelectors.forEach(sel => {
            document.querySelectorAll(sel).forEach(el => {
                el.style.display = 'none';
                el.remove();
            });
        });
        // Restore body scroll (modals often lock it)
        document.body.style.overflow = 'auto';
        document.documentElement.style.overflow = 'auto';
    """)

# ─── Estrategia 1: Highcharts JS eval ──────────────────────────────────────

JS_HC = """() => {
    try {
        if (!window.Highcharts || !window.Highcharts.charts) return null;
        const charts = window.Highcharts.charts.filter(c => c);
        if (!charts.length) return null;
        let best = null, bestLen = 0;
        for (const chart of charts) {
            for (const s of (chart.series || [])) {
                const pts = s.data || s.points || [];
                if (pts.length > bestLen) {
                    bestLen = pts.length;
                    const xAxis = chart.xAxis && chart.xAxis[0];
                    best = { pts, categories: xAxis ? xAxis.categories : null };
                }
            }
        }
        if (!best || bestLen < 3) return null;
        return best.pts.map((p, i) => ({
            x: p.x !== undefined ? p.x : null,
            y: p.y !== undefined ? p.y : (p.options ? p.options.y : null),
            label: best.categories ? best.categories[i] : null,
        }));
    } catch(e) { return null; }
}"""

def extract_highcharts(page):
    try:
        pts = page.evaluate(JS_HC)
        if not pts or len(pts) < 3:
            return []
        obs = []
        for p in pts:
            raw = p.get("label") or (str(int(p["x"])) if p.get("x") is not None else None)
            dt  = parse_te_date(raw) if raw else None
            val = clean_num(p.get("y"))
            if dt and val is not None:
                obs.append({"date": dt, "value": val})
        if obs:
            print(f"      → Highcharts: {len(obs)} pts")
        return obs
    except Exception as e:
        print(f"      → Highcharts error: {e}")
        return []

# ─── Estrategia 2: Click Historical tab → tabla ────────────────────────────

HIST_TAB_SELS = [
    "a[href='#historical']",
    "a:has-text('Historical')",
    "li:has-text('Historical') > a",
    ".nav-tabs a:has-text('Historical')",
    "a[data-toggle='tab']:has-text('Historical')",
    "#historical-tab",
    "a[href*='historical']",
]

def click_historical_tab(page):
    for sel in HIST_TAB_SELS:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                loc.click(force=True)
                page.wait_for_timeout(1500)
                print(f"      → Clicked: {sel}")
                return True
        except Exception:
            pass
    return False

def scrape_table(page, freq):
    obs = []
    try:
        page.evaluate("""
            document.querySelectorAll('table, .tab-pane, [class*="historical"]').forEach(el => {
                el.style.display = el.tagName === 'TABLE' ? 'table' : '';
                el.style.visibility = 'visible';
                el.style.opacity = '1';
                let p = el.parentElement;
                while (p && p !== document.body) {
                    if (getComputedStyle(p).display === 'none') p.style.display = '';
                    p = p.parentElement;
                }
            });
        """)
        page.wait_for_timeout(600)

        tables = page.query_selector_all("table")
        for table in tables:
            rows = table.query_selector_all("tr")
            if len(rows) < 4:
                continue
            header_cells = rows[0].query_selector_all("th, td")
            headers = [c.inner_text().strip().lower() for c in header_cells]

            date_col  = next((i for i, h in enumerate(headers) if any(k in h for k in
                ["date","reference","period","release","month","quarter","year"])), 0)
            value_col = next((i for i, h in enumerate(headers) if any(k in h for k in
                ["actual","value","last","close","previous"])), 1)

            count = 0
            for row in rows[1:]:
                cells = row.query_selector_all("td")
                if len(cells) <= max(date_col, value_col):
                    continue
                dt  = parse_te_date(cells[date_col].inner_text().strip(), freq)
                val = clean_num(cells[value_col].inner_text().strip())
                if dt and val is not None and dt >= "2015-01-01":
                    obs.append({"date": dt, "value": val})
                    count += 1

            if count >= 4:
                print(f"      → Table: {count} rows")
                return obs
    except Exception as e:
        print(f"      → Table error: {e}")
    return obs

# ─── Estrategia 3: XHR intercept ──────────────────────────────────────────

def setup_xhr(page):
    captured = []
    def handle(response):
        url = response.url
        if "tradingeconomics.com" not in url:
            return
        if not any(k in url for k in ["chart","data","series","embed","download",".csv",".json","historical","indicator"]):
            return
        ct = response.headers.get("content-type", "")
        try:
            if "json" in ct:
                body = response.json()
                if body:
                    captured.append(("json", body))
            elif any(x in ct for x in ["csv","text/plain","octet-stream"]):
                body = response.text()
                if body and len(body) > 50:
                    captured.append(("csv", body))
        except Exception:
            pass
    page.on("response", handle)
    return captured

def parse_xhr(captured):
    for dtype, body in captured:
        obs = []
        if dtype == "json":
            arr = body if isinstance(body, list) else next(
                (v for v in body.values() if isinstance(v, list) and len(v) > 5), None
            ) if isinstance(body, dict) else None
            if arr:
                for item in arr:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        dt  = parse_te_date(str(item[0]))
                        val = clean_num(item[1])
                    elif isinstance(item, dict):
                        dt  = parse_te_date(item.get("Date") or item.get("date") or item.get("DateTime") or "")
                        val = clean_num(item.get("Value") or item.get("value") or item.get("Last") or item.get("Actual") or "")
                    else:
                        continue
                    if dt and val is not None:
                        obs.append({"date": dt, "value": val})
                if len(obs) >= 4:
                    print(f"      → XHR JSON: {len(obs)} pts")
                    return obs
        elif dtype == "csv":
            try:
                reader = csv.DictReader(StringIO(body))
                for row in reader:
                    keys = list(row.keys())
                    dk = next((k for k in keys if any(x in k.lower() for x in ["date","reference","period"])), keys[0] if keys else None)
                    vk = next((k for k in keys if any(x in k.lower() for x in ["actual","value","last","close"])), keys[1] if len(keys)>1 else None)
                    if dk and vk:
                        dt  = parse_te_date(row[dk])
                        val = clean_num(row[vk])
                        if dt and val is not None:
                            obs.append({"date": dt, "value": val})
                if len(obs) >= 4:
                    print(f"      → XHR CSV: {len(obs)} pts")
                    return obs
            except Exception:
                pass
    return []

# ─── Estrategia 4: URL alternativa (para indicadores protegidos) ───────────

def try_alternate_urls(ctx, country_slug, indicator_slug, freq):
    """
    Para gdpGrowth e inflation, TE a veces sirve los datos desde URLs alternativas
    que tienen menos protección:
      - /embed/{country}/{slug}  (iframe sin paywall)
      - /{country}/{slug}?embed=1
      - el calendar endpoint que lista valores históricos
    """
    alt_urls = [
        f"{TE_BASE}/{country_slug}/{indicator_slug}?embed=1",
        f"{TE_BASE}/{country_slug}/{indicator_slug}#calendar",
    ]
    
    for url in alt_urls:
        print(f"      → Alt URL: {url}")
        page = ctx.new_page()
        captured = setup_xhr(page)
        obs = []
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)
            dismiss_overlays(page)
            page.wait_for_timeout(1000)
            
            # Try Highcharts
            try:
                page.wait_for_function(
                    "() => window.Highcharts && window.Highcharts.charts && "
                    "window.Highcharts.charts.filter(c=>c).length > 0",
                    timeout=8000
                )
            except Exception:
                pass
            
            obs = extract_highcharts(page)
            if len(obs) >= 5:
                break
            
            # Try table
            click_historical_tab(page)
            page.wait_for_timeout(1500)
            obs2 = extract_highcharts(page)
            if len(obs2) > len(obs):
                obs = obs2
            if len(obs) < 5:
                table_obs = scrape_table(page, freq)
                if len(table_obs) > len(obs):
                    obs = table_obs
            
            # XHR
            if len(obs) < 5:
                xhr_obs = parse_xhr(captured)
                if len(xhr_obs) > len(obs):
                    obs = xhr_obs
            
            if len(obs) >= 5:
                break

        except Exception as e:
            print(f"        ✗ Alt URL error: {e}")
        finally:
            page.close()
        
        random_delay(2, 4)
    
    return obs

# ─── Scraper por página ────────────────────────────────────────────────────

def scrape_page(ctx, country_slug, indicator_slug, freq, indicator_name=None, retries=2):
    url = f"{TE_BASE}/{country_slug}/{indicator_slug}"
    print(f"    → {url}")
    is_protected = indicator_name in PROTECTED_INDICATORS

    for attempt in range(retries + 1):
        if attempt > 0:
            print(f"      → Retry {attempt}/{retries}")
            random_delay(8, 15)  # Pausa más larga entre retries

        page = ctx.new_page()
        captured = setup_xhr(page)
        obs = []

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=40000)
            
            # Para indicadores protegidos, esperamos más antes de actuar
            initial_wait = 3000 if not is_protected else 5000
            page.wait_for_timeout(initial_wait)
            
            # Dismiss overlays/modals ANTES de intentar extraer datos
            dismiss_overlays(page)
            page.wait_for_timeout(800)

            # Progressive scroll to trigger lazy loads
            for y in [200, 500, 900, 1300]:
                page.evaluate(f"window.scrollTo(0, {y})")
                page.wait_for_timeout(400)
            
            # Scroll back to top (some charts only render when visible at top)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(500)

            # Wait for Highcharts to initialize
            hc_timeout = 10000 if not is_protected else 15000
            try:
                page.wait_for_function(
                    "() => window.Highcharts && window.Highcharts.charts && "
                    "window.Highcharts.charts.filter(c=>c).length > 0",
                    timeout=hc_timeout
                )
            except Exception:
                pass  # Continue without Highcharts

            # Strategy 1: Highcharts
            obs = extract_highcharts(page)
            if len(obs) >= 10:
                break

            # If Highcharts got some data but not enough, wait more
            if 0 < len(obs) < 10:
                page.wait_for_timeout(3000)
                obs2 = extract_highcharts(page)
                if len(obs2) > len(obs):
                    obs = obs2
                if len(obs) >= 8:
                    break

            # Strategy 2: Click Historical tab, then Highcharts + table
            click_historical_tab(page)
            page.wait_for_timeout(2000)

            obs2 = extract_highcharts(page)
            if len(obs2) > len(obs):
                obs = obs2

            if len(obs) < 10:
                table_obs = scrape_table(page, freq)
                if len(table_obs) > len(obs):
                    obs = table_obs

            # Strategy 3: XHR
            if len(obs) < 5:
                xhr_obs = parse_xhr(captured)
                if len(xhr_obs) > len(obs):
                    obs = xhr_obs

            if len(obs) >= 5:
                break

        except Exception as e:
            print(f"      ✗ Error (attempt {attempt+1}): {e}")
        finally:
            page.close()

    # Strategy 4: Alternate URLs for protected indicators (last resort)
    if len(obs) < 5 and is_protected:
        print(f"      → Trying alternate URLs for protected indicator...")
        alt_obs = try_alternate_urls(ctx, country_slug, indicator_slug, freq)
        if len(alt_obs) > len(obs):
            obs = alt_obs

    # Deduplicate and filter
    seen = set()
    result = []
    for o in obs:
        if o["date"] >= "2015-01-01" and o["date"] not in seen:
            seen.add(o["date"])
            result.append(o)
    return sorted(result, key=lambda o: o["date"])


# ─── COT histórico ─────────────────────────────────────────────────────────

def scrape_cot_all(ctx):
    print("\n── COT/CFTC Positioning History ──")
    for currency, slug in COT_SLUGS.items():
        existing     = load_existing(currency, "cotPositioning")
        existing_obs = existing.get("observations", []) if existing else []
        url = f"{TE_BASE}/cot/{slug}"
        print(f"\n  {currency}: {url}")
        page = ctx.new_page()
        captured = setup_xhr(page)
        obs = []
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=35000)
            page.wait_for_timeout(2000)
            dismiss_overlays(page)
            for y in [300, 700, 1100]:
                page.evaluate(f"window.scrollTo(0, {y})")
                page.wait_for_timeout(400)
            try:
                page.wait_for_function(
                    "() => window.Highcharts && window.Highcharts.charts && "
                    "window.Highcharts.charts.filter(c=>c).length > 0",
                    timeout=8000
                )
            except Exception:
                pass
            obs = extract_highcharts(page)
            if len(obs) < 5:
                click_historical_tab(page)
                page.wait_for_timeout(1500)
                obs2 = extract_highcharts(page)
                if len(obs2) > len(obs):
                    obs = obs2
            if len(obs) < 5:
                obs = scrape_table(page, "weekly") or obs
            if len(obs) < 5:
                obs = parse_xhr(captured) or obs
        except Exception as e:
            print(f"    ✗ {e}")
        finally:
            page.close()

        obs = [o for o in obs if o["date"] >= "2015-01-01"]
        if obs:
            final = merge_observations(existing_obs, obs)
            save_observations(currency, "cotPositioning", final)
        else:
            print(f"    ✗ No COT data for {currency}")
        random_delay(4, 8)


# ─── Runner ────────────────────────────────────────────────────────────────

def run(currencies=None, indicators=None, skip_existing=False):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: pip install playwright && playwright install chromium")
        sys.exit(1)

    ensure_dirs()
    target_cur = currencies  or CURRENCIES
    target_ind = indicators  or list(INDICATORS.keys())
    total      = len(target_cur) * len(target_ind)
    done = errors = 0
    requests_since_refresh = 0

    print("=" * 62)
    print("HISTORICAL ECON DATA SCRAPER v3.0 — Playwright/Chromium")
    print(f"Divisas:     {target_cur}")
    print(f"Indicadores: {target_ind}")
    print(f"Total URLs:  {total}")
    print("=" * 62)

    with sync_playwright() as pw:
        browser, ctx = create_context(pw)
        try:
            for currency in target_cur:
                country = COUNTRY_SLUGS[currency]
                print(f"\n{'─'*52}")
                print(f"  {currency}  ({country})")
                print(f"{'─'*52}")
                for indicator in target_ind:
                    slug = INDICATORS[indicator]
                    freq = INDICATOR_FREQ[indicator]
                    done += 1
                    requests_since_refresh += 1
                    print(f"\n  [{done}/{total}] {currency}/{indicator}")

                    existing     = load_existing(currency, indicator)
                    existing_obs = existing.get("observations", []) if existing else []
                    if skip_existing and len(existing_obs) >= 20:
                        print(f"    → Skip ({len(existing_obs)} obs existentes)")
                        requests_since_refresh -= 1  # No consumimos un request
                        continue

                    # Recrear context cada N requests para limpiar estado/cookies
                    if requests_since_refresh >= CONTEXT_REFRESH_EVERY:
                        print(f"\n  ⟳ Refreshing browser context (anti-tracking)...")
                        ctx.close()
                        browser.close()
                        random_delay(5, 10)
                        browser, ctx = create_context(pw)
                        requests_since_refresh = 0

                    new_obs = scrape_page(ctx, country, slug, freq, indicator_name=indicator)
                    if new_obs:
                        final = merge_observations(existing_obs, new_obs)
                        save_observations(currency, indicator, final)
                    else:
                        errors += 1
                        print(f"    ✗ Sin datos para {currency}/{indicator}")
                    random_delay(3.5, 7.0)
                random_delay(6, 12)
            
            scrape_cot_all(ctx)
        finally:
            ctx.close()
            browser.close()

    print("\n" + "=" * 62)
    print(f"COMPLETADO — {done} URLs | {errors} errores")
    print("=" * 62)


def show_summary():
    print("\n── Resumen economic-data-history/ ──")
    all_ind = list(INDICATORS.keys()) + ["cotPositioning"]
    total_ok = 0
    total_missing = 0
    for currency in CURRENCIES:
        print(f"\n  {currency}:")
        for ind in all_ind:
            path = f"{OUTPUT_DIR}/{currency}/{ind}.json"
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                obs = d.get("observations", [])
                if obs:
                    dates = sorted(o["date"] for o in obs)
                    print(f"    {ind:25s}: {len(obs):4d} obs  [{dates[0]} → {dates[-1]}]")
                    total_ok += 1
                else:
                    print(f"    {ind:25s}: vacío")
                    total_missing += 1
            else:
                print(f"    {ind:25s}: —")
                total_missing += 1
    
    print(f"\n  TOTAL: {total_ok} indicadores con datos | {total_missing} faltantes")
    print(f"  Cobertura: {100*total_ok/(total_ok+total_missing):.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper histórico TE v3.0 (Playwright)")
    parser.add_argument("--currency",  "-c", nargs="+", choices=CURRENCIES)
    parser.add_argument("--indicator", "-i", nargs="+", choices=list(INDICATORS.keys()))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary",       action="store_true")
    args = parser.parse_args()
    if args.summary:
        show_summary()
    else:
        run(currencies=args.currency, indicators=args.indicator, skip_existing=args.skip_existing)
