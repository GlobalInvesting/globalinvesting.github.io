#!/usr/bin/env python3
"""
fetch_historical_econ_data.py
─────────────────────────────
Scraper de datos históricos económicos desde Trading Economics usando
Playwright + Chromium para simular un navegador real.

Descarga series históricas completas (2020–hoy) para los 13 indicadores
del modelo de scoring forex v6.3, para las 8 divisas G10.

SALIDA:
  economic-data-history/{CURRENCY}/{indicator}.json
  Formato:
    {
      "currency": "USD",
      "indicator": "gdpGrowth",
      "source": "TradingEconomics (Playwright)",
      "fetched": "2026-03-10",
      "observations": [
        {"date": "2024-12-15", "value": 2.3},
        ...
      ]
    }

USO LOCAL:
  pip install playwright
  playwright install chromium
  python3 scripts/fetch_historical_econ_data.py
  python3 scripts/fetch_historical_econ_data.py --currency USD
  python3 scripts/fetch_historical_econ_data.py --indicator gdpGrowth
  python3 scripts/fetch_historical_econ_data.py --currency USD --indicator inflation

USO EN GITHUB ACTIONS:
  Ver .github/workflows/fetch-historical-econ-data.yml
"""

import json
import os
import re
import sys
import time
import random
import argparse
import csv
from datetime import date, datetime
from io import StringIO

# ─── Configuración ────────────────────────────────────────────────────────────

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

# Indicadores del modelo v6.3 y sus slugs en Trading Economics
INDICATORS = {
    "gdpGrowth":         "gdp-growth-rate",
    "inflation":         "inflation-rate",
    "unemployment":      "unemployment-rate",
    "currentAccount":    "current-account-to-gdp",
    "production":        "industrial-production",
    "tradeBalance":      "balance-of-trade",
    "retailSales":       "retail-sales",
    "wageGrowth":        "wage-growth",
    "manufacturingPMI":  "manufacturing-pmi",
    "servicesPMI":       "services-pmi",
    "bond10y":           "government-bond-yield",
    "consumerConfidence":"consumer-confidence",
    "businessConfidence":"business-confidence",
}

# Frecuencia esperada de cada indicador (para parseo de fechas)
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

OUTPUT_DIR = "economic-data-history"
TE_BASE    = "https://tradingeconomics.com"

# ─── Helpers ──────────────────────────────────────────────────────────────────

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
    data = {
        "currency":     currency,
        "indicator":    indicator,
        "source":       "TradingEconomics (Playwright/Chromium)",
        "fetched":      date.today().isoformat(),
        "observations": observations,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    ✓ Saved {len(observations)} obs → {path}")

def clean_num(text):
    if not text:
        return None
    text = str(text).strip().replace(",", "").replace("%", "").replace(" ", "")
    m = re.search(r"(-?\d+\.?\d*)", text)
    return float(m.group(1)) if m else None

def parse_te_date(date_text, freq="monthly"):
    """
    Parsea fechas de Trading Economics a formato ISO YYYY-MM-DD.
    TE usa formatos como: 'Mar/25', 'Q4/24', '2024', 'Mar 2025'
    """
    if not date_text:
        return None
    date_text = date_text.strip()

    # 'Mar/25' o 'Mar/2025'
    m = re.match(r"^([A-Za-z]{3})[/\s](\d{2,4})$", date_text)
    if m:
        yr = m.group(2)
        if len(yr) == 2:
            yr = "20" + yr
        try:
            dt = datetime.strptime(f"{m.group(1)} {yr}", "%b %Y")
            return dt.strftime("%Y-%m-15")
        except ValueError:
            pass

    # 'Q4/24' o 'Q4 2024'
    m = re.match(r"Q(\d)[/\s](\d{2,4})", date_text)
    if m:
        q  = int(m.group(1))
        yr = m.group(2)
        if len(yr) == 2:
            yr = "20" + yr
        month = (q - 1) * 3 + 2  # Q1→Feb, Q2→May, Q3→Aug, Q4→Nov
        return f"{yr}-{month:02d}-15"

    # Año solo: '2024'
    if re.match(r"^\d{4}$", date_text):
        return f"{date_text}-06-15"

    # ISO completo: '2024-03-15'
    m = re.match(r"^(\d{4}-\d{2}-\d{2})$", date_text)
    if m:
        return m.group(1)

    # 'March 2025' o 'March/2025'
    m = re.match(r"^([A-Za-z]+)[/\s](\d{4})$", date_text)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%B %Y")
            return dt.strftime("%Y-%m-15")
        except ValueError:
            pass

    return None

def merge_observations(existing_obs, new_obs):
    """
    Combina observaciones existentes con nuevas, eliminando duplicados por fecha.
    Las nuevas sobreescriben las existentes en caso de conflicto.
    """
    merged = {o["date"]: o["value"] for o in existing_obs}
    for o in new_obs:
        if o.get("date") and o.get("value") is not None:
            merged[o["date"]] = o["value"]
    return [{"date": d, "value": v} for d, v in sorted(merged.items())]

def random_delay(min_s=2.0, max_s=5.0):
    """Delay aleatorio para simular comportamiento humano."""
    time.sleep(random.uniform(min_s, max_s))

# ─── Playwright scraper ───────────────────────────────────────────────────────

def create_browser_context(playwright):
    """
    Crea un contexto de Chromium con configuración anti-detección.
    """
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--window-size=1366,768",
        ],
    )
    context = browser.new_context(
        viewport={"width": 1366, "height": 768},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        timezone_id="America/New_York",
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "DNT":             "1",
        },
    )
    # Ocultar webdriver flag
    context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
    """)
    return browser, context


def try_csv_download(page, url):
    """
    Intenta descargar el CSV histórico directamente desde la página TE.
    TE expone un botón de descarga que apunta a /chart-data o similar.
    Retorna lista de {date, value} o None.
    """
    observations = []
    try:
        # Buscar el enlace de descarga CSV en la página
        # TE usa un botón con texto 'Download' o un <a> con href que contiene 'download'
        download_links = page.locator("a[href*='download'], a[href*='csv'], button:has-text('Download')")
        count = download_links.count()
        if count > 0:
            # Interceptar la descarga
            with page.expect_download(timeout=15000) as download_info:
                download_links.first.click()
            download = download_info.value
            csv_path = f"/tmp/te_download_{random.randint(1000,9999)}.csv"
            download.save_as(csv_path)
            with open(csv_path) as f:
                content = f.read()
            os.unlink(csv_path)
            observations = parse_csv_content(content)
            if observations:
                print(f"      → CSV download: {len(observations)} rows")
                return observations
    except Exception as e:
        print(f"      → CSV download failed: {e}")
    return None


def try_chart_data_api(page, country_slug, indicator_slug):
    """
    TE expone datos del gráfico via endpoint /chart-data o API interna.
    Interceptamos las requests XHR para capturar los datos JSON del gráfico.
    """
    observations = []
    captured = []

    def handle_response(response):
        url = response.url
        # TE carga datos del chart via endpoints como:
        # /chart-data?s=...&d1=...&d2=...
        # /api/...
        # o via tradingeconomics.com/embed
        if any(kw in url for kw in ["chart-data", "/api/", "chartdata", "embed?s="]):
            try:
                body = response.json()
                if isinstance(body, list) and len(body) > 5:
                    captured.append(body)
                elif isinstance(body, dict):
                    # Buscar arrays de datos dentro del dict
                    for k, v in body.items():
                        if isinstance(v, list) and len(v) > 5:
                            captured.append(v)
            except Exception:
                pass

    page.on("response", handle_response)

    # Hacer scroll para activar la carga lazy del gráfico
    page.evaluate("window.scrollTo(0, 400)")
    page.wait_for_timeout(2000)
    page.evaluate("window.scrollTo(0, 800)")
    page.wait_for_timeout(2000)

    page.remove_listener("response", handle_response)

    # Procesar datos capturados
    for data_array in captured:
        obs = parse_chart_json(data_array)
        if obs:
            observations.extend(obs)
            print(f"      → XHR chart data: {len(obs)} rows")
            return observations

    return None


def parse_csv_content(content):
    """Parsea el contenido CSV de TE."""
    observations = []
    try:
        reader = csv.DictReader(StringIO(content))
        for row in reader:
            # TE CSV tiene columnas: Date, Value (o similar)
            date_val = row.get("Date") or row.get("date") or row.get("Reference")
            value_raw = row.get("Value") or row.get("value") or row.get("Actual") or row.get("Last")
            if not date_val or not value_raw:
                # Intentar con la primera y segunda columna
                keys = list(row.keys())
                if len(keys) >= 2:
                    date_val  = row[keys[0]]
                    value_raw = row[keys[1]]
            dt    = parse_te_date(date_val)
            value = clean_num(value_raw)
            if dt and value is not None:
                observations.append({"date": dt, "value": value})
    except Exception as e:
        print(f"      → CSV parse error: {e}")
    return observations


def parse_chart_json(data_array):
    """
    Parsea el array JSON del gráfico de TE.
    Formato típico: [["2024-01-01", 2.3], ...] o [{"Date":"...","Value":...}, ...]
    """
    observations = []
    if not data_array:
        return observations

    for item in data_array:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                dt    = parse_te_date(str(item[0]))
                value = clean_num(str(item[1]))
            elif isinstance(item, dict):
                dt    = parse_te_date(item.get("Date") or item.get("date") or item.get("DateTime") or "")
                value = clean_num(item.get("Value") or item.get("value") or item.get("Last") or item.get("Close") or "")
            else:
                continue

            if dt and value is not None:
                observations.append({"date": dt, "value": value})
        except Exception:
            continue

    return observations


def scrape_table_from_page(page, indicator, freq):
    """
    Extrae la tabla de datos históricos visible en la página TE.
    TE muestra la historia completa en una tabla con columnas: Date, Actual/Value, Previous, etc.
    """
    observations = []
    try:
        # Esperar a que cargue alguna tabla o contenido relevante
        page.wait_for_selector("table, #ctl00_ContentPlaceHolder1_ctl00_GridViewCalendar, .table", timeout=10000)

        # Extraer todas las tablas
        tables = page.query_selector_all("table")
        for table in tables:
            rows = table.query_selector_all("tr")
            if len(rows) < 3:
                continue

            # Detectar headers
            header_row = rows[0]
            headers = [th.inner_text().strip().lower() for th in header_row.query_selector_all("th, td")]

            # Buscar columnas de fecha y valor
            date_col  = next((i for i, h in enumerate(headers) if any(k in h for k in ["date", "reference", "period", "month", "quarter"])), None)
            value_col = next((i for i, h in enumerate(headers) if any(k in h for k in ["actual", "value", "last", "close"])), None)

            if date_col is None or value_col is None:
                # Intentar por posición: col 0 = date, col 1 = value
                if len(headers) >= 2:
                    date_col, value_col = 0, 1
                else:
                    continue

            count = 0
            for row in rows[1:]:
                cells = row.query_selector_all("td")
                if len(cells) <= max(date_col, value_col):
                    continue
                date_text  = cells[date_col].inner_text().strip()
                value_text = cells[value_col].inner_text().strip()
                dt    = parse_te_date(date_text, freq)
                value = clean_num(value_text)
                if dt and value is not None and dt >= "2020-01-01":
                    observations.append({"date": dt, "value": value})
                    count += 1

            if count > 3:
                print(f"      → Table scrape: {count} rows (headers: {headers[:4]})")
                return observations

    except Exception as e:
        print(f"      → Table scrape error: {e}")
    return observations


def scrape_from_embed_url(page, country_slug, indicator_slug):
    """
    TE tiene un endpoint embed con datos JSON:
    https://tradingeconomics.com/embed/?s=usaurtot&v=202403131513V20230410&type=line
    pero el parámetro 's' es un código interno. Lo intentamos vía la URL principal.
    También intentamos el endpoint de datos del gráfico directamente.
    """
    # Intentar el endpoint de descarga histórica de TE
    # Formato: /united-states/gdp-growth-rate#chart -> botón Download -> CSV
    observations = []

    # Navegar al tab 'Chart' que suele tener más historia
    try:
        chart_tab = page.locator("a[href*='#chart'], a:has-text('Chart'), #chart")
        if chart_tab.count() > 0:
            chart_tab.first.click()
            page.wait_for_timeout(2000)
    except Exception:
        pass

    # Buscar y hacer clic en botón de descarga
    download_selectors = [
        "a[href*='.csv']",
        "a:has-text('Download')",
        "a:has-text('CSV')",
        "button:has-text('Download')",
        "i.fa-download",
        ".download-button",
    ]
    for sel in download_selectors:
        try:
            btn = page.locator(sel)
            if btn.count() > 0:
                with page.expect_download(timeout=12000) as dl:
                    btn.first.click()
                download = dl.value
                csv_path = f"/tmp/te_{country_slug}_{indicator_slug}.csv"
                download.save_as(csv_path)
                with open(csv_path) as f:
                    content = f.read()
                os.unlink(csv_path)
                obs = parse_csv_content(content)
                if obs:
                    print(f"      → Download button ({sel}): {len(obs)} rows")
                    return obs
        except Exception:
            pass

    return observations


def scrape_indicator_page(context, country_slug, indicator_slug, currency, indicator, freq):
    """
    Scraping principal de una página de indicador en Trading Economics.
    Estrategia en cascada:
      1. XHR intercept (chart data API)
      2. Botón de descarga CSV
      3. Tabla HTML visible
    """
    url = f"{TE_BASE}/{country_slug}/{indicator_slug}"
    print(f"    → {url}")

    page = context.new_page()
    observations = []

    try:
        # Configurar intercepción XHR antes de navegar
        captured_xhr = []

        def on_response(response):
            url_r = response.url
            if any(kw in url_r for kw in [
                "chart-data", "chartdata", "/api/", "embed?s=",
                "download", ".csv", "getdata", "series"
            ]):
                ct = response.headers.get("content-type", "")
                if "json" in ct:
                    try:
                        body = response.json()
                        captured_xhr.append(("json", body))
                    except Exception:
                        pass
                elif "csv" in ct or "text/plain" in ct:
                    try:
                        body = response.text()
                        captured_xhr.append(("csv", body))
                    except Exception:
                        pass

        page.on("response", on_response)

        # Navegar a la página
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        random_delay(2.0, 4.0)

        # Scroll para activar carga lazy
        for scroll_y in [300, 600, 1000]:
            page.evaluate(f"window.scrollTo(0, {scroll_y})")
            page.wait_for_timeout(800)

        # 1. Procesar XHR capturados
        for data_type, body in captured_xhr:
            if data_type == "json":
                if isinstance(body, list):
                    obs = parse_chart_json(body)
                    if len(obs) > 5:
                        observations = obs
                        break
                elif isinstance(body, dict):
                    for k, v in body.items():
                        if isinstance(v, list) and len(v) > 5:
                            obs = parse_chart_json(v)
                            if len(obs) > 5:
                                observations = obs
                                break
            elif data_type == "csv":
                obs = parse_csv_content(body)
                if len(obs) > 5:
                    observations = obs
                    break

        if observations:
            print(f"      ✓ XHR: {len(observations)} observations")
        
        # 2. Intentar descarga CSV
        if not observations:
            obs = scrape_from_embed_url(page, country_slug, indicator_slug)
            if obs:
                observations = obs

        # 3. Scraping de tabla HTML
        if not observations:
            obs = scrape_table_from_page(page, indicator, freq)
            if obs:
                observations = obs

        # 4. Intentar navegar a la URL con #chart o ?g=10 para ampliar historia
        if len(observations) < 10:
            for suffix in ["?g=10", "?g=5", "#chart"]:
                try:
                    page.goto(url + suffix, wait_until="domcontentloaded", timeout=20000)
                    random_delay(2.0, 3.5)
                    for scroll_y in [400, 800]:
                        page.evaluate(f"window.scrollTo(0, {scroll_y})")
                        page.wait_for_timeout(600)
                    # Reintentar XHR
                    for data_type, body in captured_xhr:
                        if data_type == "json" and isinstance(body, list):
                            obs = parse_chart_json(body)
                            if len(obs) > len(observations):
                                observations = obs
                    if len(observations) >= 10:
                        break
                except Exception:
                    pass

    except Exception as e:
        print(f"      ✗ Page error: {e}")
    finally:
        page.remove_listener("response", on_response) if 'on_response' in dir() else None
        page.close()

    return observations


# ─── Scraper COT histórico ────────────────────────────────────────────────────

def scrape_cot_history(context):
    """
    Scraping del historial de posicionamiento COT/CFTC desde Trading Economics.
    Guarda en economic-data-history/{CURRENCY}/cotPositioning.json
    """
    COT_SLUGS = {
        "EUR": "euro-fx-cftc-net-speculative-positions",
        "GBP": "british-pound-cftc-net-speculative-positions",
        "JPY": "japanese-yen-cftc-net-speculative-positions",
        "AUD": "australian-dollar-cftc-net-speculative-positions",
        "CAD": "canadian-dollar-cftc-net-speculative-positions",
        "CHF": "swiss-franc-cftc-net-speculative-positions",
        "NZD": "new-zealand-dollar-cftc-net-speculative-positions",
    }

    print("\n── COT/CFTC Positioning History ──")
    for currency, slug in COT_SLUGS.items():
        existing = load_existing(currency, "cotPositioning")
        existing_obs = existing.get("observations", []) if existing else []

        url = f"{TE_BASE}/cot/{slug}"
        print(f"  {currency}: {url}")

        page = context.new_page()
        observations = []
        captured_xhr = []

        def on_response(response):
            if any(kw in response.url for kw in ["chart-data", "/api/", "embed", "download", ".csv"]):
                try:
                    ct = response.headers.get("content-type", "")
                    if "json" in ct:
                        captured_xhr.append(("json", response.json()))
                    elif "csv" in ct or "text" in ct:
                        captured_xhr.append(("csv", response.text()))
                except Exception:
                    pass

        page.on("response", on_response)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            random_delay(2.0, 4.0)
            for y in [300, 600, 1000]:
                page.evaluate(f"window.scrollTo(0, {y})")
                page.wait_for_timeout(700)

            for dtype, body in captured_xhr:
                if dtype == "json" and isinstance(body, list):
                    obs = parse_chart_json(body)
                    if len(obs) > len(observations):
                        observations = obs
                elif dtype == "csv":
                    obs = parse_csv_content(body)
                    if len(obs) > len(observations):
                        observations = obs

            if not observations:
                observations = scrape_table_from_page(page, "cotPositioning", "weekly")

        except Exception as e:
            print(f"    ✗ {e}")
        finally:
            page.close()

        if observations:
            final_obs = merge_observations(existing_obs, observations)
            save_observations(currency, "cotPositioning", final_obs)
        else:
            print(f"    ✗ No COT data for {currency}")

        random_delay(3.0, 6.0)


# ─── Función principal ────────────────────────────────────────────────────────

def run_scraper(currencies=None, indicators=None, skip_existing=False):
    """
    Ejecuta el scraping completo para las combinaciones de divisas/indicadores.
    
    Args:
        currencies:     lista de divisas a scrapear (None = todas)
        indicators:     lista de indicadores a scrapear (None = todos)
        skip_existing:  si True, salta pares que ya tienen datos
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: Playwright no está instalado.")
        print("Ejecuta: pip install playwright && playwright install chromium")
        sys.exit(1)

    ensure_dirs()

    target_currencies  = currencies  or CURRENCIES
    target_indicators  = indicators  or list(INDICATORS.keys())

    total  = len(target_currencies) * len(target_indicators)
    done   = 0
    errors = 0

    print("=" * 60)
    print("HISTORICAL ECON DATA SCRAPER — Playwright/Chromium")
    print(f"Divisas: {target_currencies}")
    print(f"Indicadores: {target_indicators}")
    print(f"Total URLs: {total}")
    print("=" * 60)

    with sync_playwright() as pw:
        browser, context = create_browser_context(pw)

        try:
            for currency in target_currencies:
                country_slug = COUNTRY_SLUGS[currency]
                print(f"\n{'─'*50}")
                print(f"  {currency} ({country_slug})")
                print(f"{'─'*50}")

                for indicator in target_indicators:
                    indicator_slug = INDICATORS[indicator]
                    freq           = INDICATOR_FREQ[indicator]
                    done          += 1
                    print(f"\n  [{done}/{total}] {currency}/{indicator}")

                    # Cargar datos existentes para merge incremental
                    existing = load_existing(currency, indicator)
                    existing_obs = existing.get("observations", []) if existing else []

                    if skip_existing and len(existing_obs) >= 20:
                        print(f"    → Skip (ya tiene {len(existing_obs)} obs)")
                        continue

                    observations = scrape_indicator_page(
                        context, country_slug, indicator_slug,
                        currency, indicator, freq
                    )

                    if observations:
                        # Filtrar: solo datos desde 2020
                        observations = [o for o in observations if o["date"] >= "2020-01-01"]
                        # Merge con existentes
                        final_obs = merge_observations(existing_obs, observations)
                        save_observations(currency, indicator, final_obs)
                    else:
                        errors += 1
                        print(f"    ✗ Sin datos para {currency}/{indicator}")

                    # Pausa entre requests para no ser detectado
                    random_delay(3.0, 7.0)

                # Pausa mayor entre divisas
                random_delay(5.0, 10.0)

            # Scraping COT histórico
            scrape_cot_history(context)

        finally:
            context.close()
            browser.close()

    print("\n" + "=" * 60)
    print(f"COMPLETADO: {done} URLs procesadas, {errors} errores")
    print(f"Datos guardados en: {OUTPUT_DIR}/")
    print("=" * 60)


def build_summary():
    """
    Genera un resumen de los datos disponibles en economic-data-history/.
    """
    print("\n── Resumen de datos históricos disponibles ──")
    for currency in CURRENCIES:
        print(f"\n  {currency}:")
        for indicator in list(INDICATORS.keys()) + ["cotPositioning"]:
            path = f"{OUTPUT_DIR}/{currency}/{indicator}.json"
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                obs = d.get("observations", [])
                if obs:
                    dates = sorted(o["date"] for o in obs)
                    print(f"    {indicator:25s}: {len(obs):4d} obs  [{dates[0]} → {dates[-1]}]")
                else:
                    print(f"    {indicator:25s}: VACÍO")
            else:
                print(f"    {indicator:25s}: NO EXISTE")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scraper histórico de datos económicos desde Trading Economics (Playwright)"
    )
    parser.add_argument(
        "--currency", "-c",
        nargs="+",
        choices=CURRENCIES,
        help="Divisas a scrapear (default: todas)"
    )
    parser.add_argument(
        "--indicator", "-i",
        nargs="+",
        choices=list(INDICATORS.keys()),
        help="Indicadores a scrapear (default: todos)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Saltar indicadores que ya tienen ≥20 observaciones"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Solo mostrar resumen de datos disponibles (sin scrapear)"
    )
    args = parser.parse_args()

    if args.summary:
        build_summary()
    else:
        run_scraper(
            currencies=args.currency,
            indicators=args.indicator,
            skip_existing=args.skip_existing,
        )
