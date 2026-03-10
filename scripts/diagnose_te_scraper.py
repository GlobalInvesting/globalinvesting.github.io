#!/usr/bin/env python3
"""
diagnose_te_scraper.py
──────────────────────
Script de diagnóstico para entender por qué fallan ciertos indicadores de TE.

Ejecutar ANTES del scraper completo para validar que todo funciona:
  python3 scripts/diagnose_te_scraper.py

O para solo testear indicadores específicos:
  python3 scripts/diagnose_te_scraper.py --currency USD --indicator gdpGrowth inflation

Salida:
  - Por cada URL: estado HTTP, presencia de Highcharts, datos encontrados
  - Resumen de qué funciona y qué no
  - Recomendaciones específicas

Esto NO modifica ningún archivo de datos.
"""

import sys, time, random, json
from datetime import datetime

COUNTRY_SLUGS = {
    "USD": "united-states", "EUR": "euro-area", "GBP": "united-kingdom",
    "JPY": "japan",         "AUD": "australia", "CAD": "canada",
    "CHF": "switzerland",   "NZD": "new-zealand",
}

INDICATORS = {
    "gdpGrowth":    "gdp-growth-rate",
    "inflation":    "inflation-rate",
    "unemployment": "unemployment-rate",
    "bond10y":      "government-bond-yield",
}

TE_BASE = "https://tradingeconomics.com"

JS_HC = """() => {
    if (!window.Highcharts || !window.Highcharts.charts) return {found: false, reason: 'no Highcharts'};
    const charts = window.Highcharts.charts.filter(c => c);
    if (!charts.length) return {found: false, reason: 'empty charts array'};
    let maxPts = 0;
    for (const chart of charts) {
        for (const s of (chart.series || [])) {
            maxPts = Math.max(maxPts, (s.data || s.points || []).length);
        }
    }
    return {found: maxPts > 0, pts: maxPts, charts: charts.length};
}"""

def check_url(page, url, wait_ms=5000):
    result = {"url": url, "status": None, "hc": None, "overlay": None, "error": None}
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=30000)
        result["status"] = resp.status if resp else "no_response"
        page.wait_for_timeout(wait_ms)
        
        # Check for overlays/paywalls
        has_modal = page.evaluate("""() => {
            const sels = ['.modal', '#subscribe-modal', '.paywall', '[class*="paywall"]'];
            return sels.some(s => {
                const el = document.querySelector(s);
                return el && getComputedStyle(el).display !== 'none';
            });
        }""")
        result["overlay"] = has_modal
        
        # Check Highcharts
        hc = page.evaluate(JS_HC)
        result["hc"] = hc
        
        # Check if page content indicates blocking
        content_check = page.evaluate("""() => {
            const text = document.body ? document.body.innerText.toLowerCase() : '';
            return {
                captcha: text.includes('captcha') || text.includes('cloudflare'),
                blocked: text.includes('access denied') || text.includes('403'),
                login: text.includes('sign in') || text.includes('log in') || text.includes('subscribe'),
                hasChart: document.querySelectorAll('[class*="highcharts"], svg.highcharts-root').length > 0,
            };
        }""")
        result["content"] = content_check
        
    except Exception as e:
        result["error"] = str(e)[:100]
    return result

def run_diagnostics(currencies=None, indicators=None):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: pip install playwright && playwright install chromium")
        sys.exit(1)

    target_cur = currencies or list(COUNTRY_SLUGS.keys())[:3]  # Default: first 3
    target_ind = indicators or list(INDICATORS.keys())
    
    results = []
    
    print("=" * 60)
    print("TE SCRAPER DIAGNOSTICS")
    print(f"Testing {len(target_cur)} currencies × {len(target_ind)} indicators")
    print("=" * 60)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        )
        ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        
        for currency in target_cur:
            country = COUNTRY_SLUGS[currency]
            print(f"\n  {currency} ({country}):")
            
            for ind_name, ind_slug in INDICATORS.items():
                if ind_name not in target_ind:
                    continue
                
                url = f"{TE_BASE}/{country}/{ind_slug}"
                page = ctx.new_page()
                
                try:
                    r = check_url(page, url, wait_ms=6000)
                    results.append({"currency": currency, "indicator": ind_name, **r})
                    
                    hc = r.get("hc", {}) or {}
                    pts = hc.get("pts", 0) if isinstance(hc, dict) else 0
                    overlay = r.get("overlay", False)
                    content = r.get("content", {}) or {}
                    
                    status_icon = "✓" if pts >= 5 else ("⚠" if pts > 0 else "✗")
                    overlay_str = " [OVERLAY]" if overlay else ""
                    blocked_str = " [BLOCKED]" if content.get("blocked") or content.get("captcha") else ""
                    login_str = " [LOGIN_WALL]" if content.get("login") else ""
                    
                    print(f"    {status_icon} {ind_name:20s} HTTP={r['status']} HC={pts}pts{overlay_str}{blocked_str}{login_str}")
                    
                    if r.get("error"):
                        print(f"      ERROR: {r['error']}")
                
                finally:
                    page.close()
                
                time.sleep(random.uniform(2, 4))
        
        ctx.close()
        browser.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    ok = [r for r in results if isinstance(r.get("hc"), dict) and r["hc"].get("pts", 0) >= 5]
    fail = [r for r in results if not (isinstance(r.get("hc"), dict) and r["hc"].get("pts", 0) >= 5)]
    
    print(f"\n  ✓ Working: {len(ok)}/{len(results)}")
    if fail:
        print(f"  ✗ Failed:  {len(fail)}/{len(results)}")
        print("\n  Failed indicators:")
        for r in fail:
            hc = r.get("hc") or {}
            content = r.get("content") or {}
            reason = "unknown"
            if r.get("error"):
                reason = f"timeout/error: {r['error'][:50]}"
            elif content.get("captcha"):
                reason = "CAPTCHA detected"
            elif content.get("blocked"):
                reason = "Access blocked (403)"
            elif content.get("login"):
                reason = "Login/paywall wall"
            elif isinstance(hc, dict) and hc.get("pts", 0) == 0:
                reason = f"Highcharts found ({hc.get('charts',0)} charts) but 0 data points"
            elif not isinstance(hc, dict) or not hc.get("found"):
                reason = f"Highcharts not found: {hc.get('reason','') if isinstance(hc,dict) else hc}"
            print(f"    - {r['currency']}/{r['indicator']}: {reason}")
    
    print("\n  RECOMMENDATIONS:")
    overlay_fails = [r for r in fail if r.get("overlay")]
    if overlay_fails:
        print("  → Overlays/modals detected: dismiss_overlays() fix is needed ✓ (included in v3.0)")
    
    login_fails = [r for r in fail if (r.get("content") or {}).get("login")]
    if login_fails:
        print("  → Login walls detected: these indicators may require TE free account cookies")
        print("    Workaround: use the country-list URL instead of individual pages")
    
    if len(ok) / max(len(results), 1) > 0.7:
        print("  → >70% success rate: scraper should work. Run full scraper.")
    else:
        print("  → <70% success rate: TE may be rate-limiting this IP.")
        print("    Wait 1 hour and retry, or use a different runner region in Actions.")
    
    # Save results
    output_path = "/tmp/te_diagnostics.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {"ok": len(ok), "fail": len(fail), "total": len(results)}
        }, f, indent=2)
    print(f"\n  Full results saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--currency",  "-c", nargs="+", choices=list(COUNTRY_SLUGS.keys()))
    parser.add_argument("--indicator", "-i", nargs="+", choices=list(INDICATORS.keys()))
    args = parser.parse_args()
    run_diagnostics(currencies=args.currency, indicators=args.indicator)
