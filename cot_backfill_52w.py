#!/usr/bin/env python3
"""
COT HISTORICAL BACKFILL — 52 WEEKS
====================================
Descarga archivos históricos anuales del CFTC y construye 52 semanas
de historial para cada divisa en cot-data/*.json.

CFTC publica ZIPs anuales en:
  Options+Futures: https://www.cftc.gov/files/dea/history/fin_com_txt_YYYY.zip
  Futures Only:    https://www.cftc.gov/files/dea/history/fin_fut_txt_YYYY.zip

Usa el MISMO parser (extract_tff_block) que el workflow de producción.

USO (desde la raíz del repo globalinvesting.github.io):
  pip install requests
  python3 scripts/cot_backfill_52w.py

  # Dry run (no escribe archivos):
  python3 scripts/cot_backfill_52w.py --dry-run

  # Ver qué semanas quedarían:
  python3 scripts/cot_backfill_52w.py --dry-run --verbose
"""

import argparse
import io
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from datetime import date, datetime

import requests

# ── Configuración ────────────────────────────────────────────────────────────

TARGET_WEEKS  = 52
OUTPUT_DIR    = "cot-data"
CURRENT_YEAR  = date.today().year
YEARS_TO_FETCH = [CURRENT_YEAR - 1, CURRENT_YEAR]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; COT-Backfill-Bot/1.0; "
        "+https://globalinvesting.github.io/)"
    )
}

CURRENCY_NAMES = {
    "USD": [
        "USD INDEX - ICE FUTURES U.S.",
        "U.S. DOLLAR INDEX - ICE FUTURES U.S.",
        "US DOLLAR INDEX - ICE FUTURES U.S.",
    ],
    "EUR": ["EURO FX - CHICAGO MERCANTILE EXCHANGE"],
    "GBP": ["BRITISH POUND - CHICAGO MERCANTILE EXCHANGE"],
    "JPY": ["JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE"],
    "CAD": ["CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE"],
    "CHF": ["SWISS FRANC - CHICAGO MERCANTILE EXCHANGE"],
    "AUD": ["AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE"],
    "NZD": [
        "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE",
        "N.Z. DOLLAR - CHICAGO MERCANTILE EXCHANGE",
        "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE",
        "NEW ZEALAND DOLLAR - CME",
    ],
}

# ── Helpers (idénticos al workflow de producción) ────────────────────────────

def parse_numbers(line):
    return [int(s.replace(",", "")) for s in re.findall(r"[\d,]+", line)]


def extract_week_ending_from_block(snippet):
    """Extrae la fecha 'As of' del bloque de una divisa."""
    for pattern in (
        r"[Aa]s\s+of\s+\w+,\s+(\w+\s+\d+,\s*\d{4})",
        r"[Pp]ositions\s+as\s+of\s+(\w+\s+\d+,\s*\d{4})",
        r"(\w+\s+\d+,\s*\d{4})",
    ):
        m = re.search(pattern, snippet[:500])
        if m:
            try:
                return datetime.strptime(m.group(1).strip(), "%B %d, %Y").strftime("%Y-%m-%d")
            except Exception:
                pass
    return None


def extract_tff_block(content, currency_code, aliases, verbose=False):
    """
    Parser TFF Disaggregated — idéntico al workflow de producción.
    Columnas después de 'Positions':
      [0] Dealer Long   [1] Dealer Short   [2] Dealer Spreading
      [3] AM Long       [4] AM Short       [5] AM Spreading
      [6] Lev Long      [7] Lev Short      [8] Lev Spreading
      ...
    """
    for name in aliases:
        m = re.compile(re.escape(name.upper()), re.IGNORECASE | re.DOTALL).search(content)
        if not m:
            continue

        snippet = content[m.start(): m.start() + 4000]
        lines   = snippet.splitlines()

        # Extraer fecha del bloque
        week_ending = extract_week_ending_from_block(snippet)

        for i, line in enumerate(lines):
            if re.match(r"\s*Positions\s*$", line, re.IGNORECASE):
                collected = []
                for j in range(i + 1, min(i + 8, len(lines))):
                    raw  = lines[j].strip()
                    if not raw:
                        continue
                    nums = parse_numbers(raw)
                    if not nums:
                        break
                    collected.extend(nums)
                    if len(collected) >= 14:
                        break

                if len(collected) < 8:
                    continue

                lev_long  = collected[6]
                lev_short = collected[7]
                lev_net   = lev_long - lev_short

                am_long  = collected[3] if len(collected) > 4 else None
                am_short = collected[4] if len(collected) > 4 else None
                am_net   = (am_long - am_short) if (am_long is not None and am_short is not None) else None

                dd_long  = collected[0] if len(collected) > 1 else None
                dd_short = collected[1] if len(collected) > 1 else None
                dd_net   = (dd_long - dd_short) if (dd_long is not None and dd_short is not None) else None

                return {
                    "weekEnding":   week_ending,
                    "leveraged":    {"long": lev_long,  "short": lev_short, "net": lev_net},
                    "assetManager": {"long": am_long,   "short": am_short,  "net": am_net},
                    "dealer":       {"long": dd_long,   "short": dd_short,  "net": dd_net},
                }
    return None


def extract_all_weeks_from_annual(content, verbose=False):
    """
    Un archivo histórico anual contiene MÚLTIPLES semanas concatenadas,
    cada una con el mismo formato que el archivo semanal actual.
    Divide el contenido en bloques por divisa+fecha y parsea cada uno.
    """
    results = []  # list of { weekEnding, ccy, levLong, levShort, levNet, assetManagerNet, dealerNet }

    # El formato anual repite el bloque de cada divisa para cada semana.
    # Estrategia: encontrar todos los bloques "EURO FX ..." y parsear cada uno.

    for ccy, aliases in CURRENCY_NAMES.items():
        all_positions = []  # posiciones de inicio de todos los bloques de esta divisa

        for name in aliases:
            pattern = re.compile(re.escape(name.upper()), re.IGNORECASE)
            for m in pattern.finditer(content):
                all_positions.append(m.start())

        # Deduplicar y ordenar
        all_positions = sorted(set(all_positions))

        if verbose:
            print(f"  {ccy}: {len(all_positions)} ocurrencias encontradas")

        for pos in all_positions:
            snippet = content[pos: pos + 4000]
            block   = extract_tff_block(snippet, ccy, aliases, verbose=False)
            if block is None:
                continue

            week_ending = block.get("weekEnding")
            if not week_ending:
                # Intentar extraer fecha del snippet directamente
                week_ending = extract_week_ending_from_block(snippet)
            if not week_ending:
                continue

            lev = block["leveraged"]
            am  = block["assetManager"]
            dd  = block["dealer"]

            results.append({
                "weekEnding":      week_ending,
                "ccy":             ccy,
                "levLong":         lev["long"],
                "levShort":        lev["short"],
                "levNet":          lev["net"],
                "assetManagerNet": am["net"],
                "dealerNet":       dd["net"],
            })

    return results


# ── Descarga ─────────────────────────────────────────────────────────────────

def fetch_annual_zip(year, report_type="com"):
    url = f"https://www.cftc.gov/files/dea/history/fin_{report_type}_txt_{year}.zip"
    print(f"  GET {url}", end="", flush=True)
    try:
        r = requests.get(url, headers=HEADERS, timeout=90)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            txt_files = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not txt_files:
                print(f" ✗ (sin .txt en el ZIP)")
                return None
            content = z.read(txt_files[0]).decode("latin-1")
        print(f" ✓ ({len(r.content)//1024}KB, {len(content):,} chars)")
        return content
    except requests.HTTPError as e:
        print(f" ✗ HTTP {e.response.status_code}")
        return None
    except Exception as e:
        print(f" ✗ {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COT Historical Backfill — 52 weeks")
    parser.add_argument("--weeks",      type=int, default=TARGET_WEEKS)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print(f"COT HISTORICAL BACKFILL — {args.weeks} weeks")
    print(f"Output: {args.output_dir}/   Dry-run: {args.dry_run}")
    print("=" * 65)

    # 1. Descargar y parsear archivos anuales
    all_blocks = []

    for year in YEARS_TO_FETCH:
        print(f"\n[{year}]")

        # Intentar Options+Futures primero, luego Futures Only
        for report_type in ("com", "fut"):
            content = fetch_annual_zip(year, report_type)
            if content:
                print(f"  Parseando ({report_type})...")
                blocks = extract_all_weeks_from_annual(content, verbose=args.verbose)
                print(f"  → {len(blocks)} registros ({len(set(b['weekEnding'] for b in blocks))} semanas únicas)")
                all_blocks.extend(blocks)
                break  # com exitoso, no necesitamos fut
        else:
            print(f"  ⚠ Sin datos para {year}")

    if not all_blocks:
        print("\nERROR: Sin datos. Verificá conexión a cftc.gov y volvé a intentar.")
        return 1

    # 2. Agrupar por divisa, deduplicar, ordenar
    by_ccy = defaultdict(dict)  # ccy → { weekEnding → block }
    for b in all_blocks:
        by_ccy[b["ccy"]][b["weekEnding"]] = b  # última entrada gana si hay dups

    print(f"\n{'='*65}")
    print("DATOS DESCARGADOS")
    print(f"{'='*65}")
    for ccy in sorted(CURRENCY_NAMES):
        entries = sorted(by_ccy[ccy].values(), key=lambda x: x["weekEnding"])
        if entries:
            print(f"  {ccy:3s}: {len(entries):2d} semanas  "
                  f"({entries[0]['weekEnding']} → {entries[-1]['weekEnding']})")
        else:
            print(f"  {ccy:3s}: ⚠ sin datos")

    # 3. Actualizar JSONs
    print(f"\n{'='*65}")
    print("ESCRIBIENDO ARCHIVOS JSON")
    print(f"{'='*65}")

    os.makedirs(args.output_dir, exist_ok=True)

    for ccy in CURRENCY_NAMES:
        path = os.path.join(args.output_dir, f"{ccy}.json")

        raw_entries = sorted(by_ccy[ccy].values(), key=lambda x: x["weekEnding"])
        if not raw_entries:
            print(f"  {ccy}: ⚠ sin datos — archivo sin cambios")
            continue

        # Construir lista history (formato idéntico al workflow de producción)
        new_history = [
            {
                "weekEnding":      b["weekEnding"],
                "levNet":          b["levNet"],
                "levLong":         b["levLong"],
                "levShort":        b["levShort"],
                "assetManagerNet": b["assetManagerNet"],
                "dealerNet":       b["dealerNet"],
            }
            for b in raw_entries
        ]

        # Leer archivo existente para preservar campos que el backfill no tiene
        # (assetManagerLong/Short, dealerLong/Short, sourceType, etc.)
        existing = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing = json.load(f)
            except Exception:
                pass

        # Combinar: historial nuevo + historial existente reciente (por si el
        # archivo existente tiene semanas más nuevas que el ZIP anual)
        existing_history = existing.get("history", [])
        combined = {h["weekEnding"]: h for h in new_history}
        for h in existing_history:
            if h["weekEnding"] not in combined:
                combined[h["weekEnding"]] = h

        history = sorted(combined.values(), key=lambda x: x["weekEnding"])[-args.weeks:]

        # La semana más reciente en el historial = datos root
        latest = history[-1]
        lev_net   = latest["levNet"]
        lev_long  = latest.get("levLong")
        lev_short = latest.get("levShort")

        updated = {
            "netPosition":    lev_net,
            "longPositions":  lev_long,
            "shortPositions": lev_short,
            "positionCategory": "Leveraged Funds (speculative)",
            "assetManagerNet":   latest.get("assetManagerNet"),
            "assetManagerLong":  existing.get("assetManagerLong"),
            "assetManagerShort": existing.get("assetManagerShort"),
            "dealerNet":   latest.get("dealerNet"),
            "dealerLong":  existing.get("dealerLong"),
            "dealerShort": existing.get("dealerShort"),
            "history":     history,
            "sourceType":  existing.get("sourceType", "options_futures_combined"),
            "source":      "CFTC Official",
            "sourceUrl":   "https://www.cftc.gov/dea/options/financial_lof.htm",
            "reportDate":  date.today().isoformat(),
            "lastUpdate":  date.today().isoformat(),
            "weekEnding":  latest["weekEnding"],
        }
        # Limpiar None (campos que no estaban en el archivo existente)
        updated = {k: v for k, v in updated.items() if v is not None}

        if args.dry_run:
            print(f"  {ccy}: [DRY-RUN] {len(history)}w  "
                  f"({history[0]['weekEnding']} → {history[-1]['weekEnding']})")
        else:
            with open(path, "w") as f:
                json.dump(updated, f, indent=2)
            print(f"  {ccy}: ✓ {len(history)}w  "
                  f"({history[0]['weekEnding']} → {history[-1]['weekEnding']})")

    print(f"\n{'='*65}")
    if args.dry_run:
        print("DRY-RUN completado. Ningún archivo fue modificado.")
    else:
        print("BACKFILL COMPLETADO.")
        print()
        print("Próximos pasos:")
        print("  1. Verificar: python3 -c \"import json; d=json.load(open('cot-data/EUR.json')); print(len(d['history']), 'semanas')\"")
        print("  2. git add cot-data/ && git commit -m 'backfill: 52w COT history' && git push")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
