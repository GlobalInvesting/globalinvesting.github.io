#!/usr/bin/env python3
"""
update_economic_calendar.py  v8.0
Fuente primaria: Finnhub Economic Calendar API (datos limpios, impact normalizado, UTC puro)
Fuente secundaria: Investing.com (enriquece con eventos que Finnhub no cubre)
"""

import requests
import json
import re
import os
import sys
from datetime import date, datetime, timedelta
from collections import defaultdict

# ── CONSTANTS ────────────────────────────────────────────────────────

TRACKED_CURRENCIES = {'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'}

CURRENCY_FLAGS = {
    'USD': '🇺🇸', 'EUR': '🇪🇺', 'GBP': '🇬🇧', 'JPY': '🇯🇵',
    'AUD': '🇦🇺', 'CAD': '🇨🇦', 'CHF': '🇨🇭', 'NZD': '🇳🇿',
}

# Finnhub devuelve country name — mapeamos a currency code
COUNTRY_TO_CURRENCY = {
    # USD
    'united states': 'USD', 'us': 'USD', 'usa': 'USD',
    # EUR — múltiples países
    'euro area': 'EUR', 'eurozone': 'EUR', 'european union': 'EUR',
    'germany': 'EUR', 'france': 'EUR', 'italy': 'EUR', 'spain': 'EUR',
    'netherlands': 'EUR', 'belgium': 'EUR', 'austria': 'EUR',
    'portugal': 'EUR', 'finland': 'EUR', 'ireland': 'EUR',
    'greece': 'EUR', 'luxembourg': 'EUR',
    # GBP
    'united kingdom': 'GBP', 'uk': 'GBP', 'great britain': 'GBP',
    'england': 'GBP', 'britain': 'GBP',
    # JPY
    'japan': 'JPY',
    # AUD
    'australia': 'AUD',
    # CAD
    'canada': 'CAD',
    # CHF
    'switzerland': 'CHF', 'swiss': 'CHF',
    # NZD
    'new zealand': 'NZD',
}

MONTH_ES = {
    1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic',
}

FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', '')
CALENDAR_PATH   = 'calendar-data/calendar.json'

# ── HELPERS ──────────────────────────────────────────────────────────

def fmt_date(dt_obj):
    return f"{dt_obj.day} {MONTH_ES[dt_obj.month]}"

def clean_val(v):
    if v is None: return ''
    s = str(v).strip()
    return '' if s.lower() in ('none', 'nan', '-', 'n/a', '') else s

def resolve_currency(country_str):
    if not country_str: return None
    t = country_str.strip().upper()
    if t in TRACKED_CURRENCIES: return t
    t_lower = country_str.strip().lower()
    for key, curr in COUNTRY_TO_CURRENCY.items():
        if key in t_lower: return curr
    return None

def fmt_number(value, unit):
    """Convierte float de Finnhub a string con unidad (e.g. 1.3 + '%' → '1.3%')"""
    if value is None: return ''
    try:
        v = float(value)
        # Formatear sin trailing zeros innecesarios
        s = f"{v:g}"
        u = (unit or '').strip()
        if u and u not in ('', 'None'):
            # Unidades que van pegadas al número
            if u in ('%', 'B', 'M', 'K', 'T'):
                return s + u
            # Unidades que van con espacio
            return s + ' ' + u
        return s
    except (ValueError, TypeError):
        return clean_val(value)

def impact_rank(impact):
    return {'high': 3, 'medium': 2, 'low': 1}.get(impact, 0)


# ════════════════════════════════════════════════════════════════════
# SOURCE 1: FINNHUB  (fuente primaria)
# ════════════════════════════════════════════════════════════════════

def fetch_finnhub_calendar(from_str, to_str, target_dates):
    """
    Finnhub /calendar/economic endpoint.
    - Retorna UTC puro (campo 'time' en ISO8601 UTC)
    - impact ya normalizado: 'high' / 'medium' / 'low'
    - actual/estimate/prev como floats con campo 'unit'
    """
    if not FINNHUB_API_KEY:
        print("  [Finnhub] ⚠️  No API key — skipping")
        return []

    url = 'https://finnhub.io/api/v1/calendar/economic'
    params = {
        'from':  from_str,
        'to':    to_str,
        'token': FINNHUB_API_KEY,
    }
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if not r.ok:
            print(f"  [Finnhub] ❌ HTTP {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
    except Exception as e:
        print(f"  [Finnhub] ❌ Request error: {e}")
        return []

    raw = data.get('economicCalendar', [])
    print(f"  [Finnhub] Raw events: {len(raw)}")

    events = []
    skipped_currency = skipped_noname = 0

    for ev in raw:
        try:
            # Resolver currency desde country
            country = ev.get('country', '')
            currency = resolve_currency(country)
            if not currency:
                skipped_currency += 1
                continue

            event_name = (ev.get('event') or '').strip()
            if not event_name:
                skipped_noname += 1
                continue

            # Parsear tiempo UTC
            time_iso = ev.get('time', '')
            event_date = None
            time_utc   = ''
            if time_iso:
                try:
                    dt = datetime.fromisoformat(time_iso.replace('Z', '+00:00'))
                    time_utc   = dt.strftime('%H:%M')
                    event_date = dt.date()
                    # Eventos 00:00-03:59 UTC = noche anterior en Investing
                    if dt.hour < 4:
                        event_date = event_date - timedelta(days=1)
                except Exception:
                    event_date = date.fromisoformat(from_str)

            if event_date not in target_dates:
                continue

            # Impact: Finnhub ya lo da normalizado
            impact = (ev.get('impact') or 'low').lower()
            if impact not in ('high', 'medium', 'low'):
                impact = 'low'

            # Valores numéricos → string con unidad
            unit = ev.get('unit', '')
            actual   = fmt_number(ev.get('actual'),   unit)
            forecast = fmt_number(ev.get('estimate'), unit)
            previous = fmt_number(ev.get('prev'),     unit)

            events.append({
                'date':     fmt_date(event_date),
                'dateISO':  event_date.isoformat(),
                'timeUTC':  time_utc,
                'country':  currency,
                'currency': currency,
                'flag':     CURRENCY_FLAGS.get(currency, ''),
                'event':    event_name,
                'impact':   impact,
                'actual':   actual,
                'forecast': forecast,
                'previous': previous,
                '_source':  'finnhub',
            })
        except Exception:
            continue

    print(f"  [Finnhub] ✅ Parsed {len(events)} events "
          f"(skipped: no_currency={skipped_currency}, no_name={skipped_noname})")
    return events


# ════════════════════════════════════════════════════════════════════
# SOURCE 2: INVESTING.COM  (fuente secundaria — solo enriquece)
# ════════════════════════════════════════════════════════════════════

def fetch_investing_calendar(from_str, to_str, target_dates):
    """
    Investing.com JSON API (__NEXT_DATA__).
    Se usa SOLO para agregar eventos que Finnhub no tiene.
    El impact de Investing NO sobreescribe el de Finnhub.
    """
    print(f"  [Investing] Fetching {from_str} → {to_str}")

    IMPACT_MAP = {
        '1': 'low', '2': 'medium', '3': 'high',
         1:  'low',  2:  'medium',  3:  'high',
    }

    url = 'https://www.investing.com/economic-calendar/'
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    params = {'dateFrom': from_str, 'dateTo': to_str}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if not r.ok:
            raise ValueError(f"HTTP {r.status_code}")

        m = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            r.text, re.DOTALL
        )
        if not m:
            raise ValueError("No __NEXT_DATA__ found")

        data  = json.loads(m.group(1))
        store = (data.get('props', {})
                     .get('pageProps', {})
                     .get('state', {})
                     .get('economicCalendarStore', {})
                     .get('calendarEventsByDate', {}))
        if not store:
            raise ValueError("Empty economicCalendarStore")

    except Exception as e:
        print(f"  [Investing] ❌ {e}")
        return []

    events = []
    for date_key, day_events in store.items():
        for ev in day_events:
            try:
                currency = str(ev.get('currency', '')).upper()
                if currency not in TRACKED_CURRENCIES:
                    continue

                time_iso = ev.get('time', '') or ev.get('actual_time', '')
                event_date = None
                time_utc   = ''
                if time_iso:
                    dt = datetime.fromisoformat(time_iso.replace('Z', '+00:00'))
                    time_utc   = dt.strftime('%H:%M')
                    event_date = dt.date()
                    if dt.hour < 4:
                        event_date = event_date - timedelta(days=1)
                else:
                    try:
                        event_date = date.fromisoformat(date_key)
                    except Exception:
                        continue

                if event_date not in target_dates:
                    continue

                imp_raw = ev.get('importance', '1')
                impact  = IMPACT_MAP.get(imp_raw, 'low')

                event_name = (ev.get('eventLong') or ev.get('event') or '').strip()
                if not event_name:
                    continue

                actual   = clean_val(str(ev.get('actual',   '') or ''))
                forecast = clean_val(str(ev.get('forecast', '') or ''))
                previous = clean_val(str(ev.get('previous', '') or ''))

                events.append({
                    'date':     fmt_date(event_date),
                    'dateISO':  event_date.isoformat(),
                    'timeUTC':  time_utc,
                    'country':  currency,
                    'currency': currency,
                    'flag':     CURRENCY_FLAGS.get(currency, ''),
                    'event':    event_name,
                    'impact':   impact,
                    'actual':   actual,
                    'forecast': forecast,
                    'previous': previous,
                    '_source':  'investing',
                })
            except Exception:
                continue

    print(f"  [Investing] ✅ Parsed {len(events)} events")
    return events


# ════════════════════════════════════════════════════════════════════
# MERGE: Finnhub primario + Investing complementario
# ════════════════════════════════════════════════════════════════════

def merge_sources(finnhub_evs, investing_evs):
    """
    Finnhub gana siempre en impact, actual, forecast, previous, timeUTC.
    Investing solo aporta eventos que Finnhub no tiene.
    """
    def make_key(ev):
        # Key corta (primeros 30 chars normalizados) para matching entre fuentes
        name = re.sub(r'\s*\([^)]*\)', '', ev['event'].lower()).strip()
        name = re.sub(r'[^a-z0-9 ]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()[:30]
        return (ev['dateISO'], ev['currency'], name)

    # Índice de Finnhub (fuente de verdad)
    finnhub_index = {}
    for ev in finnhub_evs:
        k = make_key(ev)
        # Si hay duplicado en Finnhub, quedar con el de mayor score
        if k not in finnhub_index or _score(ev) > _score(finnhub_index[k]):
            finnhub_index[k] = ev

    merged = dict(finnhub_index)  # copia

    # Agregar eventos de Investing que Finnhub no tiene
    added_from_investing = 0
    for ev in investing_evs:
        k = make_key(ev)
        if k not in merged:
            # Evento nuevo — agregarlo con impacto de Investing
            merged[k] = ev
            added_from_investing += 1
        else:
            # Evento ya en Finnhub — solo aportar actual/forecast/previous si falta
            existing = dict(merged[k])
            for field in ('actual', 'forecast', 'previous'):
                if not existing.get(field) and ev.get(field):
                    existing[field] = ev[field]
            # Investing puede tener el nombre más descriptivo (eventLong)
            if len(ev.get('event', '')) > len(existing.get('event', '')):
                existing['event'] = ev['event']
            merged[k] = existing

    print(f"  [Merge] Finnhub={len(finnhub_evs)}, Investing={len(investing_evs)}, "
          f"added_from_investing={added_from_investing}, total={len(merged)}")
    return list(merged.values())


def _score(ev):
    s = 0
    if ev.get('actual'):   s += 1000
    if ev.get('forecast'): s += 100
    if ev.get('previous'): s += 10
    s += len(ev.get('event', ''))
    return s


# ════════════════════════════════════════════════════════════════════
# DEDUP
# ════════════════════════════════════════════════════════════════════

def normalize_for_dedup(name):
    """Normalización agresiva para detectar duplicados semánticos."""
    n = name.lower().strip()
    n = re.sub(r'\(yoy\)', 'yoy', n); n = re.sub(r'\by/y\b', 'yoy', n)
    n = re.sub(r'\(mom\)', 'mom', n); n = re.sub(r'\bm/m\b', 'mom', n)
    n = re.sub(r'\(qoq\)', 'qoq', n); n = re.sub(r'\bq/q\b', 'qoq', n)
    n = re.sub(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|\d{4})\b', '', n)
    n = re.sub(r'\([^)]*\)', '', n)
    n = re.sub(r'\bgross domestic product\b', 'gdp', n)
    n = re.sub(r'\bconsumer price index\b', 'cpi', n)
    n = re.sub(r'\bpurchasing managers\b', 'pmi', n)
    n = re.sub(
        r'\b(japan|japanese|us|uk|germany|german|france|french|eurozone|euro area|'
        r'australia|australian|canada|canadian|swiss|switzerland|new zealand|u\.s\.)\b', '', n
    )
    sw = {
        'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'by', 'or',
        'rate', 'change', 'data', 'flash', 'final', 'prelim', 'preliminary',
        'revised', 'report', 'survey', 'release', 'monthly', 'quarterly',
        'annual', 'weekly', 'current', 'total', 'net', 'new', 'all', 'index',
        'q', 'y', 'm', 'sa', 'nsa',
    }
    words = [w for w in re.sub(r'[^a-z0-9 ]', ' ', n).split()
             if w not in sw and len(w) > 1]
    return ' '.join(sorted(words))


def dedup_events(events):
    """Elimina duplicados semánticos dentro del mismo día/currency."""
    # Step A: mismo slot exacto (dateISO + timeUTC + currency)
    slot_groups = defaultdict(list)
    for ev in events:
        slot = (ev['dateISO'], ev.get('timeUTC', ''), ev['currency'])
        slot_groups[slot].append(ev)

    after_slot = []
    for slot, group in slot_groups.items():
        if len(group) == 1:
            after_slot.append(group[0])
            continue
        # Quedarse con el mejor scored
        best = max(group, key=_score)
        best_impact = max(group, key=lambda e: impact_rank(e.get('impact', 'low')))
        best = dict(best)
        best['impact'] = best_impact['impact']
        for ev in group:
            for field in ('actual', 'forecast', 'previous'):
                if not best.get(field) and ev.get(field):
                    best[field] = ev[field]
        after_slot.append(best)

    # Step B: mismo día, mismo currency, nombre semánticamente igual
    semantic_groups = defaultdict(list)
    for ev in after_slot:
        norm = normalize_for_dedup(ev['event'])
        semantic_groups[(ev['dateISO'], ev['currency'], norm)].append(ev)

    final = []
    dominated = set()
    merged_count = 0
    for group_key, group in semantic_groups.items():
        if len(group) == 1:
            final.append(group[0])
            continue
        best = max(group, key=_score)
        best_impact = max(group, key=lambda e: impact_rank(e.get('impact', 'low')))
        best = dict(best)
        best['impact'] = best_impact['impact']
        for ev in group:
            for field in ('actual', 'forecast', 'previous'):
                if not best.get(field) and ev.get(field):
                    best[field] = ev[field]
            if ev is not best and id(ev) not in dominated:
                dominated.add(id(ev))
                merged_count += 1
        final.append(best)

    if merged_count:
        print(f"  [Dedup] Merged {merged_count} semantic duplicates, "
              f"{len(events)} → {len(final)}")
    else:
        print(f"  [Dedup] {len(events)} events, no duplicates found")
    return final


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("ECONOMIC CALENDAR SCRAPER v8.0")
print("Primary:   Finnhub Economic Calendar API (UTC, normalized impact)")
print("Secondary: Investing.com (events not covered by Finnhub)")
print("=" * 60)

today     = date.today()
from_date = today - timedelta(days=90)
to_date   = today + timedelta(days=30)

target_dates = set()
d = from_date
while d <= to_date:
    target_dates.add(d)
    d += timedelta(days=1)

print(f"\nRange: {from_date} → {to_date} ({len(target_dates)} days)\n")

# ── STEP 1: Cargar base existente ─────────────────────────────────
base_events = {}
try:
    with open(CALENDAR_PATH, encoding='utf-8') as f:
        prev = json.load(f)
    for ev in prev.get('events', []):
        try:
            k = (ev['dateISO'], ev['currency'], ev['event'][:25].lower().strip())
            base_events[k] = ev
        except Exception:
            pass
    all_base_dates = sorted(set(e['dateISO'] for e in base_events.values()))
    print(f"  [Base] Loaded {len(base_events)} events "
          f"({all_base_dates[0] if all_base_dates else '?'} → "
          f"{all_base_dates[-1] if all_base_dates else '?'})")
except FileNotFoundError:
    print("  [Base] No previous calendar.json — starting fresh")
except Exception as e:
    print(f"  [Base] Error loading: {e}")

# ── STEP 2: Fetch fuentes ─────────────────────────────────────────
from_str = from_date.isoformat()
to_str   = to_date.isoformat()

print(f"\n{'='*50}")
print("STEP 2A — Finnhub (primary)")
print(f"{'='*50}")
finnhub_evs = fetch_finnhub_calendar(from_str, to_str, target_dates)

print(f"\n{'='*50}")
print("STEP 2B — Investing.com (secondary, complementary only)")
print(f"{'='*50}")
investing_evs = fetch_investing_calendar(from_str, to_str, target_dates)

# ── STEP 3: Merge fuentes ─────────────────────────────────────────
print(f"\n{'='*50}")
print("STEP 3 — Merge")
print(f"{'='*50}")

if finnhub_evs or investing_evs:
    fresh_events = merge_sources(finnhub_evs, investing_evs)
else:
    fresh_events = []
    print("  ⚠️  Both sources failed — will use base only")

# ── STEP 4: Merge con base histórica ──────────────────────────────
if fresh_events:
    merged = dict(base_events)
    added = updated = 0

    for ev in fresh_events:
        k = (ev['dateISO'], ev['currency'], ev['event'][:25].lower().strip())

        if k in merged:
            existing = dict(merged[k])
            # Fresh siempre gana en todo — es más nuevo y más fiable
            for field in ('actual', 'forecast', 'previous', 'timeUTC', 'impact', 'event'):
                if ev.get(field):
                    existing[field] = ev[field]
            # Pero no bajar el impact si la base tenía high y fresh trae low
            # (solo aplica si la fuente es Investing, no Finnhub)
            if ev.get('_source') == 'investing':
                if impact_rank(existing.get('impact', 'low')) > impact_rank(ev.get('impact', 'low')):
                    existing['impact'] = merged[k].get('impact', 'low')
            merged[k] = existing
            updated += 1
        else:
            # Buscar en fecha adyacente (eventos en boundary 22:00-02:00)
            matched = False
            try:
                ev_date = date.fromisoformat(ev['dateISO'])
                for delta in (-1, 1):
                    adj = (ev_date + timedelta(days=delta)).isoformat()
                    k_adj = (adj, ev['currency'], ev['event'][:25].lower().strip())
                    if k_adj in merged:
                        existing = dict(merged[k_adj])
                        for field in ('actual', 'forecast', 'previous'):
                            if ev.get(field):
                                existing[field] = ev[field]
                        merged[k_adj] = existing
                        updated += 1
                        matched = True
                        break
            except Exception:
                pass
            if not matched:
                merged[k] = ev
                added += 1

    all_events = list(merged.values())
    print(f"  [Merge-base] base={len(base_events)} + fresh={len(fresh_events)} "
          f"→ {len(all_events)} total (added={added}, updated={updated})")
else:
    all_events = list(base_events.values())
    print(f"  [Merge-base] Using base only: {len(all_events)} events")

# ── STEP 5: Dedup ─────────────────────────────────────────────────
print(f"\n{'='*50}")
print("STEP 5 — Dedup")
print(f"{'='*50}")
final_events = dedup_events(all_events)

# Filtrar eventos sin tiempo ni datos (noise puro)
before = len(final_events)
final_events = [
    ev for ev in final_events
    if (ev.get('timeUTC', '').strip() or
        ev.get('actual', '').strip() or
        ev.get('forecast', '').strip() or
        ev.get('previous', '').strip())
]
if len(final_events) < before:
    print(f"  [Filter] Removed {before - len(final_events)} no-data noise events")

# Eliminar campo interno _source antes de guardar
for ev in final_events:
    ev.pop('_source', None)

# Sort por fecha + hora
def sort_key(ev):
    t = ev.get('timeUTC') or '00:00'
    return ev['dateISO'] + 'T' + (t if re.match(r'\d{2}:\d{2}', t) else '00:00')

final_events.sort(key=sort_key)

# ── STEP 6: Guardar ───────────────────────────────────────────────
currency_counts = {}
impact_counts   = {'high': 0, 'medium': 0, 'low': 0}
for ev in final_events:
    c = ev['currency']
    currency_counts[c] = currency_counts.get(c, 0) + 1
    impact_counts[ev.get('impact', 'low')] = \
        impact_counts.get(ev.get('impact', 'low'), 0) + 1

all_dates = sorted(set(e['dateISO'] for e in final_events))
data_ok   = len(final_events) >= 5

source_label = 'Finnhub + Investing.com' if finnhub_evs else 'Investing.com only'
if not fresh_events:
    source_label = 'Base cache only'

output = {
    'lastUpdate':    today.isoformat(),
    'generatedAt':   datetime.utcnow().isoformat() + 'Z',
    'timezoneNote':  'All timeUTC fields are UTC. Frontend converts to user local timezone.',
    'status':        'ok' if data_ok else 'error',
    'source':        source_label,
    'errorMessage':  None if data_ok else (
        'No se pudieron obtener datos. '
        'Consulte: investing.com/economic-calendar'
    ),
    'fetchErrors':   [],
    'rangeFrom':     all_dates[0] if all_dates else from_date.isoformat(),
    'rangeTo':       to_date.isoformat(),
    'totalEvents':   len(final_events),
    'currencyCounts': currency_counts,
    'impactCounts':  impact_counts,
    'events':        final_events,
}

with open(CALENDAR_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
if data_ok:
    print(f"✅ SAVED: {CALENDAR_PATH}")
    print(f"   Source:  {source_label}")
    print(f"   Total:   {len(final_events)} events")
    print(f"   Impact:  {impact_counts}")
    print(f"   Dates:   {all_dates[0]} → {all_dates[-1]}")
    print(f"   By currency: {dict(sorted(currency_counts.items()))}")
    highs = [e for e in final_events if e.get('impact') == 'high'][:10]
    if highs:
        print(f"\n   HIGH impact events:")
        for ev in highs:
            fc = f" est:{ev['forecast']}" if ev.get('forecast') else ''
            ac = f" → {ev['actual']}"     if ev.get('actual')   else ''
            print(f"   {ev['dateISO']} {(ev['timeUTC'] or '?'):5s} [{ev['currency']}] "
                  f"{ev['event'][:50]}{fc}{ac}")
else:
    print(f"⛔ SAVED EMPTY CALENDAR — all sources failed")
print("=" * 60)

if not data_ok:
    sys.exit(1)
