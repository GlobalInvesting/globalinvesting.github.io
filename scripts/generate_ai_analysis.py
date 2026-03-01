# ─────────────────────────────────────────────────────────────────────────────
# PATCH para generate_ai_analysis.py — v4
# Reutiliza análisis existente si los datos económicos no cambiaron,
# evitando llamadas innecesarias a Groq y ahorrando tokens.
#
# CAMBIOS v4 respecto a v3:
#   • load_previous_analysis(): carga el análisis guardado para una divisa
#   • data_has_changed(): compara dataSnapshot anterior vs datos actuales
#   • main() modificado: salta Groq si los datos son idénticos al snapshot
#   • Reporte detallado: cuántas divisas se reutilizaron vs regeneraron
#
# INSTRUCCIONES DE APLICACIÓN:
#   1. Agrega las nuevas funciones (load_previous_analysis, data_has_changed)
#      ANTES de la función main() en generate_ai_analysis.py
#   2. Reemplaza la función main() completa con la versión de este patch
#   3. Las demás funciones (call_groq_api, generate_analysis, etc.) no cambian
# ─────────────────────────────────────────────────────────────────────────────

# ── NUEVAS FUNCIONES — agregar antes de main() ────────────────────────────────

def load_previous_analysis(currency: str) -> dict | None:
    """
    Carga el análisis JSON guardado previamente para una divisa.
    Retorna el dict completo o None si no existe o está corrupto.
    """
    path = OUTPUT_DIR / f"{currency}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# Campos del dataSnapshot que se comparan para detectar cambios.
# Se excluyen campos volátiles (lastUpdate, fxSource) que cambian
# aunque los datos macroeconómicos sean los mismos.
SNAPSHOT_COMPARE_KEYS = [
    "interestRate",
    "gdpGrowth",
    "inflation",
    "unemployment",
    "currentAccount",
    "rateMomentum",
    "lastRateDecision",
    "cotPositioning",
    "fxPerformance1M",
]


def data_has_changed(currency: str, current_data: dict, prev_analysis: dict | None) -> bool:
    """
    Compara los datos económicos actuales con el dataSnapshot del análisis previo.

    Retorna True  → los datos cambiaron → hay que regenerar con Groq
    Retorna False → los datos son idénticos → se puede reutilizar el análisis

    Criterios de cambio:
    - No existe análisis previo → siempre regenerar
    - Algún indicador clave difiere (interestRate, gdpGrowth, inflation, etc.)
    - rateMomentum cambió (señal de nueva decisión del banco central)
    - lastRateDecision es diferente (nueva reunión procesada)

    Se ignoran diferencias mínimas de float (< 0.01%) para evitar
    regeneraciones por redondeo de API.
    """
    if prev_analysis is None:
        return True

    prev_snapshot = prev_analysis.get("dataSnapshot", {})
    if not prev_snapshot:
        return True

    for key in SNAPSHOT_COMPARE_KEYS:
        curr_val = current_data.get(key)
        prev_val = prev_snapshot.get(key)

        # Ambos None → sin cambio para este campo
        if curr_val is None and prev_val is None:
            continue

        # Uno tiene valor y el otro no → cambio
        if (curr_val is None) != (prev_val is None):
            return True

        # Comparación numérica con tolerancia mínima (evitar falsos positivos por redondeo)
        if isinstance(curr_val, (int, float)) and isinstance(prev_val, (int, float)):
            if abs(float(curr_val) - float(prev_val)) > 0.001:
                return True
            continue

        # Comparación de strings y otros tipos
        if str(curr_val).strip() != str(prev_val).strip():
            return True

    return False  # Todos los campos son iguales → reutilizar


# ── main() COMPLETO — reemplaza el existente ─────────────────────────────────

def main():
    print("=" * 60)
    print(f"🤖 Generador AI v4.0 — {GROQ_MODEL} via Groq API")
    print(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # ── Cargar y verificar keys ───────────────────────────────────────────────
    all_keys = load_groq_keys()
    if not all_keys:
        raise EnvironmentError("❌ No se encontró ninguna GROQ_API_KEY.")

    print(f"\n🔑 Keys configuradas: {len(all_keys)}")
    available_keys = []
    for i, key in enumerate(all_keys, 1):
        status = check_groq_key(key)
        label  = f"Key {i} ({mask_key(key)})"
        if status == "daily_limit":
            print(f"  ⛔ {label} — límite diario confirmado")
        elif status == "invalid":
            print(f"  ❌ {label} — inválida (401)")
        else:
            print(f"  ✅ {label} — disponible")
            available_keys.append(key)

    if not available_keys:
        print("\n⛔ Todas las keys confirmadas agotadas. Saliendo.")
        import sys
        sys.exit(0)

    current_key_idx = 0
    print(f"\n✅ {len(available_keys)} key(s) disponible(s) | Usando Key 1")
    print(f"🔧 Modelo: {GROQ_MODEL}")
    print(f"📊 Decisiones monetarias: historial propio rates/{{currency}}.json")
    print(f"💱 FX Performance: Frankfurter API con fallback estático\n")

    print("🔍 Testeando conectividad...")
    for label, url in [
        ("Internet",        "https://httpbin.org/get"),
        ("GitHub Pages",    "https://globalinvesting.github.io/economic-data/USD.json"),
        ("Frankfurter API", "https://api.frankfurter.dev/v1/latest?base=EUR&symbols=USD"),
        ("UN Comtrade",     "https://comtradeapi.un.org/public/v1/preview/C/A/HS?reporterCode=36&period=2022&partnerCode=0&cmdCode=TOTAL&flowCode=X&customsCode=C00&motCode=0"),
    ]:
        try:
            r = requests.get(url, timeout=8)
            print(f"  ✅ {label} OK ({r.status_code})")
        except Exception as e:
            print(f"  ❌ {label}: {e}")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("📥 Cargando datos económicos...")
    all_data = {}
    for currency in CURRENCIES:
        print(f"  • {currency}...", end=' ', flush=True)
        all_data[currency] = load_economic_data(currency)
        available = sum(1 for k, v in all_data[currency].items()
                        if v is not None and k not in ('lastRateDecision', 'fxSource'))
        fx_src = all_data[currency].get('fxSource', '?')
        print(f"{available} indicadores | FX:{fx_src}")

    global_context = compute_global_context(all_data)
    print(f"\n🌐 Contexto global:")
    for k, v in global_context.items():
        if v is not None:
            print(f"   {k}: {v}")
    print()

    print("📦 Obteniendo composición exportadora (UN Comtrade)...")
    export_compositions = {}
    for currency in CURRENCIES:
        print(f"  • {currency}...", end=' ', flush=True)
        composition = fetch_export_composition(currency)
        export_compositions[currency] = composition
        if not composition:
            print("sin dato")
    print()

    # ── Pre-check: ¿qué divisas necesitan regeneración? ──────────────────────
    print("🔎 Comparando datos actuales con snapshots anteriores...")
    needs_regen  = []  # divisas con datos nuevos → llamar a Groq
    can_reuse    = []  # divisas sin cambios → reutilizar análisis guardado
    prev_analyses = {}

    for currency in CURRENCIES:
        prev = load_previous_analysis(currency)
        prev_analyses[currency] = prev
        changed = data_has_changed(currency, all_data[currency], prev)
        if changed:
            needs_regen.append(currency)
            reason = "sin análisis previo" if prev is None else "datos actualizados"
            print(f"  🆕 {currency} — {reason}")
        else:
            can_reuse.append(currency)
            prev_age = ""
            if prev and prev.get("generatedAt"):
                try:
                    from datetime import datetime, timezone
                    gen_at = datetime.fromisoformat(prev["generatedAt"].replace("Z", "+00:00"))
                    hours  = (datetime.now(timezone.utc) - gen_at).total_seconds() / 3600
                    prev_age = f" (generado hace {hours:.0f}h)"
                except Exception:
                    pass
            print(f"  ♻️  {currency} — sin cambios en datos{prev_age}, reutilizando análisis")

    print(f"\n   🆕 A regenerar:  {len(needs_regen)} divisas — {', '.join(needs_regen) or 'ninguna'}")
    print(f"   ♻️  A reutilizar: {len(can_reuse)} divisas — {', '.join(can_reuse) or 'ninguna'}")

    if not needs_regen:
        print("\n✅ Todos los análisis están actualizados. Sin llamadas a Groq necesarias.")
        # Actualizar solo el index.json con timestamp fresco
        successful = [c for c in CURRENCIES if prev_analyses.get(c) is not None]
        index = {
            "generatedAt":    datetime.now(timezone.utc).isoformat(),
            "model":          GROQ_MODEL,
            "version":        "4.0",
            "currencies":     successful,
            "totalGenerated": 0,
            "totalReused":    len(can_reuse),
            "keysUsed":       0,
            "errors":         [],
            "results":        {c: {"success": True, "reused": True} for c in can_reuse},
            "globalContext":  global_context,
        }
        with open(OUTPUT_DIR / 'index.json', 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        return
    print()

    # ── Generar solo las divisas que cambiaron ────────────────────────────────
    results = {}

    # Marcar las divisas reutilizadas como exitosas sin procesamiento
    for currency in can_reuse:
        results[currency] = {"success": True, "reused": True}

    errors = []

    for i, currency in enumerate(needs_regen):
        print(f"[{i+1}/{len(needs_regen)}] {currency} — generando análisis...")
        data       = all_data[currency]
        export_comp = export_compositions.get(currency)

        available = sum(1 for v in data.values() if v is not None)
        print(f"  📊 {available} indicadores económicos")
        if export_comp:
            print(f"  🏭 Exportaciones: {export_comp[:80]}...")

        if available < 4:
            msg = f"Datos insuficientes ({available})"
            print(f"  ⚠️  {msg}, saltando...")
            errors.append(f"{currency}: {msg}")
            results[currency] = {"success": False, "error": msg}
            continue

        # ── Intento con key actual, fallback si DAILY_LIMIT ──────────────────
        analysis_text = None
        while current_key_idx < len(available_keys):
            try:
                print(f"  🧠 Groq API (Key {current_key_idx + 1})...")
                analysis_text = generate_analysis(
                    available_keys[current_key_idx], currency, data,
                    global_context, export_comp
                )
                break  # éxito

            except RuntimeError as e:
                if "DAILY_LIMIT" in str(e):
                    print(f"  ⛔ Key {current_key_idx + 1} agotada — buscando siguiente...")
                    current_key_idx += 1
                    if current_key_idx >= len(available_keys):
                        print("  ⛔ Todas las keys agotadas — deteniendo")
                        break
                    print(f"  🔄 Key {current_key_idx + 1} ({mask_key(available_keys[current_key_idx])}) — pausa {KEY_SWITCH_PAUSE}s...")
                    time.sleep(KEY_SWITCH_PAUSE)
                else:
                    raise

        if analysis_text is None:
            # Si falló Groq pero hay análisis previo, conservarlo como fallback
            prev = prev_analyses.get(currency)
            if prev and prev.get("analysis"):
                print(f"  ⚠️  Groq falló — conservando análisis previo como fallback")
                results[currency] = {"success": True, "reused": True, "fallback": True}
                continue
            msg = "Todas las keys agotadas" if current_key_idx >= len(available_keys) else "Error en generación"
            print(f"  ❌ {msg}")
            errors.append(f"{currency}: {msg}")
            results[currency] = {"success": False, "error": msg}
            if current_key_idx >= len(available_keys):
                print("\n⛔ Sin keys — deteniendo generación")
                break
            continue

        output = {
            "currency":    currency,
            "country":     COUNTRY_META[currency]['name'],
            "bank":        COUNTRY_META[currency]['bank'],
            "analysis":    analysis_text,
            "model":       GROQ_MODEL,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "exportComposition": export_comp,
            "dataSnapshot": {
                "interestRate":      data.get('interestRate'),
                "gdpGrowth":         data.get('gdpGrowth'),
                "inflation":         data.get('inflation'),
                "unemployment":      data.get('unemployment'),
                "currentAccount":    data.get('currentAccount'),
                "rateMomentum":      data.get('rateMomentum'),
                "lastRateDecision":  data.get('lastRateDecision'),
                "cotPositioning":    data.get('cotPositioning'),
                "fxPerformance1M":   data.get('fxPerformance1M'),
                "fxSource":          data.get('fxSource'),
                "lastUpdate":        data.get('lastUpdate'),
            },
            "globalContext": global_context,
        }

        output_path = OUTPUT_DIR / f"{currency}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        results[currency] = {
            "success":          True,
            "reused":           False,
            "wordCount":        len(analysis_text.split()),
            "generatedAt":      output["generatedAt"],
            "exportDataSource": "comtrade" if export_comp and "Comtrade" in export_comp else "fallback",
            "keyUsed":          current_key_idx + 1,
        }
        print(f"  💾 Guardado → {output_path}")

        if i < len(needs_regen) - 1 and not (current_key_idx >= len(available_keys)):
            print(f"  ⏸  Pausa 3s...")
            time.sleep(3)

    # ── Índice final ──────────────────────────────────────────────────────────
    successful      = [c for c, r in results.items() if r.get('success')]
    regenerated     = [c for c, r in results.items() if r.get('success') and not r.get('reused')]
    reused_final    = [c for c, r in results.items() if r.get('reused')]
    comtrade_count  = sum(1 for r in results.values() if r.get('exportDataSource') == 'comtrade')

    index = {
        "generatedAt":    datetime.now(timezone.utc).isoformat(),
        "model":          GROQ_MODEL,
        "version":        "4.0",
        "currencies":     successful,
        "totalGenerated": len(regenerated),
        "totalReused":    len(reused_final),
        "comtradeHits":   comtrade_count,
        "keysUsed":       current_key_idx + 1 if needs_regen else 0,
        "errors":         errors,
        "results":        results,
        "globalContext":  global_context,
    }
    with open(OUTPUT_DIR / 'index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📋 RESUMEN")
    print(f"   ✅ Exitosos:      {len(successful)}/{len(CURRENCIES)}")
    print(f"   🆕 Regenerados:  {len(regenerated)} — {', '.join(regenerated) or 'ninguno'}")
    print(f"   ♻️  Reutilizados: {len(reused_final)} — {', '.join(reused_final) or 'ninguno'}")
    print(f"   🌐 Comtrade:     {comtrade_count}/{len(CURRENCIES)} divisas")
    if needs_regen:
        print(f"   🔑 Keys usadas:  hasta Key {current_key_idx + 1} de {len(available_keys)}")
    else:
        print(f"   🔑 Groq:         0 llamadas (todos reutilizados)")
    if errors:
        print(f"   ❌ Errores:")
        for err in errors:
            print(f"      • {err}")
    print("=" * 60)

    if len(errors) > len(successful):
        raise RuntimeError(f"Demasiados errores: {len(errors)} fallos")
