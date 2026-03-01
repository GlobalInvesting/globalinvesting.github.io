# ─────────────────────────────────────────────────────────────────────────────
# PATCH para generate_ai_analysis.py — soporte multi-key con fallback
# v2 — check_groq_key() conservador: solo marca daily_limit si el mensaje
#      menciona EXPLÍCITAMENTE "per day" o "tpd" (nunca "per minute")
#
# Reemplaza en el script original:
#   call_groq_api(), generate_analysis(), main()
# Agrega antes de main():
#   is_daily_limit_message(), load_groq_keys(), mask_key(), check_groq_key()
# ─────────────────────────────────────────────────────────────────────────────


def is_daily_limit_message(response) -> bool:
    """
    Detecta si un 429 es por límite diario de TOKENS.
    Solo True si el mensaje menciona "per day" o "tpd" (no "per minute").
    """
    try:
        msg = response.json().get("error", {}).get("message", "").lower()
        is_daily     = any(x in msg for x in ("per day", "tpd", "tokens per day"))
        is_per_minute = "per minute" in msg or "rpm" in msg or "tpm" in msg
        return is_daily and not is_per_minute
    except Exception:
        return False


def load_groq_keys() -> list:
    """Carga todas las keys de Groq disponibles en orden de prioridad."""
    keys = []
    for var in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"):
        k = os.environ.get(var, "").strip()
        if k:
            keys.append(k)
    return keys


def mask_key(key: str) -> str:
    return key[:8] + "..." + key[-4:] if len(key) > 12 else "***"


def check_groq_key(key: str) -> str:
    """
    Verifica el estado de una key con una llamada de prueba mínima.

    Devuelve:
      'ok'          → key funciona o error ambiguo (beneficio de la duda)
      'daily_limit' → confirmado límite diario de tokens ("per day" / "tpd")
      'invalid'     → key inválida (401)

    IMPORTANTE: cualquier 429 que NO sea explícitamente límite diario
    se trata como 'ok' — no descartar keys válidas por rate limit temporal.
    """
    try:
        r = requests.post(
            GROQ_URL,
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=10,
        )
        if r.status_code == 401:
            return "invalid"
        if r.status_code == 429:
            if is_daily_limit_message(r):
                return "daily_limit"
            # 429 por RPM, TPM u otro rate limit temporal → key disponible
            return "ok"
        return "ok"
    except Exception:
        # Error de red → asumir OK, el fallo real se verá al usar la key
        return "ok"


def call_groq_api(api_key, data_summary, currency):
    """
    Lanza RuntimeError("RATE_LIMIT")  si recibe 429 por rate limit de minuto.
    Lanza RuntimeError("DAILY_LIMIT") si recibe 429 por cuota diaria de tokens.
    """
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{data_summary}\n\n"
                    f"---\n\n"
                    f"Redacta el análisis para {currency}. "
                    f"Recuerda: 3 párrafos con línea en blanco entre ellos, 150-200 palabras en total. "
                    f"El objetivo es interpretar las causas y consecuencias de los datos, "
                    f"no simplemente listarlos. Redacción en español natural y fluido, "
                    f"como un análisis de mercado profesional hispanohablante."
                ),
            },
        ],
        "max_tokens": 700,
        "temperature": 0.5,
        "top_p": 0.9,
    }
    response = requests.post(
        GROQ_URL,
        json=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        timeout=40,
    )

    if response.status_code == 429:
        if is_daily_limit_message(response):
            raise RuntimeError("DAILY_LIMIT")
        raise RuntimeError("RATE_LIMIT")

    if response.status_code == 401:
        raise RuntimeError("INVALID_KEY")

    response.raise_for_status()
    data = response.json()
    try:
        return data['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Respuesta inesperada: {data}") from e


def generate_analysis(api_key, currency, data, global_context=None, export_composition=None):
    """Propaga DAILY_LIMIT para que main() haga el cambio de key."""
    data_summary = build_data_summary(currency, data, global_context, export_composition)

    for attempt in range(3):
        try:
            text = call_groq_api(api_key, data_summary, currency)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            text = '\n\n'.join(paragraphs)
            word_count = len(text.split())
            if word_count < 80:
                raise ValueError(f"Respuesta demasiado corta: {word_count} palabras")
            print(f"  ✅ {word_count} palabras generadas")
            return text

        except RuntimeError as e:
            err_str = str(e)
            if "DAILY_LIMIT" in err_str:
                raise  # propagar — main() cambia de key
            if "RATE_LIMIT" in err_str:
                wait = 60 if attempt == 0 else 120
                print(f"  ⏳ Rate limit temporal, esperando {wait}s...")
                time.sleep(wait)
            elif "INVALID_KEY" in err_str:
                raise
            elif attempt < 2:
                wait = 15 * (attempt + 1)
                print(f"  ⚠️  Error intento {attempt+1}: {e}. Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < 2:
                wait = 15 * (attempt + 1)
                print(f"  ⚠️  Error intento {attempt+1}: {e}. Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Falló para {currency}: {e}")

    raise RuntimeError(f"Agotados reintentos para {currency}")


def main():
    print("=" * 60)
    print(f"🤖 Generador AI v2.6 — {GROQ_MODEL} via Groq API")
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
            print(f"  ⛔ {label} — límite diario de tokens confirmado")
        elif status == "invalid":
            print(f"  ❌ {label} — inválida (401)")
        else:
            # 'ok' o cualquier duda → incluir
            print(f"  ✅ {label} — disponible")
            available_keys.append(key)

    if not available_keys:
        print("\n⛔ Todas las keys confirmadas agotadas. Saliendo.")
        import sys
        sys.exit(0)

    current_key_idx = 0
    print(f"\n✅ Usando Key 1 de {len(available_keys)} disponibles")
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

    results = {}
    errors  = []

    for i, currency in enumerate(CURRENCIES):
        print(f"[{i+1}/{len(CURRENCIES)}] {currency}...")
        data = all_data[currency]
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
                    print(f"  🔄 Key {current_key_idx + 1} ({mask_key(available_keys[current_key_idx])})")
                else:
                    raise

        if analysis_text is None:
            msg = "Todas las keys de Groq agotadas" if current_key_idx >= len(available_keys) else "Error en generación"
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
            "wordCount":        len(analysis_text.split()),
            "generatedAt":      output["generatedAt"],
            "exportDataSource": "comtrade" if export_comp and "Comtrade" in export_comp else "fallback",
            "keyUsed":          current_key_idx + 1,
        }
        print(f"  💾 Guardado → {output_path}")

        if i < len(CURRENCIES) - 1:
            print(f"  ⏸  Pausa 3s...")
            time.sleep(3)

    successful = [c for c, r in results.items() if r.get('success')]
    comtrade_count = sum(1 for r in results.values() if r.get('exportDataSource') == 'comtrade')

    index = {
        "generatedAt":    datetime.now(timezone.utc).isoformat(),
        "model":          GROQ_MODEL,
        "version":        "2.6",
        "currencies":     successful,
        "totalGenerated": len(successful),
        "comtradeHits":   comtrade_count,
        "keysUsed":       current_key_idx + 1,
        "errors":         errors,
        "results":        results,
        "globalContext":  global_context,
    }
    with open(OUTPUT_DIR / 'index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📋 RESUMEN")
    print(f"   ✅ Exitosos: {len(successful)}/{len(CURRENCIES)} — {', '.join(successful) or 'ninguno'}")
    print(f"   🌐 Comtrade en vivo: {comtrade_count}/{len(CURRENCIES)} divisas")
    print(f"   🔑 Keys usadas: hasta Key {current_key_idx + 1} de {len(available_keys)}")
    if errors:
        print(f"   ❌ Errores:")
        for err in errors:
            print(f"      • {err}")
    print("=" * 60)

    if len(errors) > len(successful):
        raise RuntimeError(f"Demasiados errores: {len(errors)} fallos")
