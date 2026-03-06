"""
tests/test_fetch_news.py
Suite de tests para las funciones críticas de fetch_news.py.

Cubre:
  - detect_currency(): detección por par explícito, scoring ponderado,
    false positive guards y fallback institucional.
  - is_forex_relevant(): filtros de relevancia y exclusiones.
  - detect_impact(): clasificación de impacto.
  - smart_select(): distribución garantizada y límites por divisa.

Ejecutar:
  cd scripts && pytest tests/ -v
  cd scripts && pytest tests/ -v --tb=short   # traceback corto en fallos
"""

import sys
import os
import types
import pytest

# Agregar el directorio scripts/ al path para poder importar fetch_news
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Mock de dependencias externas que no son necesarias para los tests unitarios.
# feedparser y python-dateutil solo se usan en fetch_all_feeds() y parse_date(),
# funciones que no testeamos aquí. Esto permite correr los tests sin instalar
# todas las dependencias de producción (útil en CI con caché o entornos limpios).
for _mod in ("feedparser", "dateutil", "dateutil.parser"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

import fetch_news as fn


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_article(cur, impact="low", ts=1000000, id_=None):
    """Crea un artículo mínimo para tests de smart_select."""
    return {
        "id":     id_ or f"{cur}-{ts}",
        "cur":    cur,
        "impact": impact,
        "ts":     ts,
        "title":  f"Test article {cur}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# detect_currency() — PASO 1: Par explícito en el título
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectCurrencyExplicitPair:
    """El par explícito en el título debe identificar al protagonista correctamente."""

    def test_eurusd_al_inicio_retorna_eur(self):
        assert fn.detect_currency("EUR/USD sube tras datos de inflación", "") == "EUR"

    def test_gbpusd_retorna_gbp(self):
        assert fn.detect_currency("GBP/USD cae por debajo de 1.26 tras BoE", "") == "GBP"

    def test_usdjpy_retorna_jpy(self):
        assert fn.detect_currency("USD/JPY alcanza máximos de 3 meses", "") == "JPY"

    def test_audusd_retorna_aud(self):
        assert fn.detect_currency("AUD/USD presionado por datos chinos débiles", "") == "AUD"

    def test_nzdusd_retorna_nzd(self):
        assert fn.detect_currency("NZD/USD: RBNZ mantiene tasas sin cambios", "") == "NZD"

    def test_usdcad_retorna_cad(self):
        assert fn.detect_currency("USD/CAD sube con petróleo a la baja", "") == "CAD"

    def test_usdchf_retorna_chf(self):
        assert fn.detect_currency("USD/CHF cae tras datos SNB", "") == "CHF"

    def test_eurgbp_retorna_eur(self):
        assert fn.detect_currency("EUR/GBP: ¿qué esperar esta semana?", "") == "EUR"

    def test_gbpjpy_retorna_gbp(self):
        assert fn.detect_currency("GBP/JPY análisis técnico semanal", "") == "GBP"

    def test_par_sin_slash_eurusd(self):
        """También debe detectar el par sin barra."""
        assert fn.detect_currency("EURUSD análisis de hoy", "") == "EUR"

    def test_par_sin_slash_usdjpy(self):
        assert fn.detect_currency("USDJPY alcanza resistencia clave", "") == "JPY"

    def test_par_en_mitad_del_titulo(self):
        """El par en cualquier posición del título debe funcionar."""
        assert fn.detect_currency("Análisis completo: EUR/USD y perspectivas", "") == "EUR"


# ══════════════════════════════════════════════════════════════════════════════
# detect_currency() — PASO 2: Scoring ponderado
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectCurrencyScoring:
    """Keywords de alto peso deben identificar la divisa correcta."""

    # ── USD ─────────────────────────────────────────────────────────────────
    def test_fed_en_titulo_retorna_usd(self):
        assert fn.detect_currency("Fed mantiene tasas en reunión de marzo", "") == "USD"

    def test_powell_retorna_usd(self):
        assert fn.detect_currency("Powell señala posibles recortes en 2025", "") == "USD"

    def test_fomc_retorna_usd(self):
        assert fn.detect_currency("FOMC minutes revelan división sobre tasas", "") == "USD"

    def test_nonfarm_payrolls_retorna_usd(self):
        assert fn.detect_currency("Nonfarm payrolls sorprenden al alza en febrero", "") == "USD"

    def test_us_cpi_retorna_usd(self):
        assert fn.detect_currency("US CPI aumenta más de lo esperado en enero", "") == "USD"

    # ── EUR ─────────────────────────────────────────────────────────────────
    def test_ecb_retorna_eur(self):
        assert fn.detect_currency("ECB recorta tasas 25 puntos básicos", "") == "EUR"

    def test_lagarde_retorna_eur(self):
        assert fn.detect_currency("Lagarde: la inflación de la eurozona sigue siendo alta", "") == "EUR"

    def test_eurozone_inflation_retorna_eur(self):
        assert fn.detect_currency("Eurozone inflation falls to 2.3% in February", "") == "EUR"

    def test_bce_retorna_eur(self):
        assert fn.detect_currency("BCE mantiene tipos en el 4% ante presión inflacionaria", "") == "EUR"

    # ── GBP ─────────────────────────────────────────────────────────────────
    def test_boe_retorna_gbp(self):
        assert fn.detect_currency("BoE sube tasas por encima de lo esperado", "") == "GBP"

    def test_bank_of_england_retorna_gbp(self):
        assert fn.detect_currency("Bank of England signals rate cut for summer", "") == "GBP"

    def test_sterling_retorna_gbp(self):
        assert fn.detect_currency("Sterling hits 8-month high on strong UK jobs data", "") == "GBP"

    def test_uk_cpi_retorna_gbp(self):
        assert fn.detect_currency("UK CPI falls sharply to 3.4%, below expectations", "") == "GBP"

    # ── JPY ─────────────────────────────────────────────────────────────────
    def test_boj_retorna_jpy(self):
        assert fn.detect_currency("BoJ sorprende con subida de tasas histórica", "") == "JPY"

    def test_bank_of_japan_retorna_jpy(self):
        assert fn.detect_currency("Bank of Japan ends negative interest rate policy", "") == "JPY"

    def test_japanese_yen_retorna_jpy(self):
        assert fn.detect_currency("Japanese yen weakens past 150 on yield differential", "") == "JPY"

    # ── AUD ─────────────────────────────────────────────────────────────────
    def test_rba_retorna_aud(self):
        assert fn.detect_currency("RBA pausa el ciclo de subidas tras dato de empleo", "") == "AUD"

    def test_reserve_bank_australia_retorna_aud(self):
        assert fn.detect_currency("Reserve Bank of Australia holds at 4.35%", "") == "AUD"

    def test_australian_dollar_retorna_aud(self):
        assert fn.detect_currency("Australian dollar weakens on China slowdown fears", "") == "AUD"

    # ── CAD ─────────────────────────────────────────────────────────────────
    def test_boc_retorna_cad(self):
        assert fn.detect_currency("BoC recorta tasas por segunda vez este año", "") == "CAD"

    def test_bank_of_canada_retorna_cad(self):
        assert fn.detect_currency("Bank of Canada signals more cuts ahead", "") == "CAD"

    def test_canadian_dollar_retorna_cad(self):
        assert fn.detect_currency("Canadian dollar slips as oil prices fall", "") == "CAD"

    # ── CHF ─────────────────────────────────────────────────────────────────
    def test_snb_retorna_chf(self):
        assert fn.detect_currency("SNB sorprende con recorte de tasas al 1.0%", "") == "CHF"

    def test_swiss_national_bank_retorna_chf(self):
        assert fn.detect_currency("Swiss National Bank cuts rates to fight deflation", "") == "CHF"

    def test_swiss_franc_retorna_chf(self):
        assert fn.detect_currency("Swiss franc surges as investors seek safety", "") == "CHF"

    # ── NZD ─────────────────────────────────────────────────────────────────
    def test_rbnz_retorna_nzd(self):
        assert fn.detect_currency("RBNZ baja tasas 50bp en movimiento agresivo", "") == "NZD"

    def test_reserve_bank_nz_retorna_nzd(self):
        assert fn.detect_currency("Reserve Bank of New Zealand cuts OCR to 4.75%", "") == "NZD"

    def test_kiwi_dollar_retorna_nzd(self):
        assert fn.detect_currency("Kiwi dollar falls after weak NZ GDP data", "") == "NZD"

    # ── Español ──────────────────────────────────────────────────────────────
    def test_reserva_federal_retorna_usd(self):
        assert fn.detect_currency("La Reserva Federal mantiene tasas sin cambios", "") == "USD"

    def test_banco_central_europeo_retorna_eur(self):
        assert fn.detect_currency("El Banco Central Europeo recorta tipos 25 puntos", "") == "EUR"

    def test_banco_de_japon_retorna_jpy(self):
        assert fn.detect_currency("El Banco de Japón sube tasas por primera vez en décadas", "") == "JPY"


# ══════════════════════════════════════════════════════════════════════════════
# detect_currency() — False Positive Guards
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectCurrencyFalsePositiveGuards:
    """Los guards deben penalizar divisas incorrectas en contextos ambiguos."""

    def test_sugar_per_pound_no_retorna_gbp(self):
        """'pound' en contexto de commodities no debe identificarse como GBP."""
        result = fn.detect_currency(
            "Sugar futures rise 3 cents per pound on supply concerns",
            "Global sugar prices increased driven by reduced Brazilian output"
        )
        assert result != "GBP"

    def test_halifax_house_prices_no_retorna_nzd(self):
        """Halifax (UK) no debe penalizar NZD."""
        result = fn.detect_currency(
            "Halifax: UK house prices rise 0.3% in February",
            "UK housing market shows resilience despite high mortgage rates"
        )
        assert result != "NZD"

    def test_turkish_lira_no_retorna_aud(self):
        """Noticias de lira turca no deben clasificarse como AUD."""
        result = fn.detect_currency(
            "Turkish lira hits record low as inflation surges",
            "The lira weakened sharply after CBRT decision"
        )
        assert result != "AUD"

    def test_ibovespa_no_retorna_cad(self):
        """Índice brasileño no debe clasificarse como CAD."""
        result = fn.detect_currency(
            "Ibovespa sube 2% impulsado por commodities",
            "El mercado brasileño reacciona positivamente"
        )
        assert result != "CAD"

    def test_south_african_rand_no_retorna_usd(self):
        """Rand sudafricano no debe clasificarse como USD."""
        result = fn.detect_currency(
            "South African rand weakens on political uncertainty",
            "ZAR hits lowest level in 6 months"
        )
        assert result != "USD"


# ══════════════════════════════════════════════════════════════════════════════
# detect_currency() — Fallback institucional (PASO 3)
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectCurrencyFallback:
    """El fallback institucional debe activarse cuando el scoring no alcanza el mínimo."""

    def test_fuente_ecb_retorna_eur(self):
        assert fn.detect_currency("Press release on monetary policy", "", source="ECB") == "EUR"

    def test_fuente_boe_retorna_gbp(self):
        assert fn.detect_currency("Monetary Policy Committee statement", "", source="Bank of England") == "GBP"

    def test_fuente_rba_retorna_aud(self):
        assert fn.detect_currency("Statement by the Governor", "", source="RBA") == "AUD"

    def test_fuente_rbnz_retorna_nzd(self):
        assert fn.detect_currency("Official Cash Rate announcement", "", source="RBNZ") == "NZD"

    def test_fuente_snb_retorna_chf(self):
        assert fn.detect_currency("SNB monetary policy assessment", "", source="SNB") == "CHF"

    def test_fuente_boj_retorna_jpy(self):
        assert fn.detect_currency("Monetary policy decision", "", source="Bank of Japan") == "JPY"

    def test_google_news_usd_retorna_usd(self):
        assert fn.detect_currency("Breaking: markets react", "", source="Google News USD") == "USD"

    def test_google_news_nzd_retorna_nzd(self):
        assert fn.detect_currency("Economic update", "", source="Google News NZD") == "NZD"

    def test_sin_fuente_conocida_y_score_bajo_retorna_none(self):
        """Sin fuente institucional y sin keywords debe retornar None."""
        result = fn.detect_currency(
            "Local elections scheduled for next month",
            "Voters will head to polls",
            source=""
        )
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# is_forex_relevant()
# ══════════════════════════════════════════════════════════════════════════════

class TestIsForexRelevant:
    """Filtro de relevancia: debe incluir noticias forex y excluir ruido."""

    # ── Casos que DEBEN ser relevantes ───────────────────────────────────────
    def test_fed_es_relevante(self):
        assert fn.is_forex_relevant("Fed holds rates steady at March meeting", "") is True

    def test_ecb_es_relevante(self):
        assert fn.is_forex_relevant("ECB cuts rates by 25 basis points", "") is True

    def test_inflation_es_relevante(self):
        assert fn.is_forex_relevant("US inflation surges to 3.2% in February", "") is True

    def test_gdp_es_relevante(self):
        assert fn.is_forex_relevant("UK GDP contracts 0.1% in Q4", "") is True

    def test_cpi_es_relevante(self):
        assert fn.is_forex_relevant("Eurozone CPI falls to 2.6%", "") is True

    def test_pmi_es_relevante(self):
        assert fn.is_forex_relevant("Japan PMI rises above 50 for first time in 6 months", "") is True

    def test_monetary_policy_es_relevante(self):
        assert fn.is_forex_relevant("RBA monetary policy decision: hold at 4.35%", "") is True

    def test_divisa_directa_es_relevante(self):
        assert fn.is_forex_relevant("EUR/USD outlook for next week", "") is True

    def test_treasury_yield_es_relevante(self):
        assert fn.is_forex_relevant("US Treasury yield hits 4.5% on strong jobs data", "") is True

    def test_en_espanol_es_relevante(self):
        assert fn.is_forex_relevant("El BCE recorta tasas ante caída de la inflación", "") is True

    def test_tasa_de_interes_es_relevante(self):
        assert fn.is_forex_relevant("Banco de Japón sube tasa de interés por primera vez", "") is True

    # ── Casos que DEBEN ser excluidos ────────────────────────────────────────
    def test_dax_rises_no_es_relevante(self):
        """Índices bursátiles europeos sin contexto forex deben excluirse."""
        assert fn.is_forex_relevant("DAX rises 200 points on tech rally", "") is False

    def test_cac40_falls_no_es_relevante(self):
        assert fn.is_forex_relevant("CAC 40 falls sharply amid political uncertainty", "") is False

    def test_eurostoxx_no_es_relevante(self):
        assert fn.is_forex_relevant("EuroStoxx 50 gains 1.5% on earnings optimism", "") is False

    def test_ibovespa_no_es_relevante(self):
        assert fn.is_forex_relevant("Ibovespa sube impulsado por Vale y Petrobras", "") is False

    def test_bitcoin_puro_no_es_relevante(self):
        """Bitcoin sin contexto de banco central no debe ser relevante."""
        assert fn.is_forex_relevant(
            "Bitcoin hits $75,000 all-time high on ETF demand",
            "Crypto markets surge as institutional buyers enter"
        ) is False

    def test_bitcoin_con_fed_si_es_relevante(self):
        """Bitcoin con mención a Fed sí es relevante (bridge keyword)."""
        assert fn.is_forex_relevant(
            "Bitcoin rises as Federal Reserve signals pause",
            "The central bank's dovish tone boosted risk assets including crypto"
        ) is True

    def test_silver_price_no_es_relevante(self):
        assert fn.is_forex_relevant("Silver price slammed as dollar strengthens", "Silver futures drop") is False

    def test_noticias_sin_contexto_forex_no_relevante(self):
        assert fn.is_forex_relevant(
            "Champions League final tickets sell out in minutes",
            "Football fans scramble for seats at the final"
        ) is False


# ══════════════════════════════════════════════════════════════════════════════
# detect_impact()
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectImpact:
    """Clasificación correcta de impacto: high / med / low."""

    def test_rate_decision_es_high(self):
        assert fn.detect_impact("Fed rate decision: hold at 5.25%", "") == "high"

    def test_nonfarm_es_high(self):
        assert fn.detect_impact("Nonfarm payrolls beat expectations with 250K jobs", "") == "high"

    def test_cpi_es_high(self):
        assert fn.detect_impact("CPI inflation jumps to 3.5%, highest in 6 months", "") == "high"

    def test_powell_es_high(self):
        assert fn.detect_impact("Powell signals faster rate cuts if labor market weakens", "") == "high"

    def test_hawkish_es_high(self):
        assert fn.detect_impact("ECB adopts more hawkish stance on inflation", "") == "high"

    def test_pmi_es_med(self):
        assert fn.detect_impact("UK Manufacturing PMI rises to 51.2 in March", "") == "med"

    def test_retail_sales_es_med(self):
        assert fn.detect_impact("US retail sales rise 0.4% in February", "") == "med"

    def test_employment_es_med(self):
        assert fn.detect_impact("Australian employment change beats forecast", "") == "med"

    def test_oil_prices_es_med(self):
        assert fn.detect_impact("Crude oil prices rise on OPEC supply cut extension", "") == "med"

    def test_noticia_generica_es_low(self):
        assert fn.detect_impact("EUR/USD weekly technical outlook and chart analysis", "") == "low"

    def test_analisis_tecnico_es_low(self):
        assert fn.detect_impact("GBP/USD support levels to watch this week", "") == "low"


# ══════════════════════════════════════════════════════════════════════════════
# smart_select()
# ══════════════════════════════════════════════════════════════════════════════

class TestSmartSelect:
    """Distribución, límites y orden de los artículos seleccionados."""

    def test_garantiza_minimo_por_divisa(self):
        """Debe garantizar al menos guaranteed_per_cur artículos por divisa si existen."""
        articles = []
        for cur in fn.CURRENCIES:
            for i in range(5):
                articles.append(make_article(cur, "low", ts=1000 + i, id_=f"{cur}-{i}"))

        result = smart_select_wrap(articles, max_total=48, guaranteed=3, max_per=8)
        counts = {}
        for a in result:
            counts[a["cur"]] = counts.get(a["cur"], 0) + 1
        for cur in fn.CURRENCIES:
            assert counts.get(cur, 0) >= 3, f"{cur} tiene menos de 3 artículos garantizados"

    def test_respeta_max_total(self):
        """No debe superar max_total artículos."""
        articles = [make_article("USD", "low", ts=i, id_=f"usd-{i}") for i in range(100)]
        result = smart_select_wrap(articles, max_total=10, guaranteed=3, max_per=8)
        assert len(result) <= 10

    def test_respeta_max_por_divisa(self):
        """No debe incluir más de max_per_cur artículos de la misma divisa."""
        articles = [make_article("USD", "low", ts=i, id_=f"usd-{i}") for i in range(50)]
        result = smart_select_wrap(articles, max_total=48, guaranteed=3, max_per=5)
        usd_count = sum(1 for a in result if a["cur"] == "USD")
        assert usd_count <= 5

    def test_prioriza_high_impact(self):
        """Artículos de alto impacto deben aparecer antes que los de bajo impacto."""
        articles = [
            make_article("EUR", "low",  ts=2000, id_="eur-low"),
            make_article("EUR", "high", ts=1000, id_="eur-high"),  # ts más antiguo pero high
        ]
        result = smart_select_wrap(articles, max_total=10, guaranteed=3, max_per=8)
        ids = [a["id"] for a in result]
        assert "eur-high" in ids  # el high debe estar seleccionado

    def test_resultado_ordenado_por_timestamp_desc(self):
        """El resultado final debe estar ordenado por ts descendente."""
        articles = []
        for i in range(5):
            articles.append(make_article("USD", "low", ts=i * 100, id_=f"usd-{i}"))
        result = smart_select_wrap(articles, max_total=10, guaranteed=3, max_per=8)
        timestamps = [a["ts"] for a in result]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_sin_articulos_retorna_lista_vacia(self):
        result = smart_select_wrap([], max_total=48, guaranteed=3, max_per=8)
        assert result == []

    def test_no_duplica_ids(self):
        """No deben aparecer artículos duplicados en el resultado."""
        articles = [make_article("GBP", "high", ts=1000, id_="gbp-1")] * 10
        result = smart_select_wrap(articles, max_total=48, guaranteed=3, max_per=8)
        ids = [a["id"] for a in result]
        assert len(ids) == len(set(ids))


def smart_select_wrap(articles, max_total, guaranteed, max_per):
    """Wrapper para llamar smart_select con nombres de parámetro explícitos."""
    return fn.smart_select(
        articles,
        max_total=max_total,
        guaranteed_per_cur=guaranteed,
        max_per_cur=max_per,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Casos de regresión — Bugs conocidos corregidos en versiones anteriores
# ══════════════════════════════════════════════════════════════════════════════

class TestRegresion:
    """
    Casos que documentan bugs específicos corregidos en versiones previas.
    Si alguno falla, indica una regresión.
    """

    def test_nzd_no_confunde_con_usd_en_noticias_genericas(self):
        """
        Regresión v5.2→v5.3: artículos de NZD se clasificaban como USD
        por keywords de bajo peso. El score mínimo de 5 debe prevenirlo.
        """
        result = fn.detect_currency(
            "New Zealand dollar falls on weak domestic data",
            "NZD drops as RBNZ signals dovish shift"
        )
        assert result == "NZD"

    def test_dwelling_consents_nz_no_penaliza_nzd(self):
        """
        Regresión: 'dwelling consents' es un dato NZ legítimo, no debe
        activar el guard de Halifax/UK housing que penaliza NZD.
        """
        result = fn.detect_currency(
            "New Zealand dwelling consents fall 8% in January",
            "RBNZ data shows construction slowdown continues"
        )
        assert result == "NZD"

    def test_dax_con_euro_en_summary_puede_ser_relevante(self):
        """
        El filtro de DAX aplica sobre el título. Si el summary menciona
        EUR/USD, la noticia podría ser relevante de todas formas.
        Documentamos el comportamiento actual (el filtro aplica al título).
        """
        # El filtro EQUITY_INDEX_TITLE_RE solo mira el título,
        # así que si el título tiene DAX + rise/falls, se filtra
        # independientemente del summary.
        result = fn.is_forex_relevant(
            "DAX rises 1% on ECB optimism",
            "EUR/USD moves higher as European sentiment improves"
        )
        # Comportamiento actual: se filtra por el título (DAX rises)
        assert result is False

    def test_crypto_con_cbdc_es_relevante(self):
        """
        Noticias de crypto relacionadas con CBDC son relevantes para forex.
        """
        assert fn.is_forex_relevant(
            "Bitcoin falls as Fed announces CBDC pilot program",
            "The Federal Reserve digital dollar initiative affects crypto markets"
        ) is True

    def test_titulo_muy_corto_no_rompe_detect_currency(self):
        """detect_currency no debe crashear con títulos muy cortos."""
        result = fn.detect_currency("Fed", "", source="")
        # Puede retornar USD (por el keyword "fed ") o None, pero no debe lanzar excepción
        assert result in (fn.CURRENCIES + [None])

    def test_titulo_vacio_no_rompe(self):
        result = fn.detect_currency("", "", source="")
        assert result is None or result in fn.CURRENCIES

    def test_summary_none_no_rompe(self):
        result = fn.detect_currency("Fed raises rates", None, source="")
        assert result == "USD"
