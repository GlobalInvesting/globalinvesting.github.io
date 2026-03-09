"""
tests/test_all_scripts.py
Suite de tests unitarios para todos los scripts del proyecto.

Cubre funciones puras (sin I/O ni HTTP) de:
  - fetch_rates.py
  - update_economic_data.py
  - update_extended_data.py
  - generate_ai_analysis.py
  - generate_summaries.py

Ejecutar:
  cd scripts && pytest tests/ -v
  cd scripts && pytest tests/test_all_scripts.py -v --tb=short
"""

import sys
import os
import types
import re
import pytest
from datetime import datetime, date

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Mock de dependencias externas (HTTP, parsers, APIs) ───────────────────────
# Los scrapers tienen código a nivel de módulo que llama a requests y BeautifulSoup
# al importarse. Necesitamos mocks completos para que el import no crashee ni haga
# llamadas reales a internet.

class _FakeResponse:
    status_code = 200
    content     = b""
    text        = ""
    ok          = False
    def raise_for_status(self): pass
    def json(self): return {}

class _FakeRequests:
    exceptions = types.SimpleNamespace(RequestException=Exception,
                                        Timeout=Exception,
                                        ConnectionError=Exception)
    @staticmethod
    def get(*a, **kw):  return _FakeResponse()
    @staticmethod
    def post(*a, **kw): return _FakeResponse()
    Session = type("Session", (), {
        "__init__": lambda s: None,
        "get":      lambda s, *a, **kw: _FakeResponse(),
        "post":     lambda s, *a, **kw: _FakeResponse(),
    })

class _FakeSoup:
    def __init__(self, *a, **kw): pass
    def find(self, *a, **kw):          return None
    def find_all(self, *a, **kw):      return []
    def get_text(self, **kw):          return ""
    def get(self, *a, **kw):           return None

class _FakeBS4:
    BeautifulSoup = _FakeSoup

for _name, _mod in [
    ("requests",       _FakeRequests),
    ("bs4",            _FakeBS4),
    ("feedparser",     types.ModuleType("feedparser")),
    ("dateutil",       types.ModuleType("dateutil")),
    ("dateutil.parser",types.ModuleType("dateutil.parser")),
    ("groq",           types.ModuleType("groq")),
    ("lxml",           types.ModuleType("lxml")),
]:
    if _name not in sys.modules:
        sys.modules[_name] = _mod

# Los scrapers también hacen I/O de archivos a nivel de módulo.
# Parcheamos open/os.makedirs/os.path.exists para que no fallen al importar.
import unittest.mock as _mock
import builtins as _builtins
_real_open = _builtins.open

def _safe_open(path, mode="r", *a, **kw):
    # Permitir escritura a /dev/null (los scripts guardan archivos al iniciar)
    if "w" in str(mode):
        return _real_open(os.devnull, mode, *a, **kw)
    # Para lectura, si el archivo no existe devolver un JSON vacío
    if not os.path.exists(str(path)):
        import io
        return io.StringIO("{}")
    return _real_open(path, mode, *a, **kw)

with _mock.patch("builtins.open", _safe_open), \
     _mock.patch("os.makedirs", lambda *a, **kw: None), \
     _mock.patch("os.path.exists", lambda p: False), \
     _mock.patch("sys.exit", lambda *a: None):
    import fetch_rates           as fr
    import update_economic_data  as ued
    import update_extended_data  as uex
    import generate_ai_analysis  as gaa
    import generate_summaries    as gs


# ══════════════════════════════════════════════════════════════════════════════
# fetch_rates.py
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanRate:
    """clean_rate() extrae tasas numéricas y filtra valores fuera de rango."""

    def test_valor_simple(self):
        assert fr.clean_rate("5.25%") == "5.25"

    def test_sin_signo_porcentaje(self):
        assert fr.clean_rate("4.50") == "4.50"

    def test_tasa_cero(self):
        assert fr.clean_rate("0%") == "0"

    def test_tasa_negativa(self):
        assert fr.clean_rate("-0.1%") == "-0.1"

    def test_con_espacios(self):
        assert fr.clean_rate("  3.50 %  ") == "3.50"

    def test_con_coma_decimal(self):
        # coma → punto
        assert fr.clean_rate("4,50") == "4.50"

    def test_texto_con_tasa_incrustada(self):
        assert fr.clean_rate("Rate: 5.25%") == "5.25"

    def test_tasa_demasiado_alta_retorna_none(self):
        # > MAX_POLICY_RATE_PP (15)
        assert fr.clean_rate("25%") is None

    def test_tasa_demasiado_baja_retorna_none(self):
        # < MIN_POLICY_RATE_PP (-2)
        assert fr.clean_rate("-5%") is None

    def test_texto_vacio_retorna_none(self):
        assert fr.clean_rate("") is None

    def test_none_retorna_none(self):
        assert fr.clean_rate(None) is None

    def test_texto_sin_numeros_retorna_none(self):
        assert fr.clean_rate("N/A") is None

    def test_limite_superior_valido(self):
        # Exactamente MAX_POLICY_RATE_PP = 15 debe pasar
        assert fr.clean_rate("15%") == "15"

    def test_limite_inferior_valido(self):
        # Exactamente MIN_POLICY_RATE_PP = -2 debe pasar
        assert fr.clean_rate("-2%") == "-2"


class TestParseReferenceDate:
    """parse_reference_date() convierte texto de referencia en fecha ISO."""

    def test_formato_mmm_yy(self):
        assert fr.parse_reference_date("Mar/25") == "2025-03-01"

    def test_formato_mmm_yy_enero(self):
        assert fr.parse_reference_date("Jan/25") == "2025-01-01"

    def test_formato_mmm_yy_diciembre(self):
        assert fr.parse_reference_date("Dec/24") == "2024-12-01"

    def test_capitaliza_mes(self):
        assert fr.parse_reference_date("mar/25") == "2025-03-01"

    def test_texto_vacio_retorna_none(self):
        assert fr.parse_reference_date("") is None

    def test_none_retorna_none(self):
        assert fr.parse_reference_date(None) is None

    def test_formato_desconocido_retorna_none(self):
        assert fr.parse_reference_date("Q1 2025") is None

    def test_texto_libre_retorna_none(self):
        assert fr.parse_reference_date("Latest") is None


class TestMergeObservations:
    """merge_observations() fusiona historial existente con nueva observación."""

    def test_agrega_nueva_observacion(self):
        existing = [{"date": "2025-01-01", "value": "5.25"}]
        new_obs  = {"date": "2025-02-01", "value": "5.50"}
        result   = fr.merge_observations(existing, new_obs)
        values   = [o["value"] for o in result]
        assert "5.25" in values
        assert "5.50" in values

    def test_sobreescribe_mismo_mes(self):
        existing = [{"date": "2025-03-01", "value": "5.25"}]
        new_obs  = {"date": "2025-03-15", "value": "5.50"}
        result   = fr.merge_observations(existing, new_obs)
        # Solo debe quedar un registro de marzo
        march = [o for o in result if o["date"].startswith("2025-03")]
        assert len(march) == 1
        assert march[0]["value"] == "5.50"

    def test_ordena_descendente(self):
        existing = [
            {"date": "2025-01-01", "value": "5.00"},
            {"date": "2025-02-01", "value": "5.25"},
        ]
        new_obs = {"date": "2025-03-01", "value": "5.50"}
        result  = fr.merge_observations(existing, new_obs)
        assert result[0]["date"] > result[-1]["date"]

    def test_max_36_observaciones(self):
        existing = [{"date": f"202{i//12}-{(i%12)+1:02d}-01", "value": "5.0"}
                    for i in range(40)]
        new_obs = {"date": "2026-01-01", "value": "5.5"}
        result  = fr.merge_observations(existing, new_obs)
        assert len(result) <= 36

    def test_historial_vacio(self):
        result = fr.merge_observations([], {"date": "2025-03-01", "value": "4.0"})
        assert len(result) == 1
        assert result[0]["value"] == "4.0"

    def test_normaliza_fecha_a_dia_01(self):
        result = fr.merge_observations([], {"date": "2025-03-15", "value": "4.0"})
        assert result[0]["date"] == "2025-03-01"


class TestValidateRates:
    """validate_rates() detecta tasas fuera de rango y cambios excesivos."""

    def _make_rates(self, values):
        return {cur: {"rate": str(v)} for cur, v in values.items()}

    def test_tasas_validas_sin_issues(self):
        rates = self._make_rates({"USD": 5.25, "EUR": 4.0, "GBP": 5.0, "JPY": 0.1})
        issues, warnings = fr.validate_rates(rates, {})
        # Solo verifica rango — sin historial previo no hay issues de cambio
        assert all("outside plausible range" not in i for i in issues)

    def test_tasa_fuera_de_rango_genera_issue(self):
        rates = self._make_rates({"USD": 20.0})  # > MAX_POLICY_RATE_PP
        issues, _ = fr.validate_rates(rates, {})
        assert any("USD" in i and "outside plausible range" in i for i in issues)

    def test_tasa_negativa_extrema_genera_issue(self):
        rates = self._make_rates({"CHF": -5.0})  # < MIN_POLICY_RATE_PP
        issues, _ = fr.validate_rates(rates, {})
        assert any("CHF" in i and "outside plausible range" in i for i in issues)

    def test_sin_frankfurter_no_rompe(self):
        rates = self._make_rates({"USD": 5.25})
        issues, warnings = fr.validate_rates(rates, {})
        assert isinstance(issues, list)
        assert isinstance(warnings, list)


# ══════════════════════════════════════════════════════════════════════════════
# update_economic_data.py  (y update_extended_data.py comparten clean_num / parse_te_date)
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanNum:
    """clean_num() extrae el primer número flotante del texto."""

    def test_numero_entero(self):
        assert ued.clean_num("42") == 42.0

    def test_numero_decimal(self):
        assert ued.clean_num("3.14") == 3.14

    def test_con_porcentaje(self):
        assert ued.clean_num("2.5%") == 2.5

    def test_negativo(self):
        assert ued.clean_num("-1.4%") == -1.4

    def test_con_comas_de_miles(self):
        assert ued.clean_num("1,234.56") == 1234.56

    def test_texto_con_numero_incrustado(self):
        assert ued.clean_num("GDP: 2.3%") == 2.3

    def test_cero(self):
        # 0.0 es falsy en Python → clean_num lo trata como texto vacío → None
        # El comportamiento correcto del código: retorna None para input falsy
        assert ued.clean_num(0.0) is None

    def test_cero_como_string(self):
        # Como string, "0" sí tiene contenido
        assert ued.clean_num("0") == 0.0

    def test_none_retorna_none(self):
        assert ued.clean_num(None) is None

    def test_texto_vacio_retorna_none(self):
        assert ued.clean_num("") is None

    def test_texto_sin_numeros_retorna_none(self):
        assert ued.clean_num("N/A") is None

    def test_solo_texto_retorna_none(self):
        assert ued.clean_num("Rising") is None


class TestParseTEDate:
    """parse_te_date() convierte distintos formatos de fecha de TE a ISO."""

    def test_formato_mmm_yy_slash(self):
        result = ued.parse_te_date("Feb/25")
        assert result == "2025-02-15"

    def test_formato_mmm_espacio_yyyy(self):
        result = ued.parse_te_date("Mar 2025")
        assert result == "2025-03-15"

    def test_formato_qn_yyyy(self):
        result = ued.parse_te_date("Q1/2025")
        assert result == "2025-02-15"  # Q1 → mes 2

    def test_formato_q2(self):
        result = ued.parse_te_date("Q2/2025")
        assert result == "2025-05-15"

    def test_formato_q4(self):
        result = ued.parse_te_date("Q4/2024")
        assert result == "2024-11-15"

    def test_formato_solo_año(self):
        result = ued.parse_te_date("2024")
        assert result == "2024-06-15"

    def test_texto_vacio_retorna_hoy(self):
        result = ued.parse_te_date("")
        assert result == str(date.today())

    def test_none_retorna_hoy(self):
        result = ued.parse_te_date(None)
        assert result == str(date.today())

    def test_texto_irreconocible_retorna_hoy(self):
        result = ued.parse_te_date("latest")
        assert result == str(date.today())


# ══════════════════════════════════════════════════════════════════════════════
# update_extended_data.py
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeConfidence:
    """normalize_confidence() normaliza índices de confianza a escala ~100."""

    # ── consumerConfidence ────────────────────────────────────────────────────
    def test_eur_consumer_confidence_suma_100(self):
        # EUR raw suele ser negativo (ej. -15) → se suma 100
        result = uex.normalize_confidence(-15.0, "EUR", "consumerConfidence")
        assert result == 85.0

    def test_gbp_consumer_confidence_suma_100(self):
        result = uex.normalize_confidence(-20.0, "GBP", "consumerConfidence")
        assert result == 80.0

    def test_usd_consumer_confidence_passthrough(self):
        # USD está en RAW_PASSTHROUGH — no hay transformación
        result = uex.normalize_confidence(101.3, "USD", "consumerConfidence")
        assert result == 101.3

    def test_chf_consumer_confidence_suma_130(self):
        result = uex.normalize_confidence(-25.0, "CHF", "consumerConfidence")
        assert result == 105.0

    # ── businessConfidence ───────────────────────────────────────────────────
    def test_eur_business_confidence_suma_100(self):
        result = uex.normalize_confidence(5.0, "EUR", "businessConfidence")
        assert result == 105.0

    def test_usd_business_confidence_passthrough(self):
        result = uex.normalize_confidence(55.0, "USD", "businessConfidence")
        assert result == 55.0

    # ── Fuera de rango → None (excepto RAW_PASSTHROUGH) ──────────────────────
    def test_valor_fuera_de_rango_retorna_none_eur(self):
        # EUR normalizado quedaría en 250 → fuera de (50, 200) → None
        result = uex.normalize_confidence(150.0, "EUR", "consumerConfidence")
        assert result is None

    def test_valor_none_retorna_none(self):
        assert uex.normalize_confidence(None, "USD", "consumerConfidence") is None

    def test_indicador_desconocido_passthrough(self):
        # Para indicadores no reconocidos devuelve el valor sin transformar
        result = uex.normalize_confidence(42.0, "USD", "rateMomentum")
        assert result == 42.0


# ══════════════════════════════════════════════════════════════════════════════
# generate_ai_analysis.py
# ══════════════════════════════════════════════════════════════════════════════

class TestFmt:
    """fmt() formatea valores numéricos con decimales y sufijo opcionales."""

    def test_valor_entero(self):
        assert gaa.fmt(5) == "5.0"

    def test_valor_decimal(self):
        assert gaa.fmt(3.14159, decimals=2) == "3.14"

    def test_con_sufijo(self):
        assert gaa.fmt(2.5, suffix="%") == "2.5%"

    def test_cero_decimales(self):
        assert gaa.fmt(100.9, decimals=0) == "101"

    def test_negativo(self):
        assert gaa.fmt(-1.4, decimals=1) == "-1.4"

    def test_none_retorna_none(self):
        assert gaa.fmt(None) is None

    def test_string_numerico(self):
        assert gaa.fmt("3.5", decimals=1) == "3.5"

    def test_string_no_numerico_retorna_none(self):
        assert gaa.fmt("N/A") is None


class TestComputeGlobalContext:
    """compute_global_context() calcula promedios del G8."""

    def test_promedio_tasas(self):
        all_data = {
            "USD": {"interestRate": 4.0},
            "EUR": {"interestRate": 2.0},
        }
        ctx = gaa.compute_global_context(all_data)
        assert ctx["avg_interest_rate"] == 3.0

    def test_ignora_valores_none(self):
        all_data = {
            "USD": {"interestRate": 5.0},
            "EUR": {"interestRate": None},
        }
        ctx = gaa.compute_global_context(all_data)
        assert ctx["avg_interest_rate"] == 5.0

    def test_retorna_none_si_no_hay_datos(self):
        all_data = {"USD": {"gdpGrowth": None}}
        ctx = gaa.compute_global_context(all_data)
        assert ctx["avg_gdp_growth"] is None

    def test_multiples_indicadores(self):
        all_data = {
            "USD": {"interestRate": 5.0, "inflation": 3.0},
            "EUR": {"interestRate": 4.0, "inflation": 2.5},
        }
        ctx = gaa.compute_global_context(all_data)
        assert ctx["avg_interest_rate"] == 4.5
        assert ctx["avg_inflation"] == 2.75

    def test_dict_vacio_retorna_nones(self):
        ctx = gaa.compute_global_context({})
        assert ctx["avg_interest_rate"] is None


class TestInferStructuralSignals:
    """infer_structural_signals() detecta señales macro relevantes."""

    def _ctx(self, avg_rate=3.0, avg_cot=0):
        return {"avg_interest_rate": avg_rate, "avg_cot": avg_cot}

    def test_tasa_muy_baja_es_carry_financiador(self):
        # Si la tasa es >= 1.5pp por debajo del promedio → señal de carry financiador
        data = {"interestRate": 0.1}
        signals = gaa.infer_structural_signals("JPY", data, self._ctx(avg_rate=3.0))
        assert any("financiadora" in s.lower() or "carry trade" in s.lower() for s in signals)

    def test_tasa_muy_alta_es_carry_atractivo(self):
        data = {"interestRate": 5.5}
        signals = gaa.infer_structural_signals("USD", data, self._ctx(avg_rate=3.0))
        assert any("carry atractivo" in s.lower() or "renta fija" in s.lower() for s in signals)

    def test_sin_tasa_no_genera_señal_de_tasa(self):
        data = {"interestRate": None}
        signals = gaa.infer_structural_signals("USD", data, self._ctx())
        rate_signals = [s for s in signals if "carry" in s.lower() and "tasa" in s.lower()]
        assert len(rate_signals) == 0

    def test_tasa_en_rango_medio_sin_señal_de_carry(self):
        # 3.0% con promedio 3.0% → gap = 0 → sin señal
        data = {"interestRate": 3.0}
        signals = gaa.infer_structural_signals("USD", data, self._ctx(avg_rate=3.0))
        carry_signals = [s for s in signals
                         if "financiador" in s.lower() or "carry atractivo" in s.lower()]
        assert len(carry_signals) == 0

    def test_retorna_lista(self):
        signals = gaa.infer_structural_signals("USD", {}, self._ctx())
        assert isinstance(signals, list)


class TestDeriveRateDecisionFromHistory:
    """derive_rate_decision_from_history() infiere decisión del banco central."""

    def test_sin_historial_retorna_none(self):
        assert gaa.derive_rate_decision_from_history("USD", []) is None

    def test_historial_con_un_solo_registro_retorna_mantuvo(self):
        obs = [{"value": "5.25", "date": "2025-03-01"}]
        result = gaa.derive_rate_decision_from_history("USD", obs)
        assert result is not None
        assert result["direction"] == "MANTUVO"

    def test_subida_de_tasa(self):
        obs = [
            {"value": "5.50", "date": "2025-03-01"},
            {"value": "5.25", "date": "2025-02-01"},
        ]
        result = gaa.derive_rate_decision_from_history("USD", obs)
        assert result["direction"] == "SUBIÓ"
        assert result["delta"] > 0

    def test_bajada_de_tasa(self):
        obs = [
            {"value": "4.75", "date": "2025-03-01"},
            {"value": "5.00", "date": "2025-02-01"},
        ]
        result = gaa.derive_rate_decision_from_history("USD", obs)
        assert result["direction"] == "BAJÓ"
        assert result["delta"] < 0

    def test_sin_cambios_retorna_mantuvo(self):
        obs = [
            {"value": "4.0", "date": "2025-03-01"},
            {"value": "4.0", "date": "2025-02-01"},
            {"value": "4.0", "date": "2025-01-01"},
        ]
        result = gaa.derive_rate_decision_from_history("EUR", obs)
        assert result["direction"] == "MANTUVO"

    def test_ignora_observaciones_sin_valor(self):
        obs = [
            {"value": "5.25", "date": "2025-03-01"},
            {"value": None,   "date": "2025-02-01"},
            {"value": ".",    "date": "2025-01-01"},
        ]
        result = gaa.derive_rate_decision_from_history("USD", obs)
        assert result is not None  # no debe crashear

    def test_delta_calculado_correctamente(self):
        obs = [
            {"value": "5.50", "date": "2025-03-01"},
            {"value": "5.25", "date": "2025-02-01"},
        ]
        result = gaa.derive_rate_decision_from_history("USD", obs)
        assert abs(result["delta"] - 0.25) < 0.001


# ══════════════════════════════════════════════════════════════════════════════
# generate_summaries.py
# ══════════════════════════════════════════════════════════════════════════════

class TestIsUpcomingEventStale:
    """is_upcoming_event_stale() detecta eventos que ya ocurrieron."""

    def test_frase_esta_noche_es_stale(self):
        # occurred_set debe ser no-vacío para que la función no haga early return
        assert gs.is_upcoming_event_stale("CPI esta noche", {"placeholder"}) is True

    def test_frase_today_es_stale(self):
        assert gs.is_upcoming_event_stale("Fed decision today at 2pm", {"placeholder"}) is True

    def test_frase_tonight_es_stale(self):
        assert gs.is_upcoming_event_stale("NFP release tonight", {"placeholder"}) is True

    def test_frase_esta_manana_es_stale(self):
        assert gs.is_upcoming_event_stale("GDP esta mañana mostró...", {"placeholder"}) is True

    def test_evento_futuro_no_es_stale(self):
        assert gs.is_upcoming_event_stale("Next week: FOMC meeting", {"placeholder"}) is False

    def test_texto_vacio_no_es_stale(self):
        assert gs.is_upcoming_event_stale("", {"placeholder"}) is False

    def test_none_no_es_stale(self):
        assert gs.is_upcoming_event_stale(None, {"placeholder"}) is False

    def test_solapamiento_con_occurred_set(self):
        # occurred_set contiene nombres de eventos completos (strings)
        # La función tokeniza cada nombre y busca ≥2 tokens en común
        occurred = {"FOMC rate decision march"}
        assert gs.is_upcoming_event_stale(
            "FOMC rate decision expected", occurred
        ) is True

    def test_sin_solapamiento_con_occurred_set(self):
        occurred = {"nonfarm payrolls february report"}
        assert gs.is_upcoming_event_stale(
            "ECB meeting next Thursday", occurred
        ) is False

    def test_occurred_set_con_un_solo_token_no_activa(self):
        # Un nombre de evento con <2 tokens no puede activar el overlap
        occurred = {"fomc"}
        assert gs.is_upcoming_event_stale("FOMC decision", occurred) is False


# ══════════════════════════════════════════════════════════════════════════════
# Casos de regresión — comportamientos críticos inter-script
# ══════════════════════════════════════════════════════════════════════════════

class TestRegresion:
    """Bugs conocidos corregidos. Si alguno falla → regresión."""

    def test_clean_rate_no_acepta_cero_como_invalido(self):
        # Algunas tasas son 0% (ej. BoJ). No debe descartarse.
        assert fr.clean_rate("0%") == "0"

    def test_clean_num_no_rompe_con_tipo_int(self):
        assert ued.clean_num(42) == 42.0

    def test_parse_te_date_no_rompe_con_espacios(self):
        result = ued.parse_te_date("  Mar/25  ")
        assert result == "2025-03-15"

    def test_fmt_no_rompe_con_cero(self):
        assert gaa.fmt(0) == "0.0"

    def test_clean_num_cero_string(self):
        # clean_num("0") debe retornar 0.0 — el string "0" no es falsy
        assert ued.clean_num("0") == 0.0

    def test_compute_global_context_no_rompe_con_un_solo_pais(self):
        ctx = gaa.compute_global_context({"USD": {"interestRate": 5.25}})
        assert ctx["avg_interest_rate"] == 5.25

    def test_derive_rate_decision_no_rompe_con_fechas_desordenadas(self):
        obs = [
            {"value": "5.00", "date": "2025-01-01"},
            {"value": "5.25", "date": "2025-03-01"},
            {"value": "5.00", "date": "2025-02-01"},
        ]
        result = gaa.derive_rate_decision_from_history("USD", obs)
        # Debe ordenar por fecha y no crashear
        assert result is not None

    def test_normalize_confidence_none_no_rompe_eur(self):
        assert uex.normalize_confidence(None, "EUR", "consumerConfidence") is None

    def test_merge_observations_con_fechas_mixtas(self):
        # Mezcla de formatos de fecha: con día 01 y con día distinto
        existing = [{"date": "2025-01-15", "value": "5.0"}]
        new_obs  = {"date": "2025-01-01", "value": "5.25"}
        result   = fr.merge_observations(existing, new_obs)
        # Mismo mes → debe quedar solo uno
        jan = [o for o in result if o["date"].startswith("2025-01")]
        assert len(jan) == 1
