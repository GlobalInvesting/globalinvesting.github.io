"""
fx_config.py — Configuración compartida de divisas para todos los scripts.

Centraliza CURRENCIES, COUNTRY_META y CURRENCY_NAMES para evitar duplicación
entre fetch_news.py, generate_ai_analysis.py y generate_summaries.py.
"""

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

CURRENCY_NAMES = {
    "USD": "Dólar Estadounidense",
    "EUR": "Euro",
    "GBP": "Libra Esterlina",
    "JPY": "Yen Japonés",
    "AUD": "Dólar Australiano",
    "CAD": "Dólar Canadiense",
    "CHF": "Franco Suizo",
    "NZD": "Dólar Neozelandés",
}

COUNTRY_META = {
    "USD": {"name": "Estados Unidos",  "bank": "Reserva Federal (Fed)"},
    "EUR": {"name": "Eurozona",        "bank": "Banco Central Europeo (BCE)"},
    "GBP": {"name": "Reino Unido",     "bank": "Banco de Inglaterra (BoE)"},
    "JPY": {"name": "Japón",           "bank": "Banco de Japón (BoJ)"},
    "AUD": {"name": "Australia",       "bank": "Banco de la Reserva de Australia (RBA)"},
    "CAD": {"name": "Canadá",          "bank": "Banco de Canadá (BoC)"},
    "CHF": {"name": "Suiza",           "bank": "Banco Nacional Suizo (SNB)"},
    "NZD": {"name": "Nueva Zelanda",   "bank": "Banco de la Reserva de Nueva Zelanda (RBNZ)"},
}

CURRENCY_MACRO_CONTEXT = {
    "USD": "Activo refugio global y divisa de reserva. Se beneficia de risk-off, tensiones geopolíticas y datos macro sólidos en EEUU. Sensible a postura Fed (hawkish/dovish) y al diferencial de tasas con otras economías G10.",
    "EUR": "Importador neto de energía. Conflicto geopolítico = mayores costes energéticos = presión sobre crecimiento eurozona = dilema BCE entre inflación y recesión. Sensible a spreads de bonos periféricos y postura BCE.",
    "GBP": "No es activo refugio. Sensible a inflación UK, política del BoE y datos laborales británicos. En risk-off cae frente a USD, JPY y CHF.",
    "JPY": "Activo refugio tradicional pero debilitado cuando sube el petróleo (Japón importa casi todo su crudo). Driver dominante: diferencial tasas US-JP. Fed hawkish o BoJ dovish = JPY bajista.",
    "AUD": "Divisa de riesgo correlacionada con commodities (hierro, cobre) y ciclo económico chino. En entornos risk-off cae. RBA hawkish ofrece soporte doméstico.",
    "CAD": "Correlacionada con petróleo WTI: Canadá es exportador neto de crudo. Petróleo alto = soporte estructural CAD incluso con BoC dovish. USMCA y comercio con EEUU son el mayor riesgo de cola.",
    "CHF": "Activo refugio por excelencia. Se aprecia en crisis/guerra/risk-off. SNB puede intervenir para limitar apreciación excesiva — distinguir intervención activa (bajista) de amenaza como techo (mixto).",
    "NZD": "Divisa de riesgo de alta beta. En crisis/guerra cae por risk-off global (no por correlación directa con petróleo). RBNZ y datos domésticos NZ son drivers fundamentales propios.",
}
