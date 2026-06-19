"""
fx_config.py — Shared currency configuration for all scripts.

Centralizes CURRENCIES, COUNTRY_META and CURRENCY_NAMES to avoid
duplication across fetch_news.py, generate_narrative_signals.py and
other pipeline scripts.
"""

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "NOK", "SEK"]

CURRENCY_NAMES = {
    "USD": "US Dollar",
    "EUR": "Euro",
    "GBP": "British Pound Sterling",
    "JPY": "Japanese Yen",
    "AUD": "Australian Dollar",
    "CAD": "Canadian Dollar",
    "CHF": "Swiss Franc",
    "NZD": "New Zealand Dollar",
    "NOK": "Norwegian Krone",
    "SEK": "Swedish Krona",
}

COUNTRY_META = {
    "USD": {"name": "United States",  "bank": "Federal Reserve (Fed)"},
    "EUR": {"name": "Eurozone",        "bank": "European Central Bank (ECB)"},
    "GBP": {"name": "United Kingdom",  "bank": "Bank of England (BoE)"},
    "JPY": {"name": "Japan",           "bank": "Bank of Japan (BoJ)"},
    "AUD": {"name": "Australia",       "bank": "Reserve Bank of Australia (RBA)"},
    "CAD": {"name": "Canada",          "bank": "Bank of Canada (BoC)"},
    "CHF": {"name": "Switzerland",     "bank": "Swiss National Bank (SNB)"},
    "NZD": {"name": "New Zealand",     "bank": "Reserve Bank of New Zealand (RBNZ)"},
    "NOK": {"name": "Norway",          "bank": "Norges Bank (NB)"},
    "SEK": {"name": "Sweden",          "bank": "Sveriges Riksbank (Riksbank)"},
}

CURRENCY_MACRO_CONTEXT = {
    "USD": (
        "Global reserve currency and primary safe-haven asset. Benefits from risk-off flows, "
        "geopolitical tensions and strong US macro data. Highly sensitive to Fed stance "
        "(hawkish/dovish) and the interest rate differential vs other G10 economies."
    ),
    "EUR": (
        "Net energy importer. Geopolitical conflict raises energy costs, weighing on Eurozone "
        "growth and creating a dilemma for the ECB between fighting inflation and avoiding "
        "recession. Sensitive to peripheral bond spreads and ECB policy stance."
    ),
    "GBP": (
        "Not a safe-haven asset. Sensitive to UK inflation, BoE policy and UK labour market data. "
        "In risk-off environments it falls against USD, JPY and CHF."
    ),
    "JPY": (
        "Traditional safe-haven but weakened when oil rises (Japan imports nearly all its crude). "
        "Dominant driver: US-JP rate differential. Fed hawkish or BoJ dovish = JPY bearish."
    ),
    "AUD": (
        "Risk-correlated currency linked to commodities (iron ore, copper) and the Chinese "
        "economic cycle. Falls in risk-off environments. RBA hawkish stance provides "
        "domestic support."
    ),
    "CAD": (
        "Correlated with WTI crude: Canada is a net oil exporter. High oil = structural CAD "
        "support even with a dovish BoC. CUSMA/USMCA trade relationship with the US is the "
        "primary tail risk."
    ),
    "CHF": (
        "Quintessential safe-haven asset. Appreciates in crises, wars and risk-off episodes. "
        "SNB may intervene to limit excessive appreciation — distinguish active intervention "
        "(bearish) from a verbal ceiling threat (mixed)."
    ),
    "NZD": (
        "High-beta risk currency. Falls in crises/wars due to global risk-off (not direct "
        "oil correlation). RBNZ policy and domestic NZ data are the primary fundamental drivers."
    ),
    "NOK": (
        "Commodity-linked G10 currency with the strongest structural correlation to Brent crude: "
        "Norway is a major oil and gas exporter. Risk-sensitive — falls in global risk-off "
        "environments. Norges Bank is one of the most forward-guidance-driven CBs in G10; "
        "the rate path signal at each meeting is a primary driver of NOK volatility."
    ),
    "SEK": (
        "European risk proxy. Highly sensitive to global risk appetite and Eurozone growth "
        "given Sweden's trade links. The Riksbank has run one of the most volatile policy "
        "cycles in G10 (negative rates, QE, rapid normalization 2022-2024). "
        "EUR/SEK is the institutional benchmark pair — watched closely as a barometer "
        "of European financial stress."
    ),
}
