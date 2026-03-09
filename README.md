# Dashboard de Análisis Fundamental Forex

Plataforma cuantitativa de análisis fundamental para las 8 divisas principales del G8 (USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD). Agrega 22 indicadores macroeconómicos en un score de fortaleza 0–100, con mapa de calor interactivo, análisis de carry trade, posicionamiento COT institucional y calendario económico en tiempo real.

![Status](https://img.shields.io/badge/Status-Producción-success) ![License](https://img.shields.io/badge/License-Propietario-red) ![Version](https://img.shields.io/badge/Version-6.3.0-informational)

---

## Características principales

- **Score de Fortaleza Fundamental (0–100)** — 22 indicadores ponderados en 6 tiers. Bandas: `>60` Alcista · `50–60` Neutral · `<50` Bajista
- **Mapa de Calor de Indicadores** — 22 métricas con codificación por color, fecha de publicación por celda y columna de score con sticky scroll
- **Carry Trade** — Ranking de pares por diferencial de tasas, historial de spread 18 meses y régimen Risk-On/Risk-Off en vivo
- **COT Institucional** — Posicionamiento neto Leveraged Funds (CFTC), actualización semanal, lógica contrarian integrada
- **Calendario Económico** — Scraping automático desde Investing.com con actuals, forecasts y previous. Horarios convertidos a zona local del usuario
- **ESI Proxy** — Economic Surprise Index: compara actuals vs forecasts del calendario + z-scores G8 relativos (v6.3)
- **PMI Servicios** — Cobertura del sector servicios (>70% del PIB en G8) como indicador adelantado (v6.2)
- **Análisis AI** — Narrativa fundamental por divisa generada con Groq/LLaMA
- **Recomendaciones de Pares** — LONG/SHORT con tres niveles de confianza (Alta >20pt · Media 12–20pt · Baja <12pt)

---

## Modelo de Scoring — 22 Indicadores en 6 Tiers

| Tier | Peso | Indicadores |
|------|------|-------------|
| T1 · Política Monetaria | **29%** | Tasa de Interés (10%), Momentum de Tasas (7%), Inflación (7%), Outlook BC (5%) |
| T2 · Balance Externo | **19%** | Cuenta Corriente (7%), Balanza Comercial (4%), Deuda Pública (4%), Términos de Intercambio (4%) |
| T3 · Sentimiento de Mercado | **21%** | COT Positioning (6%), PMI Servicios (4%), ESI Proxy (4%), PMI Manufacturero (3%), Confianza Consumidor (2%), Confianza Empresarial (2%), Flujos de Capital (0%) |
| T4 · Crecimiento y Empleo | **16%** | PIB Crecimiento (7%), Desempleo (5%), Producción Industrial (4%) |
| T5 · FX Confirmador | **11%** | FX Performance 1M basket-corrected (8%), Bono 10Y (3%) |
| T6 · Consumo y Salarios | **4%** | Ventas Minoristas (3%), Crecimiento Salarial (1%) |
| **Total** | **100%** | **22 indicadores** |

### Principio de causalidad

Los fundamentales causan el precio; el precio no debe ser el mayor peso del modelo. `fxPerformance1M` actúa como **confirmador** (8%), no como driver (era 28% hasta v5.x). Si los fundamentales son sólidos y el precio confirma → convicción. Si divergen → señal de alerta.

### Ajustes contextuales (±15 pts máximo)

| Ajuste | Condición | Efecto |
|--------|-----------|--------|
| Safe Haven dinámico | COT extremo + régimen Risk-Off | Prima proporcional para CHF y JPY |
| Hawkish Pause Boost (v6.1) | `bcOutlook=Hawkish` y `rateMomentum ≈ 0` | `rm efectivo = +0.3` (pausa activa ≠ inacción) |
| Stagflation Risk (v6.1) | `inflation < 2.5%` + `production < -0.5%` + `rateMomentum < -0.5` | Penalización 0 a −5 pts |
| Rate Cycle Haircut | Inicio de ciclo de cortes | Descuento proporcional al ritmo de bajadas |

---

## Fuentes de Datos

| Dato | Fuente | Frecuencia |
|------|--------|------------|
| Tipos de cambio FX | ExchangeRate-API / Open ER / Frankfurter/ECB (cascada 6 proveedores) | 60 segundos |
| Indicadores económicos (20) | Trading Economics (scraping automatizado) | Diaria |
| COT Positioning | CFTC oficial (Leveraged Funds, Futures Only) | Semanal (viernes) |
| Calendario económico | Investing.com (HTML POST, ventana 28d pasado + 30d futuro) | 3 veces/día |
| Análisis AI | Groq API (LLaMA) | Por cambio de datos |
| Outlook BC | Scraping comunicados bancos centrales | Diaria |

---

## Arquitectura técnica

**Frontend:** React 18 via Babel CDN — sin proceso de compilación. Aplicación de un único archivo HTML con JSONs estáticos como backend.

**Backend de datos:** 11 GitHub Actions workflows en Python que actualizan los JSON del repositorio automáticamente:

- `update-economic-data` — 22 indicadores por divisa desde Trading Economics
- `update-calendar` — Calendario económico desde Investing.com
- `update-cot` — Reporte CFTC semanal (Leveraged Funds)
- `update-rates` — Tasas de política monetaria y momentum 12 meses
- `update-fx-performance` — Rendimiento FX basket-corrected vs G8
- `update-ai-analysis` — Narrativas fundamentales por divisa (Groq)
- `update-news` — Noticias forex desde fuentes RSS
- `update-meetings` — Calendario de reuniones de bancos centrales
- `update-cot-data` — Series históricas COT para gráficos
- `update-extended-data` — Términos de intercambio, deuda pública
- `update-sitemap` — Sitemap automático

**Caché:** Los datos económicos se almacenan en `localStorage` con clave `DASHBOARD_VERSION`. Cada nueva versión invalida el caché automáticamente al detectar un cambio de versión.

---

## Estructura del proyecto

```
/
├── index.html                  # Dashboard principal (React + Babel)
├── about.html                  # Metodología completa en español
├── en.html                     # Landing page en inglés
├── carry-trade.html            # Análisis y ranking carry trade
├── news.html                   # Noticias forex agregadas
├── contact.html                # FAQ y contacto
├── guia-score-fortaleza.html   # Guía del modelo de scoring
├── guia-cot.html               # Guía COT
├── guia-bancos-centrales.html  # Guía de bancos centrales
├── guia-calendario-economico.html
├── guia-carry-trade.html
├── guia-pips.html
├── glosario-forex.html
├── tecnico-vs-fundamental.html
├── economic-data/              # JSONs de indicadores por divisa
├── calendar-data/              # calendar.json (Investing.com)
├── rates/                      # Historial tasas de política
├── cot-data/                   # Posicionamiento COT por divisa
├── fx-performance/             # Rendimiento FX basket-corrected
├── meetings-data/              # Calendario reuniones BC
├── news-data/                  # Noticias scrapeadas
├── ai-analysis/                # Análisis AI por divisa
├── extended-data/              # Términos de intercambio, deuda
└── scripts/                    # Scripts Python de actualización
    ├── update_economic_data.py
    ├── update_economic_calendar.py
    ├── update_cot.py
    └── ...
```

---

## Despliegue

### GitHub Pages (producción)

```bash
git clone https://github.com/GlobalInvesting/globalinvesting.github.io.git
# Subir cambios a main → GitHub Pages despliega automáticamente
# URL: https://globalinvesting.github.io/
```

### Local

```bash
# Python
python -m http.server 8000

# Node.js
npx http-server -p 8000

# Acceder en http://localhost:8000
```

No requiere compilación ni dependencias externas. Compatible con cualquier servidor estático (GitHub Pages, Netlify, Vercel, Apache, Nginx).

---

## Compatibilidad de navegadores

| Navegador | Versión mínima | Estado |
|-----------|----------------|--------|
| Google Chrome | 90+ | ✅ Totalmente soportado (recomendado) |
| Mozilla Firefox | 88+ | ✅ Totalmente soportado |
| Microsoft Edge | 90+ | ✅ Totalmente soportado |
| Safari | 14+ | ✅ Soportado |
| Mobile Safari / Chrome Mobile | 14+ / 90+ | ⚠️ Optimizado para tableta |

---

## Historial de versiones

**v6.3.0** (Marzo 2026)
- ESI Proxy (`economicSurprise`) — 22.º indicador, peso 4%. Arquitectura híbrida: (A) actuals vs forecast del calendario económico, (B) z-scores G8 relativos como respaldo
- Umbrales de sentimiento corregidos: `>60` Alcista · `50–60` Neutral · `<50` Bajista (antes >63 / 53–63 / <53)
- `DASHBOARD_VERSION` como constante única para invalidación automática de caché en todos los archivos

**v6.2.0** (Marzo 2026)
- PMI Servicios (`servicesPMI`) — 21.º indicador, peso 4%. Sector servicios representa >70% del PIB en G8
- Pipeline del calendario económico migrado a Investing.com como fuente única
- Deduplicación semántica y filtro de ruido en el calendario (eventos sin actuals/forecasts/previous excluidos)

**v6.1.0** (Marzo 2026)
- Hawkish Pause Boost: `bcOutlook=Hawkish` + `rateMomentum ≈ 0` → trato como ciclo hawkish leve (corrige subestimación de JPY/BOJ)
- Stagflation Risk: penalización cuando inflación baja coincide con contracción industrial y momentum negativo
- Umbral Hawkish bajado de `rm > 0.50%` a `rm > 0.20%` (+0.25pp en 12M = al menos un hike)

**v6.0.0** (Marzo 2026)
- Rebalanceo fundamental: `fxPerformance1M` 28% → 8% (confirmador, no driver)
- `interestRate` 7% → 10%, `rateMomentum` 4% → 7%
- Eliminado `inflationExpectations` (redundante con `inflation` + `rateMomentum`)
- Badge de "Score estructural" para divisas con fundamentales sólidos pero momentum débil

**v5.x** (Febrero–Marzo 2026)
- Sistema COT con lógica contrarian y filtro de confirmación de precio (v5.7)
- Análisis AI por divisa con Groq/LLaMA
- Recomendaciones de pares con niveles de confianza (Alta / Media / Baja)
- Safe haven dinámico proporcional al régimen de riesgo global (CHF/JPY)
- FX Performance basket-corrected vs G8 — elimina sesgo USD (v5.9)
- 11 GitHub Actions workflows de actualización automática

---

## Aviso legal

Este dashboard es una herramienta de análisis cuantitativo con fines informativos y educativos exclusivamente. Los scores, rankings y recomendaciones de pares son cálculos basados en datos macroeconómicos públicos y **no constituyen asesoramiento financiero ni señales de trading**. El trading de divisas (forex) conlleva riesgos significativos de pérdida de capital. Consulte con un asesor financiero certificado antes de tomar decisiones de inversión.

---

## Contacto

- **GitHub:** [@GlobalInvesting](https://github.com/GlobalInvesting/)
- **Email:** globalinvestingmarkets@gmail.com
- **LinkedIn:** [santiago-pla-casuriaga](https://www.linkedin.com/in/santiago-pla-casuriaga/)

---

**© 2026 Santiago Plá Casuriaga · Global Investing. Todos los derechos reservados.**
Este software es de uso exclusivo de su autor. No se permite su reproducción, distribución ni modificación sin autorización expresa por escrito.
