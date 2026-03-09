# Dashboard de Análisis Fundamental Forex

Plataforma de análisis cuantitativo para las 8 divisas principales del G8 (USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD). Construida con datos macroeconómicos reales, posicionamiento institucional COT y calendario económico en tiempo real.

🔗 **[globalinvesting.github.io](https://globalinvesting.github.io/)**

![Status](https://img.shields.io/badge/Status-Producción-success) ![License](https://img.shields.io/badge/License-Propietario-red) ![Version](https://img.shields.io/badge/Version-6.3.0-informational)

---

## ¿Qué hace?

- Calcula un **score de fortaleza fundamental (0–100)** por divisa combinando más de 20 indicadores macroeconómicos ponderados
- Muestra un **mapa de calor interactivo** con todos los indicadores por divisa en una sola pantalla
- Genera **recomendaciones de pares** LONG/SHORT con niveles de confianza cuantificados
- Incluye análisis de **carry trade** con ranking de diferenciales de tasas y régimen de riesgo global en vivo
- Integra **posicionamiento COT institucional** (CFTC, Leveraged Funds) con lógica contrarian
- Provee un **calendario económico** con actuals vs forecasts y conversión automática de zona horaria
- Genera **narrativas de análisis AI** por divisa

Los datos se actualizan automáticamente varias veces al día mediante pipelines en GitHub Actions.

---

## Stack

- **Frontend:** React 18 (Babel CDN) — sin proceso de compilación, archivo HTML único
- **Hosting:** GitHub Pages (sitio estático)
- **Datos:** Pipelines Python automatizados → JSON estáticos en el repositorio
- **AI:** Groq API

---

## Despliegue local

```bash
git clone https://github.com/GlobalInvesting/globalinvesting.github.io.git
cd globalinvesting.github.io
python -m http.server 8000
# Acceder en http://localhost:8000
```

No requiere compilación ni dependencias adicionales.

---

## Metodología

La metodología completa, incluyendo la descripción de indicadores, fuentes de datos y criterios de interpretación, está disponible en la sección [Acerca de](https://globalinvesting.github.io/about.html) de la plataforma.

---

## Aviso legal

Herramienta de análisis cuantitativo con fines informativos y educativos exclusivamente. No constituye asesoramiento financiero ni señales de trading. Consulte con un asesor financiero certificado antes de tomar decisiones de inversión.

---

## Contacto

- **GitHub:** [@GlobalInvesting](https://github.com/GlobalInvesting/)
- **Email:** globalinvestingmarkets@gmail.com
- **LinkedIn:** [santiago-pla-casuriaga](https://www.linkedin.com/in/santiago-pla-casuriaga/)

---

**© 2026 Santiago Plá Casuriaga · Global Investing. Todos los derechos reservados.**
Este software es de uso exclusivo de su autor. No se permite su reproducción, distribución ni modificación sin autorización expresa por escrito.
