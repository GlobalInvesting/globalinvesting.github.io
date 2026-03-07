# Dashboard de Análisis Fundamental Forex

Una plataforma integral de grado institucional para el monitoreo y análisis en tiempo real de indicadores económicos fundamentales en los principales pares de divisas. Diseñado para profesionales financieros, traders institucionales y participantes sofisticados del mercado.

![Status](https://img.shields.io/badge/Status-Producción-success) ![License](https://img.shields.io/badge/License-Propietario-red) ![Version](https://img.shields.io/badge/Version-5.6.0-informational)

## Descripción General

Este dashboard proporciona herramientas de análisis fundamental de grado profesional para el mercado de divisas, integrando datos de mercado en tiempo real con indicadores macroeconómicos completos. La plataforma emplea metodologías cuantitativas para evaluar la fortaleza de divisas y el valor relativo en las monedas G10.

### Capacidades Clave

**Evaluación de Fortaleza de Divisas**
- Metodología de puntuación cuantitativa (escala 0-100) basada en nueve indicadores fundamentales
- Clasificación en tiempo real en categorías de sentimiento alcista, bajista o neutral
- Ranking sistemático de divisas por fortaleza fundamental
- Integración de postura de política del banco central y orientación futura

**Inteligencia Macroeconómica**
- Cobertura de ocho divisas principales: USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD
- Seguimiento completo de crecimiento del PIB, inflación, empleo y métricas de balance externo
- Monitoreo de política de bancos centrales incluyendo tasas de interés y calendarios de reuniones
- Tipos de cambio en vivo con intervalos de actualización de 60 segundos

**Análisis Visual**
- Visualización de mapa de calor de indicadores económicos con métricas de fortaleza codificadas por color
- Interfaz intuitiva para evaluación rápida de dinámicas entre divisas
- Tooltips interactivos proporcionando información contextual sobre cada indicador
- Diseño responsivo optimizado para dispositivos de escritorio y tabletas

## Arquitectura Técnica

**Framework Frontend**
- React 18 para arquitectura basada en componentes
- Implementación en JavaScript puro sin dependencias de compilación
- Diseño de aplicación de una sola página para rendimiento óptimo

**Infraestructura de Datos**
- Integración de API RESTful para tipos de cambio
- Motor de procesamiento y cálculo de datos del lado del cliente
- Mecanismos de actualización automática para integridad de datos en tiempo real
- Provisiones de datos de respaldo para escenarios de indisponibilidad de API

**Entorno de Hosting**
- Despliegue de sitio estático compatible con GitHub Pages, Netlify, Vercel
- No se requieren dependencias del lado del servidor
- Soporta cifrado SSL/TLS
- Capacidad de distribución CDN global

## Instrucciones de Despliegue

### Despliegue Estándar

La aplicación está diseñada como un archivo HTML independiente que no requiere compilación o proceso de construcción.

```bash
# Clonar repositorio
git clone https://github.com/GlobalInvesting/forex-dashboard.git

# Navegar al directorio
cd forex-dashboard

# Desplegar en servidor web o abrir localmente
# No se requieren dependencias adicionales
```

### Despliegue en GitHub Pages

1. Navegar a Configuración del repositorio
2. Acceder a la sección Pages
3. Configurar rama fuente: `main`
4. Establecer directorio: `/ (root)`
5. Confirmar despliegue
6. Acceder vía: `https://globalinvesting.github.io/`

### Entorno de Desarrollo Local

```bash
# Servidor HTTP Python
python -m http.server 8000

# Servidor HTTP Node.js
npx http-server -p 8000

# Acceder en http://localhost:8000
```

## Fuentes de Datos y Metodología

### Datos de Mercado

| Categoría | Fuente | Frecuencia de Actualización | Cobertura |
|----------|--------|------------------|----------|
| Tipos FX | ExchangeRate-API | 60 segundos | Pares principales vs USD |
| Indicadores Económicos | Publicaciones de Bancos Centrales | 24 horas | Divisas G10 |
| Tasas de Política | Anuncios de Autoridades Monetarias | Impulsado por eventos | Decisiones de bancos centrales |

### Proveedores de Datos Primarios

- Sistema de Reserva Federal (Estados Unidos)
- Banco Central Europeo (Eurozona)
- Banco de Inglaterra (Reino Unido)
- Banco de Japón (Japón)
- Banco de la Reserva de Australia (Australia)
- Banco de Canadá (Canadá)
- Banco Nacional Suizo (Suiza)
- Banco de la Reserva de Nueva Zelanda (Nueva Zelanda)

### Metodología Cuantitativa

La puntuación de fortaleza fundamental emplea un modelo multifactorial ponderado que incorpora:

**Dinámicas de Crecimiento** (ponderación 20%)
- Tasa de crecimiento del PIB real relativa a promedios históricos
- Tendencias de producción industrial
- Indicadores de impulso económico

**Postura de Política Monetaria** (ponderación 25%)
- Diferencial de tasa de política versus promedio global
- Orientación futura del banco central
- Cálculos de tasa de interés real

**Estabilidad de Precios** (ponderación 15%)
- Desviación del Índice de Precios al Consumidor del objetivo del 2%
- Tendencias de inflación subyacente
- Expectativas de inflación

**Condiciones del Mercado Laboral** (ponderación 15%)
- Tasa de desempleo versus estimaciones de tasa natural
- Tendencias de crecimiento del empleo
- Indicadores de presión salarial

**Balance Externo** (ponderación 25%)
- Saldo de cuenta corriente como porcentaje del PIB
- Balanza comercial en términos absolutos
- Dinámicas de flujo de capital

**Posición Fiscal** (ponderación 10%)
- Ratio deuda pública-PIB
- Tendencias de déficit fiscal
- Calificaciones crediticias

Las puntuaciones se normalizan en una escala de 0-100 con ajustes de ponderación dinámica basados en el régimen de mercado y las condiciones de volatilidad.

## Configuración y Personalización

### Configuración de Actualización de Datos

Modificar intervalos de actualización en el código:

```javascript
// Línea 860: Intervalo de actualización de tipos FX (milisegundos)
const ratesInterval = setInterval(fetchForexRates, 60000);

// Línea 861: Intervalo de actualización de datos económicos (milisegundos)
const fundamentalInterval = setInterval(loadEconomicData, 86400000);
```

### Actualizaciones de Datos Económicos

Actualizar el objeto `baseEconomicData` con las últimas publicaciones estadísticas:

```javascript
const baseEconomicData = {
    USD: { 
        gdp: 29.18,          // Trillones USD
        gdpGrowth: 1.8,      // % Anual
        interestRate: 4.25,  // Tasa de política %
        inflation: 3.4,      // IPC Anual %
        unemployment: 4.2,   // % de fuerza laboral
        currentAccount: -3.9, // % del PIB
        debt: 126.5,         // % del PIB
        tradeBalance: -68.2, // Miles de Millones USD
        production: 1.6      // Producción industrial Anual %
    },
    // Divisas adicionales...
};
```

### Personalización de Indicadores

Modificar indicadores mostrados actualizando:
1. Definiciones de tooltip en el objeto `indicatorTooltips`
2. Estructura de encabezado de tabla en componente de mapa de calor
3. Lógica de renderizado de celdas de datos en cuerpo de tabla

## Matriz de Compatibilidad de Navegadores

| Navegador | Versión Mínima | Estado | Notas |
|---------|----------------|--------|-------|
| Google Chrome | 90+ | Totalmente Soportado | Recomendado para rendimiento óptimo |
| Mozilla Firefox | 88+ | Totalmente Soportado | Todas las características funcionales |
| Safari | 14+ | Soportado | Posibles diferencias visuales menores |
| Microsoft Edge | 90+ | Totalmente Soportado | Versiones basadas en Chromium |
| Mobile Safari | 14+ | Limitado | Optimizado para tableta, soporte de teléfono mínimo |
| Chrome Mobile | 90+ | Limitado | Optimizado para tableta, soporte de teléfono mínimo |

## Divulgación de Riesgos y Descargos de Responsabilidad

**AVISO LEGAL IMPORTANTE**

Este software se proporciona solo con fines informativos y educativos y NO constituye:
- Asesoramiento financiero o recomendaciones de inversión
- Señales de trading u orientación sobre timing de mercado
- Servicios de asesoría de inversión profesional
- Una oferta o solicitud para comprar o vender instrumentos financieros

**Advertencias de Riesgo**

El trading de divisas conlleva un alto nivel de riesgo y puede no ser adecuado para todos los inversores. Los riesgos clave incluyen:

- **Riesgo de Mercado**: Los valores de las divisas pueden fluctuar significativamente debido a condiciones económicas, políticas y de mercado
- **Riesgo de Apalancamiento**: El trading con margen puede resultar en pérdidas que excedan la inversión inicial
- **Riesgo de Liquidez**: Las condiciones del mercado pueden impedir la ejecución a precios deseados
- **Riesgo Operacional**: Las fallas tecnológicas pueden afectar la precisión o disponibilidad de datos
- **Riesgo País**: La inestabilidad política y económica puede afectar los valores de las divisas

**Responsabilidades del Usuario**

Los usuarios de esta plataforma son únicamente responsables de:
- Realizar investigación independiente y debida diligencia
- Consultar con asesores financieros licenciados antes de tomar decisiones de inversión
- Entender que el rendimiento pasado no garantiza resultados futuros
- Asegurar el cumplimiento con leyes y regulaciones aplicables en su jurisdicción
- Verificar la precisión de los datos de fuentes primarias

**Sin Garantía**

Este software se proporciona "tal cual" sin garantía de ningún tipo, expresa o implícita. Los desarrolladores no hacen representaciones con respecto a la precisión, integridad o puntualidad de la información proporcionada.

## Mejoras Planificadas

**Fase 1: Expansión de Datos**
- Integración de divisas adicionales G10 y de mercados emergentes
- Datos de series temporales históricas con análisis de tendencias
- Integración de calendario económico con evaluación de impacto de eventos
- Análisis mejorado de comunicaciones de bancos centrales

**Fase 2: Mejora de Análisis**
- Capacidades avanzadas de gráficos con superposiciones técnicas
- Matrices de correlación entre pares de divisas
- Métricas de volatilidad y análisis implícita vs realizada
- Indicadores de sentimiento de múltiples fuentes de datos

**Fase 3: Experiencia de Usuario**
- Diseños de dashboard personalizables y vistas guardadas
- Sistema de alertas para publicaciones económicas significativas
- Funcionalidad de exportación (CSV, PDF, Excel)
- Acceso API para recuperación programática de datos

**Fase 4: Móvil y Accesibilidad**
- Optimización de diseño móvil responsivo
- Implementación de Aplicación Web Progresiva (PWA)
- Cumplimiento WCAG 2.1 AA para accesibilidad
- Soporte de internacionalización multiidioma

## Derechos de Autor

© 2026 Santiago Plá Casuriaga · Global Investing. Todos los derechos reservados.

Este software es de uso exclusivo de su autor. No se permite su reproducción, distribución ni modificación sin autorización expresa por escrito.

## Contacto y Soporte

**Soporte Técnico**
- Seguimiento de Problemas: https://github.com/GlobalInvesting/forex-dashboard/issues
- Foro de Discusión: https://github.com/GlobalInvesting/forex-dashboard/discussions

**Consultas Profesionales**
- GitHub: [@GlobalInvesting](https://github.com/GlobalInvesting/)
- Email: globalinvestingmarkets@gmail.com
- LinkedIn: https://www.linkedin.com/in/santiago-pla-casuriaga/

**Preocupaciones de Seguridad**
Para problemas relacionados con la seguridad, por favor contacte directamente a globalinvestingmarkets@gmail.com en lugar de abrir problemas públicos.

## Agradecimientos

Este proyecto integra datos y metodologías de varias fuentes:

- Datos de tipos de cambio proporcionados por ExchangeRate-API
- Estadísticas económicas procedentes de publicaciones de bancos centrales y agencias estadísticas
- Marco analítico informado por metodologías de investigación institucional
- Construido sobre tecnologías de código abierto incluyendo React y estándares web modernos

## Historial de Versiones

**Versión 5.6.0** (Marzo 2026)
- FX momentum 1M como indicador dominante (22%)
- Corrección safe haven CHF: deflación ya no suma puntos de prima
- Horizonte actualizado a 2-6 semanas
- Protección de rama main y control de acceso

**Versión 1.0.0** (Febrero 2026)
- Lanzamiento inicial de producción
- Marco de análisis fundamental central
- Integración de datos en tiempo real
- Visualización de mapa de calor
- Cobertura de ocho divisas principales

---

**© 2026 Santiago Plá Casuriaga · Global Investing**
