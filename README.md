# Global Investing FX Terminal

A professional FX terminal built as a static GitHub Pages site, powered by automated GitHub Actions data pipelines.

---

## Architecture

```
globalinvesting-engine (PRIVATE)     globalinvesting.github.io (PUBLIC)
├── scripts/                         ├── index.html                ← terminal UI
│   ├── fetch_rates.py               ├── ai-analysis/
│   ├── fetch_news.py                │   ├── index.json            ← narrative + regime
│   ├── generate_narrative_signals.py│   └── signals.json          ← market alerts
│   ├── update_cot_cftc.py           ├── news-data/
│   ├── fetch_econ_data_apis.py      │   └── news.json             ← RSS headlines
│   └── fx_config.py                 ├── rates/
├── .github/workflows/               │   └── {CCY}.json            ← CB rates
│   ├── generate-ai-narrative.yml    ├── cot-data/
│   ├── forex-news.yml               │   └── {CCY}.json            ← CFTC positioning
│   ├── update-rates.yml             └── economic-data-history/
│   └── update-cot-cftc-all.yml          └── {CCY}/{indicator}.json
└── fx_config.py
```

The engine (private) repo writes JSON files to the public site repo via PAT_TOKEN. The frontend reads these files with `fetch()` calls. No server required.

---

## What stays private (engine repo)

| File | Reason |
|------|--------|
| `generate_narrative_signals.py` | Contains the AI prompts that define the terminal's analysis quality |
| `generate_ai_analysis.py` | Same — per-currency AI prompt methodology |
| `generate_summaries.py` | Groq prompt engineering |
| All `.github/workflows/*.yml` | Orchestration logic and secret references |
| `fx_config.py` | Not sensitive, but included for consistency |
| `calculate_scores.py` | Legacy scoring model — no longer used in terminal, keep private anyway |

**Everything else** (fetch scripts for public APIs, the HTML frontend) can be public without meaningful competitive risk.

---

## What goes public (site repo)

- `index.html` — the entire terminal UI
- All generated JSON data files (`rates/`, `cot-data/`, `news-data/`, `ai-analysis/`, `economic-data-history/`)
- GitHub Pages serves the static site for free

---

## AI usage — intentionally limited

AI (Groq / llama-3.3-70b-versatile) is used **only** for two outputs:

1. **Narrative bar** — 2-3 sentence market summary + regime label (RISK-ON / RISK-OFF / MIXED / NEUTRAL)
2. **Alerts & Market Signals** — 4-6 prioritized actionable signals

All other sections (News Feed, ticker, COT, rates, heatmap, charts) consume real data directly without AI filtering or summarization.

Workflow: `generate-ai-narrative.yml` runs 3×/day (06:30, 12:30, 21:30 UTC).

---

## Data sources

| Widget | Source | Update freq | Cost |
|--------|--------|-------------|------|
| Advanced chart | TradingView embed | real-time | Free |
| Ticker tape | TradingView embed | real-time | Free |
| Quote bar | Frankfurter (ECB) | daily ~16:00 CET | Free |
| Forex heatmap | TradingView embed | real-time | Free |
| Economic calendar | TradingView embed | real-time | Free |
| Interest rates | CB direct APIs (RBA, BoC, ECB…) | daily GH Action | Free |
| COT positioning | CFTC.gov direct | weekly (Friday) | Free |
| FX sentiment | Dukascopy public JSON | 30 min refresh | Free |
| News headlines | RSS feeds (Reuters, FT, ForexLive…) | 3×/day GH Action | Free |
| AI narrative + signals | Groq (llama-3.3-70b) | 3×/day GH Action | Free tier |
| Economic data history | FRED + OECD + World Bank | twice/month GH Action | Free |

**Total monthly cost at current scale: $0**

---

## Required secrets (engine repo settings)

| Secret | Used by |
|--------|---------|
| `PAT_TOKEN` | All workflows — write access to public site repo |
| `GROQ_API_KEY` | `generate-ai-narrative.yml`, `forex-news.yml` |
| `FRED_API_KEY` | `fetch-econ-data-apis.yml` (free key from FRED) |
| `NEWSDATA_API_KEY` | `forex-news.yml` (optional, free tier) |

---

## Workflow schedule

| Workflow | Schedule (UTC) | Output |
|----------|----------------|--------|
| `update-rates.yml` | 08:00 daily | `rates/{CCY}.json` |
| `forex-news.yml` | 06:00, 12:00, 21:00 | `news-data/news.json` |
| `generate-ai-narrative.yml` | 06:30, 12:30, 21:30 | `ai-analysis/index.json`, `signals.json` |
| `update-cot-cftc-all.yml` | Saturday 04:00 | `cot-data/{CCY}.json` |
| `fetch-econ-data-apis.yml` | 1st & 15th of month | `economic-data-history/**` |

---

## Setup

1. Fork or clone both repos under the `GlobalInvesting` GitHub organization
2. Enable GitHub Pages on the public repo (branch: `main`, folder: `/`)
3. Add all required secrets to the **engine** (private) repo
4. Run each workflow once manually (`workflow_dispatch`) to populate initial data
5. Deploy `index.html` to the public repo root

The terminal is live at `https://globalinvesting.github.io/` immediately after step 5.
