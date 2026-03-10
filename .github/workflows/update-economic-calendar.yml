name: Update Economic Calendar v11 (Trading Economics)

on:
  schedule:
    # Asia session (JPY, AUD, NZD) — 00:00–05:30 UTC
    - cron: '5 0 * * *'
    - cron: '35 1 * * *'
    - cron: '5 3 * * *'
    - cron: '35 4 * * *'
    # Europe open (EUR, GBP, CHF) — 06:00–11:00 UTC
    - cron: '5 6 * * *'
    - cron: '35 7 * * *'
    - cron: '5 9 * * *'
    - cron: '35 10 * * *'
    # US session (USD, CAD) — 12:00–18:30 UTC
    - cron: '5 12 * * *'
    - cron: '35 12 * * *'
    - cron: '5 13 * * *'
    - cron: '35 13 * * *'
    - cron: '5 14 * * *'
    - cron: '5 15 * * *'
    - cron: '5 16 * * *'
    - cron: '5 17 * * *'
    - cron: '35 17 * * *'
    - cron: '5 18 * * *'
    # Late / API updates
    - cron: '5 20 * * *'
    - cron: '5 22 * * *'
    - cron: '35 22 * * *'
  workflow_dispatch:

jobs:
  update-calendar:
    runs-on: ubuntu-latest
    timeout-minutes: 20

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install requests beautifulsoup4 lxml playwright
          # Instalar Chromium solo si lo necesitamos (fallback)
          # Se instala siempre para tener el fallback listo
          playwright install chromium
          sudo apt-get update -qq
          sudo apt-get install -y -qq \
            libnss3 libatk-bridge2.0-0 libdrm2 libxcomposite1 \
            libxdamage1 libxrandr2 libgbm1 libxshmfence1 \
            libasound2t64 2>/dev/null || \
          sudo apt-get install -y -qq \
            libnss3 libatk-bridge2.0-0 libdrm2 libxcomposite1 \
            libxdamage1 libxrandr2 libgbm1 libxshmfence1 || true

      - name: Create output directory
        run: mkdir -p calendar-data

      - name: Run calendar scraper
        run: python3 -u scripts/update_economic_calendar.py

      - name: Commit and push if changed
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add calendar-data/calendar.json
          if git diff --staged --quiet; then
            echo "✅ No changes to commit"
          else
            git commit -m "📅 Calendar $(date -u +'%Y-%m-%d %H:%M') UTC — Trading Economics v11"
            git pull --rebase origin main || true
            git push
          fi
