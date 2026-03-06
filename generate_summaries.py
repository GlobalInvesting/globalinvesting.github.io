name: Generate AI Analysis
on:
  workflow_run:
    workflows: ["Update Economic Data"]
    types: [completed]
  schedule:
    - cron: '30 10 * * 1-5'
  workflow_dispatch:

# FIX R-01: Concurrency group para evitar ejecuciones simultáneas.
concurrency:
  group: generate-ai-analysis
  cancel-in-progress: false

jobs:
  generate:
    runs-on: ubuntu-latest
    if: ${{ github.event_name != 'workflow_run' || github.event.workflow_run.conclusion == 'success' }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        # FIX B-02: Versiones fijadas via requirements.txt
        run: pip install -r scripts/requirements.txt
      - name: Generate AI analysis
        env:
          GROQ_API_KEY:   ${{ secrets.GROQ_API_KEY }}
          GROQ_API_KEY_2: ${{ secrets.GROQ_API_KEY_2 }}
          GROQ_API_KEY_3: ${{ secrets.GROQ_API_KEY_3 }}
          FRED_API_KEY:   ${{ secrets.FRED_API_KEY }}
        run: python scripts/generate_ai_analysis.py
      - name: Commit and push
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add ai-analysis/
          if git diff --cached --quiet; then
            echo "No changes to commit"
          else
            git commit -m "chore: update AI analysis [$(date -u +'%Y-%m-%d %H:%M UTC')]"
            git push
          fi
