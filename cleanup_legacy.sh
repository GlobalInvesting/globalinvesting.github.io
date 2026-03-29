#!/usr/bin/env bash
# ============================================================
# Global Investing FX Terminal — Legacy Cleanup Script
# Run from the ROOT of the globalinvesting.github.io repo
# ============================================================
# DRY RUN by default. To actually delete, run:
#   bash cleanup_legacy.sh --execute
# ============================================================

set -euo pipefail

DRY_RUN=true
if [[ "${1:-}" == "--execute" ]]; then
  DRY_RUN=false
fi

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; NC='\033[0m'

rm_item() {
  local target="$1"
  if [[ -e "$target" || -d "$target" ]]; then
    if $DRY_RUN; then
      echo -e "${YELLOW}[DRY RUN] Would remove: $target${NC}"
    else
      rm -rf "$target"
      echo -e "${RED}Removed: $target${NC}"
    fi
  else
    echo -e "${NC}  (not found, skipping): $target"
  fi
}

echo ""
echo "======================================================="
echo " Global Investing FX Terminal — Legacy File Cleanup"
if $DRY_RUN; then
  echo " MODE: DRY RUN (no files will be deleted)"
  echo " Run with --execute to actually delete files"
fi
echo "======================================================="
echo ""

# ─── LEGACY HTML PAGES ────────────────────────────────────────
echo "── Legacy HTML pages ──"
rm_item "carry-trade.html"
rm_item "news.html"
rm_item "en.html"
rm_item "data.html"
rm_item "tecnico-vs-fundamental.html"
rm_item "publicidad.html"
rm_item "advertise.html"

# ─── LEGACY GUIDE PAGES (Spanish, old scoring system) ─────────
echo ""
echo "── Legacy guide pages (Spanish / old scoring system) ──"
rm_item "guia-carry-trade.html"
rm_item "guia-cot.html"
rm_item "guia-bancos-centrales.html"
rm_item "guia-calendario-economico.html"
rm_item "guia-score-fortaleza.html"
rm_item "guia-pips.html"
rm_item "glosario-forex.html"

# ─── LEGACY DATA DIRECTORIES ──────────────────────────────────
echo ""
echo "── Legacy data directories ──"
rm_item "backtest-results/"
# extended-data/ IS used by terminal (IV, carry, cross-asset) — DO NOT DELETE
# rm_item "extended-data/"
rm_item "fx-history/"
rm_item "scores-history/"
rm_item "strength-scores/"

# ─── LEGACY AI ANALYSIS (per-currency JSON from old scoring) ──
echo ""
echo "── Legacy ai-analysis per-currency files ──"
for f in ai-analysis/AUD.json ai-analysis/CAD.json ai-analysis/CHF.json \
          ai-analysis/EUR.json ai-analysis/GBP.json ai-analysis/JPY.json \
          ai-analysis/NZD.json ai-analysis/USD.json; do
  rm_item "$f"
done
# Keep ai-analysis/signals.json — the terminal reads this for AI signals
echo -e "${GREEN}  (keeping) ai-analysis/signals.json${NC}"

# ─── LEGACY RSS FEEDS ─────────────────────────────────────────
echo ""
echo "── Legacy RSS feeds ──"
rm_item "feed-analysis.xml"
rm_item "feed-scores.xml"
rm_item "feed.xml"

# ─── LEGACY SCRIPTS FOLDER ────────────────────────────────────
echo ""
echo "── Legacy scripts in /scripts ──"
# Only remove if the folder contains old scoring scripts
# Check before blindly deleting
if [[ -d "scripts/" ]]; then
  echo -e "${YELLOW}  WARNING: /scripts/ directory found. Review manually before deleting.${NC}"
  echo "  Contents:"
  ls scripts/ 2>/dev/null || echo "  (empty)"
fi

# ─── SUMMARY ──────────────────────────────────────────────────
echo ""
echo "======================================================="
if $DRY_RUN; then
  echo -e "${YELLOW} DRY RUN complete. No files were deleted.${NC}"
  echo " Review the list above, then run:"
  echo "   bash cleanup_legacy.sh --execute"
else
  echo -e "${GREEN} Cleanup complete.${NC}"
  echo " Next steps:"
  echo "  1. git add -A"
  echo "  2. git commit -m 'chore: remove legacy scoring system files and pages'"
  echo "  3. git push"
fi
echo "======================================================="
echo ""

# ─── FILES TO KEEP (reference) ────────────────────────────────
cat <<'EOF'

Files intentionally kept:
  index.html          — FX Terminal v4.x (main dashboard)
  about.html          — Rebuilt in English
  contact.html        — Contact page
  privacy.html        — Privacy policy
  terms.html          — Terms of use
  sitemap.xml         — Updated sitemap
  robots.txt          — Search engine directives
  ads.txt             — AdSense publisher file
  og-image.png        — Social preview image
  favicon*.png/ico    — Favicons
  apple-touch-icon.png
  _headers / netlify.toml / gitignore
  ai-analysis/signals.json   — AI signals (terminal reads this)
  calendar-data/             — Economic calendar (terminal reads this)
  cot-data/                  — COT data (terminal reads this)
  economic-data/             — Macro data (terminal reads this)
  news-data/                 — News feed (terminal reads this)
  rates/                     — FX rates cache (terminal reads this)
  meetings-data/             — CB meetings (terminal reads this)
  guide-dashboard.html       — New English guide
  guide-cross-asset-risk.html
  guide-rates-yield-curve.html
  guide-market-sentiment.html
  guide-fx-liquidity.html
  guide-cot.html

  economic-data-history/     — VERIFY if macro panel in terminal uses this
                               before deleting (check fetch calls in index.html)
EOF
