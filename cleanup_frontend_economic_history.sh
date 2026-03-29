#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Global Investing — Frontend: Remove economic-data-history/
#  Run from: root of globalinvesting.github.io repo
#  Usage:    bash cleanup_frontend_economic_history.sh
#            bash cleanup_frontend_economic_history.sh --execute
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
EXECUTE=false
[[ "${1:-}" == "--execute" ]] && EXECUTE=true

echo "======================================================="
echo " Frontend — Remove economic-data-history/"
$EXECUTE && echo " MODE: EXECUTE" || echo " MODE: DRY RUN"
echo "======================================================="

if [ -d "economic-data-history" ]; then
  COUNT=$(find economic-data-history -type f | wc -l)
  if $EXECUTE; then
    git rm -rf economic-data-history/
    echo "Removed economic-data-history/ ($COUNT files)"
    echo ""
    echo "Run:"
    echo "  git commit -m \"chore: remove economic-data-history (legacy scoring system)\""
    echo "  git push"
  else
    echo "[DRY RUN] Would remove: economic-data-history/ ($COUNT files)"
    echo ""
    echo "Zero references to this directory in index.html or any page."
    echo "Was written by fetch-econ-data-apis.yml (deleted from engine)."
    echo ""
    echo "Run with --execute when ready."
  fi
else
  echo "economic-data-history/ not found — already removed or never existed."
fi
