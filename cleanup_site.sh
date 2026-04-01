#!/usr/bin/env bash
# cleanup_site.sh — Limpieza de código muerto en globalinvesting.github.io
# Generado por auditoría 2026-03-31
#
# Ejecutar desde la raíz del repo site:
#   chmod +x cleanup_site.sh
#   ./cleanup_site.sh
#
# El script es idempotente: si ya se aplicaron cambios, no rompe nada.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   CLEANUP — globalinvesting.github.io (auditoría 2026-03)   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Renombrar gitignore → .gitignore ─────────────────────────────────────
# El archivo existía sin punto. Git nunca lo leyó — todos los patrones
# (.env, __pycache__, venv/, etc.) estaban ignorados por el propio ignorado.

if [ -f "gitignore" ] && [ ! -f ".gitignore" ]; then
  git mv gitignore .gitignore
  echo "✅ Renombrado: gitignore → .gitignore"
elif [ -f ".gitignore" ]; then
  echo "⏭  Ya existe: .gitignore"
else
  echo "⚠️  No se encontró 'gitignore' para renombrar"
fi

# ── 2. Eliminar news-data/summaries.json ─────────────────────────────────────
# generate_summaries.py fue eliminado en v7.0.0 (ver CHANGELOG).
# El frontend nunca consumió este archivo — el CHANGELOG lo confirma explícitamente.
# El workflow forex-news.yml ya no tiene ningún step que lo genere.

if [ -f "news-data/summaries.json" ]; then
  git rm "news-data/summaries.json"
  echo "✅ Eliminado: news-data/summaries.json"
else
  echo "⏭  Ya eliminado: news-data/summaries.json"
fi

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "Resumen:"
echo "  - 1 archivo renombrado (.gitignore ahora activo)"
echo "  - 1 archivo eliminado  (summaries.json)"
echo ""
echo "Los archivos _headers y netlify.toml también fueron modificados"
echo "para eliminar rutas obsoletas (/strength-scores/* y /fx-history/*)."
echo "Esos cambios vienen ya aplicados en los archivos del repo."
echo ""
echo "Próximo paso:"
echo "  git add -A"
echo "  git commit -m \"chore: audit 2026-03 — remove dead files, fix .gitignore\""
echo "  git push"
echo ""
