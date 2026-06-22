#!/usr/bin/env bash
# Build a self-contained linger-bench from the main project.
# Run from the MAIN PROJECT ROOT (the dir that contains the_new_world/ and scripts/).
# The main project is READ-ONLY here: nothing under the_new_world/ is modified.
set -euo pipefail

ROOT="$(pwd)"
LB="$ROOT/linger-bench"
SRC="$ROOT/scripts/paper_experiments"

[ -d "$ROOT/the_new_world" ] || { echo "ERROR: run me from the main project root (no the_new_world/ here)"; exit 1; }

echo "[1/4] vendoring backend: the_new_world -> linger-bench/linger_core (copy only)"
rm -rf "$LB/linger_core"
cp -r "$ROOT/the_new_world" "$LB/linger_core"
find "$LB/linger_core" -depth -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
# rewrite import prefix; UPPER-CASE env var THE_NEW_WORLD_DATA_DIR is unaffected (case-sensitive)
find "$LB/linger_core" -name '*.py' -exec sed -i 's/the_new_world/linger_core/g' {} +

echo "[2/4] copy experiment scripts + data"
mkdir -p "$LB/paper_experiments" "$LB/figures"
cp "$SRC"/e5_common.py "$SRC"/e5_probes.py "$SRC"/e5_rag.py "$SRC"/run_e5_quant.py \
   "$SRC"/stats_e5_quant.py "$SRC"/e5_supplement_common.py \
   "$SRC"/e6_personas.py "$SRC"/run_e6_persona.py "$SRC"/stats_e6_persona.py \
   "$SRC"/make_figs.py "$LB/paper_experiments/"
cp "$SRC"/results_e5_quant_A1.json "$SRC"/results_e5_quant_A2.json \
   "$SRC"/results_e5_quant_B0.json "$SRC"/results_e5_quant_Bplus.json \
   "$SRC"/results_e5_quant_stats.json \
   "$SRC"/results_e6_persona_A1.json "$SRC"/results_e6_persona_A2.json \
   "$SRC"/results_e6_persona_B0.json "$SRC"/results_e6_persona_Bplus.json \
   "$SRC"/results_e6_persona_stats.json "$LB/paper_experiments/"

echo "[3/4] adjust paths in the copied scripts"
find "$LB/paper_experiments" -name '*.py' -exec sed -i 's/the_new_world/linger_core/g' {} +
sed -i 's/parents\[2\]/parents[1]/g' "$LB/paper_experiments/e5_common.py"   # .env now at linger-bench/
sed -i 's#\.\./\.\./paper/fig_e5honesty#../figures/memory_honesty#g' "$LB/paper_experiments/make_figs.py"

echo "[4/4] backend dependency list"
cp "$ROOT/requirements.txt" "$LB/requirements-backend.txt" 2>/dev/null || true

echo
echo "DONE. linger-bench is now self-contained:"
echo "  linger_core/        backend ($(find "$LB/linger_core" -name '*.py'|wc -l) py files)"
echo "  paper_experiments/  scripts + data ($(find "$LB/paper_experiments" -type f|wc -l) files)"
echo
echo "VERIFY (offline figure, no API key):"
echo "  uv run python linger-bench/paper_experiments/make_figs.py && ls -l linger-bench/figures/"
