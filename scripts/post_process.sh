#!/bin/bash
# =============================================================================
# post_process.sh — consolidate per-animal results and render figures
# =============================================================================
# Run from $PROJECT_DIR after all 18 animal Slurm tasks complete.
#
# Outputs:
#   Results/comparison_report_all_animals.csv
#   manuscript/figures-new/*.eps,*.png
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "═══ Consolidating 18-animal results ═══"
python -u scripts/consolidate_results.py \
    --results-dir Results/python_pipeline \
    --out         Results/comparison_report_all_animals.csv

echo
echo "═══ Rendering figures ═══"
python -u scripts/make_figures.py \
    --results-dir Results/python_pipeline \
    --out-dir     manuscript/figures-new

echo
echo "═══ Done. Inspect: ═══"
echo "  Results/comparison_report_all_animals.csv"
echo "  manuscript/figures-new/*.eps  (and matching .png previews)"
