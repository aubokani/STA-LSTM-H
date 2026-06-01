#!/bin/bash
# =============================================================================
# fetch_results.sh — Retrieve per-animal results from Ada to local machine
# =============================================================================
# Run this from your LOCAL Mac terminal after all 18 Slurm tasks complete.
#
# Usage:
#   bash scripts/fetch_results.sh
#   bash scripts/fetch_results.sh --dry-run     # preview without copying
#
# Prerequisites:
#   • Ada account and VPN active (if off-campus)
#   • 18 Slurm array tasks have finished (check with: squeue -u <username>)
# =============================================================================

set -euo pipefail

# ── Configuration — edit these three variables ────────────────────────────────
REMOTE_USER="${HPC_USER:-$(whoami)}"                      # your CQU username
REMOTE_HOST="ada.cqu.edu.au"
REMOTE_PROJECT_DIR="~/projects/2026-STA-LSTM-H"          # must match rsync deploy path

LOCAL_RESULTS_DIR="Results/python_pipeline"               # local destination

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN] No files will be transferred"
fi

RSYNC_FLAGS="-avz --progress --compress-level=9"
[[ "$DRY_RUN" == "true" ]] && RSYNC_FLAGS="$RSYNC_FLAGS --dry-run"

echo "===================================================================="
echo "  Fetching results from Ada Lovelace HPC"
echo "  Remote : ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}"
echo "  Local  : $(pwd)/${LOCAL_RESULTS_DIR}"
echo "===================================================================="

# ── Fetch per-animal result directories ──────────────────────────────────────
mkdir -p "$LOCAL_RESULTS_DIR"

rsync $RSYNC_FLAGS \
    --include="animal-*/" \
    --include="animal-*/comparison_report.csv" \
    --include="animal-*/wilcoxon_stats.csv" \
    --include="animal-*/run_summary.json" \
    --include="animal-*/autoregressive_errors.csv" \
    --include="animal-*/last_fold_predictions.npy" \
    --exclude="*" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/Results/python_pipeline/" \
    "$LOCAL_RESULTS_DIR/"

# ── Fetch top-level aggregate report (written by consolidate_results.py) ─────
rsync $RSYNC_FLAGS \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/Results/comparison_report.csv" \
    "Results/comparison_report_hpc.csv" 2>/dev/null || \
    echo "  [SKIP] No aggregate comparison_report.csv yet — run consolidate_results.py first"

# ── Fetch Slurm logs ──────────────────────────────────────────────────────────
mkdir -p logs
rsync $RSYNC_FLAGS \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/logs/" \
    "logs/" 2>/dev/null || echo "  [SKIP] No logs/ directory found"

echo ""
echo "  Transfer complete."
echo ""
echo "  Next: run python scripts/consolidate_results.py"
echo "        to merge all 18 animal reports into one summary table."
echo "===================================================================="
