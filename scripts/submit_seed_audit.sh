#!/bin/bash
# =============================================================================
# submit_seed_audit.sh — Multi-seed audit on 3 representative animals
# =============================================================================
# Addresses Reviewer 1 Major Concern 3: the headline ±0.0009 std bands across
# 18 animals × 5 folds were measured at one initialisation seed per training
# run; training-stochastic variance was conflated with cross-animal
# heterogeneity. Re-runs animals 1, 9, 17 (low / mid / high RMSE percentile)
# with 5 seeds, full 5-fold CV, full headline NN_MODELS registry.
#
# Outputs:
#   Results/python_pipeline_seedaudit/animal-NN/
#       seed_audit_long.csv     (one row per seed × fold × model)
#       seed_audit_summary.csv  (per-model mean ± seed-std across folds)
# =============================================================================

#SBATCH -J STALSTMH_seed_audit
#SBATCH --array=1,9,17
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -p gpucomputeq
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00

#SBATCH -o logs/seed-audit-%a.out
#SBATCH -e logs/seed-audit-%a.err

#SBATCH --mail-type=END,FAIL

cd "$SLURM_SUBMIT_DIR"

module purge
module load Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu

VENV_DIR="$SLURM_SUBMIT_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

ANIMAL_ID=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
echo "==========================================="
echo " SEED-AUDIT  animal=${ANIMAL_ID}  node=${SLURMD_NODENAME}  start=$(date)"
echo "==========================================="

mkdir -p "Results/python_pipeline_seedaudit/animal-${ANIMAL_ID}" logs

python -u scripts/run_seed_audit.py \
    --animal-id "$SLURM_ARRAY_TASK_ID" \
    --seeds     7 42 101 1729 2026 \
    --epochs    40 \
    --folds     5 \
    --output-dir "Results/python_pipeline_seedaudit/animal-${ANIMAL_ID}"

EXIT=$?
echo "Finished SEED-AUDIT animal ${ANIMAL_ID} at $(date) — exit ${EXIT}"
exit $EXIT
