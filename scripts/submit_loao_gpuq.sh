#!/bin/bash
# =============================================================================
# submit_loao.sh — Leave-one-animal-out (LOAO) Slurm array
# =============================================================================
# Reviewer 3 flagged the per-animal 5-fold StratifiedKFold protocol as
# inflating accuracy through within-animal temporal autocorrelation. This
# script implements the LOAO external-validation standard: 18 array tasks,
# each holding out one animal and training on the other 17.
#
# Per task: load all 18 animals (~8 GB), train neural + XGBoost models on
# 17, evaluate on the held-out animal. Wall time ~3-4× the per-animal job
# because each task trains on 17 × the rows, so we set a longer self-cap.
#
# Usage (from $PROJECT_DIR on the cluster login node):
#   sbatch scripts/submit_loao.sh
#
# After all 18 tasks finish:
#   python scripts/consolidate_loao.py
# =============================================================================

#SBATCH -J STALSTMH_loao_gpuq             # Job name (gpuq spillover)
#SBATCH --array=17-18                     # Spillover tasks 17 and 18
#SBATCH -c 8                              # 8 CPU cores per task
#SBATCH --mem=64G                         # 64 GB RAM (full cohort in memory)
#SBATCH -p gpuq                           # GPU partition (separate QoS, +1 GPU)
#SBATCH --gres=gpu:1                      # 1 GPU per task
#SBATCH -t 12:00:00                       # 12-hour self-imposed cap

#SBATCH -o logs/loao-gpuq-%a.out
#SBATCH -e logs/loao-gpuq-%a.err

#SBATCH --mail-type=BEGIN,END,FAIL
# (intentionally no --mail-user; supplied by user's Slurm config)

cd "$SLURM_SUBMIT_DIR"

module purge
module load Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu

VENV_DIR="$SLURM_SUBMIT_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Run: bash scripts/setup_env.sh"
    exit 1
fi

HELD_OUT_ID=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
echo "============================================================"
echo " LOAO held-out animal: ${HELD_OUT_ID}  —  Task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}"
echo " Node: ${SLURMD_NODENAME}  GPU: ${CUDA_VISIBLE_DEVICES}"
echo " Start: $(date)"
echo "============================================================"

mkdir -p "Results/python_pipeline_loao/animal-${HELD_OUT_ID}"
mkdir -p logs

DATA_ROOT="/home/bokania/projects/2026-STA-LSTM-H/data/raw"

python -u scripts/run_loao_animal.py \
    --held-out-animal-id "$SLURM_ARRAY_TASK_ID"                       \
    --data-root          "$DATA_ROOT"                                  \
    --epochs             40                                            \
    --max-windows        86400                                         \
    --output-dir         "Results/python_pipeline_loao/animal-${HELD_OUT_ID}"

EXIT_CODE=$?
echo "Finished LOAO held-out animal ${HELD_OUT_ID} at $(date) — exit code: ${EXIT_CODE}"
exit $EXIT_CODE
