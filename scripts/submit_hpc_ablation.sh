#!/bin/bash
# =============================================================================
# submit_hpc_ablation.sh — Ada Lovelace (CQU HPC) Array Job — ABLATION RUN
# =============================================================================
# Submits one Slurm task per animal (IDs 01-18) running the (architecture x
# input-set) factorial ablation that the headline run does not cover:
#
#   STA-LSTM-H (accel)        STA-LSTM-H + bp_lag1
#   STA-LSTM   (accel)        STA-LSTM   + bp_lag1
#   LSTM       + bp           LSTM       + bp_lag1
#   GRU        + bp           GRU        + bp_lag1
#   Transformer+ bp           Transformer+ bp_lag1
#
# 10 NN cells per animal, no classical filters (input-set independent —
# already on disk in Results/python_pipeline/). Outputs land in
#   Results/python_pipeline_ablation/animal-NN/
# so the headline tree is untouched.
#
# Usage (from $PROJECT_DIR on the Ada login node):
#   sbatch scripts/submit_hpc_ablation.sh
# =============================================================================

#SBATCH -J STALSTMH_ablation               # Job name
#SBATCH --array=1-18                       # One task per animal (01-18)
#SBATCH -c 8                               # 8 CPU cores per task
#SBATCH --mem=32G                          # 32 GB RAM per task
#SBATCH -p gpucomputeq                     # H100 partition (2 nodes x 2 GPUs)
#SBATCH --gres=gpu:1                       # 1 GPU per task
#SBATCH -t 06:00:00                        # 6-hour cap (10 NN cells vs 5)

# ── Output / error files ────────────────────────────────────────────────────
#SBATCH -o logs/ablation-%a.out
#SBATCH -e logs/ablation-%a.err

# ── Email notifications ──────────────────────────────────────────────────────
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bokania@cqu.edu.au

cd "$SLURM_SUBMIT_DIR"

# ── Load Ada deep-learning Python module ────────────────────────────────────
module purge
module load Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu

# ── Activate project virtual environment ────────────────────────────────────
VENV_DIR="$SLURM_SUBMIT_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Run: bash scripts/setup_env.sh"
    exit 1
fi

# ── Map array index → zero-padded animal ID ────────────────────────────────
ANIMAL_ID=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
echo "============================================================"
echo " ABLATION  Animal ${ANIMAL_ID} — Task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}"
echo " Node: ${SLURMD_NODENAME}  GPU: ${CUDA_VISIBLE_DEVICES}"
echo " Start: $(date)"
echo "============================================================"

mkdir -p "Results/python_pipeline_ablation/animal-${ANIMAL_ID}"
mkdir -p logs

# Match the headline run's training budget exactly (40 epochs, 86 400 windows,
# 5 folds) so the only thing that changes between headline and ablation is the
# input feature set, not the optimisation regime.
DATA_ROOT="/home/bokania/projects/2026-STA-LSTM-H/data/raw"

python -u src/run_animal.py \
    --animal-id   "$SLURM_ARRAY_TASK_ID"                                          \
    --data-root   "$DATA_ROOT"                                                     \
    --folds       5                                                                \
    --epochs      40                                                               \
    --max-windows 86400                                                            \
    --max-horizon 0                                                                \
    --ablation                                                                     \
    --output-dir  "Results/python_pipeline_ablation/animal-${ANIMAL_ID}"

EXIT_CODE=$?
echo "Finished ablation animal ${ANIMAL_ID} at $(date) — exit code: ${EXIT_CODE}"
exit $EXIT_CODE
