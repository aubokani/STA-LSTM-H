#!/bin/bash
# =============================================================================
# submit_loao_ablation.sh — LOAO with the ABLATION_NN_MODELS registry
# =============================================================================
# Addresses Reviewer 1 Major Concern 2 / Reviewer 2 line comment on 05_results
# 155: the headline LOAO ran with same-step behavior_prev for the STA family
# and accel-only for the plain baselines, an input asymmetry. This rerun
# trains every architecture × every input set (accel-only, +bp, +bp_lag1) on
# the 17 training animals and evaluates on the held-out animal, so the
# cross-animal generalisation gap can be read at fixed input set.
#
# Outputs land in Results/python_pipeline_loao_ablation/animal-NN/.
# Consolidation: scripts/consolidate_loao.py picks them up automatically.
# =============================================================================

#SBATCH -J STALSTMH_loao_abl
#SBATCH --array=1-18
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -p gpucomputeq
#SBATCH --gres=gpu:1
#SBATCH -t 14:00:00

#SBATCH -o logs/loao-abl-%a.out
#SBATCH -e logs/loao-abl-%a.err

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

HELD_OUT_ID=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
echo "==========================================="
echo " LOAO-ABLATION  animal=${HELD_OUT_ID}  node=${SLURMD_NODENAME}  start=$(date)"
echo "==========================================="

mkdir -p "Results/python_pipeline_loao_ablation/animal-${HELD_OUT_ID}" logs

python -u scripts/run_loao_animal.py \
    --held-out-animal-id "$SLURM_ARRAY_TASK_ID" \
    --data-root          "/home/bokania/projects/2026-STA-LSTM-H/data/raw" \
    --epochs             40 \
    --max-windows        86400 \
    --ablation \
    --output-dir         "Results/python_pipeline_loao_ablation/animal-${HELD_OUT_ID}"

EXIT=$?
echo "Finished LOAO-ABLATION animal ${HELD_OUT_ID} at $(date) — exit ${EXIT}"
exit $EXIT
