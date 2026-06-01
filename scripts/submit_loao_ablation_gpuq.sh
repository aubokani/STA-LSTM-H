#!/bin/bash
# =============================================================================
# submit_loao_ablation_gpuq.sh — LOAO-ablation spillover to gpuq
# =============================================================================
# Runs animals 3-10 on the gpuq partition in parallel with the gpucomputeq
# array (job 154985) so wall-clock for the full 18-animal LOAO-ablation
# rerun is ~2x faster than serial. Same per-task config as
# submit_loao_ablation.sh; only partition and task-id range differ.
# =============================================================================

#SBATCH -J STALSTMH_loao_abl_gq
#SBATCH --array=3-10
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH -t 14:00:00

#SBATCH -o logs/loao-abl-gpuq-%a.out
#SBATCH -e logs/loao-abl-gpuq-%a.err

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
echo " LOAO-ABLATION-gpuq  animal=${HELD_OUT_ID}  node=${SLURMD_NODENAME}  start=$(date)"
echo "==========================================="

# Skip if already finished on gpucomputeq side
if [ -f "Results/python_pipeline_loao_ablation/animal-${HELD_OUT_ID}/loao_report.csv" ]; then
    echo "Already done; skipping."
    exit 0
fi

mkdir -p "Results/python_pipeline_loao_ablation/animal-${HELD_OUT_ID}" logs

python -u scripts/run_loao_animal.py \
    --held-out-animal-id "$SLURM_ARRAY_TASK_ID" \
    --data-root          "/home/bokania/projects/2026-STA-LSTM-H/data/raw" \
    --epochs             40 \
    --max-windows        86400 \
    --ablation \
    --output-dir         "Results/python_pipeline_loao_ablation/animal-${HELD_OUT_ID}"

EXIT=$?
echo "Finished LOAO-ABLATION-gpuq animal ${HELD_OUT_ID} at $(date) — exit ${EXIT}"
exit $EXIT
