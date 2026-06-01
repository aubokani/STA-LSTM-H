#!/bin/bash
# =============================================================================
# submit_array_l40.sh — animals 10–18 on the L40S (gpuq) partition
# =============================================================================
# Run in parallel with submit_array_h100.sh (1–9 on gpucomputeq).
# =============================================================================

#SBATCH -J STA_l40
#SBATCH --array=10-18
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/array-l40-%a-%j.out
#SBATCH -e logs/array-l40-%a-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bokania@cqu.edu.au

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ANIMAL_ID=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
echo "============================================================"
echo "  Animal ${ANIMAL_ID} (L40S) — task ${SLURM_ARRAY_TASK_ID}"
echo "  Node: ${SLURMD_NODENAME}   GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "  Start: $(date)"
echo "============================================================"

module purge
module load Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu
source .venv/bin/activate

mkdir -p Results/python_pipeline/animal-${ANIMAL_ID} logs

python -u src/run_animal.py \
    --animal-id   "$SLURM_ARRAY_TASK_ID" \
    --folds       5 \
    --epochs      10 \
    --batch-size  256 \
    --max-windows 86400 \
    --max-horizon 30 \
    --output-dir  Results/python_pipeline/animal-${ANIMAL_ID}

echo "Animal ${ANIMAL_ID} finished at $(date)"
