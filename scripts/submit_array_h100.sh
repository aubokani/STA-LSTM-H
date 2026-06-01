#!/bin/bash
# =============================================================================
# submit_array_h100.sh — animals 01–09 on the H100 (gpucomputeq) partition
# =============================================================================
# Run in parallel with submit_array_l40.sh (10–18 on gpuq) so all 18 animals
# go through both H100 nodes (4 GPUs) and both L40S nodes (2 GPUs) at once.
# =============================================================================

#SBATCH -J STA_h100
#SBATCH --array=1-9
#SBATCH -p gpucomputeq
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/array-h100-%a-%j.out
#SBATCH -e logs/array-h100-%a-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bokania@cqu.edu.au

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ANIMAL_ID=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
echo "============================================================"
echo "  Animal ${ANIMAL_ID} (H100) — task ${SLURM_ARRAY_TASK_ID}"
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
