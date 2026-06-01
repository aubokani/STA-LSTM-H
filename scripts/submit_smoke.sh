#!/bin/bash
# =============================================================================
# submit_smoke.sh — fast smoke test (animal 04, 2 folds, 2 epochs, 500 windows)
# =============================================================================
# Verifies that the pipeline runs end-to-end on a compute node before the
# real pilot submission. Should finish in ~5 minutes.
# =============================================================================

#SBATCH -J smoke-04
#SBATCH -p gpucomputeq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o logs/smoke-04-%j.out
#SBATCH -e logs/smoke-04-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "  Smoke test — Animal 04 (2 folds × 2 epochs × 500 windows)"
echo "  Node: ${SLURMD_NODENAME}   GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "  Start: $(date)"
echo "============================================================"

module purge
module load Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu
source .venv/bin/activate

mkdir -p Results/python_pipeline/animal-04 logs

python -u src/run_animal.py \
    --animal-id  4 \
    --folds      2 \
    --epochs     2 \
    --max-windows 500 \
    --max-horizon 0 \
    --output-dir Results/python_pipeline/animal-04-smoke

echo "Smoke finished at $(date)"
