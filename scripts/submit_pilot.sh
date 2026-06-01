#!/bin/bash
# =============================================================================
# submit_pilot.sh — single-animal pilot run (animal 04) on Ada
# =============================================================================
# Submit with:
#   sbatch scripts/submit_pilot.sh
#
# Outputs land in:
#   Results/python_pipeline/animal-04/
#   logs/pilot-04-<jobid>.{out,err}
# =============================================================================

#SBATCH -J pilot-04
#SBATCH -p gpucomputeq                      # H100 partition (4 GPUs total)
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/pilot-04-%j.out
#SBATCH -e logs/pilot-04-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bokania@cqu.edu.au

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "  Pilot run — Animal 04"
echo "  Node: ${SLURMD_NODENAME}   GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "  Start: $(date)"
echo "============================================================"

module purge
module load Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu
source .venv/bin/activate

mkdir -p Results/python_pipeline/animal-04 logs

python -u src/run_animal.py \
    --animal-id  4 \
    --folds      5 \
    --epochs     5 \
    --max-windows 86400 \
    --max-horizon 30 \
    --output-dir Results/python_pipeline/animal-04

echo "Pilot finished at $(date)"
