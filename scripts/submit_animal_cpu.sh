#!/bin/bash
# =============================================================================
# submit_animal_cpu.sh — single-animal CPU job (workq partition)
# =============================================================================
# Submitted from submit_loop_cpu.sh once per animal id.
#
# Args (positional):
#   $1  ANIMAL_ID  (1..18)
# =============================================================================

#SBATCH -J STA_cpu
#SBATCH -p workq
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/cpu-%x-%j.out
#SBATCH -e logs/cpu-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bokania@cqu.edu.au

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ANIMAL_ID="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [ -z "$ANIMAL_ID" ]; then
    echo "ERROR: animal id not provided as \$1 and \$SLURM_ARRAY_TASK_ID is unset"
    exit 2
fi
ANIMAL_STR=$(printf "%02d" "$ANIMAL_ID")

echo "============================================================"
echo "  STA-LSTM-H CPU job — animal ${ANIMAL_STR}"
echo "  Node: ${SLURMD_NODENAME}   CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "  Start: $(date)"
echo "============================================================"

module purge
module load Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu
source .venv/bin/activate

# Use all allocated CPU cores for PyTorch intra-op parallelism
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p Results/python_pipeline/animal-${ANIMAL_STR} logs

python -u src/run_animal.py \
    --animal-id   "$ANIMAL_ID" \
    --folds       5 \
    --epochs      10 \
    --batch-size  256 \
    --max-windows 86400 \
    --max-horizon 0 \
    --device      cpu \
    --output-dir  Results/python_pipeline/animal-${ANIMAL_STR}

echo "Animal ${ANIMAL_STR} finished at $(date)"
