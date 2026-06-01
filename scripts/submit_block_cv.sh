#!/bin/bash
# =============================================================================
# submit_block_cv.sh — Block-stratified within-animal CV
# =============================================================================
# Addresses Reviewer 1 Minor 8 / Reviewer 2 Major 6: the headline
# StratifiedKFold(shuffle=True) splits 96%-overlapping adjacent windows into
# train and test, partially explaining the within-animal ceiling accuracy.
# This array runs each of the 18 animals through 5 contiguous-block folds,
# with no temporal overlap between train and test.
#
# Output: Results/python_pipeline_blockcv/animal-NN/
# =============================================================================

#SBATCH -J STALSTMH_blockcv
#SBATCH --array=1-18
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -p gpucomputeq
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00

#SBATCH -o logs/block-cv-%a.out
#SBATCH -e logs/block-cv-%a.err

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
echo " BLOCK-CV  animal=${ANIMAL_ID}  node=${SLURMD_NODENAME}  start=$(date)"
echo "==========================================="

mkdir -p "Results/python_pipeline_blockcv/animal-${ANIMAL_ID}" logs

python -u scripts/run_block_cv_animal.py \
    --animal-id "$SLURM_ARRAY_TASK_ID" \
    --folds      5 \
    --epochs     40 \
    --output-dir "Results/python_pipeline_blockcv/animal-${ANIMAL_ID}"

EXIT=$?
echo "Finished BLOCK-CV animal ${ANIMAL_ID} at $(date) — exit ${EXIT}"
exit $EXIT
