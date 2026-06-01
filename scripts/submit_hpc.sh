#!/bin/bash
# =============================================================================
# submit_hpc.sh — Ada Lovelace (CQU HPC) Array Job
# =============================================================================
# Submits one Slurm task per animal (IDs 01–18) in parallel.
# Each task runs src/run_animal.py for a single animal and writes its
# results to $PROJECT_DIR/Results/python_pipeline/animal-NN/
#
# Usage (from $PROJECT_DIR on the Ada login node):
#   sbatch scripts/submit_hpc.sh
#
# Hardware targets (live `sinfo` on Ada, 2026-04-27 — see docs/hpc/ada_reference.md):
#   workq*       : 10× CPU-only nodes, 96 cores, ~770 GB
#   gpuq         : 2 nodes (hpc05-gln01–02), 1× L40S 48 GB each
#   gpucomputeq  : 2 nodes (hpc05-ghn01–02), 2× H100 96 GB each   ← used here
#
# Account `cquhpc` cap: cpu=500 (≈62 concurrent 8-core tasks). All partitions
# advertise infinite walltime; the -t below is a self-imposed safety cap.
# =============================================================================

#SBATCH -J STALSTMH_cattle                 # Job name
#SBATCH --array=1-18                       # One task per animal (01–18)
#SBATCH -c 8                               # 8 CPU cores per task
#SBATCH --mem=32G                          # 32 GB RAM per task
#SBATCH -p gpucomputeq                     # H100 partition (4 GPUs total across 2 nodes)
#SBATCH --gres=gpu:1                       # 1 GPU per task
#SBATCH -t 04:00:00                        # 4-hour wall time per animal (self-cap)

# ── Output / error files (one per task) ─────────────────────────────────────
#SBATCH -o logs/animal-%a.out
#SBATCH -e logs/animal-%a.err

# ── Email notifications ──────────────────────────────────────────────────────
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bokania@cqu.edu.au

# ── Change to project directory ──────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"

# ── Load Ada deep-learning Python module ────────────────────────────────────
#    Module identified from: module avail Python (deep-learning variant)
module purge
module load Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu

# ── Activate project virtual environment ────────────────────────────────────
#    Created by scripts/setup_env.sh (run once before first sbatch)
VENV_DIR="$SLURM_SUBMIT_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Run: bash scripts/setup_env.sh"
    exit 1
fi

# ── Map Slurm array index → zero-padded animal ID ───────────────────────────
ANIMAL_ID=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
echo "============================================================"
echo " Animal ${ANIMAL_ID} — Task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}"
echo " Node: ${SLURMD_NODENAME}  GPU: ${CUDA_VISIBLE_DEVICES}"
echo " Start: $(date)"
echo "============================================================"

# ── Create per-animal output directory ──────────────────────────────────────
mkdir -p "Results/python_pipeline/animal-${ANIMAL_ID}"
mkdir -p logs

# ── Run per-animal training ──────────────────────────────────────────────────
# Data root: reorganised on 2026-04-27 from data_raw/ → data/raw/
# Absolute path so compute nodes cannot inherit a wrong CWD.
DATA_ROOT="/home/bokania/projects/2026-STA-LSTM-H/data/raw"

python -u src/run_animal.py \
    --animal-id  "$SLURM_ARRAY_TASK_ID"                         \
    --data-root  "$DATA_ROOT"                                    \
    --folds      5                                               \
    --epochs     40                                              \
    --max-windows 86400                                          \
    --max-horizon 30                                             \
    --output-dir "Results/python_pipeline/animal-${ANIMAL_ID}"

EXIT_CODE=$?
echo "Finished animal ${ANIMAL_ID} at $(date) — exit code: ${EXIT_CODE}"
exit $EXIT_CODE
