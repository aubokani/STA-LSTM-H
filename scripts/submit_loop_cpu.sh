#!/bin/bash
# =============================================================================
# submit_loop_cpu.sh — submit one CPU job per animal (1..18) on workq
# =============================================================================
# Mirrors the user's earlier PBS qsub-loop pattern: a thin bash loop that
# dispatches N independent Slurm jobs, one per task. The workq partition has
# no GPU GRES cap, so all 18 jobs can run truly concurrently (cap is
# cpu=500 → 31 concurrent at 16 cores per task).
#
# Usage (from project root on Ada):
#   bash scripts/submit_loop_cpu.sh                # all 18 animals
#   bash scripts/submit_loop_cpu.sh 4              # only animal 04
#   bash scripts/submit_loop_cpu.sh 1 2 3 4        # only animals 1..4
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

# Default to all 18 animals if no args
if [ "$#" -eq 0 ]; then
    ANIMALS=$(seq 1 18)
else
    ANIMALS="$*"
fi

echo "============================================================"
echo "  Submitting CPU jobs to workq for: ${ANIMALS}"
echo "  Time: $(date)"
echo "============================================================"

JIDS=()
for ID in $ANIMALS; do
    # --export=NONE --get-user-env: re-source the user's login env on the
    # compute node so Lmod's `module` function is defined. Without this,
    # sbatch from a non-login parent (cron, automation) propagates an empty
    # MODULESHOME/LMOD env and the script aborts at `module purge` with
    # exit 127. Safe for interactive use too.
    J=$(sbatch --parsable --export=NONE --get-user-env --job-name "STA_${ID}" scripts/submit_animal_cpu.sh "$ID")
    JIDS+=("$J")
    printf "  animal %02d → job %s\n" "$ID" "$J"
done

echo
echo "Submitted ${#JIDS[@]} jobs."
echo "Track with:  squeue -u bokania -o '%.10i %.20j %.8T %.10M %.20R'"
echo
echo "Job IDs:"
printf "  %s\n" "${JIDS[@]}"
