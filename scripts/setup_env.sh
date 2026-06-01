#!/bin/bash
# =============================================================================
# setup_env.sh — One-time environment bootstrap on Ada
# =============================================================================
# Run this ONCE on the Ada login node before the first sbatch submission.
# It creates a project-local virtual environment and installs all dependencies.
#
# Usage (from $PROJECT_DIR on Ada):
#   bash scripts/setup_env.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "===================================================================="
echo "  LSTM-H Ada Environment Setup"
echo "  Project: $PROJECT_DIR"
echo "===================================================================="

# ── Load deep-learning Python module ─────────────────────────────────────────
module purge
module load Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu
echo "[1/4] Loaded module: Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu"

# ── Create virtual environment ────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[2/4] Venv already exists at $VENV_DIR — skipping creation"
else
    python -m venv "$VENV_DIR"
    echo "[2/4] Created venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "[3/4] Activated venv"

# ── Install dependencies ──────────────────────────────────────────────────────
# The deep-learning module already ships torch; pip-install remaining packages.
pip install --upgrade pip --quiet
pip install \
    pandas>=2.0 \
    scikit-learn>=1.3 \
    scipy>=1.11 \
    matplotlib>=3.7 \
    imbalanced-learn>=0.11 \
    --quiet

echo "[4/4] Dependencies installed"
echo ""
echo "  Setup complete. Submit jobs with:"
echo "    sbatch scripts/submit_hpc.sh"
echo "===================================================================="
