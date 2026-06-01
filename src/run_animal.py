"""src/run_animal.py — HPC per-animal entry point.

Called by scripts/submit_hpc.sh once per Slurm array task.

Usage (Slurm):
    python src/run_animal.py --animal-id $SLURM_ARRAY_TASK_ID \
                             --folds 5 --epochs 40 \
                             --output-dir Results/python_pipeline/animal-01

Usage (local debug):
    python src/run_animal.py --animal-id 4 --folds 2 --epochs 2 --max-windows 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── path bootstrap ────────────────────────────────────────────────────────────
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import DataConfig, load_animals
from trainer import TrainConfig, run

# ── Data root: resolved at runtime so the same script runs anywhere ──────────
# Resolution order (first existing path wins): repo-relative data/raw/, then the
# authors' cluster / local-dev paths. Override any time with --data-root.
_REPO_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
_ADA_ROOT = Path("/home/bokania/projects/2026-STA-LSTM-H/data/raw")
_LOCAL_FALLBACK = Path(
    "/Users/ayub/Library/CloudStorage/OneDrive-CQUniversity"
    "/Research/LiveStock Monitoring/data/cattle-Zenodo"
)


def _resolve_data_root() -> Path:
    for candidate in (_REPO_RAW, _ADA_ROOT, _LOCAL_FALLBACK):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Dataset not found. Download the Pavlovic et al. cohort from Zenodo "
        "(https://doi.org/10.5281/zenodo.4064802) and place the CSVs at "
        f"{_REPO_RAW} (accel-NN.csv / halter-NN.csv), or pass --data-root <path>."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM-H/LSTM/GRU for one animal.")
    p.add_argument("--animal-id",   type=int, required=True,
                   help="Animal number 1–18 (maps to accel-NN.csv / halter-NN.csv)")
    p.add_argument("--data-root",   type=Path, default=None,
                   help="Path to folder containing accel-*.csv and halter-*.csv")
    p.add_argument("--output-dir",  type=Path, default=None,
                   help="Directory to write results (default: Results/python_pipeline/animal-NN)")
    p.add_argument("--folds",        type=int, default=5)
    p.add_argument("--epochs",       type=int, default=40)
    p.add_argument("--lr",           type=float, default=2e-3)
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--seq-len",      type=int, default=25)
    p.add_argument("--max-windows",  type=int, default=86400,
                   help="Cap on 1-second windows to load (0 = no cap)")
    p.add_argument("--max-horizon",  type=int, default=30,
                   help="Autoregressive rollout length in 1-s steps (0 = skip)")
    p.add_argument("--device",       default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--ablation",     action="store_true",
                   help="Run the (architecture × input-set) factorial ablation "
                        "(10 NN cells, no classical filters). Writes to "
                        "Results/python_pipeline_ablation/animal-NN/ unless "
                        "--output-dir overrides.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    animal_id = args.animal_id
    animal_str = f"{animal_id:02d}"

    data_root = args.data_root or _resolve_data_root()
    default_root = (
        "python_pipeline_ablation" if args.ablation else "python_pipeline"
    )
    output_dir = args.output_dir or (
        Path("Results") / default_root / f"animal-{animal_str}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Animal {animal_str}  —  pid {os.getpid()}")
    print(f"  Data root : {data_root}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*64}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    data_cfg = DataConfig(
        zenodo_root=data_root,
        max_windows=args.max_windows,
    )
    data = load_animals((animal_id,), data_cfg)
    load_time = time.perf_counter() - t0
    print(f"  Loaded {len(data):,} windows in {load_time:.1f}s")

    if data.empty:
        print(f"  ERROR: No data for animal {animal_str}. Exiting.")
        sys.exit(1)

    # ── Configure training ─────────────────────────────────────────────────────
    cfg = TrainConfig(
        animal_ids=(animal_id,),
        folds=args.folds,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        device_pref=args.device,
        max_horizon=args.max_horizon,
        output_dir=output_dir,
        ablation_mode=args.ablation,
    )

    # ── Run full CV pipeline ───────────────────────────────────────────────────
    comparison, wilcoxon = run(data, cfg)

    # ── Re-save with animal-scoped filenames ──────────────────────────────────
    comp_path    = output_dir / "comparison_report.csv"
    wil_path     = output_dir / "wilcoxon_stats.csv"
    summary_path = output_dir / "run_summary.json"

    comparison.to_csv(comp_path, index=False)
    wilcoxon.to_csv(wil_path, index=False)


    elapsed = time.perf_counter() - t0
    summary = {
        "animal_id":     animal_str,
        "n_windows":     int(len(data)),
        "load_time_s":   round(load_time, 2),
        "total_time_s":  round(elapsed, 2),
        "folds":         args.folds,
        "epochs":        args.epochs,
        "best_model":    comparison.iloc[0]["model"] if len(comparison) > 0 else "N/A",
        "best_accuracy": float(comparison.iloc[0]["accuracy_mean"]) if len(comparison) > 0 else None,
        "best_f1":       float(comparison.iloc[0]["f1_mean"])       if len(comparison) > 0 else None,
        "best_rmse":     float(comparison.iloc[0]["rmse_mean"])     if len(comparison) > 0 else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n  Results saved to {output_dir}")
    print(f"  Total wall time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
