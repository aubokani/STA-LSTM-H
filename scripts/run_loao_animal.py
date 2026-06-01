"""scripts/run_loao_animal.py — HPC LOAO held-out-animal entry point.

Called by scripts/submit_loao.sh once per Slurm array task. Each task holds
out one animal (specified by --held-out-animal-id), trains the full neural
+ XGBoost battery on the other 17, and evaluates on the held-out animal.

Reviewer 3 flagged the per-animal 5-fold StratifiedKFold protocol as
inflating accuracy through temporally-autocorrelated train/test sharing of
the same animal's bouts. This driver implements the leave-one-animal-out
(LOAO) protocol that the PLF deep-learning literature uses as the
external-validation standard.

Usage (Slurm):
    python scripts/run_loao_animal.py --held-out-animal-id $SLURM_ARRAY_TASK_ID \\
                                      --epochs 40 \\
                                      --output-dir Results/python_pipeline_loao/animal-01

Usage (local debug):
    python scripts/run_loao_animal.py --held-out-animal-id 4 --epochs 2 \\
                                      --max-windows 1000

The aggregator `scripts/consolidate_loao.py` walks the per-held-out-animal
directories and produces `Results/comparison_report_loao.csv` with one row
per (model, held_out_animal) pair.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loader import DataConfig, load_animals  # noqa: E402
from trainer import TrainConfig, run_loao  # noqa: E402

# Same resolved-at-runtime data root scheme as src/run_animal.py
_ADA_ROOT = Path("/home/bokania/projects/2026-STA-LSTM-H/data/raw")
_LOCAL_FALLBACK = Path(
    "/Users/ayub/Library/CloudStorage/OneDrive-CQUniversity"
    "/Research/LiveStock Monitoring/data/cattle-Zenodo"
)


def _resolve_data_root() -> Path:
    if _ADA_ROOT.exists():
        return _ADA_ROOT
    if _LOCAL_FALLBACK.exists():
        return _LOCAL_FALLBACK
    raise FileNotFoundError(
        "Data root not found. Either:\n"
        f"  • On Ada : ensure {_ADA_ROOT} exists (rsync the dataset first).\n"
        f"  • Locally: pass --data-root <path> explicitly."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--held-out-animal-id", type=int, required=True,
                   help="Animal number 1–18 to hold out for evaluation.")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=25)
    p.add_argument("--max-windows", type=int, default=86400,
                   help="Cap on 1-second windows per animal (0 = no cap).")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--ablation", action="store_true",
                   help="LOAO with the (architecture × input-set) factorial "
                        "ablation registry instead of the headline NN_MODELS.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    held_out = args.held_out_animal_id
    held_out_str = f"{held_out:02d}"
    if not (1 <= held_out <= 18):
        raise SystemExit(f"--held-out-animal-id must be in [1, 18], got {held_out}")

    data_root = args.data_root or _resolve_data_root()
    default_root = (
        "python_pipeline_loao_ablation" if args.ablation
        else "python_pipeline_loao"
    )
    output_dir = args.output_dir or (
        ROOT / "Results" / default_root / f"animal-{held_out_str}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  LOAO held-out animal: {held_out_str}  —  pid {os.getpid()}")
    print(f"  Data root : {data_root}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*64}\n")

    t0 = time.perf_counter()
    data_cfg = DataConfig(
        zenodo_root=data_root,
        max_windows=args.max_windows,
    )
    all_ids = tuple(range(1, 19))
    full = load_animals(all_ids, data_cfg)
    load_time = time.perf_counter() - t0
    print(f"  Loaded {len(full):,} windows from {full['animal_id'].nunique()} "
          f"animals in {load_time:.1f}s")

    data_train = full[full["animal_id"] != held_out].reset_index(drop=True)
    data_test  = full[full["animal_id"] == held_out].reset_index(drop=True)
    if data_test.empty:
        print(f"  ERROR: held-out animal {held_out_str} has no rows. Exiting.")
        sys.exit(1)

    cfg = TrainConfig(
        animal_ids=all_ids,
        folds=1,                 # single split; the "fold" is the held-out animal
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        device_pref=args.device,
        max_horizon=0,           # autoregressive rollout disabled in LOAO mode
        output_dir=output_dir,
        ablation_mode=args.ablation,
    )

    ledger, _ = run_loao(data_train, data_test, cfg, held_out_animal_id=held_out)

    elapsed = time.perf_counter() - t0
    summary = {
        "held_out_animal_id": held_out_str,
        "n_train_animals":    int(data_train["animal_id"].nunique()),
        "n_train_rows":       int(len(data_train)),
        "n_test_rows":        int(len(data_test)),
        "load_time_s":        round(load_time, 2),
        "total_time_s":       round(elapsed, 2),
        "epochs":             args.epochs,
        "best_model":         ledger.iloc[0]["model"] if len(ledger) > 0 else "N/A",
        "best_accuracy":      float(ledger.iloc[0]["accuracy_mean"]) if len(ledger) > 0 else None,
        "best_f1":            float(ledger.iloc[0]["f1_mean"])       if len(ledger) > 0 else None,
        "best_rmse":          float(ledger.iloc[0]["rmse_mean"])     if len(ledger) > 0 else None,
    }
    (output_dir / "loao_run_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(f"\n  LOAO held-out animal {held_out_str} done in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
