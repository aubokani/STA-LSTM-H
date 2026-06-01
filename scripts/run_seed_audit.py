"""scripts/run_seed_audit.py — Multi-seed audit for one animal.

Addresses Reviewer 1 Major Concern 3: the headline ±0.0009 std bands across
18 animals × 5 folds were measured at one initialisation seed per training
run; training-stochastic variance was conflated with cross-animal
heterogeneity. This driver re-runs a single animal's 5-fold CV across
multiple seeds for STA-LSTM-H, STA-LSTM, and the Transformer (the three
architectures whose pairwise differences sit closest to noise), so the
manuscript can report seed-variance bands separately from cohort-variance
bands.

Usage (Slurm array):
    python scripts/run_seed_audit.py --animal-id $SLURM_ARRAY_TASK_ID \
                                     --seeds 7 42 101 1729 2026
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loader import DataConfig, load_animals  # noqa: E402
from trainer import TrainConfig, run_cross_validation  # noqa: E402

_ADA_ROOT = Path("/home/bokania/projects/2026-STA-LSTM-H/data/raw")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--animal-id", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[7, 42, 101, 1729, 2026])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    animal_id = args.animal_id
    out_dir = args.output_dir or (
        ROOT / "Results" / "python_pipeline_seedaudit" /
        f"animal-{animal_id:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[seed-audit] animal={animal_id:02d}  seeds={args.seeds}")
    print(f"[seed-audit] loading animal {animal_id:02d}...")
    data = load_animals(
        (animal_id,),
        DataConfig(zenodo_root=_ADA_ROOT, max_windows=86400),
    )

    rows: list[dict] = []
    for seed in args.seeds:
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = TrainConfig(
            folds=args.folds,
            epochs=args.epochs,
            seed=seed,
        )
        cv_results, _ = run_cross_validation(data, cfg)

        for kind, fold_metrics in cv_results.items():
            for fm in fold_metrics:
                rows.append({
                    "seed":     seed,
                    "animal":   animal_id,
                    "model":    kind,
                    "fold":     fm.get("fold", -1),
                    "accuracy": fm.get("accuracy", float("nan")),
                    "f1":       fm.get("f1", float("nan")),
                    "rmse":     fm.get("rmse", float("nan")),
                })
        print(f"[seed-audit] seed={seed} done in {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "seed_audit_long.csv", index=False)

    # Compact summary: per (model, seed) mean accuracy across folds, then std
    # across seeds → that is the training-stochastic variance band.
    if not df.empty:
        per_seed = (
            df.groupby(["model", "seed"])[["accuracy", "f1", "rmse"]]
              .mean().reset_index()
        )
        per_seed.to_csv(out_dir / "seed_audit_per_seed.csv", index=False)
        summary = (
            per_seed.groupby("model")[["accuracy", "f1", "rmse"]]
                    .agg(["mean", "std"])
        )
        summary.to_csv(out_dir / "seed_audit_summary.csv")
        print(summary)

    (out_dir / "seed_audit_run.json").write_text(json.dumps({
        "animal_id": animal_id,
        "seeds":     args.seeds,
        "epochs":    args.epochs,
        "folds":     args.folds,
    }, indent=2))


if __name__ == "__main__":
    main()
