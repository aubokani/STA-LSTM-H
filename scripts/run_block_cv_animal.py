"""scripts/run_block_cv_animal.py — Block-stratified within-animal CV.

Addresses Reviewer 1 Minor Concern 8 and Reviewer 2 Major Concern 6: the
default StratifiedKFold(shuffle=True) splits adjacent windows that share
24/25 input rows across train and test. This script implements a
contiguous-block CV variant: each animal's ~86k-window timeline is split
into K contiguous, non-overlapping blocks; fold k holds out block k and
trains on the remaining K-1 blocks. No two folds share an input row.

Output schema mirrors `comparison_report.csv` from the standard pipeline so
`scripts/aggregate_18animals.py` can pick it up via a parallel directory.

Usage (Slurm array):
    python scripts/run_block_cv_animal.py --animal-id $SLURM_ARRAY_TASK_ID

Output directory:
    Results/python_pipeline_blockcv/animal-NN/
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

from data_loader import (  # noqa: E402
    DataConfig, load_animals, make_sequences,
)
from trainer import (  # noqa: E402
    TrainConfig, MODEL_FEATURE_MAP, NN_MODELS, MODEL_DISPLAY,
    _train_fold, audit_class_balance, select_device,
)

_ADA_ROOT = Path("/home/bokania/projects/2026-STA-LSTM-H/data/raw")


def block_split_indices(n: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield k folds of contiguous-block (train_idx, test_idx) splits.

    Block i = indices [i*B, (i+1)*B). Train indices = all other blocks.
    """
    block_size = n // k
    folds = []
    for i in range(k):
        start = i * block_size
        end = (i + 1) * block_size if i < k - 1 else n
        test_idx = np.arange(start, end)
        train_idx = np.concatenate([np.arange(0, start), np.arange(end, n)])
        folds.append((train_idx, test_idx))
    return folds


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--animal-id", type=int, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max-windows", type=int, default=86400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    animal_id = args.animal_id
    out_dir = args.output_dir or (
        ROOT / "Results" / "python_pipeline_blockcv" /
        f"animal-{animal_id:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[block-cv] animal={animal_id:02d}  folds={args.folds}  "
          f"epochs={args.epochs}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_animals(
        (animal_id,),
        DataConfig(zenodo_root=_ADA_ROOT, max_windows=args.max_windows),
    )
    cfg = TrainConfig(folds=args.folds, epochs=args.epochs, seed=args.seed,
                      verbose=False)

    balance_decision, class_weights = audit_class_balance(
        data, smote_threshold=cfg.smote_threshold
    )
    device = select_device(cfg.device_pref)
    print(f"[block-cv] balance={balance_decision}  device={device}")

    rows = []
    # Build master sequence array (using lstm_h feature set for index size)
    _, _, Yc_all, _ = make_sequences(
        data, MODEL_FEATURE_MAP["lstm_h"], cfg.sequence_length
    )
    n_total = len(Yc_all)
    block_folds = block_split_indices(n_total, args.folds)

    for fold_idx, (train_idx, test_idx) in enumerate(block_folds):
        t0 = time.time()
        print(f"\n  Fold {fold_idx + 1}/{args.folds}  "
              f"(train={len(train_idx)}, test={len(test_idx)})")
        for kind in NN_MODELS:
            fcols = MODEL_FEATURE_MAP[kind]
            X_all, Yxy_all, Yc_all_m, nm = make_sequences(
                data, fcols, cfg.sequence_length
            )
            X_tr = X_all[train_idx]
            X_te = X_all[test_idx]
            Yxy_tr, Yxy_te = Yxy_all[train_idx], Yxy_all[test_idx]
            Yc_tr,  Yc_te  = Yc_all_m[train_idx], Yc_all_m[test_idx]

            metrics = _train_fold(
                kind, X_tr, Yxy_tr, Yc_tr, X_te, Yxy_te, Yc_te,
                nm, balance_decision, class_weights, cfg, device,
                verbose=False,
            )
            rows.append({
                "animal":   animal_id,
                "model":    MODEL_DISPLAY.get(kind, kind),
                "kind":     kind,
                "fold":     fold_idx + 1,
                "accuracy": metrics["accuracy"],
                "f1":       metrics["f1"],
                "rmse":     metrics["rmse"],
                "latency_ms": metrics.get("latency_ms", float("nan")),
            })
            print(f"    {MODEL_DISPLAY[kind]:12s}  Acc={metrics['accuracy']:.4f}  "
                  f"F1={metrics['f1']:.4f}  RMSE={metrics['rmse']:.2f}")
        print(f"  fold time: {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "block_cv_long.csv", index=False)

    summary = (
        df.groupby("model")[["accuracy", "f1", "rmse"]]
          .agg(["mean", "std"]).reset_index()
    )
    summary.to_csv(out_dir / "block_cv_summary.csv", index=False)

    (out_dir / "block_cv_run.json").write_text(json.dumps({
        "animal_id": animal_id,
        "folds":     args.folds,
        "epochs":    args.epochs,
        "n_windows": int(n_total),
    }, indent=2))
    print(f"\n[block-cv] wrote {out_dir}/block_cv_long.csv "
          f"({len(df)} rows)")


if __name__ == "__main__":
    main()
