"""scripts/run_trivial_baselines.py — Trivial classification baselines.

Addresses Reviewer 1 Major Concern 9: the manuscript needs a "stupid
baseline" floor before the 0.9551 accel-only number can be judged. This
script runs four floors on the same 14-feature windowed representation per
animal × 5-fold StratifiedKFold split:

  * majority         — predict the most-common class in the training fold
  * predict_previous — predict the test window's previous-step label
                       (uses behavior_prev, exposes the upper bound of
                       label-autocorrelation alone)
  * logreg           — multinomial logistic regression on the flattened
                       (T × F) input
  * gbdt             — gradient-boosted decision tree on the flattened
                       (T × F) input

Output: Results/python_pipeline_trivial/animal-NN/trivial_report.csv with
columns (model, fold, accuracy, f1, latency_ms).

Aggregated by `scripts/aggregate_trivial.py` (one-liner) into
Results/aggregate_trivial_summary.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loader import (  # noqa: E402
    DataConfig, ACCEL_FEAT_COLS, LSTMH_EXTRA_COLS, load_animals, make_sequences,
)

_ADA_ROOT = Path("/home/bokania/projects/2026-STA-LSTM-H/data/raw")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--animal-ids", type=int, nargs="+",
                   default=list(range(1, 19)))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-windows", type=int, default=86400)
    p.add_argument("--output-root", type=Path, default=None)
    return p.parse_args()


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def main() -> None:
    args = _parse_args()
    out_root = args.output_root or (ROOT / "Results" / "python_pipeline_trivial")
    out_root.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict] = []
    for animal_id in args.animal_ids:
        print(f"\n[trivial] animal={animal_id:02d}")
        out_dir = out_root / f"animal-{animal_id:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        data = load_animals(
            (animal_id,),
            DataConfig(zenodo_root=_ADA_ROOT, max_windows=args.max_windows),
        )

        # Use lstm_h feature set so we have behavior_prev available for the
        # predict_previous baseline. We only feed the 14 accel features to the
        # logreg/gbdt models (drop the last column).
        X, _, Yc, _ = make_sequences(
            data, ACCEL_FEAT_COLS + LSTMH_EXTRA_COLS, seq_len=25
        )
        # X: (N, 25, 15) with behavior_prev as the 15th channel
        n_feat = 14   # accel only
        X_accel = X[..., :n_feat]
        X_prev_lastrow = X[:, -1, n_feat]  # behavior_prev of last input row

        skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                              random_state=args.seed)
        rows: list[dict] = []
        for fold_idx, (tr, te) in enumerate(skf.split(np.zeros(len(Yc)), Yc)):
            Xt_flat = _flatten(X_accel[tr])
            Xe_flat = _flatten(X_accel[te])
            yt, ye = Yc[tr], Yc[te]

            # ── 1. predict-majority (training-set mode) ──────────────────────
            maj = int(np.bincount(yt).argmax())
            yp_maj = np.full_like(ye, maj)
            rows.append({
                "model": "majority", "fold": fold_idx + 1,
                "accuracy": accuracy_score(ye, yp_maj),
                "f1": f1_score(ye, yp_maj, average="macro", zero_division=0),
                "latency_ms": 0.0,
            })

            # ── 2. predict-previous (uses behavior_prev_lastrow) ─────────────
            # behavior_prev is stored as float in [0, 1] = (0,0.5,1.0); recover.
            yp_prev = np.round(X_prev_lastrow[te] * 2).astype(int)
            yp_prev = np.clip(yp_prev, 0, 2)
            rows.append({
                "model": "predict_prev", "fold": fold_idx + 1,
                "accuracy": accuracy_score(ye, yp_prev),
                "f1": f1_score(ye, yp_prev, average="macro", zero_division=0),
                "latency_ms": 0.0,
            })

            # ── 3. multinomial logistic regression ───────────────────────────
            sc = StandardScaler().fit(Xt_flat)
            Xt_s = sc.transform(Xt_flat)
            Xe_s = sc.transform(Xe_flat)
            t0 = time.time()
            lr = LogisticRegression(max_iter=500, solver="lbfgs",
                                    random_state=args.seed)
            lr.fit(Xt_s, yt)
            yp_lr = lr.predict(Xe_s)
            lat_lr_ms = (time.time() - t0) * 1000.0 / len(ye)
            rows.append({
                "model": "logreg", "fold": fold_idx + 1,
                "accuracy": accuracy_score(ye, yp_lr),
                "f1": f1_score(ye, yp_lr, average="macro", zero_division=0),
                "latency_ms": lat_lr_ms,
            })

            # ── 4. histogram-boosted decision tree (fast) ────────────────────
            t0 = time.time()
            gb = HistGradientBoostingClassifier(
                max_iter=100, max_depth=6, learning_rate=0.1,
                random_state=args.seed,
            )
            gb.fit(Xt_flat, yt)
            yp_gb = gb.predict(Xe_flat)
            lat_gb_ms = (time.time() - t0) * 1000.0 / len(ye)
            rows.append({
                "model": "gbdt", "fold": fold_idx + 1,
                "accuracy": accuracy_score(ye, yp_gb),
                "f1": f1_score(ye, yp_gb, average="macro", zero_division=0),
                "latency_ms": lat_gb_ms,
            })

            print(f"  fold {fold_idx + 1}: "
                  f"maj={rows[-4]['accuracy']:.4f} "
                  f"prev={rows[-3]['accuracy']:.4f} "
                  f"logreg={rows[-2]['accuracy']:.4f} "
                  f"gbdt={rows[-1]['accuracy']:.4f}")

        df = pd.DataFrame(rows)
        df["animal"] = animal_id
        df.to_csv(out_dir / "trivial_report.csv", index=False)
        long_rows.extend(df.to_dict("records"))

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out_root / "trivial_long.csv", index=False)
    # Per-animal mean across folds → then mean ± std across animals
    per_animal = (
        long_df.groupby(["animal", "model"])[["accuracy", "f1"]]
               .mean().reset_index()
    )
    summary = (
        per_animal.groupby("model")[["accuracy", "f1"]]
                  .agg(["mean", "std"])
    )
    summary.to_csv(out_root / "aggregate_trivial_summary.csv")
    print("\n[trivial] per-model cohort summary:")
    print(summary)


if __name__ == "__main__":
    main()
