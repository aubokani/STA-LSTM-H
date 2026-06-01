"""scripts/predict_prev_baseline.py — Bout-autocorrelation floor.

The previous-step baseline asks: "what fraction of windows have the same
halter label as the previous window?" This is the trivial floor any
temporally-coherent classifier must clear. We compute it analytically from
the per-second `behavior` column without going through the sliding-window
machinery, giving a clean per-animal cohort summary.

Fixes a z-scoring bug in scripts/run_trivial_baselines.py where the
`behavior_prev` column was z-scored alongside accelerometry features and
could not be recovered by simple rounding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loader import DataConfig, load_animals  # noqa: E402

_ADA_ROOT = Path("/home/bokania/projects/2026-STA-LSTM-H/data/raw")


def main() -> None:
    rows = []
    for animal_id in range(1, 19):
        try:
            data = load_animals(
                (animal_id,),
                DataConfig(zenodo_root=_ADA_ROOT, max_windows=86400),
            )
        except Exception as e:
            print(f"animal {animal_id:02d}: SKIP ({e})")
            continue

        beh = data["behavior"].astype(int).to_numpy()
        n = len(beh)
        if n < 26:
            continue

        # 25-step window: target is beh[i+25], "previous" is beh[i+24]
        seq_len = 25
        y_true = beh[seq_len:]
        y_prev = beh[seq_len - 1: -1]
        assert len(y_true) == len(y_prev) == n - seq_len

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, f1s = [], []
        for tr, te in skf.split(np.zeros(len(y_true)), y_true):
            accs.append(accuracy_score(y_true[te], y_prev[te]))
            f1s.append(f1_score(y_true[te], y_prev[te],
                                average="macro", zero_division=0))
        acc_mean, f1_mean = float(np.mean(accs)), float(np.mean(f1s))
        rows.append({
            "animal":   animal_id,
            "accuracy": acc_mean,
            "f1":       f1_mean,
        })
        print(f"animal {animal_id:02d}: predict_prev acc={acc_mean:.4f} "
              f"f1={f1_mean:.4f}")

    df = pd.DataFrame(rows)
    out = ROOT / "Results" / "predict_prev_cohort.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(f"Cohort:  acc = {df['accuracy'].mean():.4f} +/- "
          f"{df['accuracy'].std():.4f}")
    print(f"         F1  = {df['f1'].mean():.4f} +/- {df['f1'].std():.4f}")


if __name__ == "__main__":
    main()
