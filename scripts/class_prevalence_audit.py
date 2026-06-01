"""scripts/class_prevalence_audit.py — Per-animal class prevalence and
balancer-branch audit.

Addresses Reviewer 2 Major Concerns 3 and 4 and Reviewer 1 Concern 7:
the manuscript mentions SMOTE-vs-weighted-CE branching at threshold 0.80
but never reports the actual class distributions or which branch fired
per animal. This script reads each animal's cached feature CSV, computes
the (Other, Ruminating, Eating) percentages and which branch the trainer
would invoke, and writes a tidy table that the manuscript can cite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loader import (  # noqa: E402
    DataConfig, load_animals,
)

_ADA_ROOT = Path("/home/bokania/projects/2026-STA-LSTM-H/data/raw")
_SMOTE_THRESHOLD = 0.80

CLASS_NAMES = {0: "Other", 1: "Ruminating", 2: "Eating"}


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
        counts = {c: int((beh == c).sum()) for c in (0, 1, 2)}
        pct = {c: counts[c] / n for c in counts}
        dom = max(pct, key=pct.get)
        branch = "smote" if pct[dom] > _SMOTE_THRESHOLD else "weighted_ce"
        row = {
            "animal":              animal_id,
            "n_windows":           n,
            "pct_Other":           pct[0],
            "pct_Ruminating":      pct[1],
            "pct_Eating":          pct[2],
            "dominant_class_idx":  dom,
            "dominant_class_name": CLASS_NAMES[dom],
            "dominant_pct":        pct[dom],
            "balancer_branch":     branch,
        }
        rows.append(row)
        print(f"animal {animal_id:02d}: n={n}  "
              f"Other={pct[0]:.3f} Rum={pct[1]:.3f} Eat={pct[2]:.3f}  "
              f"dom={CLASS_NAMES[dom]} ({pct[dom]:.3f})  → {branch}")

    df = pd.DataFrame(rows)
    out_path = ROOT / "Results" / "cohort_class_prevalence.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    print("\nBranch counts across cohort:")
    print(df["balancer_branch"].value_counts())
    print(f"\nMean per-class prevalence (cohort):")
    print(df[["pct_Other", "pct_Ruminating", "pct_Eating"]].mean())
    print(f"Std per-class prevalence (cohort):")
    print(df[["pct_Other", "pct_Ruminating", "pct_Eating"]].std())


if __name__ == "__main__":
    main()
