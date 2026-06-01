"""scripts/consolidate_seed_audit.py — Assemble the multi-seed audit table.

The seed-audit Slurm array (`scripts/submit_seed_audit.sh`, job 155489) re-runs
protocol P1 on three representative animals (1, 9, 17) across five seeds
{7, 42, 101, 1729, 2026} and writes
`Results/python_pipeline_seedaudit/animal-NN/seed_audit_summary.csv`
(per-model accuracy/f1/rmse mean+std across the five seeds).

This script collates the three per-animal summaries into the layout of
manuscript `tab:seed_audit`:

    Architecture | Animal 01 | Animal 09 | Animal 17 | Mean across-seed std

where each animal column is the across-seed mean accuracy and the last column
is the across-seed accuracy std averaged over the three animals.

Writes `Results/aggregate_seed_audit_summary.csv` and prints LaTeX rows.

Usage:
    python scripts/consolidate_seed_audit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SA_DIR = ROOT / "Results" / "python_pipeline_seedaudit"
OUT = ROOT / "Results" / "aggregate_seed_audit_summary.csv"

ANIMALS = [1, 9, 17]
# manuscript model-key -> display name (table row order)
MODEL_DISPLAY = {
    "lstm_h": "STA-LSTM-H",
    "sta_lstm": "STA-LSTM",
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
}


def main() -> None:
    acc_mean: dict[str, dict[int, float]] = {}
    acc_std: dict[str, dict[int, float]] = {}
    for a in ANIMALS:
        f = SA_DIR / f"animal-{a:02d}" / "seed_audit_summary.csv"
        if not f.exists():
            raise SystemExit(f"missing seed-audit summary: {f}")
        df = pd.read_csv(f, header=[0, 1], index_col=0)
        for key in MODEL_DISPLAY:
            acc_mean.setdefault(key, {})[a] = float(df.loc[key, ("accuracy", "mean")])
            acc_std.setdefault(key, {})[a] = float(df.loc[key, ("accuracy", "std")])

    rows = []
    for key, disp in MODEL_DISPLAY.items():
        mean_std = float(np.mean([acc_std[key][a] for a in ANIMALS]))
        rows.append(
            dict(
                architecture=disp,
                animal_01=acc_mean[key][1],
                animal_09=acc_mean[key][9],
                animal_17=acc_mean[key][17],
                mean_across_seed_std=mean_std,
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print(f"Multi-seed audit (animals {ANIMALS}, 5 seeds) -> "
          f"{OUT.relative_to(ROOT)}\n")
    for r in rows:
        print(
            f"{r['architecture']:<12} & ${r['animal_01']:.4f}$ & "
            f"${r['animal_09']:.4f}$ & ${r['animal_17']:.4f}$ & "
            f"${r['mean_across_seed_std']:.4f}$ \\\\"
        )
    worst = max(r["mean_across_seed_std"] for r in rows)
    print(f"\nmax mean across-seed std over the 5 architectures: {worst:.4f}")


if __name__ == "__main__":
    main()
