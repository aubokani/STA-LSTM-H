"""scripts/consolidate_loao_ablation.py — Assemble the LOAO-ablation 6x3 grid.

The LOAO-ablation Slurm array (`scripts/submit_loao_ablation.sh`, jobs 154985
+ 155026) writes `Results/python_pipeline_loao_ablation/animal-NN/loao_report.csv`
for NN in 01..18. Its registry is *asymmetric* by design — it only computes the
cells that the LOAO-headline run did not already produce:

  * accel-only           : STA-LSTM-H, STA-LSTM            (2 cells)
  * +behavior_prev_lag1  : all six architectures           (6 cells)
  * +behavior_prev       : LSTM, GRU, Transformer, 1D-CNN  (4 cells)

The remaining six cells of the 6x3 grid are exactly the LOAO-headline numbers
(`Results/aggregate_loao_summary.csv`), because in the headline registry the
plain baselines see accel-only and the STA family sees same-step behavior_prev:

  * accel-only           : LSTM, GRU, Transformer, 1D-CNN  (from LOAO-headline)
  * +behavior_prev       : STA-LSTM-H, STA-LSTM            (from LOAO-headline)

This script joins the two LOAO experiments into the full 6x3 cohort-mean grid
(mean +/- std across the 18 held-out animals) and writes:

  * Results/aggregate_loao_ablation_summary.csv  — long form, one row per
    (architecture, input_set) with acc/f1 mean+/-std and the source experiment.

It also prints a LaTeX-ready accuracy grid for `tab:loao_ablation`.

Usage:
    python scripts/consolidate_loao_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABL_DIR = ROOT / "Results" / "python_pipeline_loao_ablation"
HEADLINE_AGG = ROOT / "Results" / "aggregate_loao_summary.csv"
OUT = ROOT / "Results" / "aggregate_loao_ablation_summary.csv"

ARCH_ORDER = ["STA-LSTM-H", "STA-LSTM", "LSTM", "GRU", "Transformer", "1D-CNN"]
INPUT_ORDER = ["accel", "bp_lag1", "bp"]


def parse_label(label: str) -> tuple[str, str]:
    """`STA-LSTM-H + bp_lag1` -> ('STA-LSTM-H', 'bp_lag1');
       `STA-LSTM (accel)`     -> ('STA-LSTM',   'accel')."""
    label = label.strip()
    if label.endswith("(accel)"):
        return label[: -len("(accel)")].strip(), "accel"
    if " + bp_lag1" in label:
        return label.split(" + ")[0].strip(), "bp_lag1"
    if " + bp" in label:
        return label.split(" + ")[0].strip(), "bp"
    raise ValueError(f"unrecognised LOAO-ablation label: {label!r}")


def main() -> None:
    # ---- 1. Per-cell cohort stats from the LOAO-ablation array ----------
    per_animal: dict[tuple[str, str], list[tuple[float, float]]] = {}
    n_animals = 0
    for nn in range(1, 19):
        rep = ABL_DIR / f"animal-{nn:02d}" / "loao_report.csv"
        if not rep.exists():
            raise SystemExit(f"missing LOAO-ablation report: {rep}")
        df = pd.read_csv(rep)
        n_animals += 1
        for _, row in df.iterrows():
            arch, inp = parse_label(row["model"])
            per_animal.setdefault((arch, inp), []).append(
                (float(row["accuracy_mean"]), float(row["f1_mean"]))
            )

    rows = []
    for (arch, inp), vals in per_animal.items():
        acc = np.array([v[0] for v in vals])
        f1 = np.array([v[1] for v in vals])
        rows.append(
            dict(
                architecture=arch,
                input_set=inp,
                acc_mean=acc.mean(),
                acc_std=acc.std(ddof=1),
                f1_mean=f1.mean(),
                f1_std=f1.std(ddof=1),
                n_animals=len(vals),
                source="loao_ablation",
            )
        )

    # ---- 2. Complementary six cells from the LOAO-headline aggregate ----
    hl = pd.read_csv(HEADLINE_AGG).set_index("model")
    complementary = (
        [("accel", a) for a in ("LSTM", "GRU", "Transformer", "1D-CNN")]
        + [("bp", a) for a in ("STA-LSTM-H", "STA-LSTM")]
    )
    for inp, arch in complementary:
        r = hl.loc[arch]
        rows.append(
            dict(
                architecture=arch,
                input_set=inp,
                acc_mean=float(r["acc_mean"]),
                acc_std=float(r["acc_std"]),
                f1_mean=float(r["f1_mean"]),
                f1_std=float(r["f1_std"]),
                n_animals=int(r["n_animals"]),
                source="loao_headline",
            )
        )

    summary = pd.DataFrame(rows)
    summary["architecture"] = pd.Categorical(
        summary["architecture"], ARCH_ORDER, ordered=True
    )
    summary["input_set"] = pd.Categorical(
        summary["input_set"], INPUT_ORDER, ordered=True
    )
    summary = summary.sort_values(["architecture", "input_set"]).reset_index(
        drop=True
    )
    summary.to_csv(OUT, index=False)

    # ---- 3. LaTeX-ready accuracy grid ----------------------------------
    grid = summary.pivot(
        index="architecture", columns="input_set", values="acc_mean"
    )
    grid_std = summary.pivot(
        index="architecture", columns="input_set", values="acc_std"
    )
    print(f"\nLOAO-ablation 6x3 cohort-mean accuracy (n={n_animals} held-out "
          f"animals)\nwritten to {OUT.relative_to(ROOT)}\n")
    for arch in ARCH_ORDER:
        cells = " & ".join(
            f"${grid.loc[arch, c]:.4f}\\pm{grid_std.loc[arch, c]:.4f}$"
            for c in INPUT_ORDER
        )
        print(f"{arch:<12} & {cells} \\\\")
    print()
    src = summary.pivot(
        index="architecture", columns="input_set", values="source"
    )
    print("cell provenance (which LOAO experiment each cell came from):")
    print(src.to_string())


if __name__ == "__main__":
    main()
