"""scripts/aggregate_18animals.py — Cohort-level aggregation and Wilcoxon tests.

Reads the per-animal × per-model rows in
`Results/comparison_report_all_animals.csv` (produced by
`consolidate_results.py`) and writes:

  * `Results/aggregate_neural_summary.csv`       — neural-net cohort means
  * `Results/aggregate_classical_summary.csv`    — KF / EKF-HMM cohort means
  * `Results/wilcoxon_aggregate_18animals.csv`   — STA-LSTM-H vs each baseline,
                                                   paired two-sided exact
                                                   Wilcoxon signed-rank test
                                                   across the 18 animals.

The aggregate Wilcoxon CSV that previously sat in the repo was generated
inline by an auto-loop and had no committed generator script. This file
closes that reproducibility gap with an explicit, single-call entry point.

Usage:
    python scripts/aggregate_18animals.py
    python scripts/aggregate_18animals.py --check      # diff against the
                                                       # currently committed
                                                       # CSVs (no overwrite)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats
from statsmodels.stats.multitest import multipletests

NEURAL = ("STA-LSTM-H", "STA-LSTM", "LSTM", "GRU", "Transformer")
CLASSICAL = ("KF", "EKF-HMM")
BASELINES = ("STA-LSTM", "LSTM", "GRU", "Transformer")


def aggregate_neural(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean/std across the 18 animal-means for the NN model set."""
    sub = df[df["model"].isin(NEURAL)].copy()
    out = (
        sub.groupby("model")
        .agg(
            acc_mean=("accuracy_mean", "mean"),
            acc_std=("accuracy_mean", "std"),
            f1_mean=("f1_mean", "mean"),
            f1_std=("f1_mean", "std"),
            rmse_mean=("rmse_mean", "mean"),
            rmse_std=("rmse_mean", "std"),
            lat_mean=("latency_ms_mean", "mean"),
            composite_mean=("composite_score", "mean"),
        )
        .reset_index()
    )
    return out.sort_values("model").reset_index(drop=True)


def aggregate_classical(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean/std for the classical-filter set (KF, EKF-HMM)."""
    sub = df[df["model"].isin(CLASSICAL)].copy()
    out = (
        sub.groupby("model")
        .agg(
            rmse_mean=("rmse_mean", "mean"),
            rmse_std=("rmse_mean", "std"),
            lat_mean=("latency_ms_mean", "mean"),
        )
        .reset_index()
    )
    return out.sort_values("model").reset_index(drop=True)


def _wilcoxon_two_sided(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired two-sided exact Wilcoxon signed-rank test.

    Returns (W_statistic, p_value). Drops pairs with NaN. If fewer than 2
    finite pairs remain, or all differences are zero, returns (nan, nan).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.all(a[mask] == b[mask]):
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = spstats.wilcoxon(
            a[mask], b[mask], alternative="two-sided", zero_method="wilcox",
        )
    return float(stat), float(p)


def wilcoxon_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Paired two-sided Wilcoxon, STA-LSTM-H vs each baseline, n=18 animals.

    Each animal contributes a single paired observation given by its mean over
    the 5 folds (already the row in `comparison_report_all_animals.csv`).
    """
    pivot_acc = df.pivot(index="animal", columns="model", values="accuracy_mean")
    pivot_f1 = df.pivot(index="animal", columns="model", values="f1_mean")
    pivot_rmse = df.pivot(index="animal", columns="model", values="rmse_mean")

    rows = []
    for baseline in BASELINES:
        if baseline not in pivot_acc.columns:
            continue
        acc_W, acc_p = _wilcoxon_two_sided(
            pivot_acc["STA-LSTM-H"].to_numpy(), pivot_acc[baseline].to_numpy()
        )
        f1_W, f1_p = _wilcoxon_two_sided(
            pivot_f1["STA-LSTM-H"].to_numpy(), pivot_f1[baseline].to_numpy()
        )
        rmse_W, rmse_p = _wilcoxon_two_sided(
            pivot_rmse["STA-LSTM-H"].to_numpy(), pivot_rmse[baseline].to_numpy()
        )
        n = int(
            np.isfinite(pivot_acc["STA-LSTM-H"]).sum()
            & np.isfinite(pivot_acc[baseline]).sum()
        )
        rows.append({
            "comparison": f"STA-LSTM-H_vs_{baseline}",
            "acc_W": acc_W,
            "acc_p": acc_p,
            "f1_W": f1_W,
            "f1_p": f1_p,
            "rmse_W": rmse_W,
            "rmse_p": rmse_p,
            "n_animals": n,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        # Holm correction across the full battery: 4 baselines × 3 metrics = 12
        # simultaneous tests against STA-LSTM-H. Treat NaN raw p as 1.0 so it is
        # never flagged significant (rather than dropping it from the family).
        for col in ("acc_p", "f1_p", "rmse_p"):
            out[col + "_holm"] = float("nan")
        flat = pd.concat([
            out[["comparison"]].assign(metric="acc",  p=out["acc_p"]),
            out[["comparison"]].assign(metric="f1",   p=out["f1_p"]),
            out[["comparison"]].assign(metric="rmse", p=out["rmse_p"]),
        ], ignore_index=True)
        flat["p_filled"] = flat["p"].fillna(1.0).astype(float)
        _, p_holm, _, _ = multipletests(flat["p_filled"].to_numpy(),
                                        method="holm")
        flat["p_holm"] = p_holm
        # Map back to per-row columns.
        for _, row in flat.iterrows():
            mask = out["comparison"] == row["comparison"]
            out.loc[mask, f"{row['metric']}_p_holm"] = row["p_holm"]
    return out


def _maybe_diff(label: str, fresh: pd.DataFrame, committed_path: Path) -> None:
    """Print a brief mismatch report between freshly computed and committed CSVs."""
    if not committed_path.exists():
        print(f"  [{label}] no committed file at {committed_path} (first run).")
        return
    old = pd.read_csv(committed_path)
    try:
        merged = fresh.merge(old, on=fresh.columns[0].split(",")[0]
                             if isinstance(fresh.columns[0], str) and "," in fresh.columns[0]
                             else fresh.columns[0],
                             suffixes=("_new", "_old"))
    except Exception:
        merged = None
    if merged is None or merged.empty:
        print(f"  [{label}] schema/key mismatch with {committed_path.name}; "
              f"manual diff required.")
        print(f"           fresh columns: {list(fresh.columns)}")
        print(f"           old   columns: {list(old.columns)}")
        return
    n_diffs = 0
    for col in merged.columns:
        if col.endswith("_new"):
            base = col[:-4]
            old_col = f"{base}_old"
            if old_col not in merged.columns:
                continue
            try:
                diff = (merged[col].astype(float) - merged[old_col].astype(float)).abs()
                bad = diff > 1e-6
                if bad.any():
                    n_diffs += int(bad.sum())
                    print(f"  [{label}] column {base}: "
                          f"{int(bad.sum())} mismatches (max abs Δ = "
                          f"{diff.max():.6e})")
            except (ValueError, TypeError):
                continue
    if n_diffs == 0:
        print(f"  [{label}] all values match committed CSV to 1e-6.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("Results/comparison_report_all_animals.csv"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("Results"))
    p.add_argument(
        "--check",
        action="store_true",
        help="Diff freshly computed values against the committed CSVs without overwriting.",
    )
    args = p.parse_args()

    print(f"Reading {args.input} …")
    df = pd.read_csv(args.input)
    df["animal"] = df["animal"].astype(str).str.zfill(2)

    neural = aggregate_neural(df)
    classical = aggregate_classical(df)
    wilcoxon = wilcoxon_aggregate(df)

    out_paths = {
        "neural":    args.out_dir / "aggregate_neural_summary.csv",
        "classical": args.out_dir / "aggregate_classical_summary.csv",
        "wilcoxon":  args.out_dir / "wilcoxon_aggregate_18animals.csv",
    }

    if args.check:
        print("\nDiff against committed CSVs:")
        _maybe_diff("neural",    neural,    out_paths["neural"])
        _maybe_diff("classical", classical, out_paths["classical"])
        _maybe_diff("wilcoxon",  wilcoxon,  out_paths["wilcoxon"])
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    neural.to_csv(out_paths["neural"], index=False)
    classical.to_csv(out_paths["classical"], index=False)
    wilcoxon.to_csv(out_paths["wilcoxon"], index=False)
    print(f"Wrote {out_paths['neural']}")
    print(f"Wrote {out_paths['classical']}")
    print(f"Wrote {out_paths['wilcoxon']}")

    print("\nAggregate neural summary:")
    print(neural.to_string(index=False))
    print("\nAggregate classical summary:")
    print(classical.to_string(index=False))
    print("\nWilcoxon (two-sided exact):")
    print(wilcoxon.to_string(index=False))


if __name__ == "__main__":
    main()
