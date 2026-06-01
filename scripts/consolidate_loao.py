"""scripts/consolidate_loao.py — Aggregate LOAO per-held-out-animal results.

Walks `Results/python_pipeline_loao/animal-NN/loao_report.csv` for
NN ∈ 01..18 and produces:

  * `Results/comparison_report_loao.csv` — long table, one row per
    (model, held_out_animal) pair. Schema mirrors
    `comparison_report_all_animals.csv` so downstream aggregation can reuse
    the same Wilcoxon scaffold.
  * `Results/aggregate_loao_summary.csv` — per-model mean ± std across the
    18 held-out animals. Comparable to `aggregate_neural_summary.csv` for
    the headline 5-fold protocol.
  * `Results/wilcoxon_loao_18animals.csv` — paired two-sided Wilcoxon
    signed-rank, STA-LSTM-H vs each baseline across 18 LOAO splits, with
    Holm-corrected p-values for the family of 4 baselines × 3 metrics = 12
    simultaneous tests.

Usage:
    python scripts/consolidate_loao.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats

try:
    from statsmodels.stats.multitest import multipletests
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"statsmodels is required for Holm correction: {exc}. "
        "Install with: pip install statsmodels"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
LOAO_DIR = ROOT / "Results" / "python_pipeline_loao"
OUT_DIR = ROOT / "Results"

NEURAL = ("STA-LSTM-H", "STA-LSTM", "LSTM", "GRU", "Transformer", "1D-CNN")
CLASSICAL = ("KF", "EKF-HMM")
TABULAR = ("XGBoost",)
BASELINES = ("STA-LSTM", "LSTM", "GRU", "Transformer")


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    for animal_dir in sorted(LOAO_DIR.glob("animal-*")):
        animal_id = animal_dir.name.replace("animal-", "")
        report = animal_dir / "loao_report.csv"
        if not report.exists():
            print(f"  [warn] {report} missing; skipping animal {animal_id}.")
            continue
        df = pd.read_csv(report)
        for _, row in df.iterrows():
            rows.append({
                "held_out_animal": animal_id,
                "model":           str(row["model"]),
                "accuracy_mean":   float(row.get("accuracy_mean", float("nan"))),
                "f1_mean":         float(row.get("f1_mean", float("nan"))),
                "rmse_mean":       float(row.get("rmse_mean", float("nan"))),
                "latency_ms_mean": float(row.get("latency_ms_mean", float("nan"))),
                "memory_mb_mean":  float(row.get("memory_mb_mean", float("nan"))),
            })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("model")
        .agg(
            n_animals      = ("held_out_animal", "nunique"),
            acc_mean       = ("accuracy_mean", "mean"),
            acc_std        = ("accuracy_mean", "std"),
            f1_mean        = ("f1_mean", "mean"),
            f1_std         = ("f1_mean", "std"),
            rmse_mean      = ("rmse_mean", "mean"),
            rmse_std       = ("rmse_mean", "std"),
            latency_ms_mean= ("latency_ms_mean", "mean"),
        )
        .reset_index()
        .sort_values("acc_mean", ascending=False)
        .reset_index(drop=True)
    )
    return out


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
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


def wilcoxon_battery(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot(index="held_out_animal", columns="model",
                     values=["accuracy_mean", "f1_mean", "rmse_mean"])
    rows: list[dict] = []
    for baseline in BASELINES:
        if ("accuracy_mean", baseline) not in pivot.columns:
            continue
        if ("accuracy_mean", "STA-LSTM-H") not in pivot.columns:
            continue
        a_acc = pivot[("accuracy_mean", "STA-LSTM-H")].to_numpy()
        b_acc = pivot[("accuracy_mean", baseline)].to_numpy()
        a_f1 = pivot[("f1_mean", "STA-LSTM-H")].to_numpy()
        b_f1 = pivot[("f1_mean", baseline)].to_numpy()
        a_rm = pivot[("rmse_mean", "STA-LSTM-H")].to_numpy()
        b_rm = pivot[("rmse_mean", baseline)].to_numpy()
        acc_W, acc_p = _paired_wilcoxon(a_acc, b_acc)
        f1_W, f1_p = _paired_wilcoxon(a_f1, b_f1)
        rmse_W, rmse_p = _paired_wilcoxon(a_rm, b_rm)
        rows.append({
            "comparison":  f"STA-LSTM-H_vs_{baseline}",
            "acc_W":       acc_W, "acc_p": acc_p,
            "f1_W":        f1_W,  "f1_p":  f1_p,
            "rmse_W":      rmse_W, "rmse_p": rmse_p,
            "n_animals":   int(np.isfinite(a_acc).sum()
                              & np.isfinite(b_acc).sum()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    flat = pd.concat([
        out[["comparison"]].assign(metric="acc",  p=out["acc_p"]),
        out[["comparison"]].assign(metric="f1",   p=out["f1_p"]),
        out[["comparison"]].assign(metric="rmse", p=out["rmse_p"]),
    ], ignore_index=True)
    flat["p_filled"] = flat["p"].fillna(1.0).astype(float)
    _, p_holm, _, _ = multipletests(flat["p_filled"].to_numpy(), method="holm")
    flat["p_holm"] = p_holm
    for col in ("acc_p", "f1_p", "rmse_p"):
        out[col + "_holm"] = float("nan")
    for _, row in flat.iterrows():
        mask = out["comparison"] == row["comparison"]
        out.loc[mask, f"{row['metric']}_p_holm"] = row["p_holm"]
    return out


def main() -> None:
    if not LOAO_DIR.exists():
        print(f"No LOAO results found at {LOAO_DIR}.")
        print("Run scripts/submit_loao.sh first (or scripts/run_loao_animal.py "
              "locally), then re-run this consolidation.")
        sys.exit(0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long = collect()
    if long.empty:
        print(f"No per-animal loao_report.csv files found under {LOAO_DIR}.")
        sys.exit(0)

    long_path = OUT_DIR / "comparison_report_loao.csv"
    long.to_csv(long_path, index=False)
    print(f"Wrote {long_path}  ({len(long)} rows)")

    summary = aggregate(long)
    summary_path = OUT_DIR / "aggregate_loao_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print("\nLOAO per-model aggregate (mean ± std across 18 held-out animals):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    wilc = wilcoxon_battery(long)
    if not wilc.empty:
        wilc_path = OUT_DIR / "wilcoxon_loao_18animals.csv"
        wilc.to_csv(wilc_path, index=False)
        print(f"\nWrote {wilc_path}")
        print("\nLOAO Wilcoxon battery (Holm-corrected; 4 baselines × 3 metrics):")
        print(wilc.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
