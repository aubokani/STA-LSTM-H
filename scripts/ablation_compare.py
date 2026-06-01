"""scripts/ablation_compare.py — Decisive 3-way ablation analysis.

Combines headline (Results/python_pipeline) and ablation (Results/python_pipeline_ablation)
per-animal CSVs into one long table, then runs the comparisons specified in the project memory:
  - lstm_h (headline, same-step bp)  vs  lstm_h_accel  vs  lstm_h_lag1   [decisive]
  - lstm_h_lag1 vs lstm_lag1, gru_lag1, transformer_lag1                  [clean architecture]
  - lstm_h (headline) vs lstm_h_lag1                                       [same-step vs lagged gap]

Wilcoxon signed-rank across the 18 paired per-animal accuracy_mean values.
Writes:
  Results/ablation_3way.csv
  Results/ablation_wilcoxon.csv
  Results/aggregate_neural_summary_ablation.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
HEADLINE = ROOT / "Results" / "comparison_report_all_animals.csv"
ABLATION = ROOT / "Results" / "comparison_report_all_animals_ablation.csv"
OUT_DIR  = ROOT / "Results"

# Headline display labels → ablation key (for joining with the 3-way comparison).
HEADLINE_LABELS = ("STA-LSTM-H", "STA-LSTM", "LSTM", "GRU", "Transformer")

# Map an ablation CSV `model` value to a (architecture, input_set) tuple.
ARCH_FROM = {
    "STA-LSTM-H": "STA-LSTM-H",
    "STA-LSTM":   "STA-LSTM",
    "LSTM":       "LSTM",
    "GRU":        "GRU",
    "Transformer":"Transformer",
}

def parse_ablation_label(label: str) -> tuple[str, str]:
    """`STA-LSTM-H + bp_lag1` -> ('STA-LSTM-H', 'bp_lag1')
       `STA-LSTM (accel)`     -> ('STA-LSTM',   'accel')."""
    if " + " in label:
        arch, inputs = label.split(" + ", 1)
        return arch.strip(), inputs.strip()
    if "(" in label:
        arch, inputs = label.split("(", 1)
        return arch.strip(), inputs.strip(") ")
    return label.strip(), "bp"  # headline default


def load_long() -> pd.DataFrame:
    head = pd.read_csv(HEADLINE)
    abl  = pd.read_csv(ABLATION)

    head_neu = head[head["model"].isin(HEADLINE_LABELS)].copy()
    head_neu["arch"]   = head_neu["model"]
    # Per trainer.MODEL_FEATURE_MAP: STA-LSTM-H/STA-LSTM use 15-feature input (with same-step behavior_prev),
    # LSTM/GRU/Transformer use 14-feature accel-only.
    head_neu["inputs"] = head_neu["arch"].map(
        {"STA-LSTM-H": "bp", "STA-LSTM": "bp",
         "LSTM": "accel",   "GRU": "accel",     "Transformer": "accel"}
    )
    head_neu["origin"] = "headline"

    abl_parsed = abl["model"].apply(parse_ablation_label)
    abl["arch"]   = abl_parsed.apply(lambda t: t[0])
    abl["inputs"] = abl_parsed.apply(lambda t: t[1])
    abl["origin"] = "ablation"

    cols = ["animal", "arch", "inputs", "origin",
            "accuracy_mean", "f1_mean", "rmse_mean", "latency_ms_mean", "composite_score"]
    long = pd.concat([head_neu[cols], abl[cols]], ignore_index=True)
    long["animal"] = long["animal"].astype(str).str.zfill(2)
    return long


def cohort_summary(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby(["arch", "inputs"])
    out = g.agg(
        n_animals  = ("animal", "nunique"),
        acc_mean   = ("accuracy_mean", "mean"),
        acc_std    = ("accuracy_mean", "std"),
        f1_mean    = ("f1_mean",       "mean"),
        f1_std     = ("f1_mean",       "std"),
        rmse_mean  = ("rmse_mean",     "mean"),
        rmse_std   = ("rmse_mean",     "std"),
        lat_mean   = ("latency_ms_mean","mean"),
    ).reset_index()
    return out.sort_values(["inputs", "acc_mean"], ascending=[True, False])


def paired_wilcoxon(long: pd.DataFrame,
                    a: tuple[str, str],
                    b: tuple[str, str]) -> dict:
    """Paired Wilcoxon signed-rank on per-animal accuracy_mean. a, b = (arch, inputs)."""
    pa = long[(long["arch"] == a[0]) & (long["inputs"] == a[1])][["animal", "accuracy_mean"]]
    pb = long[(long["arch"] == b[0]) & (long["inputs"] == b[1])][["animal", "accuracy_mean"]]
    m  = pa.merge(pb, on="animal", suffixes=("_a", "_b"))
    if len(m) < 6:
        return {"n": len(m), "stat": np.nan, "p": np.nan,
                "mean_a": pa["accuracy_mean"].mean(), "mean_b": pb["accuracy_mean"].mean(),
                "delta_mean": np.nan}
    diff = m["accuracy_mean_a"] - m["accuracy_mean_b"]
    if (diff == 0).all():
        stat, p = 0.0, 1.0
    else:
        stat, p = wilcoxon(m["accuracy_mean_a"], m["accuracy_mean_b"], zero_method="wilcox")
    return {"n": int(len(m)), "stat": float(stat), "p": float(p),
            "mean_a": float(m["accuracy_mean_a"].mean()),
            "mean_b": float(m["accuracy_mean_b"].mean()),
            "delta_mean": float(diff.mean())}


def main() -> None:
    long = load_long()
    summary = cohort_summary(long)
    summary.to_csv(OUT_DIR / "aggregate_neural_summary_ablation.csv", index=False)
    print("=== Cohort summary (mean ± std across 18 animals) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- Decisive: full 5x3 factorial across architectures and input sets ----
    decisive = []
    for arch in ("STA-LSTM-H", "STA-LSTM", "LSTM", "GRU", "Transformer"):
        for inp in ("accel", "bp_lag1", "bp"):
            sl = long[(long["arch"] == arch) & (long["inputs"] == inp)]
            if sl.empty:
                continue
            decisive.append({
                "arch": arch, "inputs": inp,
                "n_animals": int(sl["animal"].nunique()),
                "acc_mean":  float(sl["accuracy_mean"].mean()),
                "acc_std":   float(sl["accuracy_mean"].std()),
                "f1_mean":   float(sl["f1_mean"].mean()),
                "f1_std":    float(sl["f1_mean"].std()),
                "rmse_mean": float(sl["rmse_mean"].mean()),
                "rmse_std":  float(sl["rmse_mean"].std()),
            })
    pd.DataFrame(decisive).to_csv(OUT_DIR / "ablation_3way.csv", index=False)
    print("\n=== Decisive 5x3 factorial (architecture x input set, mean across 18 animals) ===")
    print(pd.DataFrame(decisive).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- Wilcoxon comparisons ----
    pairs = [
        # the three decisive comparisons named in project memory
        ("lstm_h_headline_vs_lstm_h_accel",   ("STA-LSTM-H", "bp"),     ("STA-LSTM-H", "accel")),
        ("lstm_h_headline_vs_lstm_h_lag1",    ("STA-LSTM-H", "bp"),     ("STA-LSTM-H", "bp_lag1")),
        ("lstm_h_lag1_vs_lstm_h_accel",       ("STA-LSTM-H", "bp_lag1"),("STA-LSTM-H", "accel")),

        # clean architectural comparisons with input set held fixed at bp_lag1
        ("lstm_h_lag1_vs_lstm_lag1",          ("STA-LSTM-H", "bp_lag1"),("LSTM",       "bp_lag1")),
        ("lstm_h_lag1_vs_gru_lag1",           ("STA-LSTM-H", "bp_lag1"),("GRU",        "bp_lag1")),
        ("lstm_h_lag1_vs_transformer_lag1",   ("STA-LSTM-H", "bp_lag1"),("Transformer","bp_lag1")),
        ("lstm_h_lag1_vs_sta_lstm_lag1",      ("STA-LSTM-H", "bp_lag1"),("STA-LSTM",   "bp_lag1")),

        # leakage check: same-step bp baselines should match STA-LSTM-H headline
        ("lstm_h_headline_vs_lstm_bp",        ("STA-LSTM-H", "bp"),     ("LSTM",       "bp")),
        ("lstm_h_headline_vs_gru_bp",         ("STA-LSTM-H", "bp"),     ("GRU",        "bp")),
        ("lstm_h_headline_vs_transformer_bp", ("STA-LSTM-H", "bp"),     ("Transformer","bp")),

        # vs original headline baselines (14-feature accel-only)
        ("lstm_h_headline_vs_lstm_headline",        ("STA-LSTM-H", "bp"), ("LSTM",       "accel")),
        ("lstm_h_headline_vs_gru_headline",         ("STA-LSTM-H", "bp"), ("GRU",        "accel")),
        ("lstm_h_headline_vs_transformer_headline", ("STA-LSTM-H", "bp"), ("Transformer","accel")),
    ]
    rows = []
    for name, a, b in pairs:
        r = paired_wilcoxon(long, a, b)
        rows.append({"comparison": name, "a": f"{a[0]}/{a[1]}", "b": f"{b[0]}/{b[1]}", **r})
    wdf = pd.DataFrame(rows)
    # Holm correction across the ablation Wilcoxon family (typically 13 tests).
    # We keep the raw `p` column for reference and add `p_holm` for the
    # corrected value, treating any NaN raw p as 1.0 so it is never flagged
    # significant rather than dropped from the family.
    if not wdf.empty:
        p_filled = wdf["p"].fillna(1.0).astype(float).to_numpy()
        _, p_holm, _, _ = multipletests(p_filled, method="holm")
        wdf["p_holm"] = p_holm
    wdf.to_csv(OUT_DIR / "ablation_wilcoxon.csv", index=False)

    print("\n=== Paired Wilcoxon (n=18 animals, accuracy_mean; Holm-corrected) ===")
    print(wdf.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
