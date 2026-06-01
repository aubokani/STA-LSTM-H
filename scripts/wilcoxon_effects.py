"""scripts/wilcoxon_effects.py — reproducible Wilcoxon + rank-biserial battery.

Recomputes, from the per-animal paired result CSVs, the three Holm-corrected
paired two-sided Wilcoxon families reported in the manuscript, and — crucially
— the matched-pairs rank-biserial effect size r_rb that the manuscript tables
print. Prior to this script r_rb was hand-entered with no generator and did not
reconcile with the stored W statistic (Q1 stats-reviewer blocker S1/S2).

Convention: for every comparison, a positive r_rb means "STA-LSTM-H (or, for
the within-architecture row, the lagged config) is better on this metric in
more animals". Accuracy/F1 are higher-better; RMSE is lower-better, so the
signed difference is oriented per metric before ranking.

    r_rb = (W+ - W-) / (n (n+1)/2) = 4 W+ / (n (n+1)) - 1,
           n = number of non-zero paired differences  (Eq. rbs of the manuscript)

Families (Holm-corrected within each, matching Section meth-eval):
  F1  headline within-animal P1   — Results/comparison_report_all_animals.csv
  F2  factorial input-set ablation — Results/comparison_report_all_animals_ablation.csv
                                     (+ same-step STA-LSTM-H from the F1 CSV)
  F3  leave-one-animal-out         — Results/comparison_report_loao.csv

Writes <stem>_effects.csv next to each source family and prints LaTeX rows.

Usage:
    python scripts/wilcoxon_effects.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "Results"
METRICS = [("accuracy_mean", "acc", True),
           ("f1_mean", "f1", True),
           ("rmse_mean", "rmse", False)]


def paired(df, model_col, key_col, a_label, b_label):
    """Return (vec_a, vec_b) aligned on key_col for two model labels."""
    a = df[df[model_col] == a_label].set_index(key_col)
    b = df[df[model_col] == b_label].set_index(key_col)
    keys = sorted(set(a.index) & set(b.index))
    return a.loc[keys], b.loc[keys]


def one(a_vals, b_vals, higher_better):
    """Wilcoxon two-sided p, W+, and rank-biserial for one metric."""
    a = np.asarray(a_vals, float)
    b = np.asarray(b_vals, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    d = (a - b) if higher_better else (b - a)        # positive => "a" better
    nz = d[d != 0]
    n = nz.size
    if n == 0:
        return dict(p=1.0, W_plus=0.0, r_rb=0.0, n=0)
    ranks = rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    T = float(ranks.sum())                           # = n(n+1)/2
    # matched-pairs rank-biserial (Kerby 2014): (W+ - W-)/T, W-=T-W+
    r_rb = (2.0 * w_plus - T) / T
    try:
        p = float(spstats.wilcoxon(d, alternative="two-sided",
                                   zero_method="wilcox").pvalue)
    except ValueError:                                # all-zero diffs
        p = 1.0
    return dict(p=p, W_plus=w_plus, r_rb=r_rb, n=n)


def run_family(name, rows, out_csv):
    recs = []
    for label, res in rows:
        recs.append(dict(comparison=label, **res))
    fam = pd.DataFrame(recs)
    fam["p_holm"] = multipletests(fam["p"].values, method="holm")[1]
    fam.to_csv(out_csv, index=False)
    print(f"\n===== {name}  (Holm over {len(fam)} tests)  -> "
          f"{out_csv.relative_to(ROOT)} =====")
    print(fam.to_string(index=False,
          formatters={"p": lambda x: f"{x:.3e}",
                      "p_holm": lambda x: f"{x:.3e}",
                      "r_rb": lambda x: f"{x:+.2f}",
                      "W_plus": lambda x: f"{x:.1f}"}))
    return fam


def fmt_p(p):
    if p >= 0.9995:
        return "1.000"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def latex_block(fam, label_map):
    for _, r in fam.iterrows():
        disp = label_map.get(r["comparison"], r["comparison"])
        accs = fam[fam.comparison == r["comparison"]]
    # printed separately below per family


def main():
    # ---------------- F1 : headline within-animal -----------------------
    f1 = pd.read_csv(R / "comparison_report_all_animals.csv")
    f1 = f1[f1["model"] != "model"]
    for c in ("accuracy_mean", "f1_mean", "rmse_mean"):
        f1[c] = pd.to_numeric(f1[c], errors="coerce")
    rows = []
    for base in ["STA-LSTM", "LSTM", "GRU", "Transformer"]:
        A, B = paired(f1, "model", "animal", "STA-LSTM-H", base)
        for col, tag, hb in METRICS:
            rows.append((f"STA-LSTM-H_vs_{base}|{tag}",
                         one(A[col], B[col], hb)))
    famF1 = run_family("F1 headline within-animal", rows,
                       R / "wilcoxon_aggregate_18animals_effects.csv")

    # ---------------- F2 : factorial ablation ---------------------------
    ab = pd.read_csv(R / "comparison_report_all_animals_ablation.csv")
    ab = ab[ab["model"] != "model"]
    for c in ("accuracy_mean", "f1_mean", "rmse_mean"):
        ab[c] = pd.to_numeric(ab[c], errors="coerce")
    # same-step STA-LSTM-H lives in the headline CSV (headline registry)
    ss = f1[f1["model"] == "STA-LSTM-H"][["animal", "accuracy_mean",
                                          "f1_mean", "rmse_mean"]].copy()
    ss["model"] = "STA-LSTM-H + bp (same-step)"
    abx = pd.concat([ab, ss], ignore_index=True)
    rows = []
    for base in ["STA-LSTM + bp_lag1", "LSTM + bp_lag1",
                 "GRU + bp_lag1", "Transformer + bp_lag1"]:
        A, B = paired(abx, "model", "animal", "STA-LSTM-H + bp_lag1", base)
        for col, tag, hb in METRICS:
            rows.append((f"STA-LSTM-H+bp_lag1_vs_{base}|{tag}",
                         one(A[col], B[col], hb)))
    A, B = paired(abx, "model", "animal", "STA-LSTM-H + bp_lag1",
                  "STA-LSTM-H + bp (same-step)")
    for col, tag, hb in METRICS:
        rows.append((f"STA-LSTM-H_lagged_vs_samestep|{tag}",
                     one(A[col], B[col], hb)))
    famF2 = run_family("F2 factorial ablation (5 contrasts x 3 metrics)",
                       rows, R / "ablation_wilcoxon_effects.csv")

    # ---------------- F3 : leave-one-animal-out -------------------------
    lo = pd.read_csv(R / "comparison_report_loao.csv")
    lo = lo[lo["model"] != "model"]
    for c in ("accuracy_mean", "f1_mean", "rmse_mean"):
        lo[c] = pd.to_numeric(lo[c], errors="coerce")
    rows = []
    for base in ["STA-LSTM", "LSTM", "GRU", "Transformer"]:
        A, B = paired(lo, "model", "held_out_animal", "STA-LSTM-H", base)
        for col, tag, hb in METRICS:
            rows.append((f"STA-LSTM-H_vs_{base}|{tag}",
                         one(A[col], B[col], hb)))
    famF3 = run_family("F3 leave-one-animal-out", rows,
                       R / "wilcoxon_loao_18animals_effects.csv")

    # ---------------- LaTeX rows for the three tables -------------------
    def trip(fam, key):
        out = {}
        for _, r in fam.iterrows():
            comp, tag = r["comparison"].split("|")
            out.setdefault(comp, {})[tag] = (r["p_holm"], r["r_rb"])
        return out

    print("\n\n########## LaTeX table rows ##########")
    for fam, order, title in [
        (famF1, ["STA-LSTM-H_vs_STA-LSTM", "STA-LSTM-H_vs_LSTM",
                 "STA-LSTM-H_vs_GRU", "STA-LSTM-H_vs_Transformer"],
         "tab:wilcoxon (F1)"),
        (famF2, ["STA-LSTM-H+bp_lag1_vs_STA-LSTM + bp_lag1",
                 "STA-LSTM-H+bp_lag1_vs_LSTM + bp_lag1",
                 "STA-LSTM-H+bp_lag1_vs_GRU + bp_lag1",
                 "STA-LSTM-H+bp_lag1_vs_Transformer + bp_lag1",
                 "STA-LSTM-H_lagged_vs_samestep"], "tab:ablation_wilcoxon (F2)"),
        (famF3, ["STA-LSTM-H_vs_STA-LSTM", "STA-LSTM-H_vs_LSTM",
                 "STA-LSTM-H_vs_GRU", "STA-LSTM-H_vs_Transformer"],
         "tab:loao_wilcoxon (F3)"),
    ]:
        t = trip(fam, title)
        print(f"\n--- {title} ---")
        for comp in order:
            cells = t[comp]
            seg = " & ".join(
                f"{fmt_p(cells[m][0])} & ${cells[m][1]:+.2f}$"
                for m in ("acc", "f1", "rmse"))
            print(f"{comp}  & {seg} \\\\")


if __name__ == "__main__":
    main()
