"""scripts/make_figures.py — publication-quality figures for the STA-LSTM-H paper.

Reads the consolidated results from `Results/python_pipeline/` (per-animal
`comparison_report.csv`, `wilcoxon_stats.csv`, `autoregressive_errors.csv`,
`last_fold_predictions.npy`) and renders EPS+PNG figures into
`manuscript/figures-new/`.

Figures produced (matching the prior MDPI Drones paper's lineup):
  * cattle_movements.eps     — example trajectory of one animal
  * movements.eps            — multi-animal heatmap
  * prediction_LSTM.eps      — actual vs predicted position over time
  * errors_over_time.eps     — per-step error series, all models
  * boxplot_errors.eps       — error distribution boxplot per model
  * cdf_errors.eps           — cumulative error distribution
  * allSteps_errors.eps      — multi-step horizon error curve (1..30)
  * confusion_matrices.png   — per-model confusion matrices for behaviour
  * radar_chart.eps          — accuracy / F1 / 1/RMSE / 1/latency / 1/memory
  * signal_comparison.eps    — accel signal vs predicted behaviour vs truth

Usage:
    python scripts/make_figures.py \
        --results-dir Results/python_pipeline \
        --out-dir     manuscript/figures-new
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.3,
})

MODEL_COLORS = {
    "STA-LSTM-H": "#d62728",
    "LSTM-H":     "#ff7f0e",
    "STA-LSTM":   "#9467bd",
    "Transformer":"#2ca02c",
    "LSTM":       "#1f77b4",
    "GRU":        "#17becf",
    "KF":         "#7f7f7f",
    "EKF-HMM":    "#8c564b",
}

# Classical filters consume the noisy-position observation channel directly,
# while every deep model consumes only accelerometry. Plotting them on the
# same axes is misleading — separate them throughout.
CLASSICAL_NAMES: set[str] = {"KF", "EKF-HMM"}

# Per-model line style for time-series plots. STA-LSTM-H is the proposed model
# and is rendered solid + heavy; STA-LSTM is dashed; the unguarded baselines
# get distinguishable dotted/dash-dot/long-dash patterns.
MODEL_LINESTYLE = {
    "STA-LSTM-H":  ("-",         2.4, 11),
    "STA-LSTM":    ((0, (5, 2)), 2.0, 10),
    "LSTM":        ((0, (1, 1)), 1.6,  6),
    "GRU":         ((0, (3, 1, 1, 1)), 1.6,  6),
    "Transformer": ((0, (5, 1)), 1.6,  6),
}


def compute_input_noise_floor(
    per_animal_preds: dict, noise_pct: float = 0.05,
) -> float:
    """Mean over animals of `noise_pct * range(true_xy)` averaged over (x, y).

    Reproduces the synthetic UAV-noise σ that the classical KF / EKF-HMM
    baselines consume, so we can plot a horizontal reference line marking
    the input noise floor on the classical-filter panel.
    """
    if not per_animal_preds:
        return float("nan")
    sigmas: list[float] = []
    for animal_d in per_animal_preds.values():
        # Use whichever key carries a saved true_xy — the truth is shared
        # across models per animal so any present key works.
        for d in animal_d.values():
            t = np.asarray(d.get("true_xy"))
            if t.ndim == 2 and t.shape[1] == 2:
                rng = (t.max(axis=0) - t.min(axis=0)).mean()
                sigmas.append(noise_pct * float(rng))
                break
    return float(np.mean(sigmas)) if sigmas else float("nan")


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use bbox_inches='tight' with a small pad so axis labels are never clipped
    # but the figure is not artificially padded either.
    fig.savefig(out_dir / f"{name}.eps", format="eps",
                bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_dir / f"{name}.png", format="png",
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  ✓ {name}.{{eps,png}}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Boxplot of errors per model (across folds × animals)
# ══════════════════════════════════════════════════════════════════════════════

def plot_error_boxplot(
    consolidated: pd.DataFrame,
    out_dir: Path,
    noise_floor: float | None = None,
) -> None:
    """Two-panel boxplot, framed as Blind Forecasting vs Sensor-Fusion Reference.

    Top panel  : deep-net RMSE — Blind Forecasting from accelerometry only.
    Bottom panel: classical KF / EKF-HMM RMSE — Sensor-Fusion Reference fed
                  with a synthetic 5%-Gaussian noisy-position observation
                  channel. A horizontal dotted line marks the input noise
                  floor σ ≈ 0.05·range(coord) so the reader can see the
                  classical filters approach but do not beat the noise of
                  their own input.

    The two panels share nothing; their y-axes are unrelated and the
    classical filters are NOT comparable to the deep models.
    """
    if consolidated.empty:
        return
    all_models = consolidated["model"].unique().tolist()
    nn_models  = [m for m in all_models if m not in CLASSICAL_NAMES]
    cl_models  = [m for m in all_models if m in CLASSICAL_NAMES]

    if not nn_models and not cl_models:
        return

    n_panels = (1 if nn_models else 0) + (1 if cl_models else 0)
    # Side-by-side layout: deep nets (left) | classical filters (right).
    width_ratios = []
    if nn_models: width_ratios.append(3)
    if cl_models: width_ratios.append(1.7)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(9.5, 3.6),
        gridspec_kw={"width_ratios": width_ratios},
    )
    if n_panels == 1:
        axes = [axes]

    panel_iter = iter(axes)

    if nn_models:
        ax = next(panel_iter)
        data = [consolidated.loc[consolidated["model"] == m, "rmse_mean"].dropna()
                for m in nn_models]
        bp = ax.boxplot(data, tick_labels=nn_models, patch_artist=True, widths=0.55)
        for patch, m in zip(bp["boxes"], nn_models):
            patch.set_facecolor(MODEL_COLORS.get(m, "#cccccc"))
            patch.set_alpha(0.75)
        ax.set_ylabel("RMSE (proxy units)")
        ax.set_title("Blind Forecasting (deep nets)", fontsize=10)
        ax.tick_params(axis="x", rotation=20)

    if cl_models:
        ax = next(panel_iter)
        data = [consolidated.loc[consolidated["model"] == m, "rmse_mean"].dropna()
                for m in cl_models]
        bp = ax.boxplot(data, tick_labels=cl_models, patch_artist=True, widths=0.45)
        for patch, m in zip(bp["boxes"], cl_models):
            patch.set_facecolor(MODEL_COLORS.get(m, "#cccccc"))
            patch.set_alpha(0.75)
        ax.set_ylabel("RMSE (proxy units)")
        ax.set_title("Sensor-Fusion Reference\n(noisy obs input)", fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        if noise_floor is not None and np.isfinite(noise_floor):
            ax.axhline(
                noise_floor, color="grey", ls=":", lw=1.4,
                label=f"Noise floor σ ≈ {noise_floor:,.0f}",
            )
            ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18, wspace=0.30)
    _save(fig, out_dir, "boxplot_errors")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CDF of errors
# ══════════════════════════════════════════════════════════════════════════════

def plot_cdf(
    consolidated: pd.DataFrame,
    out_dir: Path,
    noise_floor: float | None = None,
) -> None:
    """Two-panel CDF, framed as Blind Forecasting vs Sensor-Fusion Reference.

    Mixing classical filters and deep nets on a shared x-axis collapses the
    deep-net spread to the right edge because KF/EKF-HMM RMSE is ~5x smaller
    on a different input signal. We split them so each CDF actually conveys
    the per-animal spread within its input regime, and mark the input noise
    floor on the classical-filter panel so the reader can see the filters
    approach but do not beat the noise of their own input.
    """
    if consolidated.empty:
        return
    all_models = consolidated["model"].unique().tolist()
    nn_models  = [m for m in all_models if m not in CLASSICAL_NAMES]
    cl_models  = [m for m in all_models if m in CLASSICAL_NAMES]

    n_panels = (1 if nn_models else 0) + (1 if cl_models else 0)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(11 if n_panels == 2 else 7, 4.6))
    if n_panels == 1:
        axes = [axes]
    panel_iter = iter(axes)

    def _draw(ax, models, ylabel_extra):
        for m in models:
            rmse_vals = consolidated.loc[consolidated["model"] == m, "rmse_mean"].dropna()
            if len(rmse_vals) == 0:
                continue
            sorted_vals = np.sort(rmse_vals.to_numpy())
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, cdf, label=m,
                    color=MODEL_COLORS.get(m, None), lw=2)
        ax.set_xlabel(f"RMSE — {ylabel_extra}")
        ax.set_ylabel("Cumulative probability")
        ax.legend(loc="lower right")

    if nn_models:
        ax = next(panel_iter)
        _draw(ax, nn_models, "Blind Forecasting (deep nets, accel only)")
        ax.set_title("Blind Forecasting")
        ax.text(
            0.02, 0.98, "Input: accelerometry only",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.85),
        )
    if cl_models:
        ax = next(panel_iter)
        _draw(ax, cl_models, "Sensor-Fusion Reference (noisy-obs input)")
        ax.set_title("Sensor-Fusion Reference (different input modality)")
        ax.text(
            0.02, 0.98,
            "Input: noisy synthetic\nposition (5 % Gaussian σ)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.85),
        )
        if noise_floor is not None and np.isfinite(noise_floor):
            ax.axvline(
                noise_floor, color="grey", ls=":", lw=1.4,
                label=f"Input noise floor σ ≈ {noise_floor:,.0f}",
            )
            ax.legend(loc="lower right", fontsize=9, frameon=True,
                      framealpha=0.9)

    fig.suptitle(
        "CDF of per-animal mean RMSE across the 18-animal cohort\n"
        "(deep-net RMSE is NOT directly comparable to the Sensor-Fusion Reference)",
        y=1.04, fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out_dir, "cdf_errors")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Multi-step horizon error curve
# ══════════════════════════════════════════════════════════════════════════════

def plot_horizon_errors(ar_dfs: list[pd.DataFrame], out_dir: Path) -> None:
    if not ar_dfs:
        return
    combined = pd.concat(ar_dfs, ignore_index=True)
    horizon_cols = [c for c in combined.columns if c.startswith("h")]
    if not horizon_cols:
        return
    horizons = [int(c[1:]) for c in horizon_cols]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in combined["model"].unique():
        rows = combined.loc[combined["model"] == m, horizon_cols]
        mean = rows.mean(axis=0).to_numpy()
        std  = rows.std(axis=0).to_numpy()
        ax.plot(horizons, mean, label=m, color=MODEL_COLORS.get(m, None), lw=2)
        ax.fill_between(horizons, mean - std, mean + std,
                        color=MODEL_COLORS.get(m, "#cccccc"), alpha=0.15)
    ax.set_xlabel("Prediction horizon (1-second steps)")
    ax.set_ylabel("Mean Euclidean error (relative-motion units)")
    ax.set_title("Autoregressive multi-step prediction error")
    ax.legend(loc="lower right")
    _save(fig, out_dir, "allSteps_errors")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Confusion matrices
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["Other", "Ruminating", "Eating"]


def plot_confusion_matrices(
    per_animal_preds: dict,
    out_dir: Path,
    *,
    models: list[str] | None = None,
    basename: str = "confusion_matrices",
) -> None:
    """Per-model 3-class confusion matrices, deep nets only.

    Classical filters (KF / EKF-HMM) are excluded because they store sentinel
    pred_cls = -1 (they have no classification head); their panels were
    rendering as all-zeros matrices in the prior version, which made the
    figure look broken even though the underlying numbers are fine.

    Pass ``models`` to emit a subset (e.g. main-text contrast panels only).
    """
    if not per_animal_preds:
        return
    raw_models = sorted({m for d in per_animal_preds.values() for m in d.keys()})
    available = [m for m in raw_models if m not in CLASSICAL_NAMES]
    preferred_order = ["LSTM", "GRU", "Transformer", "STA-LSTM", "STA-LSTM-H"]
    if models is None:
        models = [m for m in preferred_order if m in available] + \
                 [m for m in available if m not in preferred_order]
    else:
        models = [m for m in models if m in available]
    if not models:
        return
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.8), squeeze=False)

    for ax, m in zip(axes[0], models):
        cm = np.zeros((3, 3), dtype=np.int64)
        for animal_d in per_animal_preds.values():
            d = animal_d.get(m)
            if d is None:
                continue
            pred = np.asarray(d["pred_cls"]).astype(np.int64)
            true = np.asarray(d["true_cls"]).astype(np.int64)
            mask = (pred >= 0) & (pred < 3) & (true >= 0) & (true < 3)
            for t, p in zip(true[mask], pred[mask]):
                cm[int(t), int(p)] += 1
        if cm.sum() == 0:
            ax.set_axis_off()
            ax.set_title(f"{m}\n(no behaviour predictions)")
            continue
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3));  ax.set_xticklabels(CLASS_NAMES, rotation=30)
        ax.set_yticks(range(3));  ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted");  ax.set_ylabel("True")
        ax.set_title(m)
        for i in range(3):
            for j in range(3):
                txt_color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]:,})",
                        ha="center", va="center", color=txt_color, fontsize=9)
    fig.suptitle("Confusion matrices per model (pooled across animals)")
    fig.tight_layout()
    _save(fig, out_dir, basename)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Radar chart of multi-criterion model comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_radar(consolidated: pd.DataFrame, out_dir: Path) -> None:
    """Radar over deep-net cohort means.

    Classical KF / EKF-HMM are excluded — they have no behaviour-classification
    metrics (NaN accuracy/F1) and they consume a different input signal, so
    putting them on the same polar axes as the deep nets produced broken
    polygons in the prior version of this figure.
    """
    if consolidated.empty:
        return
    nn_only = consolidated[~consolidated["model"].isin(CLASSICAL_NAMES)].copy()
    if nn_only.empty:
        return
    agg = (nn_only
           .groupby("model")[["accuracy_mean", "f1_mean", "rmse_mean",
                              "latency_ms_mean"]]
           .mean()
           .reset_index()
           .dropna(subset=["accuracy_mean", "f1_mean", "rmse_mean",
                           "latency_ms_mean"]))
    if agg.empty:
        return

    # Higher = better on every axis; invert RMSE and latency before normalising
    norm = pd.DataFrame({
        "Accuracy":    agg["accuracy_mean"]   / agg["accuracy_mean"].max(),
        "F1":          agg["f1_mean"]         / agg["f1_mean"].max(),
        "1 / RMSE":    (1 / agg["rmse_mean"]) / (1 / agg["rmse_mean"]).max(),
        "1 / Latency": (1 / agg["latency_ms_mean"]) / (1 / agg["latency_ms_mean"]).max(),
    })
    labels = list(norm.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.0, 7.8))
    ax  = fig.add_subplot(111, projection="polar")
    for i, m in enumerate(agg["model"]):
        ls, lw, zorder = MODEL_LINESTYLE.get(m, ("-", 1.6, 5))
        vals = norm.iloc[i].tolist() + [norm.iloc[i, 0]]
        ax.plot(angles, vals, label=m, color=MODEL_COLORS.get(m, None),
                lw=lw, ls=ls, zorder=zorder)
        ax.fill(angles, vals, color=MODEL_COLORS.get(m, "#cccccc"),
                alpha=0.10, zorder=zorder - 0.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1.05)
    ax.set_title("Cohort-averaged multi-criterion comparison (deep nets only)",
                 pad=22)
    # Legend below the polar plot, horizontal — keeps the radar uncluttered.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=min(4, len(agg)), frameon=False)
    fig.text(
        0.5, 0.02,
        "All four axes oriented so outer = better "
        "(Accuracy and F1 raw; 1/RMSE and 1/Latency inverted before normalisation).",
        ha="center", fontsize=9, style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out_dir, "radar_chart")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Per-animal trajectory + per-step errors (uses last_fold_predictions.npy)
# ══════════════════════════════════════════════════════════════════════════════

FIGURE5_SLICE_SEED = 20260510
"""Fixed seed for reproducible random slice selection in Figure 5.

Reviewer 3 flagged the prior `pick_best_slice` as selection-on-the-dependent-
variable (it picked the window minimising the reference model's residual,
which biases the qualitative panel toward the model's best-case behaviour).
We replace it with a fixed-seed random window so the slice is independent of
model performance. Update this seed only if the manuscript caption is updated
in the same change.
"""


def pick_random_slice(
    true_xy: np.ndarray,
    slice_len: int = 1000,
    seed: int = FIGURE5_SLICE_SEED,
) -> tuple[int, int]:
    """Pick a fixed-seed random contiguous slice of the test fold.

    Independent of any model's predictions, so cannot bias the qualitative
    panel toward a specific architecture. The seed is fixed across runs to
    keep the figure reproducible, but the slice is no longer chosen to
    minimise residuals on the dependent variable.

    Args:
        true_xy:  (N, 2) ground-truth pseudo-position over the test fold.
        slice_len: window length in samples.
        seed: RNG seed; fixed at module level for reproducibility.

    Returns:
        (start, end) sample indices of the chosen slice. Falls back to the
        full fold if it is shorter than `slice_len`.
    """
    n_total = len(true_xy)
    if n_total <= slice_len:
        return 0, n_total
    rng = np.random.default_rng(seed)
    s0 = int(rng.integers(0, n_total - slice_len + 1))
    return s0, s0 + slice_len


def plot_predictions_vs_truth(per_animal_preds: dict, out_dir: Path,
                              animal_focus: str | None = None) -> None:
    """Four-panel position-prediction figure.

    (top-left)    contiguous slice, x-coordinate, every deep-net model.
    (top-right)   2-D density of true x vs predicted x for STA-LSTM-H over
                  the full test fold (hexbin, log colour scale, y = x ref).
    (bottom-left) same slice, y-coordinate.
    (bottom-right) hexbin density on y.

    The slice is chosen by `pick_random_slice` with a fixed seed; it is
    independent of any model's predictions and therefore cannot bias the
    panel toward a specific architecture (replaces the prior best-residual
    selection flagged by Reviewer 3 as selection-on-the-dependent-variable).
    """
    if not per_animal_preds:
        return
    if animal_focus is None:
        animal_focus = sorted(per_animal_preds.keys())[0]
    if animal_focus not in per_animal_preds:
        return

    d = per_animal_preds[animal_focus]
    ref_model = "STA-LSTM-H" if "STA-LSTM-H" in d else (
                "LSTM-H" if "LSTM-H" in d else next(iter(d)))
    if ref_model not in d:
        return

    nn_models_present = [m for m in
                         ["STA-LSTM-H", "STA-LSTM", "LSTM", "GRU", "Transformer"]
                         if m in d and d[m]["pred_xy"].ndim == 2]
    if not nn_models_present:
        return

    true_xy = d[ref_model]["true_xy"]
    ref_pred = d[ref_model]["pred_xy"]
    n_total = len(true_xy)
    slice_start, slice_end = pick_random_slice(
        true_xy, slice_len=min(1000, n_total)
    )
    t = np.arange(slice_start, slice_end)
    slice_len = slice_end - slice_start

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    def _slice_panel(ax, axis_idx: int, axis_label: str) -> None:
        ax.plot(t, true_xy[slice_start:slice_end, axis_idx],
                label=f"{axis_label} (true)", color="black",
                lw=2.2, zorder=12)
        for m in nn_models_present:
            ls, lw, zorder = MODEL_LINESTYLE.get(m, ("--", 1.2, 5))
            pxy = d[m]["pred_xy"]
            ax.plot(t, pxy[slice_start:slice_end, axis_idx],
                    label=f"{axis_label} ({m})",
                    color=MODEL_COLORS.get(m, None),
                    lw=lw * 0.8, ls=ls, alpha=0.85, zorder=zorder)
        ax.set_xlabel("Time index (1-second windows)")
        ax.set_ylabel(f"Relative {axis_label}-position (proxy units)")
        ax.legend(loc="lower center", ncol=3, frameon=True, framealpha=0.92,
                  borderpad=0.4, handlelength=2.0, fontsize=8)

    def _hexbin_panel(ax, axis_idx: int, axis_label: str) -> None:
        true_v = true_xy[:, axis_idx]
        pred_v = ref_pred[:, axis_idx]
        hb = ax.hexbin(
            true_v, pred_v, gridsize=50, cmap="Blues",
            mincnt=1, bins="log",
        )
        lo = float(min(true_v.min(), pred_v.min()))
        hi = float(max(true_v.max(), pred_v.max()))
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls=":",
                label="y = x")
        ax.set_xlabel(f"True {axis_label} (proxy units)")
        ax.set_ylabel(f"Predicted {axis_label} (proxy units)")
        ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.92)
        cbar = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label("count (log)", fontsize=9)

    _slice_panel(axes[0, 0], 0, "x")
    axes[0, 0].set_title(
        f"Animal {animal_focus}: x-coordinate, fixed-seed random "
        f"{slice_len}-window slice (rows {slice_start}-{slice_end})"
    )
    _hexbin_panel(axes[0, 1], 0, "x")
    axes[0, 1].set_title(f"{ref_model}: full-fold density on x (n = {n_total:,})")

    _slice_panel(axes[1, 0], 1, "y")
    axes[1, 0].set_title(
        f"Animal {animal_focus}: y-coordinate, same slice"
    )
    _hexbin_panel(axes[1, 1], 1, "y")
    axes[1, 1].set_title(f"{ref_model}: full-fold density on y (n = {n_total:,})")

    fig.tight_layout()
    _save(fig, out_dir, "prediction_LSTM")


def plot_multi_animal_predictions(per_animal_preds: dict, out_dir: Path) -> None:
    """Three-animal position-prediction figure spanning RMSE quartiles.

    Plots Animals 04 (low RMSE), 01 (median RMSE), 13 (high RMSE) in a 3-column
    layout, each with a time-series panel for x-coordinate. Addresses Reviewer 3
    feedback to show variability across animals.
    """
    if not per_animal_preds:
        return

    animals_to_plot = ["04", "01", "13"]
    available = [a for a in animals_to_plot if a in per_animal_preds]
    if len(available) < 2:
        return  # Need at least 2 animals

    fig, axes = plt.subplots(1, len(available), figsize=(5.0 * len(available), 4.0),
                             squeeze=False)

    for col, animal_id in enumerate(available):
        ax = axes[0, col]
        d = per_animal_preds[animal_id]

        ref_model = "STA-LSTM-H" if "STA-LSTM-H" in d else (
                    "LSTM-H" if "LSTM-H" in d else next(iter(d)))
        if ref_model not in d:
            continue

        nn_models = [m for m in ["STA-LSTM-H", "STA-LSTM", "LSTM", "GRU", "Transformer"]
                     if m in d and d[m]["pred_xy"].ndim == 2]
        if not nn_models:
            continue

        true_xy = d[ref_model]["true_xy"]
        n_total = len(true_xy)
        slice_start, slice_end = pick_random_slice(true_xy, slice_len=min(1000, n_total))
        t = np.arange(slice_start, slice_end)

        ax.plot(t, true_xy[slice_start:slice_end, 0],
                label="x (true)", color="black", lw=2.2, zorder=12)
        for m in nn_models:
            ls, lw, zorder = MODEL_LINESTYLE.get(m, ("--", 1.2, 5))
            pxy = d[m]["pred_xy"]
            ax.plot(t, pxy[slice_start:slice_end, 0],
                    label=f"{m}",
                    color=MODEL_COLORS.get(m, None),
                    lw=lw * 0.8, ls=ls, alpha=0.85, zorder=zorder)

        ax.set_xlabel("Time index (1-second windows)", fontsize=9)
        ax.set_ylabel("Relative x-position (proxy units)", fontsize=9)
        ax.set_title(f"Animal {animal_id} (n = {n_total:,} windows)", fontsize=10)
        ax.legend(loc="best", ncol=2, frameon=True, framealpha=0.92, fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Position prediction across animals spanning RMSE range")
    fig.tight_layout()
    _save(fig, out_dir, "prediction_multi_animal")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def collect(results_dir: Path) -> tuple[pd.DataFrame, list[pd.DataFrame], dict]:
    """Walk per-animal directories and return consolidated tables."""
    rows = []
    ar_frames = []
    per_animal_preds: dict = {}

    for animal_dir in sorted(results_dir.glob("animal-*")):
        animal_id = animal_dir.name.replace("animal-", "")
        comp_path = animal_dir / "comparison_report.csv"
        ar_path   = animal_dir / "autoregressive_errors.csv"
        npy_path  = animal_dir / "last_fold_predictions.npy"
        if comp_path.exists():
            comp = pd.read_csv(comp_path)
            comp["animal"] = animal_id
            rows.append(comp)
        if ar_path.exists():
            ar = pd.read_csv(ar_path)
            ar["animal"] = animal_id
            ar_frames.append(ar)
        if npy_path.exists():
            d = np.load(npy_path, allow_pickle=True).item()
            per_animal_preds[animal_id] = d

    # Also accept a flat layout (single-animal pilot)
    flat_comp = results_dir / "comparison_report.csv"
    flat_ar   = results_dir / "autoregressive_errors.csv"
    flat_npy  = results_dir / "last_fold_predictions.npy"
    if flat_comp.exists() and not rows:
        comp = pd.read_csv(flat_comp)
        comp["animal"] = "pilot"
        rows.append(comp)
    if flat_ar.exists() and not ar_frames:
        ar = pd.read_csv(flat_ar)
        ar["animal"] = "pilot"
        ar_frames.append(ar)
    if flat_npy.exists() and not per_animal_preds:
        per_animal_preds["pilot"] = np.load(flat_npy, allow_pickle=True).item()

    consolidated = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return consolidated, ar_frames, per_animal_preds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("Results/python_pipeline"))
    p.add_argument("--out-dir",     type=Path, default=Path("manuscript/figures-new"))
    p.add_argument("--animal-focus", type=str, default=None,
                   help="Animal id (e.g. '04') for trajectory plots")
    args = p.parse_args()

    print(f"\n  Reading results from: {args.results_dir}")
    consolidated, ar_frames, per_animal_preds = collect(args.results_dir)
    print(f"  Loaded {len(consolidated)} report rows from "
          f"{len(per_animal_preds) or '?'} animals; "
          f"{len(ar_frames)} autoregressive curves")

    noise_floor = compute_input_noise_floor(per_animal_preds)
    if np.isfinite(noise_floor):
        print(f"  Input noise floor σ ≈ {noise_floor:,.1f} (proxy units, "
              f"5 % of mean coord range)")

    print(f"\n  Writing figures → {args.out_dir}")
    plot_error_boxplot(consolidated, args.out_dir, noise_floor=noise_floor)
    plot_cdf(consolidated, args.out_dir, noise_floor=noise_floor)
    plot_horizon_errors(ar_frames, args.out_dir)
    plot_confusion_matrices(per_animal_preds, args.out_dir)
    # Main text: clearest architectural contrast at the Ruminating↔Eating boundary.
    plot_confusion_matrices(
        per_animal_preds,
        args.out_dir,
        models=["Transformer", "STA-LSTM-H"],
        basename="confusion_matrices_main",
    )
    # Supplementary: remaining deep-net panels.
    plot_confusion_matrices(
        per_animal_preds,
        args.out_dir,
        models=["LSTM", "GRU", "STA-LSTM"],
        basename="confusion_matrices_supp",
    )
    plot_radar(consolidated, args.out_dir)
    plot_predictions_vs_truth(per_animal_preds, args.out_dir,
                              animal_focus=args.animal_focus)
    plot_multi_animal_predictions(per_animal_preds, args.out_dir)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
