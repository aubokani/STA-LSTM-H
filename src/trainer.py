"""src/trainer.py — 5-Fold CV Engine, Autoregressive Evaluation, Statistical Tests.

Stage 2 — MODELLING:
  * StratifiedKFold (k=5) on behaviour labels.
  * SMOTE or weighted CE on training fold only.
  * Train LSTM-H, LSTM, GRU per fold.
  * KF post-processing with auto-retune (model.py).
  * Per-fold metrics: accuracy, F1, RMSE, latency, memory.

Stage 3 — ANALYTICS:
  * Performance Ledger with mean ± std across folds.
  * Composite weighted score: 80 % Acc + 10 % (1/Lat) + 10 % (1/Mem).
  * Wilcoxon signed-rank test LSTM-H vs each baseline.
  * Autoregressive multi-step rollout error curve.

Stage 4 — OUTPUT:
  * results/comparison_report.csv
  * results/wilcoxon_stats.csv
  * results/autoregressive_errors.csv  (if max_horizon > 0)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats as spstats
import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import sys
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import (
    ACCEL_FEAT_COLS,
    LSTMH_EXTRA_COLS,
    audit_class_balance,
    build_autoregressive_sequences,
    denormalize_xy,
    make_sequences,
    make_sequences_split,
)
from model import (
    BaselineRNN,
    LSTMHModel,
    ModelKind,
    add_uav_observation_noise,
    build_model,
    ekf_hmm_baseline,
    evaluate,
    kalman_adaptive,
    kalman_only_baseline,
    kalman_with_retune,
    macro_f1,
    model_memory_mb,
    select_device,
    train_epoch,
)

RESULTS_DIR = Path("Results/python_pipeline")

from data_loader import LAG1_EXTRA_COLS  # noqa: E402  (kept here to group registry imports)

# Neural-network models (trained per fold).
NN_MODELS: tuple[str, ...] = (
    "lstm_h", "sta_lstm", "lstm", "gru", "transformer", "cnn1d",
)
# Deterministic classical baselines (no training).
CLASSICAL_MODELS: tuple[str, ...] = ("kf", "ekf_hmm")
# Tabular gradient-boosting baseline (trained, but not a PyTorch module).
# Reported separately because it has no notion of position regression in
# the per-second 14-feature representation; we report behaviour accuracy
# and F1 only and leave RMSE NaN, matching the classical-baseline shape.
XGBOOST_MODELS: tuple[str, ...] = ("xgboost",)

# ── Ablation registry ───────────────────────────────────────────────────────────
# 10 cells filling the (architecture × input-set) factorial that the headline
# run does not cover. Each ablation key aliases back to one of the four canonical
# build_model() kinds; only the input feature set differs.
#
#   *_accel  : strip the behaviour channel from the STA family       (cols 14)
#   *_bp     : add behavior_prev (=behavior/2, no shift) to the      (cols 15)
#              plain LSTM/GRU/Transformer baselines — same column the
#              STA family already consumes in the headline run
#   *_lag1   : add behavior_prev_lag1 (=behavior.shift(1)/2) to every (cols 15)
#              architecture so the channel reflects only the previous
#              window's GT label, not the current step
ABLATION_NN_MODELS: tuple[str, ...] = (
    "lstm_h_accel",   "sta_lstm_accel",
    "lstm_bp",        "gru_bp",         "transformer_bp",  "cnn1d_bp",
    "lstm_h_lag1",    "sta_lstm_lag1",
    "lstm_lag1",      "gru_lag1",       "transformer_lag1", "cnn1d_lag1",
)

# Map every key (headline + ablation) to the canonical build_model() kind so we
# can register new keys without touching model.py.
MODEL_KIND_ALIAS: dict[str, str] = {
    "lstm_h_accel":     "lstm_h",
    "sta_lstm_accel":   "sta_lstm",
    "lstm_bp":          "lstm",
    "gru_bp":           "gru",
    "transformer_bp":   "transformer",
    "cnn1d_bp":         "cnn1d",
    "lstm_h_lag1":      "lstm_h",
    "sta_lstm_lag1":    "sta_lstm",
    "lstm_lag1":        "lstm",
    "gru_lag1":         "gru",
    "transformer_lag1": "transformer",
    "cnn1d_lag1":       "cnn1d",
}

MODEL_FEATURE_MAP: dict[str, list[str]] = {
    # ── Headline ─────────────────────────────────────────────────────────────
    "lstm_h":      ACCEL_FEAT_COLS + LSTMH_EXTRA_COLS,
    "sta_lstm":    ACCEL_FEAT_COLS + LSTMH_EXTRA_COLS,  # same backbone, no KF
    "lstm":        ACCEL_FEAT_COLS,
    "gru":         ACCEL_FEAT_COLS,
    "transformer": ACCEL_FEAT_COLS,
    "cnn1d":       ACCEL_FEAT_COLS,
    "xgboost":     ACCEL_FEAT_COLS,
    # ── Ablation: accel-only on STA family (14 features) ─────────────────────
    "lstm_h_accel":     ACCEL_FEAT_COLS,
    "sta_lstm_accel":   ACCEL_FEAT_COLS,
    # ── Ablation: + behavior_prev (current-GT) on plain baselines (15 feat) ──
    "lstm_bp":          ACCEL_FEAT_COLS + LSTMH_EXTRA_COLS,
    "gru_bp":           ACCEL_FEAT_COLS + LSTMH_EXTRA_COLS,
    "transformer_bp":   ACCEL_FEAT_COLS + LSTMH_EXTRA_COLS,
    "cnn1d_bp":         ACCEL_FEAT_COLS + LSTMH_EXTRA_COLS,
    # ── Ablation: + behavior_prev_lag1 (true previous GT) on all (15 feat) ───
    "lstm_h_lag1":      ACCEL_FEAT_COLS + LAG1_EXTRA_COLS,
    "sta_lstm_lag1":    ACCEL_FEAT_COLS + LAG1_EXTRA_COLS,
    "lstm_lag1":        ACCEL_FEAT_COLS + LAG1_EXTRA_COLS,
    "gru_lag1":         ACCEL_FEAT_COLS + LAG1_EXTRA_COLS,
    "transformer_lag1": ACCEL_FEAT_COLS + LAG1_EXTRA_COLS,
    "cnn1d_lag1":       ACCEL_FEAT_COLS + LAG1_EXTRA_COLS,
}
MODEL_DISPLAY: dict[str, str] = {
    "lstm_h":      "STA-LSTM-H",
    "sta_lstm":    "STA-LSTM",
    "lstm":        "LSTM",
    "gru":         "GRU",
    "transformer": "Transformer",
    "cnn1d":       "1D-CNN",
    "xgboost":     "XGBoost",
    "kf":          "KF",
    "ekf_hmm":     "EKF-HMM",
    "lstm_h_accel":     "STA-LSTM-H (accel)",
    "sta_lstm_accel":   "STA-LSTM (accel)",
    "lstm_bp":          "LSTM + bp",
    "gru_bp":           "GRU + bp",
    "transformer_bp":   "Transformer + bp",
    "cnn1d_bp":         "1D-CNN + bp",
    "lstm_h_lag1":      "STA-LSTM-H + bp_lag1",
    "sta_lstm_lag1":    "STA-LSTM + bp_lag1",
    "lstm_lag1":        "LSTM + bp_lag1",
    "gru_lag1":         "GRU + bp_lag1",
    "transformer_lag1": "Transformer + bp_lag1",
    "cnn1d_lag1":       "1D-CNN + bp_lag1",
}

# UAV-style observation noise for the KF/EKF-HMM denoising task. 5% of the
# coordinate range matches the prior MDPI Drones paper (Bokani et al. 2025).
UAV_NOISE_PCT: float = 0.05


@dataclass
class TrainConfig:
    animal_ids:           tuple[int, ...] = (1,)
    folds:                int   = 5
    epochs:               int   = 5
    batch_size:           int   = 64
    sequence_length:      int   = 25
    learning_rate:        float = 0.002
    weight_decay:         float = 1e-4
    grad_clip:            float = 1.2
    train_fraction:       float = 0.8
    max_horizon:          int   = 0        # 0 = skip autoregressive eval
    device_pref:          str   = "auto"
    smote_threshold:      float = 0.80
    seed:                 int   = 42
    # When True, post-process STA-LSTM-H predictions with the behaviour-adaptive
    # Kalman filter. The pilot showed the KF hurts predictions when there is no
    # separate noisy-observation source, so the default is False; the KF lives
    # on as a standalone classical baseline (`kf`, `ekf_hmm`).
    use_kalman_for_lstmh: bool  = False
    # Where to write CSV/NPY artefacts. None → use the module-level RESULTS_DIR
    # (the legacy flat layout). When parallel per-animal jobs share that flat
    # directory they overwrite each other — so run_animal.py passes an explicit
    # per-animal path here.
    output_dir:           Path | None = None
    # Switch to the (architecture × input-set) factorial ablation registry.
    # When True, run_cross_validation iterates ABLATION_NN_MODELS and skips
    # CLASSICAL_MODELS (input-set independent — already on disk).
    ablation_mode:        bool = False


# ══════════════════════════════════════════════════════════════════════════════
# SMOTE helper — operates on flattened (N × T×F) feature matrix
# ══════════════════════════════════════════════════════════════════════════════

def _apply_smote(
    X: np.ndarray,          # (N, T, F)
    Yxy: np.ndarray,        # (N, 2)
    Yc: np.ndarray,         # (N,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, T, F = X.shape
    X_flat = X.reshape(N, T * F)
    X_aug  = np.hstack([X_flat, Yxy])

    k_nb = max(1, min(5, np.unique(Yc, return_counts=True)[1].min() - 1))
    try:
        sm = SMOTE(random_state=42, k_neighbors=k_nb)
        X_res, Yc_res = sm.fit_resample(X_aug, Yc)
        return (
            X_res[:, : T * F].reshape(-1, T, F).astype(np.float32),
            X_res[:, T * F :].astype(np.float32),
            Yc_res,
        )
    except Exception as exc:
        warnings.warn(f"SMOTE failed ({exc}); falling back to weighted CE loss.")
        return X, Yxy, Yc


# ══════════════════════════════════════════════════════════════════════════════
# Single-fold training + evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _train_fold(
    model_kind: str,
    X_tr: np.ndarray, Yxy_tr: np.ndarray, Yc_tr: np.ndarray,
    X_te: np.ndarray, Yxy_te: np.ndarray, Yc_te: np.ndarray,
    norm_meta: dict,
    balance_decision: str,
    class_weights: np.ndarray | None,
    cfg: TrainConfig,
    device: torch.device,
    verbose: bool = False,
) -> dict:
    # Apply balance fix to training fold
    if balance_decision == "smote":
        X_tr, Yxy_tr, Yc_tr = _apply_smote(X_tr, Yxy_tr, Yc_tr)
        ce_w = None
    else:
        ce_w = (
            torch.tensor(class_weights, dtype=torch.float32).to(device)
            if class_weights is not None else None
        )

    ds_tr = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(Yxy_tr),
        torch.from_numpy(Yc_tr),
    )
    ds_te = TensorDataset(
        torch.from_numpy(X_te),
        torch.from_numpy(Yxy_te),
        torch.from_numpy(Yc_te),
    )
    loader_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    loader_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False)

    input_size = X_tr.shape[-1]
    model = build_model(MODEL_KIND_ALIAS.get(model_kind, model_kind), input_size)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    for _ in range(cfg.epochs):
        train_epoch(model, loader_tr, optimizer, device, ce_w, cfg.grad_clip)

    metrics, pred_xy, true_xy, pred_cls, true_cls = evaluate(
        model, loader_te, norm_meta, device
    )

    # KF post-processing for STA-LSTM-H (opt-in; default off — see TrainConfig)
    if model_kind == "lstm_h" and getattr(cfg, "use_kalman_for_lstmh", False):
        behavior_seq = Yc_te   # true behaviour at each test target step
        tracked, kf_rmse, reduction, Q_used, R_used = kalman_with_retune(
            metrics["rmse_backbone"], pred_xy, true_xy,
            behavior_seq=behavior_seq,
            verbose=verbose,
        )
        metrics["rmse"]            = kf_rmse
        metrics["kf_reduction_pct"]= reduction * 100
        metrics["kf_Q"]            = Q_used
        metrics["kf_R"]            = R_used
        final_pred_xy = tracked
    else:
        metrics["rmse"]            = metrics["rmse_backbone"]
        metrics["kf_reduction_pct"]= 0.0
        metrics["kf_Q"]            = float("nan")
        metrics["kf_R"]            = float("nan")
        final_pred_xy = pred_xy

    metrics["model"]    = model_kind
    metrics["pred_xy"]  = final_pred_xy
    metrics["true_xy"]  = true_xy
    metrics["pred_cls"] = pred_cls
    metrics["true_cls"] = true_cls
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Classical-baseline single-fold evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _eval_classical(
    kind:        str,
    true_xy:     np.ndarray,
    true_cls:    np.ndarray,
    noise_pct:   float = UAV_NOISE_PCT,
) -> dict:
    """Run a deterministic classical filter on noisy observations of true_xy.

    Returns a metrics dict shaped like `_train_fold` so it can be merged into
    cv_results: keys {accuracy, f1, rmse_backbone, rmse, latency_ms, memory_mb,
    kf_reduction_pct, kf_Q, kf_R, model, pred_xy, true_xy, pred_cls, true_cls}.
    """
    # Build noisy UAV-style observations
    obs = add_uav_observation_noise(true_xy, noise_pct=noise_pct)

    t0 = time.perf_counter()
    if kind == "kf":
        tracked = kalman_only_baseline(obs)
    elif kind == "ekf_hmm":
        tracked = ekf_hmm_baseline(obs, behavior_seq=true_cls)
    else:
        raise ValueError(f"Unknown classical kind: {kind}")
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    rmse = float(np.sqrt(np.mean(np.sum((tracked - true_xy) ** 2, axis=1))))
    obs_rmse = float(np.sqrt(np.mean(np.sum((obs - true_xy) ** 2, axis=1))))
    reduction = (obs_rmse - rmse) / max(obs_rmse, 1e-12)

    return {
        "model":           kind,
        "accuracy":        float("nan"),
        "f1":              float("nan"),
        "rmse_backbone":   obs_rmse,    # observation noise as backbone
        "rmse":            rmse,
        "latency_ms":      elapsed_ms / max(len(true_xy), 1),
        "memory_mb":       0.0,
        "kf_reduction_pct": reduction * 100,
        "kf_Q":            float("nan"),
        "kf_R":            float("nan"),
        "pred_xy":         tracked,
        "true_xy":         true_xy,
        "pred_cls":        np.full(len(true_cls), -1, dtype=np.int64),
        "true_cls":        np.asarray(true_cls, dtype=np.int64),
    }


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost tabular baseline (industry-standard PLF behaviour classifier)
# ══════════════════════════════════════════════════════════════════════════════

def _eval_xgboost(
    X_tr: np.ndarray, Yc_tr: np.ndarray,
    X_te: np.ndarray, Yc_te: np.ndarray,
    true_xy: np.ndarray,
    seed: int = 42,
) -> dict:
    """Fit an XGBClassifier on per-second 14-feature representations and
    return a metrics dict shaped like `_train_fold` so it can be merged into
    the cross-validation results.

    Reviewer 2 asked for an XGBoost-on-features baseline as the de-facto PLF
    industry baseline. We flatten the (N, T, F) sliding-window representation
    to (N, T*F), fit a gradient-boosted classifier on the behaviour label,
    and report accuracy / macro-F1. We do not run an XGBoost regressor on
    the position channel (the per-second representation has no temporal
    state for double-integrated motion to be meaningful), so RMSE is NaN —
    matching the classical-baseline treatment.
    """
    # Lazy import so the dependency is only required when the baseline runs.
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score, accuracy_score

    N_tr, T, F = X_tr.shape
    X_tr_flat = X_tr.reshape(N_tr, T * F)
    X_te_flat = X_te.reshape(X_te.shape[0], T * F)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    t0 = time.perf_counter()
    clf.fit(X_tr_flat, Yc_tr.astype(int))
    train_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    pred_cls = clf.predict(X_te_flat)
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    acc = float(accuracy_score(Yc_te, pred_cls))
    f1 = float(f1_score(Yc_te, pred_cls, average="macro", zero_division=0))

    # Memory footprint: dump to a buffer and measure (gives a deployment-relevant
    # XGBoost model size; comparable in spirit to PyTorch parameter memory).
    try:
        booster_size_mb = clf.get_booster().save_raw().__len__() / 1024 ** 2
    except Exception:
        booster_size_mb = float("nan")

    return {
        "model":            "xgboost",
        "accuracy":         acc,
        "f1":               f1,
        "rmse_backbone":    float("nan"),
        "rmse":             float("nan"),
        "latency_ms":       elapsed_ms / max(len(Yc_te), 1),
        "memory_mb":        float(booster_size_mb),
        "kf_reduction_pct": 0.0,
        "kf_Q":             float("nan"),
        "kf_R":             float("nan"),
        "pred_xy":          np.full((len(Yc_te), 2), float("nan")),
        "true_xy":          np.asarray(true_xy, dtype=float),
        "pred_cls":         np.asarray(pred_cls, dtype=np.int64),
        "true_cls":         np.asarray(Yc_te, dtype=np.int64),
        "_train_ms":        float(train_ms),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5-fold CV main loop
# ══════════════════════════════════════════════════════════════════════════════

def run_cross_validation(
    data: pd.DataFrame,
    cfg:  TrainConfig,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """5-fold stratified CV for all three models.

    Returns:
        cv_results       — {model_kind: [fold_metrics, …]}
        last_fold_preds  — {model_kind: {pred_xy, true_xy, pred_cls, true_cls}}
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = select_device(cfg.device_pref)

    print(f"\n  Device: {device}  |  Folds: {cfg.folds}  |  "
          f"Epochs: {cfg.epochs}  |  Animals: {list(cfg.animal_ids)}")

    # Class-balance audit once on full dataset
    balance_decision, class_weights = audit_class_balance(
        data, smote_threshold=cfg.smote_threshold
    )

    # Build master sequence array for splitting (use LSTM-H features for stratification)
    _, _, Yc_all, _ = make_sequences(data, MODEL_FEATURE_MAP["lstm_h"], cfg.sequence_length)
    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)

    nn_set = ABLATION_NN_MODELS if cfg.ablation_mode else NN_MODELS
    classical_set: tuple[str, ...] = () if cfg.ablation_mode else CLASSICAL_MODELS
    xgboost_set:   tuple[str, ...] = () if cfg.ablation_mode else XGBOOST_MODELS
    all_models = list(nn_set) + list(classical_set) + list(xgboost_set)
    cv_results:      dict[str, list[dict]] = {m: [] for m in all_models}
    last_fold_preds: dict[str, dict]       = {}

    print(f"\n{'─'*64}")
    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(Yc_all)), Yc_all)
    ):
        print(f"  Fold {fold_idx+1}/{cfg.folds}  "
              f"(train={len(train_idx)}, test={len(test_idx)})")

        # ── Neural-network models ───────────────────────────────────────────
        for kind in nn_set:
            fcols = MODEL_FEATURE_MAP[kind]
            X_all, Yxy_all, Yc_all_m, nm = make_sequences(
                data, fcols, cfg.sequence_length
            )
            X_tr,  Yxy_tr,  Yc_tr  = X_all[train_idx], Yxy_all[train_idx], Yc_all_m[train_idx]
            X_te,  Yxy_te,  Yc_te  = X_all[test_idx],  Yxy_all[test_idx],  Yc_all_m[test_idx]

            verbose_kf = (fold_idx == 0 and kind == "lstm_h")
            fold_metrics = _train_fold(
                kind, X_tr, Yxy_tr, Yc_tr, X_te, Yxy_te, Yc_te,
                nm, balance_decision, class_weights, cfg, device,
                verbose=verbose_kf,
            )
            fold_metrics["fold"] = fold_idx + 1
            cv_results[kind].append(fold_metrics)

            print(f"    {MODEL_DISPLAY[kind]:12s}  "
                  f"Acc={fold_metrics['accuracy']:.3f}  "
                  f"F1={fold_metrics['f1']:.3f}  "
                  f"RMSE={fold_metrics['rmse']:.4f}  "
                  f"Lat={fold_metrics['latency_ms']:.3f}ms")

            if fold_idx == cfg.folds - 1:
                last_fold_preds[kind] = {
                    "pred_xy":  fold_metrics["pred_xy"],
                    "true_xy":  fold_metrics["true_xy"],
                    "pred_cls": fold_metrics["pred_cls"],
                    "true_cls": fold_metrics["true_cls"],
                }

        # ── Classical baselines (KF-only, EKF-HMM) on noisy observations ────
        # Pull the LSTM-H test fold's denormalised true_xy + behaviour to feed
        # both classical filters consistently with the NN evaluation.
        if cfg.ablation_mode:
            continue
        ref = cv_results["lstm_h"][-1]
        true_xy_fold = ref["true_xy"]
        true_cls_fold = ref["true_cls"]
        for kind in classical_set:
            classical = _eval_classical(kind, true_xy_fold, true_cls_fold)
            classical["fold"] = fold_idx + 1
            cv_results[kind].append(classical)
            print(f"    {MODEL_DISPLAY[kind]:12s}  "
                  f"Acc=N/A    F1=N/A    "
                  f"RMSE={classical['rmse']:.4f}  "
                  f"Lat={classical['latency_ms']:.3f}ms")
            if fold_idx == cfg.folds - 1:
                last_fold_preds[kind] = {
                    "pred_xy":  classical["pred_xy"],
                    "true_xy":  classical["true_xy"],
                    "pred_cls": classical["pred_cls"],
                    "true_cls": classical["true_cls"],
                }

        # ── XGBoost tabular baseline (PLF industry baseline) ───────────────
        # Train and evaluate on the same per-fold split as the NN models so
        # the comparison is paired. Uses the 14-feature accelerometry-only
        # representation (no behaviour channel) so it matches the LSTM/GRU
        # baselines' input set.
        for kind in xgboost_set:
            fcols = MODEL_FEATURE_MAP.get(kind, ACCEL_FEAT_COLS)
            X_all_xgb, Yxy_all_xgb, Yc_all_xgb, _ = make_sequences(
                data, fcols, cfg.sequence_length
            )
            X_tr_xgb = X_all_xgb[train_idx]
            X_te_xgb = X_all_xgb[test_idx]
            Yc_tr_xgb = Yc_all_xgb[train_idx]
            Yc_te_xgb = Yc_all_xgb[test_idx]
            true_xy_xgb = Yxy_all_xgb[test_idx]
            xgb_metrics = _eval_xgboost(
                X_tr_xgb, Yc_tr_xgb, X_te_xgb, Yc_te_xgb, true_xy_xgb,
                seed=cfg.seed,
            )
            xgb_metrics["fold"] = fold_idx + 1
            cv_results[kind].append(xgb_metrics)
            print(f"    {MODEL_DISPLAY[kind]:12s}  "
                  f"Acc={xgb_metrics['accuracy']:.3f}  "
                  f"F1={xgb_metrics['f1']:.3f}  "
                  f"RMSE=N/A      "
                  f"Lat={xgb_metrics['latency_ms']:.3f}ms")
            if fold_idx == cfg.folds - 1:
                last_fold_preds[kind] = {
                    "pred_xy":  xgb_metrics["pred_xy"],
                    "true_xy":  xgb_metrics["true_xy"],
                    "pred_cls": xgb_metrics["pred_cls"],
                    "true_cls": xgb_metrics["true_cls"],
                }

        print(f"{'─'*64}")

    return cv_results, last_fold_preds


# ══════════════════════════════════════════════════════════════════════════════
# Leave-one-animal-out (LOAO) cross-validation driver
# ══════════════════════════════════════════════════════════════════════════════

def run_loao_split(
    data_train: pd.DataFrame,
    data_test:  pd.DataFrame,
    cfg:        TrainConfig,
    held_out_animal_id: int,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Single LOAO train/test split: train on `data_train` (e.g. 17 animals),
    evaluate on `data_test` (the held-out animal).

    Reviewer 3 flagged the per-animal 5-fold StratifiedKFold protocol as
    inflating accuracy via temporally-autocorrelated train/test sharing of the
    same animal's bouts. LOAO addresses this by splitting at the animal level:
    no within-animal data is shared between train and test.

    Returns the same `(cv_results, last_fold_preds)` shape as
    `run_cross_validation`, with a single "fold" per model corresponding to
    the held-out-animal split. The Slurm array driver in
    `scripts/run_loao_animal.py` calls this once per `--array=1-18` task with
    `held_out_animal_id` set to that task's animal.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = select_device(cfg.device_pref)

    print(f"\n  [LOAO] held-out animal: {held_out_animal_id:02d}  "
          f"|  Device: {device}  |  Epochs: {cfg.epochs}")
    print(f"  Train: {data_train['animal_id'].nunique()} animals, "
          f"{len(data_train):,} rows  |  "
          f"Test: 1 animal, {len(data_test):,} rows")

    balance_decision, class_weights = audit_class_balance(
        data_train, smote_threshold=cfg.smote_threshold
    )

    nn_set = ABLATION_NN_MODELS if cfg.ablation_mode else NN_MODELS
    classical_set: tuple[str, ...] = () if cfg.ablation_mode else CLASSICAL_MODELS
    xgboost_set:   tuple[str, ...] = () if cfg.ablation_mode else XGBOOST_MODELS
    all_models = list(nn_set) + list(classical_set) + list(xgboost_set)
    cv_results:      dict[str, list[dict]] = {m: [] for m in all_models}
    last_fold_preds: dict[str, dict]       = {}

    print(f"\n{'─'*64}")
    for kind in nn_set:
        fcols = MODEL_FEATURE_MAP[kind]
        (X_tr, Yxy_tr, Yc_tr), (X_te, Yxy_te, Yc_te), nm = make_sequences_split(
            data_train, data_test, fcols, cfg.sequence_length
        )
        verbose_kf = (kind == "lstm_h")
        fold_metrics = _train_fold(
            kind, X_tr, Yxy_tr, Yc_tr, X_te, Yxy_te, Yc_te,
            nm, balance_decision, class_weights, cfg, device,
            verbose=verbose_kf,
        )
        fold_metrics["fold"] = held_out_animal_id
        cv_results[kind].append(fold_metrics)
        print(f"    {MODEL_DISPLAY[kind]:12s}  "
              f"Acc={fold_metrics['accuracy']:.3f}  "
              f"F1={fold_metrics['f1']:.3f}  "
              f"RMSE={fold_metrics['rmse']:.4f}  "
              f"Lat={fold_metrics['latency_ms']:.3f}ms")
        last_fold_preds[kind] = {
            "pred_xy":  fold_metrics["pred_xy"],
            "true_xy":  fold_metrics["true_xy"],
            "pred_cls": fold_metrics["pred_cls"],
            "true_cls": fold_metrics["true_cls"],
        }

    if not cfg.ablation_mode:
        ref = cv_results["lstm_h"][-1]
        true_xy_fold = ref["true_xy"]
        true_cls_fold = ref["true_cls"]
        for kind in classical_set:
            classical = _eval_classical(kind, true_xy_fold, true_cls_fold)
            classical["fold"] = held_out_animal_id
            cv_results[kind].append(classical)
            print(f"    {MODEL_DISPLAY[kind]:12s}  "
                  f"Acc=N/A    F1=N/A    "
                  f"RMSE={classical['rmse']:.4f}  "
                  f"Lat={classical['latency_ms']:.3f}ms")
            last_fold_preds[kind] = {
                "pred_xy":  classical["pred_xy"],
                "true_xy":  classical["true_xy"],
                "pred_cls": classical["pred_cls"],
                "true_cls": classical["true_cls"],
            }
        for kind in xgboost_set:
            fcols = MODEL_FEATURE_MAP.get(kind, ACCEL_FEAT_COLS)
            (X_tr_xgb, Yxy_tr_xgb, Yc_tr_xgb), (X_te_xgb, Yxy_te_xgb, Yc_te_xgb), _ = (
                make_sequences_split(data_train, data_test, fcols, cfg.sequence_length)
            )
            xgb_metrics = _eval_xgboost(
                X_tr_xgb, Yc_tr_xgb, X_te_xgb, Yc_te_xgb,
                Yxy_te_xgb, seed=cfg.seed,
            )
            xgb_metrics["fold"] = held_out_animal_id
            cv_results[kind].append(xgb_metrics)
            print(f"    {MODEL_DISPLAY[kind]:12s}  "
                  f"Acc={xgb_metrics['accuracy']:.3f}  "
                  f"F1={xgb_metrics['f1']:.3f}  "
                  f"RMSE=N/A      "
                  f"Lat={xgb_metrics['latency_ms']:.3f}ms")
            last_fold_preds[kind] = {
                "pred_xy":  xgb_metrics["pred_xy"],
                "true_xy":  xgb_metrics["true_xy"],
                "pred_cls": xgb_metrics["pred_cls"],
                "true_cls": xgb_metrics["true_cls"],
            }
    print(f"{'─'*64}")
    return cv_results, last_fold_preds


# ══════════════════════════════════════════════════════════════════════════════
# Performance Ledger & scoring# ══════════════════════════════════════════════════════════════════════════════

def build_ledger(cv_results: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for kind, folds in cv_results.items():
        df = pd.DataFrame(folds)
        row: dict = {"model": MODEL_DISPLAY[kind]}
        for col in ["accuracy","f1","rmse","latency_ms","memory_mb","kf_reduction_pct"]:
            vals = df[col].to_numpy(float)
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"]  = float(np.std(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def composite_score(ledger: pd.DataFrame) -> pd.DataFrame:
    """Accuracy-first composite ranking for paper-quality model selection.

    Weights: 80 % Accuracy + 10 % Speed (1/Latency) + 10 % Memory efficiency.
    Earlier 50/30/20 weighting buried the most-accurate model behind faster but
    less-accurate baselines — fine for a deployment scoring scheme, but not a
    research paper's "which model wins" headline.

    Each component normalised to [0,1] within the *NN* model set. Classical
    baselines (KF, EKF-HMM) have NaN classification metrics and are excluded
    from the composite score; their RMSE-only comparison is in `rmse_mean`.
    """
    df = ledger.copy()
    nn_mask = df["accuracy_mean"].notna() & df["latency_ms_mean"].notna() & df["memory_mb_mean"].notna()

    df["composite_score"] = float("nan")
    df["rank"] = pd.NA

    if nn_mask.any():
        sub = df.loc[nn_mask].copy()
        acc_max = sub["accuracy_mean"].max() or 1.0
        lat_max = (1.0 / sub["latency_ms_mean"]).max() or 1.0
        mem_max = (1.0 / sub["memory_mb_mean"].replace(0.0, np.nan)).max() or 1.0
        acc_n = sub["accuracy_mean"] / acc_max
        lat_n = (1.0 / sub["latency_ms_mean"]) / lat_max
        mem_n = (1.0 / sub["memory_mb_mean"].replace(0.0, np.nan)) / mem_max
        sub["composite_score"] = 0.80 * acc_n + 0.10 * lat_n + 0.10 * mem_n.fillna(0)
        sub["rank"] = sub["composite_score"].rank(ascending=False).astype("Int64")
        df.loc[nn_mask, "composite_score"] = sub["composite_score"]
        df.loc[nn_mask, "rank"]            = sub["rank"]

    return df.sort_values(["rank", "rmse_mean"], na_position="last").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Statistical significance# ══════════════════════════════════════════════════════════════════════════════

def wilcoxon_tests(cv_results: dict[str, list[dict]]) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests, STA-LSTM-H vs each baseline.

    Reports three metrics (accuracy, F1, RMSE) under a two-sided alternative,
    matching the cohort-level aggregation in `scripts/aggregate_18animals.py`
    so the per-animal and aggregate p-values are produced by the same test.
    Direction of any significant effect is read off the per-model means.

    Classical baselines (KF, EKF-HMM) carry NaN accuracy/F1 — they are tested
    on RMSE only.
    """
    if "lstm_h" not in cv_results or not cv_results["lstm_h"]:
        return pd.DataFrame()

    def _arr(folds, key):
        return np.array([f.get(key, np.nan) for f in folds], dtype=float)

    lstmh_acc  = _arr(cv_results["lstm_h"], "accuracy")
    lstmh_f1   = _arr(cv_results["lstm_h"], "f1")
    lstmh_rmse = _arr(cv_results["lstm_h"], "rmse")

    def _wilcoxon_one(a, b, alt):
        a, b = np.asarray(a, float), np.asarray(b, float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2 or np.all(a[mask] == b[mask]):
            return np.nan, np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = spstats.wilcoxon(a[mask], b[mask], alternative=alt)
        return float(stat), float(p)

    rows = []
    for baseline in ("sta_lstm", "lstm", "gru", "transformer", "kf", "ekf_hmm"):
        if baseline not in cv_results or not cv_results[baseline]:
            continue
        base_acc  = _arr(cv_results[baseline], "accuracy")
        base_f1   = _arr(cv_results[baseline], "f1")
        base_rmse = _arr(cv_results[baseline], "rmse")
        if len(base_rmse) != len(lstmh_rmse):
            continue

        s_a, p_a = _wilcoxon_one(lstmh_acc, base_acc, "two-sided")
        s_f, p_f = _wilcoxon_one(lstmh_f1,  base_f1,  "two-sided")
        s_r, p_r = _wilcoxon_one(lstmh_rmse, base_rmse, "two-sided")

        rows.append({
            "comparison":          f"STA-LSTM-H vs {MODEL_DISPLAY[baseline]}",
            "acc_stat":            s_a,
            "acc_p_value":         p_a,
            "acc_significant":     bool(p_a < 0.05) if np.isfinite(p_a) else False,
            "f1_stat":             s_f,
            "f1_p_value":          p_f,
            "f1_significant":      bool(p_f < 0.05) if np.isfinite(p_f) else False,
            "rmse_stat":           s_r,
            "rmse_p_value":        p_r,
            "rmse_significant":    bool(p_r < 0.05) if np.isfinite(p_r) else False,
            "lstmh_mean_acc":      float(np.nanmean(lstmh_acc))  if np.isfinite(lstmh_acc).any()  else float("nan"),
            "lstmh_mean_f1":       float(np.nanmean(lstmh_f1))   if np.isfinite(lstmh_f1).any()   else float("nan"),
            "lstmh_mean_rmse":     float(np.nanmean(lstmh_rmse)) if np.isfinite(lstmh_rmse).any() else float("nan"),
            f"{baseline}_mean_acc":  float(np.nanmean(base_acc))  if np.isfinite(base_acc).any()  else float("nan"),
            f"{baseline}_mean_f1":   float(np.nanmean(base_f1))   if np.isfinite(base_f1).any()   else float("nan"),
            f"{baseline}_mean_rmse": float(np.nanmean(base_rmse)) if np.isfinite(base_rmse).any() else float("nan"),
        })
    return pd.DataFrame(rows)


# Note: The earlier autoregressive multi-step evaluation was removed in 2026-04
# after a results review. The "rolling window" loop only updated the
# behaviour_prev channel between horizon steps and copied the previous accel
# feature vector unchanged — so the model was asked to predict from
# near-identical input each step, producing a flat error curve regardless of
# horizon. There is no honest path to autoregressive prediction with this
# pipeline (the input is accel features, not positions, so the predicted
# position cannot be fed back). The 1-step-ahead RMSE per fold (already
# captured in `cv_results`) is the only defensible position-prediction metric.


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_loao(
    data_train: pd.DataFrame,
    data_test:  pd.DataFrame,
    cfg:        TrainConfig,
    held_out_animal_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-animal-out single-split entry point.

    Mirrors `run()` but uses `run_loao_split` (one train/test split) instead of
    the per-animal 5-fold protocol. Each held-out animal contributes one row
    per model. The Slurm-array driver `scripts/run_loao_animal.py` calls this
    once per `--array=1-18` task.
    """
    print("\n══ Phase 2: LOAO single-split train/test ═══════════════════════════")
    cv_results, last_preds = run_loao_split(
        data_train, data_test, cfg, held_out_animal_id=held_out_animal_id,
    )

    print("\n══ Phase 3: Performance Ledger (single-split) ══════════════════════")
    ledger = build_ledger(cv_results)
    ledger = composite_score(ledger)

    print("\n  LOAO held-out summary (animal "
          f"{held_out_animal_id:02d}):")
    for _, row in ledger.iterrows():
        acc_str  = (f"Acc={row['accuracy_mean']:.3f}"
                    if pd.notna(row["accuracy_mean"]) else "Acc=  N/A ")
        f1_str   = (f"F1={row['f1_mean']:.3f}"
                    if pd.notna(row["f1_mean"]) else "F1=  N/A ")
        rmse_str = (f"RMSE={row['rmse_mean']:.4f}"
                    if pd.notna(row["rmse_mean"]) else "RMSE=  N/A ")
        print(f"    {row['model']:12s}  {acc_str}  {f1_str}  {rmse_str}")

    out_dir = Path(cfg.output_dir) if cfg.output_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(out_dir / "loao_report.csv", index=False)
    print(f"\n  Saved → {out_dir}/loao_report.csv")

    last_preds_display = {
        MODEL_DISPLAY.get(k, k): v for k, v in last_preds.items()
    }
    np.save(str(out_dir / "loao_last_predictions.npy"), last_preds_display)

    # No per-fold Wilcoxon: one observation per model per held-out animal.
    # Aggregation across the 18 held-out animals is done by
    # scripts/consolidate_loao.py.
    empty_wilcoxon = pd.DataFrame()
    return ledger, empty_wilcoxon


def run(data: pd.DataFrame, cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full CV pipeline; return (comparison_report, wilcoxon_stats)."""
    print("\n══ Phase 2: Cross-Validation ═══════════════════════════════════════")
    cv_results, last_preds = run_cross_validation(data, cfg)

    print("\n══ Phase 3: Performance Ledger ══════════════════════════════════════")
    ledger   = build_ledger(cv_results)
    ledger   = composite_score(ledger)
    stats_df = wilcoxon_tests(cv_results)

    print("\n  5-fold CV summary:")
    for _, row in ledger.iterrows():
        rank_str = f"[{int(row['rank'])}]" if pd.notna(row["rank"]) else "[ -]"
        acc_str  = (f"Acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}"
                    if pd.notna(row["accuracy_mean"]) else "Acc=  N/A         ")
        f1_str   = (f"F1={row['f1_mean']:.3f}"
                    if pd.notna(row["f1_mean"]) else "F1=  N/A ")
        score_str = (f"Score={row['composite_score']:.3f}"
                     if pd.notna(row["composite_score"]) else "Score=  N/A ")
        print(f"    {rank_str} {row['model']:12s}  {acc_str}  {f1_str}  "
              f"RMSE={row['rmse_mean']:.4f}±{row['rmse_std']:.4f}  {score_str}")

    print("\n  Wilcoxon signed-rank tests (paired across folds, one-sided):")
    for _, row in stats_df.iterrows():
        def _fmt(p, sig):
            ps = f"p={p:.4f}" if np.isfinite(p) else "p=N/A    "
            return f"{ps} {'✓' if sig else '✗'}"
        comp = row["comparison"]
        acc  = _fmt(row["acc_p_value"],  row["acc_significant"])
        f1   = _fmt(row["f1_p_value"],   row["f1_significant"])
        rmse = _fmt(row["rmse_p_value"], row["rmse_significant"])
        print(f"    {comp:30s}  Acc {acc}   F1 {f1}   RMSE {rmse}")

    out_dir = Path(cfg.output_dir) if cfg.output_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    comparison = ledger
    comparison.to_csv(out_dir / "comparison_report.csv", index=False)
    stats_df.to_csv(out_dir / "wilcoxon_stats.csv", index=False)
    print(f"\n  Saved → {out_dir}/comparison_report.csv")
    print(f"  Saved → {out_dir}/wilcoxon_stats.csv")

    # Save last-fold predictions for figures, keyed by display name.
    last_preds_display = {
        MODEL_DISPLAY.get(k, k): v for k, v in last_preds.items()
    }
    np.save(str(out_dir / "last_fold_predictions.npy"), last_preds_display)

    return comparison, stats_df
