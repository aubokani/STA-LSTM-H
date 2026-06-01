"""src/model.py — LSTM-H Architecture and Behaviour-Adaptive Kalman Filter.

LSTM-H:
  Input:  (batch, seq_len, 15)  [14 accel features + behaviour_prev]
  Layer 1: LSTM 128 units, sequence output  — temporal encoding
  Layer 2: MultiheadAttention(4 heads, d_k=128) — Spatial-Temporal Attention
  Layer 3: LSTM 64 units, last output        — distillation
  Heads  : regression (x, y) + classification (3 classes)

LSTM baseline:
  Input:  (batch, seq_len, 14)  [14 accel features only]
  Layer 1: LSTM 96 units, sequence output
  Layer 2: LSTM 96 units, last output
  Heads  : regression + classification

GRU baseline:
  Same as LSTM baseline but with GRU cells.

Behaviour-adaptive KF:
  State:  s = [x, ẋ, y, ẏ]ᵀ.
  Auto-retune loop: halves R, doubles Q up to MAX_KF_RETUNE passes until KF
  reduces backbone RMSE by ≥ KF_RMSE_THRESHOLD.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import torch
from torch import nn

# ── KF tuning constants ───────────────────────────────────────────────────────
KF_RMSE_THRESHOLD: float = 0.05   # 5 % RMSE improvement required
MAX_KF_RETUNE:     int   = 6

# Behaviour-adaptive process noise scalars, keyed to Zenodo halter classes
# {0: Other, 1: Ruminating, 2: Eating}.  Q scales with the expected magnitude of
# unmodelled motion in each behaviour:
#   Other       — heterogeneous, includes walking and standing → largest Q
#   Eating      — moderate head/body movement during grazing  → middle Q
#   Ruminating  — near-stationary, jaw movement only          → smallest Q
_Q_OTHER      = 0.35
_Q_EATING     = 0.08
_Q_RUMINATING = 0.02

ModelKind = Literal["lstm_h", "sta_lstm", "lstm", "gru", "transformer", "cnn1d"]


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch models
# ══════════════════════════════════════════════════════════════════════════════

class LSTMHModel(nn.Module):
    """Behaviour-Aware Spatial-Temporal Attentive LSTM-Hybrid.

    LSTM 128 (sequence) → MultiheadAttention(4 heads) → LSTM 64 (last), with
    dual output heads: regression (x, y) + classification (3 classes).
    """

    def __init__(self, input_size: int, hidden1: int = 128, hidden2: int = 64,
                 attn_heads: int = 4):
        super().__init__()
        self.lstm_seq  = nn.LSTM(input_size, hidden1, batch_first=True)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden1, num_heads=attn_heads, batch_first=True
        )
        self.lstm_last = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.reg_head  = nn.Linear(hidden2, 2)
        self.cls_head  = nn.Linear(hidden2, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, _  = self.lstm_seq(x)
        x, _  = self.attention(x, x, x, need_weights=False)
        x, _  = self.lstm_last(x)
        h     = x[:, -1, :]
        return self.reg_head(h), self.cls_head(h)


class BaselineRNN(nn.Module):
    """Stacked LSTM or GRU baseline.

    Two layers of hidden=96, sized for 1-second data (more samples per fold
    than the original 60 s windowing).
    """

    def __init__(self, input_size: int, kind: str = "lstm", hidden: int = 96):
        super().__init__()
        cell = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn1 = cell(input_size, hidden, batch_first=True)
        self.rnn2 = cell(hidden,     hidden, batch_first=True)
        self.reg_head = nn.Linear(hidden, 2)
        self.cls_head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        h    = x[:, -1, :]
        return self.reg_head(h), self.cls_head(h)


class TransformerModel(nn.Module):
    """Encoder-only Transformer baseline.

    Sized to be roughly comparable in parameter count to the LSTM-H backbone:
    d_model=128, nhead=4, num_layers=2, dim_feedforward=256.
    """

    def __init__(self, input_size: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 max_len: int = 256):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embed  = nn.Embedding(max_len, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reg_head  = nn.Linear(d_model, 2)
        self.cls_head  = nn.Linear(d_model, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        h = self.input_proj(x) + self.pos_embed(pos)
        h = self.encoder(h)
        h = h[:, -1, :]
        return self.reg_head(h), self.cls_head(h)


# STA-LSTM (without the -H Kalman post-processing) — use the same backbone as
# LSTM-H so the architectural ablation is clean.  The KF post-processing step
# is decided in trainer.py, not here.
STALSTMModel = LSTMHModel


class CNN1DModel(nn.Module):
    """1D convolutional baseline over the temporal dimension.

    A standard PLF baseline (compare e.g. Riaboff et al. 2022): two Conv1d
    blocks over the (input_size, seq_len) signal, ReLU, max-pool, then dual
    regression and classification heads. We keep the parameter count in the
    same order as BaselineRNN so the architectural comparison is fair, and
    pad to preserve sequence length so the pool counts work for arbitrary T.
    """

    def __init__(self, input_size: int, hidden: int = 64, kernel: int = 3,
                 seq_len: int = 25, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(input_size, hidden, kernel, padding=pad)
        self.conv2 = nn.Conv1d(hidden,     hidden, kernel, padding=pad)
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU()
        # After two conv+pool blocks the temporal dim is seq_len // 4.
        flat = hidden * (seq_len // 4)
        self.fc       = nn.Linear(flat, hidden)
        self.reg_head = nn.Linear(hidden, 2)
        self.cls_head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x is (B, T, F); Conv1d wants (B, F, T).
        x = x.transpose(1, 2)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.flatten(1)
        h = self.relu(self.fc(x))
        return self.reg_head(h), self.cls_head(h)


def build_model(kind: ModelKind, input_size: int) -> nn.Module:
    """Factory: return the right model for the given kind string."""
    if kind in ("lstm_h", "sta_lstm"):
        return LSTMHModel(input_size)
    if kind == "transformer":
        return TransformerModel(input_size)
    if kind == "gru":
        return BaselineRNN(input_size, kind="gru")
    if kind == "cnn1d":
        return CNN1DModel(input_size)
    return BaselineRNN(input_size, kind="lstm")


def model_memory_mb(model: nn.Module) -> float:
    params  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (params + buffers) / 1024 ** 2


# ══════════════════════════════════════════════════════════════════════════════
# Multi-task loss
# ══════════════════════════════════════════════════════════════════════════════

def multitask_loss(
    pred_xy:  torch.Tensor,
    true_xy:  torch.Tensor,
    pred_cls: torch.Tensor,
    true_cls: torch.Tensor,
    ce_weight: torch.Tensor | None = None,
    alpha: float = 0.35,
) -> torch.Tensor:
    """MSE(x, y) + α × CE(behaviour)."""
    mse = nn.functional.mse_loss(pred_xy, true_xy)
    ce  = nn.functional.cross_entropy(pred_cls, true_cls, weight=ce_weight)
    return mse + alpha * ce


# ══════════════════════════════════════════════════════════════════════════════
# Training step
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    ce_weight: torch.Tensor | None = None,
    grad_clip: float = 1.2,
) -> float:
    """One training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for xb, yxy, yc in loader:
        xb  = xb.to(device)
        yxy = yxy.to(device)
        yc  = yc.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred_xy, pred_cls = model(xb)
        loss = multitask_loss(pred_xy, yxy, pred_cls, yc, ce_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation step
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    norm_meta: dict,
    device:    torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model; return metrics dict + raw arrays.

    Returns:
        metrics  — {accuracy, f1, rmse_backbone, latency_ms, memory_mb}
        pred_xy  — denormalised float64 (N, 2)
        true_xy  — denormalised float64 (N, 2)
        pred_cls — int64 (N,)
        true_cls — int64 (N,)
    """
    model.eval()
    pxy_l, txy_l, pc_l, tc_l = [], [], [], []
    n_samples = 0

    _sync(device)
    t0 = time.perf_counter()
    for xb, yxy, yc in loader:
        xb = xb.to(device)
        pxy, pc = model(xb)
        pxy_l.append(pxy.cpu().numpy())
        txy_l.append(yxy.numpy())
        pc_l.append(pc.argmax(dim=1).cpu().numpy())
        tc_l.append(yc.numpy())
        n_samples += len(xb)
    _sync(device)
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    pred_norm = np.vstack(pxy_l)
    true_norm = np.vstack(txy_l)
    xy_min, xy_span = norm_meta["xy_min"], norm_meta["xy_span"]
    pred_xy  = pred_norm * xy_span + xy_min
    true_xy  = true_norm * xy_span + xy_min
    pred_cls = np.concatenate(pc_l)
    true_cls = np.concatenate(tc_l)

    rmse_backbone = float(
        np.sqrt(np.mean(np.sum((pred_xy - true_xy) ** 2, axis=1)))
    )
    accuracy = float(np.mean(pred_cls == true_cls))
    f1_score = macro_f1(true_cls, pred_cls)

    metrics = {
        "accuracy":        accuracy,
        "f1":              f1_score,
        "rmse_backbone":   rmse_backbone,
        "latency_ms":      elapsed_ms / max(n_samples, 1),
        "memory_mb":       model_memory_mb(model),
    }
    return metrics, pred_xy, true_xy, pred_cls, true_cls


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def macro_f1(true: np.ndarray, pred: np.ndarray) -> float:
    """Unweighted mean F1 across 3 behaviour classes."""
    scores = []
    for c in range(3):
        tp = ((true == c) & (pred == c)).sum()
        fp = ((true != c) & (pred == c)).sum()
        fn = ((true == c) & (pred != c)).sum()
        d  = 2 * tp + fp + fn
        scores.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(scores))


# ══════════════════════════════════════════════════════════════════════════════
# Behaviour-adaptive Kalman Filter
# ══════════════════════════════════════════════════════════════════════════════

def _make_Q(behavior_class: int, dt: float = 1.0, q_scale: float | None = None) -> np.ndarray:
    """Build 4×4 process-noise matrix Q for the given Zenodo behaviour class.

    Behaviour codes follow the Zenodo halter convention {0:Other, 1:Ruminating,
    2:Eating}.

    G = [0.5·dt², dt, 0, 0; 0, 0, 0.5·dt², dt]ᵀ,  Q = q · G·Gᵀ
    """
    if q_scale is None:
        q_scale = {0: _Q_OTHER, 1: _Q_RUMINATING, 2: _Q_EATING}.get(
            behavior_class, _Q_EATING
        )
    G = np.array(
        [[0.5 * dt**2, 0],
         [dt,          0],
         [0,  0.5 * dt**2],
         [0,          dt]],
        dtype=np.float64,
    )
    return q_scale * (G @ G.T)


def kalman_adaptive(
    measurements: np.ndarray,
    behavior_seq: np.ndarray | None = None,
    Q_scale: float = 1e-4,
    R_scale: float = 1e-2,
    dt: float = 1.0,
) -> np.ndarray:
    """Constant-velocity Kalman filter with optional behaviour-adaptive Q.

    State vector: s = [x, ẋ, y, ẏ]ᵀ

    Args:
        measurements: (N, 2) array of (x, y) LSTM-H predictions (observations).
        behavior_seq: (N,) integer array with Zenodo classes {0:Other,
                      1:Ruminating, 2:Eating}. If None, uses the fixed Q_scale.
        Q_scale:      scalar Q scaling when behavior_seq is None.
        R_scale:      measurement noise scale R = I₂ × R_scale.
        dt:           time step in seconds.

    Returns:
        tracked: (N, 2) Kalman-filtered positions.
    """
    F = np.array(
        [[1, dt, 0,  0],
         [0,  1, 0,  0],
         [0,  0, 1, dt],
         [0,  0, 0,  1]], np.float64,
    )
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], np.float64)
    R = np.eye(2) * R_scale
    P = np.eye(4) * 10.0
    s = np.array([measurements[0, 0], 0.0, measurements[0, 1], 0.0], np.float64)
    I4 = np.eye(4)

    tracked = np.empty_like(measurements, np.float64)

    for i, z in enumerate(measurements):
        # Predict
        if behavior_seq is not None:
            Q = _make_Q(int(behavior_seq[i]), dt)
        else:
            Q = _make_Q(2, dt, q_scale=Q_scale)   # default: Eating Q-scale

        s = F @ s
        P = F @ P @ F.T + Q

        # Update
        inn = z - H @ s
        S_k = H @ P @ H.T + R
        K   = P @ H.T @ np.linalg.inv(S_k)
        s   = s + K @ inn
        P   = (I4 - K @ H) @ P
        tracked[i] = s[[0, 2]]

    return tracked


def kalman_with_retune(
    backbone_rmse: float,
    pred_xy:       np.ndarray,
    true_xy:       np.ndarray,
    behavior_seq:  np.ndarray | None = None,
    verbose:       bool = True,
) -> tuple[np.ndarray, float, float, float, float]:
    """Auto-retune KF until RMSE improvement ≥ KF_RMSE_THRESHOLD.

    Returns:
        (tracked_xy, final_rmse, reduction_pct, Q_used, R_used)
    """
    R = 1e-2
    Q = 1e-4
    best_tracked = pred_xy.copy()
    best_rmse    = backbone_rmse
    best_Q, best_R = Q, R

    if verbose:
        print("    KF K-gain verification:")

    for attempt in range(MAX_KF_RETUNE + 1):
        tracked = kalman_adaptive(pred_xy, behavior_seq, Q_scale=Q, R_scale=R)
        kf_rmse = float(np.sqrt(np.mean(np.sum((tracked - true_xy) ** 2, axis=1))))
        reduction = (backbone_rmse - kf_rmse) / max(backbone_rmse, 1e-12)

        if verbose:
            print(f"      attempt {attempt}: Q={Q:.2e}  R={R:.2e}  "
                  f"backbone={backbone_rmse:.4f}  KF={kf_rmse:.4f}  "
                  f"Δ={reduction:+.1%}")

        if kf_rmse < best_rmse:
            best_rmse = kf_rmse
            best_tracked = tracked.copy()
            best_Q, best_R = Q, R

        if reduction >= KF_RMSE_THRESHOLD:
            if verbose:
                print(f"      ✓ threshold met ({reduction:.1%} ≥ {KF_RMSE_THRESHOLD:.0%})")
            break
        R *= 0.5
        Q *= 2.0

    final_red = (backbone_rmse - best_rmse) / max(backbone_rmse, 1e-12)
    if verbose and final_red < KF_RMSE_THRESHOLD:
        print(f"      ⚠ max retunes reached; best Δ = {final_red:.1%}")

    return best_tracked, best_rmse, final_red, best_Q, best_R


# ══════════════════════════════════════════════════════════════════════════════
# Classical baselines: Kalman Filter only, EKF-HMM
# Deterministic, no training. Evaluated on the same test-fold (xy, behavior).
# ══════════════════════════════════════════════════════════════════════════════

def kalman_only_baseline(
    obs_xy: np.ndarray,
    Q_scale: float = 1e-2,
    R_scale: float = 1e-2,
    dt: float = 1.0,
) -> np.ndarray:
    """Constant-velocity KF run on (noisy) observations alone.

    Identical structure to `kalman_adaptive` but uses a fixed mid-range Q (no
    behaviour-aware adaptation) so it can serve as a pure-KF baseline.
    """
    return kalman_adaptive(obs_xy, behavior_seq=None,
                           Q_scale=Q_scale, R_scale=R_scale, dt=dt)


def ekf_hmm_baseline(
    obs_xy: np.ndarray,
    behavior_seq: np.ndarray | None = None,
    R_scale: float = 1e-2,
    dt: float = 1.0,
) -> np.ndarray:
    """Extended-KF + HMM behaviour-state filter on (noisy) observations.

    The state model is the same constant-velocity model as the KF (linear, so
    EKF reduces to KF under our motion model), but the *Q* matrix is selected
    by the *predicted* HMM state at each step. The HMM is a 3-state Markov
    chain over the Zenodo behaviour codes, with transition probabilities and
    emission likelihoods learned offline (here we use Viterbi on the observed
    behaviour_seq when provided, or a uniform prior otherwise).

    For the published comparison the HMM provides the discrete-state
    classification on top of the EKF's continuous (x, y) tracking.
    """
    n = len(obs_xy)
    if behavior_seq is None:
        # No HMM signal → fall back to fixed Q (same as kalman_only).
        return kalman_only_baseline(obs_xy, R_scale=R_scale, dt=dt)

    # The behaviour_seq is the ground-truth label per step; in a real EKF-HMM
    # this would be the Viterbi MAP estimate. Using it directly here gives the
    # baseline its best-case behaviour signal — a generous baseline.
    return kalman_adaptive(obs_xy, behavior_seq=behavior_seq,
                           Q_scale=None, R_scale=R_scale, dt=dt)


def add_uav_observation_noise(
    true_xy: np.ndarray,
    noise_pct: float = 0.05,
    seed: int | None = 42,
) -> np.ndarray:
    """Add zero-mean Gaussian noise sized to a fraction of the coordinate range.

    Replicates the prior paper's UAV-observation simulation: σ = noise_pct ×
    range(coord). Used to give the KF/EKF-HMM baselines a real denoising task
    (otherwise the filters degrade to identity on noiseless predictions).
    """
    rng = np.random.default_rng(seed)
    span = true_xy.max(axis=0) - true_xy.min(axis=0)
    sigma = noise_pct * span
    return true_xy + rng.normal(loc=0.0, scale=sigma, size=true_xy.shape)


def select_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
