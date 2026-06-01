"""src/data_loader.py — 10 Hz Zenodo Data Engine.

  * Streaming 10 Hz CSV ingestion (memory-safe, chunk-based).
  * 1-second feature-window extraction: {mean, std, min, max} × 3 axes + magnitude stats.
  * Majority-vote halter behaviour label per window.
  * Double-integration pseudo-position (x, y) from demeaned horizontal accel:
        k = dt² / 1e6
        v_x = cumsum(ax_demeaned) * k
        x   = cumsum(v_x)
  * Class-balance audit: SMOTE if dominant class > 80 %, else weighted CE.
  * Per-fold min-max normalisation.
  * Autoregressive sequence builder for multi-step horizon evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ── Default paths (override via DataConfig) ────────────────────────────────────
# Resolution order (first existing path wins):
#   1. <repo>/data/raw/   — repository-relative default. Place the Zenodo CSVs
#      here after `git clone` and everything runs with no further config
#      (see README → "Dataset").
#   2. The original cluster / local-dev paths, kept so the authors' machines
#      keep working unchanged.
_REPO_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
_CLUSTER_ROOT = Path("/home/bokania/projects/2026-STA-LSTM-H/data/raw")
_LOCAL_FALLBACK_ROOT = Path(
    "/Users/ayub/Library/CloudStorage/OneDrive-CQUniversity/Research/"
    "LiveStock Monitoring/data/cattle-Zenodo"
)
ZENODO_ROOT = next(
    (p for p in (_REPO_RAW, _CLUSTER_ROOT, _LOCAL_FALLBACK_ROOT) if p.exists()),
    _REPO_RAW,
)
CACHE_DIR = Path("Results/python_pipeline/feature_cache")

# ── Feature column registry ────────────────────────────────────────────────────
ACCEL_FEAT_COLS: list[str] = [
    "ax_mean", "ay_mean", "az_mean",
    "ax_std",  "ay_std",  "az_std",
    "ax_min",  "ay_min",  "az_min",
    "ax_max",  "ay_max",  "az_max",
    "mag_mean", "mag_std",
]
# LSTM-H adds normalised previous-window behaviour as an input channel
LSTMH_EXTRA_COLS: list[str] = ["behavior_prev"]
# Honest "previous-second" label channel (behaviour shifted by 1 row, /2). Used
# only by the ablation runs to disentangle architecture from same-step label
# leakage that `behavior_prev` (= behaviour/2 with no shift) carries through
# the 25-step sliding window.
LAG1_EXTRA_COLS: list[str] = ["behavior_prev_lag1"]


@dataclass(frozen=True)
class DataConfig:
    zenodo_root: Path = ZENODO_ROOT
    cache_dir: Path = CACHE_DIR
    max_windows: int = 3_600          # first N 1-s windows per animal (0 = all)
    chunk_size: int = 500_000         # CSV streaming chunk size
    rebuild_cache: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# I. Streaming ingestion helpers
# ══════════════════════════════════════════════════════════════════════════════

def _accel_path(root: Path, animal_id: int) -> Path:
    return root / f"accel-{animal_id:02d}.csv"


def _halter_path(root: Path, animal_id: int) -> Path:
    return root / f"halter-{animal_id:02d}.csv"


def _timestamp_start(csv_path: Path, chunk_size: int) -> pd.Timestamp:
    """Return the first valid timestamp, floored to whole seconds."""
    for chunk in pd.read_csv(csv_path, usecols=["timestamp"], chunksize=chunk_size):
        ts = pd.to_datetime(chunk["timestamp"], errors="coerce").dropna()
        if not ts.empty:
            return ts.iloc[0].floor("s")
    raise ValueError(f"No valid timestamps found in {csv_path}")


def _bin_index(timestamps: pd.Series, start: pd.Timestamp, n_bins: int) -> np.ndarray:
    """Map timestamps to integer bin indices [0, n_bins). Returns -1 for out-of-range."""
    elapsed = (timestamps - start).dt.total_seconds().to_numpy()
    raw = np.floor(elapsed).astype("float64")
    valid = np.isfinite(raw) & (raw >= 0) & (raw < n_bins)
    out = np.full(len(timestamps), -1, dtype=np.int64)
    out[valid] = raw[valid].astype(np.int64)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# II. Feature extraction — stream accel CSV → 14-feature windows
# ══════════════════════════════════════════════════════════════════════════════

def stream_accel_features(
    csv_path: Path,
    start: pd.Timestamp,
    n_bins: int,
    chunk_size: int,
) -> pd.DataFrame:
    """Stream `accel-NN.csv` and return a DataFrame of per-second feature vectors.

    Columns: ax_mean/ay_mean/az_mean, ax_std/ay_std/az_std,
             ax_min/ay_min/az_min, ax_max/ay_max/az_max,
             mag_mean, mag_std, sample_count.
    """
    end = start + pd.to_timedelta(n_bins, unit="s")

    sums   = np.zeros((n_bins, 3), np.float64)
    sums2  = np.zeros((n_bins, 3), np.float64)
    mins   = np.full((n_bins, 3),  np.inf,  np.float64)
    maxs   = np.full((n_bins, 3), -np.inf,  np.float64)
    counts = np.zeros(n_bins, np.int64)
    mag_s  = np.zeros(n_bins, np.float64)
    mag_s2 = np.zeros(n_bins, np.float64)

    for chunk in pd.read_csv(
        csv_path, usecols=["timestamp", "x", "y", "z"], chunksize=chunk_size
    ):
        ts = pd.to_datetime(chunk["timestamp"], errors="coerce")
        # Early-exit: first row of chunk is already past our window
        if ts.notna().any() and ts.dropna().iloc[0] >= end:
            break

        vals = chunk[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").to_numpy()
        keep = ts.notna().to_numpy() & np.isfinite(vals).all(axis=1)
        if not keep.any():
            continue

        idx = _bin_index(ts[keep], start, n_bins)
        in_range = idx >= 0
        if not in_range.any():
            continue

        idx  = idx[in_range]
        vals = vals[keep][in_range]
        mag  = np.sqrt(np.sum(vals ** 2, axis=1))

        for axis in range(3):
            np.add.at(sums[:, axis],  idx, vals[:, axis])
            np.add.at(sums2[:, axis], idx, vals[:, axis] ** 2)
            np.minimum.at(mins[:, axis], idx, vals[:, axis])
            np.maximum.at(maxs[:, axis], idx, vals[:, axis])
        np.add.at(counts, idx, 1)
        np.add.at(mag_s,  idx, mag)
        np.add.at(mag_s2, idx, mag ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums  / counts[:, None]
        vars_ = np.maximum((sums2 / counts[:, None]) - means ** 2, 0.0)
        mag_mean = mag_s  / counts
        mag_var  = np.maximum((mag_s2 / counts) - mag_mean ** 2, 0.0)

    mins[mins ==  np.inf] = np.nan
    maxs[maxs == -np.inf] = np.nan

    out = pd.DataFrame({
        "ax_mean": means[:, 0], "ay_mean": means[:, 1], "az_mean": means[:, 2],
        "ax_std":  np.sqrt(vars_[:, 0]), "ay_std":  np.sqrt(vars_[:, 1]),
        "az_std":  np.sqrt(vars_[:, 2]),
        "ax_min":  mins[:, 0],  "ay_min":  mins[:, 1],  "az_min":  mins[:, 2],
        "ax_max":  maxs[:, 0],  "ay_max":  maxs[:, 1],  "az_max":  maxs[:, 2],
        "mag_mean": mag_mean,
        "mag_std":  np.sqrt(mag_var),
        "sample_count": counts,
    })
    out.loc[counts == 0, :] = np.nan
    return out


# ══════════════════════════════════════════════════════════════════════════════
# III. Halter labels — stream majority-vote per 1-second window
# ══════════════════════════════════════════════════════════════════════════════

def stream_halter_labels(
    csv_path: Path,
    start: pd.Timestamp,
    n_bins: int,
    chunk_size: int,
) -> np.ndarray:
    """Return integer majority-vote label (0/1/2) per 1-second window.

    Returns NaN for empty bins. Zenodo classes: 0=Other, 1=Ruminating, 2=Eating
    (drinking samples are merged into Eating per the Zenodo README).
    """
    end = start + pd.to_timedelta(n_bins, unit="s")
    header = pd.read_csv(csv_path, nrows=0)
    cls_col = next(
        (c for c in header.columns if c.lower() == "classification"), None
    )
    if cls_col is None:
        raise ValueError(f"No 'classification' column found in {csv_path}")

    tally = np.zeros((n_bins, 3), np.int64)

    for chunk in pd.read_csv(
        csv_path, usecols=["timestamp", cls_col], chunksize=chunk_size
    ):
        ts  = pd.to_datetime(chunk["timestamp"], errors="coerce")
        if ts.notna().any() and ts.dropna().iloc[0] >= end:
            break
        cls = pd.to_numeric(chunk[cls_col], errors="coerce").to_numpy()
        ok  = ts.notna().to_numpy() & np.isfinite(cls) & (cls >= 0) & (cls <= 2)
        if not ok.any():
            continue
        idx = _bin_index(ts[ok], start, n_bins)
        in_range = idx >= 0
        if not in_range.any():
            continue
        np.add.at(tally, (idx[in_range], cls[ok][in_range].astype(np.int64)), 1)

    labels = np.full(n_bins, np.nan, np.float64)
    voted  = tally.sum(axis=1) > 0
    labels[voted] = tally[voted].argmax(axis=1).astype(np.float64)
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# IV. Pseudo-position via double-integration of demeaned horizontal accel
# ══════════════════════════════════════════════════════════════════════════════

def pseudo_position_from_accel(
    ax: np.ndarray,
    ay: np.ndarray,
    dt_sec: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Double-integrate demeaned horizontal acceleration → 2-D pseudo-trajectory.

        k    = dt² / 1e6
        v_x  = cumsum(ax - mean(ax)) * k
        x    = cumsum(v_x)

    The scale `k = dt² / 1e6` keeps the doubly-integrated proxy in a
    numerically convenient range; the result is a *relative-motion proxy*,
    not metric position (no GPS receiver in the deployment). The track is
    suitable for short-horizon motion-activity inference but accumulates
    drift under double integration and is not appropriate for fixed-frame
    mapping. NaN mask from the original signal is preserved.
    """
    ax_arr = np.asarray(ax, dtype=np.float64)
    ay_arr = np.asarray(ay, dtype=np.float64)
    nan_mask = np.isnan(ax_arr) | np.isnan(ay_arr)

    ax_f = pd.Series(ax_arr).interpolate(limit_direction="both").bfill().ffill().to_numpy()
    ay_f = pd.Series(ay_arr).interpolate(limit_direction="both").bfill().ffill().to_numpy()

    ax_d = ax_f - np.nanmean(ax_f)
    ay_d = ay_f - np.nanmean(ay_f)

    k  = (float(dt_sec) ** 2) / 1e6
    vx = np.cumsum(ax_d) * k
    vy = np.cumsum(ay_d) * k
    px = np.cumsum(vx)
    py = np.cumsum(vy)

    px[nan_mask] = np.nan
    py[nan_mask] = np.nan
    return px, py


# ══════════════════════════════════════════════════════════════════════════════
# V. Per-animal window builder
# ══════════════════════════════════════════════════════════════════════════════

def build_windows(animal_id: int, cfg: DataConfig) -> pd.DataFrame:
    """Return a DataFrame of 1-second feature windows for one animal.

    Columns: ACCEL_FEAT_COLS + [sample_count, timestamp, animal_id,
                                 behavior, x, y]

    Uses cache if available (and rebuild_cache is False).
    """
    cache_path = cfg.cache_dir / f"animal-{animal_id:02d}_1s_features.csv"
    if cache_path.is_file() and not cfg.rebuild_cache:
        frame = pd.read_csv(cache_path, parse_dates=["timestamp"])
        # Invalidate cache when caller requests more windows than were cached
        # (the cache stores exactly the n_bins built by an earlier run).
        if cfg.max_windows == 0 or len(frame) >= cfg.max_windows:
            if cfg.max_windows > 0:
                frame = frame.iloc[: cfg.max_windows].copy()
            return frame.dropna().reset_index(drop=True)
        print(f"  [cache] {cache_path.name}: cached {len(frame)} < requested "
              f"{cfg.max_windows} → rebuilding")

    accel_path  = _accel_path(cfg.zenodo_root, animal_id)
    halter_path = _halter_path(cfg.zenodo_root, animal_id)

    if not accel_path.is_file():
        raise FileNotFoundError(f"Missing: {accel_path}")
    if not halter_path.is_file():
        raise FileNotFoundError(f"Missing: {halter_path}")

    n_bins = cfg.max_windows if cfg.max_windows > 0 else 86_400  # default: 24 hours
    start  = _timestamp_start(accel_path, cfg.chunk_size)

    frame = stream_accel_features(accel_path, start, n_bins, cfg.chunk_size)
    frame["timestamp"]  = (
        pd.date_range(start=start, periods=n_bins, freq="1s")
        + pd.to_timedelta(0.5, unit="s")
    )
    frame["animal_id"]  = animal_id
    frame["behavior"]   = stream_halter_labels(halter_path, start, n_bins, cfg.chunk_size)
    frame["x"], frame["y"] = pseudo_position_from_accel(
        frame["ax_mean"].to_numpy(), frame["ay_mean"].to_numpy(), dt_sec=1
    )

    frame = frame.dropna().reset_index(drop=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def load_animals(animal_ids: Iterable[int], cfg: DataConfig) -> pd.DataFrame:
    """Load and concatenate feature windows for multiple animals."""
    frames = [build_windows(int(aid), cfg) for aid in animal_ids]
    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# VI. Class-balance audit# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = {0: "Other", 1: "Ruminating", 2: "Eating"}


def audit_class_balance(
    frame: pd.DataFrame,
    smote_threshold: float = 0.80,
) -> tuple[str, np.ndarray | None]:
    """Print class-balance table. Return ('smote'|'weighted'|'none', weights).

    Matches the decision logic in research_pipeline.py Phase 1.
    """
    labels  = frame["behavior"].astype(int).to_numpy()
    classes, counts = np.unique(labels, return_counts=True)
    total   = counts.sum()
    ratios  = counts / total
    dominant = ratios.max()

    print("\n── Class Balance Audit ────────────────────────────────────────────")
    for cls, cnt, rat in zip(classes, counts, ratios):
        bar = "█" * int(rat * 40)
        print(f"  Class {cls} ({CLASS_NAMES.get(int(cls), '?')}): {cnt:6d}  ({rat:.1%})  {bar}")

    if dominant > smote_threshold:
        print(f"\n  ⚠  Dominant class {dominant:.1%} > {smote_threshold:.0%} "
              f"→ SMOTE oversampling")
        return "smote", None

    weights = total / (len(classes) * counts)
    w_norm  = weights / weights.sum() * len(classes)
    print(f"\n  ✓  Balance OK ({dominant:.1%} ≤ {smote_threshold:.0%}) "
          f"→ Weighted CE loss: { {int(c): round(float(w), 3) for c, w in zip(classes, w_norm)} }")
    return "weighted", w_norm


# ══════════════════════════════════════════════════════════════════════════════
# VII. Normalisation & sequence building# ══════════════════════════════════════════════════════════════════════════════

def make_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build (X, Y_xy, Y_cls, norm_meta) sliding-window tensors.

    Normalisation:
        x_norm = (x - min_x) / max(max_x - min_x, ε)
        b_norm = (b - 1) / 2          ∈ [0, 1]
        features z-scored per column.

    Returns:
        X        — float32 (N, seq_len, F)
        Y_xy     — float32 (N, 2)   normalised pseudo-position target
        Y_cls    — int64   (N,)     raw behaviour class {0,1,2}
        norm_meta — dict with keys mu, sigma, xy_min, xy_span (for denormalisation)
    """
    df = frame.copy()
    df["behavior_prev"] = df["behavior"] / 2.0   # behaviour ∈ {0,1,2} → [0,1]
    df["behavior_prev_lag1"] = df["behavior"].shift(1).fillna(0).astype(float) / 2.0

    mu    = df[feature_cols].mean().to_numpy(np.float32)
    sigma = df[feature_cols].std().to_numpy(np.float32)
    sigma[sigma < 1e-9] = 1.0

    xy_min  = df[["x", "y"]].min().to_numpy(np.float32)
    xy_span = (df[["x", "y"]].max() - df[["x", "y"]].min()).to_numpy(np.float32)
    xy_span[xy_span < 1e-9] = 1.0

    feats   = (df[feature_cols].to_numpy(np.float32) - mu) / sigma
    xy_norm = (df[["x", "y"]].to_numpy(np.float32) - xy_min) / xy_span
    cls_arr = df["behavior"].to_numpy(np.int64)

    n = len(df) - seq_len
    if n <= 0:
        raise ValueError(
            f"Cannot build sequences: need > {seq_len} rows, got {len(df)}"
        )
    X   = np.empty((n, seq_len, len(feature_cols)), np.float32)
    Yxy = np.empty((n, 2), np.float32)
    Yc  = np.empty(n, np.int64)
    for i in range(n):
        X[i]   = feats[i : i + seq_len]
        Yxy[i] = xy_norm[i + seq_len]
        Yc[i]  = cls_arr[i + seq_len]

    return X, Yxy, Yc, {
        "mu": mu, "sigma": sigma,
        "xy_min": xy_min, "xy_span": xy_span,
    }


def denormalize_xy(pred_norm: np.ndarray, norm_meta: dict) -> np.ndarray:
    """Inverse of the min-max normalisation applied in make_sequences."""
    return pred_norm * norm_meta["xy_span"] + norm_meta["xy_min"]


def make_sequences_split(
    data_train: pd.DataFrame,
    data_test:  pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    dict,
]:
    """Build train/test sliding-window tensors for leave-one-animal-out CV.

    The key difference vs. `make_sequences` is that the normalisation stats
    (mu, sigma, xy_min, xy_span) are fitted on `data_train` only and applied
    identically to `data_test`. This is the standard LOAO protocol: the
    held-out animal must not contribute to the normalisation, otherwise the
    test-set leak invalidates the CV.

    Returns:
        (X_tr, Yxy_tr, Yc_tr), (X_te, Yxy_te, Yc_te), norm_meta
    """
    def _augment(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["behavior_prev"]      = df["behavior"] / 2.0
        df["behavior_prev_lag1"] = (
            df["behavior"].shift(1).fillna(0).astype(float) / 2.0
        )
        return df

    tr = _augment(data_train)
    te = _augment(data_test)

    mu    = tr[feature_cols].mean().to_numpy(np.float32)
    sigma = tr[feature_cols].std().to_numpy(np.float32)
    sigma[sigma < 1e-9] = 1.0
    xy_min  = tr[["x", "y"]].min().to_numpy(np.float32)
    xy_span = (tr[["x", "y"]].max() - tr[["x", "y"]].min()).to_numpy(np.float32)
    xy_span[xy_span < 1e-9] = 1.0

    def _build(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats   = (df[feature_cols].to_numpy(np.float32) - mu) / sigma
        xy_norm = (df[["x", "y"]].to_numpy(np.float32) - xy_min) / xy_span
        cls_arr = df["behavior"].to_numpy(np.int64)
        n = len(df) - seq_len
        if n <= 0:
            raise ValueError(
                f"Cannot build sequences: need > {seq_len} rows, got {len(df)}"
            )
        X   = np.empty((n, seq_len, len(feature_cols)), np.float32)
        Yxy = np.empty((n, 2), np.float32)
        Yc  = np.empty(n, np.int64)
        for i in range(n):
            X[i]   = feats[i : i + seq_len]
            Yxy[i] = xy_norm[i + seq_len]
            Yc[i]  = cls_arr[i + seq_len]
        return X, Yxy, Yc

    train = _build(tr)
    test  = _build(te)
    norm_meta = {
        "mu": mu, "sigma": sigma,
        "xy_min": xy_min, "xy_span": xy_span,
    }
    return train, test, norm_meta


# ══════════════════════════════════════════════════════════════════════════════
# VIII. Autoregressive multi-step sequence builder
# ══════════════════════════════════════════════════════════════════════════════

def build_autoregressive_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    max_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Yield all (start_idx, horizon) prediction targets for autoregressive eval.

    Returns:
        X_starts — float32 (M, seq_len, F): seed windows for each start index
        Y_ar     — float32 (M, max_horizon, 2): true xy targets at each horizon
        Cls_ar   — int64   (M, max_horizon): true class at each horizon
        norm_meta
    """
    df = frame.copy()
    df["behavior_prev"] = df["behavior"] / 2.0

    mu    = df[feature_cols].mean().to_numpy(np.float32)
    sigma = df[feature_cols].std().to_numpy(np.float32)
    sigma[sigma < 1e-9] = 1.0
    xy_min  = df[["x", "y"]].min().to_numpy(np.float32)
    xy_span = (df[["x", "y"]].max() - df[["x", "y"]].min()).to_numpy(np.float32)
    xy_span[xy_span < 1e-9] = 1.0

    feats   = (df[feature_cols].to_numpy(np.float32) - mu) / sigma
    xy_norm = (df[["x", "y"]].to_numpy(np.float32) - xy_min) / xy_span
    cls_arr = df["behavior"].to_numpy(np.int64)

    max_start = len(df) - seq_len - max_horizon
    M = max(0, max_start - seq_len + 1)

    X_starts = np.empty((M, seq_len, len(feature_cols)), np.float32)
    Y_ar     = np.empty((M, max_horizon, 2), np.float32)
    Cls_ar   = np.empty((M, max_horizon), np.int64)

    for i, t0 in enumerate(range(seq_len, max_start + 1)):
        X_starts[i] = feats[t0 - seq_len : t0]
        for h in range(max_horizon):
            Y_ar[i, h]   = xy_norm[t0 + h]
            Cls_ar[i, h] = cls_arr[t0 + h]

    return X_starts, Y_ar, Cls_ar, {
        "mu": mu, "sigma": sigma,
        "xy_min": xy_min, "xy_span": xy_span,
    }
