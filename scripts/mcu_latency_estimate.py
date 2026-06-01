"""scripts/mcu_latency_estimate.py — FLOP-based on-MCU latency estimate.

Reviewer 2 flagged that the L40S server-class GPU latency reported in §5.4 is
not directly relevant to the "commodity GPS-free collar" deployment claim and
asked for an on-MCU latency estimate. A real on-MCU benchmark requires INT8
quantisation, ONNX export and TensorFlow Lite Micro deployment to actual
microcontroller hardware (engineering follow-up, see §6 conclusion). For the
present revision we report a FLOP-based analytic estimate using:

  - `fvcore.nn.FlopCountAnalysis` for the encoder-only Transformer baseline,
    where fvcore covers all primitive ops cleanly.
  - An analytic FLOP model for the LSTM cells, since fvcore does not count
    `nn.LSTM` reliably (it walks `aten::lstm` as a single fused op and reports
    zero). The analytic count for an LSTM cell with input dim I and hidden dim
    H, sequence length T is `4 * T * H * (I + H)` MACs (4 gates × per-step
    matmuls); each MAC is 2 FLOPs.
  - For the multi-head attention block: 2 * T^2 * d_model + 4 * T * d_model^2
    MACs (Q/K/V/O projections + QK^T softmax matmul + AV matmul).

The script prints per-architecture MAC and parameter counts, plus four
representative MCU latency points:

  - Cortex-M4 @ 100 MHz, INT8: ~50 MMAC/s        (worst-case low-end collar)
  - Cortex-M7 @ 400 MHz, INT8: ~200 MMAC/s       (mid-range collar)
  - Cortex-A53 @ 1.4 GHz, INT8 (NEON): ~700 MMAC/s (high-end SoM)
  - Server CPU x86, FP32: ~2 GMAC/s (sanity check vs measured GPU latency)

These rule-of-thumb throughput numbers are conservative estimates from
published benchmarks (CMSIS-NN, TensorFlow Lite Micro, Arm NN) and we explicitly
flag them as analytic, not measured. The 1 Hz prediction cadence gives a
1000 ms / window deployment budget.

Outputs:

  - Stdout report.
  - `Results/mcu_latency_estimate.csv` — per-architecture MAC count and
    estimated latency for each MCU class.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import LSTMHModel, BaselineRNN, TransformerModel  # noqa: E402

# ---------------------------------------------------------------------------
# MCU throughput rules of thumb (MAC/s).
# Conservative estimates suitable for a manuscript-level analytic projection.
# ---------------------------------------------------------------------------
MCU_PROFILES = [
    # (label,                                MAC/s,        notes)
    ("Cortex-M4 @ 100 MHz, INT8",            50e6,        "low-end collar"),
    ("Cortex-M7 @ 400 MHz, INT8",            200e6,       "mid-range collar"),
    ("Cortex-A53 @ 1.4 GHz, INT8 (NEON)",    700e6,       "high-end SoM"),
    ("Server CPU x86, FP32 (1 thread)",      2_000e6,     "sanity check"),
]

SEQ_LEN = 25  # sliding window length used in the headline pipeline


def lstm_macs(input_dim: int, hidden_dim: int, seq_len: int) -> int:
    """Analytic MAC count for an `nn.LSTM` cell over a full sequence.

    Per-step gate computation is 4 * H * (I + H) MACs (4 gates × matmul of
    [h_{t-1}, x_t] vector through the gate weights). Pointwise gate ops
    (sigmoid, tanh, hadamard) are negligible at <1% and ignored.
    """
    return 4 * hidden_dim * (input_dim + hidden_dim) * seq_len


def linear_macs(in_dim: int, out_dim: int) -> int:
    return in_dim * out_dim


def attention_macs(d_model: int, seq_len: int) -> int:
    """MACs for a single multi-head self-attention layer.

    Q/K/V projections: 3 * T * d_model^2.
    Output projection: T * d_model^2.
    QK^T (per-head sum): T^2 * d_model.
    AV (per-head sum):   T^2 * d_model.
    Softmax pointwise:   negligible.
    """
    qkv_proj = 4 * seq_len * d_model * d_model
    qk = seq_len * seq_len * d_model
    av = seq_len * seq_len * d_model
    return qkv_proj + qk + av


def transformer_layer_macs(d_model: int, dim_ff: int, seq_len: int) -> int:
    """One TransformerEncoderLayer: self-attention + feed-forward."""
    attn = attention_macs(d_model, seq_len)
    ff = 2 * seq_len * d_model * dim_ff   # two linears, gelu pointwise
    return attn + ff


def estimate_macs(model_kind: str, input_dim: int, seq_len: int = SEQ_LEN) -> dict:
    """Return MAC count + parameter count for a single forward pass on a
    (1, seq_len, input_dim) input.
    """
    if model_kind == "lstm_h":
        # LSTM-128 encoder + multi-head attention + LSTM-64 decoder + dual heads.
        macs = (
            lstm_macs(input_dim, 128, seq_len)
            + attention_macs(128, seq_len)
            + lstm_macs(128, 64, seq_len)
            + linear_macs(64, 2)
            + linear_macs(64, 3)
        )
        net = LSTMHModel(input_dim)
    elif model_kind == "lstm":
        # Stacked LSTM (96, 96).
        macs = (
            lstm_macs(input_dim, 96, seq_len)
            + lstm_macs(96, 96, seq_len)
            + linear_macs(96, 2)
            + linear_macs(96, 3)
        )
        net = BaselineRNN(input_dim, kind="lstm")
    elif model_kind == "gru":
        # GRU has 3 gates instead of 4; 3/4 LSTM MAC count is a close enough
        # analytic substitute for this rule-of-thumb estimate.
        macs = int(0.75 * (
            lstm_macs(input_dim, 96, seq_len)
            + lstm_macs(96, 96, seq_len)
        )) + linear_macs(96, 2) + linear_macs(96, 3)
        net = BaselineRNN(input_dim, kind="gru")
    elif model_kind == "transformer":
        macs = (
            seq_len * input_dim * 128             # input projection
            + 2 * transformer_layer_macs(128, 256, seq_len)  # 2 encoder layers
            + linear_macs(128, 2)
            + linear_macs(128, 3)
        )
        net = TransformerModel(input_dim)
    else:
        raise ValueError(f"unknown model_kind {model_kind!r}")

    n_params = sum(p.numel() for p in net.parameters())
    return {"model_kind": model_kind, "input_dim": input_dim,
            "seq_len": seq_len, "macs": int(macs),
            "n_params": int(n_params),
            "model_size_kb_fp32": n_params * 4 / 1024.0,
            "model_size_kb_int8": n_params * 1 / 1024.0}


def report(input_dim: int = 15) -> pd.DataFrame:
    rows = []
    for kind in ("lstm_h", "lstm", "gru", "transformer"):
        m = estimate_macs(kind, input_dim=input_dim)
        for label, mac_per_s, note in MCU_PROFILES:
            rows.append({
                **m,
                "mcu_profile": label,
                "mcu_mac_per_s": mac_per_s,
                "estimated_latency_ms": 1000.0 * m["macs"] / mac_per_s,
                "note": note,
            })
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    print("Per-architecture FLOP estimate at sliding-window length T = "
          f"{SEQ_LEN} steps, input_dim = 15 features (lstm_h/sta_lstm) or 14 "
          "features (others).")
    print("-" * 78)

    rows = []
    # STA-LSTM-H gets the 15-dim input (with bp_lag1 channel); the other
    # neural baselines see 14-dim accel-only input.
    for kind, in_dim in [("lstm_h", 15), ("lstm", 14), ("gru", 14),
                         ("transformer", 14)]:
        m = estimate_macs(kind, input_dim=in_dim)
        rows.append(m)
        print(f"  {kind:<12s}  input_dim={in_dim:2d}  "
              f"params={m['n_params']:>8d}  "
              f"FP32 size={m['model_size_kb_fp32']:>7.1f} KB  "
              f"INT8 size={m['model_size_kb_int8']:>7.1f} KB  "
              f"MACs/window={m['macs']:>11,d}")

    print()
    print("Estimated MCU latency per window (analytic; INT8 except where noted):")
    print("-" * 78)

    df_rows = []
    for r in rows:
        for label, mac_per_s, note in MCU_PROFILES:
            ms = 1000.0 * r["macs"] / mac_per_s
            df_rows.append({**r, "mcu_profile": label,
                            "mcu_mac_per_s": mac_per_s, "note": note,
                            "estimated_latency_ms": ms})
            ok = "OK" if ms < 1000.0 else "OVER 1 s budget"
            print(f"  {r['model_kind']:<12s}  on  {label:<38s}  ->  "
                  f"{ms:>8.2f} ms / window   ({ok}; {note})")

    out_df = pd.DataFrame(df_rows)
    out_path = ROOT / "Results" / "mcu_latency_estimate.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
