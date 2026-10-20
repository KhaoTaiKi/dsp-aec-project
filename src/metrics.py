"""
Evaluation Metrics for Acoustic Echo Cancellation (AEC).

Implements:
- Full-clip and post-warmup ERLE (dB) on far-end single-talk segments:
    ERLE = 10 * log10( E[d^2[n]] / (E[e^2[n]] + eps) )
- Sliding-window ERLE curve over time (for convergence visualization and post-warmup checks).
- Root Mean Square Error (RMSE) between residual e[n] and desired target signal:
    RMSE = sqrt( E[(e[n] - target[n])^2] )
  (In S1 echo-only scenario, target[n] = 0, so RMSE = sqrt(E[e^2[n]])).
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def compute_erle_db(
    mic: np.ndarray,
    residual: np.ndarray,
    start_sample: int = 0,
    active_mask: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> float:
    """Compute Echo Return Loss Enhancement (ERLE) in dB on far-end single-talk samples.

    Parameters
    ----------
    mic : np.ndarray, shape (L,)
        Microphone signal d[n] (echo only in S1).
    residual : np.ndarray, shape (L,)
        Residual error signal e[n] = d[n] - y[n].
    start_sample : int
        Optional warmup offset in samples before measuring steady-state ERLE.
    active_mask : Optional[np.ndarray], shape (L,)
        Optional boolean mask restricting evaluation to active far-end single-talk frames.
    eps : float
        Numerical floor preventing log(0) or division by zero.

    Returns
    -------
    erle_db : float
        ERLE in decibels.
    """
    d = np.asarray(mic, dtype=np.float64).reshape(-1)
    e = np.asarray(residual, dtype=np.float64).reshape(-1)
    if d.shape[0] != e.shape[0]:
        raise ValueError("mic and residual must have identical length.")

    d_seg = d[start_sample:]
    e_seg = e[start_sample:]
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool).reshape(-1)[start_sample:]
        if np.any(mask):
            d_seg = d_seg[mask]
            e_seg = e_seg[mask]

    p_d = float(np.mean(d_seg ** 2))
    p_e = float(np.mean(e_seg ** 2))
    return float(10.0 * np.log10((p_d + eps) / (p_e + eps)))


def compute_sliding_erle_db(
    mic: np.ndarray,
    residual: np.ndarray,
    sample_rate: int = 16000,
    window_sec: float = 0.25,
    hop_sec: float = 0.05,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sliding-window ERLE(t) in dB over time.

    Returns
    -------
    times_sec : np.ndarray
        Window center timestamps in seconds.
    erle_curve_db : np.ndarray
        ERLE values (dB) per window.
    """
    d = np.asarray(mic, dtype=np.float64).reshape(-1)
    e = np.asarray(residual, dtype=np.float64).reshape(-1)
    if d.shape[0] != e.shape[0]:
        raise ValueError("mic and residual must have identical length.")

    win_len = max(1, int(round(window_sec * sample_rate)))
    hop_len = max(1, int(round(hop_sec * sample_rate)))
    n_samples = d.shape[0]

    times = []
    values = []
    for start in range(0, n_samples - win_len + 1, hop_len):
        end = start + win_len
        d_win = d[start:end]
        e_win = e[start:end]
        p_d = float(np.mean(d_win ** 2))
        p_e = float(np.mean(e_win ** 2))
        # Only record windows with non-trivial echo energy
        if p_d > 1e-8:
            erle_val = 10.0 * np.log10((p_d + eps) / (p_e + eps))
        else:
            erle_val = 0.0
        times.append((start + 0.5 * win_len) / float(sample_rate))
        values.append(erle_val)

    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)


def compute_rmse(
    residual: np.ndarray,
    target: Optional[np.ndarray] = None,
    start_sample: int = 0,
) -> float:
    """Compute Root Mean Square Error (RMSE) between residual e[n] and target s[n].

    In S1 (echo-only), target is silence (zeros), so RMSE measures residual amplitude.
    """
    e = np.asarray(residual, dtype=np.float64).reshape(-1)[start_sample:]
    if target is None:
        return float(np.sqrt(np.mean(e ** 2)))
    t = np.asarray(target, dtype=np.float64).reshape(-1)[start_sample:]
    if e.shape[0] != t.shape[0]:
        raise ValueError("residual and target must have identical length.")
    return float(np.sqrt(np.mean((e - t) ** 2)))


def check_no_post_warmup_growth(
    residual: np.ndarray,
    mic: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    warmup_sec: float = 1.0,
) -> bool:
    """Verify that post-warmup residual energy does not exhibit persistent growth
    relative to the input microphone echo energy (or finite numerical stability)."""
    e = np.asarray(residual, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(e)):
        return False

    warmup_idx = int(round(warmup_sec * sample_rate))
    if warmup_idx >= e.shape[0] // 2:
        warmup_idx = e.shape[0] // 4

    if mic is not None:
        d = np.asarray(mic, dtype=np.float64).reshape(-1)
        post_erle_db = compute_erle_db(d, e, start_sample=warmup_idx)
        late_erle_db = compute_erle_db(d, e, start_sample=e.shape[0] - warmup_idx)
        return bool(post_erle_db > 0.0 and late_erle_db > 0.0)

    early_rms = float(np.sqrt(np.mean(e[:warmup_idx] ** 2))) + 1e-12
    late_rms = float(np.sqrt(np.mean(e[-warmup_idx:] ** 2)))
    return bool(late_rms <= early_rms * 2.0)
