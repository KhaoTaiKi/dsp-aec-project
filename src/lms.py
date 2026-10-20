"""
Standard Least Mean Squares (LMS) Adaptive Filter & Room Simulation Utilities.

Implements:
- Causal zero-filled FIR input vector x_vec[n] = [x[n], x[n-1], ..., x[n-N+1]]^T
  without circular wraparound or future samples.
- Standard LMS adaptive filter update:
    y[n] = w^T[n] x_vec[n]
    e[n] = d[n] - y[n]
    w[n+1] = w[n] + 2 * mu * e[n] * x_vec[n]
- Piecewise room impulse response simulator supporting:
    * S1: One far-end speaker, one fixed RIR (active).
    * S2: Far-end + near-end speaker, one fixed RIR (reserved hook).
    * S3: Piecewise-fixed RIR switching at declared sample indices (reserved hook).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def create_fixed_room_ir(
    length: int = 128,
    rir_id: str = "rir_0",
) -> np.ndarray:
    """Create a deterministic, causal room impulse response (RIR).

    Parameters
    ----------
    length : int
        Total number of FIR taps (default 128 taps = 8 ms at 16 kHz).
    rir_id : str
        Identifier ('rir_0' for static baseline room, 'rir_1' for reserved S3 room).

    Returns
    -------
    h_room : np.ndarray, shape (length,)
        Causal room impulse response normalized so its L2 norm is bounded and realistic.
    """
    if length < 16:
        raise ValueError("RIR length must be at least 16 taps.")

    h = np.zeros(length, dtype=np.float64)

    if rir_id == "rir_0":
        # Direct path + 6 discrete early reflections + exponential late tail
        sparse_taps: Dict[int, float] = {
            0: 0.80,
            5: -0.35,
            12: 0.24,
            23: -0.16,
            37: 0.11,
            54: -0.07,
            78: 0.045,
            102: -0.025,
        }
        for idx, amp in sparse_taps.items():
            if idx < length:
                h[idx] = amp
        # Add deterministic damped sinusoidal room modes for realistic coloration
        n = np.arange(length, dtype=np.float64)
        tail = 0.04 * np.exp(-n / 28.0) * np.cos(2.0 * np.pi * n / 11.0)
        tail[:4] = 0.0
        h = h + tail
    elif rir_id == "rir_1":
        # Secondary room path reserved for S3 moving-path scenario
        sparse_taps = {
            0: 0.65,
            8: 0.32,
            19: -0.22,
            33: 0.15,
            49: -0.09,
            71: 0.05,
            96: -0.03,
        }
        for idx, amp in sparse_taps.items():
            if idx < length:
                h[idx] = amp
        n = np.arange(length, dtype=np.float64)
        tail = 0.035 * np.exp(-n / 32.0) * np.sin(2.0 * np.pi * n / 13.0)
        tail[:6] = 0.0
        h = h + tail
    else:
        raise ValueError(f"Unknown rir_id: {rir_id}")

    assert np.all(np.isfinite(h)), "RIR contains non-finite values."
    return h


def simulate_microphone_signal(
    farend: np.ndarray,
    rir_list: Sequence[np.ndarray],
    rir_switch_samples: Optional[Sequence[int]] = None,
    nearend: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate microphone signal d[n] using causal linear convolution without wraparound.

    Supports S1 (single fixed RIR, no nearend), S2 (single fixed RIR + nearend),
    and S3 (piecewise RIR switching at declared sample indices).

    Parameters
    ----------
    farend : np.ndarray, shape (L,)
        Clean far-end reference signal x[n].
    rir_list : Sequence[np.ndarray]
        One or more causal FIR room impulse responses.
    rir_switch_samples : Optional[Sequence[int]]
        Sample indices where the active RIR switches to the next entry in rir_list.
        Must be empty/None for S1 and S2.
    nearend : Optional[np.ndarray], shape (L,)
        Optional near-end clean speech signal s[n] (None for S1).

    Returns
    -------
    mic : np.ndarray, shape (L,)
        Simulated microphone signal d[n] = echo[n] + target[n].
    echo : np.ndarray, shape (L,)
        True acoustic echo component (x * h_room)[n].
    target : np.ndarray, shape (L,)
        Desired target signal (all zeros for S1 echo-only; nearend s[n] for S2/S3).
    """
    x = np.asarray(farend, dtype=np.float64).reshape(-1)
    n_samples = x.shape[0]
    if len(rir_list) == 0:
        raise ValueError("rir_list must contain at least one RIR.")

    switches = list(rir_switch_samples) if rir_switch_samples is not None else []
    if len(switches) != len(rir_list) - 1:
        raise ValueError("Number of rir_switch_samples must equal len(rir_list) - 1.")

    if len(switches) == 0:
        # S1 / S2 fast causal FIR convolution with zero initial history
        h = np.asarray(rir_list[0], dtype=np.float64).reshape(-1)
        m_taps = h.shape[0]
        echo = np.zeros(n_samples, dtype=np.float64)
        x_padded = np.concatenate([np.zeros(m_taps - 1, dtype=np.float64), x])
        h_rev = h[::-1]
        for n in range(n_samples):
            echo[n] = float(np.dot(h_rev, x_padded[n : n + m_taps]))
    else:
        # S3 piecewise-fixed RIR simulation with causal zero-filled history
        boundaries = [0] + [int(s) for s in switches] + [n_samples]
        max_taps = max(len(r) for r in rir_list)
        x_padded = np.concatenate([np.zeros(max_taps - 1, dtype=np.float64), x])
        echo = np.zeros(n_samples, dtype=np.float64)
        for seg_idx, h_raw in enumerate(rir_list):
            h = np.asarray(h_raw, dtype=np.float64).reshape(-1)
            m_taps = h.shape[0]
            h_rev = h[::-1]
            start_n = boundaries[seg_idx]
            end_n = boundaries[seg_idx + 1]
            offset = max_taps - m_taps
            for n in range(start_n, end_n):
                echo[n] = float(np.dot(h_rev, x_padded[n + offset : n + offset + m_taps]))

    if nearend is not None:
        target = np.asarray(nearend, dtype=np.float64).reshape(-1)
        if target.shape[0] != n_samples:
            raise ValueError("nearend length must match farend length.")
    else:
        target = np.zeros(n_samples, dtype=np.float64)

    mic = echo + target
    assert np.all(np.isfinite(mic)), "Simulated microphone signal contains non-finite values."
    return mic, echo, target


def compute_lms_stability_bound(
    farend: np.ndarray, num_taps: int
) -> Tuple[float, float, float, float]:
    """Compute average input power P_x, peak instantaneous window energy, and LMS bounds.

    For standard LMS update w[n+1] = w[n] + 2 * mu * e[n] * x_vec[n]:
    - Stationary mean-power bound: mu_avg_bound = 1.0 / (num_taps * P_x)
    - Strict instantaneous bound for non-stationary speech bursts:
      mu_peak_bound = 1.0 / max_n( ||x_vec[n]||^2 )

    Returns
    -------
    p_x : float
        Mean squared power of farend signal E[x^2[n]].
    max_window_energy : float
        Maximum instantaneous tap-vector energy max_n(x_vec^T[n] x_vec[n]).
    mu_avg_bound : float
        Stationary bound 1 / (N * P_x).
    mu_peak_bound : float
        Strict non-stationary burst bound 1 / max_n(||x_vec[n]||^2).
    """
    x = np.asarray(farend, dtype=np.float64).reshape(-1)
    p_x = float(np.mean(x ** 2))
    if p_x <= 0.0:
        raise ValueError("Input signal power P_x must be positive.")
    # Sliding sum of squares of length num_taps
    win_energies = np.convolve(x ** 2, np.ones(num_taps, dtype=np.float64), mode="full")[: len(x)]
    max_window_energy = float(np.max(win_energies))
    mu_avg_bound = 1.0 / (float(num_taps) * p_x)
    mu_peak_bound = 1.0 / max(max_window_energy, 1e-12)
    return p_x, max_window_energy, mu_avg_bound, mu_peak_bound


def run_lms(
    farend: np.ndarray,
    mic: np.ndarray,
    num_taps: int = 128,
    mu: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run causal Standard LMS adaptive filter.

    Equations:
        x_vec[n] = [x[n], x[n-1], ..., x[n-N+1]]^T  (zero-filled for n < N-1)
        y[n]     = w^T[n] x_vec[n]
        e[n]     = d[n] - y[n]
        w[n+1]   = w[n] + 2 * mu * e[n] * x_vec[n]

    Parameters
    ----------
    farend : np.ndarray, shape (L,)
        Far-end reference signal x[n].
    mic : np.ndarray, shape (L,)
        Microphone signal d[n].
    num_taps : int
        Filter length N.
    mu : float
        LMS step size.

    Returns
    -------
    echo_estimate : np.ndarray, shape (L,)
        Estimated echo signal y[n].
    residual : np.ndarray, shape (L,)
        Error / residual signal e[n] = d[n] - y[n].
    weights : np.ndarray, shape (num_taps,)
        Final learned filter weights w[L].
    """
    x = np.asarray(farend, dtype=np.float64).reshape(-1)
    d = np.asarray(mic, dtype=np.float64).reshape(-1)

    if x.shape[0] != d.shape[0]:
        raise ValueError(f"Length mismatch: farend ({x.shape[0]}) != mic ({d.shape[0]})")
    if num_taps <= 0:
        raise ValueError("num_taps must be positive.")
    if mu <= 0.0:
        raise ValueError("mu must be positive.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(d))):
        raise ValueError("Input signals must contain only finite values.")

    n_samples = x.shape[0]
    w = np.zeros(num_taps, dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.float64)
    e = np.zeros(n_samples, dtype=np.float64)

    # Zero-filled history buffer: x_vec[0] = x[n], x_vec[1] = x[n-1], ..., x_vec[N-1] = x[n-N+1]
    x_buf = np.zeros(num_taps, dtype=np.float64)

    for n in range(n_samples):
        # Shift history strictly causally without circular indexing
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = x[n]

        y_n = float(np.dot(w, x_buf))
        e_n = d[n] - y_n

        y[n] = y_n
        e[n] = e_n

        w = w + (2.0 * mu * e_n) * x_buf

    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(e)) and np.all(np.isfinite(w))):
        raise FloatingPointError("LMS filter diverged (non-finite values encountered).")

    return y, e, w
