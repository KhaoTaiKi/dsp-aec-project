"""
Normalized Least Mean Squares (NLMS) Adaptive Filter.

Implements:
- Causal zero-filled FIR input vector x_vec[n] = [x[n], x[n-1], ..., x[n-N+1]]^T
  without circular wraparound or future samples.
- Normalized LMS adaptive filter update:
    y[n]   = w^T[n] x_vec[n]
    e[n]   = d[n] - y[n]
    w[n+1] = w[n] + (mu / (x_vec^T[n] x_vec[n] + eps)) * e[n] * x_vec[n]
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def run_nlms(
    farend: np.ndarray,
    mic: np.ndarray,
    num_taps: int = 128,
    mu: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run causal Normalized LMS (NLMS) adaptive filter.

    Parameters
    ----------
    farend : np.ndarray, shape (L,)
        Far-end reference signal x[n].
    mic : np.ndarray, shape (L,)
        Microphone signal d[n].
    num_taps : int
        Filter length N.
    mu : float
        Normalized step size (0 < mu < 2 for stability).
    eps : float
        Regularization constant preventing division by zero during silence.

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
    if mu <= 0.0 or mu >= 2.0:
        raise ValueError("NLMS step size mu must satisfy 0 < mu < 2.")
    if eps <= 0.0:
        raise ValueError("Regularization eps must be positive.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(d))):
        raise ValueError("Input signals must contain only finite values.")

    n_samples = x.shape[0]
    w = np.zeros(num_taps, dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.float64)
    e = np.zeros(n_samples, dtype=np.float64)

    # Zero-filled history buffer: x_buf[0] = x[n], x_buf[1] = x[n-1], ..., x_buf[N-1] = x[n-N+1]
    x_buf = np.zeros(num_taps, dtype=np.float64)

    for n in range(n_samples):
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = x[n]

        y_n = float(np.dot(w, x_buf))
        e_n = d[n] - y_n

        y[n] = y_n
        e[n] = e_n

        norm_power = float(np.dot(x_buf, x_buf)) + eps
        w = w + (mu / norm_power) * e_n * x_buf

    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(e)) and np.all(np.isfinite(w))):
        raise FloatingPointError("NLMS filter diverged (non-finite values encountered).")

    return y, e, w
