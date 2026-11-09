"""
src/metrics/rmse.py
===================
คำนวณค่า Root Mean Squared Error (RMSE)
สำหรับวัดความคลาดเคลื่อนระหว่างสัญญาณเป้าหมายและเอาต์พุต
"""

from __future__ import annotations
import numpy as np


def calculate_rmse(target: np.ndarray, output: np.ndarray) -> float:
    """
    คำนวณค่า Root Mean Squared Error (RMSE)

    Parameters
    ----------
    target : np.ndarray
        สัญญาณเป้าหมาย (Clean Near-End หรือ Ground Truth)
    output : np.ndarray
        สัญญาณที่ได้จากระบบ

    Returns
    -------
    float
        ค่า RMSE
    """
    target = np.asarray(target, dtype=np.float64)
    output = np.asarray(output, dtype=np.float64)

    N = min(len(target), len(output))
    if N == 0:
        raise ValueError("Signals must not be empty.")

    mse = np.mean((target[:N] - output[:N]) ** 2)
    return float(np.sqrt(mse))
