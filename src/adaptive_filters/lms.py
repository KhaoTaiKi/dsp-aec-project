"""
src/adaptive_filters/lms.py
===========================
ตัวกรองปรับตัวได้ Standard Least Mean Squares (LMS)
พร้อมฟังก์ชันการชดเชย Bulk Delay
พัฒนาด้วย Pure NumPy จากพื้นฐานทางคณิตศาสตร์
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def lms_filter(
    x: np.ndarray,
    d: np.ndarray,
    M: int = 128,
    mu: float = 0.005,
    delay_samples: int = 0,
    adapt_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ประมวลผลสัญญาณด้วยตัวกรองปรับตัวได้ Standard LMS

    สมการการปรับค่าน้ำหนัก:
    w[n+1] = w[n] + mu * e[n] * x[n]

    Parameters
    ----------
    x : np.ndarray
        สัญญาณอ้างอิงลำโพง Far-End (Reference Signal)
    d : np.ndarray
        สัญญาณไมโครโฟนที่มีเสียงสะท้อน (Desired / Mic Signal)
    M : int
        จำนวน Tap ของตัวกรอง (Filter Length)
    mu : float
        ขนาดสเต็ปการปรับค่าน้ำหนัก (Standard Step Size)
    delay_samples : int
        จำนวนตัวอย่างที่ต้องการชดเชยดีเลย์ (Bulk Delay)
    adapt_mask : np.ndarray, optional
        บิตแมสก์กำหนดจุดที่ยอมให้ปรับค่าน้ำหนัก

    Returns
    -------
    y : np.ndarray
        สัญญาณเสียงสะท้อนที่ประมาณค่าได้
    e : np.ndarray
        สัญญาณเอาต์พุตคงเหลือหลังตัดเสียงสะท้อน
    w : np.ndarray
        ค่าน้ำหนักสุดท้ายของตัวกรอง
    mse_history : np.ndarray
        ประวัติค่าความผิดพลาดกำลังสองในแต่ละตัวอย่าง
    """
    N = min(len(x), len(d))
    if delay_samples > 0:
        x_in = np.zeros(N, dtype=np.float64)
        x_in[delay_samples:] = x[:N - delay_samples]
    else:
        x_in = x[:N].astype(np.float64)

    d_in = d[:N].astype(np.float64)

    if adapt_mask is None:
        mask = np.ones(N, dtype=bool)
    else:
        mask = np.asarray(adapt_mask, dtype=bool)[:N]

    w = np.zeros(M, dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)
    e = np.zeros(N, dtype=np.float64)
    mse_history = np.zeros(N, dtype=np.float64)
    buf = np.zeros(M, dtype=np.float64)

    for n in range(N):
        buf[1:] = buf[:-1]
        buf[0] = x_in[n]

        y[n] = np.dot(w, buf)
        e[n] = d_in[n] - y[n]
        mse_history[n] = e[n] ** 2

        if mask[n]:
            w += mu * e[n] * buf

    return y, e, w, mse_history
