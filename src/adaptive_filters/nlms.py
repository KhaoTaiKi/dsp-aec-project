"""
src/adaptive_filters/nlms.py
============================
ตัวกรองปรับตัวได้ Normalized Least Mean Squares (NLMS)
พร้อมฟังก์ชันการชดเชย Bulk Delay (Time-of-Flight Compensation)
พัฒนาด้วย Pure NumPy จากพื้นฐานทางคณิตศาสตร์
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def nlms_filter(
    x: np.ndarray,
    d: np.ndarray,
    M: int = 128,
    mu: float = 0.5,
    delay_samples: int = 0,
    eps: float = 1e-8,
    adapt_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ประมวลผลสัญญาณด้วยตัวกรองปรับตัวได้ NLMS

    สมการการปรับค่าน้ำหนัก:
    w[n+1] = w[n] + (mu / (||x[n]||^2 + eps)) * e[n] * x[n]

    Parameters
    ----------
    x : np.ndarray
        สัญญาณอ้างอิงลำโพง Far-End (Reference Signal)
    d : np.ndarray
        สัญญาณไมโครโฟนที่มีเสียงสะท้อน (Desired / Mic Signal)
    M : int
        จำนวน Tap ของตัวกรอง (Filter Length)
    mu : float
        ขนาดสเต็ปการปรับค่าน้ำหนัก (Normalized Step Size, 0 < mu < 2)
    delay_samples : int
        จำนวนตัวอย่างที่ต้องการชดเชยดีเลย์ (Bulk Delay / Time-of-Flight)
    eps : float
        ค่าคงที่ขนาดเล็กเพื่อป้องกันการหารด้วยศูนย์ (Regularization)
    adapt_mask : np.ndarray, optional
        บิตแมสก์กำหนดจุดที่ยอมให้ปรับค่าน้ำหนัก (True = adapt, False = freeze)

    Returns
    -------
    y : np.ndarray
        สัญญาณเสียงสะท้อนที่ประมาณค่าได้ (Estimated Echo)
    e : np.ndarray
        สัญญาณเอาต์พุตคงเหลือหลังตัดเสียงสะท้อน (Error / Residual Signal)
    w : np.ndarray
        ค่าน้ำหนักสุดท้ายของตัวกรอง (Final Filter Weights)
    mse_history : np.ndarray
        ประวัติค่าความผิดพลาดกำลังสองในแต่ละตัวอย่าง (e[n]^2)
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
        # เลื่อนบัฟเฟอร์ข้อมูลย้อนหลัง
        buf[1:] = buf[:-1]
        buf[0] = x_in[n]

        # ประมาณค่าเสียงสะท้อน
        y[n] = np.dot(w, buf)

        # คำนวณสัญญาณผิดพลาดคงเหลือ
        e[n] = d_in[n] - y[n]
        mse_history[n] = e[n] ** 2

        # ปรับค่าน้ำหนักแบบ Normalized Energy
        if mask[n]:
            norm = np.dot(buf, buf)
            w += (mu * e[n] / (norm + eps)) * buf

    return y, e, w, mse_history
