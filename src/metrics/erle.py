"""
src/metrics/erle.py
===================
คำนวณค่า Echo Return Loss Enhancement (ERLE)
เพื่อวัดประสิทธิภาพการลดทอนพลังงานเสียงสะท้อนในหน่วยเดซิเบล (dB)
"""

from __future__ import annotations
import numpy as np


def calculate_erle(
    echo_before: np.ndarray,
    residual_echo: np.ndarray,
    burn_in_samples: int = 0,
    eps: float = 1e-12
) -> float:
    """
    คำนวณค่า ERLE รวม (Scalar ERLE in dB)

    สูตร:
    ERLE = 10 * log10(E[d^2[n]] / E[e^2[n]])

    Parameters
    ----------
    echo_before : np.ndarray
        สัญญาณเสียงสะท้อนก่อนตัด (d[n])
    residual_echo : np.ndarray
        สัญญาณคงเหลือหลังตัด (e[n])
    burn_in_samples : int
        จำนวนตัวอย่างช่วงเริ่มต้นที่ไม่นำมาคิด (Transient period)
    eps : float
        ค่าคงที่ขนาดเล็กเพื่อป้องกันการหารด้วยศูนย์

    Returns
    -------
    float
        ค่า ERLE ในหน่วยเดซิเบล (dB) ยิ่งสูงยิ่งตัดเสียงสะท้อนได้ดี
    """
    N = min(len(echo_before), len(residual_echo))
    d = echo_before[burn_in_samples:N]
    e = residual_echo[burn_in_samples:N]

    power_before = np.mean(d ** 2)
    power_after = np.mean(e ** 2)

    erle = 10.0 * np.log10((power_before + eps) / (power_after + eps))
    return float(erle)


def calculate_erle_curve(
    echo_before: np.ndarray,
    residual_echo: np.ndarray,
    window_len: int = 512,
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    คำนวณเส้นโค้ง ERLE ตามเวลาด้วย Moving Average Window (Time-Varying ERLE Curve)

    Parameters
    ----------
    echo_before : np.ndarray
        สัญญาณก่อนตัด
    residual_echo : np.ndarray
        สัญญาณหลังตัด
    window_len : int
        ขนาดหน้าต่างเฉลี่ยกำลังงาน

    Returns
    -------
    indices : np.ndarray
        ตำแหน่งเวลา (Sample Indices)
    erle_curve : np.ndarray
        ค่า ERLE (dB) ในแต่ละช่วงเวลา
    """
    N = min(len(echo_before), len(residual_echo))
    half = window_len // 2
    indices = []
    erle_vals = []

    for i in range(half, N - half, window_len // 4):
        p_in = np.mean(echo_before[i - half:i + half] ** 2)
        p_out = np.mean(residual_echo[i - half:i + half] ** 2)
        val = 10.0 * np.log10((p_in + eps) / (p_out + eps))
        indices.append(i)
        erle_vals.append(val)

    return np.array(indices), np.array(erle_vals)
