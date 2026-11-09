"""
src/common/dsp_utils.py
=======================
รวบรวมฟังก์ชันอรรถประโยชน์ทาง DSP ที่ใช้งานได้ดี
(คัดสรรและปรับปรุงให้เสถียรจากโค้ดทดลอง ProjectDSPFinal):
1. check_eda: ตรวจสอบคุณลักษณะ Sampling Rate, Nyquist, และ Sampling Interval
2. add_noise: จำลองการเพิ่มสัญญาณรบกวนตามระดับ SNR (dB) ที่กำหนด
3. correlation_offset: ตรวจหา Delay / Lag ระหว่าง 2 สัญญาณด้วย Cross-Correlation แบบ Normalized
4. shift_signal: ปรับสัญญาณให้ตรงกันตามค่า Lag ที่ตรวจพบ
5. iir_filter: ตัวกรอง IIR อเนกประสงค์ (Butterworth, Chebyshev I & II, Bessel, Elliptic)
6. resample_ratio: แปลง Sampling Rate ด้วยอัตราส่วนจำนวนเต็มแม่นยำ (Fraction & Polyphase)
"""

from __future__ import annotations
import numpy as np
from scipy import signal
from fractions import Fraction
from typing import Tuple, Dict, Any, Optional


def check_eda(fs: int) -> Dict[str, float]:
    """
    ตรวจสอบข้อมูลพื้นฐานทางเวลาและความถี่ของสัญญาณ (Exploratory Data Analysis)
    
    Parameters
    ----------
    fs : int
        Sampling Rate (Hz)
        
    Returns
    -------
    dict
        {'fs': fs, 'nyquist': nyquist_hz, 'sampling_interval': dt_sec}
    """
    nyquist = fs / 2.0
    dt = 1.0 / fs
    print(f"📊 [EDA] Sampling Rate: {fs} Hz | Nyquist: {nyquist} Hz | Sampling Interval (dt): {dt:.6f} s")
    return {"fs": fs, "nyquist": nyquist, "sampling_interval": dt}


def add_noise(
    audio: np.ndarray,
    snr_db: float,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    เพิ่มสัญญาณรบกวนเกาส์เซียนสีขาว (AWGN) ลงในสัญญาณเสียงตามระดับ SNR (dB) ที่ระบุ
    
    Parameters
    ----------
    audio : np.ndarray
        สัญญาณเสียงต้นฉบับ 1D
    snr_db : float
        อัตราส่วนสัญญาณต่อสัญญาณรบกวน (Signal-to-Noise Ratio) ในหน่วยเดซิเบล (dB)
    seed : int, optional
        ตัวตั้งต้นสร้างตัวเลขสุ่ม (Default = 42)
        
    Returns
    -------
    noisy_audio : np.ndarray
        สัญญาณเสียงที่ผสมสัญญาณรบกวนแล้ว
    noise : np.ndarray
        เฉพาะสัญญาณรบกวนที่สร้างขึ้น
    """
    rng = np.random.default_rng(seed)
    audio = np.asarray(audio, dtype=np.float64)
    signal_power = np.mean(audio ** 2)

    if signal_power == 0:
        raise ValueError("Audio signal has zero power (signal is all zeros).")

    # คำนวณพลังงานสัญญาณรบกวนจากสูตร SNR (dB) = 10 * log10(P_signal / P_noise)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), len(audio))
    noisy_audio = audio + noise

    return noisy_audio, noise


def correlation_offset(
    x1: np.ndarray,
    x2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    ตรวจหาความหน่วงเวลา (Lag / Time Offset) ระหว่าง 2 สัญญาณ
    ด้วยเทคนิค Cross-Correlation แบบ Normalized (Z-Score)
    
    Parameters
    ----------
    x1 : np.ndarray
        สัญญาณอ้างอิง (Reference Signal)
    x2 : np.ndarray
        สัญญาณเป้าหมายที่อาจมีความหน่วง (Delayed Signal)
        
    Returns
    -------
    corr : np.ndarray
        ค่า Correlation ครบทุกช่วง Lag
    lags : np.ndarray
        แกน Lag (ตัวอย่าง / Samples)
    best_lag : int
        ค่า Lag ที่ตำแหน่ง Correlation สูงสุด (d > 0 หมายถึง x2 หน่วงตามหลัง x1)
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    # ปรับ Normalization ป้องกัน Scale เบี่ยงเบน
    std1 = np.std(x1)
    std2 = np.std(x2)
    x1_norm = (x1 - np.mean(x1)) / (std1 if std1 > 0 else 1.0)
    x2_norm = (x2 - np.mean(x2)) / (std2 if std2 > 0 else 1.0)

    corr = signal.correlate(x1_norm, x2_norm, mode="full")
    lags = signal.correlation_lags(len(x1_norm), len(x2_norm), mode="full")

    best_lag = int(lags[np.argmax(corr)])
    return corr, lags, best_lag


def shift_signal(
    x1: np.ndarray,
    x2: np.ndarray,
    d: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    เลื่อนชดเชยและตัดส่วนขอบของสัญญาณทั้งสองให้ตรงกันตามค่า Lag `d`
    
    Parameters
    ----------
    x1 : np.ndarray
        สัญญาณตัวที่ 1
    x2 : np.ndarray
        สัญญาณตัวที่ 2
    d : int
        ค่าความหน่วง (จาก correlation_offset)
        
    Returns
    -------
    x1_aligned, x2_aligned : Tuple[np.ndarray, np.ndarray]
        สัญญาณทั้งสองที่เลื่อนตรงกันและมีความยาวเท่ากัน
    """
    if d < 0:
        x1_aligned = x1[:d]
        x2_aligned = x2[-d:]
    elif d > 0:
        x1_aligned = x1[d:]
        x2_aligned = x2[:-d]
    else:
        x1_aligned = x1.copy()
        x2_aligned = x2.copy()
        
    n_min = min(len(x1_aligned), len(x2_aligned))
    return x1_aligned[:n_min], x2_aligned[:n_min]


def iir_filter(
    x: np.ndarray,
    fs: int,
    fil_type: str = "butter",
    order: int = 4,
    cutoff: Any = 1000.0,
    btype: str = "lowpass",
    rp: float = 1.0,
    rs: float = 40.0
) -> np.ndarray:
    """
    ตัวกรอง IIR อเนกประสงค์ ครอบคลุมตัวกรองที่ศึกษาในห้องปฏิบัติการ DSP
    (Butterworth, Chebyshev Type I & II, Bessel, Elliptic)
    
    Parameters
    ----------
    x : np.ndarray
        สัญญาณอินพุต
    fs : int
        Sampling Rate (Hz)
    fil_type : str
        ประเภทตัวกรอง ('butter', 'cheby1', 'cheby2', 'bessel', 'elliptic')
    order : int
        อันดับของตัวกรอง (Filter Order)
    cutoff : float or list
        ความถี่ Cutoff (Hz)
    btype : str
        'lowpass', 'highpass', 'bandpass', 'bandstop'
    rp : float
        Passband ripple (dB) สำหรับ cheby1, elliptic
    rs : float
        Stopband attenuation (dB) สำหรับ cheby2, elliptic
        
    Returns
    -------
    np.ndarray
        สัญญาณเอาต์พุตที่ผ่านการกรองแล้ว
    """
    fil_type = fil_type.lower()
    if fil_type == "butter":
        b, a = signal.butter(order, cutoff, fs=fs, btype=btype)
    elif fil_type == "cheby1":
        b, a = signal.cheby1(order, rp, cutoff, fs=fs, btype=btype)
    elif fil_type == "cheby2":
        b, a = signal.cheby2(order, rs, cutoff, fs=fs, btype=btype)
    elif fil_type == "bessel":
        b, a = signal.bessel(order, cutoff, btype=btype, fs=fs, norm="mag")
    elif fil_type == "elliptic":
        b, a = signal.ellip(order, rp, rs, cutoff, btype=btype, fs=fs)
    else:
        raise ValueError(f"Unsupported filter type: {fil_type}")

    return signal.lfilter(b, a, x)


def resample_ratio(
    x: np.ndarray,
    fs_old: int,
    fs_new: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    แปลงความถี่สุ่มสัญญาณ (Sampling Rate Conversion) ด้วยวิธี Rational Polyphase
    โดยหาอัตราส่วนทอนเป็นเศษส่วนอย่างต่ำ (L/M) เพื่อความแม่นยำสูงสุด
    
    Parameters
    ----------
    x : np.ndarray
        สัญญาณต้นฉบับ
    fs_old : int
        Sampling Rate เดิม (Hz)
    fs_new : int
        Sampling Rate ใหม่ (Hz)
        
    Returns
    -------
    x_new : np.ndarray
        สัญญาณใหม่ที่แปลง Sampling Rate แล้ว
    t_new : np.ndarray
        แกนเวลาใหม่
    """
    ratio = Fraction(fs_new, fs_old)
    up, down = ratio.numerator, ratio.denominator
    x_new = signal.resample_poly(x, up, down)
    t_new = np.arange(len(x_new)) / fs_new
    return x_new, t_new


def time_axis(fs: int, duration: float) -> np.ndarray:
    """สร้างแกนเวลา (Time Vector) ตั้งแต่ 0 ถึง duration วินาที"""
    T = 1.0 / fs
    return np.arange(0, duration, T)


def generate_sine(f: float, t: np.ndarray, A: float = 1.0, p: float = 0.0) -> np.ndarray:
    """สร้างสัญญาณ Sine Wave: y = A * sin(2*pi*f*t + p)"""
    return A * np.sin(2.0 * np.pi * f * t + p)


def spectrum(x: np.ndarray, fs: float = 16000, NFFT: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    คำนวณสเปกตรัมขนาด (Magnitude Spectrum) แบบ Single-Sided FFT
    
    Parameters
    ----------
    x : np.ndarray
        สัญญาณอินพุต 1D
    fs : float
        Sampling Rate (Hz), ค่าเริ่มต้น 16000 Hz
    NFFT : int
        จำนวนจุด FFT (-1 หมายถึงใช้เท่ากับ len(x))
        
    Returns
    -------
    freq : np.ndarray
        แกนความถี่ (0 ถึง fs/2 Hz)
    mag : np.ndarray
        ขนาดสเปกตรัม (Magnitude)
    """
    if NFFT == -1:
        NFFT = len(x)
    X = np.fft.fft(x, NFFT)
    N = len(X)
    X_half = X[:N // 2]
    mag = np.abs(X_half) * 2.0 / N
    freq = np.arange(len(mag)) * fs / N
    return freq, mag

