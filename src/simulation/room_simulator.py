"""
src/simulation/room_simulator.py
================================
โมดูลสำหรับ:
1. สร้างและโหลดผลตอบสนองอิมพัลส์ของห้อง (Room Impulse Response: RIR)
   - สามารถกำหนดระดับความก้องของเสียงสะท้อนได้ (reverb_level="low" | "medium" | "high"
     หรือกำหนดค่าความดังของเสียงสะท้อน echo_gain และความยาวหางเสียงก้อง decay_tau เองได้)
   - รองรับทั้งห้องคงที่ (fixed_room) สำหรับ S1, S2 และห้องเปลี่ยนตำแหน่ง (moving_room) สำหรับ S3
2. จำลองสัญญาณที่ไมโครโฟนรับได้จากการสะท้อนในห้อง:
   - รองรับการเลือกโหมดความยาวทั้งแบบ "trim" (ตัดหางเสียงก้องที่เกินความยาวต้นฉบับทิ้ง)
     และแบบ "pad" (เก็บหางเสียงก้องท้ายคลิปไว้ครบ แล้วเติม 0 ต่อท้ายสัญญาณอ้างอิงให้ยาวเท่ากัน)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
from scipy import signal

from src.common.constants import (
    DEFAULT_LENGTH_MODE,
    DEFAULT_RIR_LENGTH,
    DEFAULT_SAMPLE_RATE,
    RIR_FIXED_DIR,
    RIR_MOVING_DIR,
    SOURCE_CLIPS_S1_DIR,
)
from src.io.audio_io import LengthMode, align_signal_lengths, load_wav_mono, save_wav_mono

ReverbLevel = Literal["low", "medium", "high"]

# ค่าตั้งต้นของแต่ละระดับความก้อง (Preset Reverberation Profiles)
REVERB_PRESETS: Dict[str, Dict[str, float]] = {
    "low": {
        "reflection_scale": 0.45,  # เสียงสะท้อนผนังเบา (ห้องบุผนังซับเสียง)
        "tail_gain": 0.015,        # หางเสียงก้องเบา
        "decay_tau": 14.0,         # หางเสียงก้องจางหายเร็วมาก
    },
    "medium": {
        "reflection_scale": 1.00,  # เสียงสะท้อนระดับห้องทำงานทั่วไป (ค่ามาตรฐาน)
        "tail_gain": 0.040,        # หางเสียงก้องปานกลาง
        "decay_tau": 28.0,         # อัตราการจางหายปานกลาง
    },
    "high": {
        "reflection_scale": 1.45,  # เสียงสะท้อนผนังแรง (ห้องผนังปูน/กระจก)
        "tail_gain": 0.085,        # หางเสียงก้องชัด
        "decay_tau": 52.0,         # หางเสียงก้องค้างนาน
    },
}


def create_fixed_room_ir(
    length: int = DEFAULT_RIR_LENGTH,
    reverb_level: ReverbLevel = "medium",
    echo_gain: float = 1.0,
    decay_tau: Optional[float] = None,
    direct_path_gain: float = 0.80,
    room_name: Optional[str] = None,
    save_to_disk: bool = True,
) -> np.ndarray:
    """สร้างผลตอบสนองอิมพัลส์ของห้องคงที่ (Fixed Room Impulse Response) ที่ปรับความก้องได้

    Parameters
    ----------
    length : int
        ความยาวของ RIR เป็นจำนวนจุด (ค่าเริ่มต้น 128 จุด = 8 ms ที่ 16 kHz)
    reverb_level : {"low", "medium", "high"}
        ระดับความก้องของห้อง:
            - "low"    : ห้องซับเสียงดี ก้องน้อย สลายตัวเร็ว
            - "medium" : ห้องทั่วไป ก้องปานกลาง (ค่าเริ่มต้น)
            - "high"   : ห้องสะท้อนเสียงมาก หางเสียงก้องค้างนาน
    echo_gain : float
        ตัวคูณปรับความดังรวมของเสียงก้องทั้งหมด (ค่าเริ่มต้น 1.0)
    decay_tau : float, optional
        ค่าคงที่การสลายตัวของหางเสียงก้อง (ยิ่งมาก หางเสียงก้องยิ่งค้างนาน)
        หากไม่ระบุ จะใช้ค่าจาก reverb_level อัตโนมัติ
    direct_path_gain : float
        ความแรงของเสียงที่เดินทางจากลำโพงเข้าไมโครโฟนโดยตรง (ค่าเริ่มต้น 0.80)
    room_name : str, optional
        ชื่อไฟล์สำหรับบันทึก (หากไม่ระบุ จะตั้งตามระดับความก้อง เช่น "rir_fixed_medium")
    save_to_disk : bool
        บันทึกไฟล์ .npy, .wav, และ .json ลงใน data/room_impulse_responses/fixed_room/
    """
    if length < 16:
        raise ValueError("ความยาวของ Room Impulse Response ต้องไม่น้อยกว่า 16 จุด")
    if reverb_level not in REVERB_PRESETS:
        raise ValueError(f"reverb_level ต้องเป็น 'low', 'medium', หรือ 'high' (ได้รับ: {reverb_level})")

    preset = REVERB_PRESETS[reverb_level]
    ref_scale = preset["reflection_scale"]
    tail_gain = preset["tail_gain"]
    tau = float(decay_tau) if decay_tau is not None else preset["decay_tau"]

    if room_name is None:
        room_name = f"rir_fixed_{reverb_level}"

    h = np.zeros(length, dtype=np.float64)

    # 1. เสียงเดินทางตรงจากลำโพงเข้าไมโครโฟน (Direct Path)
    h[0] = direct_path_gain

    # 2. เสียงสะท้อนผนังช่วงต้น (Early Reflections) ปรับความแรงตาม ref_scale
    base_reflections: Dict[int, float] = {
        5: -0.35,
        12: 0.24,
        23: -0.16,
        37: 0.11,
        54: -0.07,
        78: 0.045,
        102: -0.025,
    }
    for tap_idx, amp in base_reflections.items():
        if tap_idx < length:
            h[tap_idx] = amp * ref_scale

    # 3. หางเสียงก้องที่ค่อยๆ สลายตัวตามค่า tau (Late Reverberation Decay)
    n = np.arange(length, dtype=np.float64)
    tail = tail_gain * np.exp(-n / tau) * np.cos(2.0 * np.pi * n / 11.0)
    tail[:4] = 0.0
    h = (h + tail) * float(echo_gain)

    if save_to_disk:
        RIR_FIXED_DIR.mkdir(parents=True, exist_ok=True)
        npy_path = RIR_FIXED_DIR / f"{room_name}.npy"
        json_path = RIR_FIXED_DIR / f"{room_name}.json"
        wav_path = RIR_FIXED_DIR / f"{room_name}.wav"

        np.save(npy_path, h)
        save_wav_mono(wav_path, h, sample_rate=DEFAULT_SAMPLE_RATE)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "room_name": room_name,
                    "reverb_level": reverb_level,
                    "echo_gain": float(echo_gain),
                    "decay_tau": tau,
                    "sample_rate_hz": DEFAULT_SAMPLE_RATE,
                    "num_taps": int(length),
                    "duration_ms": round(length / DEFAULT_SAMPLE_RATE * 1000.0, 3),
                    "energy_l2": round(float(np.sum(h ** 2)), 6),
                    "coefficients": [round(float(v), 8) for v in h],
                },
                f,
                indent=2,
            )

    return h


def simulate_static_room_echo(
    farend: np.ndarray,
    h_room: np.ndarray,
    nearend: Optional[np.ndarray] = None,
    length_mode: LengthMode = DEFAULT_LENGTH_MODE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """จำลองเสียงที่ไมโครโฟนรับได้เมื่อเสียงลำโพง (farend) สะท้อนผ่านห้องคงที่ (h_room)
    โดยสามารถเลือกวิธีจัดการความยาวได้ 2 รูปแบบเพื่อใช้เปรียบเทียบ:
        1. length_mode="trim":
           - ตัดหางผลลัพธ์จากการทำ Convolution (ซึ่งเดิมยาว L + M - 1 จุด) ให้เหลือเท่ากับความยาวที่สั้นที่สุด (L จุด)
        2. length_mode="pad":
           - เก็บหางเสียงก้องท้ายคลิปจากการทำ Convolution ไว้ทั้งหมด (L + M - 1 จุด)
             และเติมเลข 0 (ความเงียบ) ต่อท้ายสัญญาณลำโพง farend และ nearend ให้ยาวเท่ากัน

    Returns
    -------
    farend_aligned : np.ndarray
        สัญญาณลำโพงอ้างอิงที่ปรับความยาวให้ตรงกับไมโครโฟนแล้ว
    mic_signal : np.ndarray
        สัญญาณที่ไมโครโฟนรับได้ d[n] = echo[n] + nearend[n]
    echo_only : np.ndarray
        สัญญาณเสียงก้องจากลำโพงล้วนๆ echo[n] = (farend * h_room)[n]
    target_signal : np.ndarray
        สัญญาณเป้าหมายที่ต้องการให้เหลืออยู่หลังตัดเสียงก้อง (สำหรับ S1 คือความเงียบ 0 ทั้งหมด)
    """
    x = np.asarray(farend, dtype=np.float64).reshape(-1)
    h = np.asarray(h_room, dtype=np.float64).reshape(-1)

    # หากมีเสียงคนพูดฝั่งใกล้ (สำหรับ S2 ในอนาคต) ให้ปรับความยาวต้นทางระหว่าง farend กับ nearend ก่อน
    if nearend is not None:
        s_raw = np.asarray(nearend, dtype=np.float64).reshape(-1)
        x, s_raw = align_signal_lengths([x, s_raw], mode=length_mode)
    else:
        s_raw = np.zeros_like(x)

    # คำนวณ Linear Convolution ระหว่างเสียงลำโพง x[n] กับห้อง h[n]
    # ความยาวเต็มของ full_echo จะเท่ากับ len(x) + len(h) - 1 (ยาวกว่า x อยู่ M - 1 จุด)
    full_echo = signal.fftconvolve(x, h, mode="full")

    if length_mode == "trim":
        # แบบที่ 1 (Trim): ตัดหางที่เกินมา M - 1 จุดออก เพื่อให้ทุกสัญญาณยาวเท่ากับ len(x) พอดี
        farend_aligned, echo_only, target_signal = align_signal_lengths(
            [x, full_echo, s_raw],
            mode="trim",
            target_length=len(x),
        )
    elif length_mode == "pad":
        # แบบที่ 2 (Pad): เก็บหางเสียงก้องไว้ครบ len(full_echo) จุด แล้วเติม 0 ต่อท้าย x และ target ให้ยาวเท่ากัน
        farend_aligned, echo_only, target_signal = align_signal_lengths(
            [x, full_echo, s_raw],
            mode="pad",
            target_length=len(full_echo),
        )
    else:
        raise ValueError(f"length_mode ต้องเป็น 'trim' หรือ 'pad' เท่านั้น (ได้รับ: {length_mode})")

    mic_signal = echo_only + target_signal

    # ตรวจสอบซ้ำว่าทุกสัญญาณมีความยาวเท่ากันเป๊ะก่อนส่งออก
    assert len(farend_aligned) == len(mic_signal) == len(echo_only) == len(target_signal)
    return farend_aligned, mic_signal, echo_only, target_signal


def apply_position_movement(
    signal_in: np.ndarray,
    change_ratio: float = 0.5,
    gain_pos1: float = 1.0,
    gain_pos2: float = 0.55,
    fade_samples: int = 800,
) -> Tuple[np.ndarray, int]:
    """จำลองการขยับเปลี่ยนตำแหน่งของผู้พูดแบบยังไม่ผ่านห้องสะท้อน (Free-Field Movement)
    โดยปรับระดับความดังตามระยะห่างที่เปลี่ยนไปจากตำแหน่งที่ 1 (gain_pos1) ไปยังตำแหน่งที่ 2 (gain_pos2)
    พร้อมเกลี่ยช่วงรอยต่อ (Cross-fade) เพื่อไม่ให้เกิดเสียงคลิกตรงจุดเปลี่ยนตำแหน่ง.

    Returns
    -------
    moved_signal : np.ndarray
        สัญญาณที่ปรับระดับความดังตามการเปลี่ยนตำแหน่งแล้ว
    change_idx : int
        จุดตัวอย่าง (Sample Index) ที่เริ่มขยับเปลี่ยนตำแหน่ง
    """
    x = np.asarray(signal_in, dtype=np.float64).reshape(-1)
    n_total = len(x)
    change_idx = int(np.clip(round(n_total * change_ratio), 1, n_total - 1))

    envelope = np.full(n_total, float(gain_pos1), dtype=np.float64)
    envelope[change_idx:] = float(gain_pos2)

    half_fade = min(fade_samples // 2, change_idx, n_total - change_idx)
    if half_fade > 0:
        start = change_idx - half_fade
        end = change_idx + half_fade
        envelope[start:end] = np.linspace(float(gain_pos1), float(gain_pos2), end - start)

    return x * envelope, change_idx


def create_moving_room_irs(
    length: int = DEFAULT_RIR_LENGTH,
    save_to_disk: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """สร้างผลตอบสนองของห้อง 2 ตำแหน่ง (Position 1 และ Position 2) สำหรับจำลองการเคลื่อนที่ (S3)
    - Position 1: อยู่ใกล้ลำโพง/ไมค์มากกว่า เสียงตรงชัดและสะท้อนระดับปานกลาง
    - Position 2: ขยับห่างออกไป เสียงตรงลดลง ดีเลย์สะท้อนเปลี่ยนตำแหน่ง และหางเสียงก้องชัดขึ้น
    """
    if length < 16:
        raise ValueError("ความยาวของ Room Impulse Response ต้องไม่น้อยกว่า 16 จุด")

    # ตำแหน่งที่ 1 (Position 1)
    h_pos1 = create_fixed_room_ir(
        length=length,
        reverb_level="medium",
        echo_gain=1.15,
        direct_path_gain=0.90,
        save_to_disk=False,
    )

    # ตำแหน่งที่ 2 (Position 2): เปลี่ยนตำแหน่งจุดสะท้อนผนังและเพิ่มความก้อง
    h_pos2 = np.zeros(length, dtype=np.float64)
    h_pos2[2] = 0.85  # ขยับห่างออกทำให้มีดีเลย์เริ่มต้นเล็กน้อยและเกิดการสะท้อนผนังแรงขึ้น
    pos2_reflections: Dict[int, float] = {
        8: -0.52,
        17: 0.40,
        29: -0.30,
        44: 0.22,
        63: -0.15,
        89: 0.10,
        112: -0.065,
    }
    for tap_idx, amp in pos2_reflections.items():
        if tap_idx < length:
            h_pos2[tap_idx] = amp

    n = np.arange(length, dtype=np.float64)
    tail2 = 0.070 * np.exp(-n / 45.0) * np.sin(2.0 * np.pi * n / 9.5)
    tail2[:5] = 0.0
    h_pos2 = h_pos2 + tail2

    if save_to_disk:
        RIR_MOVING_DIR.mkdir(parents=True, exist_ok=True)
        for name, ir_arr in (("rir_moving_pos1", h_pos1), ("rir_moving_pos2", h_pos2)):
            np.save(RIR_MOVING_DIR / f"{name}.npy", ir_arr)
            save_wav_mono(RIR_MOVING_DIR / f"{name}.wav", ir_arr, sample_rate=DEFAULT_SAMPLE_RATE)
            with open(RIR_MOVING_DIR / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "room_name": name,
                        "sample_rate_hz": DEFAULT_SAMPLE_RATE,
                        "num_taps": int(length),
                        "duration_ms": round(length / DEFAULT_SAMPLE_RATE * 1000.0, 3),
                        "energy_l2": round(float(np.sum(ir_arr ** 2)), 6),
                        "coefficients": [round(float(v), 8) for v in ir_arr],
                    },
                    f,
                    indent=2,
                )

    return h_pos1, h_pos2


def simulate_moving_room_echo(
    farend: np.ndarray,
    h_pos1: np.ndarray,
    h_pos2: np.ndarray,
    nearend: Optional[np.ndarray] = None,
    change_ratio: float = 0.5,
    length_mode: LengthMode = DEFAULT_LENGTH_MODE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """จำลองเสียงที่ไมโครโฟนรับได้เมื่อมีการขยับเปลี่ยนตำแหน่งจากห้อง h_pos1 ไปยัง h_pos2
    ณ สัดส่วนเวลา change_ratio (ค่าเริ่มต้น 0.5 คือครึ่งคลิป).

    Returns
    -------
    farend_aligned : np.ndarray
        สัญญาณลำโพงอ้างอิงที่ปรับความยาวตรงกับไมโครโฟนแล้ว
    mic_signal : np.ndarray
        สัญญาณไมโครโฟนรวม d[n] = echo_moving[n] + target_signal[n]
    echo_only : np.ndarray
        สัญญาณเสียงก้องจากลำโพงที่เปลี่ยนผลตอบสนองของห้องตามตำแหน่ง
    target_signal : np.ndarray
        สัญญาณเป้าหมาย (0 สำหรับ 1 คน หรือเสียงคนพูดที่ 2 สำหรับ 2 คน)
    change_idx : int
        ตำแหน่งจุดตัวอย่าง (Sample Index) ที่เปลี่ยนจากตำแหน่งที่ 1 ไปตำแหน่งที่ 2
    """
    x = np.asarray(farend, dtype=np.float64).reshape(-1)
    h1 = np.asarray(h_pos1, dtype=np.float64).reshape(-1)
    h2 = np.asarray(h_pos2, dtype=np.float64).reshape(-1)

    if nearend is not None:
        s_raw = np.asarray(nearend, dtype=np.float64).reshape(-1)
        x, s_raw = align_signal_lengths([x, s_raw], mode=length_mode)
    else:
        s_raw = np.zeros_like(x)

    echo1_full = signal.fftconvolve(x, h1, mode="full")
    echo2_full = signal.fftconvolve(x, h2, mode="full")
    full_len = max(len(echo1_full), len(echo2_full))
    echo1_full, echo2_full = align_signal_lengths([echo1_full, echo2_full], mode="pad", target_length=full_len)

    change_idx = int(np.clip(round(len(x) * change_ratio), 1, len(x) - 1))

    # ผสมเสียงสะท้อนช่วงก่อนขยับ (h_pos1) และหลังขยับ (h_pos2) ด้วย Cross-fade สั้นๆ
    weight_pos2 = np.zeros(full_len, dtype=np.float64)
    weight_pos2[change_idx:] = 1.0
    half_fade = min(400, change_idx, full_len - change_idx)
    if half_fade > 0:
        weight_pos2[change_idx - half_fade : change_idx + half_fade] = np.linspace(0.0, 1.0, 2 * half_fade)

    full_echo = (1.0 - weight_pos2) * echo1_full + weight_pos2 * echo2_full

    target_len = len(x) if length_mode == "trim" else full_len
    farend_aligned, echo_only, target_signal = align_signal_lengths(
        [x, full_echo, s_raw],
        mode=length_mode,
        target_length=target_len,
    )
    mic_signal = echo_only + target_signal

    assert len(farend_aligned) == len(mic_signal) == len(echo_only) == len(target_signal)
    return farend_aligned, mic_signal, echo_only, target_signal, change_idx


def get_scenario_audio(
    scenario: Union[int, str] = 1,
    clip_idx1: int = 0,
    clip_idx2: int = 1,
    reverb_level: Optional[Union[int, str]] = None,
    lag_ms: Optional[float] = -1,
    change_ratio: float = 0.5,
):
    """
    ฟังก์ชันสำหรับโหลดและเตรียมสัญญาณเสียงจริงสำหรับทำกราฟ พร้อมกำหนดหรือสุ่มความหน่วง (Lag) ของ Echo
    
    Parameters:
    -----------
    scenario : int หรือ str (ค่าเริ่มต้น 1)
        - 1 หรือ "1p_static" : แบบที่ 1 (1 คน ไม่เคลื่อนที่)
        - 2 หรือ "1p_moving" : แบบที่ 2 (1 คน เคลื่อนที่)
        - 3 หรือ "2p_static" : แบบที่ 3 (2 คน ไม่เคลื่อนที่)
        - 4 หรือ "2p_moving" : แบบที่ 4 (2 คน เคลื่อนที่)
        
    clip_idx1 : int (ค่าเริ่มต้น 0)
        ลำดับ Index ของคลิปเสียงคนที่ 1 (Far-End) เช่น 0 ถึง 9
    clip_idx2 : int (ค่าเริ่มต้น 1)
        ลำดับ Index ของคลิปเสียงคนที่ 2 (Near-End) สำหรับกรณี 2 คน
        
    reverb_level : None, int, หรือ str (ค่าเริ่มต้น None)
        - None                  : ไม่ผ่านห้องสะท้อนเสียง (ดึงเฉพาะสัญญาณดิบ)
        - 0 หรือ "low"          : ผ่านห้องสะท้อนเสียงความก้องต่ำ
        - 1 หรือ "mid"/"medium" : ผ่านห้องสะท้อนเสียงความก้องปานกลาง
        - 2 หรือ "high"         : ผ่านห้องสะท้อนเสียงความก้องสูง
        
    lag_ms : float หรือ None (ค่าเริ่มต้น -1)
        ความหน่วงเวลาของเสียงสะท้อน (Acoustic Propagation Delay / Lag) หน่วยเป็นมิลลิวินาที (ms):
        - ไม่ใส่ค่า, None, หรือ -1 : สุ่มค่าความหน่วงอัตโนมัติในช่วงปกติ (3.0 - 20.0 ms)
        - ตัวเลข >= 0 (เช่น 0, 5, 12) : เลื่อนเสียงสะท้อนไปข้างหลังตามจำนวน ms ที่กำหนด
        
    change_ratio : float (ค่าเริ่มต้น 0.5)
        สัดส่วนเวลาที่ขยับเปลี่ยนตำแหน่ง (0.5 = กึ่งกลางคลิปที่ 4 วินาที)

    Returns:
    --------
    - ถ้า reverb_level is None:
        คืนค่า 3 ตัว: (t, raw, fs)
    - ถ้าส่ง reverb_level (0, 1, 2 หรือ 'low', 'mid', 'high'):
        คืนค่า 4 ตัว: (t, raw, echo, fs)
    """
    # 1. โหลดคลิปเสียงจริง
    clips = sorted(SOURCE_CLIPS_S1_DIR.glob("*.wav"))
    spk1, fs = load_wav_mono(clips[clip_idx1])
    spk2, _ = load_wav_mono(clips[clip_idx2])
    spk1, spk2 = align_signal_lengths([spk1, spk2], mode="trim")
    t = np.arange(len(spk1)) / fs

    # 2. จัดการสัญญาณต้นฉบับ raw ตามสถานการณ์ที่เลือก
    if scenario in (1, "1", "1p_static"):
        raw = spk1.copy()
    elif scenario in (2, "2", "1p_moving"):
        raw, _ = apply_position_movement(spk1, change_ratio=change_ratio)
    elif scenario in (3, "3", "2p_static"):
        raw = spk1 + 0.85 * spk2
    elif scenario in (4, "4", "2p_moving"):
        raw_moved, _ = apply_position_movement(spk1, change_ratio=change_ratio)
        raw = raw_moved + 0.85 * spk2
    else:
        raise ValueError(f"scenario ไม่ถูกต้อง: {scenario} (ใส่ 1, 2, 3 หรือ 4)")

    # 3. ถ้าไม่ต้องการห้องสะท้อนเสียง (reverb_level is None) คืนค่าแค่ 3 ตัว
    if reverb_level is None:
        return t, raw, fs

    # 4. แปลงค่าพารามิเตอร์ห้องสะท้อน 0, 1, 2 เป็นชื่อ preset
    reverb_map = {
        0: "low", "0": "low", "low": "low",
        1: "medium", "1": "medium", "mid": "medium", "medium": "medium",
        2: "high", "2": "high", "high": "high"
    }
    if reverb_level not in reverb_map:
        raise ValueError(f"reverb_level ไม่ถูกต้อง: {reverb_level} (ใส่ None, 0/'low', 1/'mid', หรือ 2/'high')")
    preset = reverb_map[reverb_level]

    # 5. คำนวณความหน่วงเวลา (Lag) ของเสียงสะท้อน
    if lag_ms is None or float(lag_ms) == -1.0:
        # สุ่มในช่วงปกติที่พบในห้องจริง: 3.0 ถึง 20.0 มิลลิวินาที
        actual_lag_ms = round(float(np.random.uniform(3.0, 20.0)), 2)
        lag_mode_str = "สุ่มอัตโนมัติ (Random 3-20 ms)"
    elif float(lag_ms) >= 0.0:
        actual_lag_ms = float(lag_ms)
        lag_mode_str = "กำหนดค่าเอง (Custom)"
    else:
        raise ValueError(f"lag_ms ไม่ถูกต้อง: {lag_ms} (ใส่ -1 เพื่อสุ่ม หรือใส่ค่า >= 0)")

    lag_samples = int(round(actual_lag_ms * fs / 1000.0))
    lag_zeros = np.zeros(lag_samples, dtype=np.float64)
    print(f"[Echo Lag] โหมด: {lag_mode_str} -> ใช้ Lag = {actual_lag_ms:.2f} ms ({lag_samples} samples)")

    # 6. จำลองเสียงผ่านห้องสะท้อน (เติม 0 นำหน้า Room IR เพื่อสร้าง Lag จริงตามฟิสิกส์)
    if scenario in (1, "1", "1p_static"):
        h_static = create_fixed_room_ir(reverb_level=preset, save_to_disk=False)
        h_static_lagged = np.concatenate([lag_zeros, h_static])
        _, echo, _, _ = simulate_static_room_echo(raw, h_static_lagged, length_mode="trim")
    elif scenario in (2, "2", "1p_moving"):
        h_pos1, h_pos2 = create_moving_room_irs(save_to_disk=False)
        h_pos1_lagged = np.concatenate([lag_zeros, h_pos1])
        h_pos2_lagged = np.concatenate([lag_zeros, h_pos2])
        _, echo, _, _, _ = simulate_moving_room_echo(raw, h_pos1_lagged, h_pos2_lagged, change_ratio=change_ratio, length_mode="trim")
    elif scenario in (3, "3", "2p_static"):
        h_static = create_fixed_room_ir(reverb_level=preset, save_to_disk=False)
        h_static_lagged = np.concatenate([lag_zeros, h_static])
        _, echo, _, _ = simulate_static_room_echo(spk1, h_static_lagged, nearend=0.85 * spk2, length_mode="trim")
    elif scenario in (4, "4", "2p_moving"):
        h_pos1, h_pos2 = create_moving_room_irs(save_to_disk=False)
        h_pos1_lagged = np.concatenate([lag_zeros, h_pos1])
        h_pos2_lagged = np.concatenate([lag_zeros, h_pos2])
        raw_spk1_moved, _ = apply_position_movement(spk1, change_ratio=change_ratio)
        _, echo, _, _, _ = simulate_moving_room_echo(raw_spk1_moved, h_pos1_lagged, h_pos2_lagged, nearend=0.85 * spk2, change_ratio=change_ratio, length_mode="trim")

    return t, raw, echo, fs
