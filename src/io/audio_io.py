"""
src/io/audio_io.py
==================
โมดูลสำหรับโหลดไฟล์เสียง บันทึกไฟล์เสียง ตรวจสอบความถี่สุ่มสัญญาณ (Sample Rate)
และจัดการกรณีที่สัญญาณมีความยาวไม่เท่ากัน 2 รูปแบบ:
    1. mode="trim" : ตัดหางสัญญาณที่ยาวกว่าทิ้ง ให้เหลือความยาวเท่ากับตัวที่สั้นที่สุด
    2. mode="pad"  : เติมค่า 0 (ความเงียบ) ต่อท้ายสัญญาณที่สั้นกว่า ให้มีความยาวเท่ากับตัวที่ยาวที่สุด
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal

from src.common.constants import DEFAULT_SAMPLE_RATE, SOURCE_CLIPS_S1_DIR

LengthMode = Literal["trim", "pad"]


def load_wav_mono(
    wav_path: Path | str,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """อ่านไฟล์ .wav และแปลงเป็นสัญญาณ Mono ความถี่ target_sr (ค่าเริ่มต้น 16000 Hz)
    พร้อมปรับสเกลให้อยู่ในช่วง [-1.0, 1.0).
    """
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์เสียง: {path}")

    with wave.open(str(path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        orig_sr = wf.getframerate()
        num_frames = wf.getnframes()
        raw_bytes = wf.readframes(num_frames)

    if sample_width == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"ไม่รองรับขนาดข้อมูลเสียง {sample_width} bytes ในไฟล์ {path.name}")

    # หากเป็นไฟล์หลายช่องสัญญาณ (เช่น Stereo) ให้เฉลี่ยรวมเป็น Mono
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    # หาก Sample Rate ไม่ตรงกับ target_sr (เช่น ไม่ใช่ 16000 Hz) ให้ Resample อัตโนมัติ
    if orig_sr != target_sr:
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        audio = signal.resample_poly(audio, up, down).astype(np.float64)

    return audio, target_sr


def save_wav_mono(
    wav_path: Path | str,
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """บันทึกอาเรย์สัญญาณเสียงลงเป็นไฟล์ .wav 16-bit PCM Mono."""
    path = Path(wav_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(np.asarray(audio, dtype=np.float64).reshape(-1), -0.999969, 0.999969)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def align_signal_lengths(
    signals: Sequence[np.ndarray],
    mode: LengthMode = "trim",
    target_length: Optional[int] = None,
) -> List[np.ndarray]:
    """ปรับให้สัญญาณทุกตัวในรายการมีความยาวเท่ากันทุกจุด (Sample-by-Sample)
    โดยมีตัวเลือกให้เปรียบเทียบ 2 แบบ:
        - mode="trim" : ตัดหางสัญญาณทุกตัวให้เท่ากับความยาวที่สั้นที่สุด (หรือ target_length)
        - mode="pad"  : เติมเลข 0 (ความเงียบ) ต่อท้ายสัญญาณที่สั้นกว่าให้เท่ากับความยาวที่ยาวที่สุด (หรือ target_length)
    """
    if not signals:
        return []

    arrays = [np.asarray(s, dtype=np.float64).reshape(-1) for s in signals]
    lengths = [arr.shape[0] for arr in arrays]

    if mode not in ("trim", "pad"):
        raise ValueError(f"mode ต้องเป็น 'trim' หรือ 'pad' เท่านั้น (ได้รับ: {mode})")

    if target_length is not None:
        final_len = int(target_length)
    elif mode == "trim":
        final_len = min(lengths)
    else:  # mode == "pad"
        final_len = max(lengths)

    aligned: List[np.ndarray] = []
    for arr in arrays:
        cur_len = arr.shape[0]
        if cur_len == final_len:
            aligned.append(arr.copy())
        elif cur_len > final_len:
            # ถ้ายาวเกินเป้าหมาย: หากเป็นโหมด trim หรือกำหนด target_length ให้ตัดหางทิ้ง
            aligned.append(arr[:final_len].copy())
        else:
            # ถ้าสั้นกว่าเป้าหมาย: เติมเลข 0 (ความเงียบ) ต่อท้ายให้ครบ final_len
            padded = np.zeros(final_len, dtype=np.float64)
            padded[:cur_len] = arr
            aligned.append(padded)

    # ตรวจสอบความถูกต้องว่าหลังปรับแล้ว สัญญาณทุกตัวมีความยาวเท่ากันเป๊ะ
    for arr in aligned:
        assert arr.shape[0] == final_len, "เกิดข้อผิดพลาด: ความยาวสัญญาณหลังปรับยังไม่เท่ากัน!"

    return aligned


def inspect_dataset(target: Optional[Union[int, str]] = None) -> Union[Dict[str, int], List[str]]:
    """
    ฟังก์ชันสำหรับสำรวจชุดข้อมูลเสียงคนพูดจริง (ไม่รวม fixed_room)
    
    Parameters:
    -----------
    target : int หรือ str (ค่าเริ่มต้น None)
        - ถ้าไม่ระบุ (None หรือ "all") : โชว์ภาพรวมทั้ง 4 แบบ และจำนวนตัวอย่างที่มี
        - ถ้าใส่ระบุแบบ (1, 2, 3, 4 หรือชื่อแบบ) : เจาะจงแสดงรายชื่อคลิป/คู่ตัวอย่างของแบบนั้น
            * 1 หรือ "1p_static" : เจาะดู 1 คน ไม่เคลื่อนที่
            * 2 หรือ "1p_moving" : เจาะดู 1 คน เคลื่อนที่
            * 3 หรือ "2p_static" : เจาะดู 2 คน ไม่เคลื่อนที่ (แสดงเป็นคู่คนพูด)
            * 4 หรือ "2p_moving" : เจาะดู 2 คน เคลื่อนที่ (แสดงเป็นคู่คนพูด)

    Returns:
    --------
    - ถ้า target=None : ส่งกลับ Dict สรุปจำนวนตัวอย่าง {"1_person_static": 10, ...}
    - ถ้าเจาะจง target : ส่งกลับ List[str] แสดงรายชื่อคลิปหรือคู่คลิปของแบบนั้น
    """
    clips = sorted([f.name for f in SOURCE_CLIPS_S1_DIR.glob("*.wav")])
    n_clips = len(clips)
    pairs_2p = [f"คู่ที่ {i+1:02d}: [คนพูด 1] {clips[i]}  +  [คนพูด 2] {clips[i+1]}" for i in range(n_clips - 1)]
    
    if target is None or target == "all":
        summary = {
            "1) 1_person_static (1 คน ไม่เคลื่อนที่)": n_clips,
            "2) 1_person_moving (1 คน เคลื่อนที่)": n_clips,
            "3) 2_persons_static (2 คน ไม่เคลื่อนที่)": len(pairs_2p),
            "4) 2_persons_moving (2 คน เคลื่อนที่)": len(pairs_2p)
        }
        print("=" * 65)
        print("สรุปภาพรวมชุดข้อมูลเสียงคนพูดจริง (ทั้งหมด 4 รูปแบบ):")
        print("=" * 65)
        for name, count in summary.items():
            print(f"- {name:<42} : มี {count:02d} ตัวอย่าง")
        print("=" * 65)
        print("💡 Tip: ใส่ target=1 ถึง 4 เพื่อเจาะจงดูรายละเอียดของแต่ละแบบได้")
        return summary

    elif target in (1, "1", "1p_static"):
        print(f"--- [แบบที่ 1: 1 คน ไม่เคลื่อนที่] มีทั้งหมด {n_clips} ตัวอย่าง ---")
        for idx, name in enumerate(clips):
            print(f"[{idx:02d}] {name}")
        return clips

    elif target in (2, "2", "1p_moving"):
        print(f"--- [แบบที่ 2: 1 คน เคลื่อนที่] มีทั้งหมด {n_clips} ตัวอย่าง ---")
        items = [f"[{idx:02d}] {name} (จำลองขยับเปลี่ยนตำแหน่ง)" for idx, name in enumerate(clips)]
        for it in items:
            print(it)
        return items

    elif target in (3, "3", "2p_static"):
        print(f"--- [แบบที่ 3: 2 คน ไม่เคลื่อนที่] มีทั้งหมด {len(pairs_2p)} คู่ตัวอย่าง ---")
        for p in pairs_2p:
            print(p)
        return pairs_2p

    elif target in (4, "4", "2p_moving"):
        print(f"--- [แบบที่ 4: 2 คน เคลื่อนที่] มีทั้งหมด {len(pairs_2p)} คู่ตัวอย่าง ---")
        pairs_moving = [p + " (มีคนขยับตำแหน่ง)" for p in pairs_2p]
        for p in pairs_moving:
            print(p)
        return pairs_moving

    else:
        raise ValueError(f"target ไม่ถูกต้อง: {target} (ใส่ 1, 2, 3, 4 หรือปล่อยว่างเพื่อดูภาพรวม)")
