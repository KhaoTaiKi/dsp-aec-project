"""
notebooks/setup_dsp.py
======================
ไฟล์ Helper สำหรับ Jupyter Notebooks ในโฟลเดอร์ notebooks/
ทำหน้าที่:
1. เชื่อมต่อ sys.path ไปยังโฟลเดอร์โปรเจกต์หลักอัตโนมัติ
2. รวมศูนย์และ Re-export ฟังก์ชันทั้งหมด ทำให้ใน notebook เรียกแค่:
   `from setup_dsp import *`
   ก็พร้อมใช้งานทุกฟังก์ชันทันที ไม่ติด ModuleNotFoundError อีกต่อไป!
"""

import sys
from pathlib import Path

# 1. ค้นหาโฟลเดอร์หลักของโปรเจกต์ (ถอย 1 ขั้นจาก notebooks/)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

# Fallback หาก path ไม่ตรง
if not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = Path("C:/Users/khaok/69-1/DSP/FinalProject")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. นำเข้าฟังก์ชันที่หยิบมาจากเพื่อน (จาก src.common.dsp_utils)
from src.common.dsp_utils import (
    check_eda,
    add_noise,
    correlation_offset,
    shift_signal,
    iir_filter,
)

# 3. นำเข้าฟังก์ชันจำลองห้องสะท้อนเสียงและเตรียมเสียง (จาก src.simulation.room_simulator)
from src.simulation.room_simulator import (
    create_fixed_room_ir,
    simulate_static_room_echo,
    get_scenario_audio,
    SOURCE_CLIPS_S1_DIR,
)

# 4. นำเข้าฟังก์ชันตัวกรองปรับตัวได้ (Adaptive Filters)
from src.adaptive_filters import nlms_filter, lms_filter

# 5. นำเข้าตัวชี้วัดประสิทธิภาพ (Metrics)
from src.metrics import calculate_erle, calculate_erle_curve, calculate_rmse

# 6. นำเข้าฟังก์ชันจัดการไฟล์เสียง (Audio I/O)
from src.io.audio_io import load_wav_mono, align_signal_lengths

__all__ = [
    "PROJECT_ROOT",
    "check_eda",
    "add_noise",
    "correlation_offset",
    "shift_signal",
    "iir_filter",
    "create_fixed_room_ir",
    "simulate_static_room_echo",
    "get_scenario_audio",
    "SOURCE_CLIPS_S1_DIR",
    "nlms_filter",
    "lms_filter",
    "calculate_erle",
    "calculate_erle_curve",
    "calculate_rmse",
    "load_wav_mono",
    "align_signal_lengths",
]

print("🚀 [setup_dsp] โหลดและเชื่อมต่อฟังก์ชัน DSP ทั้งหมดสำเร็จ พร้อมลุย!")
