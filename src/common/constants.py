"""
src/common/constants.py
=======================
เก็บค่าคงที่มาตรฐานและพาธของโฟลเดอร์ทั้งหมดในโปรเจกต์
เพื่อให้ทั้ง S1, S2, และ S3 เรียกใช้จากจุดเดียวกัน
"""

from __future__ import annotations
from pathlib import Path

# พาธหลักของโปรเจกต์ (C:\Users\khaok\69-1\DSP\FinalProject)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ค่ามาตรฐานทางสัญญาณเสียง
DEFAULT_SAMPLE_RATE = 16000       # ความถี่สุ่มสัญญาณมาตรฐาน 16 kHz
DEFAULT_RIR_LENGTH = 128          # ความยาวผลตอบสนองของห้องเริ่มต้น 128 จุด (8 ms ที่ 16 kHz)
DEFAULT_LENGTH_MODE = "trim"      # โหมดจัดการความยาวเริ่มต้น: "trim" (ตัดให้เท่าตัวสั้น) หรือ "pad" (เติม 0 ให้เท่าตัวยาว)

# พาธของโฟลเดอร์ข้อมูล (data/ - ไม่เอาขึ้น GitHub)
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_CLIPS_S1_DIR = DATA_DIR / "source_clips" / "s1_one_person"
SOURCE_CLIPS_S2_DIR = DATA_DIR / "source_clips" / "s2_two_people"
SOURCE_CLIPS_S3_DIR = DATA_DIR / "source_clips" / "s3_moving"

RIR_FIXED_DIR = DATA_DIR / "room_impulse_responses" / "fixed_room"
RIR_MOVING_DIR = DATA_DIR / "room_impulse_responses" / "moving_room"

SIMULATED_S1_DIR = DATA_DIR / "simulated" / "s1_one_person"
SIMULATED_S2_DIR = DATA_DIR / "simulated" / "s2_two_people_static"
SIMULATED_S3_DIR = DATA_DIR / "simulated" / "s3_moving_position"

# พาธของโฟลเดอร์ตั้งค่า ตารางดัชนี และผลลัพธ์
CONFIGS_S1_DIR = PROJECT_ROOT / "configs" / "s1_one_person"
CONFIGS_S2_DIR = PROJECT_ROOT / "configs" / "s2_two_people_static"
CONFIGS_S3_DIR = PROJECT_ROOT / "configs" / "s3_moving_position"

MANIFESTS_S1_DIR = PROJECT_ROOT / "manifests" / "s1_manifest"
MANIFESTS_S2_DIR = PROJECT_ROOT / "manifests" / "s2_manifest"
MANIFESTS_S3_DIR = PROJECT_ROOT / "manifests" / "s3_manifest"

RESULTS_S1_DIR = PROJECT_ROOT / "results" / "s1_one_person"
RESULTS_S2_DIR = PROJECT_ROOT / "results" / "s2_two_people_static"
RESULTS_S3_DIR = PROJECT_ROOT / "results" / "s3_moving_position"
