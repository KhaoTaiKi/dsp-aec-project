r"""
src/student_step_by_step_s1.py
==============================
สคริปต์ฝึกปฏิบัติทีละขั้นสำหรับนักศึกษา 2 คน (Student-Led Learning Runner for Scenario S1)
ตามข้อกำหนด CDX-04 และ CDX-05

วิธีรัน:
    & "C:\Users\khaok\dsp_env\Scripts\python.exe" "C:\Users\khaok\69-1\DSP\FinalProject\src\student_step_by_step_s1.py"

โครงสร้างการเรียนรู้ 5 ขั้นตอน:
    - Step 1 (Student A): โหลดไฟล์เสียงพูด 1 คลิปแรก (x[n]) และสร้างผลตอบสนองของห้องคงที่ (h_room[n])
    - Step 2 (Student A): จำลองเสียงก้องที่ไมโครโฟน d[n] = x[n] * h_room[n] และตรวจสอบค่าพลังงาน
    - Step 3 (Student B): รัน LMS เทียบกับ NLMS ภายใต้พารามิเตอร์เดียวกันเป๊ะ (N=128, mu=0.01) บนคลิปที่ 1
    - Step 4 (Student B): ทดลองใช้ก้าวปรับตัวแบบ Normalize ของ NLMS (N=128, mu=0.50) บนคลิปที่ 1
    - Step 5 (Student A & B): รันครบทั้ง 10 คลิป สรุปค่าเฉลี่ย ERLE / RMSE และเปรียบเทียบผล
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lms import (
    compute_lms_stability_bound,
    create_fixed_room_ir,
    run_lms,
    simulate_microphone_signal,
)
from src.metrics import compute_erle_db, compute_rmse
from src.nlms import run_nlms
from src.run_s1_experiment import prepare_10_clean_speech_clips

# =============================================================================
# โซนพารามิเตอร์ที่นักศึกษาสามารถทดลองเปลี่ยนค่าได้อย่างปลอดภัย (Safe Student Parameters)
# =============================================================================
SAMPLE_RATE = 16000

# [Student A สามารถลองเปลี่ยน]: เลือกหมายเลขคลิปตัวอย่างสำหรับดูผลละเอียดใน Step 1-4 (เลือกได้ตั้งแต่ 0 ถึง 9)
SELECTED_CLIP_INDEX = 0

# [Student B สามารถลองเปลี่ยน]: ความยาวตัวกรอง FIR (ค่าเริ่มต้น N = 128 จุด เท่ากับความยาวห้อง rir_0)
# ลองเปลี่ยนเป็น 64 (สั้นกว่าห้องจริง) หรือ 128 (เท่าห้องจริง) เพื่อดูผลต่อค่า ERLE
FILTER_TAPS = 128

# [Student B สามารถลองเปลี่ยน]: ค่า Step Size (mu) สำหรับเทียบ LMS vs NLMS ที่พารามิเตอร์เท่ากัน
# ค่าที่ปลอดภัยสำหรับทั้ง 10 คลิปคือช่วง 0.002 ถึง 0.020 (ค่าเริ่มต้น = 0.010)
SAME_SETTING_MU = 0.010

# [Student B สามารถลองเปลี่ยน]: ค่า Normalized Step Size (mu) สำหรับ NLMS
# ค่าที่ปลอดภัยและลู่เข้าเร็วสำหรับ NLMS คือช่วง 0.10 ถึง 0.80 (ค่าเริ่มต้น = 0.50)
NLMS_NORMALIZED_MU = 0.50


def read_wav_mono16_float(wav_path: Path) -> np.ndarray:
    """อ่านไฟล์ WAV 16-bit PCM Mono มาเป็นอาเรย์ทศนิยมในช่วง [-1.0, 1.0)."""
    with wave.open(str(wav_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return arr


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 78)
    print("STUDENT-LED STEP-BY-STEP RUNNER: SCENARIO S1 (STATIC-ROOM ECHO CANCELLATION)")
    print("หมายเหตุ: นักศึกษาสามารถปรับค่าพารามิเตอร์ด้านบนไฟล์เพื่อทดลองสังเกตผลลัพธ์ได้")
    print("=" * 78)

    # เตรียมหรือตรวจสอบว่ามีไฟล์เสียงสะอาด 10 คลิปใน data/source_clips/ พร้อมใช้งาน
    clips_meta = prepare_10_clean_speech_clips()
    selected_meta = clips_meta[SELECTED_CLIP_INDEX]
    clip_path = PROJECT_ROOT / str(selected_meta["local_path"])

    # -------------------------------------------------------------------------
    # STEP 1 (รับผิดชอบโดย Student A): โหลดสัญญาณลำโพง x[n] และสร้างห้องคงที่ h_room[n]
    # -------------------------------------------------------------------------
    print("\n--- [STEP 1 - Student A] โหลดเสียงต้นทาง x[n] และสร้างผลตอบสนองห้องคงที่ h_room[n] ---")
    x = read_wav_mono16_float(clip_path)
    h_room = create_fixed_room_ir(length=128, rir_id="rir_0")

    print(f"  * ไฟล์เสียงที่เลือกทดสอบ : {clip_path.name} (Index = {SELECTED_CLIP_INDEX})")
    print(f"  * ความถี่สุ่มสัญญาณ      : {SAMPLE_RATE} Hz, จำนวนจุด = {len(x)} samples ({len(x)/SAMPLE_RATE:.1f} วินาที)")
    print(f"  * ความยาวห้องจำลอง h[n] : {len(h_room)} taps ({len(h_room)/SAMPLE_RATE*1000:.1f} ms)")
    print("  [สิ่งที่ควรสังเกต]: ค่าสูงสุดของ h_room อยู่ที่ Direct Path (tap แรกๆ) แล้วค่อยๆ สลายตัวตามเวลา")
    print("  [ค่าที่ลองเปลี่ยนได้]: ลองเปลี่ยน SELECTED_CLIP_INDEX เป็น 0..9 เพื่อลองคลิปเสียงพูดคนอื่น")

    # -------------------------------------------------------------------------
    # STEP 2 (รับผิดชอบโดย Student A): จำลองสัญญาณไมโครโฟน d[n] = x[n] * h_room[n]
    # -------------------------------------------------------------------------
    print("\n--- [STEP 2 - Student A] จำลองเสียงสะท้อนในห้อง d[n] = x[n] * h_room[n] ---")
    d, echo, target = simulate_microphone_signal(x, [h_room], rir_switch_samples=None, nearend=None)

    x_rms = compute_rmse(x)
    d_rms = compute_rmse(d)
    target_rms = compute_rmse(target)

    print(f"  * ขนาด RMS ของเสียงลำโพงต้นทาง x[n]      : {x_rms:.5f}")
    print(f"  * ขนาด RMS ของเสียงก้องที่ไมโครโฟนรับได้ d[n] : {d_rms:.5f}")
    print(f"  * ขนาด RMS ของสัญญาณเป้าหมายฝั่งใกล้ s[n] : {target_rms:.5f} (เป็น 0 เพราะไม่มีคนพูดฝั่งใกล้)")
    print("  [สิ่งที่ควรสังเกต]: ใน S1 ไม่มีคนพูดฝั่งใกล้ ดังนั้นถ้าตัดเสียงก้องสมบูรณ์แบบ สัญญาณคงเหลือ e[n] ต้องเข้าใกล้ 0 (เงียบสนิท)")

    # -------------------------------------------------------------------------
    # STEP 3 (รับผิดชอบโดย Student B): เทียบ LMS กับ NLMS ที่พารามิเตอร์เท่ากันเป๊ะ
    # -------------------------------------------------------------------------
    print(f"\n--- [STEP 3 - Student B] เปรียบเทียบ LMS vs NLMS ที่พารามิเตอร์เดียวกัน (N={FILTER_TAPS}, mu={SAME_SETTING_MU}) ---")
    p_x, max_win_e, mu_avg_bound, mu_peak_bound = compute_lms_stability_bound(x, FILTER_TAPS)

    y_lms, e_lms, _ = run_lms(x, d, num_taps=FILTER_TAPS, mu=SAME_SETTING_MU)
    y_nlms_same, e_nlms_same, _ = run_nlms(x, d, num_taps=FILTER_TAPS, mu=SAME_SETTING_MU)

    erle_lms = compute_erle_db(d, e_lms)
    rmse_lms = compute_rmse(e_lms, target)
    erle_nlms_same = compute_erle_db(d, e_nlms_same)
    rmse_nlms_same = compute_rmse(e_nlms_same, target)

    print(f"  * พลังงานเฉลี่ย P_x = {p_x:.6f}, พลังงานหน้าต่างสูงสุด max(||x[n]||^2) = {max_win_e:.4f}")
    print(f"  * ขอบเขตความเสถียรจริงแบบจุดต่อจุดของ LMS (1 / max_win_energy) = {mu_peak_bound:.4f}")
    print(f"  * [Standard LMS ที่ mu={SAME_SETTING_MU:.3f}] -> ERLE = {erle_lms:6.2f} dB | Residual RMSE = {rmse_lms:.5f}")
    print(f"  * [NLMS         ที่ mu={SAME_SETTING_MU:.3f}] -> ERLE = {erle_nlms_same:6.2f} dB | Residual RMSE = {rmse_nlms_same:.5f}")
    print("  [สิ่งที่ควรสังเกต]: เมื่อใช้ค่า mu เล็กเท่ากัน ทั้ง LMS และ NLMS ลดเสียงก้องได้ใกล้เคียงกันและเสถียรทั้งคู่")
    print("  [ค่าที่ลองเปลี่ยนได้]: ลองเปลี่ยน SAME_SETTING_MU ในช่วง 0.002 ถึง 0.020 แล้วดูว่าค่า ERLE เปลี่ยนอย่างไร")

    # -------------------------------------------------------------------------
    # STEP 4 (รับผิดชอบโดย Student B): ทดสอบข้อได้เปรียบของการหารด้วยพลังงานใน NLMS
    # -------------------------------------------------------------------------
    print(f"\n--- [STEP 4 - Student B] รัน NLMS ด้วยก้าวปรับตัวแบบ Normalize (N={FILTER_TAPS}, mu={NLMS_NORMALIZED_MU}) ---")
    y_nlms_norm, e_nlms_norm, _ = run_nlms(x, d, num_taps=FILTER_TAPS, mu=NLMS_NORMALIZED_MU)

    erle_nlms_norm = compute_erle_db(d, e_nlms_norm)
    rmse_nlms_norm = compute_rmse(e_nlms_norm, target)

    print(f"  * [NLMS ที่ mu={NLMS_NORMALIZED_MU:.2f}] -> ERLE = {erle_nlms_norm:6.2f} dB | Residual RMSE = {rmse_nlms_norm:.5f}")
    print("  [สิ่งที่ควรสังเกต]: เพราะ NLMS หารด้วย ||x[n]||^2 + eps จึงใช้ mu=0.50 ได้โดยไม่ระเบิด และกด RMSE ลงจนเกือบเป็น 0")
    print("  [ค่าที่ลองเปลี่ยนได้]: ลองเปลี่ยน FILTER_TAPS เป็น 64 (สั้นกว่าห้องจริง 128 จุด) เพื่อดูว่า ERLE ตกลงเพราะอะไร")

    # -------------------------------------------------------------------------
    # STEP 5 (ทำร่วมกันทั้ง Student A และ Student B): รันครบทั้ง 10 คลิปและสรุปตาราง
    # -------------------------------------------------------------------------
    print("\n--- [STEP 5 - Student A & B] รันทดสอบครบทั้ง 10 คลิปเพื่อหาค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน ---")
    header = (
        f"{'Trial':<7} | {'Mic RMS':>8} | "
        f"{'LMS(mu=' + str(SAME_SETTING_MU) + ') ERLE':>18} | {'LMS RMSE':>9} | "
        f"{'NLMS(mu=' + str(SAME_SETTING_MU) + ') ERLE':>19} | "
        f"{'NLMS(mu=' + str(NLMS_NORMALIZED_MU) + ') ERLE':>18} | {'NLMS RMSE':>9}"
    )
    print(header)
    print("-" * len(header))

    lms_erles, lms_rmses = [], []
    nlms_same_erles, nlms_same_rmses = [], []
    nlms_norm_erles, nlms_norm_rmses = [], []

    for idx, cmeta in enumerate(clips_meta, start=1):
        trial_id = f"S1_T{idx:02d}"
        x_i = read_wav_mono16_float(PROJECT_ROOT / str(cmeta["local_path"]))
        d_i, _, tgt_i = simulate_microphone_signal(x_i, [h_room])

        _, e_lms_i, _ = run_lms(x_i, d_i, num_taps=FILTER_TAPS, mu=SAME_SETTING_MU)
        _, e_nlms_same_i, _ = run_nlms(x_i, d_i, num_taps=FILTER_TAPS, mu=SAME_SETTING_MU)
        _, e_nlms_norm_i, _ = run_nlms(x_i, d_i, num_taps=FILTER_TAPS, mu=NLMS_NORMALIZED_MU)

        mic_r = compute_rmse(d_i)
        e1 = compute_erle_db(d_i, e_lms_i)
        r1 = compute_rmse(e_lms_i, tgt_i)
        e2 = compute_erle_db(d_i, e_nlms_same_i)
        r2 = compute_rmse(e_nlms_same_i, tgt_i)
        e3 = compute_erle_db(d_i, e_nlms_norm_i)
        r3 = compute_rmse(e_nlms_norm_i, tgt_i)

        lms_erles.append(e1)
        lms_rmses.append(r1)
        nlms_same_erles.append(e2)
        nlms_same_rmses.append(r2)
        nlms_norm_erles.append(e3)
        nlms_norm_rmses.append(r3)

        print(
            f"{trial_id:<7} | {mic_r:8.5f} | "
            f"{e1:15.2f} dB | {r1:9.5f} | "
            f"{e2:16.2f} dB | "
            f"{e3:15.2f} dB | {r3:9.5f}"
        )

    print("-" * len(header))
    print(
        f"{'MEAN':<7} | {'':>8} | "
        f"{np.mean(lms_erles):15.2f} dB | {np.mean(lms_rmses):9.5f} | "
        f"{np.mean(nlms_same_erles):16.2f} dB | "
        f"{np.mean(nlms_norm_erles):15.2f} dB | {np.mean(nlms_norm_rmses):9.5f}"
    )
    print(
        f"{'STD':<7} | {'':>8} | "
        f"{np.std(lms_erles):15.2f} dB | {np.std(lms_rmses):9.5f} | "
        f"{np.std(nlms_same_erles):16.2f} dB | "
        f"{np.std(nlms_norm_erles):15.2f} dB | {np.std(nlms_norm_rmses):9.5f}"
    )
    print("\nเสร็จสิ้นการรันฝึกปฏิบัติ S1 ทั้ง 5 ขั้นตอน! นักศึกษาสามารถเปิดอ่าน LEARNING_GUIDE_S1_TH.md เพื่อซ้อมตอบคำถามได้ทันที")


if __name__ == "__main__":
    main()
