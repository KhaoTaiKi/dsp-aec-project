"""
src/visualization/signal_plots.py
=================================
โมดูลสำหรับพล็อตกราฟเปรียบเทียบสัญญาณเสียงทั้ง 4 กรณี:
    1. คนพูด 1 คน ไม่เคลื่อนที่ (1 Speaker, Static)
    2. คนพูด 1 คน เคลื่อนที่ (1 Speaker, Moving Position)
    3. คนพูด 2 คน ไม่เคลื่อนที่ (2 Speakers, Static)
    4. คนพูด 2 คน เคลื่อนที่ (2 Speakers, Moving Position)

แบ่งการแสดงผลออกเป็น 2 ชุด:
    - ชุดที่ 1 (plot_four_raw_scenarios):
      แสดงกราฟสัญญาณต้นฉบับทั้ง 4 กรณี แบบยังไม่ผ่านห้องสะท้อนและไม่ผ่านตัวกรองใดๆ
    - ชุดที่ 2 (plot_four_room_echo_comparisons):
      นำสัญญาณทั้ง 4 กรณีไปผ่านห้องเสียงสะท้อน แล้วพล็อตเทียบระหว่าง
      "เสียงต้นฉบับ (ไม่ผ่านห้อง)" กับ "เสียงที่ผ่านห้องสะท้อน"
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.common.constants import DEFAULT_SAMPLE_RATE
from src.io.audio_io import align_signal_lengths
from src.simulation.room_simulator import (
    apply_position_movement,
    create_fixed_room_ir,
    create_moving_room_irs,
    simulate_moving_room_echo,
    simulate_static_room_echo,
)

# ตั้งค่าฟอนต์ภาษาไทยสำหรับ Windows ให้แสดงผลบนกราฟได้สวยงามไม่เป็นสี่เหลี่ยม
matplotlib.rcParams["font.family"] = ["Tahoma", "Leelawadee UI", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def prepare_four_scenarios_data(
    speaker1: np.ndarray,
    speaker2: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    change_ratio: float = 0.5,
) -> Dict[str, Dict[str, np.ndarray | float | int]]:
    """เตรียมข้อมูลสัญญาณทั้ง 4 กรณี ทั้งแบบต้นฉบับ (ไม่ผ่านห้อง) และแบบผ่านห้องเสียงสะท้อนแล้ว."""
    s1, s2 = align_signal_lengths([speaker1, speaker2], mode="trim")
    t_axis = np.arange(len(s1), dtype=np.float64) / float(sample_rate)

    # 1. สัญญาณต้นฉบับ (ยังไม่ผ่านห้องสะท้อนและไม่ผ่านตัวกรอง)
    raw_1p_static = s1.copy()
    raw_1p_moving, change_idx = apply_position_movement(
        s1, change_ratio=change_ratio, gain_pos1=1.0, gain_pos2=0.55
    )
    raw_2p_static = s1 + s2
    raw_2p_moving = raw_1p_moving + s2
    change_time_sec = change_idx / float(sample_rate)

    # 2. สร้างผลตอบสนองของห้อง (Room Impulse Response)
    h_fixed = create_fixed_room_ir(reverb_level="high", echo_gain=1.15, save_to_disk=False)
    h_pos1, h_pos2 = create_moving_room_irs(save_to_disk=False)

    # 3. นำเสียงทั้ง 4 กรณีไปผ่านห้องเสียงสะท้อน
    # กรณีที่ 1: 1 คน ไม่เคลื่อนที่ (ผ่านห้องคงที่)
    _, room_1p_static, echo_1p_static, _ = simulate_static_room_echo(
        raw_1p_static, h_fixed, nearend=None, length_mode="trim"
    )

    # กรณีที่ 2: 1 คน เคลื่อนที่ (ผ่านห้องเคลื่อนที่ h_pos1 -> h_pos2)
    _, room_1p_moving, echo_1p_moving, _, _ = simulate_moving_room_echo(
        raw_1p_moving, h_pos1, h_pos2, nearend=None, change_ratio=change_ratio, length_mode="trim"
    )

    # กรณีที่ 3: 2 คน ไม่เคลื่อนที่ (ผ่านห้องคงที่ + เสียงคนพูดที่ 2)
    _, room_2p_static, echo_2p_static, _ = simulate_static_room_echo(
        raw_1p_static, h_fixed, nearend=s2, length_mode="trim"
    )

    # กรณีที่ 4: 2 คน เคลื่อนที่ (ผ่านห้องเคลื่อนที่ + เสียงคนพูดที่ 2)
    _, room_2p_moving, echo_2p_moving, _, _ = simulate_moving_room_echo(
        raw_1p_moving, h_pos1, h_pos2, nearend=s2, change_ratio=change_ratio, length_mode="trim"
    )

    return {
        "meta": {
            "time_sec": t_axis,
            "sample_rate": sample_rate,
            "change_idx": change_idx,
            "change_time_sec": change_time_sec,
            "speaker1_clean": s1,
            "speaker2_clean": s2,
        },
        "1_one_person_static": {
            "title": "1) คนพูด 1 คน — ไม่เคลื่อนที่ (1 Speaker, Static)",
            "raw": raw_1p_static,
            "room": room_1p_static,
            "echo_only": echo_1p_static,
            "is_moving": False,
            "is_two_people": False,
        },
        "2_one_person_moving": {
            "title": "2) คนพูด 1 คน — เคลื่อนที่เปลี่ยนตำแหน่ง (1 Speaker, Moving)",
            "raw": raw_1p_moving,
            "room": room_1p_moving,
            "echo_only": echo_1p_moving,
            "is_moving": True,
            "is_two_people": False,
        },
        "3_two_people_static": {
            "title": "3) คนพูด 2 คน — ไม่เคลื่อนที่ (2 Speakers, Static)",
            "raw": raw_2p_static,
            "room": room_2p_static,
            "echo_only": echo_2p_static,
            "is_moving": False,
            "is_two_people": True,
        },
        "4_two_people_moving": {
            "title": "4) คนพูด 2 คน — เคลื่อนที่เปลี่ยนตำแหน่ง (2 Speakers, Moving)",
            "raw": raw_2p_moving,
            "room": room_2p_moving,
            "echo_only": echo_2p_moving,
            "is_moving": True,
            "is_two_people": True,
        },
    }


def plot_four_raw_scenarios(
    scenarios_data: Dict[str, Dict[str, np.ndarray | float | int]],
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """พล็อตกราฟ 4 รูป แสดงสัญญาณเสียงต้นฉบับทั้ง 4 กรณี (ยังไม่ผ่านห้องสะท้อนและไม่ผ่านตัวกรอง)."""
    meta = scenarios_data["meta"]
    t = meta["time_sec"]
    change_t = float(meta["change_time_sec"])
    s2_clean = meta["speaker2_clean"]

    keys = [
        "1_one_person_static",
        "2_one_person_moving",
        "3_two_people_static",
        "4_two_people_moving",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    fig.suptitle(
        "ชุดที่ 1: สัญญาณเสียงต้นฉบับทั้ง 4 กรณี (ยังไม่ผ่านห้องสะท้อน และไม่ผ่านตัวกรอง)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    for ax, key in zip(axes.flat, keys):
        item = scenarios_data[key]
        raw_sig = item["raw"]

        if item["is_two_people"]:
            ax.plot(
                t,
                raw_sig,
                color="#1f77b4",
                linewidth=0.8,
                alpha=0.85,
                label="เสียงรวมต้นฉบับ (คนที่ 1 + คนที่ 2)",
            )
            ax.plot(
                t,
                s2_clean,
                color="#2ca02c",
                linewidth=0.65,
                alpha=0.65,
                label="เสียงคนที่ 2 (พูดแทรกหน้าไมค์)",
            )
        else:
            ax.plot(
                t,
                raw_sig,
                color="#1f77b4",
                linewidth=0.85,
                alpha=0.9,
                label="เสียงต้นฉบับคนที่ 1 (ไม่ผ่านห้อง)",
            )

        if item["is_moving"]:
            ax.axvline(
                change_t,
                color="#d62728",
                linestyle="--",
                linewidth=1.8,
                label=f"จุดขยับเปลี่ยนตำแหน่ง (t = {change_t:.1f} s)",
            )
            ax.axvspan(t[0], change_t, color="#e8f4f8", alpha=0.35, label="ตำแหน่งที่ 1 (ใกล้)")
            ax.axvspan(change_t, t[-1], color="#fff3cd", alpha=0.35, label="ตำแหน่งที่ 2 (ขยับห่างออก)")

        rms_val = float(np.sqrt(np.mean(raw_sig ** 2)))
        ax.set_title(f"{item['title']} | RMS = {rms_val:.4f}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8.5)

    axes[1, 0].set_xlabel("Time (seconds)")
    axes[1, 1].set_xlabel("Time (seconds)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_four_room_echo_comparisons(
    scenarios_data: Dict[str, Dict[str, np.ndarray | float | int]],
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """พล็อตกราฟ 4 รูป เปรียบเทียบ 'เสียงต้นฉบับ (ไม่ผ่านห้อง)' กับ 'เสียงที่ผ่านห้องสะท้อนแล้ว' ทั้ง 4 กรณี."""
    meta = scenarios_data["meta"]
    t = meta["time_sec"]
    change_t = float(meta["change_time_sec"])

    keys = [
        "1_one_person_static",
        "2_one_person_moving",
        "3_two_people_static",
        "4_two_people_moving",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5), sharex=True, sharey=True)
    fig.suptitle(
        "ชุดที่ 2: เปรียบเทียบ เสียงต้นฉบับ (ไม่ผ่านห้อง) vs เสียงที่ผ่านห้องสะท้อนแล้ว ทั้ง 4 กรณี",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    for ax, key in zip(axes.flat, keys):
        item = scenarios_data[key]
        raw_sig = item["raw"]
        room_sig = item["room"]

        # พล็อตเสียงที่ผ่านห้องสะท้อนไว้ด้านหลัง (สีส้ม/แดง) และเสียงต้นฉบับไว้ด้านหน้า (สีน้ำเงิน)
        ax.plot(
            t,
            room_sig,
            color="#ff7f0e",
            linewidth=0.85,
            alpha=0.80,
            label="ผ่านห้องเสียงสะท้อน (With Room Echo)",
        )
        ax.plot(
            t,
            raw_sig,
            color="#1f77b4",
            linewidth=0.75,
            alpha=0.75,
            label="ต้นฉบับ ไม่ผ่านห้อง (Original Raw)",
        )

        if item["is_moving"]:
            ax.axvline(
                change_t,
                color="#d62728",
                linestyle="--",
                linewidth=1.8,
                label=f"สลับห้องสะท้อน RIR Pos1 -> Pos2 (t = {change_t:.1f} s)",
            )

        raw_rms = float(np.sqrt(np.mean(raw_sig ** 2)))
        room_rms = float(np.sqrt(np.mean(room_sig ** 2)))
        diff_rms = float(np.sqrt(np.mean((room_sig - raw_sig) ** 2)))
        ax.set_title(
            f"{item['title']}\nRMS ต้นฉบับ = {raw_rms:.4f} | RMS ผ่านห้อง = {room_rms:.4f} | เสียงก้องที่เพิ่มมา (RMSE) = {diff_rms:.4f}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8.5)

    axes[1, 0].set_xlabel("Time (seconds)")
    axes[1, 1].set_xlabel("Time (seconds)")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
