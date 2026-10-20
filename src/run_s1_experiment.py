"""
Experiment Runner for CDX-01B (Phase 0 Gate), CDX-02 (S1 Static-Room AEC), and CDX-03 (Scenario Layout).

Executes:
1. Phase 0 Synthetic Correctness Gate (CDX-01B):
   - Causal FIR path with >= 5 non-zero taps (`rir_0`, 128 taps).
   - Far-end signal combining a 440-Hz sine wave + broadband speech-like signal, no near-end signal.
   - Asserts strict causal zero-filled tap ordering, finite values, no post-warm-up growth,
     and NLMS steady-state ERLE >= 30 dB.
2. Scenario S1 — Simple Static-Room Echo Experiment (CDX-02 & CDX-03):
   - Extracts 10 real clean speech clips (16 kHz mono, fixed 8.0-second segments = 128,000 samples)
     into ignored `data/source_clips/`.
   - Passes all 10 clips through one identical fixed room impulse response `rir_0` (128 taps = 8 ms)
     to generate `data/simulated/s1_one_speaker_static/` wav files.
   - Runs LMS and NLMS under identical filter length (N = 128) and safe step size settings.
   - Records manifests (`manifests/rir_catalog.json`, `manifests/s1_one_speaker_static_manifest.csv`,
     and reserved empty manifests for S2 and S3).
   - Computes ERLE (full-clip and steady-state) and RMSE for all 10 clips, reporting mean and std.
   - Generates the 4 required figures in `results/s1_one_speaker_static/` (and copies to `results/figures/`):
     (1) Room impulse response plot
     (2) Representative waveform before/after cancellation
     (3) Representative spectrogram before/after cancellation
     (4) Bar chart of ERLE across all 10 clips (LMS vs NLMS)
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import wave
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lms import (
    compute_lms_stability_bound,
    create_fixed_room_ir,
    run_lms,
    simulate_microphone_signal,
)
from src.nlms import run_nlms
from src.metrics import (
    check_no_post_warmup_growth,
    compute_erle_db,
    compute_rmse,
    compute_sliding_erle_db,
)


SAMPLE_RATE = 16000
SEGMENT_SECONDS = 8.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)  # 128,000 samples
FILTER_TAPS = 128
SAFE_MU = 0.01  # Safe step size strictly below instantaneous burst bound 1/max(||x_vec||^2) across all 10 clips
NLMS_MU = 0.5   # Standard normalized step size (also evaluated alongside identical mu=0.01)

SELECTED_ZIP_CLIPS: List[str] = [
    "1EbxxdNtTU6sfStbSi-xbA_doubletalk.wav",
    "6qaoi9k6Qk2RWUgx68erSQ_doubletalk.wav",
    "AbbfwwrUbU2rrCgs69mg6A_doubletalk.wav",
    "B6yQraHQwUmDdgts6zSckg_doubletalk.wav",
    "CchZxh-H7kGCeACO8l_MZg_doubletalk.wav",
    "GP5G30waQ0qEii-l8avBdw_doubletalk.wav",
    "KG7t-N2Q9EG_lY_eJaJWlA_doubletalk.wav",
    "KTSv-QqiXE6AtMHghJN4Pg_doubletalk.wav",
    "MlZqzzdyBE6pvFaHBmi_5g_doubletalk.wav",
    "NXzM6tR0KkaqmNCr9PDnMg_doubletalk.wav",
]


def write_wav_mono16(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Save 1D float64 array in [-1, 1] as 16-bit PCM mono WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -0.999969, 0.999969)
    pcm16 = (clipped * 32768.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def read_wav_mono16(path: Path) -> Tuple[np.ndarray, int]:
    """Read 16-bit PCM WAV into float64 mono array in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm, sr


def verify_causal_buffer_ordering() -> None:
    """Explicit unit check for CDX-01B requirement 1:
    Input vector must be [x[n], x[n-1], ..., x[n-N+1]] with zero-filled history.
    """
    test_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    test_h = np.array([0.5, -0.25, 0.1], dtype=np.float64)
    mic, echo, _ = simulate_microphone_signal(test_x, [test_h])
    expected = np.array([
        0.5 * 1.0,
        0.5 * 2.0 - 0.25 * 1.0,
        0.5 * 3.0 - 0.25 * 2.0 + 0.1 * 1.0,
        0.5 * 4.0 - 0.25 * 3.0 + 0.1 * 2.0,
        0.5 * 5.0 - 0.25 * 4.0 + 0.1 * 3.0,
    ])
    assert np.allclose(mic, expected, atol=1e-12), "Causal convolution buffer ordering check failed!"


def run_phase0_sanity_gate(h_room: np.ndarray) -> Dict[str, object]:
    """Execute CDX-01B Phase 0 synthetic known-room correctness gate."""
    verify_causal_buffer_ordering()

    non_zero_taps = int(np.sum(np.abs(h_room) > 1e-6))
    assert non_zero_taps >= 5, f"Expected >= 5 non-zero taps in RIR, got {non_zero_taps}"

    rng = np.random.default_rng(seed=20261020)
    n_samples = SAMPLE_RATE * 6  # 6.0 seconds
    t = np.arange(n_samples, dtype=np.float64) / float(SAMPLE_RATE)

    # 440-Hz tone + broadband speech-shaped AR(1)/bandpass excitation
    tone_440 = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    broadband = rng.standard_normal(n_samples)
    b_bp, a_bp = signal.butter(4, [100.0, 3800.0], btype="bandpass", fs=SAMPLE_RATE)
    speech_like = signal.lfilter(b_bp, a_bp, broadband)
    speech_like = 0.35 * speech_like / (np.max(np.abs(speech_like)) + 1e-12)
    # Modulate envelope like speech syllables (4 Hz + 1.5 Hz)
    env = 0.65 + 0.35 * np.sin(2.0 * np.pi * 3.5 * t)
    x_syn = np.clip(tone_440 + env * speech_like, -0.95, 0.95)

    d_syn, _, target_syn = simulate_microphone_signal(x_syn, [h_room])
    assert np.allclose(target_syn, 0.0), "Phase 0 must have zero near-end signal."

    p_x, max_win_e, mu_avg_bound, mu_peak_bound = compute_lms_stability_bound(x_syn, FILTER_TAPS)
    # Safe LMS step size well below instantaneous stability bound
    mu_lms = min(0.02, 0.25 * mu_peak_bound)
    mu_nlms = 0.5

    y_lms, e_lms, w_lms = run_lms(x_syn, d_syn, num_taps=FILTER_TAPS, mu=mu_lms)
    y_nlms, e_nlms, w_nlms = run_nlms(x_syn, d_syn, num_taps=FILTER_TAPS, mu=mu_nlms)

    assert check_no_post_warmup_growth(e_lms, d_syn, SAMPLE_RATE), "LMS failed post-warmup growth check!"
    assert check_no_post_warmup_growth(e_nlms, d_syn, SAMPLE_RATE), "NLMS failed post-warmup growth check!"

    warmup_idx = SAMPLE_RATE * 2  # Measure steady-state over last 4 seconds
    erle_lms_steady = compute_erle_db(d_syn, e_lms, start_sample=warmup_idx)
    erle_nlms_steady = compute_erle_db(d_syn, e_nlms, start_sample=warmup_idx)
    erle_lms_full = compute_erle_db(d_syn, e_lms, start_sample=0)
    erle_nlms_full = compute_erle_db(d_syn, e_nlms, start_sample=0)

    assert erle_nlms_steady >= 30.0, (
        f"Phase 0 Gate failed: NLMS steady-state ERLE = {erle_nlms_steady:.2f} dB (< 30 dB)"
    )

    out_dir = PROJECT_ROOT / "results" / "s1_one_speaker_static"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save Phase 0 ERLE learning curve
    t_lms, curve_lms = compute_sliding_erle_db(d_syn, e_lms, SAMPLE_RATE)
    t_nlms, curve_nlms = compute_sliding_erle_db(d_syn, e_nlms, SAMPLE_RATE)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    ax.plot(t_lms, curve_lms, label=f"LMS (mu={mu_lms:.4f}, steady ERLE={erle_lms_steady:.1f} dB)", color="#1f77b4", lw=1.8)
    ax.plot(t_nlms, curve_nlms, label=f"NLMS (mu={mu_nlms:.2f}, steady ERLE={erle_nlms_steady:.1f} dB)", color="#d62728", lw=1.8)
    ax.axhline(30.0, color="#2ca02c", ls="--", lw=1.2, label="30 dB Gate Threshold")
    ax.set_title("Phase 0 Sanity Gate: Known-Room ERLE Convergence (440 Hz Tone + Broadband)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ERLE (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "phase0_sanity_erle_curve.png")
    plt.close(fig)

    gate_record = {
        "gate": "CDX-01B_Phase0_Synthetic_Known_Room",
        "sample_rate_hz": SAMPLE_RATE,
        "duration_sec": 6.0,
        "num_taps": FILTER_TAPS,
        "non_zero_rir_taps": non_zero_taps,
        "measured_input_power_px": round(p_x, 6),
        "max_instantaneous_window_energy": round(max_win_e, 4),
        "lms_stationary_bound_mu_max": round(mu_avg_bound, 4),
        "lms_instantaneous_burst_bound_mu_max": round(mu_peak_bound, 4),
        "lms_mu_used": round(mu_lms, 4),
        "nlms_mu_used": round(mu_nlms, 4),
        "lms_full_erle_db": round(erle_lms_full, 2),
        "lms_steady_state_erle_db": round(erle_lms_steady, 2),
        "nlms_full_erle_db": round(erle_nlms_full, 2),
        "nlms_steady_state_erle_db": round(erle_nlms_steady, 2),
        "lms_weight_rmse_vs_true_rir": round(float(np.sqrt(np.mean((w_lms - h_room) ** 2))), 6),
        "nlms_weight_rmse_vs_true_rir": round(float(np.sqrt(np.mean((w_nlms - h_room) ** 2))), 6),
        "gate_passed": bool(erle_nlms_steady >= 30.0),
    }
    with open(out_dir / "phase0_sanity_gate.json", "w", encoding="utf-8") as f:
        json.dump(gate_record, f, indent=2)

    return gate_record


def prepare_10_clean_speech_clips() -> List[Dict[str, object]]:
    """Extract 10 clean speech clips from AEC-Challenge-main.zip into ignored data/source_clips/."""
    zip_path = Path(r"C:\Users\khaok\Downloads\AEC-Challenge-main.zip")
    source_dir = PROJECT_ROOT / "data" / "source_clips"
    source_dir.mkdir(parents=True, exist_ok=True)

    clips_meta: List[Dict[str, object]] = []

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            for idx, fname in enumerate(SELECTED_ZIP_CLIPS, start=1):
                member = f"AEC-Challenge-main/baseline/icassp2021/clean/{fname}"
                raw_bytes = z.read(member)
                with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                    sr = wf.getframerate()
                    ch = wf.getnchannels()
                    pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
                if ch > 1:
                    pcm = pcm.reshape(-1, ch).mean(axis=1)
                if sr != SAMPLE_RATE:
                    pcm = signal.resample_poly(pcm, SAMPLE_RATE, sr)

                # Detrend & extract fixed 8.0s segment (128,000 samples)
                pcm = pcm - np.mean(pcm)
                assert len(pcm) >= SEGMENT_SAMPLES, f"Clip {fname} shorter than {SEGMENT_SECONDS}s"
                seg = pcm[:SEGMENT_SAMPLES].copy()
                # Normalize peak to 0.80 so all 10 clips have consistent realistic speech level
                peak = float(np.max(np.abs(seg)))
                if peak > 1e-6:
                    seg = 0.80 * (seg / peak)

                clip_id = f"clip_{idx:02d}"
                out_wav = source_dir / f"{clip_id}_{fname}"
                write_wav_mono16(out_wav, seg, SAMPLE_RATE)

                clips_meta.append({
                    "clip_id": clip_id,
                    "source_zip_member": member,
                    "local_path": f"data/source_clips/{out_wav.name}",
                    "sample_rate": SAMPLE_RATE,
                    "duration_sec": SEGMENT_SECONDS,
                    "num_samples": SEGMENT_SAMPLES,
                    "rms": round(float(np.sqrt(np.mean(seg ** 2))), 5),
                    "peak": round(float(np.max(np.abs(seg))), 5),
                })
    else:
        raise FileNotFoundError(f"Expected archive not found: {zip_path}")

    return clips_meta


def save_rir_catalog(h_rir0: np.ndarray, h_rir1: np.ndarray) -> None:
    """Save deterministic room impulse responses to manifests/rir_catalog.json."""
    manifests_dir = PROJECT_ROOT / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    catalog = {
        "sample_rate_hz": SAMPLE_RATE,
        "rirs": {
            "rir_0": {
                "description": "Fixed static room impulse response for S1 and S2 (128 taps = 8.0 ms at 16 kHz)",
                "num_taps": int(len(h_rir0)),
                "l2_norm": round(float(np.linalg.norm(h_rir0)), 6),
                "coefficients": [round(float(v), 8) for v in h_rir0],
            },
            "rir_1": {
                "description": "Reserved secondary room impulse response for S3 moving-path scenario",
                "num_taps": int(len(h_rir1)),
                "l2_norm": round(float(np.linalg.norm(h_rir1)), 6),
                "coefficients": [round(float(v), 8) for v in h_rir1],
            },
        },
    }
    with open(manifests_dir / "rir_catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)


def create_reserved_s2_s3_manifests() -> None:
    """Create empty header-only manifests and simulated folders for reserved S2 and S3."""
    manifests_dir = PROJECT_ROOT / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario_id",
        "trial_id",
        "sample_rate",
        "farend_clip",
        "nearend_clip",
        "rir_id",
        "rir_switch_samples",
        "segment_start",
        "segment_end",
        "mic_path",
        "reference_path",
        "target_path",
        "seed",
    ]

    for scen_name in ["s2_two_speakers_static", "s3_moving_path"]:
        (PROJECT_ROOT / "data" / "simulated" / scen_name).mkdir(parents=True, exist_ok=True)
        manifest_path = manifests_dir / f"{scen_name}_manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def run_s1_experiment(
    clips_meta: List[Dict[str, object]],
    h_rir0: np.ndarray,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Run S1 static-room echo cancellation across all 10 clean speech clips."""
    sim_dir = PROJECT_ROOT / "data" / "simulated" / "s1_one_speaker_static"
    res_dir = PROJECT_ROOT / "results" / "s1_one_speaker_static"
    fig_dir = PROJECT_ROOT / "results" / "figures"
    manifests_dir = PROJECT_ROOT / "manifests"

    sim_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []

    rep_data: Dict[str, np.ndarray] = {}

    for idx, cmeta in enumerate(clips_meta, start=1):
        trial_id = f"S1_T{idx:02d}"
        src_wav = PROJECT_ROOT / str(cmeta["local_path"])
        x, sr = read_wav_mono16(src_wav)
        assert sr == SAMPLE_RATE and len(x) == SEGMENT_SAMPLES

        # Simulate microphone signal d[n] = x[n] * h_room[n] (no nearend, no noise)
        d, echo, target = simulate_microphone_signal(x, [h_rir0], rir_switch_samples=None, nearend=None)

        ref_rel = f"data/simulated/s1_one_speaker_static/{trial_id}_reference.wav"
        mic_rel = f"data/simulated/s1_one_speaker_static/{trial_id}_mic.wav"
        tgt_rel = f"data/simulated/s1_one_speaker_static/{trial_id}_target.wav"

        write_wav_mono16(PROJECT_ROOT / ref_rel, x, SAMPLE_RATE)
        write_wav_mono16(PROJECT_ROOT / mic_rel, d, SAMPLE_RATE)
        write_wav_mono16(PROJECT_ROOT / tgt_rel, target, SAMPLE_RATE)

        manifest_rows.append({
            "scenario_id": "S1",
            "trial_id": trial_id,
            "sample_rate": SAMPLE_RATE,
            "farend_clip": cmeta["local_path"],
            "nearend_clip": "",
            "rir_id": "rir_0",
            "rir_switch_samples": "[]",
            "segment_start": 0,
            "segment_end": SEGMENT_SAMPLES,
            "mic_path": mic_rel,
            "reference_path": ref_rel,
            "target_path": tgt_rel,
            "seed": 20261020 + idx,
        })

        p_x, max_win_e, mu_avg_bound, mu_peak_bound = compute_lms_stability_bound(x, FILTER_TAPS)
        assert SAFE_MU < mu_peak_bound, (
            f"SAFE_MU ({SAFE_MU}) exceeds instantaneous stability bound ({mu_peak_bound}) for {trial_id}"
        )

        # Controlled comparison 1: Identical safe settings (N=128, mu=0.01) for LMS vs NLMS
        y_lms, e_lms, w_lms = run_lms(x, d, num_taps=FILTER_TAPS, mu=SAFE_MU)
        y_nlms_same, e_nlms_same, w_nlms_same = run_nlms(x, d, num_taps=FILTER_TAPS, mu=SAFE_MU)
        # Controlled comparison 2: Standard NLMS setting (N=128, mu=0.50)
        y_nlms_opt, e_nlms_opt, w_nlms_opt = run_nlms(x, d, num_taps=FILTER_TAPS, mu=NLMS_MU)

        assert check_no_post_warmup_growth(e_lms, d, SAMPLE_RATE), f"LMS post-warmup growth in {trial_id}"
        assert check_no_post_warmup_growth(e_nlms_same, d, SAMPLE_RATE), f"NLMS (mu=0.01) post-warmup growth in {trial_id}"
        assert check_no_post_warmup_growth(e_nlms_opt, d, SAMPLE_RATE), f"NLMS (mu=0.50) post-warmup growth in {trial_id}"

        # Save residual WAVs into ignored simulated directory
        write_wav_mono16(sim_dir / f"{trial_id}_residual_lms_mu001.wav", e_lms, SAMPLE_RATE)
        write_wav_mono16(sim_dir / f"{trial_id}_residual_nlms_mu001.wav", e_nlms_same, SAMPLE_RATE)
        write_wav_mono16(sim_dir / f"{trial_id}_residual_nlms_mu050.wav", e_nlms_opt, SAMPLE_RATE)

        warmup_samples = int(1.0 * SAMPLE_RATE)
        row = {
            "trial_id": trial_id,
            "clip_name": Path(str(cmeta["local_path"])).name,
            "input_power_px": round(p_x, 6),
            "max_win_energy": round(max_win_e, 4),
            "lms_mu_avg_bound": round(mu_avg_bound, 4),
            "lms_mu_peak_bound": round(mu_peak_bound, 4),
            "mic_rms": round(compute_rmse(d), 5),
            "lms_erle_db": round(compute_erle_db(d, e_lms), 2),
            "lms_steady_erle_db": round(compute_erle_db(d, e_lms, start_sample=warmup_samples), 2),
            "lms_rmse": round(compute_rmse(e_lms, target), 5),
            "nlms_same_mu_erle_db": round(compute_erle_db(d, e_nlms_same), 2),
            "nlms_same_mu_steady_erle_db": round(compute_erle_db(d, e_nlms_same, start_sample=warmup_samples), 2),
            "nlms_same_mu_rmse": round(compute_rmse(e_nlms_same, target), 5),
            "nlms_erle_db": round(compute_erle_db(d, e_nlms_opt), 2),
            "nlms_steady_erle_db": round(compute_erle_db(d, e_nlms_opt, start_sample=warmup_samples), 2),
            "nlms_rmse": round(compute_rmse(e_nlms_opt, target), 5),
        }
        metric_rows.append(row)

        if idx == 1:
            rep_data = {
                "x": x,
                "d": d,
                "e_lms": e_lms,
                "e_nlms_same": e_nlms_same,
                "e_nlms_opt": e_nlms_opt,
                "w_lms": w_lms,
                "w_nlms_opt": w_nlms_opt,
            }

    # Write S1 manifest
    manifest_fields = list(manifest_rows[0].keys())
    with open(manifests_dir / "s1_one_speaker_static_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Write S1 per-clip metrics CSV
    metric_fields = list(metric_rows[0].keys())
    with open(res_dir / "s1_metrics_per_clip.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    # Compute aggregate summary (Mean & Std across 10 clips)
    def stats(key: str) -> Dict[str, float]:
        arr = np.array([float(r[key]) for r in metric_rows], dtype=np.float64)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    summary = {
        "scenario_id": "S1",
        "num_clips": len(metric_rows),
        "sample_rate_hz": SAMPLE_RATE,
        "segment_seconds": SEGMENT_SECONDS,
        "filter_length_taps": FILTER_TAPS,
        "settings": {
            "lms_safe": {"num_taps": FILTER_TAPS, "mu": SAFE_MU},
            "nlms_identical_mu": {"num_taps": FILTER_TAPS, "mu": SAFE_MU},
            "nlms_standard_mu": {"num_taps": FILTER_TAPS, "mu": NLMS_MU},
        },
        "aggregates": {
            "mic_unprocessed_rms": stats("mic_rms"),
            "lms_erle_db": stats("lms_erle_db"),
            "lms_steady_erle_db": stats("lms_steady_erle_db"),
            "lms_rmse": stats("lms_rmse"),
            "nlms_same_mu_erle_db": stats("nlms_same_mu_erle_db"),
            "nlms_same_mu_steady_erle_db": stats("nlms_same_mu_steady_erle_db"),
            "nlms_same_mu_rmse": stats("nlms_same_mu_rmse"),
            "nlms_erle_db": stats("nlms_erle_db"),
            "nlms_steady_erle_db": stats("nlms_steady_erle_db"),
            "nlms_rmse": stats("nlms_rmse"),
        },
    }
    with open(res_dir / "s1_summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Produce the 4 mandatory figures (and save to both results/s1_one_speaker_static/ and results/figures/)
    generate_s1_figures(h_rir0, rep_data, metric_rows, [res_dir, fig_dir])

    return metric_rows, summary


def generate_s1_figures(
    h_rir0: np.ndarray,
    rep_data: Dict[str, np.ndarray],
    metric_rows: List[Dict[str, object]],
    out_dirs: List[Path],
) -> None:
    """Generate the 4 required S1 figures specified in CDX-02."""
    taps = np.arange(len(h_rir0))
    time_ms = taps * 1000.0 / float(SAMPLE_RATE)

    # 1. Room Impulse Response plot (True RIR vs Learned Weights)
    fig1, ax1 = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    markerline, stemlines, baseline = ax1.stem(
        time_ms, h_rir0, linefmt="#1f77b4", markerfmt="o", basefmt="k-", label="True Room RIR h_room[n]"
    )
    plt.setp(markerline, markersize=4)
    ax1.plot(
        time_ms, rep_data["w_nlms_opt"], "--", color="#d62728", lw=1.5,
        label="Learned NLMS Weights w[n] (S1_T01)"
    )
    ax1.plot(
        time_ms, rep_data["w_lms"], ":", color="#2ca02c", lw=1.5,
        label="Learned LMS Weights w[n] (S1_T01)"
    )
    ax1.set_title("Figure 1: Fixed Simulated Room Impulse Response (rir_0, N=128 Taps, 8 ms)")
    ax1.set_xlabel("Delay Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    fig1.tight_layout()

    # 2. Representative Waveform Before/After Cancellation (S1_T01)
    t_sec = np.arange(len(rep_data["d"])) / float(SAMPLE_RATE)
    fig2, axes2 = plt.subplots(3, 1, figsize=(9.5, 6.2), sharex=True, sharey=True, dpi=150)
    axes2[0].plot(t_sec, rep_data["d"], color="#444444", lw=0.7)
    axes2[0].set_title("Simulated Microphone Echo Signal d[n] = x[n] * h_room[n] (Before Cancellation)")
    axes2[0].set_ylabel("Amplitude")
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(t_sec, rep_data["e_lms"], color="#1f77b4", lw=0.7)
    axes2[1].set_title(f"Residual e[n] After Standard LMS (N=128, mu={SAFE_MU:.2f}) — Approaches Silence")
    axes2[1].set_ylabel("Amplitude")
    axes2[1].grid(True, alpha=0.3)

    axes2[2].plot(t_sec, rep_data["e_nlms_opt"], color="#d62728", lw=0.7)
    axes2[2].set_title(f"Residual e[n] After NLMS (N=128, mu={NLMS_MU:.2f}) — Rapid Convergence to Silence")
    axes2[2].set_xlabel("Time (s)")
    axes2[2].set_ylabel("Amplitude")
    axes2[2].grid(True, alpha=0.3)
    fig2.suptitle("Figure 2: Representative Waveform Before and After Echo Cancellation (Clip S1_T01)", y=0.99)
    fig2.tight_layout()

    # 3. Representative Spectrogram Before/After Cancellation (S1_T01)
    fig3, axes3 = plt.subplots(1, 3, figsize=(12.0, 4.2), sharey=True, dpi=150)
    signals_to_plot = [
        (rep_data["d"], "Before AEC: Mic Echo d[n]"),
        (rep_data["e_lms"], f"After Standard LMS (mu={SAFE_MU:.2f}): Residual e[n]"),
        (rep_data["e_nlms_opt"], f"After NLMS (mu={NLMS_MU:.2f}): Residual e[n]"),
    ]
    im = None
    for ax, (sig, title) in zip(axes3, signals_to_plot):
        f_bins, t_bins, zxx = signal.stft(sig, fs=SAMPLE_RATE, window="hann", nperseg=512, noverlap=256)
        mag_db = 20.0 * np.log10(np.abs(zxx) + 1e-8)
        im = ax.pcolormesh(t_bins, f_bins / 1000.0, mag_db, vmin=-120.0, vmax=-10.0, cmap="magma", shading="gouraud")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
    axes3[0].set_ylabel("Frequency (kHz)")
    fig3.colorbar(im, ax=axes3.ravel().tolist(), label="Magnitude (dB)")
    fig3.suptitle("Figure 3: Representative STFT Spectrogram Before and After Cancellation (Clip S1_T01)", y=1.02)

    # 4. Bar Chart of ERLE for all 10 clips (LMS vs NLMS)
    trials = [str(r["trial_id"]) for r in metric_rows]
    erle_lms = [float(r["lms_erle_db"]) for r in metric_rows]
    erle_nlms_same = [float(r["nlms_same_mu_erle_db"]) for r in metric_rows]
    erle_nlms_opt = [float(r["nlms_erle_db"]) for r in metric_rows]

    x_pos = np.arange(len(trials))
    width = 0.26
    fig4, ax4 = plt.subplots(figsize=(10.0, 4.8), dpi=150)
    ax4.bar(x_pos - width, erle_lms, width, label=f"Standard LMS (N=128, mu={SAFE_MU:.2f})", color="#1f77b4")
    ax4.bar(x_pos, erle_nlms_same, width, label=f"NLMS Identical Setting (N=128, mu={SAFE_MU:.2f})", color="#ff7f0e")
    ax4.bar(x_pos + width, erle_nlms_opt, width, label=f"NLMS Normalized Step (N=128, mu={NLMS_MU:.2f})", color="#2ca02c")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(trials)
    ax4.set_ylabel("Full-Clip ERLE (dB)")
    ax4.set_xlabel("Clip Trial ID (10 Clean Speech Clips)")
    ax4.set_title("Figure 4: Echo Return Loss Enhancement (ERLE) Across All 10 Static-Room Clips")
    ax4.grid(True, axis="y", alpha=0.3)
    ax4.legend(loc="upper right")
    fig4.tight_layout()

    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig1.savefig(d / "fig1_room_impulse_response.png", bbox_inches="tight")
        fig2.savefig(d / "fig2_waveform_before_after.png", bbox_inches="tight")
        fig3.savefig(d / "fig3_spectrogram_before_after.png", bbox_inches="tight")
        fig4.savefig(d / "fig4_erle_bar_chart_10clips.png", bbox_inches="tight")

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)


def main() -> None:
    print("=== Step 1: Creating Fixed Room Impulse Responses (rir_0, rir_1) ===")
    h_rir0 = create_fixed_room_ir(length=FILTER_TAPS, rir_id="rir_0")
    h_rir1 = create_fixed_room_ir(length=FILTER_TAPS, rir_id="rir_1")
    save_rir_catalog(h_rir0, h_rir1)
    create_reserved_s2_s3_manifests()

    print("=== Step 2: Running Phase 0 Synthetic Known-Room Gate (CDX-01B) ===")
    gate0 = run_phase0_sanity_gate(h_rir0)
    print(json.dumps(gate0, indent=2))

    print("=== Step 3: Preparing 10 Clean Speech Clips (16 kHz, 8.0s Mono) ===")
    clips_meta = prepare_10_clean_speech_clips()
    print(f"Prepared {len(clips_meta)} clean speech clips in data/source_clips/.")

    print("=== Step 4: Running S1 Static-Room Echo Experiment (CDX-02 & CDX-03) ===")
    metric_rows, summary = run_s1_experiment(clips_meta, h_rir0)
    print(json.dumps(summary["aggregates"], indent=2))
    print("=== All S1 deliverables generated and verified successfully! ===")


if __name__ == "__main__":
    main()
