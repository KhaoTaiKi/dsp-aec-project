# Acoustic Echo Cancellation & Noise Analysis (DSP Final Project)

โปรเจกต์ไฟนอลวิชา **Digital Signal Processing (DSP)**  
เรื่อง **การวิเคราะห์ชุดข้อมูลเสียงสะท้อนและเปรียบเทียบกระบวนการตัดเสียงรบกวนด้วยเทคนิค DSP เชิงทดลอง (Dataset-Driven Acoustic Echo & Noise Analysis: A Comparative Study of Classical DSP Techniques)**

---

## 1. โครงสร้างโปรเจกต์ (Repository Structure)

```text
FinalProject/
├── src/                        # โมดูลโค้ดหลักของโปรเจกต์
│   ├── lms.py                  # อัลกอริทึม Standard LMS Adaptive Filter & Preprocessing/Alignment
│   ├── nlms.py                 # อัลกอริทึม Normalized LMS (NLMS) & Comparative DSP Methods
│   ├── metrics.py              # การคำนวณตัวชี้วัด ERLE (dB), SNR, PESQ, STOI
│   └── baseline_webrtc.py      # การรันเทียบกับ Baseline มาตรฐานอุตสาหกรรม (WebRTC AEC)
├── notebooks/
│   └── experiment.ipynb        # สมุดทดลองเปรียบเทียบผลลัพธ์ (What-If & Case-by-Case Study)
├── results/
│   └── figures/                # กราฟ Waveform, Cross-Correlation Lag, Learning Curve, STFT Spectrogram
├── data/                       # [GIT-IGNORED] โฟลเดอร์เก็บไฟล์เสียง Dataset (ห้ามอัปโหลดขึ้น GitHub!)
├── .gitignore                  # กฎป้องกันการอัปโหลดไฟล์ Dataset (.wav, .mp3, .m4a, data/)
├── README_AEC_Project_DataFlow.md  # เอกสารสถาปัตยกรรม Data Flow ภาพรวมและรายส่วนงาน
└── README.md                   # คู่มือการตั้งค่าและดาวน์โหลด Dataset ฉบับนี้
```

---

## 2. ข้อควรระวังสำคัญ: ห้ามอัปโหลด Dataset ขึ้น GitHub เด็ดขาด!

> [!CAUTION]
> **ห้าม Commit หรือ Push ไฟล์เสียงและโฟลเดอร์ `data/` ขึ้นบน GitHub Repository เด็ดขาด**  
> เนื่องจากชุดข้อมูล **Microsoft AEC Challenge** มีขนาดใหญ่มาก (หลาย GB) และอยู่ภายใต้เงื่อนไขสิทธิ์การใช้งานเพื่อการวิจัยและการศึกษาเท่านั้น ไฟล์ `.gitignore` ในโปรเจกต์นี้ได้บล็อก `data/`, `*.wav`, `*.mp3`, `*.m4a` ไว้เรียบร้อยแล้ว

---

## 3. วิธีดาวน์โหลดและเตรียมชุดข้อมูลสำหรับสมาชิกในทีม (Dataset Setup Guide)

สมาชิกในทีมสามารถเตรียมชุดข้อมูลเข้าสู่โฟลเดอร์ `data/` ได้ 2 ช่องทางดังนี้:

### ช่องทางที่ 1: ดาวน์โหลดจาก Official GitHub ของ Microsoft AEC Challenge
1. เข้าไปที่ Repository ทางการของ Microsoft:  
   [https://github.com/microsoft/AEC-Challenge](https://github.com/microsoft/AEC-Challenge)
2. ดาวน์โหลดชุดข้อมูลทดสอบ (`datasets/real_recordings` และ `datasets/synthetic`)
3. นำไฟล์ที่คัดเลือกแล้ว (อย่างละ 10–25 คู่ไฟล์) มาวางไว้ในโฟลเดอร์ `data/` ตามโครงสร้างนี้:
   ```text
   data/
   ├── real_recordings/
   │   ├── farend_speech/      # เสียงต้นทางที่ลำโพงเล่น (farend.wav)
   │   ├── mic_signals/        # เสียงที่ไมโครโฟนรับได้ มีเสียงก้องปน (mic.wav)
   │   └── nearend_speech/     # เสียงพูดสะอาดฝั่งใกล้ ใช้เป็น Reference วัดผล (nearend.wav)
   └── synthetic/
       ├── farend_speech/
       ├── mic_signals/
       └── nearend_speech/
   ```

### ช่องทางที่ 2: ใช้งานร่วมกันผ่าน Kaggle Dataset (แนะนำสำหรับรันบน Cloud / แชร์ในกลุ่ม)
- หากทำงานร่วมกับเพื่อนผ่าน **Kaggle Notebooks** สามารถอัปโหลด Subset ที่คัดเลือกแล้ว (เช่น 20–50 คู่ไฟล์) ขึ้นเป็น **Private/Team Kaggle Dataset** แล้วเชื่อมเข้ากับ `notebooks/experiment.ipynb` ได้โดยตรง ไม่ต้องให้แต่ละคนดาวน์โหลดไฟล์ขนาดหลาย GB ลงเครื่องตัวเองซ้ำ

---

## 4. การติดตั้งไลบรารีที่จำเป็น (Dependencies)

```bash
pip install numpy scipy matplotlib soundfile librosa pesq pystoi
```
*(สำหรับรันเทียบกับ WebRTC AEC Baseline สามารถติดตั้งเพิ่มเติมด้วย `pip install webrtc-audio-processing` หรือ `speechbrain`)*
