# Acoustic Echo Cancellation & Noise Analysis (DSP Final Project)

โปรเจกต์ไฟนอลวิชา **517354 Digital Signal Processing (DSP)**  
เรื่อง **การวิเคราะห์และเปรียบเทียบการตัดเสียงสะท้อนในห้องด้วยตัวกรองปรับตัวได้เชิงทดลอง (LMS vs NLMS with Delay Compensation)**  
ภาควิชาคอมพิวเตอร์ คณะวิทยาศาสตร์ มหาวิทยาลัยศิลปากร

---

## ⚡ ทางลัดเอกสารและผลการทดลอง (Quick Navigation)

| รายการ | รูปแบบไฟล์ | คำอธิบาย |
| :--- | :--- | :--- |
| 📄 **เล่มรายงานฉบับสมบูรณ์** | [PDF](docs/report/DSP_FINAL_PROJECT_REPORT.pdf) \| [HTML](docs/report/DSP_FINAL_PROJECT_REPORT.html) \| [MD](docs/report/DSP_FINAL_PROJECT_REPORT.md) | รายงานฉบับเต็ม 13 หัวข้อ อ้างอิงทฤษฎี สมการคณิตศาสตร์ และกราฟผลลัพธ์ |
| 🎙️ **สไลด์นำเสนอ 15 สไลด์** | [PDF](docs/presentation/DSP_FINAL_PROJECT_PRESENTATION.pdf) \| [HTML](docs/presentation/DSP_FINAL_PROJECT_PRESENTATION.html) \| [MD](docs/presentation/DSP_FINAL_PROJECT_PRESENTATION_SLIDES.md) | สไลด์ 16:9 ออกแบบกระชับตามเวลาพูด 12 นาที + ตอบคำถาม 5 นาที |
| 🧭 **สรุปตอบคำถามนำเสนอ** | [Markdown](docs/presentation/PRESENTATION_NOTES.md) | โพยเจาะลึกคำถามทางเทคนิค (DAF, Spectral Leakage, Savitzky-Golay) |
| 📓 **Main Experiment Notebook** | [main.ipynb](notebooks/main.ipynb) | โน้ตบุ๊กหลักสำหรับรันการทดลองจริง มีตัวเล่นเสียงฟัง Clean vs Echo |
| 📓 **Step-by-Step Learning Guide** | [step_by_step_dsp_guide.ipynb](notebooks/step_by_step_dsp_guide.ipynb) | คู่มือการทดลองทีละสเต็ป 6 ขั้นตอน รันผ่าน 100% พร้อมคำอธิบายภาษาไทย |

---

## 1. สถานการณ์การทดลองทั้ง 3 กรณี (Experimental Scenarios)

1. **S1 (`s1_one_person`) — คนพูด 1 คน ในห้องและตำแหน่งคงที่ (Static Room, One Speaker)**:
   - มีเฉพาะเสียงพูดต้นทางจากลำโพง (Far-End) สะท้อนผ่านผลตอบสนองของห้องคงที่ (`fixed_room`) เข้าสู่ไมโครโฟน โดยไม่มีคนพูดแทรกหน้าไมค์ เป้าหมายคือตัดเสียงสะท้อนออกจนสัญญาณคงเหลือเข้าใกล้ความเงียบ (NLMS ทำได้สูงถึง **69.32 dB ERLE**)
2. **S2 (`s2_two_people_static`) — คนพูด 2 คน ในห้องและตำแหน่งคงที่ (Static Room, Two Speakers - Double-talk)**:
   - มีทั้งเสียงสะท้อนจากลำโพง (Far-End) และเสียงคนพูดฝั่งใกล้หน้าไมโครโฟน (Near-End) ในห้องที่มีการสะท้อนคงที่ (`fixed_room`) เป้าหมายคือตัดเสียงสะท้อนจากลำโพงออกโดยไม่ทำลายเสียงคนพูดฝั่งใกล้
3. **S3 (`s3_moving_position`) — มีการขยับเปลี่ยนตำแหน่ง (Moving Position / Time-Varying Room Path)**:
   - จำลองการขยับตำแหน่งของผู้พูดหรือไมโครโฟน โดยการสลับผลตอบสนองของห้อง (`moving_room`) ตามช่วงเวลาที่กำหนด เพื่อทดสอบความสามารถในการปรับตัวตามการเปลี่ยนแปลงของตัวกรอง (Tracking & Re-convergence)

---

## 2. แผนผังโครงสร้างโฟลเดอร์และหน้าที่การทำงาน (Project Folder Map)

```text
FinalProject/
├── docs/                                   # รวมเอกสารรายงาน สไลด์นำเสนอ และโพยติวทั้งหมด
│   ├── report/                             # เล่มรายงานโครงงานฉบับสมบูรณ์ (PDF, HTML, MD)
│   │   ├── DSP_FINAL_PROJECT_REPORT.pdf
│   │   ├── DSP_FINAL_PROJECT_REPORT.html
│   │   └── DSP_FINAL_PROJECT_REPORT.md
│   ├── presentation/                       # สไลด์นำเสนอและโน้ตเตรียมพรีเซนต์
│   │   ├── DSP_FINAL_PROJECT_PRESENTATION.pdf
│   │   ├── DSP_FINAL_PROJECT_PRESENTATION.html
│   │   ├── DSP_FINAL_PROJECT_PRESENTATION_SLIDES.md
│   │   ├── PRESENTATION_NOTES.md
│   │   └── assets/                         # ภาพกราฟความละเอียดสูง fig1-fig7 และพรีวิวสไลด์
│   └── archive/                            # เอกสารระดมความคิดตอนเริ่มต้นโครงงาน
│       └── DSP_Final_Project_20_Ideas.md
│
├── notebooks/                              # สมุดทดลอง Jupyter Notebooks
│   ├── main.ipynb                          # โน้ตบุ๊กหลักสำหรับรันการทดลองจริงและฟังเสียง
│   ├── step_by_step_dsp_guide.ipynb        # โน้ตบุ๊กคู่มือทำความเข้าใจ 6 ขั้นตอนตามลำดับตรรกะ
│   └── project_guide.ipynb                 # สมุดแนะนำโครงสร้างโปรเจกต์
│
├── scripts/                                # สคริปต์อัตโนมัติสำหรับคอมไพล์เอกสารและสร้างกราฟ
│   ├── generate_all_plots.py               # สร้างภาพกราฟความละเอียดสูง fig1 ถึง fig7
│   ├── build_report_pdf.py                 # คอมไพล์เล่มรายงานฉบับสมบูรณ์เป็น HTML และ PDF
│   ├── build_presentation_pdf.py           # คอมไพล์สไลด์นำเสนอ 16:9 เป็น HTML และ PDF
│   └── build_step_by_step_notebook.py      # สร้างและรัน notebook step_by_step_dsp_guide.ipynb
│
├── src/                                    # ซอร์สโค้ด DSP ทั้งหมด (Pure NumPy & Clean Architecture)
│   ├── io/                                 # โหลดไฟล์เสียง บันทึกไฟล์เสียง และจัดการ Sample Rate
│   ├── simulation/                         # จัดการ RIR และจำลองสัญญาณเสียงสะท้อนในห้อง
│   ├── adaptive_filters/                   # อัลกอริทึมตัวกรองปรับตัวได้ LMS และ NLMS
│   ├── metrics/                            # คำนวณตัวชี้วัด ERLE, RMSE, และ SNR
│   ├── visualization/                      # วาดกราฟ Waveform, Spectrogram, และกราฟสรุป
│   ├── pipelines/                          # ไปป์ไลน์สั่งรันการทดลอง (S1, S2, S3)
│   └── common/                             # ฟังก์ชันตรวจสอบความถูกต้อง ค่าคงที่กลาง
│
├── results/                                # โฟลเดอร์เก็บกราฟและข้อมูลผลการทดลองแยกตามสถานการณ์
│   ├── s1_one_person/                      # ผลลัพธ์และรูปกราฟกรณี S1 (fig4_s1_erle.png)
│   ├── s2_two_people_static/               # ผลลัพธ์และรูปกราฟกรณี S2 (fig5_s2_doubletalk.png)
│   └── s3_moving_position/                 # ผลลัพธ์และรูปกราฟกรณี S3 (fig6_s3_moving.png)
│
├── configs/                                # การตั้งค่าพารามิเตอร์ของการทดลองแต่ละสถานการณ์
├── manifests/                              # ตารางดัชนีรายชื่อคลิปเสียงและห้องจำลอง
├── data/                                   # [GIT-IGNORED] โฟลเดอร์ข้อมูลเสียงในเครื่อง (ห้ามขึ้น Git)
├── start-jupyter.cmd                       # คำสั่งเปิด Jupyter Lab/Notebook สำหรับ Windows
├── .gitignore                              # กฎบล็อกไฟล์เสียง โฟลเดอร์ data/ และ scratch/
└── README.md                               # เอกสารแนะนำโปรเจกต์ฉบับนี้
```

---

## 3. รายละเอียดหน้าที่ของแต่ละโฟลเดอร์ใน `src/` (Responsibility-Based Modules)

เพื่อให้โค้ดไม่ซ้ำซ้อน ทั้ง 3 สถานการณ์ (`S1`, `S2`, `S3`) จะเรียกใช้โมดูลพื้นฐานชุดเดียวกันใน `src/` โดยแบ่งหน้าที่ดังนี้:

| โฟลเดอร์ใน `src/` | หน้าที่รับผิดชอบหลัก | การใช้งานร่วมกันใน S1 / S2 / S3 |
| :--- | :--- | :--- |
| **`src/io/`** | อ่านไฟล์เสียงต้นฉบับ บันทึกไฟล์เสียงจำลอง และตรวจสอบ/แปลงความถี่สุ่มสัญญาณ (Sample Rate) | ใช้เหมือนกันทุกกรณี |
| **`src/simulation/`** | โหลดผลตอบสนองของห้อง (RIR) และนำเสียงพูดมาจำลองเป็นสัญญาณไมโครโฟน (ทั้งห้องนิ่งและห้องที่มีการสลับตำแหน่ง) | ใช้เหมือนกันทุกกรณี |
| **`src/adaptive_filters/`** | ประมวลผลตัวกรองปรับตัวได้ **LMS** และ **NLMS** เพื่อประมาณค่าเสียงสะท้อนและหักล้างออกจากสัญญาณไมโครโฟน | ใช้เหมือนกันทุกกรณี |
| **`src/metrics/`** | คำนวณค่าตัวชี้วัดเชิงปริมาณ ได้แก่ **ERLE** และ **RMSE** รวมถึงการวัดผลเฉพาะของแต่ละสถานการณ์ | ใช้เหมือนกันทุกกรณี |
| **`src/visualization/`** | พล็อตกราฟผลตอบสนองของห้อง กราฟเปรียบเทียบคลื่นเสียง (Waveform) กราฟสเปกโตรแกรม (Spectrogram) และกราฟสรุปค่าวัดผล | ใช้เหมือนกันทุกกรณี |
| **`src/common/`** | เก็บค่าคงที่กลาง การจัดการพาธของโฟลเดอร์ และฟังก์ชันตรวจสอบความถูกต้องของข้อมูล | ใช้เหมือนกันทุกกรณี |
| **`src/pipelines/`** | ตัวเชื่อมต่อลำดับการทำงานของแต่ละสถานการณ์ (`s1_one_person`, `s2_two_people_static`, `s3_moving_position`) | แยกตามสถานการณ์ S1, S2, S3 |

---

## 4. คำสั่งสำหรับคอมไพล์เอกสารใหม่ (Build Automation)

หากมีการแก้ไขเนื้อหาใน Markdown หรือต้องการเจนกราฟใหม่ สามารถรันผ่าน Command Line:

```bash
# 1. รันเจนภาพกราฟประกอบทั้งหมด (fig1 - fig7)
python scripts/generate_all_plots.py

# 2. คอมไพล์เล่มรายงานฉบับสมบูรณ์เป็น PDF & HTML
python scripts/build_report_pdf.py

# 3. คอมไพล์สไลด์นำเสนอเป็น PDF & HTML
python scripts/build_presentation_pdf.py
```

---

## 5. ข้อควรระวังสำคัญเรื่อง GitHub

> [!NOTE]
> **ข้อมูลสำหรับการทดลอง (Curated Benchmark Clips):**  
> ชุดข้อมูลเสียงที่จำเป็นในการรันการทดลองทั้งหมด (~2.5 MB) ได้รับการจัดเตรียมไว้ในโฟลเดอร์ `data/` เรียบร้อยแล้ว เพื่อให้เพื่อนร่วมทีมหรือผู้ตรวจสามารถ clone โครงงานและรันการทดลองซ้ำได้ทันที (ส่วนชุดข้อมูลดิบภายนอกขนาดใหญ่ เช่น Microsoft AEC Challenge ฉบับเต็มหลายกิกะไบต์ จะถูกบล็อกใน `.gitignore`)
