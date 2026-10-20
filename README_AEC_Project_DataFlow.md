# สถาปัตยกรรมและแผนผังการไหลของข้อมูล (Data Flow Architecture): DSP Final Project

> **ชื่อโครงงาน**: การวิเคราะห์ชุดข้อมูลเสียงสะท้อนและเปรียบเทียบกระบวนการตัดเสียงรบกวนด้วยเทคนิค DSP เชิงทดลอง (Dataset-Driven Acoustic Echo & Noise Analysis: A Comparative & Ablation Study of Classical DSP Techniques)  
> **วิชา**: Digital Signal Processing (DSP) | **อาจารย์ผู้สอน**: อ.ดร.ณัฐพงศ์ จิวมั่งมี  
> **กำหนดส่ง**: วันอังคารที่ 10 พฤศจิกายน 2569  
> **หัวใจหลักของโปรเจกต์ตามแนวทางอาจารย์ผู้สอน (Professor's Core Philosophy)**:  
> 1. **ไม่จำเป็นต้องใช้โมเดล AI/ML (และไม่จำเป็นต้องยึดติดกับ `if-else` ซับซ้อน)**: แก่นแท้ของงานคือการนำ **Dataset จริง (Microsoft AEC Challenge: `real_recordings` vs `synthetic`)** มาทดลองด้วย **เครื่องมือและทฤษฎีที่เรียนในวิชา DSP (Week 1–12)**  
> 2. **เน้นการตั้งคำถามและเปรียบเทียบว่า "ถ้าเปลี่ยนวิธีจะเป็นอย่างไร?" (What-If & Method Comparison)**:  
>    - ถ้าเปลี่ยนชนิดตัวกรอง (FIR vs IIR, 5 ตระกูล IIR, `lfilter` vs `filtfilt`) ผลลัพธ์ต่างกันอย่างไร?  
>    - ถ้าทำ Time Alignment ด้วย Cross-Correlation ก่อนเข้าตัวกรอง เทียบกับไม่ทำ อะไรจะเกิดขึ้น?  
>    - ถ้าเปลี่ยนวิธีการตัดเสียงก้อง/เสียงรบกวน (Frequency-domain Spectral Subtraction บน STFT vs Time-domain LMS vs NLMS vs Source Separation PCA/ICA) และเปลี่ยนพารามิเตอร์ ($N\text{ taps}, \mu$) จะเจอปัญหาอะไร?  
>    - **สรุปให้ชัดว่า**: วิธีไหน **ดีกว่าในเคสไหน** (เช่น ห้องเล็ก, ห้องจำลอง, เสียงนิ่ง) และวิธีไหน **แย่กว่า/พังในเคสไหน** (เช่น ห้องก้องยาว, ห้องอัดจริงที่มีความไม่เป็นเชิงเส้น, ช่วงพูดสวนกัน Double-Talk, หรือตั้ง $\mu$ ใหญ่เกินจนระบบระเบิด)  
>
> **การแบ่งบทบาทเอเจนต์ (Agent Governance)**:  
> - **Antigravity (`AGY`) — ผู้ลงมือพัฒนาหลัก (Primary Implementer)**: เขียนสคริปต์สำรวจ Dataset, สร้างชุดการทดลองเปรียบเทียบแต่ละวิธีตามเนื้อหา Week 1–12, รันวัดผล (ERLE, SNR, PESQ, STOI), บันทึกเคสที่สำเร็จและเคสที่ล้มเหลว, พล็อตกราฟ และเขียนรายงานวิเคราะห์เชิงลึก  
> - **OpenAI Codex (`CDX`) — ผู้ควบคุมและตรวจสอบคุณภาพ (Supervisor, Controller & QA Auditor)**: ตรวจสอบความถูกต้องของสมการและโค้ดทุกการทดลอง, ตรวจสอบว่าการเปรียบเทียบยุติธรรม (Controlled Experiment), ตรวจทานตัวเลขและเหตุผลทางทฤษฎี DSP ว่าทำไมวิธีหนึ่งจึงดีกว่าหรือแย่กว่าอีกวิธีในแต่ละเคส  

---

## ส่วนที่ 1: แผนผังการไหลของข้อมูลระดับภาพรวมทั้งระบบ (1. Overall System Data Flow)

แผนผังภาพรวมแสดงการนำ **Dataset** ไหลผ่าน **4-Stage DSP Pipeline** โดยในแต่ละขั้นจะมีการ **"สลับเปลี่ยนวิธีการที่เรียนมา (Method Variations)"** เพื่อสังเกตปัญหาและเปรียบเทียบว่าวิธีใดเหมาะกับสถานการณ์ใด:

```mermaid
flowchart TD
    subgraph S1["Stage 1: Read Signal & Dataset Scenarios (Microsoft AEC Challenge)"]
        DS["ชุดข้อมูลทดสอบ (10 -> 30 -> 50 คู่สัญญาณ)<br/>- เคสที่ 1: Synthetic (จำลองห้องอุดมคติ เชิงเส้น)<br/>- เคสที่ 2: Real Recordings Single-Talk (ห้องจริง มีแต่เสียงลำโพงสะท้อน)<br/>- เคสที่ 3: Real Recordings Double-Talk & Noisy (พูดสวนกัน + เสียงรบกวนแทรก)"]
    end

    subgraph S2["Stage 2: Preprocess & Alignment Variations (Week 2, 5, 7, 8, 9, 10)"]
        PRE_A["การทดลองย่อย 2A: Time Alignment (Week 5, 10)<br/>- ไม่จัดแนวเวลา (Raw Delay)<br/>- จัดแนวเวลาด้วย Cross-Correlation (np.correlate)"]
        PRE_B["การทดลองย่อย 2B: Pre-Filtering ย่านเสียงพูด 80-4000 Hz (Week 7, 8, 9)<br/>- FIR Window Method (firwin: Boxcar, Hann, Hamming, Flat-top)<br/>- IIR 5 ตระกูล (Butterworth, Cheby I, Cheby II, Elliptic, Bessel)<br/>- Causal (lfilter) vs Zero-Phase (filtfilt)"]
        PRE_A --> PRE_B
    end

    subgraph S3["Stage 3: Represent, Measure & Core Cancellation Methods (Week 1, 3, 4, 6, 11, 12)"]
        direction TB
        M_FREQ["แนวทางที่ 1: โดเมนความถี่ & เวลา-ความถี่ (Week 6, 12)<br/>FFT Masking / STFT Spectral Subtraction<br/>(ลบสเปกตรัมเสียงลำโพงออกจากไมค์บน Spectrogram)"]
        M_BSS["แนวทางที่ 2: การแยกแหล่งกำเนิดสัญญาณ (Week 11)<br/>Covariance -> PCA -> Whitening -> FastICA<br/>(มองไมค์และลำโพงเป็น Mixed Signals 2 ช่อง)"]
        M_LMS["แนวทางที่ 3: ตัวกรองปรับตัวได้ FIR (LMS) (Week 1, 3, 4, 7, 8)<br/>y(n) = w^T x(n), e(n) = d(n) - y(n)<br/>ทดลองเปลี่ยน N = 128, 256, 512, 1024 และ mu = 0.001, 0.01, 0.1"]
        M_NLMS["แนวทางที่ 4: Normalized LMS (NLMS) & Simple Energy Gate (Week 3, 9)<br/>ปรับ Step size ตามพลังงานอินพุต P_x อัตโนมัติ<br/>+ เทียบกรณีล็อกค่าน้ำหนักเมื่อพลังงานไมค์พุ่งสูง"]
    end

    subgraph S4["Stage 4: Evaluation, Failure Analysis & Case-by-Case Comparison"]
        EVAL["วัดผลเชิงปริมาณ (Metrics)<br/>- ERLE (dB) (ลดเสียงก้องได้กี่ dB)<br/>- SNR / RMSE เทียบกับ nearend.wav<br/>- PESQ & STOI (คุณภาพและความชัดเจนของเสียงพูด)"]
        ANALYZE["วิเคราะห์เปรียบเทียบเชิงวิศวกรรม (What Works & What Fails)<br/>- เปลี่ยนวิธีแล้วกราฟ Waveform / Spectrogram เปลี่ยนอย่างไร?<br/>- เจอปัญหาอะไร? (เช่น เฟสเลื่อน, Musical Noise, ตัวกรองระเบิด, ลู่เข้าช้า)<br/>- สรุปตาราง: วิธีไหนชนะในเคสไหน และแพ้ในเคสไหน เพราะเหตุใด?"]
        EVAL --> ANALYZE
    end

    DS --> PRE_A
    PRE_B --> M_FREQ
    PRE_B --> M_BSS
    PRE_B --> M_LMS
    PRE_B --> M_NLMS

    M_FREQ --> EVAL
    M_BSS --> EVAL
    M_LMS --> EVAL
    M_NLMS --> EVAL
```

---

## ส่วนที่ 2: แผนผังการไหลของข้อมูลแยกตามส่วนงานและจุดเปรียบเทียบ (2. Component Data Flows & Experimental Comparisons)

### 2.1 ส่วนงานที่ 1: การสำรวจ Dataset และผลกระทบของการจัดแนวเวลา & การกรองเบื้องต้น (Preprocessing & Alignment Experiments)
นำความรู้ **Week 2 (Normalization/Detrend)**, **Week 5 & 10 (Cross-Correlation & Multi-rate)** และ **Week 7, 8, 9 (FIR vs IIR, `lfilter` vs `filtfilt`)** มาทดลองกับ Dataset เพื่อดูว่าถ้าทำหรือไม่ทำแต่ละขั้น จะเกิดปัญหาอะไร:

```mermaid
flowchart LR
    IN["โหลดคู่ไฟล์จาก Dataset<br/>farend x[n], mic d[n], nearend s[n]<br/>(16 kHz, ตัดช่วง 10-30s, Detrend & Normalize)"] --> EXP1{"การทดลองที่ 1:<br/>ชดเชย Echo Lag ด้วย np.correlate (Week 5, 10)"}

    EXP1 -->|"วิธีที่ 1A: ไม่ทำ Time Alignment"| P1A["ปัญหาที่พบ:<br/>หากลำโพงดีเลย์เข้าไมค์ 40 ms (640 samples)<br/>แต่ใช้ตัวกรอง N=128 หรือ 256 taps<br/>ตัวกรองจะครอบคลุมไปไม่ถึงจุดที่มีเสียงก้องเลย! (ERLE ~ 0 dB)"]
    EXP1 -->|"วิธีที่ 1B: ทำ Cross-Correlation Alignment"| P1B["ผลลัพธ์ที่ได้:<br/>หาจุด Peak Lag แล้วเลื่อนสัญญาณ x[n] ให้ตรงกับ d[n]<br/>ช่วยให้ตัวกรองขนาดเล็ก (N=128, 256) ตัดเสียงก้องได้ทันที!"]

    P1A --> EXP2{"การทดลองที่ 2:<br/>กรองย่านเสียงพูด 80-4000 Hz (Week 7, 8, 9)"}
    P1B --> EXP2

    EXP2 -->|"วิธีที่ 2A: FIR (firwin) vs IIR (butter/ellip/bessel)"| P2A["เปรียบเทียบ:<br/>- FIR รักษา Linear Phase แต่ใช้จำนวน Taps เยอะ<br/>- IIR (Cheby/Elliptic) ขอบชันมากแต่เกิด Phase Distortion"]
    EXP2 -->|"วิธีที่ 2B: lfilter vs filtfilt (Week 9)"| P2B["ปัญหาสำคัญที่ค้นพบ:<br/>หากใช้ lfilter กรองเฉพาะ x[n] หรือกรองไม่เหมือนกัน<br/>เฟสของ x[n] กับ d[n] จะบิดไม่เท่ากัน ทำให้หักล้างเสียงก้องไม่ได้!<br/>แต่ถ้าใช้ filtfilt (Zero-Phase) เฟสไม่เลื่อน จึงหักล้างคลื่นได้แม่นยำ"]
```

---

### 2.2 ส่วนงานที่ 2: การเปรียบเทียบวิธีตัดเสียงก้องข้ามโดเมน (Time-Domain vs Frequency-Domain vs Source Separation)
นำความรู้จาก **Week 6 & 12 (FFT / STFT Spectrogram)**, **Week 11 (PCA / FastICA)** และ **Week 1, 3, 4, 7 (FIR Convolution & LMS)** มาประชันกันบนข้อมูลชุดเดียวกัน:

```mermaid
flowchart TD
    SIG["สัญญาณที่ผ่านการเตรียมข้อมูลแล้ว<br/>x[n] (ลำโพง) และ d[n] (ไมโครโฟน)"] --> C1["วิธีที่ 1: STFT Spectral Subtraction (Week 6, 12)<br/>แปลง STFT -> ประมาณขนาดสเปกตรัมเสียงก้อง |X(m,k)|<br/>-> ลบออกจากขนาดสเปกตรัมไมค์ |D(m,k)| -> แปลง ISTFT กลับ"]
    SIG --> C2["วิธีที่ 2: PCA & FastICA Source Separation (Week 11)<br/>รวม [x[n], d[n]] เป็นเมทริกซ์ 2 ช่อง -> ทำ Whitening & FastICA<br/>เพื่อแยกแหล่งกำเนิดเสียงลำโพงออกจากเสียงพูดฝั่งใกล้"]
    SIG --> C3["วิธีที่ 3: Time-Domain Adaptive FIR (LMS / NLMS)<br/>จำลอง Impulse Response ของห้อง h_room[n] ด้วยเวกเตอร์ w(n)<br/>คำนวณ y(n) = w^T(n)x(n) แล้วหักล้าง e(n) = d(n) - y(n)"]

    C1 --> R1["วิเคราะห์ข้อดี-ข้อเสียที่พบ:<br/>+ ทำความเข้าใจง่ายบน Spectrogram ไม่เสี่ยงระเบิด<br/>- ลบได้แต่ขนาด (Magnitude) แต่เฟสไม่ตรง และเกิดเสียงน้ำไหล (Musical Noise)"]
    C2 --> R2["วิเคราะห์ข้อดี-ข้อเสียที่พบ:<br/>+ แยกสัญญาณผสมเชิงเส้นแบบ Instantaneous (ไม่มีดีเลย์) ได้ดี<br/>- พังเมื่อเจอห้องจริงที่เป็น Convolutive Mixture (มีเสียงสะท้อนหลายจังหวะตามเวลา)"]
    C3 --> R3["วิเคราะห์ข้อดี-ข้อเสียที่พบ:<br/>+ จำลองการสะท้อนของห้อง (LTI Convolution) ได้ตรงตามฟิสิกส์จริง หักล้างได้ทั้งขนาดและเฟส<br/>- ต้องจูน N taps และ Step size (mu) อย่างระมัดระวัง"]
```

---

### 2.3 ส่วนงานที่ 3: เจาะลึกการปรับตัวกรอง LMS vs NLMS และการเปลี่ยนพารามิเตอร์ ($N, \mu$)
ทดสอบเปลี่ยนค่าพารามิเตอร์ตามแผนใน `Update งาน.txt` เพื่อสังเกตพฤติกรรมความเสถียร (**Week 8 Stability**), ความเร็วในการลู่เข้า, และปัญหาเมื่อคนสองฝั่งพูดพร้อมกัน:

```mermaid
flowchart TD
    IN_LMS["ตัวกรองปรับตัวได้ FIR<br/>y(n) = w^T(n)x(n)<br/>e(n) = d(n) - y(n)"] --> EXP_N["การทดลองที่ 3A: เปลี่ยนความยาวตัวกรอง N taps<br/>N in {128 (8ms), 256 (16ms), 512 (32ms), 1024 (64ms)}"]
    IN_LMS --> EXP_MU["การทดลองที่ 3B: เปลี่ยน Step Size (mu)<br/>mu in {0.001, 0.01, 0.1, 0.5}"]
    IN_LMS --> EXP_ALG["การทดลองที่ 3C: เปรียบเทียบสมการอัปเดตค่าน้ำหนัก<br/>Standard LMS vs Normalized LMS (NLMS)"]

    EXP_N --> RES_N["สิ่งที่ค้นพบจากการเปลี่ยน N:<br/>- ห้องเล็ก / Synthetic สั้น: N=128, 256 ลู่เข้าไวและเพียงพอ<br/>- ห้องใหญ่ก้องนาน: N=128 ตัดหางเสียงก้องไม่หมด ต้องใช้ N=512 หรือ 1024<br/>- ข้อเสียของ N=1024: คำนวณช้าลงและลู่เข้าช้ากว่าในช่วงต้น"]

    EXP_MU --> RES_MU["สิ่งที่ค้นพบจากการเปลี่ยน mu (Stability - Week 8):<br/>- mu เล็ก (0.001): เสถียรมาก แต่ลู่เข้าช้า เสียงก้องค่อยๆ หายช้า<br/>- mu พอดี (0.01): สมดุลระหว่างความเร็วและความนิ่ง<br/>- mu ใหญ่เกินขอบเขต 1/(N*P_x) (เช่น 0.1 หรือ 0.5 ใน LMS):<br/>  ค่าน้ำหนัก w(n) แกว่งและระเบิดสู่ Infinity (Blows up)!"]

    EXP_ALG --> RES_ALG["สิ่งที่ค้นพบระหว่าง LMS vs NLMS:<br/>- LMS ธรรมดา: เมื่อเสียงลำโพงดังขึ้นกะทันหัน พลังงาน P_x พุ่ง ทำให้ตัวกรองระเบิดง่าย<br/>- NLMS: หารด้วยพลังงาน x^T(n)x(n) + eps ทำให้ปรับตัวคงที่แม้เสียงพูดจะดัง-เบาไม่สม่ำเสมอ"]
```

---

### 2.4 ส่วนงานที่ 4: การวัดผลและการจัดทำตารางสรุป "วิธีไหนดีกว่า/แย่กว่าในเคสไหน" (Evaluation & Case Matrix Data Flow)

```mermaid
flowchart LR
    subgraph CASES["3 สถานการณ์ใน Dataset (Test Cases)"]
        CASE1["เคส A: Synthetic Echo (ห้องจำลองเชิงเส้น)"]
        CASE2["เคส B: Real Recording - Far-End Only (ห้องจริง เสียงก้องล้วน)"]
        CASE3["เคส C: Real Recording - Double-Talk (คนสองฝั่งพูดสวนกัน)"]
    end

    subgraph METRICS["ตัวชี้วัดมาตรฐาน (Metrics)"]
        M_ERLE["ERLE (dB): 10*log10(E[d^2]/E[e^2])<br/>Sliding Window 1 วินาที"]
        M_QUAL["SNR (dB), PESQ (-0.5..4.5), STOI (0..1)"]
    end

    CASE1 --> M_ERLE
    CASE2 --> M_ERLE
    CASE3 --> M_ERLE
    CASE1 --> M_QUAL
    CASE2 --> M_QUAL
    CASE3 --> M_QUAL

    M_ERLE --> TABLE["ตารางวิเคราะห์สรุปผล (Comparison & Failure Matrix)<br/>ระบุชัดเจนว่าแต่ละวิธีได้คะแนนเท่าไรในแต่ละเคส<br/>พร้อมอธิบายด้วยทฤษฎี DSP ว่าทำไมวิธีนั้นถึงชนะหรือพังในเคสนั้น"]
    M_QUAL --> TABLE
```

---

## ส่วนที่ 3: ตารางสรุปสมมติฐานการเปรียบเทียบ (What-If & Case-by-Case Comparison Matrix)

ตารางนี้คือหัวใจที่เราจะนำไปพิสูจน์ด้วยตัวเลขและกราฟจาก Dataset เพื่อส่งอาจารย์:

| การทดลอง / วิธีการที่เปรียบเทียบ | เนื้อหาวิชา DSP ที่ใช้ | ดีกว่าในเคสไหน? (จุดแข็ง) | แย่กว่าหรือเจอปัญหาอะไรในเคสไหน? (จุดอ่อน / ข้อจำกัด) |
| :--- | :---: | :--- | :--- |
| **1. ไม่ทำ vs ทำ Cross-Correlation Alignment (`np.correlate`)** | Week 5, Week 10 | **การทำ Alignment ดีกว่ามากในเคสที่ลำโพงห่างจากไมค์หรือมี Buffer Delay สูง** เพราะช่วยเลื่อนจุดเริ่มต้นเสียงก้องมาอยู่ในระยะของตัวกรองทันที | **ถ้าไม่ทำ Alignment**: ในไฟล์ `real_recordings` ที่มี Acoustic/System Delay เกินความยาว $N$ taps ตัวกรองจะมองไม่เห็นเสียงก้องเลย (ตัดไม่ออก) |
| **2. การกรองความถี่ด้วย `lfilter` vs `filtfilt` (IIR vs FIR)** | Week 7, Week 8, Week 9 | **`filtfilt` (Zero-Phase) และ Linear-Phase FIR ดีกว่าในการเตรียมสัญญาณ Offline** เพราะรักษาเฟสและความสัมพันธ์เชิงเวลาระหว่าง $x[n]$ กับ $d[n]$ ไว้ได้ 100% | **`lfilter` (Non-linear Phase IIR เช่น Elliptic/Cheby)**: หากนำไปกรอง $x[n]$ โดยไม่ระวัง จะทำให้เฟสของความถี่ต่างๆ เลื่อนไม่เท่ากัน ส่งผลให้หักล้างคลื่น ($d[n] - y[n]$) ไม่สนิท |
| **3. STFT Spectral Subtraction (ลบในโดเมนความถี่)** | Week 6, Week 12 | **ดีกว่าในเคสที่มีเสียงรบกวนคงที่ (Stationary Noise / 50 Hz Hum)** หรือเมื่อไม่ทราบเวลา Delay ที่แน่นอน เพราะลบขนาดสเปกตรัมออกโดยตรงและไม่เสี่ยงที่ตัวกรองจะระเบิด | **แย่กว่าในเคสเสียงพูดสะท้อนในห้องจริง**: เพราะการสะท้อนในห้องยาวข้ามหลายเฟรม STFT และลบได้เฉพาะขนาด (Magnitude) ไม่ได้แก้เฟส อีกทั้งทำให้เกิดเสียงเพี้ยนคล้ายน้ำไหล (Musical Noise) |
| **4. PCA / FastICA (Blind Source Separation)** | Week 11 | **ดีกว่าในเคสที่สัญญาณผสมกันแบบทันทีทันใด (Instantaneous Linear Mixture)** โดยไม่มีการหน่วงเวลาหรือการสะท้อนซ้ำ | **แย่กว่าในเคสห้องก้อง (Convolutive Mixture)**: เพราะเสียงในห้องสะท้อนมาหลายระลอก ($x[n-1], x[n-2], \dots$) ทำให้ PCA/ICA แยกเสียงก้องที่มีหางยาวไม่ออก |
| **5. Standard LMS ($N$ เล็ก vs $N$ ใหญ่, $\mu$ เล็ก vs $\mu$ ใหญ่)** | Week 1, Week 3, Week 4, Week 8 | **$N=128, 256$ ดีกว่าในห้องเล็ก (ลู่เข้าไว)** ส่วน **$N=512, 1024$ ดีกว่าในห้องโถงที่เสียงก้องค้างนาน** | **ปัญหาสำคัญของ LMS**: ถ้าตั้ง $\mu$ ใหญ่เกินขอบเขตเสถียรภาพ $\frac{1}{N \cdot P_x}$ หรือเจอจังหวะที่เสียงลำโพงดังพุ่งขึ้นกะทันหัน ตัวกรองจะ **ระเบิด (Diverge / Blows up)** ทันที |
| **6. Normalized LMS (NLMS) vs Standard LMS** | Week 3, Week 7, Week 8 | **NLMS ดีกว่า LMS อย่างชัดเจนในสัญญาณเสียงพูดจริง (`real_recordings`)** เพราะเสียงพูดมีพลังงานขึ้นลงตลอดเวลา NLMS หารด้วยกำลังงาน $\mathbf{x}^T\mathbf{x}$ จึงปรับตัวได้เร็วโดยไม่ระเบิด | **ปัญหาเมื่อเกิด Double-Talk (คนสองฝั่งพูดสวนกัน)**: หากทั้ง LMS และ NLMS ปรับค่าน้ำหนักตลอดเวลาโดยไม่หยุดพัก เสียงพูดของคนฝั่งใกล้ ($s[n]$) จะไปกวนการอัปเดตค่าน้ำหนัก ทำให้เสียงพูดตัวเองแหว่งหรือบิดเบี้ยว (ซึ่งแสดงให้เห็นว่าทำไมระบบระดับ Production อย่าง WebRTC จึงต้องมีกลไกตรวจจับ Double-Talk เพิ่มเติม) |

---

## ส่วนที่ 4: โครงสร้างไฟล์โปรเจกต์ กฎ GitHub และจุดที่ผิดบ่อย (Repo Structure, Git Guardrails & Sanity Checks)

### 4.0 โครงสร้าง Repository และกฎเหล็กห้ามอัปโหลด Dataset ขึ้น GitHub
- **ห้าม Push Dataset ขึ้น GitHub เด็ดขาด**: ชุดข้อมูล Microsoft AEC Challenge มีขนาดหลาย GB และติดเงื่อนไข License การวิจัย ดังนั้นต้องบล็อกผ่านไฟล์ [`.gitignore`](file:///C:/Users/khaok/69-1/DSP/FinalProject/.gitignore) (`data/`, `*.wav`, `*.mp3`, `*.m4a`) และให้เพื่อนในทีมดาวน์โหลดตามคู่มือใน [`README.md`](file:///C:/Users/khaok/69-1/DSP/FinalProject/README.md) หรือแชร์ผ่าน Kaggle Dataset แทน
- **โครงสร้างมาตรฐานของโปรเจกต์ (`DSP/FinalProject/`)**:
  ```text
  FinalProject/
  ├── src/
  │   ├── lms.py               # LMS Adaptive Filter + Delay Estimation (Cross-Correlation)
  │   ├── nlms.py              # NLMS + Comparative DSP Methods (STFT / PCA-ICA / FIR-IIR)
  │   ├── metrics.py           # ERLE (Sliding Window), SNR, PESQ, STOI
  │   └── baseline_webrtc.py   # WebRTC AEC Baseline + Internal Delay Alignment
  ├── notebooks/
  │   └── experiment.ipynb     # สมุดทดลองรันเปรียบเทียบและพล็อตกราฟ
  ├── results/
  │   └── figures/             # กราฟและตารางสรุปผลทั้งหมด
  ├── data/                    # [GIT-IGNORED] โฟลเดอร์เก็บ Dataset (ห้ามขึ้น GitHub)
  ├── .gitignore
  └── README.md
  ```

### 4.1 Phase 0: Mandatory Sanity Check (ต้องรันให้ผ่านก่อนแตะ Dataset จริง!)
- **การทดสอบ Sine-Wave Echo Cancellation**:
  - สร้างสัญญาณทดสอบสังเคราะห์เอง: ให้ `farend` $x[n]$ เป็นคลื่น Sine Wave (เช่น $440\text{ Hz}$) แล้วนำมาบวกเข้ากับสัญญาณไมค์ $d[n]$ โดยตรง (หรือผ่าน FIR ง่ายๆ 3-5 taps ที่ทราบค่าแน่นอน)
  - รันโค้ด LMS / NLMS ของเราเพื่อดูว่าตัวกรองสามารถลู่เข้าและลบคลื่น Sine Wave นั้นออกจนเกือบเป็นศูนย์ ($\text{ERLE} > 30\text{ dB}$) ได้จริงหรือไม่
  - **เกณฑ์ตัดสิน**: หากลบ Sine Wave พื้นฐานไม่ออก แสดงว่าการวาง Buffer $\mathbf{x}[n]$ หรือสมการอัปเดต $\mathbf{w}[n]$ มีบั๊ก ห้ามรันกับ Dataset จริงจนกว่าจะแก้ผ่าน

### 4.2 Phase 1 Pitfalls — Dataset & Preprocessing
1. **Sample Rate ไม่ตรงกัน**:
   - `farend.wav`, `mic.wav`, `nearend.wav` ต้องมี $f_s = 16,000\text{ Hz}$ เท่ากันทุกไฟล์ หากไม่ตรงกัน ตัวกรองจะลู่เข้าผิดทั้งหมด ต้องสั่ง `assert sr_far == sr_mic == sr_near == 16000` เสมอ
2. **Length ไม่เท่ากัน (Sample-by-Sample Mismatch)**:
   - ความยาวของสัญญาณต้องเท่ากันเป๊ะระดับ Sample (`len(farend) == len(mic) == len(nearend)`) หากคลาดเคลื่อนแม้แต่ 1 sample จะเกิดการ Misalign และทำให้ค่า ERLE/PESQ ตกฮวบ

### 4.3 Phase 2 Pitfalls — LMS / NLMS Implementation
1. **Delay ระหว่าง `farend` กับเสียงก้องใน `mic` (จุดที่สำคัญและผิดบ่อยที่สุด!)**:
   - เสียงจากลำโพงต้องเดินทางผ่านอากาศก่อนถึงไมค์ ทำให้เสียงก้องใน `mic` มาช้ากว่า `farend` เสมอ
   - หากไม่ชดเชย Delay นี้ก่อน ตัวกรองที่มีความยาว $N$ จำกัดจะพยายามจับคู่สัญญาณที่ไม่ตรงเวลากันและไม่ลู่เข้าเลย
   - **วิธีแก้บังคับ**: ประมาณค่า Delay Lag ด้วย **Cross-Correlation (`np.correlate`)** ก่อน แล้วทำการ **Shift สัญญาณ `farend`** ให้ตรงกับจุดเริ่มต้นของเสียงก้องใน `mic` ก่อนส่งเข้า LMS/NLMS
2. **การตั้งค่า $\mu$ (Step Size) ใหญ่เกินไป**:
   - หากค่าน้ำหนัก $\mathbf{w}[n]$ เกิดการ Diverge สัญญาณขาออกจะกลายเป็น Noise รุนแรงหรือค่า `NaN` / `Inf` ทันที
   - ต้องเริ่มไล่จาก $\mu$ เล็กๆ ก่อนเสมอ (เช่น $\mu = 0.001$) และใส่การเช็ก `np.isnan` / ขอบเขตเสถียรภาพ $\mu < \frac{1}{N \cdot P_x}$
3. **การจัดการ Buffer $\mathbf{x}(n)$ ผิดพลาด**:
   - LMS ต้องการเวกเตอร์อินพุตย้อนหลัง $N$ จุดในลำดับเวลาที่ถูกต้อง: $\mathbf{x}(n) = [x(n), x(n-1), \dots, x(n-N+1)]^T$
   - ระวังปัญหา Index ชนขอบ การสไลซ์อาเรย์กลับด้าน หรือการเลื่อน Buffer ผิดทิศทางซึ่งจะทำให้การอัปเดตค่าน้ำหนักผิดทั้งหมด

### 4.4 Phase 3 Pitfalls — การวัดผล (Evaluation Metrics)
1. **การวัดค่า ERLE ผิดช่วงเวลา**:
   - สมการ $\text{ERLE} = 10 \log_{10}(E[d^2(n)] / E[e^2(n)])$ จะวัดประสิทธิภาพการตัดเสียงก้องได้ถูกต้อง **เฉพาะในช่วงที่มีแต่เสียงก้อง (Far-end Single-Talk / ช่วงที่ `nearend` เงียบ)** เท่านั้น
   - ถ้านำช่วงที่มีคนฝั่งใกล้พูด (`nearend` มีเสียง) มาคำนวณรวมในสมการ ERLE ตรงๆ ตัวหาร $E[e^2(n)]$ จะมีพลังงานเสียงพูดปนอยู่ ทำให้ค่า ERLE ออกมาต่ำหรือบิดเบือนจากความเป็นจริง จึงต้องคัดเลือกเฉพาะ Segment/เฟรมที่ `nearend` เงียบในการวัด ERLE (หรือหากมี `nearend.wav` สามารถวัด Echo Attenuation เทียบกับส่วนประกอบเสียงก้องจริงได้)
2. **การใส่ตัวแปร Reference ใน PESQ / STOI ผิดตัว**:
   - ทั้ง `pesq(16000, ref, deg, 'wb')` และ `stoi(ref, deg, 16000)` ต้องใช้ **`nearend.wav` (เสียงพูดสะอาด) เป็น `ref`** เทียบกับ **`e(n)` (สัญญาณขาออกจากตัวกรอง) เป็น `deg`** (สำหรับเคสที่มีเสียงพูด nearend) ห้ามใส่สลับตำแหน่งหรือเทียบผิดคู่โดยเด็ดขาด

### 4.5 Phase 4 Pitfalls — การเปรียบเทียบกับ Baseline (WebRTC AEC)
1. **Internal Delay ของ WebRTC AEC**:
   - โมดูล WebRTC AEC จะมีการหน่วงเวลาภายใน (Internal Pipeline Delay) อัตโนมัติ ทำให้สัญญาณขาออกของ WebRTC เลื่อนเวลาไปจาก `nearend.wav` ต้นฉบับ
   - **วิธีแก้บังคับ**: ก่อนนำเอาต์พุตของ WebRTC AEC ไปคำนวณค่า ERLE / PESQ / STOI ต้องทำ Time Alignment (ด้วย Cross-Correlation เทียบกับ `nearend.wav` หรือ `mic.wav`) เพื่อชดเชย Internal Delay ก่อนเสมอ มิฉะนั้นคะแนนของ WebRTC จะออกมาต่ำกว่าความเป็นจริง

