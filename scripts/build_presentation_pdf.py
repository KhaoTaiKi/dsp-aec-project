import base64
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(r"C:\Users\khaok\69-1\DSP\FinalProject")
ASSETS_DIR = PROJECT_ROOT / "docs" / "presentation" / "assets"

# Helper to encode images to base64
def get_b64(img_name):
    p = ASSETS_DIR / img_name
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

img1 = get_b64("fig1_spectral_overlap.png")
img2 = get_b64("fig2_lag_detection.png")
img3 = get_b64("fig3_lab_benchmarks.png")
img4 = get_b64("fig4_s1_erle.png")
img5 = get_b64("fig5_s2_doubletalk.png")
img6 = get_b64("fig6_s3_moving.png")
img7 = get_b64("fig7_padasip_validation.png")

html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<title>DSP Final Project Presentation</title>
<style>
  @page {{
    size: 16in 9in;
    margin: 0;
  }}
  * {{
    box-sizing: border-box;
  }}
  body {{
    margin: 0;
    padding: 0;
    font-family: 'Leelawadee UI', 'Tahoma', 'Segoe UI', sans-serif;
    background-color: #0b0f19;
    color: #f1f5f9;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  .slide {{
    width: 16in;
    height: 9in;
    page-break-after: always;
    break-after: page;
    position: relative;
    overflow: hidden;
    padding: 0.55in 0.8in;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: radial-gradient(circle at 10% 20%, #172554 0%, #0b0f19 80%);
  }}
  
  /* Header styling */
  .slide-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 2px solid rgba(56, 189, 248, 0.25);
    padding-bottom: 12px;
  }}
  .slide-title-group h2 {{
    margin: 0;
    font-size: 32px;
    font-weight: 800;
    color: #38bdf8;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .slide-title-group p {{
    margin: 4px 0 0 0;
    font-size: 17px;
    color: #94a3b8;
  }}
  .badge-timer {{
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid #0284c7;
    color: #38bdf8;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 15px;
    font-weight: 700;
    white-space: nowrap;
  }}
  
  /* Body content */
  .slide-body {{
    flex: 1;
    display: flex;
    gap: 0.4in;
    align-items: center;
    margin: 0.25in 0;
  }}
  
  /* Footer styling */
  .slide-footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    padding-top: 10px;
    font-size: 14px;
    color: #64748b;
  }}
  .slide-footer .author {{
    color: #94a3b8;
  }}
  
  /* Layout helpers */
  .col-half {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }}
  .col-graph {{
    flex: 1.25;
    display: flex;
    justify-content: center;
    align-items: center;
  }}
  .col-graph img {{
    max-width: 100%;
    max-height: 4.8in;
    border-radius: 12px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.1);
    background: #0f172a;
  }}
  
  /* Cards & Callouts */
  .card {{
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
  }}
  .card-highlight {{
    background: rgba(14, 165, 233, 0.08);
    border-left: 4px solid #38bdf8;
  }}
  .card-success {{
    background: rgba(16, 185, 129, 0.08);
    border-left: 4px solid #10b981;
  }}
  .card-danger {{
    background: rgba(244, 63, 94, 0.08);
    border-left: 4px solid #f43f5e;
  }}
  
  /* Typography */
  h1 {{ margin: 0; }}
  h3 {{ margin: 0 0 8px 0; font-size: 20px; color: #f8fafc; font-weight: 700; }}
  p {{ margin: 0 0 8px 0; line-height: 1.5; font-size: 16px; color: #cbd5e1; }}
  ul {{ margin: 0 0 8px 0; padding-left: 24px; font-size: 16px; color: #cbd5e1; line-height: 1.6; }}
  li {{ margin-bottom: 6px; }}
  
  /* Key Stat Box */
  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 10px;
  }}
  .stat-box {{
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
  }}
  .stat-val {{
    font-size: 28px;
    font-weight: 900;
    color: #38bdf8;
  }}
  .stat-lbl {{
    font-size: 13px;
    color: #94a3b8;
    margin-top: 4px;
  }}
  
  /* Tables */
  table.pres-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 15px;
    margin-top: 8px;
  }}
  table.pres-table th {{
    background: rgba(14, 165, 233, 0.15);
    color: #38bdf8;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 2px solid rgba(56, 189, 248, 0.3);
  }}
  table.pres-table td {{
    padding: 9px 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    color: #e2e8f0;
  }}
  table.pres-table tr:hover {{
    background: rgba(255, 255, 255, 0.03);
  }}
  
  /* Formula Display */
  .math-box {{
    background: #090d16;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px 18px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 17px;
    color: #38bdf8;
    margin: 10px 0;
    display: flex;
    justify-content: center;
    align-items: center;
  }}

  /* Title Slide Specifics */
  .title-container {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    height: 100%;
    padding: 0 1in;
  }}
  .title-tag {{
    background: linear-gradient(90deg, #0284c7, #2563eb);
    color: white;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 20px;
    text-transform: uppercase;
  }}
  .main-title {{
    font-size: 46px;
    font-weight: 900;
    line-height: 1.25;
    background: linear-gradient(135deg, #ffffff 30%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 16px;
  }}
  .sub-title {{
    font-size: 22px;
    color: #94a3b8;
    margin-bottom: 35px;
    font-weight: 400;
  }}
  .meta-grid {{
    display: flex;
    gap: 40px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 20px 35px;
  }}
  .meta-item {{
    text-align: left;
  }}
  .meta-item .meta-label {{
    font-size: 13px;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
  }}
  .meta-item .meta-val {{
    font-size: 16px;
    color: #e2e8f0;
    font-weight: 600;
    margin-top: 4px;
  }}
</style>
</head>
<body>

<!-- ======================================================================= -->
<!-- SLIDE 1: COVER -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="title-container">
    <div class="title-tag">Final Project Presentation &bull; DSP 2569</div>
    <div class="main-title">การตัดเสียงสะท้อนอะคูสติกในห้อง<br>ด้วยตัวกรองปรับตัวได้ (AEC)</div>
    <div class="sub-title">Acoustic Echo Cancellation: LMS vs NLMS with Cross-Correlation Delay Compensation</div>
    
    <div class="meta-grid">
      <div class="meta-item">
        <div class="meta-label">รายวิชา</div>
        <div class="meta-val">517354 Digital Signal Processing</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">อาจารย์ผู้สอน</div>
        <div class="meta-val">อ.ดร.ณัฐพงศ์ จิวมั่งมี</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">สถาบัน</div>
        <div class="meta-val">ภาควิชาคอมพิวเตอร์ ม.ศิลปากร</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">เวลาที่กำหนด</div>
        <div class="meta-val" style="color: #38bdf8;">นำเสนอ 12 นาที | ตอบคำถาม 5 นาที</div>
      </div>
    </div>
  </div>
  <div class="slide-footer">
    <div class="author">กลุ่มผู้จัดทำโครงงาน: นักศึกษาภาควิชาคอมพิวเตอร์ มหาวิทยาลัยศิลปากร</div>
    <div>สไลด์ที่ 1 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 2: AGENDA & TIMELINE -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>🧭 โครงสร้างการนำเสนอ (Agenda & 12-Min Roadmap)</h2>
      <p>จัดสรรเวลาให้กระชับและตรงตามเกณฑ์การประเมิน Term Project (12 นาที)</p>
    </div>
    <div class="badge-timer">⏱️ 00:00 - 00:45</div>
  </div>
  
  <div class="slide-body">
    <div style="width: 100%;">
      <table class="pres-table">
        <thead>
          <tr>
            <th style="width: 15%;">ช่วงเวลา</th>
            <th style="width: 35%;">หัวข้อการนำเสนอ</th>
            <th style="width: 50%;">วัตถุประสงค์และผลลัพธ์สำคัญ</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong style="color: #38bdf8;">00:00 - 01:30</strong></td>
            <td><strong>1. ที่มาและปัญหาในชีวิตจริง</strong></td>
            <td>ทำไม Discord/Zoom ถึงสะท้อน? และจิตวิทยา Delayed Auditory Feedback (DAF)</td>
          </tr>
          <tr>
            <td><strong style="color: #38bdf8;">01:30 - 03:00</strong></td>
            <td><strong>2. EDA & ทำไมตัวกรองดั้งเดิมถึงสอบตก</strong></td>
            <td>พิสูจน์ด้วย FFT ว่าทำไม Butterworth Lowpass / Notch ถึงทำลายเนื้อเสียงพูด</td>
          </tr>
          <tr>
            <td><strong style="color: #38bdf8;">03:00 - 04:30</strong></td>
            <td><strong>3. การค้นพบทาง DSP: Bulk Delay</strong></td>
            <td>การใช้ Cross-Correlation ค้นหาความหน่วงเวลาเสียงสะท้อน (7.64 ms / 122 จุด)</td>
          </tr>
          <tr>
            <td><strong style="color: #38bdf8;">04:30 - 06:00</strong></td>
            <td><strong>4. สถาปัตยกรรมโมเดล LMS vs NLMS</strong></td>
            <td>พิมพ์เขียว DSP Pipeline 5 สเตจ และสมการคณิตศาสตร์ที่เขียนด้วย Pure NumPy</td>
          </tr>
          <tr>
            <td><strong style="color: #38bdf8;">06:00 - 08:00</strong></td>
            <td><strong>5. มวยเปรียบเทียบ 4 เสาหลัก (Benchmarks)</strong></td>
            <td>ประชันกับ 4 วิธีที่เรียนในแล็บ (W7-W12) และไลบรารีสากล <code>padasip</code></td>
          </tr>
          <tr>
            <td><strong style="color: #38bdf8;">08:00 - 10:30</strong></td>
            <td><strong>6. ผลการทดลอง 3 สถานการณ์จริง</strong></td>
            <td>S1 (คนเดียวนิ่ง ERLE 69.32 dB), S2 (Double-talk), S3 (คนเดินขยับตำแหน่ง)</td>
          </tr>
          <tr>
            <td><strong style="color: #38bdf8;">10:30 - 12:00</strong></td>
            <td><strong>7. สรุปผลสัมฤทธิ์ & ก้าวต่อไป</strong></td>
            <td>ข้อคิดเชิงวิศวกรรม Double-Talk Detector และการประยุกต์ใช้จริงในอุตสาหกรรม</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 2 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 3: REAL-WORLD PROBLEM & DAF -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>🗣️ 1. ที่มาของปัญหาในชีวิตจริง (The Acoustic Echo Loop)</h2>
      <p>ทำไมเวลาเราคุย Discord หรือ Zoom ถึงได้ยิน "เสียงสะท้อนของตัวเอง"?</p>
    </div>
    <div class="badge-timer">⏱️ 00:45 - 01:45</div>
  </div>
  
  <div class="slide-body">
    <div class="col-half">
      <div class="card card-highlight">
        <h3>📍 ความจริงเชิงอะคูสติกส์: ปัญหาเกิดที่เครื่องเพื่อน!</h3>
        <p>ปัญหาไม่ได้อยู่ที่เครื่องเรา แต่อยู่ที่ <strong>"เพื่อนเปิดลำโพงทิ้งไว้โดยไม่ใส่หูฟัง"</strong></p>
        <ul>
          <li>เราพูดใส่ไมค์ ➔ วิ่งผ่านเน็ตไปดังออก <strong>ลำโพงของเพื่อน</strong> (Far-End: <code>x[n]</code>)</li>
          <li>เสียงสะท้อนโต๊ะ กำแพง เพดานในห้องเพื่อน (<code>h[n]</code>) ➔ มุดเข้า <strong>ไมค์ของเพื่อน</strong> (<code>d[n]</code>)</li>
          <li>ไมค์เพื่อนดูดเสียงเราย้อนกลับมาเข้าหูเรา เกิดเป็น <strong>Acoustic Echo Loop</strong></li>
        </ul>
      </div>
      
      <div class="card card-danger">
        <h3>🧠 ปรากฏการณ์ Delayed Auditory Feedback (DAF)</h3>
        <p>ในทางจิตวิทยาการได้ยิน เมื่อมนุษย์ได้ยินเสียงตัวเองดีเลย์ย้อนกลับมาประมาณ <strong>50 - 200 ms</strong> สมองจะเกิดอาการสะดุด ลิ้นพันกัน และหยุดพูดโดยอัตโนมัติ <strong>AEC จึงเป็นระบบคอขาดบาดตายของแอปโทรศัพท์ทุกตัว!</strong></p>
      </div>
    </div>
    
    <div class="col-half">
      <div class="card" style="background: rgba(15, 23, 42, 0.9); border: 1px solid #0284c7; padding: 25px; text-align: center;">
        <h3 style="color: #38bdf8; margin-bottom: 20px;">แผนภาพการเดินทางของเสียง (Acoustic Echo Path)</h3>
        <div style="font-family: monospace; font-size: 15px; line-height: 1.8; text-align: left; background: #090d16; padding: 18px; border-radius: 8px;">
          [ตัวเรา (Far-End)] ────(ส่งผ่านเน็ต)────► [ลำโพงของเพื่อน: x[n]]<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▲&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼ (สะท้อนห้อง h[n])<br>
          (ได้ยินเสียงสะท้อน) ◄───(ส่งย้อนกลับ)─── [ไมค์เพื่อน: d[n] + v[n]]<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▲<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[เสียงเพื่อนพูด Near-End]
        </div>
        <p style="margin-top: 15px; color: #10b981; font-weight: bold; font-size: 15px;">
          🎯 เป้าหมายของโปรเจกต์: สร้างสมองกล DSP ประจำเครื่องเพื่อน เพื่อดักตัดเสียงสะท้อนทิ้ง 100% ก่อนส่งกลับ!
        </p>
      </div>
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 3 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 4: EDA & SPECTRAL OVERLAP PROOF -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>🔍 2. การสำรวจข้อมูล (EDA) & ทำไมตัวกรองดั้งเดิมถึงสอบตก?</h2>
      <p>การพิสูจน์ด้วย FFT: สัญญาณเสียงสะท้อนทับซ้อนกับเสียงพูด 100% (Spectral Overlap)</p>
    </div>
    <div class="badge-timer">⏱️ 01:45 - 03:00</div>
  </div>
  
  <div class="slide-body">
    <div class="col-half">
      <div class="card card-danger">
        <h3>🚫 ทำไม Classical LTI Filters ถึงใช้งานไม่ได้?</h3>
        <ul>
          <li>ทั้งเสียงพูดจริง (<code>x[n]</code>) และเสียงสะท้อน (<code>d[n]</code>) ต่างเป็น <strong>เสียงมนุษย์เหมือนกัน</strong></li>
          <li>แชร์ย่านความถี่ Formants เดียวกันเป๊ะ (<strong>100 Hz - 4,000 Hz</strong>)</li>
        </ul>
      </div>
      
      <div class="card">
        <h3>🧪 ผลการทดลองกรองด้วยฟิลเตอร์แบบดั้งเดิม:</h3>
        <p><strong>1. Lowpass Filter (&lt; 500 Hz):</strong> ตัดเสียงก้องสูงได้จริง แต่เสียงคนพูดกลายเป็น <em>"เสียงอู้อี้เหมือนคุยในโอ่ง"</em></p>
        <p><strong>2. Bandstop Filter (300-1500 Hz):</strong> เนื้อเสียงแหว่งเว้า บิดเบี้ยว แต่เสียงสะท้อนย่านอื่นยังคงอยู่ครบ</p>
      </div>
      
      <div class="card card-highlight">
        <p style="margin: 0; font-weight: bold; color: #38bdf8;">
          📌 บทสรุปเชิงทฤษฎี: ตัวกรองแบบเลือกความถี่คงที่ไม่สามารถแยกสัญญาณที่แชร์ย่านความถี่เดียวกันได้ ➔ จึงต้องใช้ Adaptive Filter!
        </p>
      </div>
    </div>
    
    <div class="col-graph">
      <img src="{img1}" alt="Spectral Overlap & Classical Filter Failure">
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 4 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 5: BULK DELAY & CROSS-CORRELATION -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>🎯 3. การค้นพบทาง DSP: การตรวจจับความหน่วงเวลา (Bulk Delay)</h2>
      <p>ค้นหาจุดเริ่มต้นของเสียงสะท้อนด้วย Cross-Correlation เพื่อทำ Delay Compensation</p>
    </div>
    <div class="badge-timer">⏱️ 03:00 - 04:30</div>
  </div>
  
  <div class="slide-body">
    <div class="col-half">
      <div class="card card-highlight">
        <h3>⏳ ปัญหา Time-of-Flight ในอากาศ</h3>
        <p>ลำโพงกับไมค์อยู่ห่างกัน คลื่นเสียงเดินทางด้วยความเร็ว 343 m/s ส่งผลให้เกิดความหน่วงเวลาเริ่มต้น (Bulk Delay: &tau;<sub>prop</sub>)</p>
        <p style="color: #f43f5e; font-weight: bold;">
          ⚠️ หากไม่ชดเชยดีเลย์: ตัวกรองปรับตัวได้จะต้องเสียค่าน้ำหนัก (Filter Taps) ไปกับช่วงความเงียบ ทำให้เหลือแท็ปไม่พอจำลองหางเสียงสะท้อน!
        </p>
      </div>
      
      <div class="card">
        <h3>🔍 การตรวจจับด้วยสมการ Cross-Correlation:</h3>
        <div class="math-box">r_dx[k] = &Sigma; d[n] &bull; x[n - k] &nbsp;&longrightarrow;&nbsp; k* = argmax |r_dx[k]|</div>
        <p>จากการทดสอบบนสัญญาณจริงในห้องจำลอง: ตรวจพบค่าความหน่วงเวลาแม่นยำที่ <strong>Lag = 122 จุด (7.64 ms)</strong></p>
        <p style="color: #10b981; font-weight: bold;">
          ✅ ทำ Delay Compensation เลื่อนบัฟเฟอร์ $x[n - k^*]$ ทำให้ตัวกรองเริ่มทำงาน ณ จุดเกิดเหตุทันที!
        </p>
      </div>
    </div>
    
    <div class="col-graph">
      <img src="{img2}" alt="Echo Lag Detection with Cross-Correlation">
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 5 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 6: DSP PIPELINE ARCHITECTURE -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>🏗️ 4. สถาปัตยกรรมระบบ (DSP Pipeline Blueprint 5 สเตจ)</h2>
      <p>โครงสร้างสถาปัตยกรรมตามพิมพ์เขียว Term Project ที่ได้รับคำแนะนำจากอาจารย์ผู้สอน</p>
    </div>
    <div class="badge-timer">⏱️ 04:30 - 05:30</div>
  </div>
  
  <div class="slide-body">
    <div style="width: 100%;">
      <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 25px;">
        <div class="card" style="border-top: 4px solid #38bdf8; text-align: center;">
          <div style="font-size: 28px; margin-bottom: 8px;">📥</div>
          <h3 style="font-size: 17px; color: #38bdf8;">Stage 1: Ingestion</h3>
          <p style="font-size: 14px;">รับสัญญาณเสียง $x[n]$ และ $d[n]$ ปรับ Sample Rate ที่ 16,000 Hz</p>
        </div>
        <div class="card" style="border-top: 4px solid #f59e0b; text-align: center;">
          <div style="font-size: 28px; margin-bottom: 8px;">⏱️</div>
          <h3 style="font-size: 17px; color: #f59e0b;">Stage 2: Lag Check</h3>
          <p style="font-size: 14px;">คำนวณ Cross-Correlation ตรวจหา Bulk Delay ($k^* = 122$ samples)</p>
        </div>
        <div class="card" style="border-top: 4px solid #8b5cf6; text-align: center;">
          <div style="font-size: 28px; margin-bottom: 8px;">🔄</div>
          <h3 style="font-size: 17px; color: #8b5cf6;">Stage 3: Delay Comp</h3>
          <p style="font-size: 14px;">หน่วงเวลาสัญญาณอ้างอิงล่วงหน้า $x[n - k^*]$ ให้ตรงตำแหน่งเสียงจริง</p>
        </div>
        <div class="card" style="border-top: 4px solid #10b981; text-align: center;">
          <div style="font-size: 28px; margin-bottom: 8px;">🧠</div>
          <h3 style="font-size: 17px; color: #10b981;">Stage 4: NLMS Engine</h3>
          <p style="font-size: 14px;">ประมาณค่าเสียงสะท้อน d_hat[n] และหักล้างออก e[n] = d[n] - d_hat[n]</p>
        </div>
        <div class="card" style="border-top: 4px solid #ec4899; text-align: center;">
          <div style="font-size: 28px; margin-bottom: 8px;">📊</div>
          <h3 style="font-size: 17px; color: #ec4899;">Stage 5: Evaluation</h3>
          <p style="font-size: 14px;">วัดค่า ERLE (dB), RMSE และส่งเสียงสะอาดปราศจากเสียงก้อง</p>
        </div>
      </div>
      
      <div class="card card-highlight" style="padding: 16px 24px;">
        <h3 style="margin-bottom: 6px;">💡 จุดเด่นของการออกแบบเชิงสถาปัตยกรรม (Responsibility-Based Architecture)</h3>
        <p style="margin: 0; font-size: 15px;">
          โมดูลในโฟลเดอร์ <code>src/</code> ถูกแบ่งตามหน้าที่รับผิดชอบ (<code>io/</code>, <code>simulation/</code>, <code>adaptive_filters/</code>, <code>metrics/</code>, <code>visualization/</code>) ทำให้โค้ดทั้ง 3 สถานการณ์ (S1, S2, S3) เรียกใช้ Engine ชุดเดียวกัน 100% ไม่มีโค้ดซ้ำซ้อน และพร้อมนำไปรันบนระบบ Embedded DSP ในชีวิตจริงได้ทันที
        </p>
      </div>
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 6 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 7: MATHEMATICAL CORE: LMS vs NLMS -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>🧠 4. หัวใจทางคณิตศาสตร์: Standard LMS vs Normalized LMS</h2>
      <p>พัฒนาขึ้นด้วย Pure NumPy จากสมการคณิตศาสตร์พื้นฐานโดยตรง</p>
    </div>
    <div class="badge-timer">⏱️ 05:30 - 06:30</div>
  </div>
  
  <div class="slide-body">
    <div class="col-half">
      <div class="card" style="border-left: 4px solid #0284c7;">
        <h3>1. Standard LMS (Least Mean Squares)</h3>
        <div class="math-box">w[n+1] = w[n] + &mu; &bull; e[n] &bull; x[n]</div>
        <ul>
          <li>ปรับค่าน้ำหนักตามความชันแบบ Stochastic Gradient Descent</li>
          <li><span style="color: #f43f5e; font-weight: bold;">จุดอ่อนวิกฤต:</span> ค่า &mu; เป็นค่าคงที่ ถ้าผู้พูดตะโกนเสียงดัง พลังงาน ||x[n]||&sup2; พุ่งสูง จะทำให้ตัวกรอง <strong>หลุดลอยและระเบิด (Diverge)</strong></li>
          <li>หากตั้ง &mu; เล็กเกินไป ตัวกรองจะลู่เข้าช้ามากจนกำจัดเสียงสะท้อนไม่ทัน</li>
        </ul>
      </div>
    </div>
    
    <div class="col-half">
      <div class="card card-success">
        <h3>2. Normalized LMS (NLMS - พระเอกของโครงงาน)</h3>
        <div class="math-box">w[n+1] = w[n] + [ &mu; / (||x[n]||&sup2; + &epsilon;) ] &bull; e[n] &bull; x[n]</div>
        <ul>
          <li><strong>ไดนามิกสเกลลิ่ง:</strong> ปรับอัตราการเรียนรู้ตามพลังงานรวมของสัญญาณอินพุต ณ ขณะนั้นแบบอัตโนมัติ</li>
          <li><strong>เสถียรภาพ 100%:</strong> เสียงจะเบาหรือดัง ตัวกรองก็ไม่หลุดลอย และลู่เข้าเร็วกว่ามาก</li>
          <li>ใส่ค่า &epsilon; = 10<sup>-4</sup> เพื่อป้องกันการหารด้วยศูนย์ในช่วงที่ผู้พูดเงียบเสียง</li>
        </ul>
      </div>
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 7 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 8: PILLAR 3 LAB BENCHMARKS -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>⚔️ 5. มวยเปรียบเทียบกับ 4 วิธีในห้องแล็บ (Pillar 3 Benchmark)</h2>
      <p>ทำไม 4 อัลกอริทึมที่เราได้เรียนในวิชา DSP (W4 - W12) ถึงแก้ปัญหาเสียงสะท้อนไม่ได้?</p>
    </div>
    <div class="badge-timer">⏱️ 06:30 - 08:00</div>
  </div>
  
  <div class="slide-body">
    <div class="col-half">
      <table class="pres-table" style="font-size: 14px;">
        <thead>
          <tr>
            <th>วิธีที่เรียนในแล็บ</th>
            <th>ข้อจำกัดทางฟิสิกส์ & สัญญาณ</th>
            <th>ผลลัพธ์</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Butterworth (W7, W9)</strong></td>
            <td>เกิด Spectral Overlap</td>
            <td><span style="color: #f43f5e; font-weight: bold;">❌ เสียงอู้อี้ในโอ่ง</span></td>
          </tr>
          <tr>
            <td><strong>Inverse Filter (W4, W8)</strong></td>
            <td>Zero ของห้องกลายเป็น Pole นอกวงกลม</td>
            <td><span style="color: #f43f5e; font-weight: bold;">❌ ตัวกรองระเบิด</span></td>
          </tr>
          <tr>
            <td><strong>FastICA (W11)</strong></td>
            <td>ICA รองรับเฉพาะ Instantaneous Mixture</td>
            <td><span style="color: #f43f5e; font-weight: bold;">❌ แยกไม่ออก</span></td>
          </tr>
          <tr>
            <td><strong>STFT Subtraction (W12)</strong></td>
            <td>หักลบเฟสไม่สัมพันธ์กัน</td>
            <td><span style="color: #f59e0b; font-weight: bold;">⚠️ Musical Noise</span></td>
          </tr>
          <tr style="background: rgba(16, 185, 129, 0.15);">
            <td><strong style="color: #10b981;">NLMS ของพวกเรา</strong></td>
            <td>ประมาณค่า $h[n]$ ด้วยสหสัมพันธ์</td>
            <td><strong style="color: #10b981;">🏆 เงียบสนิท 100%</strong></td>
          </tr>
        </tbody>
      </table>
      <div class="card card-success" style="margin-top: 15px; padding: 12px 16px;">
        <p style="margin: 0; font-size: 14.5px;">
          <strong>🎯 บทพิสูจน์ทางวิชาการ:</strong> การประมวลผลเสียงก้องสะท้อนในห้อง (Convolutive Reverb) มีเพียงตัวกรองปรับตัวได้ (Adaptive Filter) เท่านั้นที่สามารถประมาณค่าผลตอบสนองของห้องได้อย่างสมบูรณ์
        </p>
      </div>
    </div>
    
    <div class="col-graph">
      <img src="{img3}" alt="Lab Benchmarks Comparison">
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 8 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 9: SCENARIO 1 RESULTS (ERLE 69.32 dB) -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>🧪 6.1 สถานการณ์ที่ 1 (S1: คนเดียว ห้องคงที่)</h2>
      <p>เป้าหมาย: ตัดเสียงสะท้อนจากลำโพงออกจนสัญญาณเอาต์พุตเข้าใกล้ความเงียบสนิท</p>
    </div>
    <div class="badge-timer">⏱️ 08:00 - 09:00</div>
  </div>
  
  <div class="slide-body">
    <div class="col-half">
      <div class="card card-highlight">
        <h3>📊 การวัดผลด้วยค่า ERLE (Echo Return Loss Enhancement):</h3>
        <div class="math-box">ERLE = 10 &bull; log10( E[d&sup2;[n]] / E[e&sup2;[n]] ) &nbsp; (dB)</div>
        <p>วัดว่าตัวกรองสามารถลดทอนพลังงานเสียงสะท้อนลงได้กี่เดซิเบล (ยิ่งสูง ยิ่งดี)</p>
      </div>
      
      <div class="stat-grid">
        <div class="stat-box">
          <div class="stat-val" style="color: #94a3b8;">9.05 dB</div>
          <div class="stat-lbl">Standard LMS</div>
        </div>
        <div class="stat-box">
          <div class="stat-val" style="color: #f59e0b;">14.94 dB</div>
          <div class="stat-lbl">NLMS (ไม่ชดเชย Delay)</div>
        </div>
        <div class="stat-box" style="border-color: #10b981; background: rgba(16, 185, 129, 0.1);">
          <div class="stat-val" style="color: #10b981;">69.32 dB</div>
          <div class="stat-lbl">NLMS + Delay Comp 🏆</div>
        </div>
      </div>
      
      <div class="card card-success" style="margin-top: 15px;">
        <p style="margin: 0; font-size: 14.5px;">
          <strong>🚀 พลังของการชดเชย Delay:</strong> ค่า ERLE พุ่งขึ้นถึง <strong>+54.38 dB</strong> พิสูจน์ให้เห็นว่าการหา Lag ด้วย Cross-Correlation คือกุญแจสำคัญที่สุดในการทำ AEC!
        </p>
      </div>
    </div>
    
    <div class="col-graph">
      <img src="{img4}" alt="S1 ERLE & Waveform Results">
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 9 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 10: SCENARIO 2 (DOUBLE-TALK) -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>🗣️ 6.2 สถานการณ์ที่ 2 (S2: Double-talk 2 คนพูดพร้อมกัน)</h2>
      <p>เป้าหมาย: ตัดเสียงก้องสะท้อนจากลำโพงออก แต่ต้องคงเสียงคนพูดหน้าไมค์ไว้ 100%</p>
    </div>
    <div class="badge-timer">⏱️ 09:00 - 10:00</div>
  </div>
  
  <div class="slide-body">
    <div class="col-half">
      <div class="card card-highlight">
        <h3>🎙️ สัญญาณไมโครโฟนผสม (Double-talk Mixture)</h3>
        <div class="math-box">Mic Input: m[n] = d[n] (Echo) + v[n] (Near-End Voice)</div>
        <p>ในชีวิตจริง ขณะที่เพื่อนพูดออกจากลำโพง เราอาจจะพูดแทรกสวนกลับไปทันที ระบบจะต้องไม่ตัดเสียงเราทิ้งเด็ดขาด!</p>
      </div>
      
      <div class="card card-success">
        <h3>🌟 ผลลัพธ์เชิงประจักษ์:</h3>
        <ul>
          <li><strong>หักล้างเสียงสะท้อนหมดจด:</strong> เสียงลำโพง <code>d[n]</code> ถูกหักล้างจนเงียบกริบ</li>
          <li><strong>รักษาเสียงพูดหน้าไมค์สมบูรณ์:</strong> สัญญาณเอาต์พุต <code>e[n]</code> ทาบทับตรงกับเสียงต้นฉบับ <code>v[n]</code> เป๊ะ</li>
          <li>เนื้อเสียงคงรูป Formants ครบถ้วน ไม่บี้แบน ไม่เพี้ยน และไม่มีเสียงก้องตกค้าง</li>
        </ul>
      </div>
    </div>
    
    <div class="col-graph">
      <img src="{img5}" alt="S2 Double-Talk Results">
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 10 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 11: SCENARIO 3 (MOVING POSITION) -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>🚶‍♂️ 6.3 สถานการณ์ที่ 3 (S3: มีการขยับเปลี่ยนตำแหน่งในห้อง)</h2>
      <p>เป้าหมาย: ทดสอบความสามารถในการปรับตัวติดตามและลู่เข้าใหม่ (Tracking & Re-convergence)</p>
    </div>
    <div class="badge-timer">⏱️ 10:00 - 10:45</div>
  </div>
  
  <div class="slide-body">
    <div class="col-half">
      <div class="card card-highlight">
        <h3>🔄 การจำลอง Time-Varying Acoustic Environment</h3>
        <p>จำลองคนเดินขยับตำแหน่งในห้อง โดยผลตอบสนองของห้อง h[n] จะถูกสลับเปลี่ยนตำแหน่งกะทันหัน ณ กึ่งกลางเวลา (t = 2.5 s)</p>
      </div>
      
      <div class="card card-success">
        <h3>⚡ ผลการทดสอบการลู่เข้าใหม่ (Re-convergence):</h3>
        <ul>
          <li><strong>ณ วินาทีที่ 2.5:</strong> เกิดค่าความผิดพลาดดีดตัวขึ้นชั่วคราว ($e^2[n]$ พุ่งขึ้น) เนื่องจากสภาพห้องเปลี่ยนทันที</li>
          <li><strong>การปรับตัวฉับพลัน:</strong> อัลกอริทึม NLMS ใช้เวลาเพียง <strong>0.3 - 0.5 วินาที</strong> ในการคำนวณปรับค่าน้ำหนักตัวกรองใหม่และกลับมาลู่เข้าสู่จุดเงียบได้อีกครั้ง</li>
          <li>พิสูจน์ความทนทานต่อการขยับเปลี่ยนตำแหน่งของตัวกรอง NLMS ในโลกจริง</li>
        </ul>
      </div>
    </div>
    
    <div class="col-graph">
      <img src="{img6}" alt="S3 Moving Position Tracking Results">
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 11 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 12: PADASIP VALIDATION & PILLAR 4 -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>🏆 7. การพิสูจน์เทียบมาตรฐานสากล (Pillar 4 Validation)</h2>
      <p>เปรียบเทียบความถูกต้องของ Custom NLMS (Pure NumPy) กับไลบรารีสากล padasip</p>
    </div>
    <div class="badge-timer">⏱️ 10:45 - 11:30</div>
  </div>
  
  <div class="slide-body">
    <div class="col-half">
      <div class="card card-highlight">
        <h3>🏅 ไม้บรรทัดวัดมาตรฐานโลก (`padasip` Benchmark)</h3>
        <p>เพื่อพิสูจน์ให้อาจารย์เห็นว่า โค้ด NLMS ที่พวกเราเขียนมือเองด้วย Pure NumPy มีความถูกต้องทางคณิตศาสตร์ 100%</p>
      </div>
      
      <div class="stat-grid" style="grid-template-columns: repeat(2, 1fr);">
        <div class="stat-box" style="border-color: #10b981;">
          <div class="stat-val" style="color: #10b981;">100.000000 %</div>
          <div class="stat-lbl">Correlation Coefficient (r)</div>
        </div>
        <div class="stat-box" style="border-color: #38bdf8;">
          <div class="stat-val" style="color: #38bdf8;">1.36 &times; 10⁻¹⁸</div>
          <div class="stat-lbl">Mean Absolute Error (MAE)</div>
        </div>
      </div>
      
      <div class="card card-success" style="margin-top: 15px;">
        <p style="margin: 0; font-size: 14.5px;">
          <strong>🎯 การวิเคราะห์ความแม่นยำ:</strong> ค่าความคลาดเคลื่อน $10^{-18}$ ถือเป็น Machine Epsilon ของระบบตัวเลขทศนิยม 64 บิต (IEEE 754) จึงเป็นการพิสูจน์ทางคณิตศาสตร์อย่างเป็นทางการว่าโมเดลของเรามีความถูกต้องระดับสมบูรณ์แบบ!
        </p>
      </div>
    </div>
    
    <div class="col-graph">
      <img src="{img7}" alt="padasip Benchmark Validation">
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 12 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 13: CONCLUSION & FUTURE WORK -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>🌟 8. บทสรุปผลสัมฤทธิ์ & การต่อยอดในอนาคต (Conclusion)</h2>
      <p>สรุปสิ่งที่เราได้เรียนรู้และการนำเทคนิค DSP ไปประยุกต์ใช้ในอุตสาหกรรมจริง</p>
    </div>
    <div class="badge-timer">⏱️ 11:30 - 12:00</div>
  </div>
  
  <div class="slide-body">
    <div class="col-half">
      <div class="card card-success">
        <h3>🎉 สรุปผลสัมฤทธิ์หลัก 4 ประการ:</h3>
        <ul>
          <li><strong>พิสูจน์ข้อจำกัดฟิลเตอร์ดั้งเดิม:</strong> Butterworth / Notch สอบตกเพราะ Spectral Overlap</li>
          <li><strong>กุญแจสำคัญคือ Delay Compensation:</strong> ดันค่า ERLE พุ่งทะยานจาก 14.94 dB เป็น <strong>69.32 dB</strong></li>
          <li><strong>รับมือได้ทุกสถานการณ์:</strong> ทั้งคนเดียว, Double-talk และการเคลื่อนที่เปลี่ยนตำแหน่ง</li>
          <li><strong>ความแม่นยำมาตรฐานสากล:</strong> ตรงกับไลบรารีสากล <code>padasip</code> 100%</li>
        </ul>
      </div>
    </div>
    
    <div class="col-half">
      <div class="card card-highlight">
        <h3>🚀 ทิศทางการพัฒนาต่อยอดในอนาคต (Future Work):</h3>
        <ul>
          <li><strong>Double-Talk Detector (DTD):</strong> พัฒนา Geigel Algorithm ตรวจจับช่วงมีเสียงพูดแทรกเพื่อ <em>Freeze Weights</em> ป้องกันตัวกรอง Diverge เมื่อคนหน้าไมค์ตะโกนเสียงดัง</li>
          <li><strong>Subband AEC (Filter Bank):</strong> ซอยสัญญาณเป็นหลายแถบความถี่ก่อนกรอง ช่วยลดภาระการคำนวณลง 70% และรองรับห้องโถงขนาดใหญ่ (RT60 &gt; 500 ms)</li>
          <li><strong>Hardware Deployment:</strong> คอมไพล์โค้ด Pure NumPy สู่ C / Fixed-Point เพื่อฝังลงบนชิป DSP ของหูฟัง TWS</li>
        </ul>
      </div>
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 13 / 14</div>
  </div>
</div>

<!-- ======================================================================= -->
<!-- SLIDE 14: Q&A DEFENSE MASTER SHEET -->
<!-- ======================================================================= -->
<div class="slide">
  <div class="slide-header">
    <div class="slide-title-group">
      <h2>❓ ช่วงถาม-ตอบ 5 นาที (Q&A Defense Master Sheet)</h2>
      <p>คลังแนวทางการตอบคำถามอาจารย์ผู้สอน (เตรียมพร้อมรับมือทุกคำถามเชิงลึก)</p>
    </div>
    <div class="badge-timer">⏱️ 12:00 - 17:00 (Q&A 5 นาที)</div>
  </div>
  
  <div class="slide-body">
    <div style="width: 100%; display: flex; flex-direction: column; gap: 12px;">
      <div class="card card-highlight" style="padding: 12px 18px; margin: 0;">
        <h3 style="font-size: 16px; color: #38bdf8;">1. ถาม: ทำไมถึงไม่ใช้ <code>np.argsort()</code> หาความถี่เด่นของเสียงคนพูดจริง?</h3>
        <p style="margin: 0; font-size: 14.5px;">
          <strong>ตอบ:</strong> เสียงคนพูดไม่ใช่ Sine wave เดี่ยวๆ แต่มีแถบพลังงานกว้างตามสรีระคอหอย (Formants $F_1, F_2, F_3$) หากใช้ <code>argsort</code> จะได้เฉพาะ bin เสี้ยนสัญญาณที่อยู่ติดกัน วิธีที่ถูกต้องคือการทำ <strong>Smoothing (Savitzky-Golay)</strong> เพื่อลบเสี้ยนออก แล้วใช้ <strong>Dynamic Threshold (% of Max)</strong> ร่วมกับ <code>find_peaks</code> จึงจะได้ Formants ที่แท้จริงครับ
        </p>
      </div>
      
      <div class="card card-success" style="padding: 12px 18px; margin: 0;">
        <h3 style="font-size: 16px; color: #10b981;">2. ถาม: ทำไมการชดเชยดีเลย์ (Delay Compensation) ถึงดันค่า ERLE ขึ้นมหาศาล (+54 dB)?</h3>
        <p style="margin: 0; font-size: 14.5px;">
          <strong>ตอบ:</strong> เพราะระยะห่างระหว่างลำโพงกับไมค์ทำให้เกิด Bulk Delay (122 จุด) หากไม่ชดเชย ตัวกรองขนาด 128 แท็ปจะต้องเสีย 122 แท็ปแรกไปกับการรอสัญญาณ ทำให้เหลือแท็ปเพียง 6 จุดในการประมาณค่าหางเสียงสะท้อน แต่เมื่อชดเชยล่วงหน้า แท็ปทั้ง 128 จุดจะถูกนำมาใช้จำลองหางเสียงสะท้อนของห้องได้อย่างสมบูรณ์แบบครับ
        </p>
      </div>
      
      <div class="card card-danger" style="padding: 12px 18px; margin: 0;">
        <h3 style="font-size: 16px; color: #f43f5e;">3. ถาม: ในสภาวะ Double-talk ถ้าคนพูดหน้าไมค์เสียงดังมาก ตัวกรองจะเพี้ยนไหม?</h3>
        <p style="margin: 0; font-size: 14.5px;">
          <strong>ตอบ:</strong> ตัวกรองอาจพยายามเรียนรู้เสียงคนพูดหน้าไมค์เข้าไป ซึ่งในระบบอุตสาหกรรมจริงจะแก้ด้วยการติด <strong>Double-Talk Detector (DTD)</strong> เช่น Geigel Algorithm เพื่อคอยแช่แข็งค่าน้ำหนัก (Freeze Weights) ชั่วคราวเมื่อตรวจพบว่ามีคนพูดแทรกครับ
        </p>
      </div>
    </div>
  </div>
  
  <div class="slide-footer">
    <div class="author">517354 Digital Signal Processing &bull; Term Project</div>
    <div>สไลด์ที่ 14 / 14 &bull; จบการนำเสนอ ขอบคุณครับ</div>
  </div>
</div>

</body>
</html>
"""

html_path = PROJECT_ROOT / "docs" / "presentation" / "DSP_FINAL_PROJECT_PRESENTATION.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"HTML presentation written to {html_path}")

# Compile to PDF using headless Chrome
pdf_path = PROJECT_ROOT / "docs" / "presentation" / "DSP_FINAL_PROJECT_PRESENTATION.pdf"
chrome_cmd = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "--headless=new",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={pdf_path}",
    "--no-pdf-header-footer",
    str(html_path),
]

print("Compiling to PDF with headless Chrome...")
res = subprocess.run(chrome_cmd, capture_output=True, text=True)
print(res.stdout)
print(res.stderr)

if pdf_path.exists():
    print(f"SUCCESS: Created PDF at {pdf_path} (size: {pdf_path.stat().st_size} bytes)")
else:
    print("FAILED to create PDF")
