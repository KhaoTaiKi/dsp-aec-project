import base64
import mistune
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

with open(PROJECT_ROOT / "docs" / "report" / "DSP_FINAL_PROJECT_REPORT.md", "r", encoding="utf-8") as f:
    md_text = f.read()

# Enhance markdown by inserting image figures at appropriate sections
md_text = md_text.replace(
    "### 4.2 ผลการทดสอบตัวกรอง LTI ในแล็บ",
    f"### 4.2 ผลการทดสอบตัวกรอง LTI ในแล็บ\n\n<div class='fig-container'><img src='{img1}' alt='Spectral Overlap'/><p class='fig-cap'>รูปที่ 1: การเปรียบเทียบสเปกตรัมความถี่ FFT และความเสียหายของรูปคลื่นเมื่อใช้ตัวกรอง LTI แบบดั้งเดิม</p></div>\n"
)

md_text = md_text.replace(
    "### 5.2 การหาค่า Lag อัตโนมัติด้วย Cross-Correlation",
    f"### 5.2 การหาค่า Lag อัตโนมัติด้วย Cross-Correlation\n\n<div class='fig-container'><img src='{img2}' alt='Lag Detection'/><p class='fig-cap'>รูปที่ 2: กราฟความสัมพันธ์ Cross-Correlation แสดงตำแหน่งจุดเกิดเสียงสะท้อน (Lag = 122 จุด / 7.64 ms)</p></div>\n"
)

md_text = md_text.replace(
    "## 8. ชุดการทดลอง 4 เสาหลัก (The 4 Pillars Benchmark Suite)",
    f"## 8. ชุดการทดลอง 4 เสาหลัก (The 4 Pillars Benchmark Suite)\n\n<div class='fig-container'><img src='{img3}' alt='Pillar 3 Lab Benchmarks'/><p class='fig-cap'>รูปที่ 3: การเปรียบเทียบรูปคลื่นสัญญาณเอาต์พุตของทั้ง 4 วิธีที่เรียนในห้องปฏิบัติการเทียบกับตัวกรอง NLMS ที่พัฒนาขึ้น</p></div>\n"
)

md_text = md_text.replace(
    "* **ผลการทดลองเชิงตัวเลข:**",
    f"* **ผลการทดลองเชิงตัวเลข:**\n\n<div class='fig-container'><img src='{img4}' alt='Scenario 1 Results'/><p class='fig-cap'>รูปที่ 4: รูปคลื่นสัญญาณเอาต์พุตและเส้นกราฟการลู่เข้า ERLE ของสถานการณ์ที่ 1 (S1)</p></div>\n"
)

md_text = md_text.replace(
    "* **ผลการทดลอง:**  \n  เมื่อนำสัญญาณเอาต์พุต $e[n]$ มาเปรียบเทียบกับสัญญาณต้นฉบับ $v[n]$:",
    f"<div class='fig-container'><img src='{img5}' alt='Scenario 2 Results'/><p class='fig-cap'>รูปที่ 5: การเปรียบเทียบสัญญาณในสภาวะ Double-talk (สถานการณ์ S2)</p></div>\n\n* **ผลการทดลอง:**  \n  เมื่อนำสัญญาณเอาต์พุต $e[n]$ มาเปรียบเทียบกับสัญญาณต้นฉบับ $v[n]$:"
)

md_text = md_text.replace(
    "* **ผลการทดลอง:**\n  - ในช่วง $0 - 2.5$ วินาทีแรก:",
    f"<div class='fig-container'><img src='{img6}' alt='Scenario 3 Results'/><p class='fig-cap'>รูปที่ 6: การปรับตัวติดตามและลู่เข้าใหม่ (Tracking & Re-convergence) ในสถานการณ์ S3</p></div>\n\n* **ผลการทดลอง:**\n  - ในช่วง $0 - 2.5$ วินาทีแรก:"
)

md_text = md_text.replace(
    "## 10. การเปรียบเทียบความถูกต้องกับมาตรฐานสากล (Validation with World-Standard Library)",
    f"## 10. การเปรียบเทียบความถูกต้องกับมาตรฐานสากล (Validation with World-Standard Library)\n\n<div class='fig-container'><img src='{img7}' alt='Padasip Validation'/><p class='fig-cap'>รูปที่ 7: การเปรียบเทียบสัญญาณเอาต์พุตระหว่าง Custom NLMS กับไลบรารีสากล padasip</p></div>\n"
)

html_body = mistune.html(md_text)

full_html = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<title>DSP Final Project Report</title>
<style>
  @page {{
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
    @bottom-center {{
      content: counter(page);
      font-size: 10pt;
      color: #64748b;
    }}
  }}
  body {{
    font-family: 'Leelawadee UI', 'Tahoma', 'Segoe UI', sans-serif;
    font-size: 11pt;
    line-height: 1.65;
    color: #1e293b;
    background: #fff;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    color: #0f172a;
    font-size: 20pt;
    border-bottom: 2px solid #0284c7;
    padding-bottom: 8px;
    margin-top: 24pt;
    margin-bottom: 12pt;
    page-break-after: avoid;
  }}
  h2 {{
    color: #0369a1;
    font-size: 15pt;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
    margin-top: 20pt;
    margin-bottom: 10pt;
    page-break-after: avoid;
  }}
  h3 {{
    color: #0f172a;
    font-size: 12pt;
    margin-top: 14pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
  }}
  p, li {{
    text-align: justify;
    margin-bottom: 8pt;
  }}
  ul, ol {{
    padding-left: 20pt;
  }}
  blockquote {{
    background: #f0f9ff;
    border-left: 4px solid #0284c7;
    margin: 12pt 0;
    padding: 10pt 14pt;
    font-size: 10.5pt;
    color: #0369a1;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14pt 0;
    font-size: 10pt;
    page-break-inside: avoid;
  }}
  th {{
    background: #0284c7;
    color: #fff;
    padding: 8pt 10pt;
    text-align: left;
    font-weight: 600;
  }}
  td {{
    padding: 7pt 10pt;
    border-bottom: 1px solid #e2e8f0;
  }}
  tr:nth-child(even) {{
    background: #f8fafc;
  }}
  code {{
    background: #f1f5f9;
    padding: 2pt 5pt;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
    font-size: 9.5pt;
    color: #0369a1;
  }}
  pre {{
    background: #0f172a;
    color: #f8fafc;
    padding: 12pt;
    border-radius: 8px;
    font-family: 'Consolas', monospace;
    font-size: 9.5pt;
    overflow-x: auto;
    page-break-inside: avoid;
  }}
  pre code {{
    background: transparent;
    color: inherit;
    padding: 0;
  }}
  .fig-container {{
    text-align: center;
    margin: 14pt 0;
    page-break-inside: avoid;
  }}
  .fig-container img {{
    max-width: 95%;
    height: auto;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }}
  .fig-cap {{
    font-size: 9.5pt;
    color: #64748b;
    margin-top: 6pt;
    font-weight: 600;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

report_html_path = PROJECT_ROOT / "docs" / "report" / "DSP_FINAL_PROJECT_REPORT.html"
with open(report_html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

report_pdf_path = PROJECT_ROOT / "docs" / "report" / "DSP_FINAL_PROJECT_REPORT.pdf"
chrome_cmd = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "--headless=new",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={report_pdf_path}",
    "--no-pdf-header-footer",
    str(report_html_path),
]

print("Compiling Report to PDF...")
subprocess.run(chrome_cmd, capture_output=True)

if report_pdf_path.exists():
    print(f"SUCCESS: Created Report PDF at {report_pdf_path} (size: {report_pdf_path.stat().st_size} bytes)")
