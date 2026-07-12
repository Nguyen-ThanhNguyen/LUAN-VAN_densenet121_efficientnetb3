from datetime import datetime
from html import escape
from pathlib import Path


def _pct(value):
    return f"{value * 100:.2f}%"


def _fmt(value, digits=4):
    return f"{value:.{digits}f}"


def _class_name(class_names, class_id):
    return class_names.get(class_id) or class_names.get(str(class_id), "")


def _severity_label(class_id):
    labels = {
        0: "Bình thường",
        1: "Nhẹ",
        2: "Trung bình",
        3: "Nặng",
        4: "Tăng sinh",
    }
    return labels.get(class_id, "Cần xem lại")


def format_probs(probs, class_names):
    lines = []
    for i, p in enumerate(probs):
        lines.append(f"- class {i}: {p:.4f} - {_class_name(class_names, i)}")
    return "\n".join(lines)


def _prob_rows(probs, class_names, active_id=None):
    rows = []
    for i, p in enumerate(probs):
        active = " is-active" if i == active_id else ""
        rows.append(
            f"""
            <div class="prob-row{active}">
              <div class="prob-main">
                <strong>Class {i}</strong>
                <span>{escape(_class_name(class_names, i))}</span>
              </div>
              <div class="prob-track" aria-hidden="true">
                <span style="width:{max(0, min(100, p * 100)):.2f}%"></span>
              </div>
              <div class="prob-value">{_pct(p)}</div>
            </div>
            """
        )
    return "\n".join(rows)


def _top3_cards(items):
    cards = []
    for index, item in enumerate(items, start=1):
        cards.append(
            f"""
            <article class="rank-card">
              <span class="rank-number">#{index}</span>
              <div>
                <strong>Class {item["class_id"]}</strong>
                <p>{escape(item["class_name"])}</p>
              </div>
              <b>{_pct(item["probability"])}</b>
            </article>
            """
        )
    return "\n".join(cards)


def _warning_items(warnings):
    if not warnings:
        warnings = ["Không có cảnh báo đặc biệt."]
    return "\n".join(f"<li>{escape(w)}</li>" for w in warnings)


def _image_block(result):
    original_url = result.get("original_image_url", "")
    processed_url = result.get("processed_image_url", "")
    heatmap_url = result.get("heatmap_url", "")

    figures = []
    if original_url:
        figures.append(
            f"""
            <figure>
              <img src="{escape(original_url)}" alt="Ảnh gốc">
              <figcaption>Ảnh gốc</figcaption>
            </figure>
            """
        )
    if processed_url:
        figures.append(
            f"""
            <figure>
              <img src="{escape(processed_url)}" alt="Ảnh sau tiền xử lý">
              <figcaption>Ảnh sau tiền xử lý</figcaption>
            </figure>
            """
        )
    if heatmap_url:
        figures.append(
            f"""
            <figure>
              <img src="{escape(heatmap_url)}" alt="Heatmap giải thích">
              <figcaption>Heatmap giải thích</figcaption>
            </figure>
            """
        )

    if not figures:
        return ""

    return f"""
    <section class="report-section">
      <div class="section-heading">
        <span>03</span>
        <h2>Ảnh và heatmap</h2>
      </div>
      <div class="image-grid">
        {"".join(figures)}
      </div>
      <p class="caption-note">
        Heatmap chỉ hỗ trợ trực quan hóa vùng mô hình chú ý, không phải bản đồ phân đoạn tổn thương y khoa.
      </p>
    </section>
    """


def _build_html_report(result, generated_at, text_filename):
    class_names = result["class_names"]
    predicted_class = int(result["predicted_class"])
    confidence = float(result["confidence"])
    uncertainty = float(result["uncertainty"])
    entropy = float(result["entropy"])
    severity = float(result["expected_severity_score"])
    weights = result["ensemble_weights"]

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Báo cáo phân tích ảnh đáy mắt</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --paper: #ffffff;
      --ink: #18212f;
      --muted: #657084;
      --line: #dbe2ee;
      --blue: #2367d1;
      --green: #15803d;
      --amber: #b45309;
      --red: #b91c1c;
      --violet: #6d28d9;
      --soft-blue: #eaf1ff;
      --soft-green: #e8f7ef;
      --soft-amber: #fff3d8;
      --soft-red: #feecec;
      --shadow: 0 18px 50px rgba(27, 39, 66, 0.12);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(180deg, rgba(35, 103, 209, 0.10), rgba(21, 128, 61, 0.05) 36%, var(--bg) 64%);
      font-family: Arial, "Helvetica Neue", sans-serif;
      line-height: 1.55;
    }}

    .page {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}

    .toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 18px;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--ink);
      text-decoration: none;
    }}

    .brand-mark {{
      display: grid;
      place-items: center;
      width: 44px;
      height: 44px;
      border-radius: 8px;
      color: #fff;
      background: linear-gradient(135deg, var(--blue), var(--green));
      font-weight: 800;
    }}

    .brand strong, .brand span {{ display: block; }}
    .brand span {{ color: var(--muted); font-size: 13px; }}

    .actions {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }}

    .btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      padding: 0 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--ink);
      background: var(--paper);
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
    }}

    .btn.primary {{ color: #fff; border-color: var(--blue); background: var(--blue); }}

    .report-shell {{
      overflow: hidden;
      border: 1px solid rgba(219, 226, 238, 0.9);
      border-radius: 8px;
      background: var(--paper);
      box-shadow: var(--shadow);
    }}

    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 28px;
      padding: 34px;
      color: #fff;
      background:
        linear-gradient(135deg, rgba(24, 33, 47, 0.98), rgba(35, 103, 209, 0.90)),
        radial-gradient(circle at 88% 20%, rgba(255,255,255,0.18), transparent 28%);
    }}

    .eyebrow {{
      margin: 0 0 10px;
      color: #b9d5ff;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0;
      text-transform: uppercase;
    }}

    h1 {{
      margin: 0;
      max-width: 720px;
      font-size: 34px;
      line-height: 1.15;
    }}

    .hero-meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 24px;
    }}

    .meta-item {{
      padding: 12px;
      border: 1px solid rgba(255,255,255,0.18);
      border-radius: 8px;
      background: rgba(255,255,255,0.08);
    }}

    .meta-item span {{
      display: block;
      color: #c9d7eb;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}

    .meta-item strong {{
      display: block;
      margin-top: 4px;
      word-break: break-word;
    }}

    .result-card {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 250px;
      padding: 22px;
      border-radius: 8px;
      color: var(--ink);
      background: #fff;
    }}

    .result-card .label {{
      margin: 0;
      color: var(--muted);
      font-weight: 800;
      text-transform: uppercase;
      font-size: 12px;
    }}

    .result-card h2 {{
      margin: 10px 0 8px;
      font-size: 24px;
      line-height: 1.2;
    }}

    .severity-badge {{
      display: inline-flex;
      align-items: center;
      width: fit-content;
      min-height: 34px;
      padding: 0 12px;
      border-radius: 8px;
      color: #fff;
      background: var(--blue);
      font-weight: 800;
    }}

    .severity-0 {{ background: var(--green); }}
    .severity-1 {{ background: var(--amber); }}
    .severity-2 {{ background: #ea580c; }}
    .severity-3 {{ background: var(--red); }}
    .severity-4 {{ background: var(--violet); }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      padding: 24px 34px 0;
    }}

    .metric {{
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
    }}

    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }}

    .metric strong {{
      display: block;
      margin-top: 8px;
      font-size: 24px;
    }}

    .report-section {{
      padding: 30px 34px;
      border-top: 1px solid var(--line);
    }}

    .section-heading {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 18px;
    }}

    .section-heading span {{
      display: grid;
      place-items: center;
      width: 34px;
      height: 34px;
      border-radius: 8px;
      color: var(--blue);
      background: var(--soft-blue);
      font-weight: 900;
    }}

    .section-heading h2 {{ margin: 0; font-size: 22px; }}

    .top3-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}

    .rank-card {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      align-items: center;
      gap: 12px;
      min-height: 96px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}

    .rank-number {{
      display: grid;
      place-items: center;
      width: 38px;
      height: 38px;
      border-radius: 8px;
      color: var(--green);
      background: var(--soft-green);
      font-weight: 900;
    }}

    .rank-card p {{ margin: 4px 0 0; color: var(--muted); font-size: 13px; }}
    .rank-card b {{ color: var(--blue); font-size: 18px; }}

    .prob-list {{
      display: grid;
      gap: 10px;
    }}

    .prob-row {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(160px, 2fr) 74px;
      align-items: center;
      gap: 14px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}

    .prob-row.is-active {{
      border-color: rgba(35, 103, 209, 0.45);
      background: var(--soft-blue);
    }}

    .prob-main strong, .prob-main span {{ display: block; }}
    .prob-main span {{ color: var(--muted); font-size: 13px; }}

    .prob-track {{
      overflow: hidden;
      height: 12px;
      border-radius: 999px;
      background: #e8edf5;
    }}

    .prob-track span {{
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--green), var(--blue));
    }}

    .prob-value {{
      text-align: right;
      color: var(--ink);
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }}

    .model-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}

    .model-card {{
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
    }}

    .model-card h3 {{ margin: 0 0 14px; }}
    .model-card .prob-row {{ grid-template-columns: minmax(160px, 1fr) minmax(120px, 1.4fr) 70px; }}

    .image-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }}

    figure {{
      margin: 0;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
    }}

    figure img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #0f172a;
    }}

    figcaption {{
      padding: 10px 12px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }}

    .caption-note {{
      margin: 14px 0 0;
      color: var(--muted);
    }}

    .warning-panel {{
      border-color: rgba(180, 83, 9, 0.30);
      background: var(--soft-amber);
    }}

    .warning-panel ul {{
      margin: 0;
      padding-left: 22px;
    }}

    .warning-panel li + li {{ margin-top: 8px; }}

    .note {{
      display: grid;
      gap: 10px;
      padding: 18px;
      border-radius: 8px;
      color: #713f12;
      background: #fff8e5;
    }}

    .note strong {{ color: #5b350f; }}

    .footer {{
      padding: 18px 34px 30px;
      color: var(--muted);
      border-top: 1px solid var(--line);
      font-size: 13px;
    }}

    @media (max-width: 900px) {{
      .hero, .metrics, .top3-grid, .model-grid, .image-grid {{
        grid-template-columns: 1fr;
      }}

      .hero-meta {{ grid-template-columns: 1fr; }}
    }}

    @media (max-width: 640px) {{
      .page {{ width: min(100% - 20px, 1180px); padding-top: 16px; }}
      .toolbar {{ align-items: stretch; flex-direction: column; }}
      .actions {{ justify-content: stretch; }}
      .btn {{ flex: 1; }}
      .hero, .report-section, .metrics, .footer {{ padding-left: 18px; padding-right: 18px; }}
      h1 {{ font-size: 27px; }}
      .prob-row, .model-card .prob-row {{
        grid-template-columns: 1fr;
        gap: 8px;
      }}
      .prob-value {{ text-align: left; }}
    }}

    @media print {{
      body {{ background: #fff; }}
      .page {{ width: 100%; padding: 0; }}
      .toolbar {{ display: none; }}
      .report-shell {{ border: 0; box-shadow: none; }}
      .hero {{ color: var(--ink); background: #fff; border-bottom: 2px solid var(--line); }}
      .result-card {{ border: 1px solid var(--line); }}
      .report-section, .metrics {{ break-inside: avoid; }}
      .btn {{ display: none; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <div class="toolbar">
      <a class="brand" href="/">
        <span class="brand-mark">DR</span>
        <strong>DR Diagnosis</strong>
        <span>Medical AI v3</span>
      </a>
      <div class="actions">
        <button class="btn primary" type="button" onclick="window.print()">In / Lưu PDF</button>
        <a class="btn" href="./{escape(text_filename)}" download>Tải TXT</a>
        <a class="btn" href="/">Về trang chính</a>
      </div>
    </div>

    <article class="report-shell">
      <header class="hero">
        <div>
          <p class="eyebrow">Báo cáo phân tích ảnh đáy mắt</p>
          <h1>Kết quả sàng lọc bệnh võng mạc đái tháo đường</h1>
          <div class="hero-meta">
            <div class="meta-item">
              <span>Thời gian</span>
              <strong>{escape(generated_at)}</strong>
            </div>
            <div class="meta-item">
              <span>File ảnh</span>
              <strong>{escape(result.get("filename", ""))}</strong>
            </div>
            <div class="meta-item">
              <span>Mô hình</span>
              <strong>v3 final - Ensemble argmax</strong>
            </div>
            <div class="meta-item">
              <span>Trọng số</span>
              <strong>EfficientNetB3 {weights["efficientnetb3"]} | DenseNet121 {weights["densenet121"]}</strong>
            </div>
          </div>
        </div>

        <aside class="result-card">
          <div>
            <p class="label">Kết quả cuối</p>
            <h2>Class {predicted_class} - {escape(result["predicted_class_name"])}</h2>
          </div>
          <span class="severity-badge severity-{predicted_class}">{_severity_label(predicted_class)}</span>
        </aside>
      </header>

      <section class="metrics" aria-label="Chỉ số dự đoán">
        <div class="metric">
          <span>Confidence top-1</span>
          <strong>{_pct(confidence)}</strong>
        </div>
        <div class="metric">
          <span>Uncertainty</span>
          <strong>{_pct(uncertainty)}</strong>
        </div>
        <div class="metric">
          <span>Entropy</span>
          <strong>{_fmt(entropy)}</strong>
        </div>
        <div class="metric">
          <span>Expected severity</span>
          <strong>{_fmt(severity)}</strong>
        </div>
      </section>

      <section class="report-section">
        <div class="section-heading">
          <span>01</span>
          <h2>Xác suất ensemble</h2>
        </div>
        <div class="top3-grid">
          {_top3_cards(result.get("top3", []))}
        </div>
        <div class="prob-list">
          {_prob_rows(result["probabilities"], class_names, predicted_class)}
        </div>
      </section>

      <section class="report-section">
        <div class="section-heading">
          <span>02</span>
          <h2>Kết quả từng mô hình</h2>
        </div>
        <div class="model-grid">
          <section class="model-card">
            <h3>DenseNet121</h3>
            <div class="prob-list">
              {_prob_rows(result["model_outputs"]["densenet121"], class_names)}
            </div>
          </section>
          <section class="model-card">
            <h3>EfficientNetB3</h3>
            <div class="prob-list">
              {_prob_rows(result["model_outputs"]["efficientnetb3"], class_names)}
            </div>
          </section>
        </div>
      </section>

      {_image_block(result)}

      <section class="report-section warning-panel">
        <div class="section-heading">
          <span>!</span>
          <h2>Cảnh báo và ghi chú y tế</h2>
        </div>
        <ul>
          {_warning_items(result.get("warnings", []))}
        </ul>
        <div class="note">
          <strong>Kết quả chỉ mang tính hỗ trợ tham khảo, không thay thế chẩn đoán của bác sĩ chuyên khoa.</strong>
          <span>Nếu ảnh đầu vào không rõ nét, không đúng ảnh đáy mắt hoặc mô hình có độ bất định cao, cần kiểm tra lại.</span>
        </div>
      </section>

      <footer class="footer">
        DR Diagnosis System v3 - Academic Medical AI Project
      </footer>
    </article>
  </main>
</body>
</html>"""


def write_prediction_report(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    warnings = result.get("warnings", [])
    warning_text = "\n".join([f"- {w}" for w in warnings]) if warnings else "- Không có cảnh báo đặc biệt."

    top3_text = "\n".join(
        [
            f"- class {item['class_id']}: {item['probability']:.4f} - {item['class_name']}"
            for item in result.get("top3", [])
        ]
    )

    class_names = result["class_names"]

    text = f"""
BÁO CÁO PHÂN TÍCH ẢNH ĐÁY MẮT
================================

Thời gian: {generated_at}
File ảnh: {result.get("filename", "")}

PHIÊN BẢN MÔ HÌNH
-----------------
v3 final - Ensemble argmax
EfficientNetB3 weight: {result["ensemble_weights"]["efficientnetb3"]}
DenseNet121 weight: {result["ensemble_weights"]["densenet121"]}

KẾT QUẢ DỰ ĐOÁN
---------------
Kết quả cuối: class {result["predicted_class"]} - {result["predicted_class_name"]}
Confidence top-1: {result["confidence"]:.4f}
Uncertainty: {result["uncertainty"]:.4f}
Entropy: {result["entropy"]:.4f}
Expected severity score: {result["expected_severity_score"]:.4f}

TOP 3 DỰ ĐOÁN
-------------
{top3_text}

XÁC SUẤT ENSEMBLE 5 LỚP
-----------------------
{format_probs(result["probabilities"], class_names)}

DENSENET121 OUTPUT
------------------
{format_probs(result["model_outputs"]["densenet121"], class_names)}

EFFICIENTNETB3 OUTPUT
---------------------
{format_probs(result["model_outputs"]["efficientnetb3"], class_names)}

CẢNH BÁO
--------
{warning_text}

GHI CHÚ Y TẾ
------------
Kết quả chỉ mang tính chất hỗ trợ tham khảo, không thay thế chẩn đoán của bác sĩ chuyên khoa.
Nếu ảnh đầu vào không rõ nét, không đúng ảnh đáy mắt hoặc mô hình có độ bất định cao, cần kiểm tra lại.
Class 3 - Severe DR là lớp còn hạn chế trong thực nghiệm, nên cần cảnh báo khi kết quả nằm giữa Moderate DR và Severe DR.
""".strip()

    output_path.write_text(text, encoding="utf-8")

    html_path = output_path.with_suffix(".html")
    html = _build_html_report(result, generated_at, output_path.name)
    html_path.write_text(html, encoding="utf-8")
    return html_path
