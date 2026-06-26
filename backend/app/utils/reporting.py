from datetime import datetime
from pathlib import Path


def format_probs(probs, class_names):
    lines = []
    for i, p in enumerate(probs):
        lines.append(f"- class {i}: {p:.4f} - {class_names[i]}")
    return "\n".join(lines)


def write_prediction_report(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    warnings = result.get("warnings", [])
    warning_text = "\n".join([f"- {w}" for w in warnings]) if warnings else "- Không có cảnh báo đặc biệt."

    top3_text = "\n".join([
        f"- class {item['class_id']}: {item['probability']:.4f} - {item['class_name']}"
        for item in result.get("top3", [])
    ])

    class_names = result["class_names"]

    text = f'''
BÁO CÁO PHÂN TÍCH ẢNH ĐÁY MẮT
================================

Thời gian: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
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
'''.strip()

    output_path.write_text(text, encoding="utf-8")
