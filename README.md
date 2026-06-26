---
title: DenseNet121 EfficientNetB3 DR Diagnosis
emoji: 🩺
colorFrom: teal
colorTo: blue
sdk: docker
pinned: false
---

# DR Diagnosis System v3 Final — Medical AI Web Project

Hệ thống hỗ trợ phân loại **5 mức độ bệnh võng mạc đái tháo đường** từ ảnh đáy mắt.

Phiên bản cuối đã chốt:

```text
Model chính: Ensemble argmax
EfficientNetB3: 55%
DenseNet121: 45%

Preprocessing:
crop viền đen → CLAHE nhẹ → resize 320×320 → normalize theo từng backbone

Không dùng ESRGAN.
```

> ⚕️ Hệ thống chỉ phục vụ học thuật / hỗ trợ tham khảo, không thay thế chẩn đoán của bác sĩ chuyên khoa.

---

## 1. Kết quả mô hình v3

| Metric | Giá trị |
|---|---:|
| Accuracy | 84.73% |
| QWK | 0.9042 |
| Macro-F1 | 0.7066 |
| AUC macro | 0.9522 |

Công thức ensemble:

```text
P_final = 0.55 × P_EfficientNetB3 + 0.45 × P_DenseNet121
Predicted class = argmax(P_final)
```

---

## 2. Cấu trúc dự án

```text
DR_Diagnosis_System_v3_Final_With_Frontend/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   └── routes.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── services/
│   │   │   └── inference_service.py
│   │   └── utils/
│   │       ├── image_processing.py
│   │       ├── heatmap.py
│   │       ├── reporting.py
│   │       └── validation.py
│   ├── models/
│   │   ├── PLACE_MODELS_HERE.txt
│   │   └── inference_config.json
│   ├── static/
│   │   ├── uploads/
│   │   ├── processed/
│   │   ├── heatmaps/
│   │   └── reports/
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── docs/
│   ├── MODEL_CARD.md
│   ├── DATASET_CARD.md
│   ├── LIMITATIONS.md
│   ├── MEDICAL_DISCLAIMER.md
│   ├── VALIDATION_CHECKLIST.md
│   └── API_REFERENCE.md
├── training/
│   └── README_TRAINING.md
├── reports/
│   └── README_REPORTS.md
├── tests/
│   └── test_basic.py
├── run.py
├── .env.example
├── .gitignore
└── Dockerfile
```

---

## 3. Việc bạn cần làm

Copy 2 model đã train vào:

```text
backend/models/densenet121_best.h5
backend/models/efficientnetb3_best.h5
```

Nếu bạn có file `backend_model_artifacts.zip` từ Kaggle, giải nén rồi copy:

```text
densenet121_best.h5
efficientnetb3_best.h5
inference_config.json
```

vào thư mục:

```text
backend/models/
```

---

## 4. Cài thư viện

Khuyến nghị Python **3.10 hoặc 3.11**.

### Windows PowerShell

```powershell
cd DR_Diagnosis_System_v3_Final_With_Frontend

python -m venv venv
.\venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

### macOS / Linux

```bash
cd DR_Diagnosis_System_v3_Final_With_Frontend

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

---

## 5. Chạy dự án

```bash
python run.py
```

Mở trình duyệt:

```text
http://127.0.0.1:8000
```

Kiểm tra API:

```text
http://127.0.0.1:8000/api/health
```

---

## 6. Giao diện frontend

Frontend nằm riêng trong thư mục:

```text
frontend/
```

Backend Flask sẽ tự serve giao diện này tại `/`.

Frontend có:

- Upload ảnh đáy mắt
- Preview ảnh
- Gọi API `/api/predict`
- Hiển thị kết quả cuối
- Hiển thị xác suất 5 lớp
- Hiển thị xác suất từng mô hình
- Hiển thị uncertainty / entropy / expected severity score
- Hiển thị cảnh báo y tế class 2/class 3
- Hiển thị ảnh gốc, ảnh tiền xử lý, heatmap
- Mở báo cáo TXT

---

## 7. Lưu ý y tế

- Chỉ dùng ảnh đáy mắt rõ nét.
- Không dùng ảnh không phải fundus.
- Class 3 — Severe DR còn là điểm hạn chế của mô hình.
- Nếu xác suất class 2 và class 3 gần nhau, hệ thống sẽ cảnh báo.
- Hệ thống chỉ hỗ trợ tham khảo, không thay thế bác sĩ.

