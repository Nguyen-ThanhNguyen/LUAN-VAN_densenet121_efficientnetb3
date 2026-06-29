---
title: DenseNet121 EfficientNetB3 DR Diagnosis
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# DR Diagnosis System v3

Hệ thống web ứng dụng trí tuệ nhân tạo hỗ trợ phân loại 5 mức độ bệnh võng mạc đái tháo đường từ ảnh đáy mắt. Dự án sử dụng ensemble giữa hai mô hình học sâu **DenseNet121** và **EfficientNetB3**, kết hợp hiển thị xác suất dự đoán, độ tin cậy, cảnh báo bất định và heatmap giải thích trực quan.

> **Cảnh báo y tế:** Hệ thống chỉ phục vụ mục đích học thuật, nghiên cứu và hỗ trợ tham khảo. Kết quả từ AI không phải chẩn đoán y khoa cuối cùng, không thay thế bác sĩ chuyên khoa mắt và không được dùng độc lập để đưa ra quyết định điều trị.

## Thông Tin Đề Tài

| Mục | Thông tin |
|---|---|
| Tên dự án | DR Diagnosis System v3 |
| Bài toán | Phân loại bệnh võng mạc đái tháo đường từ ảnh đáy mắt |
| Lĩnh vực | AI y tế, thị giác máy tính, hỗ trợ sàng lọc bệnh mắt |
| Tác giả | Nguyễn Thành Nguyên |
| GVHD | ThS. Trần Văn Thiện |
| Demo | [Hugging Face Space](https://huggingface.co/spaces/thanhnguyen-nguyen/densenet121_efficientnetb3) |

## Mục Tiêu

Dự án được xây dựng nhằm hỗ trợ thử nghiệm mô hình AI trong bài toán phân loại bệnh võng mạc đái tháo đường (Diabetic Retinopathy - DR) trên ảnh fundus.

Các chức năng chính:

- Phân loại ảnh đáy mắt thành 5 mức độ DR.
- Trả về xác suất dự đoán cho từng lớp bệnh.
- Tính độ tin cậy, độ bất định và điểm mức độ bệnh kỳ vọng.
- Hiển thị ảnh gốc, ảnh sau tiền xử lý và heatmap hỗ trợ giải thích.
- Cảnh báo khi mô hình thiếu chắc chắn hoặc khi các mức bệnh gần nhau.
- Sinh báo cáo kết quả phục vụ lưu trữ và demo học thuật.

## Phạm Vi Sử Dụng

### Intended Use

- Demo khóa luận, nghiên cứu khoa học và thử nghiệm mô hình AI y tế.
- Hỗ trợ tham khảo trong bài toán sàng lọc ảnh đáy mắt.
- Minh họa cách mô hình chú ý đến các vùng ảnh thông qua heatmap.
- Phục vụ đánh giá mô hình trong môi trường học thuật.

### Not Intended For

- Không dùng như công cụ chẩn đoán độc lập.
- Không dùng để quyết định điều trị hoặc thay thế chỉ định của bác sĩ.
- Không dùng cho ảnh không phải ảnh đáy mắt, ảnh mờ, ảnh lệch vùng võng mạc hoặc ảnh đã chỉnh sửa mạnh.
- Không dùng trong môi trường lâm sàng nếu chưa được kiểm định y khoa, pháp lý và đánh giá độc lập.

## Nhãn Phân Loại

| Class | Nhãn | Ý nghĩa |
|---:|---|---|
| 0 | No DR | Không có dấu hiệu bệnh võng mạc đái tháo đường |
| 1 | Mild DR | Bệnh mức độ nhẹ |
| 2 | Moderate DR | Bệnh mức độ trung bình |
| 3 | Severe DR | Bệnh mức độ nặng |
| 4 | Proliferative DR | Bệnh tăng sinh |

## Kiến Trúc Mô Hình

Hệ thống sử dụng ensemble xác suất từ hai backbone CNN:

```text
P_final = 0.55 * P_EfficientNetB3 + 0.45 * P_DenseNet121
Predicted class = argmax(P_final)
```

| Thành phần | Cấu hình |
|---|---|
| DenseNet121 | `backend/models/densenet121_best.h5`, weight 0.45 |
| EfficientNetB3 | `backend/models/efficientnetb3_best.h5`, weight 0.55 |
| Input size | 320 x 320 |
| Prediction rule | Ensemble argmax |
| Explainability | Gradient saliency heatmap |
| ESRGAN | Không sử dụng |

## Pipeline Xử Lý Ảnh

1. Đọc ảnh đầu vào ở định dạng RGB.
2. Kiểm tra định dạng và kích thước file.
3. Crop viền đen quanh ảnh fundus.
4. Tăng tương phản nhẹ bằng CLAHE trên không gian màu LAB.
5. Resize ảnh về `320 x 320`.
6. Chuẩn hóa input theo từng backbone.
7. Dự đoán bằng DenseNet121 và EfficientNetB3.
8. Trộn xác suất theo trọng số ensemble.
9. Sinh heatmap và báo cáo kết quả.

Định dạng ảnh hỗ trợ:

- PNG
- JPG
- JPEG

Kích thước file tối đa mặc định: `10MB`.

## Kết Quả Thực Nghiệm

Kết quả thực nghiệm phiên bản v3:

| Metric | Giá trị |
|---|---:|
| Accuracy | 84.73% |
| Quadratic Weighted Kappa | 0.9042 |
| Macro-F1 | 0.7066 |
| AUC macro | 0.9522 |

Trong bài toán y tế, accuracy không nên được xem là chỉ số duy nhất, đặc biệt khi dữ liệu có thể mất cân bằng lớp. Cần xem thêm macro-F1, recall từng lớp, confusion matrix và kết quả kiểm chứng trên external dataset trước khi ứng dụng thực tế.

## Output Hệ Thống

Sau khi upload ảnh, hệ thống trả về:

- Lớp dự đoán cuối cùng.
- Confidence top-1.
- Uncertainty = `1 - confidence`.
- Entropy của phân phối xác suất.
- Expected severity score trong khoảng 0 đến 4.
- Xác suất ensemble của 5 lớp.
- Xác suất riêng từ DenseNet121 và EfficientNetB3.
- Cảnh báo y tế khi mô hình bất định hoặc class 2/class 3 gần nhau.
- Ảnh gốc, ảnh sau tiền xử lý và heatmap.
- Link báo cáo kết quả.

## Giải Thích Heatmap

Heatmap thể hiện mức độ đóng góp tương đối của từng vùng ảnh vào dự đoán của mô hình.

| Vùng màu | Diễn giải |
|---|---|
| Xanh tím | Vùng ít ảnh hưởng đến dự đoán |
| Xanh lục/vàng | Vùng có tín hiệu trung bình, nên đối chiếu với ảnh gốc |
| Cam/đỏ/trắng | Vùng mô hình chú ý mạnh |

Heatmap chỉ là công cụ hỗ trợ trực quan, không phải bản đồ phân đoạn tổn thương và không chứng minh chắc chắn nguyên nhân y khoa của dự đoán.

## Cảnh Báo Và Cơ Chế An Toàn

Hệ thống có thể đưa ra cảnh báo khi:

- Confidence thấp hoặc uncertainty cao.
- Xác suất giữa class 2 và class 3 gần nhau.
- Mô hình dự đoán Moderate DR nhưng Severe DR cũng có xác suất đáng chú ý.
- Mô hình dự đoán Severe DR nhưng chưa tách biệt rõ với Moderate DR.

Khi xuất hiện các cảnh báo trên, kết quả cần được kiểm tra lại bởi chuyên gia y tế.

## Giới Hạn

- Class 3 - Severe DR còn là điểm yếu trong thực nghiệm.
- Dữ liệu huấn luyện có thể mất cân bằng giữa các lớp.
- Chưa có module đánh giá chất lượng ảnh đầu vào một cách đầy đủ.
- Chưa phân đoạn tổn thương y khoa.
- Chưa kiểm chứng đầy đủ trên external dataset hoặc dữ liệu bệnh viện thực tế tại Việt Nam.
- Có thể nhạy với ảnh mờ, thiếu sáng, lệch vùng võng mạc hoặc có artifact.
- Không phù hợp để dùng làm hệ thống hỗ trợ quyết định lâm sàng nếu chưa qua kiểm định độc lập.

## Dữ Liệu Và Thiên Lệch

Dự án tham chiếu bài toán **APTOS 2019 Blindness Detection** cho phân loại DR 5 lớp.

Các rủi ro dữ liệu cần lưu ý:

- Phân phối lớp không cân bằng.
- Ảnh từ một nguồn dữ liệu có thể không đại diện cho nhiều loại thiết bị chụp, dân số bệnh nhân hoặc quy trình bệnh viện khác nhau.
- Nhãn mức độ bệnh có thể có sai khác giữa các chuyên gia.
- Hiệu năng thực tế có thể giảm khi triển khai trên dữ liệu ngoài miền huấn luyện.

## Quyền Riêng Tư Và Dữ Liệu Người Dùng

Khi chạy demo, ảnh upload được lưu tạm trong thư mục static của ứng dụng để hiển thị kết quả và sinh báo cáo. Không nên upload ảnh chứa thông tin định danh bệnh nhân nếu chưa được ẩn danh.

Khuyến nghị khi triển khai thực tế:

- Ẩn danh dữ liệu trước khi upload.
- Thiết lập cơ chế tự động xóa file upload, heatmap và report sau một khoảng thời gian.
- Không lưu thông tin cá nhân nếu không cần thiết.
- Tuân thủ quy định bảo mật dữ liệu y tế tại nơi triển khai.

## API

### `GET /api/health`

Kiểm tra trạng thái hệ thống và trạng thái load model.

### `GET /api/class-names`

Trả danh sách nhãn phân loại.

### `POST /api/predict`

Upload ảnh fundus và nhận kết quả dự đoán.

Request:

```text
multipart/form-data
field: file
```

Response mẫu:

```json
{
  "predicted_class": 2,
  "predicted_class_name": "Moderate DR",
  "confidence": 0.82,
  "uncertainty": 0.18,
  "entropy": 0.5,
  "expected_severity_score": 2.1,
  "probabilities": [0.01, 0.04, 0.82, 0.10, 0.03],
  "model_outputs": {
    "densenet121": [],
    "efficientnetb3": []
  },
  "warnings": [],
  "original_image_url": "",
  "processed_image_url": "",
  "heatmap_url": "",
  "report_url": ""
}
```

## Cấu Trúc Dự Án

```text
DR_Diagnosis_System_v3/
├── app.py                         # Entry point tương thích Hugging Face Spaces/Docker
├── run.py                         # Entry point chạy Flask app ở local
├── Dockerfile                     # Cấu hình build container triển khai
├── README.md                      # Tài liệu tổng quan của repo
├── backend/                       # Backend Flask, inference model và API
│   ├── app/                       # Source code chính của backend
│   │   ├── api/                   # Định nghĩa API routes: health, class-names, predict
│   │   ├── core/                  # Cấu hình đường dẫn, model, class names, ngưỡng cảnh báo
│   │   ├── services/              # Logic load model, predict, ensemble và tính metric output
│   │   └── utils/                 # Tiện ích xử lý ảnh, heatmap, report và validate upload
│   ├── models/                    # Nơi lưu model weights và cấu hình inference
│   │   ├── densenet121_best.h5    # Model DenseNet121 đã huấn luyện
│   │   ├── efficientnetb3_best.h5 # Model EfficientNetB3 đã huấn luyện
│   │   └── inference_config.json  # Cấu hình ensemble, image size, class names
│   ├── static/                    # File sinh khi chạy app: uploads, processed, heatmaps, reports
│   └── requirements.txt           # Dependencies Python cho backend
├── frontend/                      # Giao diện web upload ảnh và hiển thị kết quả
│   ├── index.html                 # Trang chính của ứng dụng
│   ├── css/                       # Style giao diện
│   └── js/                        # Logic frontend gọi API và render kết quả
├── docs/                          # Tài liệu kỹ thuật: API, model card, dataset card, disclaimer
├── reports/                       # Ghi chú/bảng biểu tổng hợp phục vụ báo cáo nghiên cứu
├── training/                      # Notebook và script huấn luyện model trên Kaggle
├── tests/                         # Test cơ bản cho cấu trúc và backend
└── baocao/                        # File báo cáo khóa luận/NCKH dạng DOCX/PDF
```

## Cài Đặt Và Chạy Local

Cài dependencies:

```bash
pip install -r backend/requirements.txt
```

Chạy ứng dụng:

```bash
python run.py
```

Mặc định ứng dụng chạy tại:

```text
http://127.0.0.1:7860
```

Có thể đổi port bằng biến môi trường:

```bash
set PORT=8000
python run.py
```

Hoặc:

```bash
set DR_PORT=8000
python run.py
```

## Triển Khai Bằng Docker

Build image:

```bash
docker build -t dr-diagnosis-system .
```

Chạy container:

```bash
docker run -p 7860:7860 dr-diagnosis-system
```

## Triển Khai Hugging Face Spaces

Dự án hỗ trợ triển khai trên Hugging Face Spaces bằng Docker.

Các file chính:

- `README.md`: metadata Space và mô tả dự án.
- `Dockerfile`: cấu hình build environment.
- `app.py` và `run.py`: entrypoint Flask.
- `backend/requirements.txt`: dependencies.
- `backend/models/*.h5`: model weights.

Demo:

```text
https://huggingface.co/spaces/thanhnguyen-nguyen/densenet121_efficientnetb3
```

## Checklist Trước Khi Demo

- [ ] Ứng dụng khởi động thành công.
- [ ] `/api/health` trả trạng thái model sẵn sàng.
- [ ] Upload ảnh fundus thành công.
- [ ] Hiển thị lớp dự đoán cuối cùng.
- [ ] Hiển thị xác suất 5 lớp.
- [ ] Hiển thị kết quả riêng của từng model.
- [ ] Hiển thị ảnh sau tiền xử lý.
- [ ] Hiển thị heatmap.
- [ ] Sinh báo cáo kết quả.
- [ ] Disclaimer y tế hiển thị rõ ràng.

## Tài Liệu Liên Quan

- [API Reference](docs/API_REFERENCE.md)
- [Dataset Card](docs/DATASET_CARD.md)
- [Model Card](docs/MODEL_CARD.md)
- [Medical Disclaimer](docs/MEDICAL_DISCLAIMER.md)
- [Limitations](docs/LIMITATIONS.md)
- [Validation Checklist](docs/VALIDATION_CHECKLIST.md)

## Trách Nhiệm Sử Dụng

Người dùng cần hiểu rằng mô hình AI có thể sai, đặc biệt trong các trường hợp ảnh chất lượng thấp, dữ liệu ngoài miền huấn luyện hoặc mức độ bệnh nằm gần ranh giới giữa các lớp. Mọi quyết định y khoa phải được thực hiện bởi nhân viên y tế có chuyên môn.

## Tác Giả

**Nguyễn Thành Nguyên**

Dự án được thực hiện dưới sự hướng dẫn của **GVHD: ThS. Trần Văn Thiện**.

Nội dung thuộc phạm vi học thuật, nghiên cứu và demo hệ thống AI hỗ trợ phân loại bệnh võng mạc đái tháo đường từ ảnh đáy mắt.
