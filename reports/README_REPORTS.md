# Reports Guide

Thư mục `reports/` dùng để lưu tài liệu tổng hợp, biểu đồ, bảng kết quả và nội dung phục vụ báo cáo khóa luận hoặc nghiên cứu. Đây không phải thư mục lưu báo cáo dự đoán runtime của ứng dụng web.

## Mục Đích

- Tập hợp kết quả thực nghiệm dùng trong báo cáo.
- Lưu biểu đồ, bảng metric, confusion matrix hoặc hình minh họa khi cần.
- Ghi chú các kết luận quan trọng sau khi huấn luyện và đánh giá mô hình.
- Tách biệt tài liệu báo cáo nghiên cứu với file sinh tự động khi người dùng upload ảnh.

## Phân Biệt Các Loại Báo Cáo

| Vị trí | Mục đích | Có sinh tự động không |
|---|---|---|
| `reports/` | Tài liệu tổng hợp cho luận văn/nghiên cứu | Không |
| `backend/static/reports/` | Báo cáo TXT cho từng lần dự đoán từ web app | Có |
| `baocao/` | File báo cáo khóa luận/NCKH dạng DOCX/PDF | Không |
| `training_outputs_bundle.zip` | Kết quả huấn luyện tải từ Kaggle | Có sau khi chạy notebook |

## Báo Cáo Dự Đoán Runtime

Khi người dùng upload ảnh qua API:

```text
POST /api/predict
```

Backend sẽ sinh báo cáo TXT tại:

```text
backend/static/reports/{uuid}_report.txt
```

Đường dẫn này được trả về trong response qua field:

```json
{
  "report_url": "/static/reports/{uuid}_report.txt"
}
```

File TXT được tạo bởi:

```text
backend/app/utils/reporting.py
```

## Nội Dung Báo Cáo Runtime

Mỗi báo cáo dự đoán gồm các phần chính:

- Thời gian phân tích.
- Tên file ảnh đầu vào.
- Phiên bản mô hình.
- Trọng số ensemble của DenseNet121 và EfficientNetB3.
- Lớp dự đoán cuối cùng.
- Confidence top-1.
- Uncertainty.
- Entropy.
- Expected severity score.
- Top 3 dự đoán.
- Xác suất ensemble của 5 lớp.
- Output riêng của DenseNet121.
- Output riêng của EfficientNetB3.
- Cảnh báo y tế nếu có.
- Ghi chú rằng hệ thống không thay thế chẩn đoán của bác sĩ.

## Kết Quả Nên Lưu Trong `reports/`

Các nội dung phù hợp để đưa vào thư mục này:

- Bảng metric tổng hợp của từng model.
- Bảng so sánh DenseNet121, EfficientNetB3 và ensemble.
- Confusion matrix trên test set.
- Classification report.
- Biểu đồ phân phối lớp của dataset.
- Biểu đồ training/validation loss và accuracy.
- Biểu đồ hoặc hình minh họa heatmap/Grad-CAM dùng trong báo cáo.
- Ghi chú phân tích lỗi, đặc biệt với class `Severe DR`.

## Kết Quả Thực Nghiệm Chính

Phiên bản đang được chọn cho hệ thống:

```text
v3 final - ensemble argmax
EfficientNetB3 weight = 0.55
DenseNet121 weight = 0.45
Input size = 320 x 320
Preprocessing = crop black border + CLAHE LAB + resize
ESRGAN = False
```

Metric tổng hợp:

| Metric | Giá trị |
|---|---:|
| Accuracy | 84.73% |
| QWK | 0.9042 |
| Macro-F1 | 0.7066 |
| AUC macro | 0.9522 |

## Quy Ước Lưu File

Nếu bổ sung file vào `reports/`, nên đặt tên rõ ràng theo nội dung:

```text
reports/
|-- README_REPORTS.md
|-- metrics_summary_v3.md
|-- confusion_matrix_v3.png
|-- model_comparison_v3.csv
`-- error_analysis_severe_dr.md
```

Không nên lưu:

- Ảnh bệnh nhân chưa ẩn danh.
- Raw dataset.
- File output runtime quá nhiều hoặc có thông tin nhạy cảm.
- Artifact model lớn, vì model đã nằm ở `backend/models/`.

## Quyền Riêng Tư

Báo cáo dự đoán có thể chứa tên file ảnh và kết quả phân tích. Khi dùng dữ liệu thật, cần:

- Ẩn danh tên file và thông tin bệnh nhân.
- Không đưa dữ liệu nhận diện cá nhân vào Git.
- Xóa các báo cáo runtime không cần thiết trước khi public repo.
- Chỉ đưa hình minh họa đã được phép sử dụng vào tài liệu khóa luận.

## Liên Kết Với Các Tài Liệu Khác

- `README.md`: mô tả tổng quan repo và cách chạy hệ thống.
- `training/README_TRAINING.md`: mô tả quy trình huấn luyện.
- `docs/MODEL_CARD.md`: mô tả model và giới hạn.
- `docs/DATASET_CARD.md`: mô tả dataset và thiên lệch.
- `docs/MEDICAL_DISCLAIMER.md`: cảnh báo y tế.

## Ghi Chú Y Tế

Các báo cáo trong repo chỉ phục vụ học thuật, nghiên cứu và demo. Kết quả AI không phải kết luận chẩn đoán cuối cùng và không thay thế đánh giá của bác sĩ chuyên khoa mắt.
