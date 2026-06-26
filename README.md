---
title: DenseNet121 EfficientNetB3 DR Diagnosis
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# DR Diagnosis System v3

**DR Diagnosis System v3** là ứng dụng web AI hỗ trợ phân loại 5 mức độ bệnh võng mạc đái tháo đường từ ảnh đáy mắt. Hệ thống sử dụng ensemble giữa **DenseNet121** và **EfficientNetB3**, hiển thị xác suất từng lớp, độ tin cậy, cảnh báo bất định và heatmap hỗ trợ giải thích vùng ảnh mà mô hình chú ý.

Space demo: [thanhnguyen-nguyen/densenet121_efficientnetb3](https://huggingface.co/spaces/thanhnguyen-nguyen/densenet121_efficientnetb3)

> Hệ thống chỉ phục vụ mục đích học thuật, nghiên cứu và hỗ trợ tham khảo. Kết quả AI không phải chẩn đoán y khoa cuối cùng và không thay thế bác sĩ chuyên khoa mắt.

## 1. Mục Tiêu Hệ Thống

Ứng dụng được xây dựng để hỗ trợ quy trình thử nghiệm/sàng lọc ảnh fundus trong bài toán phát hiện bệnh võng mạc đái tháo đường.

Hệ thống hướng tới:

- Phân loại ảnh đáy mắt thành 5 mức độ DR.
- Cung cấp xác suất dự đoán của từng lớp.
- Cảnh báo khi mô hình thiếu chắc chắn hoặc nhầm lẫn giữa Moderate DR và Severe DR.
- Hiển thị ảnh gốc, ảnh sau tiền xử lý và heatmap giải thích.
- Sinh báo cáo TXT phục vụ lưu trữ/kèm kết quả thử nghiệm.

## 2. Phạm Vi Sử Dụng

### Intended Use

- Học thuật, demo luận văn, thử nghiệm mô hình AI y tế.
- Hỗ trợ tham khảo trong sàng lọc ảnh đáy mắt.
- Giải thích trực quan xu hướng chú ý của mô hình qua heatmap.

### Not Intended For

- Không dùng như công cụ chẩn đoán độc lập.
- Không dùng để quyết định điều trị.
- Không dùng thay thế khám mắt, soi đáy mắt hoặc đánh giá bởi bác sĩ chuyên khoa.
- Không dùng cho ảnh không phải fundus, ảnh mờ, ảnh lệch vùng võng mạc hoặc ảnh đã chỉnh sửa mạnh.

## 3. Phân Lớp Bệnh

| Class | Nhãn | Ý nghĩa |
|---:|---|---|
| 0 | No DR | Không có dấu hiệu DR |
| 1 | Mild DR | Bệnh võng mạc đái tháo đường nhẹ |
| 2 | Moderate DR | Bệnh mức trung bình |
| 3 | Severe DR | Bệnh mức nặng |
| 4 | Proliferative DR | Bệnh tăng sinh |

## 4. Kiến Trúc Mô Hình

Hệ thống sử dụng ensemble xác suất từ hai backbone CNN:

```text
P_final = 0.55 * P_EfficientNetB3 + 0.45 * P_DenseNet121
Predicted class = argmax(P_final)
```

| Thành phần | Cấu hình |
|---|---|
| DenseNet121 | Model `.h5`, weight ensemble 0.45 |
| EfficientNetB3 | Model `.h5`, weight ensemble 0.55 |
| Rule | Ensemble argmax |
| Input size | 320 x 320 |
| Heatmap | Gradient saliency heatmap |
| ESRGAN | Không sử dụng |

## 5. Tiền Xử Lý Ảnh

Pipeline tiền xử lý:

1. Đọc ảnh RGB.
2. Crop viền đen quanh ảnh fundus.
3. Tăng tương phản nhẹ bằng CLAHE trên không gian màu LAB.
4. Resize về `320 x 320`.
5. Chuẩn hóa input theo từng backbone.

Định dạng ảnh hỗ trợ:

- PNG
- JPG
- JPEG

Kích thước file tối đa mặc định: `10MB`.

## 6. Kết Quả Thực Nghiệm

Kết quả thực nghiệm phiên bản v3:

| Metric | Giá trị |
|---|---:|
| Accuracy | 84.73% |
| QWK | 0.9042 |
| Macro-F1 | 0.7066 |
| AUC macro | 0.9522 |

Lưu ý: Accuracy có thể bị ảnh hưởng bởi mất cân bằng lớp. Với bài toán y tế, cần xem thêm macro-F1, recall từng lớp, confusion matrix và kiểm chứng trên external dataset trước khi dùng trong bối cảnh thực tế.

## 7. Output Của Hệ Thống

Sau khi upload ảnh, hệ thống trả về:

- Lớp dự đoán cuối cùng.
- Confidence top-1.
- Uncertainty = `1 - confidence`.
- Expected severity score từ 0 đến 4.
- Xác suất ensemble của 5 lớp.
- Xác suất riêng từ DenseNet121 và EfficientNetB3.
- Cảnh báo y tế nếu mô hình bất định hoặc class 2/class 3 gần nhau.
- Ảnh gốc, ảnh sau tiền xử lý và heatmap.
- Link báo cáo TXT.

## 8. Giải Thích Heatmap

Heatmap thể hiện mức độ đóng góp tương đối của từng vùng ảnh vào dự đoán của mô hình.

| Màu | Ý nghĩa |
|---|---|
| Xanh tím | Vùng ít ảnh hưởng, mô hình ít chú ý |
| Xanh lục/vàng | Vùng có tín hiệu trung bình, nên đối chiếu với ảnh gốc |
| Cam/đỏ/trắng | Vùng mô hình chú ý mạnh, có thể liên quan tổn thương hoặc cấu trúc nổi bật |

Heatmap chỉ là công cụ hỗ trợ trực quan. Đây không phải bản đồ phân đoạn tổn thương và không chứng minh chắc chắn nguyên nhân y khoa của dự đoán.

## 9. Cảnh Báo Và Cơ Chế An Toàn

Hệ thống có các cảnh báo:

- Confidence thấp hoặc uncertainty cao.
- Xác suất class 2 và class 3 gần nhau.
- Mô hình dự đoán Moderate DR nhưng Severe DR cũng có xác suất đáng chú ý.
- Mô hình dự đoán Severe DR nhưng chưa tách biệt rõ với Moderate DR.

Trong các trường hợp trên, kết quả cần được kiểm tra lại bởi chuyên gia.

## 10. Giới Hạn Hệ Thống

- Class 3 - Severe DR còn là điểm yếu trong thực nghiệm.
- Dữ liệu huấn luyện có thể mất cân bằng giữa các lớp.
- Chưa có module kiểm tra chất lượng ảnh đầu vào.
- Chưa phân đoạn tổn thương y khoa.
- Chưa kiểm chứng đầy đủ trên external dataset hoặc dữ liệu bệnh viện thực tế tại Việt Nam.
- Có thể nhạy với ảnh mờ, ảnh thiếu sáng, ảnh lệch vùng võng mạc hoặc ảnh có artifact.
- Không phù hợp để dùng làm hệ thống hỗ trợ quyết định lâm sàng nếu chưa qua đánh giá y khoa, pháp lý và kiểm định độc lập.

## 11. Dữ Liệu Và Thiên Lệch

Dự án tham chiếu bài toán **APTOS 2019 Blindness Detection** cho phân loại 5 lớp DR.

Các rủi ro dữ liệu cần lưu ý:

- Phân phối lớp không cân bằng.
- Ảnh từ một nguồn dữ liệu có thể không đại diện cho nhiều loại thiết bị chụp, dân số bệnh nhân hoặc quy trình bệnh viện khác nhau.
- Nhãn mức độ bệnh có thể có sai khác giữa chuyên gia.
- Hiệu năng thực tế có thể giảm khi triển khai trên dữ liệu ngoài miền huấn luyện.

## 12. Quyền Riêng Tư Và Dữ Liệu Người Dùng

Khi chạy demo, ảnh upload được lưu tạm trong thư mục static của ứng dụng để hiển thị kết quả và sinh báo cáo. Không nên upload ảnh chứa thông tin định danh bệnh nhân nếu chưa được ẩn danh.

Khuyến nghị khi triển khai thực tế:

- Ẩn danh dữ liệu trước khi upload.
- Thiết lập cơ chế tự động xóa file upload/report sau một khoảng thời gian.
- Không lưu thông tin cá nhân nếu không cần thiết.
- Tuân thủ quy định bảo mật dữ liệu y tế tại nơi triển khai.

## 13. API

### `GET /api/health`

Kiểm tra trạng thái hệ thống và model.

### `GET /api/class-names`

Trả danh sách class.

### `POST /api/predict`

Upload ảnh và nhận kết quả dự đoán.

Request:

```text
multipart/form-data
field: file
```

Response chính:

```json
{
  "predicted_class": 2,
  "predicted_class_name": "Moderate DR - Bệnh trung bình",
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

## 14. Cấu Trúc Dự Án

```text
DR_Diagnosis_System_v3/
├── app.py
├── run.py
├── Dockerfile
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── services/
│   │   └── utils/
│   ├── models/
│   │   ├── densenet121_best.h5
│   │   ├── efficientnetb3_best.h5
│   │   └── inference_config.json
│   ├── static/
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── docs/
└── tests/
```

## 15. Chạy Local

Cài dependencies:

```bash
pip install -r backend/requirements.txt
```

Chạy ứng dụng:

```bash
python run.py
```

Mặc định app chạy ở:

```text
http://127.0.0.1:7860
```

Có thể đổi port:

```bash
set PORT=8000
python run.py
```

## 16. Triển Khai Hugging Face Spaces

Space dùng Docker.

Các file chính:

- `README.md`: metadata Space, mô tả hệ thống.
- `Dockerfile`: build environment.
- `app.py` / `run.py`: entrypoint Flask.
- `backend/requirements.txt`: dependencies.
- `backend/models/*.h5`: model weights, track bằng Git LFS.

Link Space:

```text
https://huggingface.co/spaces/thanhnguyen-nguyen/densenet121_efficientnetb3
```

## 17. Checklist Trước Khi Demo

- [ ] Space build thành công.
- [ ] `/api/health` trả trạng thái model sẵn sàng.
- [ ] Upload ảnh fundus thành công.
- [ ] Hiển thị kết quả dự đoán cuối.
- [ ] Hiển thị xác suất 5 lớp.
- [ ] Hiển thị kết quả từng model.
- [ ] Hiển thị ảnh sau tiền xử lý.
- [ ] Hiển thị heatmap.
- [ ] Sinh báo cáo TXT.
- [ ] Cảnh báo y tế hiển thị đúng khi class 2/class 3 gần nhau.
- [ ] Disclaimer y tế hiển thị rõ ràng.

## 18. Trách Nhiệm Sử Dụng

Người dùng cần hiểu rằng mô hình AI có thể sai, đặc biệt trong các trường hợp ảnh chất lượng thấp, dữ liệu ngoài miền huấn luyện hoặc mức độ bệnh nằm gần ranh giới giữa các lớp. Mọi quyết định y khoa phải được thực hiện bởi nhân viên y tế có chuyên môn.

## 19. Tác Giả

Dự án học thuật phục vụ luận văn và nghiên cứu ứng dụng AI trong hỗ trợ phân loại bệnh võng mạc đái tháo đường.
