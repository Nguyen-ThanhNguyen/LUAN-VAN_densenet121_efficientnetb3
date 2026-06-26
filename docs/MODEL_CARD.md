# Model Card — DR Diagnosis System v3

## Mục tiêu

Phân loại ảnh đáy mắt thành 5 mức độ bệnh võng mạc đái tháo đường.

| Class | Ý nghĩa |
|---|---|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

## Mô hình chính

```text
v3 ensemble argmax
EfficientNetB3 weight = 0.55
DenseNet121 weight = 0.45
```

## Kết quả thực nghiệm v3

| Metric | Giá trị |
|---|---:|
| Accuracy | 84.73% |
| QWK | 0.9042 |
| Macro-F1 | 0.7066 |
| AUC macro | 0.9522 |

## Tiền xử lý

- Crop viền đen
- CLAHE nhẹ trên LAB
- Resize 320×320
- Normalize theo backbone
- Không dùng ESRGAN

## Hạn chế

- Class 3 - Severe DR còn yếu.
- Chưa kiểm chứng external dataset.
- Chưa có module đánh giá chất lượng ảnh đầu vào.
- Không thay thế bác sĩ.
