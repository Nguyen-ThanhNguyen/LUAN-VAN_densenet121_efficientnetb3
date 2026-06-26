# Dataset Card — APTOS 2019

## Dataset

APTOS 2019 Blindness Detection.

## Bài toán

Phân loại 5 lớp DR từ ảnh đáy mắt.

## Hạn chế

- Dữ liệu mất cân bằng.
- Class 1, class 3, class 4 ít hơn class 0 và class 2.
- Không có nhãn phân đoạn tổn thương.
- Chưa đại diện đầy đủ cho dữ liệu bệnh viện thực tế tại Việt Nam.

## Ảnh hưởng

Accuracy có thể bị ảnh hưởng bởi lớp chiếm đa số. Vì vậy cần báo cáo thêm:

- QWK
- Macro-F1
- Macro Recall
- Confusion Matrix
- Recall từng lớp
