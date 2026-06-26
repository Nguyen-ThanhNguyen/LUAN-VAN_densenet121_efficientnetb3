---
title: DenseNet121 EfficientNetB3 DR Diagnosis
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# DR Diagnosis System v3

Ứng dụng web AI hỗ trợ phân loại 5 mức độ bệnh võng mạc đái tháo đường từ ảnh đáy mắt.

## Mô hình

- Ensemble argmax
- EfficientNetB3: 55%
- DenseNet121: 45%
- Tiền xử lý: crop viền đen, CLAHE LAB, resize 320x320
- Không dùng ESRGAN

## Lưu ý y tế

Hệ thống chỉ hỗ trợ học thuật, sàng lọc và tham khảo. Kết quả không thay thế chẩn đoán của bác sĩ chuyên khoa mắt.

## Cách chạy trên Hugging Face Spaces

Space dùng Docker. Flask app lắng nghe biến môi trường `PORT`, mặc định `7860`.
