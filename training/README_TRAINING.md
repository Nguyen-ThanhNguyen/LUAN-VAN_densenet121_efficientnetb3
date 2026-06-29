# Training Guide

Thư mục `training/` lưu notebook và script phục vụ huấn luyện mô hình cho **DR Diagnosis System v3**. Nội dung ở đây dành cho giai đoạn thực nghiệm trên Kaggle, không phải luồng inference khi chạy ứng dụng web.

## Mục Đích

- Huấn luyện mô hình phân loại bệnh võng mạc đái tháo đường 5 lớp từ ảnh đáy mắt.
- So sánh hai backbone **DenseNet121** và **EfficientNetB3**.
- Xử lý mất cân bằng dữ liệu bằng balanced batch hoặc class weight.
- Đánh giá mô hình bằng các chỉ số phù hợp hơn với bài toán y tế như QWK, Macro-F1, Macro Recall và AUC.
- Xuất model artifact `.h5` để đưa vào backend Flask.

## File Chính

| File | Vai trò |
|---|---|
| `kaggle_train_v3_IMBALANCE_DenseNet121_EfficientNetB3_No_ESRGAN.ipynb` | Notebook huấn luyện chính trên Kaggle |
| `kaggle_train_v3_IMBALANCE_DenseNet121_EfficientNetB3_No_ESRGAN.py` | Bản script Python export từ notebook |
| `README_TRAINING.md` | Tài liệu mô tả quy trình huấn luyện |

## Dataset

Notebook được thiết kế cho dataset **APTOS 2019 Blindness Detection**.

Cấu trúc dataset kỳ vọng trên Kaggle:

```text
/kaggle/input/aptos2019-blindness-detection/
|-- train.csv
`-- train_images/
```

Trong đó `train.csv` cần có các cột:

| Cột | Ý nghĩa |
|---|---|
| `id_code` | Tên ảnh |
| `diagnosis` | Nhãn mức độ DR từ 0 đến 4 |

## Nhãn Phân Loại

| Class | Nhãn |
|---:|---|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

## Cấu Hình Huấn Luyện v3

Các cấu hình chính trong `TrainConfig`:

| Tham số | Giá trị mặc định |
|---|---:|
| `img_size` | 320 |
| `batch_size` | 8 |
| `seed` | 42 |
| `val_split` | 0.15 |
| `test_split` | 0.15 |
| `epochs_head` | 8 |
| `epochs_finetune` | 30 |
| `head_lr` | 1e-3 |
| `finetune_lr` | 2e-5 |
| `dropout` | 0.45 |
| `unfreeze_last_n` | 180 |
| `imbalance_strategy` | `balanced_batch` |
| `monitor_metric` | `val_macro_f1` |
| `mixed_precision` | `True` |

## Pipeline Huấn Luyện

```text
APTOS 2019
-> đọc train.csv và train_images/
-> chia train/validation/test theo stratified split
-> crop viền đen ảnh fundus
-> CLAHE nhẹ trên không gian màu LAB
-> resize về 320 x 320
-> augmentation cho tập train
-> xử lý mất cân bằng bằng balanced batch
-> train DenseNet121
-> train EfficientNetB3
-> đánh giá từng model trên test set
-> tìm trọng số ensemble tốt nhất trên validation set
-> đánh giá ensemble trên test set
-> export artifacts cho backend
```

Phiên bản v3 **không sử dụng ESRGAN**.

## Mô Hình

Hai backbone được huấn luyện riêng:

- `DenseNet121`
- `EfficientNetB3`

Backend hiện dùng ensemble xác suất:

```text
P_final = 0.55 * P_EfficientNetB3 + 0.45 * P_DenseNet121
Predicted class = argmax(P_final)
```

Các trọng số này được lưu trong `backend/models/inference_config.json` và được sử dụng khi chạy inference.

## Xử Lý Mất Cân Bằng Dữ Liệu

APTOS 2019 có phân phối lớp không cân bằng, đặc biệt các lớp `Mild DR`, `Severe DR` và `Proliferative DR` thường ít hơn lớp `No DR`.

Notebook hỗ trợ:

- `balanced_batch`: lấy mẫu theo lớp trong mỗi batch để mô hình nhìn thấy nhiều hơn các lớp thiểu số.
- `class_weight`: dùng trọng số lớp khi train.
- `none`: không áp dụng chiến lược cân bằng.

Thiết lập mặc định của v3 là `balanced_batch` và không dùng đồng thời `class_weight`, nhằm tránh over-correct.

## Chỉ Số Đánh Giá

Các chỉ số được tính trong quá trình huấn luyện và đánh giá:

- Accuracy
- Quadratic Weighted Kappa (QWK)
- Macro-F1
- Macro Precision
- Macro Recall
- AUC macro one-vs-rest
- Mean Absolute Error cho severity score
- Confusion matrix
- Classification report

Với bài toán AI y tế, cần ưu tiên xem QWK, Macro-F1, Macro Recall và recall từng lớp thay vì chỉ dựa vào Accuracy.

## Kết Quả Thực Nghiệm Được Chọn

Phiên bản được chọn cho backend hiện tại:

```text
v3 final - ensemble argmax
EfficientNetB3 weight = 0.55
DenseNet121 weight = 0.45
Input size = 320 x 320
Preprocessing = crop black border + CLAHE LAB + resize
ESRGAN = False
```

Kết quả tổng hợp đang được ghi trong README chính của repo:

| Metric | Giá trị |
|---|---:|
| Accuracy | 84.73% |
| QWK | 0.9042 |
| Macro-F1 | 0.7066 |
| AUC macro | 0.9522 |

Bản thử nghiệm tập trung riêng vào class `Severe DR` đã được cân nhắc nhưng không chọn làm bản chính vì làm giảm accuracy, QWK và Macro-F1 tổng thể.

## Output Sau Khi Train

Notebook tạo các thư mục output mặc định:

```text
/kaggle/working/training_outputs/
/kaggle/working/backend_artifacts/
```

Các artifact quan trọng:

| Artifact | Mục đích |
|---|---|
| `densenet121_best.h5` | Model DenseNet121 tốt nhất |
| `efficientnetb3_best.h5` | Model EfficientNetB3 tốt nhất |
| `ensemble_results.json` | Kết quả ensemble và trọng số tốt nhất |
| `threshold_results.json` | Kết quả thử nghiệm threshold nếu có |
| `model_metadata.json` | Metadata phục vụ backend và báo cáo |
| `backend_model_artifacts.zip` | Gói model artifact để tải từ Kaggle |
| `training_outputs_bundle.zip` | Gói toàn bộ kết quả training |

## Đưa Model Vào Backend

Sau khi train xong, copy hai file model vào:

```text
backend/models/
|-- densenet121_best.h5
`-- efficientnetb3_best.h5
```

Đảm bảo file cấu hình inference tồn tại:

```text
backend/models/inference_config.json
```

Backend sẽ đọc hai model này khi gọi `/api/health` hoặc `/api/predict`.

## Cách Chạy Trên Kaggle

1. Tạo Kaggle Notebook mới.
2. Add dataset **APTOS 2019 Blindness Detection** vào notebook.
3. Upload hoặc mở file notebook trong thư mục `training/`.
4. Bật GPU trong phần Accelerator.
5. Chạy lần lượt các cell từ đầu đến cuối.
6. Tải `backend_model_artifacts.zip` và `training_outputs_bundle.zip` từ Kaggle Output.
7. Copy model `.h5` vào `backend/models/` của repo.

## Lưu Ý Tái Lập Thực Nghiệm

- Giữ `seed = 42` để giảm biến động khi chia tập và huấn luyện.
- Kết quả vẫn có thể dao động do GPU, phiên bản TensorFlow/Keras và quá trình augmentation.
- Nếu Kaggle không tải được ImageNet pretrained weights, notebook có cơ chế tìm offline weights hoặc fallback sang random init.
- Không đưa raw dataset APTOS vào GitHub repo.
- Không dùng kết quả training như bằng chứng triển khai lâm sàng nếu chưa có external validation.

## Giới Hạn

- Dataset có mất cân bằng lớp.
- Chưa có nhãn phân đoạn tổn thương.
- Class `Severe DR` là vùng khó và cần được theo dõi riêng khi đánh giá.
- Chưa kiểm chứng đầy đủ trên dữ liệu bệnh viện thực tế tại Việt Nam.
- Notebook phục vụ nghiên cứu và khóa luận, chưa phải pipeline MLOps hoàn chỉnh.
