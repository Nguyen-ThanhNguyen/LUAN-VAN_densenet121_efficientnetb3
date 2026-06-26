# API Reference

## GET `/api/health`

Kiểm tra trạng thái hệ thống và model.

## GET `/api/class-names`

Trả danh sách class.

## POST `/api/predict`

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
