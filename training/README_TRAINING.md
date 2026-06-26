# Training Notes

Bản cuối đã chốt:

```text
v3 ensemble argmax
EfficientNetB3: 0.55
DenseNet121: 0.45
```

Pipeline train:

```text
APTOS 2019
→ crop viền đen
→ CLAHE nhẹ
→ resize 320×320
→ train DenseNet121
→ train EfficientNetB3
→ xử lý mất cân bằng bằng balanced batch
→ ensemble weighted average
→ argmax
```

Bản v4 Severe DR Focus đã được thử nghiệm nhưng không chọn làm bản chính vì giảm accuracy/QWK/macro-F1 tổng thể.
