import json

import numpy as np
from tensorflow import keras

from backend.app.core.config import (
    CLASS_NAMES,
    DENSENET_MODEL_PATH,
    EFFICIENTNET_MODEL_PATH,
    HIGH_UNCERTAINTY,
    IMG_SIZE,
    INFERENCE_CONFIG_PATH,
    MODERATE_SEVERE_MARGIN,
    SHORT_CLASS_NAMES,
    WARN_P2_WHEN_CLASS3,
    WARN_P3_WHEN_CLASS2,
    WEIGHT_DENSENET121,
    WEIGHT_EFFICIENTNETB3,
)
from backend.app.utils.heatmap import gradient_saliency_heatmap
from backend.app.utils.image_processing import (
    prepare_model_input,
    preprocess_fundus_image,
)

_SERVICE = None


def get_inference_service():
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = DRInferenceService()
    return _SERVICE


class DRInferenceService:
    def __init__(self):
        self.densenet_model = None
        self.efficientnet_model = None
        self.config = self._load_config()

    def _load_config(self):
        if INFERENCE_CONFIG_PATH.exists():
            try:
                return json.loads(INFERENCE_CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    @property
    def is_ready(self):
        return self.densenet_model is not None and self.efficientnet_model is not None

    def load_models(self):
        missing = []
        if not DENSENET_MODEL_PATH.exists():
            missing.append(str(DENSENET_MODEL_PATH))
        if not EFFICIENTNET_MODEL_PATH.exists():
            missing.append(str(EFFICIENTNET_MODEL_PATH))

        if missing:
            raise FileNotFoundError(
                "Thiếu file model. Hãy copy model vào backend/models/: "
                + ", ".join(missing)
            )

        self.densenet_model = keras.models.load_model(DENSENET_MODEL_PATH, compile=False)
        self.efficientnet_model = keras.models.load_model(EFFICIENTNET_MODEL_PATH, compile=False)

    def health(self):
        return {
            "ready": self.is_ready,
            "img_size": IMG_SIZE,
            "models": {
                "densenet121": {
                    "exists": DENSENET_MODEL_PATH.exists(),
                    "loaded": self.densenet_model is not None,
                    "path": str(DENSENET_MODEL_PATH),
                    "weight": WEIGHT_DENSENET121,
                },
                "efficientnetb3": {
                    "exists": EFFICIENTNET_MODEL_PATH.exists(),
                    "loaded": self.efficientnet_model is not None,
                    "path": str(EFFICIENTNET_MODEL_PATH),
                    "weight": WEIGHT_EFFICIENTNETB3,
                },
            },
            "prediction_rule": "ensemble_argmax",
            "preprocessing": "crop_black_border + CLAHE_LAB + resize; no_ESRGAN",
        }

    def _entropy(self, probs):
        probs = np.clip(probs.astype("float64"), 1e-12, 1.0)
        return float(-(probs * np.log(probs)).sum())

    def _expected_severity(self, probs):
        return float(np.sum(probs * np.arange(len(probs))))

    def _top3(self, probs):
        ids = np.argsort(probs)[::-1][:3]
        return [
            {
                "class_id": int(i),
                "class_name": CLASS_NAMES[int(i)],
                "probability": float(probs[int(i)]),
            }
            for i in ids
        ]

    def _warnings(self, predicted_class, probs, uncertainty):
        warnings = []

        p2 = float(probs[2])
        p3 = float(probs[3])

        if predicted_class == 2 and p3 >= WARN_P3_WHEN_CLASS2:
            warnings.append(
                "Mô hình dự đoán Moderate DR nhưng xác suất Severe DR cũng đáng chú ý. "
                "Khuyến nghị kiểm tra lại bởi chuyên gia để tránh đánh giá thấp mức độ bệnh."
            )

        if predicted_class == 3 and p2 >= WARN_P2_WHEN_CLASS3:
            warnings.append(
                "Mô hình chưa chắc chắn giữa Moderate DR và Severe DR. "
                "Cần đối chiếu thêm bởi bác sĩ chuyên khoa."
            )

        if abs(p2 - p3) <= MODERATE_SEVERE_MARGIN and (p2 >= 0.20 or p3 >= 0.20):
            warnings.append(
                "Xác suất class 2 và class 3 khá gần nhau. "
                "Đây là vùng ranh giới khó, kết quả cần được xem như tham khảo."
            )

        if uncertainty >= HIGH_UNCERTAINTY:
            warnings.append(
                "Độ bất định cao. Mô hình chưa đủ chắc chắn với ảnh đầu vào này."
            )

        warnings.append(
            "Hệ thống chỉ hỗ trợ tham khảo/sàng lọc, không thay thế chẩn đoán của bác sĩ."
        )
        return warnings

    def predict(self, image_path, filename=""):
        if not self.is_ready:
            self.load_models()

        processed_rgb = preprocess_fundus_image(image_path, img_size=IMG_SIZE, use_clahe=True)

        x_dn = prepare_model_input(processed_rgb, "densenet121")
        x_en = prepare_model_input(processed_rgb, "efficientnetb3")

        probs_dn = self.densenet_model.predict(x_dn, verbose=0)[0].astype("float64")
        probs_en = self.efficientnet_model.predict(x_en, verbose=0)[0].astype("float64")

        probs_dn = probs_dn / (probs_dn.sum() + 1e-12)
        probs_en = probs_en / (probs_en.sum() + 1e-12)

        final_probs = WEIGHT_EFFICIENTNETB3 * probs_en + WEIGHT_DENSENET121 * probs_dn
        final_probs = final_probs / (final_probs.sum() + 1e-12)

        predicted_class = int(np.argmax(final_probs))
        confidence = float(final_probs[predicted_class])
        uncertainty = float(1.0 - confidence)
        entropy = self._entropy(final_probs)
        expected_severity = self._expected_severity(final_probs)

        heatmaps = []
        try:
            hm_dn = gradient_saliency_heatmap(self.densenet_model, x_dn, predicted_class)
            if hm_dn is not None:
                heatmaps.append((WEIGHT_DENSENET121, hm_dn))
        except Exception:
            pass

        try:
            hm_en = gradient_saliency_heatmap(self.efficientnet_model, x_en, predicted_class)
            if hm_en is not None:
                heatmaps.append((WEIGHT_EFFICIENTNETB3, hm_en))
        except Exception:
            pass

        if heatmaps:
            total_w = sum(w for w, _ in heatmaps)
            heatmap = sum((w / total_w) * hm for w, hm in heatmaps)
        else:
            heatmap = None

        result = {
            "filename": filename,
            "predicted_class": predicted_class,
            "predicted_class_name": CLASS_NAMES[predicted_class],
            "short_class_name": SHORT_CLASS_NAMES[predicted_class],
            "confidence": confidence,
            "uncertainty": uncertainty,
            "entropy": entropy,
            "expected_severity_score": expected_severity,
            "probabilities": [float(x) for x in final_probs],
            "model_outputs": {
                "densenet121": [float(x) for x in probs_dn],
                "efficientnetb3": [float(x) for x in probs_en],
            },
            "ensemble_weights": {
                "efficientnetb3": WEIGHT_EFFICIENTNETB3,
                "densenet121": WEIGHT_DENSENET121,
            },
            "top3": self._top3(final_probs),
            "warnings": self._warnings(predicted_class, final_probs, uncertainty),
            "class_names": CLASS_NAMES,
        }

        return result, processed_rgb, heatmap
