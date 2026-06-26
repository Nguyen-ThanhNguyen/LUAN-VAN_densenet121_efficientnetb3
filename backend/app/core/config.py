import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

MODEL_DIR = Path(os.getenv("DR_MODEL_DIR", str(BACKEND_DIR / "models"))).resolve()
DENSENET_MODEL_PATH = MODEL_DIR / os.getenv("DR_DENSENET_MODEL", "densenet121_best.h5")
EFFICIENTNET_MODEL_PATH = MODEL_DIR / os.getenv("DR_EFFICIENTNET_MODEL", "efficientnetb3_best.h5")
INFERENCE_CONFIG_PATH = MODEL_DIR / "inference_config.json"

IMG_SIZE = int(os.getenv("DR_IMG_SIZE", "320"))
NUM_CLASSES = 5

WEIGHT_EFFICIENTNETB3 = float(os.getenv("DR_WEIGHT_EFFICIENTNETB3", "0.55"))
WEIGHT_DENSENET121 = float(os.getenv("DR_WEIGHT_DENSENET121", "0.45"))

WARN_P3_WHEN_CLASS2 = float(os.getenv("DR_WARN_P3_WHEN_CLASS2", "0.20"))
WARN_P2_WHEN_CLASS3 = float(os.getenv("DR_WARN_P2_WHEN_CLASS3", "0.25"))
MODERATE_SEVERE_MARGIN = float(os.getenv("DR_MODERATE_SEVERE_MARGIN", "0.12"))
HIGH_UNCERTAINTY = float(os.getenv("DR_HIGH_UNCERTAINTY", "0.40"))

MAX_CONTENT_LENGTH = 10 * 1024 * 1024

STATIC_DIR = BACKEND_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
PROCESSED_DIR = STATIC_DIR / "processed"
HEATMAP_DIR = STATIC_DIR / "heatmaps"
REPORT_DIR = STATIC_DIR / "reports"

for folder in [MODEL_DIR, UPLOAD_DIR, PROCESSED_DIR, HEATMAP_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "No DR - Không có bệnh võng mạc đái tháo đường",
    1: "Mild DR - Bệnh nhẹ",
    2: "Moderate DR - Bệnh trung bình",
    3: "Severe DR - Bệnh nặng",
    4: "Proliferative DR - Tăng sinh",
}

SHORT_CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}
