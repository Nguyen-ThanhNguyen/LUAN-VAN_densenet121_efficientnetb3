from uuid import uuid4

from flask import Blueprint, jsonify, request

from backend.app.core.config import (
    UPLOAD_DIR,
    PROCESSED_DIR,
    HEATMAP_DIR,
    REPORT_DIR,
    CLASS_NAMES,
)
from backend.app.services.inference_service import get_inference_service
from backend.app.utils.validation import validate_upload, safe_extension
from backend.app.utils.image_processing import save_rgb_image
from backend.app.utils.heatmap import save_heatmap_overlay
from backend.app.utils.reporting import write_prediction_report

api_bp = Blueprint("api", __name__)

@api_bp.route("/health", methods=["GET"])
def health():
    service = get_inference_service()
    try:
        if not service.is_ready:
            service.load_models()
        status = service.health()
        status["status"] = "ok"
        return jsonify(status), 200
    except Exception as e:
        status = service.health()
        status["status"] = "error"
        status["error"] = str(e)
        return jsonify(status), 503

@api_bp.route("/class-names", methods=["GET"])
def class_names():
    return jsonify(CLASS_NAMES), 200

@api_bp.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    valid, error_message = validate_upload(file)
    if not valid:
        return jsonify({"error": error_message}), 400

    uid = uuid4().hex
    ext = safe_extension(file.filename)

    upload_path = UPLOAD_DIR / f"{uid}.{ext}"
    processed_path = PROCESSED_DIR / f"{uid}_processed.jpg"
    heatmap_path = HEATMAP_DIR / f"{uid}_heatmap.jpg"
    report_path = REPORT_DIR / f"{uid}_report.txt"

    file.save(str(upload_path))

    service = get_inference_service()

    try:
        result, processed_rgb, heatmap = service.predict(upload_path, filename=file.filename)

        save_rgb_image(processed_rgb, processed_path)
        result["original_image_url"] = f"/static/uploads/{upload_path.name}"
        result["processed_image_url"] = f"/static/processed/{processed_path.name}"

        if heatmap is not None:
            save_heatmap_overlay(processed_rgb, heatmap, heatmap_path)
            result["heatmap_url"] = f"/static/heatmaps/{heatmap_path.name}"
        else:
            result["heatmap_url"] = ""

        write_prediction_report(result, report_path)
        result["report_url"] = f"/static/reports/{report_path.name}"

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
