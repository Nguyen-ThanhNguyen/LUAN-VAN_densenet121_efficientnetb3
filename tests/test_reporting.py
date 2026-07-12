from backend.app.utils.reporting import write_prediction_report


def test_write_prediction_report_creates_text_and_html(tmp_path):
    result = {
        "filename": "sample.jpg",
        "predicted_class": 0,
        "predicted_class_name": "No DR - Khong co benh vong mac dai thao duong",
        "confidence": 0.7711,
        "uncertainty": 0.2289,
        "entropy": 0.7023,
        "expected_severity_score": 0.3122,
        "probabilities": [0.7711, 0.1552, 0.0685, 0.0006, 0.0045],
        "model_outputs": {
            "densenet121": [0.5067, 0.3361, 0.1481, 0.0004, 0.0087],
            "efficientnetb3": [0.9874, 0.0073, 0.0033, 0.0007, 0.0013],
        },
        "ensemble_weights": {
            "efficientnetb3": 0.55,
            "densenet121": 0.45,
        },
        "top3": [
            {
                "class_id": 0,
                "class_name": "No DR - Khong co benh vong mac dai thao duong",
                "probability": 0.7711,
            }
        ],
        "warnings": ["Ket qua chi ho tro tham khao."],
        "class_names": {
            0: "No DR - Khong co benh vong mac dai thao duong",
            1: "Mild DR - Benh nhe",
            2: "Moderate DR - Benh trung binh",
            3: "Severe DR - Benh nang",
            4: "Proliferative DR - Tang sinh",
        },
        "original_image_url": "/static/uploads/sample.jpg",
        "processed_image_url": "/static/processed/sample_processed.jpg",
        "heatmap_url": "/static/heatmaps/sample_heatmap.jpg",
    }

    text_path = tmp_path / "sample_report.txt"
    html_path = write_prediction_report(result, text_path)

    assert text_path.exists()
    assert html_path.exists()
    assert html_path.name == "sample_report.html"
    assert "BÁO CÁO PHÂN TÍCH ẢNH ĐÁY MẮT" in text_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "Kết quả sàng lọc bệnh võng mạc đái tháo đường" in html
    assert "./sample_report.txt" in html
