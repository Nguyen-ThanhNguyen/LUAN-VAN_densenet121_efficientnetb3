from flask import Flask, send_from_directory
from flask_cors import CORS

from backend.app.api.routes import api_bp
from backend.app.core.config import (
    FRONTEND_DIR,
    MAX_CONTENT_LENGTH,
    STATIC_DIR,
)


def create_app():
    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    app.register_blueprint(api_bp, url_prefix="/api")

    @app.route("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.route("/frontend/<path:filename>")
    def frontend_assets(filename):
        return send_from_directory(FRONTEND_DIR, filename)

    @app.errorhandler(413)
    def file_too_large(_):
        return {"error": "File quá lớn. Kích thước tối đa là 10MB."}, 413

    return app
