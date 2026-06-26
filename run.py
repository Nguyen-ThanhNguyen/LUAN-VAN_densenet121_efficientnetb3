import os
from backend.app.main import create_app

app = create_app()

if __name__ == "__main__":
    host = os.getenv("DR_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("DR_PORT", "7860")))
    debug = os.getenv("DR_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
