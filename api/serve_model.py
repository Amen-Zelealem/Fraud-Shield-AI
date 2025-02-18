from flask import Flask
import logging
from routes.predict import predict_bp
from routes.health_check import health_check_bp
from logger import get_logger  

logger = get_logger()

def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True

    # Register blueprints
    app.register_blueprint(predict_bp)
    app.register_blueprint(health_check_bp)

    logger.info("Flask app initialized")
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
