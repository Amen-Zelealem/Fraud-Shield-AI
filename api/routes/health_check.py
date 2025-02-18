from flask import Blueprint, jsonify
from datetime import datetime
from logger import get_logger  # Import from logger.py

health_check_bp = Blueprint("health", __name__)
logger = get_logger()

@health_check_bp.route("/health", methods=["GET"])
def health_check():
    """✅ Health Check Endpoint - Returns API Status."""
    
    logger.info("🩺 Health check requested ✅")
    
    return jsonify({
        "status": "🟢 Healthy",
        "timestamp": str(datetime.now()),
        "message": "🚀 API is running smoothly!"
    })
