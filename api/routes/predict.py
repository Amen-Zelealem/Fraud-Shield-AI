from flask import Blueprint, request, jsonify
import numpy as np
from datetime import datetime
from models import model
from logger import get_logger  # Import from logger.py

predict_bp = Blueprint("predict", __name__)
logger = get_logger()

@predict_bp.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    try:
        # Log request
        logger.info(f"New prediction request from {request.remote_addr}")
        
        # Validate input
        data = request.get_json()
        if not data or "features" not in data:
            logger.warning("Invalid request format")
            return jsonify({"error": "Invalid input format"}), 400
            
        # Convert features to numpy array
        features = np.array(data["features"]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[:, 1].tolist()
        
        # Log prediction
        logger.info(f"Prediction: {prediction[0]}, Probability: {probability[0]:.4f}")
        
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "timestamp": str(datetime.now())
        })
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
