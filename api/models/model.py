import joblib
import os
import logging

# Configurations
MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest_fraud_best_model.pkl")

def load_model():
    """Loads and returns the trained fraud detection model."""
    try:
        model = joblib.load(MODEL_PATH)
        logging.info(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        raise

model = load_model()
