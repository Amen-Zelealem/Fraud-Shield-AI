import os

# Logging configuration
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "api.log")

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "random_forest_fraud_best_model.pkl")

# Ensure required directories exist
os.makedirs(LOG_DIR, exist_ok=True)
