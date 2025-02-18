import logging
import sys

def get_logger(name="fraud_api"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler with UTF-8 support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler("logs/fraud_api.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Formatter with timestamps
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Attach handlers to logger
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
