# Fraud Prediction API

This is a Flask-based API for fraud detection, utilizing machine learning models such as Random Forest, Gradient Boosting, LSTM, and CNN. The API serves predictions and includes health checks for monitoring.

## Project Structure

```
api/
│── logs/
│── fraud_api.log        # Contains log files
│── models/ 
|   │── model.py         # Loads and manages the fraud detection model
|   │──gradient_boosting_creditcard_best_model.pkl
|   │──gradient_boosting_fraud_best_model.pkl
|   │──random_forest_fraud_best_model.pkl
│── routes/              # API route handlers
│   ├── predict.py       # Endpoint for fraud predictions
│   ├── health_check.py  # API health check endpoint
│── config.py            # configuration files
│── serve_model.py       # Main API server script
│── logger.py            # Logger setup
│── requirements.txt     # Python dependencies
│── Dockerfile           # Docker container setup
│── README.md            # Project documentation
```

## Features
- Machine learning-based fraud detection
- API health check endpoint
- Logging integration
- Containerized deployment with Docker
- Supports MLflow for model tracking

## Setup and Installation

### Prerequisites
- Python 3.8+
- Flask and dependencies (see `requirements.txt`)
- Docker (optional for containerized deployment)

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Running the API

```sh
python serve_model.py
```

The API should be running on `http://localhost:5000`.

### Running with Docker

```sh
docker build -t fraud-detection-api .
docker run -p 5000:5000 fraud-detection-api
```

### API Endpoints

#### Health Check
```
GET /health
```
Response:
```json
{
  "message": "🚀 API is running smoothly!",
  "status": "🟢 Healthy",
  "timestamp": "2025-02-18 15:33:39.737320"
}
```

## Logging
Logs are stored in the `logs/` directory. The API logs requests, responses, and errors for monitoring and debugging.

## MLflow Integration
MLflow is used for model tracking, hyperparameter tuning, and evaluation. Make sure to configure MLflow properly before running experiments.

## Contributing
Feel free to open issues or submit pull requests to improve the API.

## License
This project is licensed under the MIT License.

