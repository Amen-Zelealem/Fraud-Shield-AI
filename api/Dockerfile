# Use an official Python image (latest stable version for production)
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    LOG_LEVEL=info

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code after dependencies are installed
COPY . .

# Expose the API port
EXPOSE 5000

# Set a health check for the container (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:5000/health || exit 1

# Start the application with Gunicorn (optimized for production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "serve_model:application"]