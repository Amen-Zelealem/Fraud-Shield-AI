# Use an official Python image (latest stable version for production)
FROM python:3.11-slim


# Set the working directory inside the container
WORKDIR /app

# Copy application code after dependencies are installed
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Create the logs directory
RUN mkdir -p logs && chmod 777 logs


# Expose the API port
EXPOSE 5000

# Set a health check for the container (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:5000/health || exit 1

# Start the application with Gunicorn (optimized for production)
CMD ["python","serve_model.py"]
