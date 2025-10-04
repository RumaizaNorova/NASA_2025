# Dockerfile for Shark Habitat Prediction Model on GCP
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libhdf5-dev \
    libnetcdf-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements_ai_enhanced.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_ai_enhanced.txt

# Install GCP dependencies
RUN pip install --no-cache-dir \
    google-cloud-storage \
    google-cloud-bigquery \
    google-cloud-aiplatform \
    google-cloud-functions-framework

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/interim models logs

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/gcp_credentials.json

# Expose port for Cloud Run
EXPOSE 8080

# Default command
CMD ["python", "src/train_real_data_model.py"]
