# Dockerfile for Politician_Classifier_MLOps
# Uses Python 3.11, installs all requirements, exposes port 5000 for MLflow or API

FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files
COPY . .

# Expose port for MLflow UI or FastAPI/Flask app
EXPOSE 5000

# Default command (can be changed as needed)
CMD ["mlflow", "ui", "--backend-store-uri", "./mlruns", "--host", "0.0.0.0"]
