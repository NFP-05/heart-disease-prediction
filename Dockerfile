# =============================================================================
# Dockerfile – FastAPI Backend
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if any needed by scikit-learn / numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project (artefacts, data, source)
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "src.app_api:app", "--host", "0.0.0.0", "--port", "8000"]
