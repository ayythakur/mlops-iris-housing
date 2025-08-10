# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system deps if needed (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY api/ api/
COPY src/ src/
COPY params/ params/

# Bake the trained model into the image
# (Assumes you ran training locally and models/registry exists)
COPY models/registry/ models/registry/

# Expose API port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
