FROM python:3.11-slim

# System dependencies for audio DSP, Praat phonetics, and libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose TeleGuard port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start server with Uvicorn
CMD ["uvicorn", "prototype.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
