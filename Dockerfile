# Document Screening Engine - production image
# Runtime dependencies only: OpenCV (libgl), Tesseract OCR, Google's face model.
FROM python:3.13-slim-trixie

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV API_ENV=production
ENV LOG_LEVEL=INFO
ENV FACE_MODELS_DIR=/opt/insightface
ENV FACE_MODEL_NAME=buffalo_sc

# libgl/libglib are required by opencv-python; tesseract-ocr for MRZ OCR.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the face recognition model so startup does not need the network.
# Runs before the app code is copied so the Docker layer is cache-friendly.
RUN python - <<'EOF'
from insightface.app import FaceAnalysis
FaceAnalysis(name="buffalo_sc", root="/opt/insightface", download=True)
EOF

COPY . .

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
  && chown -R appuser:appuser /app \
  && mkdir -p /data && chown -R appuser:appuser /data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Graceful shutdown: uvicorn handles SIGTERM/SIGINT for in-flight requests.
# Run Alembic migrations to HEAD before starting the API (idempotent).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1"]