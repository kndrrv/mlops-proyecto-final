# ── Stage 1: dependencias ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: imagen final ─────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copiar dependencias instaladas
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar código fuente
COPY src/ ./src/
COPY api/ ./api/
COPY .dvc/ ./.dvc/

# Crear directorios para datos y modelos (serán montados o descargados en runtime)
RUN mkdir -p data models

# Variables de entorno por defecto (sobreescribir en producción)
ENV MODEL_PATH=/app/models/model.pkl
ENV DATA_PATH=/app/data/dataset.csv
ENV METRICS_PATH=/app/metrics.json
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Healthcheck nativo de Docker
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
