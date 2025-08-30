FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt \
    && echo "=== Installed packages in builder stage ===" \
    && pip list --no-cache-dir

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    PORT=8080 \
    CHECKPOINT_PATH="/app/checkpoint-266490"

# Ensure offline operation; never attempt Hub downloads/resolution
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

RUN echo "=== Installed packages in final image ===" \
    && pip list --no-cache-dir

# Copy application code
COPY . .

EXPOSE $PORT

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
