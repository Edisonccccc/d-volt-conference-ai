# Conference AI Assistant — single-stage backend image.
#
# Built from the repo root so that both backend/ and web/ are in the
# build context (the FastAPI app mounts web/ at /app/).
FROM python:3.11-slim

# Don't write .pyc files; flush stdout/stderr immediately so we see logs live.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

# Minimal system layer. Pillow + reportlab + anthropic + openai all ship
# manylinux wheels, so we don't need build-essential here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer caches across code edits.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the app and the static web tester.
COPY backend/ ./backend/
COPY web/ ./web/

# Persistent disk mount point on Render. SQLite, uploads, and reports live here.
RUN mkdir -p /data
VOLUME ["/data"]

# Run from inside backend/ so app.* imports resolve and WEB_DIR (= ../web)
# resolves relative to /app/backend.
WORKDIR /app/backend

# Render injects PORT; we default to 8000 for `docker run` locally.
EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
