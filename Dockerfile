# syntax=docker/dockerfile:1
#
# Multi-stage build for the EasyOCR FastAPI microservice.
#   1. deps    — installs Python deps into a venv AND prefetches the
#      EasyOCR model weights (app/ocr_engine.py builds two Readers at
#      *import* time — a cold start with no cached models would otherwise
#      try to download them from GitHub on the first real request).
#   2. runtime — only the venv, the prefetched models, and app/ are
#      copied in. No pip, no compiler, no source outside app/.
#
# gpu=False is hardcoded in app/ocr_engine.py, so torch/torchvision are
# installed from the CPU-only PyTorch index instead of the default PyPI
# wheel (which bundles CUDA on Linux and would add ~1-2GB for nothing).

############################################
# Stage 1: deps + model prefetch
############################################
FROM python:3.10-slim AS deps

# libgl1/libglib2.0-0/libgomp1: opencv-python-headless and torch still
# dynamically link against these even in "headless" builds — without them
# `import cv2` / `import torch` fails at runtime with a missing .so error.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
COPY requirements.txt ./
# torch/torchvision versions match what's actually installed and working
# in the project's local venv (pip freeze) — but the CPU-only wheel for
# this exact version pair on the Linux index below could not be
# pulled/verified in this environment (no internet-connected Docker
# available). If this install step 404s, check
# https://download.pytorch.org/whl/cpu/torch/ for the closest available
# version to 2.11.0 and update both pins together (torch/torchvision
# versions must stay compatible with each other).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0 torchvision==0.26.0 \
    && pip install --no-cache-dir -r requirements.txt

# tests/ exercises app/parser.py and the extract_best control flow with
# easyocr.Reader patched out (see tests/test_en_first_fallback.py) — no
# model download needed to run these, so they run before the expensive
# model-prefetch step below and fail the build fast on a real regression.
COPY app/ app/
COPY tests/ tests/
RUN pip install --no-cache-dir pytest==9.0.3 \
    && python -m pytest tests/ -q

# Prefetch model weights into a fixed, copyable path (independent of any
# user's home directory) instead of leaving them to download on first
# request — see easyocr/config.py: EASYOCR_MODULE_PATH overrides the
# default ~/.EasyOCR/ location.
ENV EASYOCR_MODULE_PATH=/opt/easyocr-models
RUN python -c "\
import easyocr; \
easyocr.Reader(['en'], gpu=False, download_enabled=True); \
easyocr.Reader(['ar', 'en'], gpu=False, detector=False, download_enabled=True)"

############################################
# Stage 2: runtime
############################################
FROM python:3.10-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 ocr \
    && useradd --uid 1000 --gid ocr --shell /usr/sbin/nologin --no-create-home ocr

COPY --from=deps /opt/venv /opt/venv
COPY --from=deps /opt/easyocr-models /opt/easyocr-models

WORKDIR /app
COPY app/ app/

# temp_uploads is created on demand by app/utils.py's ensure_upload_dir(),
# but pre-creating + chowning it here avoids relying on the app to create
# a directory inside a root-owned tree at request time.
RUN mkdir -p /app/temp_uploads && chown -R ocr:ocr /app /opt/easyocr-models

ENV PATH="/opt/venv/bin:${PATH}" \
    EASYOCR_MODULE_PATH=/opt/easyocr-models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER ocr:ocr

EXPOSE 8000

# No extra HTTP client installed for this — GET / (app/main.py's
# health_check) is checked with Python's own stdlib, which is guaranteed
# present, instead of adding curl/wget to the runtime image.
# start-period is generous: model loading happens at process import time
# (see app/ocr_engine.py's module-level `ocr_engine = EasyOCREngine()`),
# which is CPU-bound and can take a while on a cold container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)" || exit 1

# --workers 1: app/ocr_engine.py builds its EasyOCR Readers as a single
# process-level singleton (`ocr_engine = EasyOCREngine()` at import time,
# no shared state across processes) — each additional uvicorn worker would
# load its own full copy of the model weights into memory rather than
# sharing anything, so more workers means more RAM for no correctness
# benefit here. Raise this only after provisioning enough memory per
# worker.
ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app", \
    "--host", "0.0.0.0", "--port", "8000", \
    "--workers", "1", "--timeout-keep-alive", "30"]
