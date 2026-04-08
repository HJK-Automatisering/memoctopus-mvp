FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# -------------------------
# IMPORTANT: build deps FIRST
# -------------------------
RUN pip install --no-cache-dir \
    cython \
    numpy \
    packaging

# -------------------------
# PyTorch (CUDA 11.8 stable)
# -------------------------
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# -------------------------
# ML stack
# -------------------------
RUN pip install --no-cache-dir \
    'nemo_toolkit[asr]==1.21.0' \
    pyannote.audio==3.1.1 \
    fastapi \
    'uvicorn[standard]' \
    python-multipart

# -------------------------
# App
# -------------------------
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]