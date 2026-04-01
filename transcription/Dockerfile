FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install PyTorch using the CUDA 12.8 index (Supported by your 570 driver)
# Note: TorchVision 0.22.0 might be 0.26.0 in the 2.11.0 release; 
# letting pip pick the companion versions is safer.
RUN pip install --no-cache-dir \
    torch==2.11.0+cu128 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# 2. Install the application dependencies
RUN pip install --no-cache-dir \
    'nemo_toolkit[asr]' \
    'pyannote.audio==3.1.1' \
    fastapi \
    'uvicorn[standard]' \
    python-multipart \
    --extra-index-url https://download.pytorch.org/whl/cu128

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]