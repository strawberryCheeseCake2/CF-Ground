FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git git-lfs \
      build-essential \
      libgl1 libglib2.0-0 libsm6 libxext6 \
      ffmpeg \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["bash"]
