FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    python3-distutils \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch 2.0.0 + CUDA 11.8 and related dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.0+cu118 \
        torchvision==0.15.1+cu118 \
        torchaudio==2.0.1+cu118

# Install Python libraries (match your working env)
RUN pip install \
    wheel==0.45.1 \
    mmengine==0.10.6 \
    openmim==0.3.9 \
    opencv-python==4.9.0.80 \
    opencv-python-headless==4.11.0.86 \
    supervision==0.19.0 \
    matplotlib==3.7.5 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scipy==1.10.0 \
    scikit-learn==1.3.2 \
    albumentations==1.4.0 \
    addict==2.4.0 \
    pycocotools==2.0.7 \
    tqdm==4.65.2 \
    shapely==2.0.7 \
    Pillow==10.2.0 \
    timm==0.6.13 \
    onnx==1.17.0 \
    onnxruntime==1.19.2 \
    onnxsim==0.4.36 \
    rich==13.4.2 \
    prettytable==3.11.0 \
    openxlab==0.1.2 \
    huggingface-hub==0.30.2 \
    transformers==4.33.0

# Install MMCV 2.0.0 using MIM
RUN mim install mmcv==2.0.0

RUN pip install \
    mmdet==3.3.0 \
    mmyolo==0.6.0 \
    git+https://github.com/lvis-dataset/lvis-api.git 