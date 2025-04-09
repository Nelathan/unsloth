FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  wget git git-lfs curl nano vim build-essential pkg-config cmake \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --upgrade pip setuptools wheel

RUN pip install -U --no-cache-dir torch xformers 'unsloth[huggingface] @ git+https://github.com/unslothai/unsloth.git' bitsandbytes wandb --extra-index-url=https://download.pytorch.org/whl/cu126

# RUN pip install -U --no-cache-dir torch==2.5.1 xformers 'unsloth[huggingface] @ git+https://github.com/unslothai/unsloth.git' bitsandbytes wandb pip --extra-index-url=https://download.pytorch.org/whl/cu124
