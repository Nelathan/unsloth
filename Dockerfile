FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
  wget git git-lfs curl nano build-essential \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV VERSION=2025.1.7

RUN pip install -U --no-cache-dir pip

RUN pip install -U --no-cache-dir \
  "unsloth[huggingface] @ git+https://github.com/unslothai/unsloth.git" \
  bitsandbytes wandb

RUN pip install --no-cache-dir \
  xformers --index-url https://download.pytorch.org/whl/cu124

RUN python -m xformers.info
