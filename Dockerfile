FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
  wget git git-lfs curl nano build-essential \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
  "unsloth[huggingface] @ git+https://github.com/unslothai/unsloth.git" \
  bitsandbytes wandb

RUN pip install --no-cache-dir \
  xformers --index-url https://download.pytorch.org/whl/cu126

RUN python -m xformers.info
