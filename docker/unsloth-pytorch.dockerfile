FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
  wget git git-lfs curl nano build-essential \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade uv && uv pip install --system --upgrade pip setuptools wheel

RUN uv pip install --system --upgrade xformers unsloth[huggingface] bitsandbytes wandb apollo-torch \
  --no-build-isolation \
  --index-strategy=unsafe-best-match \
  --extra-index-url=https://download.pytorch.org/whl/cu126
