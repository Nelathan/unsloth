FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
  wget git git-lfs curl nano build-essential \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
  "unsloth[cu124-torch251] @ git+https://github.com/unslothai/unsloth.git" \
  wandb --root-user-action=ignore
# RUN pip install --no-cache-dir cut_cross_entropy==24.12.3 safetensors==0.4.5 xformers==0.0.28.post3 --root-user-action=ignore
# RUN pip install --no-cache-dir --root-user-action=ignore

ENV VERSION=2025.1.2

RUN pip install --no-cache-dir "unsloth[cu124-torch251] @ git+https://github.com/unslothai/unsloth.git" --root-user-action=ignore
