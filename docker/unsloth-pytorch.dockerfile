
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --yes && \
  apt install --yes --no-install-recommends git wget curl nano openssh-server && \
  apt-get autoremove -y && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --no-cache-dir uv && uv pip install --system --upgrade pip setuptools wheel packaging ninja

RUN uv pip install --system --upgrade torch==2.6.0+cu126 unsloth[huggingface] bitsandbytes wandb apollo-torch flash-attn xformers\
  --no-build-isolation \
  --index-strategy=unsafe-best-match \
  --extra-index-url=https://download.pytorch.org/whl/cu126

# COPY docker/start.sh /
# RUN chmod +x /start.sh
# CMD [ "/start.sh" ]
