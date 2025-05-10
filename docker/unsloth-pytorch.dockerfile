FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

RUN apt-get update --yes && \
  apt-get upgrade --yes && \
  apt install --yes --no-install-recommends git git-lfs wget curl nano openssh-server build-essential && \
  apt-get autoremove -y && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN pip install --upgrade --no-cache-dir uv && uv pip install --system --upgrade pip setuptools wheel

RUN uv pip install --system --upgrade xformers unsloth[huggingface] bitsandbytes wandb apollo-torch \
  --no-build-isolation \
  --index-strategy=unsafe-best-match \
  --extra-index-url=https://download.pytorch.org/whl/cu126 # update

COPY docker/start.sh /
RUN chmod +x /start.sh
CMD [ "/start.sh" ]
