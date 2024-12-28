ARG MAX_JOBS=8

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update \
  && apt-get install -y wget git build-essential ninja-build git-lfs libaio-dev \
  && apt-get install -y --allow-change-held-packages vim curl nano libnccl2 libnccl-dev rsync s3fs \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && mkdir /root/.conda \
  && bash Miniconda3-latest-Linux-x86_64.sh -b \
  && rm -f Miniconda3-latest-Linux-x86_64.sh \
  && conda create -n unsloth python="3.11" pytorch-cuda="12.4" \
  pytorch cudatoolkit -c pytorch -c nvidia -y \
  && conda clean -afy

ENV PATH="/root/miniconda3/envs/unsloth/bin:${PATH}"

WORKDIR /workspace

RUN pip install "unsloth[cu124-ampere-torch251] @ git+https://github.com/unslothai/unsloth.git" wandb --root-user-action=ignore\
  && pip install --no-deps trl peft accelerate bitsandbytes --root-user-action=ignore
