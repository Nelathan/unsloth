# Base image for PyTorch + CUDA
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  wget git git-lfs curl nano \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
  && bash /tmp/miniconda.sh -b -p /root/miniconda3 \
  && rm /tmp/miniconda.sh

# Update PATH for Conda
ENV PATH="/root/miniconda3/bin:${PATH}"

# Install PyTorch with CUDA support
RUN conda install python=3.12 pytorch-cuda=12.4 pytorch cudatoolkit -c pytorch -c nvidia -y \
  && conda clean -afy

# Optional: Set a default work directory
WORKDIR /workspace
