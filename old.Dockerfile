# Use the specified base image with CUDA 12.6.1
FROM nvidia/cuda:12.6.1-devel-ubuntu24.04

# Set environment variables for non-interactive mode and add Miniconda to PATH
ENV DEBIAN_FRONTEND=noninteractive \
  PATH="/root/miniconda3/bin:$PATH"

# Install Miniconda and essential dependencies in a single RUN command to minimize layers
RUN apt-get update && apt-get install -y \
  wget \
  git \
  curl \
  bzip2 \
  build-essential \
  && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh \
  && bash /root/miniconda.sh -b -u -p /root/miniconda3 \
  && rm /root/miniconda.sh \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create the Conda environment with Python 3.11 and required CUDA packages
RUN conda create --name unsloth_env python=3.11 \
  pytorch-cuda=12.1 \
  pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y \
  && conda clean -afy

# Install unsloth and additional dependencies using pip in the Conda environment
RUN /root/miniconda3/bin/conda run -n unsloth_env \
  pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' \
  && /root/miniconda3/bin/conda run -n unsloth_env \
  pip install --no-deps trl peft accelerate bitsandbytes flash-attn einops

# Set the working directory
WORKDIR /workspace

# Set the default command to activate the environment when starting the container
CMD ["/bin/bash", "-c", "source activate unsloth_env && exec bash"]
