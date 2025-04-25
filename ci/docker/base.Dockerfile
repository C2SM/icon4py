FROM ubuntu:22.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    strace \
    build-essential \
    gfortran \
    tar \
    wget \
    curl \
    ca-certificates \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    libhdf5-dev \
    liblzma-dev \
    python3-openssl \
    libreadline-dev \
    git \
    jq \
    htop && \
    rm -rf /var/lib/apt/lists/*

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustc --version && which rustc && cargo --version && which cargo

# Install Bencher for performance monitoring
RUN curl --proto '=https' --tlsv1.2 -sSfL https://bencher.dev/download/install-cli.sh | sh
RUN bencher --version && which bencher

# Install NVIDIA HPC SDK for nvfortran
ARG HPC_SDK_VERSION=24.11
ARG HPC_SDK_NAME=nvhpc_2024_2411_Linux_aarch64_cuda_12.6
ENV HPC_SDK_URL=https://developer.download.nvidia.com/hpc-sdk/${HPC_SDK_VERSION}/${HPC_SDK_NAME}.tar.gz

RUN wget -q ${HPC_SDK_URL} -O /tmp/nvhpc.tar.gz && \
    mkdir -p /opt/nvidia && \
    tar -xzf /tmp/nvhpc.tar.gz -C /opt/nvidia && \
    rm /tmp/nvhpc.tar.gz

ENV NVHPC_SILENT=1
RUN cd /opt/nvidia/${HPC_SDK_NAME} && ./install

# Set environment variables
ARG ARCH=aarch64
ENV HPC_SDK_PATH=/opt/nvidia/hpc_sdk/Linux_${ARCH}/${HPC_SDK_VERSION}
# The variable CUDA_PATH is used by cupy to find the cuda toolchain
ENV CUDA_PATH=${HPC_SDK_PATH}/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PATH=${HPC_SDK_PATH}/compilers/bin:${HPC_SDK_PATH}/comm_libs/mpi/bin:${PATH} \
    MANPATH=${HPC_SDK_PATH}/compilers/man:${MANPATH} \
    LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${HPC_SDK_PATH}/math_libs/lib64:${LD_LIBRARY_PATH}

# Install pyenv and Python version specified by PYVERSION
ARG PYVERSION
RUN curl https://pyenv.run | bash

ENV PYENV_ROOT /root/.pyenv
ENV PATH="/root/.pyenv/bin:${PATH}"

RUN pyenv update && \
    pyenv install ${PYVERSION} && \
    echo 'eval "$(pyenv init -)"' >> /root/.bashrc && \
    eval "$(pyenv init -)" && \
    pyenv global ${PYVERSION}

ENV PATH="/root/.pyenv/shims:${PATH}"

RUN pip install --upgrade pip setuptools wheel uv nox clang-format
