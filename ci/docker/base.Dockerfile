ARG CUDA_VERSION=11.8.0
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-base-ubuntu20.04

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
    python-openssl \
    libreadline-dev \
    git \
    rustc \
    htop && \
    rm -rf /var/lib/apt/lists/*

# Install NVIDIA HPC SDK for nvfortran
ARG HPC_SDK_VERSION=22.11
ARG HPC_SDK_NAME=nvhpc_2022_2211_Linux_x86_64_cuda_11.8
ENV HPC_SDK_URL=https://developer.download.nvidia.com/hpc-sdk/${HPC_SDK_VERSION}/${HPC_SDK_NAME}.tar.gz

RUN wget -q ${HPC_SDK_URL} -O /tmp/nvhpc.tar.gz && \
    mkdir -p /opt/nvidia && \
    tar -xzf /tmp/nvhpc.tar.gz -C /opt/nvidia && \
    rm /tmp/nvhpc.tar.gz

ENV NVHPC_SILENT=1
RUN cd /opt/nvidia/${HPC_SDK_NAME} && ./install

# Set environment variables
ARG ARCH=x86_64
ENV HPC_SDK_PATH=/opt/nvidia/hpc_sdk/Linux_${ARCH}/${HPC_SDK_VERSION}
# The variable CUDA_PATH is used by cupy to find the cuda toolchain
ENV CUDA_PATH=${HPC_SDK_PATH}/cuda

ENV PATH=${HPC_SDK_PATH}/compilers/bin:${HPC_SDK_PATH}/comm_libs/mpi/bin:${PATH} \
    MANPATH=${HPC_SDK_PATH}/compilers/man:${MANPATH} \
    LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${HPC_SDK_PATH}/math_libs/lib64:${LD_LIBRARY_PATH}

# Install Boost
RUN wget --quiet https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz && \
    echo be0d91732d5b0cc6fbb275c7939974457e79b54d6f07ce2e3dfdd68bef883b0b boost_1_85_0.tar.gz > boost_hash.txt && \
    sha256sum -c boost_hash.txt && \
    tar xzf boost_1_85_0.tar.gz && \
    mv boost_1_85_0/boost /usr/local/include/ && \
    rm boost_1_85_0.tar.gz boost_hash.txt

ENV BOOST_ROOT /usr/local/

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

ARG CUPY_PACKAGE=cupy-cuda11x
RUN pip install --upgrade pip setuptools wheel tox clang-format ${CUPY_PACKAGE}
