FROM ubuntu:24.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        curl \
        git \
        libboost-dev \
        libconfig-dev \
        libcurl4-openssl-dev \
        libfuse-dev \
        libjson-c-dev \
        libnl-3-dev \
        libnuma-dev \
        libreadline-dev \
        libsensors-dev \
        libssl-dev \
        libtool \
        libuv1-dev \
        libyaml-dev \
        nvidia-cuda-dev \
        nvidia-cuda-toolkit \
        nvidia-cuda-toolkit-gcc \
        pkg-config \
        python3 \
        strace \
        tar \
        wget && \
    rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/cuda-gcc
ENV CXX=/usr/bin/cuda-g++
ENV CUDAHOSTCXX=/usr/bin/cuda-g++

# Install OpenMPI configured with libfabric, libcxi, and gdrcopy support for use
# on Alps. This is based on examples in
# https://github.com/eth-cscs/cray-network-stack.
ARG gdrcopy_version=2.5.1
RUN set -eux; \
    git clone --depth 1 --branch "v${gdrcopy_version}" https://github.com/NVIDIA/gdrcopy.git; \
    cd gdrcopy; \
    make lib -j"$(nproc)" lib_install; \
    cd /; \
    rm -rf /gdrcopy; \
    ldconfig

ARG cassini_headers_version=release/shs-13.0.0
RUN set -eux; \
    git clone --depth 1 --branch "${cassini_headers_version}" https://github.com/HewlettPackard/shs-cassini-headers.git; \
    cd shs-cassini-headers; \
    cp -r include/* /usr/include/; \
    cp -r share/* /usr/share/; \
    rm -rf /shs-cassini-headers

ARG cxi_driver_version=release/shs-13.0.0
RUN set -eux; \
    git clone --depth 1 --branch "${cxi_driver_version}" https://github.com/HewlettPackard/shs-cxi-driver.git; \
    cd shs-cxi-driver; \
    cp -r include/* /usr/include/; \
    rm -rf /shs-cxi-driver

ARG libcxi_version=release/shs-13.0.0
RUN set -eux; \
    git clone --depth 1 --branch "${libcxi_version}" https://github.com/HewlettPackard/shs-libcxi.git; \
    cd shs-libcxi; \
    ./autogen.sh; \
    ./configure \
      --with-cuda; \
    make -j"$(nproc)" install; \
    cd /; \
    rm -rf /shs-libcxi; \
    ldconfig

ARG xpmem_version=0d0bad4e1d07b38d53ecc8f20786bb1328c446da
RUN set -eux; \
    git clone https://github.com/hpc/xpmem.git; \
    cd xpmem; \
    git checkout "${xpmem_version}"; \
    ./autogen.sh; \
    ./configure --disable-kernel-module; \
    make -j"$(nproc)" install; \
    cd /; \
    rm -rf /xpmem; \
    ldconfig

# NOTE: xpmem is not found correctly without setting the prefix explicitly in
# --enable-xpmem
ARG libfabric_version=v2.4.0
RUN set -eux; \
    git clone --depth 1 --branch "${libfabric_version}" https://github.com/ofiwg/libfabric.git; \
    cd libfabric; \
    ./autogen.sh; \
    ./configure \
      --with-cuda \
      --enable-xpmem=/usr \
      --enable-tcp \
      --enable-cxi; \
    make -j"$(nproc)" install; \
    cd /; \
    rm -rf /libfabric; \
    ldconfig

ARG openmpi_version=5.0.9
RUN set -eux; \
    curl -fsSL "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${openmpi_version}.tar.gz" -o /tmp/ompi.tar.gz; \
    tar -C /tmp -xzf /tmp/ompi.tar.gz; \
    cd "/tmp/openmpi-${openmpi_version}"; \
    ./configure \
      --with-ofi \
      --with-cuda=/usr; \
    make -j"$(nproc)" install; \
    cd /; \
    rm -rf "/tmp/openmpi-${openmpi_version}" /tmp/ompi.tar.gz; \
    ldconfig

# Install uv: https://docs.astral.sh/uv/guides/integration/docker
COPY --from=ghcr.io/astral-sh/uv:0.9.24@sha256:816fdce3387ed2142e37d2e56e1b1b97ccc1ea87731ba199dc8a25c04e4997c5 /uv /uvx /bin/
