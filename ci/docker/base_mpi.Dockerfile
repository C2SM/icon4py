FROM ubuntu:25.04

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
        libmpich-dev \
        libnuma-dev \
        libreadline-dev \
        libssl-dev \
        libtool \
        nvidia-cuda-dev \
        nvidia-cuda-toolkit \
        pkg-config \
        strace \
        tar \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Install uv: https://docs.astral.sh/uv/guides/integration/docker
COPY --from=ghcr.io/astral-sh/uv:0.9.24@sha256:816fdce3387ed2142e37d2e56e1b1b97ccc1ea87731ba199dc8a25c04e4997c5 /uv /uvx /bin/
