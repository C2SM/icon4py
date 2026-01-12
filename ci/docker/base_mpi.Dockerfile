FROM ubuntu:25.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    strace \
    build-essential \
    tar \
    wget \
    curl \
    libboost-dev \
    libnuma-dev \
    libopenmpi-dev\
    ca-certificates \
    libssl-dev \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libreadline-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install uv: https://docs.astral.sh/uv/guides/integration/docker
COPY --from=ghcr.io/astral-sh/uv:0.9.24@sha256:816fdce3387ed2142e37d2e56e1b1b97ccc1ea87731ba199dc8a25c04e4997c5 /uv /uvx /bin/
