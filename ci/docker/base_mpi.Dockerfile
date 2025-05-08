FROM ubuntu:22.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    strace \
    build-essential \
    tar \
    wget \
    curl \
    libopenmpi-dev\
    ca-certificates \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libsqlite3-dev \
    libnuma-dev \
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


# Set environment variables
# Install Boost
RUN wget --quiet https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz && \
    echo be0d91732d5b0cc6fbb275c7939974457e79b54d6f07ce2e3dfdd68bef883b0b boost_1_85_0.tar.gz > boost_hash.txt && \
    sha256sum -c boost_hash.txt && \
    tar xzf boost_1_85_0.tar.gz && \
    mv boost_1_85_0/boost /usr/local/include/ && \
    rm boost_1_85_0.tar.gz boost_hash.txt

ENV BOOST_ROOT /usr/local/

ARG PYVERSION
# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
