ARG BASE_IMAGE
FROM $BASE_IMAGE

# Propagate this as environment variable for use in e.g. gt4py cache directories
ARG BASE_IMAGE
ENV BASE_IMAGE=$BASE_IMAGE

COPY . /icon4py
WORKDIR /icon4py

ARG PYVERSION
ENV UV_PYTHON=${PYVERSION}
ENV UV_MANAGED_PYTHON=1
ENV UV_CACHE_DIR=/opt/uv-cache
ENV MPI4PY_BUILD_BACKEND=scikit-build-core
ENV GHEX_USE_GPU=ON
ENV GHEX_GPU_TYPE=NVIDIA
ENV GHEX_GPU_ARCH=90
ENV GHEX_TRANSPORT_BACKEND=MPI
RUN uv sync \
    --no-dev \
    --extra all \
    --extra cuda12 \
    --group test \
    --python $PYVERSION && \
    chmod -R a+rwX "$UV_CACHE_DIR"
ENV PATH="/icon4py/.venv/bin:$PATH"
