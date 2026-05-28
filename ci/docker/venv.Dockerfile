ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY . /icon4py
WORKDIR /icon4py

ARG PYVERSION
ENV UV_CACHE_DIR=/opt/uv-cache
RUN uv sync \
    --no-dev \
    --extra cuda12 \
    --extra fortran \
    --extra io \
    --extra profiling \
    --extra testing \
    --group test \
    --python $PYVERSION && \
    rm -rf .venv && \
    chmod -R a+rwX "$UV_CACHE_DIR"
