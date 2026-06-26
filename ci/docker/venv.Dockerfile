ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY . /icon4py
WORKDIR /icon4py

ARG PYVERSION
ARG ICON4PY_NOX_UV_CUSTOM_SESSION_EXTRAS
ENV UV_CACHE_DIR=/opt/uv-cache
ENV MPI4PY_BUILD_BACKEND=scikit-build-core
ENV GHEX_USE_GPU=ON
ENV GHEX_GPU_TYPE=NVIDIA
ENV GHEX_GPU_ARCH=90
ENV GHEX_TRANSPORT_BACKEND=MPI
RUN uv sync \
    --no-dev \
    --extra all \
    --extra $ICON4PY_NOX_UV_CUSTOM_SESSION_EXTRAS \
    --group test \
    --python $PYVERSION && \
    chmod -R a+rwX "$UV_CACHE_DIR"
# Remove everything except the .venv directory. icon4py is copied again in the
# checkout.Dockerfile stage.
RUN find /icon4py -mindepth 1 -maxdepth 1 -not -name '.venv' -exec rm -rf {} +
ENV PATH="/icon4py/.venv/bin:$PATH"

# Propagate base image info so child images (checkout) and cache scripts can use it
ARG BASE_IMAGE
ENV BASE_IMAGE=$BASE_IMAGE
