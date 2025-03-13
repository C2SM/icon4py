ARG BASE_IMAGE
ARG PYVERSION
ARG VENV
FROM $BASE_IMAGE

COPY . /icon4py
ENV MPI4PY_BUILD_BACKEND="scikit-build-core"
# DO WE NEED: MPI4PY_BUILD_MPICC: nvc or si the setting the build backend enough?
ENV USE_MPI="YES"
WORKDIR /icon4py
RUN uv sync --extra distributed --python=$PYVERSION --no-dev --venv=$VENV

