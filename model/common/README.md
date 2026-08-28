# icon4py-common

## Description

Utilities shared by several ICON4Py components.

## Installation instructions

Check the `README.md` at the root of the `model` folder for installation instructions.

## Contents

### IO

module containing IO funcitonality for ICON4Py.

The IO module requires the installation of the `io` optional dependencies defined in [pyproject.toml](./pyproject.toml)
of the `icon4py-common` package and can be installed with the following command:

```bash
uv pip install .[io]
```

or even better by running:

```bash
uv sync --extra io  # or `uv sync --extra all` which includes everything
```

at the top-level folder of the repository, which installs all `ICON4Py` packages including the IO dependencies.

### Distributed run

The package `decomposition` contains infrastructure for parallel implementation of `icon4py/model`.
`icon4py` uses [GHEX](https://github.com/ghex-org/GHEX) for halo exchanges. In order to run in parallel
optional dependencies `mpi4py` and `ghex` need to be installed, which can be done through

```bash
uv sync --extra distributed  # or `uv sync --extra all` which includes everything
```

from the top-level folder of the repository.

### Profiling markers

[profiling.py](src/icon4py/model/common/utils/profiling.py) emits ranges that show up on the
timeline of `nsys` (NVIDIA, via NVTX) or `rocprofv3` (AMD, via ROCTx). The halo exchanges in
`decomposition` are annotated with them: `halo_exchange[dim,nfields]` for a full exchange,
plus `halo_exchange.start`, `halo_exchange.comm` (spanning the whole time the communication is
in flight) and `halo_exchange.wait` for the multi-node implementation.

Markers are off by default and are enabled with `ICON4PY_PROFILE_MARKERS`, which takes
`off` (default), `auto`, `nvtx` or `roctx`. `auto` picks whichever of the two libraries is
importable; naming one explicitly raises if it is missing, so a profiling run cannot silently
record nothing.

The marker libraries are not dependencies of `icon4py-common` and have to be installed
separately:

```bash
# NVIDIA
pip install nvtx
ICON4PY_PROFILE_MARKERS=auto nsys profile --trace=nvtx,cuda <command>

# AMD: `roctx` is not on PyPI, it ships with ROCprofiler-SDK and needs to be put on the path
# explicitly. It is only built for the Python versions selected at ROCm build time, so check
# that it is importable for the Python version used here.
export PYTHONPATH="/opt/rocm/lib/python3.12/site-packages:$PYTHONPATH"
ICON4PY_PROFILE_MARKERS=auto rocprofv3 --marker-trace -- <command>
```

Note that the ranges are recorded on the host: around an exchange scheduled on a stream they
measure the submission, not the device-side execution.

### Grid

Contains basic infrastructure regarding the (unstructured) grid used in `icon4py`. There are
two implementations of the general grid a small simple grid with periodic boundaries in
[simple.py](src/icon4py/model/common/grid/simple.py) used for testing and the
ICON grid [icon.py](src/icon4py/model/common/grid/icon.py) both implement the same protocl.
The ICON grid can be initialized from an ICON grid file via the [grid_manager.py](src/icon4py/model/common/grid/grid_manager.py)
(THIS is still EXPERIMENTAL!!) or from serialized data.
The `grid_manager.py` needs netcdf as an optional dependency, which can be installed with

```bash
uv sync --extra io  # or `uv sync --extra all` which includes everything
```

from the top-level folder of the repository.

### interpolation

Contains interpolation stencils and port of interpolation fields in ICON.

### math

math utilities.

### states

contains type for the ICON prognostic state used by several packages.

### test_utils

Utilities used in tests made available here for usage in other packages
