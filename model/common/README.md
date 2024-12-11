# icon4py-common

## Description

Utilities shared by several ICON4Py components.

## Installation instructions

Check the `README.md` at the root of the `model` folder for installation instructions.

## Contents

### IO

module containing IO funcitonality for ICON4Py.

The IO module requires the installation of the `io` optional dependencies defined in [pyproject.toml](../../../pyproject.toml)
of the `icon4py-common` package and can be installed with the following command:

```bash
pip install icon4py-common[io]
```

or by using

```bash
pip install -r requirements-dev.txt
```

which installs all `ICON4Py` packages including the IO dependencies.

### decomposition

Contains infrastructure for parallel implementation of `icon4py/model`.
`icon4py` uses [GHEX](https://github.com/ghex-org/GHEX) for halo exchanges. In order to run in parallel
optional dependencies `mpi4py` and `ghex` need to be installed, which can be done through

```bash
pip install -r requirements-dev-opt.txt
```

### grid

Contains basic infrastructure regarding the (unstructured) grid used in `icon4py`. There are
two implementations of the general grid a small simple grid with periodic boundaries in
[simple.py](src/icon4py/model/common/grid/simple.py) used for testing and the
ICON grid [icon.py](src/icon4py/model/common/grid/icon.py) both implement the same protocl.
The ICON grid can be initialized from an ICON grid file via the [grid_manager.py](src/icon4py/model/common/grid/grid_manager.py)
(THIS is still EXPERIMENTAL!!) or from serialized data.
The `grid_manager.py` needs netcdf as an optional dependency, which can be installed with

```bash
pip install -r requirements-dev-opt.txt
```

### interpolation

Contains interpolation stencils and port of interpolation fields in ICON.

### math

math utilities.

### states

contains type for the ICON prognostic state used by several packages.

### test_utils

Utilities used in tests made available here for usage in other packages