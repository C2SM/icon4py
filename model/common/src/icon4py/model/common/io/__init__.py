# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Module containing the IO functionality for ICON4Py.

### General concept

The module provides an `IOMonitor` that captures fields from the model state and writes them to a file
if called at the configured output time of the field. Upon each call the monitor decides on its own
what fields it needs to write.

The Datafiles produced are NETCDF4 files or zarr stores (per field group, see `backend`
below) and conform to
[CF conventions](https://cfconventions.org/cf-conventions/cf-conventions.html).
In addition, upon start-up the monitor writes a copy of the original ICON grid file enhanced with a
[UGRID](https://ugrid-conventions.github.io/ugrid-conventions/) conforming mesh, which is referenced
in the datafiles. This grid file has the same name as the original grid file with the suffix `_ugrid.nc`.

The model state is a dictionary of `xarray.DataArrays` containing the ICON4Py fields as data buffers and
CF conventional metadata. For some basic examples see [data.py](../states/data.py).

#### Adding fields

When adding new fields to the state the `short_name` should be taken from the
[CF standard name table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
or, if not available there, built up according to [guidelines of CF standard names](http://cfconventions.org/Data/cf-standard-names/docs/guidelines.html).

### Usage

#### Configuration

The IO module is configurable and can be configured with:

- `output_path`: path where all output files will be stored.
- `field_groups`: list of field group configuration (see below).
- `time_units` (optional, default is "seconds since 1970-01-01 00:00:00"): unit used with the time dimension in the data files.
- `calendar` (optional, default is "proleptic_gregorian"). Caleandar used with the time dimension in the data files.

Field groups are stored in the same file and share a common setting of

- `output_interval`: the output schedule, given as either a positive integer N (write every N model steps, i.e. every N calls to `store`) or a `datetime.timedelta` (a simulation-time delta, e.g. `timedelta(hours=2)`, must be a multiple of the model time step). A time delta is normalized to a number of steps using the model time step, so the schedule is always evaluated in steps. Defaults to every step.
- `filename`: File name to be used for the datafile, it may contain a _relative_ path which is appended to the `output_path` . Files will be appended with a counter for roll over (see `timesteps_per_file`).
- `timesteps_per_file` (default=10): Number of timesteps to be recorded in one file, if the value is negative all captured times go into the same file.
- `variables`: List of variables names to be output. Variable names are the CF names used as keys in the model state (see [data.py](../states/data.py)).
- `backend` (default="zarr"): File format of the group, `"netcdf"` or `"zarr"`.
- `mode` (default="distributed"): Write strategy of distributed (MPI) runs: `"gather"` collects all fields on the root rank which writes them in global order; `"distributed"` lets every rank write its owned entries into a rank-contiguous block of a shared store (see `io.distributed`). Single-rank runs write the full state either way, but `"distributed"` with the `"netcdf"` backend requires an MPI-parallel netCDF4 installation and is rejected at configuration time on serial installations regardless of the rank count (see "Parallel netCDF" below).
- `horizontal_chunk_size` (optional): Entries per chunk along the horizontal (cell/edge/vertex) axes. Default: one chunk per rank block in `"distributed"` mode; otherwise the whole axis (zarr) or the library default (netcdf). In `"distributed"` mode the rank-block size is rounded up to a multiple of this value so chunks never cross rank-block boundaries.
- `horizontal_shard_size` (optional, `"zarr"` backend only): Entries per shard along the horizontal axes; must be a multiple of `horizontal_chunk_size`. A shard groups whole chunks into a single storage file, so this controls the number of files written per time slice -- the critical tuning knob on parallel file systems. Default: no sharding (one file per chunk). In `"distributed"` mode the rank-block size is rounded up to the shard size instead.
- `nc_title` (optional): Title attribute of the generated files (netcdf and zarr).
- `nc_comment` (optional): Comment attribute of the generated files (netcdf and zarr).

As we have no general handling of configuration files in `ICON4Py` yet, the configuration needs to
be instantiated as Python dataclasses for now. A valid configuration could look like this:

```python
import datetime

prognostic_group = FieldGroupIOConfig(
    output_interval=datetime.timedelta(hours=2),
    filename="icon4py_prognostics",
    timesteps_per_file=12,
    variables=["air_density", "exner_function", "upward_air_velocity"],
    nc_title="prognostics from my experiment",
    nc_comment="Writing prognostic fields data from icon4py ",
)

wind_group = FieldGroupIOConfig(
    output_interval=1,
    filename="icon4py_diagnostics",
    timesteps_per_file=24,
    variables=["eastward_wind", "northward_wind"],
    nc_comment="Writing additional wind fields data from icon4py",
)

io_config = IOConfig(
    output_path="simulation1/output",
    field_groups=[prognostic_group, wind_group],
)
```

This configuration must then be passed to an instance of the `IOMonitor`, which will decide upon a
call to `IOMonitor.store` what fields of the model state need to be written at that time:

```
io_monitor = IOMonitor(io_config)
(...)
monitor.store(model_state, time)
```

### Parallel netCDF

`mode="distributed"` with the `"netcdf"` backend means every rank writes its block of one
shared netCDF file, which the installed `netCDF4` package must support:

- **The PyPI wheels are serial builds.** `pip install netcdf4` (and therefore the icon4py
  dependencies) ships `netCDF4` compiled against serial netCDF-C/HDF5 libraries --
  MPI-parallel HDF5 cannot be distributed as a portable wheel. On such an installation
  `netCDF4.__has_parallel4_support__` is `0` and distributed netCDF output is rejected
  when the configuration is validated.
- **Steps to enable the path**:
  1. Provide MPI-enabled netCDF-C and HDF5 libraries: HPC environment modules or spack
     (e.g. `spack install netcdf-c+mpi ^hdf5+mpi`), or on conda the
     `conda-forge::netcdf4=*=mpi_*` builds (which replace steps 2-3).
  2. In the icon4py environment, install `mpi4py` and the build requirements
     (`Cython`, `numpy`, `setuptools`, `setuptools_scm`), then build the Python
     package against the parallel libraries:
     `HDF5_DIR=<hdf5-prefix> NETCDF4_DIR=<netcdf-c-prefix>
     pip install --no-binary netcdf4 --no-build-isolation --force-reinstall netcdf4`.
     `mpi4py` must be importable while the package builds -- with pip's default build
     isolation the build runs in an environment without it and aborts, hence
     `--no-build-isolation`.
  3. Verify: `python -c "import netCDF4; print(netCDF4.__has_parallel4_support__)"`
     must print `1`.
- **Runtime checks**: the installation is verified at configuration validation, again
  when a parallel writer is constructed and finally when the shared file is opened
  (`netcdf_writers.missing_parallel_support`); every parallel open logs the netCDF4,
  netCDF-C and HDF5 versions in use. A wrong installation therefore fails loudly with
  the steps above -- never by writing corrupt files.
- **Without a parallel installation**: use the `"zarr"` backend (parallel with any
  installation) or `mode="gather"`.

### Restrictions

- We support NETCDF4 files and zarr stores (zarr format 3).
- No transformation are applied to any output data: Fields are written with the same unstructured grid resolutions as they are computed.
- Horizontal coordinates the latitude and longitude in radians as provided by the ICON grid file.
- Vertical coordinates are the model levels, there is no transformation to pressure levels.
- Writing is synchronous: `store` returns once the data is written.
- Distributed (MPI) runs write either gathered on the root rank (any backend) or with every rank writing its own block of a shared store (zarr with any installation; netCDF with an MPI-parallel netCDF4 installation, see "Parallel netCDF" above).
- Global attributes of the datafiles and field metadata is only scarcely available and needs to be augmented.

"""

import importlib.util


if not importlib.util.find_spec("xarray"):
    raise RuntimeError("Optional icon4py-common[io] dependencies are missing!")
