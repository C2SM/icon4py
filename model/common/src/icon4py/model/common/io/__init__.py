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
- `basename`: Base name of the datafiles, without an extension (the backend's extension and a roll-over counter are appended, see `timesteps_per_file`); it may contain a _relative_ path which is appended to the `output_path`.
- `timesteps_per_file` (default=10): Number of timesteps to be recorded in one file, if the value is negative all captured times go into the same file.
- `variables`: List of variables names to be output. Variable names are the CF names used as keys in the model state (see [data.py](../states/data.py)).
- `backend` (default="zarr"): File format of the group, `"netcdf"` or `"zarr"`.
- `mode` (default="distributed"): Write strategy of distributed (MPI) runs; single-rank runs write the full state either way:
    - `"gather"`: the owned entries of all ranks are collected on the root rank, which writes them in global order.
    - `"distributed"`: every rank writes its owned entries into a rank-contiguous block of a shared store (see `io.distributed`). The horizontal axes of such a store are rank-ordered and padded, so its data variables carry the marker attribute `icon4py_layout = "rank_block"` instead of a UGRID mesh association: a consumer must reorder them by the store's `global_index_<dim>` coordinates before the UGRID mesh applies. With the `"netcdf"` backend, multi-rank runs need an MPI-parallel netCDF4 installation (checked when the writer is created, see "Parallel netCDF" below).
- `asynchronous` (optional): Whether the group's data writes run on a background thread (one per group), overlapping the file output with the model computation: `store` performs the store-metadata operations (and any communication) and queues the data write. Supported by the `"zarr"` backend only; the default (unconfigured) is asynchronous exactly for the backends that support it. At most `writers.MAX_PENDING_WRITES` captures are queued, then `store` blocks until the writer catches up (phase `"async_wait"` of the timing report -- the signal that writing does not keep up with the model).
- `horizontal_chunk_size` (optional): Entries per chunk along the horizontal (cell/edge/vertex) axes. Default: one chunk per rank block in `"distributed"` mode; otherwise the whole axis (zarr) or the library default (netcdf). In `"distributed"` mode the rank-block size is rounded up to a multiple of this value so chunks never cross rank-block boundaries.
- `horizontal_shard_size` (optional, `"zarr"` backend only): Entries per shard along the horizontal axes; must be a multiple of `horizontal_chunk_size`. A shard groups whole chunks into a single storage file, so this controls the number of files written per time slice -- the critical tuning knob on parallel file systems. Default: no sharding (one file per chunk). In `"distributed"` mode the rank-block size is rounded up to the shard size instead.
- `nc_title` (optional): Title attribute of the generated files (netcdf and zarr).
- `nc_comment` (optional): Comment attribute of the generated files (netcdf and zarr).

The configuration is instantiated as Python dataclasses; `backend` and `mode` take enum
members, their value strings are only converted at the config-file boundary
(`common.config.config_io`, where the IO enums are registered). A valid configuration
could look like this:

```python
import datetime

prognostic_group = FieldGroupIOConfig(
    output_interval=datetime.timedelta(hours=2),
    basename="icon4py_prognostics",
    timesteps_per_file=12,
    variables=["air_density", "exner_function", "upward_air_velocity"],
    nc_title="prognostics from my experiment",
    nc_comment="Writing prognostic fields data from icon4py ",
)

wind_group = FieldGroupIOConfig(
    output_interval=1,
    basename="icon4py_diagnostics",
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
  in multi-rank runs, when the writer is created (single-rank runs write through a
  serial file handle and never need parallel support).
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
- **Runtime checks**: the installation is verified when a parallel writer is
  constructed and again when the shared file is opened
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
- Zarr groups write asynchronously by default (see `asynchronous` above): `store` returns once the data write is queued, and `IOMonitor.close` blocks until every queued write is on disk. netCDF groups write synchronously: `store` returns once the data is written.
- Distributed (MPI) runs write either gathered on the root rank (any backend) or with every rank writing its own block of a shared store (zarr with any installation; netCDF with an MPI-parallel netCDF4 installation, see "Parallel netCDF" above).
- Global attributes of the datafiles and field metadata is only scarcely available and needs to be augmented.

"""

import importlib.util


if not importlib.util.find_spec("xarray"):
    raise RuntimeError("Optional icon4py-common[io] dependencies are missing!")
