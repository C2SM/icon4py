## How to use the I/O module

### Pre-requisites and installation

The IO module needs optional dependencies in the `io`-group `io` in the [pyproject.toml](../../../pyproject.toml)
of the `icon4py-driver` package and can be installed with the following command:

```bash
pip install icon4py-driver[io]
```

or by using

```bash
pip install -r requirements-dev-opt.txt
```

which installs all `ICON4Py` packages.

### Concept

The module provides a `IoMonitor` that captures fields from a model state and writes them to files, if called at
the configured output time of the field, upon each call the monitor decides on its own what fields
it needs to write.

Datafiles produced are NETCDF4 files and conform to [CF conventions](https://cfconventions.org/cf-conventions/cf-conventions.html). In addition the monitor writes a copy of the
original ICON grid file enhanced with a [UGRID](https://ugrid-conventions.github.io/ugrid-conventions/) conforming mesh, which is referenced in the datafiles.
This grid file has the same name as the original grid file with the suffix `_ugrid.nc`.

The model state is a dictionary of `xarray.DataArrays` containing the ICON4Py model fields as data buffers and
CF conventional metadata. See examples in [data.py](../../../icon4py/model/driver/io/data.py).

#### Adding fields

When adding new fields to the state the `short_name` should be taken from the [CF standard name table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
or if not available there built up according to [guidelines of CF standard names](http://cfconventions.org/Data/cf-standard-names/docs/guidelines.html).

### Usage

#### Configuration

Configuration is hierarchical for the general IO system and for fields collected in field groups which
are written to the same data file. The Io System can be configured with

- `output_path`: path where all output files will be stored.
- `field_groups`: list of field group configuration
- `time_units` (optional, default is "seconds since 1970-01-01 00:00:00"): unit to used with the time dimension
- `calendar` (optional, default is "proleptic_gregorian"):

Field groups share a common setting of

- `start_time` (optional, default is model init time): a timestamp (string in iso format, for example "2024-01-01T12:00:00") when to start with the output, should be after model init time.
- `output_interval`: string, one of ["DAY", "HOUR", "MINUTE", "SECOND"] (or plural) combined with a number, eg "10 HOURS", "1 DAY", describing how often the fields will be written to file.
- `filename`: file name to be used for the datafile. Files will be appended with a counter for roll over (see `timesteps_per_file`).
- `timesteps_per_file` (default= 10): number of timesteps to be recorded in one file.
- `variables`: list of variables names to be output. Variable names are the `short_name` of the CF conventions or similar names used in the model state.
- `nc_title` (optional): title field of the generated netcdf file.
- `nc_comment` (optional): comment to be put to generated netcdf file.

All fields in the variable list will be written out to file at regular time output intervals starting from the `start_time`. The output
times **must match a model time step**.

As we have no general handling of configuration files in `ICON4Py` yet. It must be instantiated as python dataclasses for now.

```python
prognostic_group = FieldGroupIoConfig(
            start_time="2024-01-01T12:00:00",
            output_interval="2 HOURS",
            filename="icon4py_prognostics",
            timsteps_per_file=12,
            variables=["air_density", "dimensionless_exner_function", "upward_air_velocity"],
            nc_title="prognostics from my experiment",
            nc_comment="Writing prognostic fields data from icon4py ",
        )

wind_group = FieldGroupIoConfig(
            output_interval="1 HOUR",
            filename="icon4py_diagnostics",
            timsteps_per_file=24,
            variables=["eastward_wind","tendency_of_eastward_wind" ],
            nc_comment="Writing additional wind fields data from icon4py",
        )

io_config = IoConfig(
    output_path="simulation1/output",
    field_groups=[prognostic_group, wind_group],
)

io_monitor = IoMonitor(io_config)
(...)
monitor.store(model_state, time)

```

This configuration must then be passed to an instance of the `IoMonitor`, which will decide upon a call to `IoMonitor.store` what fields
need be output from the model state.

#### Restrictions

- No transformation are applied to any output data: Fields are written with the same unstructured grid resolutions as they are computed.
- Horizontal coordinates the latitude and longitude in radians as provided by the ICON grid file.
- Vertical coordinates are the model levels, there is no transformation to pressure levels.
- Parallel writing is not yet implemented.
