## I/O module

### Pre-requisites and installation

The IO module needs optional dependencies in the `io`-group `io` in [pyproject.toml](../../../pyproject.toml)
of the `icon4py-driver` package and can be installed with the following command:

```bash
pip install icon4py-driver[io]
```

or by using

```bash
pip install -r requirements-dev.txt
```

which installs all `ICON4Py` packages.

### General concept

The module provides a `IoMonitor` that captures fields from the model state and writes them to file 
if called at the configured output time of the field. Upon each call the monitor decides on its own 
what fields it needs to write.

The Datafiles produced are NETCDF4 files and conform to 
[CF conventions](https://cfconventions.org/cf-conventions/cf-conventions.html). 
In addition, upon start-up the monitor writes a copy of the original ICON grid file enhanced with a 
[UGRID](https://ugrid-conventions.github.io/ugrid-conventions/) conforming mesh, which is referenced
in the datafiles. This grid file has the same name as the original grid file with the suffix `_ugrid.nc`.

The model state is a dictionary of `xarray.DataArrays` containing the ICON4Py fields as data buffers and
CF conventional metadata. For some basic examples see [data.py](../../../icon4py/model/driver/io/data.py).

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

- `start_time` (optional, default is model init time): A timestamp (string in iso format, for example "2024-01-01T12:00:00") when to start with the output, should be after model init time.
- `output_interval`: String, one of ["DAY", "HOUR", "MINUTE", "SECOND"] (or plural) combined with a positive number, eg "10 HOURS", "1 DAY", describing how often the fields will be written to file.
- `filename`: File name to be used for the datafile, it may contain a path which is appended to the `output_path` . Files will be appended with a counter for roll over (see `timesteps_per_file`).
- `timesteps_per_file` (default=10): Number of timesteps to be recorded in one file, if the value is negative all captured times go into the same file.
- `variables`: List of variables names to be output. Variable names are the `short_name` of the CF conventions used in the model state.
- `nc_title` (optional): Title field of the generated netcdf file.
- `nc_comment` (optional): Comment to be put to generated netcdf file.

All fields in the `variables` list will be written out to the same file at regular 
`output_intervals` starting from the `start_time`. The output times **must exactly match a model time step**.

As we have no general handling of configuration files in `ICON4Py` yet,  the configuration needs to 
be instantiated as Python dataclasses for now. A valid configuration could look like this:



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
```
This configuration must then be passed to an instance of the `IoMonitor`, which will decide upon a 
call to `IoMonitor.store` what fields of the model state need to be written at that time:

```
io_monitor = IoMonitor(io_config)
(...)
monitor.store(model_state, time)
```



### Restrictions
- We only support NETCDF4 files.
- No transformation are applied to any output data: Fields are written with the same unstructured grid resolutions as they are computed.
- Horizontal coordinates the latitude and longitude in radians as provided by the ICON grid file.
- Vertical coordinates are the model levels, there is no transformation to pressure levels.
- Parallel writing is not yet implemented.
- Global attributes of the datafiles and field metadata is only scarcely available and needs to be augmented. 
