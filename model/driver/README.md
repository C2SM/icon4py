# Standalone driver for ICON4Py

`main.py` contains a simple Python program to run the experimental ICON4Py port.

The driver reads its configuration from a configuration directory, initializes
the grid and model state, and runs the time integration. Which granules are
active (diffusion, solve_nonhydro, tracer advection, microphysics) is determined
by the provided configuration rather than being hardcoded.

It supports both single-node and distributed (MPI) runs, including distributed
output: in MPI runs the ranks write either through the root rank (`gather` mode)
or each into its own block of a shared store (`distributed` mode, the default;
see `icon4py.model.common.io`). Which variables are written, how often and into
which files is configured in field groups (see [Output](#output)).

## Installation

See the general instructions in the [README.md](../../README.md) in the base
folder of the repository.

## Usage

```bash
# set environment variables (optional but convenient)
export ICON4PY_ROOT=<path to the icon4py clone>
export GRID_FOLDER=<path to the folder holding grids>
export CONFIG_FOLDER=<path to the configuration directory>

# command line arguments
icon4py-driver \
    --grid-file-path $GRID_FOLDER/icon_grid_0013_R02B04_R.nc \
    --config-file-path $CONFIG_FOLDER \
    --icon4py-backend gtfn_cpu \
    --output-path $ICON4PY_ROOT/output_path \
    --enable-output
```

A distributed (MPI) run with output only differs in the launcher:

```bash
mpirun -np 4 icon4py-driver \
    --grid-file-path $GRID_FOLDER/icon_grid_0013_R02B04_R.nc \
    --config-file-path $CONFIG_FOLDER \
    --icon4py-backend gtfn_cpu \
    --output-path $ICON4PY_ROOT/output_path \
    --enable-output --output-backend zarr --output-mode distributed
```

## Output

Output is switched on with `enable_output` in the `driver` section of the
configuration file (or `--enable-output` on the command line). What is written is
configured in the optional top-level `io` section: a list of field groups, each
with its own variables, schedule, file size and format, written into its own file
series below `driver.output_path` (see `icon4py.model.common.io` for every
option):

```yaml
driver:
  enable_output: true
  # ...
io:
  field_groups:
    - basename: winds
      variables: [eastward_wind, northward_wind]
      output_interval:
        type: numsteps
        value: 10
      timesteps_per_file: 10
    - basename: precipitation
      variables: [precipitation_flux, rainfall_flux, qv]
      output_interval:
        type: relative
        value: 30 minutes
      timesteps_per_file: 0 # all captures in one file
      backend: netcdf
      mode: gather
```

The variables a run can write are validated at startup. They are

- the prognostics: `air_density`, `exner_function`, `virtual_potential_temperature`,
  `upward_air_velocity`, `normal_velocity`;
- the diagnostics computed for the output: `eastward_wind`, `northward_wind`,
  `temperature`, `virtual_temperature`, `pressure`;
- the tracers active in `tracer_config`: `qv`, `qc`, `qi`, `qr`, `qs`, `qg`;
- the precipitation fluxes of the microphysics (`precipitation_flux`,
  `rainfall_flux`, `snowfall_flux`, `graupelfall_flux`, `icefall_flux`,
  `precipitation_energy_flux`), when `muphys` is configured. The physics needs all
  six tracers active. The fluxes are written per model level (the surface value is the
  lowest level; purely horizontal fields are not supported yet), and the initial time
  slice is zero since the physics has not run yet.

Without an `io` section a single default group (`icon4py_output`, all prognostic and
diagnostic variables, every step) is written in the format and write strategy of
`driver.output_backend` and `driver.output_mode`.

## Configuration directory

The driver expects a configuration directory containing the following JSON files:

- `NAMELIST_ICON_output_atm.json`
- `icon_master.namelist.json`
- `NAMELIST_expname.json`

These are generated from the corresponding Fortran namelists and describe the
experiment, the atmosphere setup, and the input parameters.

To generate these from an experiment run with Fortran ICON you can use the [f90nml](https://f90nml.readthedocs.io/en/latest/index.html) Python package to generate the JSON files from the original `NAMELIST` files of the Fortran ICON simulation.
Once the Fortran ICON simulation has finished, there is a folder generated in `<ICON_ROOT>/<BUILD_TYPE>/experiments/<EXPERIMENT_NAME>` that includes the necessary `NAMELIST` files to configure the `ICON4Py` `driver`.
Using the following instruction you can export the necessary files to their JSON equivalent format:

```
mkdir CONFIG_DIR
f90nml -g config_nml <ICON_ROOT>/<BUILD_TYPE>/experiments/<EXPERIMENT_NAME>/NAMELIST_ICON_output_atm CONFIG_DIR/NAMELIST_ICON_output_atm.json
f90nml -g config_nml <ICON_ROOT>/<BUILD_TYPE>/experiments/<EXPERIMENT_NAME>/icon_master.namelist CONFIG_DIR/icon_master.namelist.json
f90nml -g config_nml <ICON_ROOT>/<BUILD_TYPE>/experiments/<EXPERIMENT_NAME>/NAMELIST_<EXPERIMENT_NAME> CONFIG_DIR/NAMELIST_expname.json
```

Once the above is done you can provide the `CONFIG_DIR` to the `--config-file-path` of the `icon4py-driver` to configure the simulation the same way as the ICON Fortran one.

Of course you can write the necessary configuration files manually or start by some template files and edit them yourself.

### Required options

- `--grid-file-path`: path to the ICON grid file.
- `--config-file-path`: path to the directory containing the configuration JSON
  files.
- `--icon4py-backend`: GT4Py backend for running the driver. Run with
  `--help` to see the available backends.

### Optional options

- `--output-path`: override the output path from the configuration file.
- `--log-level`: logging level. Possible values are `notset` (default), `debug`,
  `info`, `warning`, `error`, `critical`.
- `--print-distributed-debug-msg`: print debug logging messages from all MPI
  ranks (only effective when `--log-level debug` is set).
- `--enable-output/--no-enable-output`: override `driver.enable_output` of the
  configuration file. Works in single-node and MPI runs alike (output is
  collective in MPI runs).
- `--output-backend`: override the file format, `zarr` or `netcdf`, of every
  configured field group (of the default group when there is no `io` section).
- `--output-mode`: override how the ranks of an MPI run write every field group:
  `distributed` (every rank writes its own block of a shared store) or `gather`
  (owned entries are collected and written by the root rank). Distributed
  netCDF requires an MPI-parallel netCDF4 installation (checked at startup, see
  `icon4py.model.common.io`); zarr is parallel with any installation.

Without these options the configuration file decides.
