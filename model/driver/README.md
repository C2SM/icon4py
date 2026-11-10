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
which files is configured in field groups (see [Output](#output)). Optionally,
some ranks are dedicated to output (`io.io_ranks_per_node`): the ranks running the
model ship their fields to them and continue while the IO ranks write (see
[Dedicated IO ranks](#dedicated-io-ranks)).

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
    --icon4py-backend gtfn_cpu
```

The output, if any, is configured in the `io` section of the configuration file
(see "Output"). A distributed (MPI) run only differs in the launcher:

```bash
mpirun -np 4 icon4py-driver \
    --grid-file-path $GRID_FOLDER/icon_grid_0013_R02B04_R.nc \
    --config-file-path $CONFIG_FOLDER \
    --icon4py-backend gtfn_cpu
```

With dedicated IO ranks (`io_ranks_per_node` in the `io` section) the launcher
provides the compute *and* the IO ranks: on every node the last ranks only write.
Here two nodes with four GPUs each run four compute ranks and one IO rank per node,
i.e. ten tasks with `io_ranks_per_node: 1`; with the usual block placement, ranks 4
and 9 write:

```bash
mpirun -np 10 icon4py-driver \
    --grid-file-path $GRID_FOLDER/icon_grid_0013_R02B04_R.nc \
    --config-file-path $CONFIG_FOLDER \
    --icon4py-backend gtfn_cpu
```

Fewer IO ranks than nodes, or IO ranks on a node of their own, are launched as
described in [Dedicated IO ranks](#dedicated-io-ranks).

## Output

Output is configured in the optional top-level `io` section of the configuration
file: the output directory and a list of field groups, each with its own variables,
schedule, file size and format, written into its own file series below
`io.output_path` (see `icon4py.model.common.io` for every option). Without an `io`
section no output is written.

```yaml
io:
  output_path: output
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
  `temperature`, `virtual_temperature`, `pressure` and the purely horizontal
  `surface_pressure`;
- the tracers active in `tracer_config`: `qv`, `qc`, `qi`, `qr`, `qs`, `qg`;
- the precipitation fluxes of the microphysics (`precipitation_flux`,
  `rainfall_flux`, `snowfall_flux`, `graupelfall_flux`, `icefall_flux`,
  `precipitation_energy_flux`), when `muphys` is configured. The physics needs all
  six tracers active. The fluxes are written per model level (the surface value is the
  lowest level), and the initial time slice is zero since the physics has not run yet.

Fields are written with their horizontal dimension (`cell`, `edge` or `vertex`) and,
if they have one, their vertical dimension (model or half levels): a surface field
such as `surface_pressure` is stored as `(time, cell)`.

### Dedicated IO ranks

Some ranks of an MPI run can be dedicated to output: the model runs on the remaining
compute ranks, which ship their output fields to the IO ranks at every capture step
and continue computing while the IO ranks write (through the same backends and
modes, on the IO ranks' communicator; see "Dedicated IO ranks" in
`icon4py.model.common.io`). The launcher provides the tasks of both roles, and the
`io` section says which of them write. There are three ways to say it; the examples
run four compute ranks per node on ten nodes under Slurm (`C` computes, `O` writes).

**IO ranks on every node** (`io_ranks_per_node`, the default way). On every node the
last ranks write for the compute ranks of that node, so no output crosses the
network. Every node needs at least twice as many ranks as IO ranks, so launch one
task per node more than compute ranks:

```yaml
io:
  io_ranks_per_node: 1
```

```bash
# nodes 1-10:  C C C C O
srun --nodes=10 --ntasks-per-node=5 icon4py-driver ...
```

**A total number of IO ranks** (`io_ranks`), for fewer IO ranks than nodes. The IO
ranks are the last ranks of the nodes holding the most ranks, so launch as many tasks
more than compute ranks and let the launcher place them: by default Slurm distributes
the tasks evenly (`SLURM_TASKS_PER_NODE` reads `5(x2),4(x8)`), two nodes get a fifth
task, and these two tasks write. The compute ranks of those two nodes ship to the IO
rank of their node; the other nodes are shared evenly between the two IO ranks:

```yaml
io:
  io_ranks: 2
```

```bash
# nodes 1-2:   C C C C O
# nodes 3-10:  C C C C
srun --nodes=10 --ntasks=42 icon4py-driver ...
```

**IO ranks on a node of their own** (`io_ranks` with `io_rank_placement: last_ranks`).
The last ranks of the run write, wherever they run, so launch the compute tasks first
and the IO tasks last. Here a host file with one line per task, in rank order, puts
the IO tasks on an eleventh node; the compute ranks are shared evenly between them:

```yaml
io:
  io_ranks: 2
  io_rank_placement: last_ranks
```

```bash
# nodes 1-10:  C C C C
# node 11:     O O
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
{
  for node in "${nodes[@]:0:10}"; do yes "$node" | head -n 4; done # compute tasks
  yes "${nodes[10]}" | head -n 2                                   # IO tasks, last
} > hostfile
SLURM_HOSTFILE=hostfile srun --ntasks=42 --distribution=arbitrary icon4py-driver ...
```

In a batch script the task count is the compute ranks plus the IO ranks
(`--ntasks-per-node=5`, or `--nodes=10 --ntasks=42`): the driver takes the tasks the
launcher provides and reads no Slurm variables. Request the GPUs per node
(`--gpus-per-node=4`) rather than per task, since the IO tasks need none, and leave
room for the extra task in `--cpus-per-task`.

`io_ranks_per_node` and `io_ranks` exclude each other, and every IO rank needs at
least one compute rank to serve. IO rank 0 logs the node it runs on and the compute
ranks it serves.

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

- `--log-level`: logging level. Possible values are `notset` (default), `debug`,
  `info`, `warning`, `error`, `critical`.
- `--print-distributed-debug-msg`: print debug logging messages from all MPI
  ranks (only effective when `--log-level debug` is set).

The output has no command-line options: it is configured in the `io` section of the
configuration file only (see "Output"), in single-node and MPI runs alike.
