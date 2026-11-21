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
some ranks are dedicated to output (`io.io_ranks`): the ranks running the
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

With dedicated IO ranks (`io_ranks` in the `io` section) the launcher provides the
compute *and* the IO ranks; [Dedicated IO ranks](#dedicated-io-ranks) describes how to
launch them.

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
`icon4py.model.common.io`). The launcher provides the tasks of both roles and the
`io` section says which of them write; the driver reads no Slurm variables. So launch
the compute ranks plus the IO ranks, and aim for

- one GPU per compute rank, and cores of the socket that GPU is attached to;
- a few cores per IO rank (it computes nothing on a GPU);
- the same number of compute ranks on every node.

There are two ways to say which ranks write. The examples below were run as shown on
four nodes with four GPUs each (one GPU per 72-core socket, Slurm, Cray MPICH), with
four compute ranks per node (`C` computes, `O` writes).

**IO ranks on every node** (`type: per_node`). On every node the
last ranks write for the compute ranks of that node, so no output crosses the
network. Every node needs at least twice as many ranks as IO ranks; launch one task
per node more than compute ranks:

```yaml
io:
  io_ranks:
    type: per_node
    count: 1
```

```bash
# nodes 1-4:  C C C C O
srun --ntasks-per-node=5 --cpu-bind=sockets ./select_gpu.sh icon4py-driver ...
```

`select_gpu.sh` is a three-line wrapper that gives every task a GPU; its contents are
shown with the batch script below.

**A total number of IO ranks** (`type: total`), for fewer IO ranks than nodes. The IO
ranks are the last ranks of the nodes holding the most ranks, so launch as many tasks
more than compute ranks. Slurm fills the nodes in order and puts the surplus tasks on
the first ones (`SLURM_TASKS_PER_NODE` reads `5(x2),4(x2)` here), and these two tasks
write. The compute ranks of those two nodes ship to the IO rank of their node; the
other nodes are shared evenly between the two IO ranks:

```yaml
io:
  io_ranks:
    type: total
    count: 2
```

```bash
# nodes 1-2:  C C C C O
# nodes 3-4:  C C C C
srun --ntasks=18 --cpu-bind=sockets ./select_gpu.sh icon4py-driver ...
```

A batch script around the first one:

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --gpus-per-node=4

srun --ntasks-per-node=5 --cpu-bind=sockets ./select_gpu.sh icon4py-driver \
  --grid-file-path grid.nc --config-file-path config.yml --icon4py-backend gtfn_gpu
```

and `select_gpu.sh`, an executable file next to it:

```bash
#!/bin/bash
# task i of a node uses GPU i; the fifth task (the IO rank) shares GPU 0
export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID % 4))
exec "$@"
```

What the options are for:

- `--cpu-bind=sockets` binds task `i` of a node to socket `i`, next to GPU `i`, and
  the fifth task (the IO rank) to the first socket again, which it shares with the
  first compute rank. Without a binding option the tasks of a five-task node are not
  bound at all: each may run on all 288 cores, while the tasks of a four-task node get
  a socket each.

- The GPUs are requested per node, since `--gpus-per-task=1` cannot place five tasks
  on four GPUs. Every task then sees all four GPUs, and `select_gpu.sh` gives every
  compute rank its own. The IO rank, a node's fifth task, shares GPU 0. It computes
  nothing there, but a GPU-aware MPI library initializes on the GPU it sees (about
  0.5 GB of device memory); with Cray MPICH and `MPICH_GPU_SUPPORT_ENABLED=1`, a run
  whose IO rank sees no GPU hangs at startup.

The IO ranks are taken from the tasks the launcher provides, whatever their number,
so check the layout the driver logs at startup, the compute and IO ranks of every
node:

```
Dedicated IO ranks: 16 compute rank(s) and 2 IO rank(s) on 4 node(s); compute+IO ranks per node: 4+1 (x2), 4+0 (x2).
```

A launch that does not match the configuration still runs. With a `count` of 3 the 18
tasks above give `3+2, 4+1, 4+0 (x2)`: the third IO rank takes a compute rank from
the first node, the model runs on 15 ranks and one GPU idles. The driver then warns
that the nodes hold unequal numbers of compute ranks. It cannot warn when every node
is short of the same number of tasks: `--ntasks-per-node=4` in the first example gives
`3+1 (x4)`, three compute ranks and an idle GPU on every node, and only the layout
line shows it. IO rank 0 also logs the node it runs on and the compute ranks it
serves, once the compute ranks have initialized.

The IO rank needs few cores, on one socket, and more than one only for zarr output
in many chunks. The output code starts no threads, but the zarr library compresses
and writes the chunks of a capture on a thread pool of its own; netCDF output is
written by HDF5 on a single thread. With output at every step (four compute
ranks shipping about 1 GB per capture to one IO rank, zarr with
`horizontal_chunk_size: 16384`), a capture took the IO rank 2.4 s to write on one
core, 1.0 s on four and 0.8 to 0.9 s on 8 to 72 cores of one socket, but 1.1 s on
cores spread over all four sockets: that is what `--cpus-per-task=57` gives it (the
cores left over on every socket) in exchange for cores no compute rank shares. With
the default chunking, one chunk per rank block, the cores made no difference (3.7 to
4.0 s per capture).

Without `io_ranks` the ranks running the model write themselves. Every IO rank needs
at least one compute rank to serve.

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
