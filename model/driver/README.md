# Dummy driver for Python ICON port

`dycore_driver.py` contains a simple python program to run the experimental ICON python port. So far code mostly draws on serialized ICON data until we increasingly can initialize and run the model by independently.

It initializes the grid from serialized data from a `mch_ch_r04b09_dsl` run and configures a timeloop functionality based on that configuration.

Currently, there is does _no real timestepping_, instead it calls a dummy timestep that serves a batch of new serialized input fields from ICON.

The code is meant to be changed and enlarged as we port new parts of the model.

It runs single node or parallel versions. For parallel runs the domain has to be decomposed previousely through a full ICON run that generates the necessary serialized data. Test data for runs with 1, 2, 4 nodes are available.

## Installation

See the general instructions in the [README.md](../../README.md) in the base folder of the repository.

## Usage

```bash
export ICON4PY_ROOT=<path to the icon4py clone>
dycore_driver $ICON4PY_ROOT/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data --n_steps=2 --run_path=/home/magdalena/temp/icon
```

or if running in parallel

```bash
mpirun -np 2 dycore_driver $ICON4PY_ROOT/testdata/ser_icondata/mpitask2/mch_ch_r04b09_dsl/ser_data --mpi=True --n_steps=2 --run_path=/home/magdalena/temp/icon
```

#### remarks

- First (required) arg is the folder where the serialized input data is stored. The same input data is for the unit tests is used. The path in the example is where the data is put when downloaded via the unit tests.
  - data for a serial (single node) run can be downloaded from https://polybox.ethz.ch/index.php/s/vcsCYmCFA9Qe26p.
- parallel runs are possible if corresponding data is provided, which is currently available for test with 2 or 4 MPI processes: check [fixtures.py](../common/src/icon4py/model/common/test_utils/fixtures.py) for download urls.
- The serialized data used contains only 5 timesteps so `--n_steps > 2` will throw an exception.
- The code logs to file and to console. Debug logging is only going to file. The log directory can be changed with the --run_path option.
