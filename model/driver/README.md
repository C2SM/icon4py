# Dycore driver for Python ICON port

`dycore_driver.py` contains a simple python program to run the experimental ICON python port. So far the code mostly draws on serialized ICON data until we increasingly can initialize and run the model independently.

It initializes the grid from serialized data from a `mch_ch_r04b09_dsl` run and configures a timeloop functionality based on that configuration.

Currently, it does only diffusion and solve_nonhydro (dry atmosphere with no physics).

The code is meant to be changed and enlarged as we port new parts of the model.

It runs single node or parallel versions. For parallel runs the domain has to be decomposed previously through a full ICON run that generates the necessary serialized data. Test data for runs with 1, 2, 4 nodes are available.

## Installation

See the general instructions in the [README.md](../../README.md) in the base folder of the repository.

## Usage

```bash
export ICON4PY_ROOT=<path to the icon4py clone>
dycore_driver $ICON4PY_ROOT/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data --run_path=$ICON4PY_ROOT/output
```

The driver code runs in parallel, in order to do this you need to install the optional communication libraries with:

```bash
cd ICON4PY_ROOT
pip install -r requirements-dev-opt.txt

```

then run

```bash
mpirun -np 2 dycore_driver $ICON4PY_ROOT/testdata/ser_icondata/mpitask2/mch_ch_r04b09_dsl/ser_data --mpi=True --run_path=$ICON4PY_ROOT/output
```

#### Remarks

- First (required) arg is the folder where the serialized input data is stored. The input data is the same as is used in the unit tests. The path in the example is where the data is put when downloaded via the unit tests.
- data for a serial (single node) run can be downloaded from https://polybox.ethz.ch/index.php/s/vcsCYmCFA9Qe26p.
- parallel runs are possible if corresponding data is provided, which is currently available for test with 2 or 4 MPI processes: check [fixtures.py](../common/src/icon4py/model/common/test_utils/fixtures.py) for download urls.
- The code logs to file and to console. Debug logging is only going to file. The log directory can be changed with the --run_path option.
- The simulation start and end dates, time step, number of substeps, and a logical switch of whether diffusion is run at the initial time step can be configured for the dycore driver. The number of time steps is calculated by (end date - start date) / time step.
