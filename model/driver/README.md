# model driver for Python ICON port

`icon4py_driver.py` contains a simple python program to run the experimental ICON python port. So far the code mostly draws on serialized ICON data until we increasingly can initialize and run the model independently.

Currently, users need serialized data of the pre-computed metric and interpolation coefficients and grid to run the driver.

Currently, it does only diffusion and solve_nonhydro (dry atmosphere with no physics). The configuration for the granules and driver is hardcoded in [icon4py_configuration.py](src/icon4py/model/driver/icon4py_configuration.py). Time step, total integration time, number of substeps, and etc. can be configured there. 

The code is meant to be changed and enlarged as we port new parts of the model.

It runs single node or parallel versions. For parallel runs the domain has to be decomposed previously through a full ICON run that generates the necessary serialized data. Test data for runs with 1, 2, 4 nodes are available.

## Installation

See the general instructions in the [README.md](../../README.md) in the base folder of the repository.

## Usage

```bash
export ICON4PY_ROOT=<path to the icon4py clone>
icon4py_driver $ICON4PY_ROOT/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data --run_path=$ICON4PY_ROOT/output
```

The driver code runs in parallel, in order to do this you need to install the optional communication libraries with:

```bash
cd ICON4PY_ROOT
pip install -r requirements-dev-opt.txt

```

then run

```bash
mpirun -np 2 icon4py_driver $ICON4PY_ROOT/testdata/ser_icondata/mpitask2/mch_ch_r04b09_dsl/ser_data --mpi=True --run_path=$ICON4PY_ROOT/output --grid_root=4 --grid_level=9 --experiment_type=any
```

#### Remarks

- First (required) arg is the folder where the serialized input data is stored. The input data is the same as is used in the unit tests. The path in the example is where the data is put when downloaded via the unit tests. As an example, you can use the DATA_URIS in [datatest_utils.py](../common/src/icon4py/model/common/test_utils/datatest_utils.py) to download the serialized data for a serial (single node) run generated from the MeteoSwiss regional experiment. You can also generate your own serialized data and put it in an arbitrary folder.
- Second arg is an option for parallel runs. Parallel runs are possible if corresponding data is provided, which is currently available for test with 2 or 4 MPI processes: check [datatest_fixtures.py](../common/src/icon4py/model/common/test_utils/datatest_fixtures.py) for download urls.
- The code logs to file and to console. Debug logging is only going to file. The log directory can be changed with the --run_path option.
- --grid_root is the root division of the grid. When torus grid is used, it must be set to 2. Please refer to ICON documentation for more information.
- --grid_level is the refinement division of the grid. When torus grid is used, it must be set to 0.
- --experiment_type is an option for configuration and how the initial state is generated. Setting it default value "any" will instruct the model to use the default configuration of MeteoSwiss regional experiment and read the initial state from serialized data.
