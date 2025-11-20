# model driver for Python ICON port

`run_driver.py` contains a simple python program to run the experimental ICON python port.

Currently, it does only diffusion and solve_nonhydro (dry atmosphere with no physics). The configuration for the granules and driver is hardcoded in [standalone_driver.py](src/icon4py/model/standalone_driver/standalone_driver.py). Time step, total integration time, number of substeps, and etc. can be configured there.

The code is meant to be changed and enlarged as we port new parts of the model.

It runs single node.

## Installation

See the general instructions in the [README.md](../../README.md) in the base folder of the repository.

## Usage

```bash
export ICON4PY_ROOT=<path to the icon4py clone>
run_driver ICON4PY_ROOT/configuration_path/ ICON4PY_ROOT/output_path --grid-file-path /scratch/mch/cong/grid-generator/grids/icon_grid_0013_R02B04_R.nc --icon4py-backend gtfn_cpu
```

#### Remarks
