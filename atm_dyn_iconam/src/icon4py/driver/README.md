# dummy driver for python dycore code

`dycore_driver.py` contains a simple python program to run the experimental ICON python port. The code mostly draws on serialized ICON data. It initializes the grid from serialized data from a `mch_ch_r04b09_dsl` run and configures a timeloop functionality based on that configuration.

Currently, there is does no real timestepping instead it calls a dummy timestep that serves a batch of new serialized input fields from ICON.

The code is meant to be changed and enlarged as we port new parts of the model.

## usage

```bash
cd atm_dyn_iconam/src/icon4py
python driver/dycore_driver.py ../../../testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data --n_steps=2 --run_path=/home/magdalena/temp/icon
```

or if running in parallel

```bash
cd atm_dyn_iconam/src/icon4py
mpirun -np 2 python driver/dycore_driver.py ../../../testdata/ser_icondata/mpitask2/mch_ch_r04b09_dsl/ser_data --mpi=True --n_steps=2 --run_path=/home/magdalena/temp/icon

```

#### remarks

- First (required) arg is the folder where the serialized input data is stored. The same input data s for the unit tests is used. It can be obtained from [polybox](https://polybox.ethz.ch/index.php/s/rzuvPf7p9sM801I/download)
- The driver heavily relies on serialized input data. The paths in the example are where the data is put when downloaded via the unit tests.
  - data for a serial (single node) run can be downloaded from https://polybox.ethz.ch/index.php/s/vcsCYmCFA9Qe26p.
- parallel runs are possible if corresponding data is provided, which is currently available for test with 2 or 4 MPI processes:

  - 2 processes: https://polybox.ethz.ch/index.php/s/NUQjmJcMEoQxFiK
  - 4 processes: https://polybox.ethz.ch/index.php/s/QC7xt7xLT5xeVN5

- The serialized data used contains only 5 timesteps so `--n_steps > 2` will throw an exception.
- The code logs to file and to console. Debug logging is only going to file. The log directory can be changed with the --run_path option.
