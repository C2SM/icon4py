# dummy driver for python dycore code

`dycore_driver.py` contains a simple python program to run the experimental ICON python port. The code mostly draws on serialized ICON data. It initializes the grid from serialized data from a `mch_ch_r04b09_dsl` run and configures a timeloop functionality based on that configuration.

Currently, there is does no real timestepping instead it calls a dummy timestep that serves a batch of new serialized input fields from ICON.

The code is meant to be changed and enlarged as we port new parts of the model.

## usage

```
cd atm_dyn_iconam/src/icon4py
python driver/dycore_driver.py ../../../testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data --n_steps=2 --run_path=/home/magdalena/temp/icon
```

#### remarks

- first (required) arg is the folder where the serialized input data is stored. The same input data s for the unit tests is used. It can be obtained from [polybox](https://polybox.ethz.ch/index.php/s/rzuvPf7p9sM801I/download)
- the serialized data used contains only 5 timesteps so `--n_steps > 2` will throw an exception.
- the code logs to file and to console. Debug logging is only going to file. The log directory can be changed with the --run_path option.
