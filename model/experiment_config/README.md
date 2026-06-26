# icon4py-experiment_config

Shared ICON experiment configuration for ICON4Py.

This package holds the configuration types and the Fortran-namelist reader that are
shared between the standalone driver and the testing fixtures:

- `ExperimentConfig` / `DriverConfig` / `ProfilingStats` (`config.py`)
- the initial-condition configuration dataclasses (`initial_condition_config.py`)
- `read_experiment_config(...)`, which assembles an `ExperimentConfig` from a directory
  of serialized Fortran namelists (`reader.py`)

It sits above the atmosphere packages but below both `testing` and `standalone_driver`, so
that neither of those needs to depend on the other.
