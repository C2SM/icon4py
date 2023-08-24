## Running parallel version of diffusion

### Installation

The parallelized code uses [GHEX](https://github.com/ghex-org/GHEX) with MPI for halo exchanges. The GHEX python bindings can be installed with pip from VCS. However it has some system dependencies

1.You need a running MPI installation in the system. On linux (apt base system) do

```bash
sudo apt-get install libopenmpi-dev
```

on MacOs

```bash
brew install mpich
```

2. You need to have boost (headers) installed in the system
3. clone and install `icon4py` as described in the [Readme](../../../README.md) install python dependencies In an existing local `icon4py` clone rerun `pip` install:

```
pip install --src _external_src -r requirements-dev.txt
```

### Run parallel test

The project includes the `pytest-mpi` utility which augments `pytest` with some MPI specific utilities. MPI dependent tests are marked with `@pytest.mark.mpi` and are skipped when not run under mpi (`--with-mpi`) All MPI related tests are in the folder `icon4py/atm_dyn_iconam/tests/mpi_tests` tests depending on MPI are marked with `@pytest.mark.mpi` are therefore skipped by any serial pytest run.

Run the tests with

```bash
mpirun -np 2 pytest -v -s --with-mpi tests/mpi_tests/
```

Test using serialized test data can be run with either 1, 2 or 4 nodes.

When you run the tests for the first time the serialized test data will be downloaded from an online storage. That make some time...
