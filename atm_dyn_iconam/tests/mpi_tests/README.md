## running parallel version of diffusion

### installation

The parallelized code uses [GHEX](https://github.com/ghex-org/GHEX) with MPI for halo exchanges. GHEX has a CMake build but no setup script for pip, so it needs to be installed manually:

1. You need a running MPI installation in the system.
2. You need to have boost (headers) installed in the system 3clone GHEX

```bash
cd {icon4py}/_external_src
git clone --recursive -b refactoring2 git@github.com:boeschf/GHEX.git
```

3. build GHEX

```
cd GHEX
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug \
-DGHEX_GIT_SUBMODULE=OFF \
-DGHEX_USE_BUNDLED_LIBS=ON \
-DGHEX_USE_BUNDLED_GRIDTOOLS=ON \
-DBUILD_TESTING=OFF \
-DGT_BUILD_TESTING=OFF \
-DGT_INSTALL_EXAMPLES=OFF \
-DGHEX_USE_BUNDLED_OOMPH=ON \
-DGHEX_TRANSPORT_BACKEND=MPI \
-DGHEX_USE_XPMEM=OFF \
-DGHEX_WITH_TESTING=ON \
-DGHEX_BUILD_PYTHON_BINDINGS=ON \
-DGHEX_BUILD_BENCHMARKS=ON \
-DGHEX_BUILD_FORTRAN=ON \
-DGHEX_ENABLE_ATLAS_BINDINGS=OFF \
-DGHEX_ENABLE_PARMETIS_BINDINGS=ON \
-DMETIS_INCLUDE_DIR=/opt/metis/include \
-DMETIS_LIB_DIR=/opt/metis/lib \
-DPARMETIS_INCLUDE_DIR=/opt/parmetis/include \
-DPARMETIS_LIB_DIR=/opt/parmetis/lib \
-DMPIEXEC_PREFLAGS=--oversubscribe \
-DGHEX_USE_GPU=OFF
make
make test  ## runs the C++ tests
```

#### generating python bindings

```bash
cmake -DGHEX_BUILD_PYTHON_BINDINGS=ON .  # turns on python bindings, need pybind11 wo we install it in the
```

builds GHEX including python bindings. If the build fails with an error message that `pybind11` was not found, you can either install it int the Python `.venv` in GHEX build folder or set the `pybind11_DIR` variable to some other location.

```bash
cd pyghex_venv/
source bin/activate
pip install pybind11
export pybind11_DIR=./pyghex_venv/lib/python3.10/site-packages/pybind11
make
make test ## will now run the python tests
```

#### use GHEX bindings from icon4py

simply create a sym link the the installation above:

```
cd {icon4py}/.venv/lib/python3.10/site-packages
ln -s ../../../../_external_src/GHEX/build/bindings/python/ghex ghex
```

### run parallel test

all MPI related tests are in the folder `icon4py/atm_dyn_iconam/tests/mpi_tests` tests depending on MPI are marked with `@pytest.mark.mpi` are therefore skipped by any serial pytest run.

run them with

```bash
mpirun -np 2 pytest -v -s --with-mpi tests/mpi_tests/
```
