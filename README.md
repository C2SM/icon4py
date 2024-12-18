[![Open in Gitpod](https://img.shields.io/badge/Gitpod-ready--to--code-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/C2SM/icon4py)
[![Nox](https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg)](https://github.com/wntrblm/nox)


# ICON4Py

This repository hosts a work-in-progress Python implementation of the ICON climate and weather model. Additionally, it includes `icon4pytools`, a collection of command-line interfaces (CLIs), and utilities required for the integration of ICON4Py code into the ICON Fortran model. ICON4Py leverages [GT4Py](https://github.com/GridTools/gt4py) to ensure efficient and performance portable implementations of these components.

## Project Structure

The repository is organized as a _monorepo_, where various ICON model components and utilities are developed as independent Python namespace packages in subfolders. An `icon4py` Python package is defined at the root folder just to collect the specific versions of the different components as project dependencies. The component packages can also be installed independently, although since they are not (yet) available from a package repository, they need to be installed from their specific location within this repository. This can be easily done by following the [Installation instructions](#installation-instructions) explained below.

## License

ICON4Py is licensed under the terms of the BSD-3-Clause.

## Installation instructions

Since this project is still in a highly experimental state, it is not yet available as a regular Python distribution project through PyPI. The expected installation procedure is to clone the [https://github.com/C2SM/icon4py](https://github.com/C2SM/icon4py) repository and install it in a venv using the following development workflow.  

### System dependencies

ICON4Py requires **_Python >= 3.10_** and **_boost >= 1.85.0_**, and uses the `uv` tool to manage the development workflow. `uv` is a versatile tool which bundles together functionality from different applications: it can work as a _fast_ Python package manager (like `pip`), as a dependency version exporter (like `pip-tools`), as a Python application runner (like `pipx`) or as a full project development manager (like `hatch`). 
`uv` can be installed in different ways (check its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)), like using the recommended standalone installer:

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh 
```
although it can also be installed with `pip` / `pipx`

```bash
$ pipx install uv
```

### ICON4Py development environment

Once `uv` is installed in your system, it is enough to clone this repository and let `uv` handling the installation of the development environment.

```bash
# Clone the repository
git clone git@github.com:C2SM/icon4py.git
cd icon4py

# Let uv create the development environment at `.venv`.
# The `--extra all` option tells uv to install all optional
# dependencies of icon4py so it is not strictly necessary
uv sync --extra all

# Activate the virtual environment and start writing code!
source .venv/bin/activate
```

The new `venv` is a standard Python virtual environment with all the necessary run-time and development dependencies, where all the icon4py subpackages have been already installed in editable mode. New packages can be installed with the `uv pip` subcommand which emulates the `pip` interface or even using the regular `pip` which is also available in the venv (although it's usually much slower and not really recommended).   

The `pyproject.toml` file at the root folder contains both the definition of the `icon4py` Python distribution package and the settings of the development tools used in this project, most notably `uv`, `ruff`, `mypy` and `pytest`. It also contains _dependency groups_ (see [PEP 735](https://peps.python.org/pep-0735/) for further reference) with the development requirements listed in different groups (`build`, `docs`, `lint`, `test`, `typing`, ...) and collected together in the general `dev` group which gets installed by default by `uv`.


## Development instructions

By following the installation instructions above, the source files are imported directly by the Python interpreter meaning that any code change is available and executed by the interpreter.

To add new dependencies to the project, either core/optional run-time or development-only dependencies, it is possible to use the `uv` cli direcly or to modify by hand the appropriate tables in the corresponding `pyproject.toml` (check `uv` documentation for more information [https://docs.astral.sh/uv/concepts/projects/dependencies/](https://docs.astral.sh/uv/concepts/projects/dependencies/)).


### Code quality checks

[pre-commit](https://pre-commit.com/) is used to run several linting and checking tools. It should always be executed locally before opening a pull request. When executing pre-commit locally you can either run it for the `model` or `tools` folder:

For example to run code checks on all components in `icon4py.model` you can do:

```bash
# running precommit for all components in model
cd model/
pre-commit run --all-files
```

### Testing

[pytest](https://pytest.org/) is used for testing and it also comes with a command tool to easily run tests:

```bash
# Run all tests (verbose mode)
pytest -v

# Run only tests in `path/to/test/folder`
pytest -v path/to/test/folder
```

Nonetheless, we also recommended to use `nox` to run the parametrized test suite:

```bash
# List all available test sessions (colored items are the default sessions)
nox -l

# Run all parametrized cases of a session
nox -s 'test_common'

# Run a test session for a specific python version and parameter values
nox -s 'test_atmosphere_advection-3.10(datatest=True)'
```

The default `tox` environment is configured to generate HTML test coverage reports in `_reports/coverage_html/`.

### Benchmarking

We use [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/) to benchmark the execution time of stencils in icon4py. To disable benchmarking during testing you can use `--benchmark-disable` when invoking `pytest`.

### Documentation

The documentation is at a very early stage given the constant state of development. Some effort is ongoing to document the dycore and can be compiled as follows.

You can install the required packages by using the provided `requirements-dev.txt` file in the root of the repository.

Then move to the dycore docs folder and build the html documentation with the provided makefile:

```bash
cd model/atmosphere/dycore/docs
make html
```

The documentation can then be accessed at `docs/_build/html/index.html`

### More Information

For more information please consult the package specific READMEs found in the `model` and `tools` folders.
