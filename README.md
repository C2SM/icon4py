[![Open in Gitpod](https://img.shields.io/badge/Gitpod-ready--to--code-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/C2SM/icon4py)
[![Nox](https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg)](https://github.com/wntrblm/nox)
[![Open the docs](https://readthedocs.org/projects/docs/badge/?version=latest)](https://c2sm.github.io/icon4py)


# ICON4Py

This repository hosts a work-in-progress Python implementation of the ICON climate and weather model. Additionally, it includes `icon4py.tools`, a collection of command-line interfaces (CLIs), and utilities required for the integration of ICON4Py code into the ICON Fortran model. ICON4Py leverages [GT4Py](https://github.com/GridTools/gt4py) to ensure efficient and performance portable implementations of these components.

## Project Structure

The repository is organized as a _monorepo_, where various ICON model components and utilities are developed as independent Python namespace packages in subfolders. An `icon4py` Python package is defined at the root folder with the purpose to collect specific versions of the different components as package dependencies. The component can also be installed independently, although since they are not (yet) available from a package repository, they need to be installed from their specific location within this repository.

## License

ICON4Py is licensed under the terms of the BSD-3-Clause.

## Installation Instructions

Since this project is still in a highly experimental state, it is not yet available as a regular Python distribution package through PyPI. The installation procedure involves cloning the [ICON4Py GitHub repository](https://github.com/C2SM/icon4py) and installing it in a Python virtual environment (_venv_).

ICON4Py uses the [`uv`](https://docs.astral.sh/uv/) project manager for development workflow. `uv` is a versatile tool that consolidates functionality previously distributed across different applications into subcommands.

- The `uv pip` subcommand provides a _fast_ Python package manager, emulating [`pip`](https://pip.pypa.io/en/stable/).
- The `uv export | lock | sync` subcommands manage dependency versions in a manner similar to the [`pip-tools`](https://pip-tools.readthedocs.io/en/stable/) command suite.
- The `uv init | add | remove | build | publish | ...` subcommands facilitate project development workflows, akin to [`hatch`](https://hatch.pypa.io/latest/).
- The `uv tool` subcommand serves as a runner for Python applications in isolation, similar to [`pipx`](https://pipx.pypa.io/stable/).
- The `uv python` subcommands manage different Python installations and versions, much like [`pyenv`](https://github.com/pyenv/pyenv).

`uv` can be installed in various ways (see its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)), with the recommended method being the standalone installer:

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh 
```

Finally, make sure **_boost >= 1.85.0_** is installed in your system, which is required by `gt4py` to compile generated C++ code. 

### ICON4Py Development Environment

Once `uv` is installed in your system, it is enough to clone this repository and let `uv` handling the installation of the development environment. 

**Important**: the `uv sync` command should always be executed from the **root folder** of the repository, to make sure it installs all the workspace dependencies and not only the dependencies of a subproject. 

```bash
# Clone the repository
git clone git@github.com:C2SM/icon4py.git
cd icon4py

# Let uv create the development environment at `.venv`.
# The `--extra all` option tells uv to install all the optional
# dependencies of icon4py, and thus it is not strictly necessary.
# Note that if no dependency groups are provided as an option,
# uv uses `--group dev` by default so the development dependencies
# are also installed. 
uv sync --extra all

# Activate the virtual environment and start writing code!
source .venv/bin/activate
```

The new _venv_ is a standard Python virtual environment preconfigured with all necessary runtime and development dependencies. Additionally, all icon4py subpackages are installed in editable mode, allowing for seamless development and testing.

To install new packages, use the `uv pip` subcommand, which emulates the `pip` interface and is generally much faster. Alternatively, the standard `pip` command is also available within the venv, although using `pip` directly is slower and not recommended.

The `pyproject.toml` file at the root folder contains both the definition of the `icon4py` Python distribution package and the settings of the development tools used in this project, most notably `uv`, `ruff`, `mypy` and `pytest`. It also contains _dependency groups_ (see [PEP 735](https://peps.python.org/pep-0735/) for further reference) with the development requirements listed in different groups (`build`, `docs`, `lint`, `test`, `typing`, ...) and collected together in the general `dev` group which gets installed by default by `uv`.


## Development Instructions

By following the installation instructions above, the source files are imported directly by the Python interpreter meaning that any code change is available and executed by the interpreter.

To add new dependencies to the project, either core/optional run-time or development-only dependencies, it is possible to use the `uv` cli direcly or to modify by hand the appropriate tables in the corresponding `pyproject.toml` (check `uv` documentation for more information [https://docs.astral.sh/uv/concepts/projects/dependencies/](https://docs.astral.sh/uv/concepts/projects/dependencies/)).


### Code Quality Checks

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

`nox` is recommended for running comprehensive test suites across multiple Python versions and configurations, mirroring the setup used in the CI pipeline.

```bash
# List all available test sessions (colored items are the default sessions)
nox -l

# Run all parametrized cases of a session
nox -s 'test_common'

# Run a test session for a specific python version and parameter value
nox -s 'test_atmosphere_advection-3.10(datatest=True)'
```

### Benchmarking

We use [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/) to benchmark the execution time of stencils in icon4py. To disable benchmarking during testing you can use `--benchmark-disable` when invoking `pytest`.

### Documentation

Documentation is at a very early stage given the constant state of development.
Ongoing efforts to document the dycore can be viewed at [c2sm.github.io/icon4py](https://c2sm.github.io/icon4py).

You can install the required packages by using the provided `docs` dependency group, which is included in the `dev` installed by default by `uv sync` if no dependency groups are specified.

To build the html documentation in your local repository copy starting from the most upwards directory, apply the following commands: 

```bash
cd model/atmosphere/dycore/docs
make html
```

The local documentation could then be accessed at `docs/_build/html/index.html`

### More Information

For more information please consult the package specific READMEs found in the `model` and `tools` folders.
