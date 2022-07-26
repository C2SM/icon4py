[![Open in Gitpod](https://img.shields.io/badge/Gitpod-ready--to--code-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/C2SM/icon4py)

# ICON4Py

## Description

ICON4Py contains Python (GT4Py) implementations of ICON (inspired) components for weather and climate models.

## Project structure

Each directory contains Python packages of ICON components or utility packages that are deployable on its own. As these packages are not available from a package repository (yet), location of dependencies within this repository have to be provided explicitly, e.g. by installing the dependencies first. See [Installation instructions](#installation-instructions).

## Installation instructions

We recommend to use [tox](https://tox.wiki/en/latest/) for the automatic installation of all packages in development mode in a single step:

```bash
# Clone the repository
git clone git@github.com:C2SM/icon4py.git
cd icon4py

# Use tox to create and set up a development environment (usually at `.venv`) in verbose mode
tox -vv -e dev --devenv .venv

# Activate the virtual environment and check that everything works
source .venv/bin/activate
pytest -v
```

If you want to proceed manually, you should install all packages at once by using the provided `requirements.txt` or `requirements-dev.txt` files in the root of the repository. For example:

```bash
# Clone the repository
git clone git@github.com:C2SM/icon4py.git
cd icon4py

# Create a (Python 3.10) virtual environment (usually at `.venv`)
python3.10 -m venv .venv

# Activate the virtual environment and make sure that 'wheel' is installed
source .venv/bin/activate
pip install --upgrade wheel

# Install all the ICON4Py packages and its dependencies
# External dependencies would be checked out at './_external_src'
pip install --src _external_src -r requirements-dev.txt

# Finally, check that everything works
pytest -v
```

The `--src _external_src` option tells `pip` to use a specific folder as base path for checked out sources, which is very convenient for development tasks involving changes in external dependencies like `gt4py`. For convenience, `./_external_src` has been already added to the repository `.gitignore`.

### Installation of specific subpackages

In case you only want to install a specific subpackage, use the actual subpackage `requirements.txt` or `requirements-dev.txt` files.

For example:

```bash
# Clone the repository
git clone git@github.com:C2SM/icon4py.git
cd icon4py

# Create a (Python 3.10) virtual environment (usually at `.venv`)
python3.10 -m venv .venv

# Activate the virtual environment and make sure that 'wheel' is installed
source .venv/bin/activate
pip install --upgrade wheel

# Install a specific ICON4Py subpackage and its dependencies
cd _SUBPACKAGE_  # where _SUBPACKAGE_ in atm_dyn_iconam | common | pyutils | testutils | ...
pip install -r requirements-dev.txt
```

## Development instructions

After following the installation instructions above using the development requirements (`*-dev.txt` files), an _editable_ installation of all the packages will be active in the virtual environment. In this mode, code changes are immediately visible since source files are imported directly by the Python interpreter.

### Code quality checks

[pre-commit](https://pre-commit.com/) is used to run several linting and checking tools. It should always be executed locally before opening a pull request. `pre-commit` can also be installed as a _git hook_ to automatically check the staged changes before committing:

```bash
# Install pre-commit as a git hook and set up all the tools
pre-commit install --install-hooks
```

Or it can be executed on demand from the command line:

```bash
# Check only the staged changes
pre-commit run

# Check all the files in the repository
pre-commit run -a

# Run only some of the tools (e.g. mypy)
pre-commit run -a mypy
```

### Testing

[pytest](https://pytest.org/) is used for testing and it also comes with a command tool to easily run tests:

```bash
# Run all tests (verbose mode)
pytest -v

# Run only tests in `path/to/test/folder`
pytest -v path/to/test/folder
```

Nonetheless, we also recommended to use `tox` to run the complete test suite:

```bash
# Run test suite in the default environment
tox

# Run test suite in a specific environment (use `tox -a` to see list of envs)
tox -e py310
```

The default `tox` environment is configured to generate HTML test coverage reports in `_reports/coverage_html/`.
