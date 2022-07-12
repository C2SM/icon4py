# ICON4Py

## Description

ICON4Py contains Python (GT4Py) implementations of ICON (inspired) components for weather and climate models.

## Project structure

Each directory contains Python packages of ICON components or utility packages that are deployable on its own. As these packages are not available from a package repository (yet), location of dependencies within this repository have to be provided explicitly, e.g. by installing the dependencies first. See [Installation instructions](#installation-instructions)

## Installation instructions

We recommend to use `tox` for the automatic installation of all packages in development mode in a single step:

```bash
# Clone the repository
git clone git@github.com:C2SM/icon4py.git
cd icon4py

# Use tox to create and set up a development environment (usually at `.venv`)
tox -e dev --devenv .venv

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

The `--src _external_src` option tells `pip` to use a specific folder as base path for checked out sources, which is very convenient for development tasks involving changes in external dependencies like `gt4py`. For convenience, `./_external_src` has already been added to the repository `.gitignore`.


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
