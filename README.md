# ICON4Py

## Description

ICON4Py contains Python (GT4Py) implementations of ICON (inspired) components for weather and climate models.

## Project structure

Each directory contains Python packages of ICON components or utility packages that are deployable on its own. As these packages are not available from a package repository (yet), location of dependencies within this repository have to be provided explicitly, e.g. by installing the dependencies first. See [Installation instructions](#installation-instructions)

## Installation instructions

We recommend to install all packages at once by using the provided `requirements.txt` or `requirements-dev.txt` files in the root of the repository.

For example by using the following set of instructions:

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
pip install -r requirements-dev.txt

# Finally, check that everything works
pytest -v
```

Alternatively, you can use `tox` to reduce the number of manual steps:

```bash
# Clone the repository
git clone git@github.com:C2SM/icon4py.git
cd icon4py

# Use tox to create and set up a development environment (usually at `.venv`)
tox -e dev --devenv .venv 
```

### Installation with editable external sources

For advanced development tasks involving changes in external dependencies, you can tell `pip` to use a specific folder as base path for checked out sources:

```bash
# Install all the ICON4Py packages.
# External dependencies would be checked out at './_external_src'
pip install --src _external_src -r requirements-dev.txt

# Check that everything works (skipping external tests)
pytest --ignore _external_src -v
```

For convenience, `./_external_src` has been added to the repository `.gitignore`.

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
