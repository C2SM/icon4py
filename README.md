# ICON4Py

## Description

ICON4Py contains Python (GT4Py) implementations of ICON (inspired) components for weather and climate models.

## Project structure

Each directory contains Python packages of ICON components or utility packages that are deployable on its own. As these packages are not available from a package repository (yet), location of dependencies within this repository have to be provided explicitly, e.g. by installing the dependencies first. See [Installation instructions](#installation-instructions)

## Installation instructions

We recommend to install all packages at once by using the provided `requirements-dev.txt` file in the root of the repository.

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

# Install the all ICON4Py packages and its dependencies
pip install -r requirements-dev.txt

# Finally, check that everything works
pytest -v
```
