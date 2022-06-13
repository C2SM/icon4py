# icon4py-atm_dyn_iconam

## Description

Contains code ported from ICON `src/atm_dyn_iconam`.

## Installation instructions

We recommend to install all packages at once by using the provided `requirements.txt` or `requirements-dev.txt` files in the root of the repository.

In case you only want to install this subpackage, use the subpackage `requirements.txt` or `requirements-dev.txt` files.

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

# Install this specific ICON4Py subpackage and its dependencies
cd atm_dyn_iconam
pip install -r requirements-dev.txt
```
