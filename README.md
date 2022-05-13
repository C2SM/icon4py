# icon4py

Repository holding implementation of ICON stencils in [Gt4Py](https://github.com/GridTools/gt4py), ported from [Dusk](https://github.com/dawn-ico/dusk).


## Installation

Execute the following commands to setup a local development environment.

```bash
# Installing pre-commit hooks and testing hooks
pre-commit install
pre-commit run --all-files

# Setting up virtual environment and installing dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
pip install -r requirements-dev.txt
```

## Testing

The test suite can be executed by running `pytest`.
