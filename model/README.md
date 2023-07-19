# ICON4Py Model

This folder contains Python implementations for multiple ICON components.

It includes the following packages:

- `atmosphere/dycore`: Contains implementations of the dynamical core of the ICON model
- `common`: Contains shared functionality that is required by multiple components.

## Installation Instructions

You can follow the general installation instructions used for the entire icon4py repository, or install each namespace package within this folder independently using the individual folder-specific `requirements.txt` or `requirements-dev.txt` files.

In the following example it is assumed that you have already created and activated a virtual environment.

```bash
# changing into the corresponding directory
cd model/atmosphere/dycore

# installing a development version
pip install -r requirements-dev.txt
```

**Note**: For detailed information specific to each component, please refer to the README in their respective subfolders.
