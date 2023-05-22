# icon4py-pyutils

## Description

Python utilities for ICON4Py.

### icongen

Generates from GT4Py Field View programs

- GridTools C++ `gridtools::fn` code,
- field metadata to be used in dawn bindings generator for GT4Py.

## Installation instructions

Part of the `icon4pytools` package, check `README.md` file in the root of the repository for installation instructions.

## Autocomplete

In order to turn on autocomplete in your shell for `icon4pygen` you need to execute the following in your shell:

```bash
eval "$(_ICON4PYGEN_COMPLETE=bash_source icon4pygen)"
```

To permanently enable autocomplete on your system add the above statement to your `~/.bashrc` file.
