# icon4py-pyutils

## icon4pygen

The `icon4pygen` tool generates GridTools C++ code as well as Fortran and C++ bindings for an icon4py fencil, so that it can be executed from within ICON.

### Usage:

`icon4pygen [OPTIONS] FENCIL BLOCK_SIZE LEVELS_PER_THREAD OUTPATH`

#### Arguments:

```
FENCIL: The fencil to generate code for. It can be specified as <module>:<member>, where <module> is the dotted name of the containing module and <member> is the name of the fencil.
BLOCK_SIZE: The number of threads per block to use in a cuda kernel.
LEVELS_PER_THREAD: How many k-levels to process per thread.
OUTPATH: The folder in which to write all generated code.
```

#### Options

```
--is_global: Flag to indicate if the stencil is global.
--imperative: Flag to indicate if the generated code should be written imperatively.
```

#### Example:

`icon4pygen icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_1:mo_velocity_advection_stencil_1 128 4 /path/to/output/folder`

#### Autocomplete

In order to turn on autocomplete in your shell for `icon4pygen` you need to execute the following in your shell:

```bash
eval "$(_ICON4PYGEN_COMPLETE=bash_source icon4pygen)"
```

To permanently enable autocomplete on your system add the above statement to your `~/.bashrc` file.

## f2ser

This tool is designed to parse a well-defined Fortran granule interface and generate ppser statements for each variable in the interface. It uses the `f2py` library to perform the parsing and `liskov` for the generation tasks.

### Usage

`f2ser [OPTIONS] GRANULE_PATH OUTPUT_FILEPATH`

### Arguments

```
GRANULE_PATH      A path to the Fortran source file to be parsed.
OUTPUT_FILEPATH   A path to the output Fortran source file to be generated.
```

### Options

```
--dependencies PATH  Optional list of dependency paths.
--directory TEXT      The directory to serialise the variables to.
--prefix TEXT         The prefix to use for each serialised variable.
```

**Note:** The output of f2ser still has to be preprocessed using `pp_ser.py`, which then yields a compilable unit. The serialised files will have `f2ser` as their prefix in the default folder location of the experiment.
