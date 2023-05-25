# ICON4PyTools

## Description

Tools and utilities for integrating icon4py code into the ICON model.

## Installation instructions

To install `icon4pytools` in a virtual environment, one can use pip with either the `requirements-dev.txt` or `requirements.txt` file. While the `requirements.txt` file will install the package along with its runtime dependencies, the `requirements-dev.txt` file additionally includes development dependencies required for running tests, generating documentation, and building the package from source. Furthermore by using the `requirements-dev.txt` file, the package will be installed in editable mode, allowing the user to make changes to the package's source code and immediately see the effects without having to reinstall the package every time. This is particularly useful during development and testing phases.

```bash
# create a virtual environment
python3 -m venv .venv

# activate the virtual environment
source .venv/bin/activate

# install all dependencies
pip install -r requirements-dev.txt
```

## Command-line tools

A variety of command-line tools are available in the shell after installation of `icon4pytools`.

### `icon4pygen`

A bindings generator that generates C++ and Fortran bindings from Gt4Py programs. This tools generates the following code:

- GridTools C++ `gridtools::fn` header file (`.hpp`).
- A Fortran interface file containing all wrapper functions which can be called from within ICON (`.f90`).
- A corresponding C (`.cpp, .h`) interface which in turn calls the Gridtools C++ code, as well as the C++ verification utils.

#### Usage

```
Usage: icon4pygen [OPTIONS] FENCIL [BLOCK_SIZE] [LEVELS_PER_THREAD] [OUTPATH]

  Generate Gridtools C++ code for an icon4py fencil as well as all the associated C++ and Fortran bindings.

  Arguments:
    FENCIL: may be specified as <module>:<member>, where <module>
            is the dotted name of the containing module and <member> is the name of the fencil.

    BLOCK_SIZE: refers to the number of threads per block to use in a cuda kernel.

    LEVELS_PER_THREAD: how many k-levels to process per thread.

    OUTPATH: represents a path to the folder in which to write all generated code.

Options:
  --is_global   Whether this is a global run.
  --imperative  Whether to use the imperative code generation backend.
  --help        Show this message and exit.
```

#### Autocomplete

In order to turn on autocomplete for the available fencils in your shell for `icon4pygen` you need to execute the following in your shell:

```bash
eval "$(_ICON4PYGEN_COMPLETE=bash_source icon4pygen)"
```

To permanently enable autocomplete on your system add the above statement to your `~/.bashrc` file.

## `icon_liskov`

A preprocessor that facilitates integration of GT4Py code into the ICON model. `icon_liskov` is a CLI tool which takes a Fortran file as input and processes it with the ICON-Liskov DSL Preprocessor. This preprocessor adds the necessary `USE` statements and generates OpenACC `DATA CREATE` statements and declares DSL input/output fields based on directives in the input file. The preprocessor also processes stencils defined in the input file using the `START STENCIL` and `END STENCIL` directives, inserting the necessary code to run the stencils and adding nvtx profile statements if specified with the `--profile` or `-p` flag. Additionally, specifying the `--metadatagen` or `-m` flag will result in the generation of runtime metadata at the top of the generated file.

### Usage

To use the `icon_liskov` tool, run the following command:

```bash
icon_liskov <input_filepath> <output_filepath> [--profile] [--metadatagen]
```

Where `input_filepath` is the path to the input file to be processed, and `output_filepath` is the path to the output file. The optional `--profile` flag adds nvtx profile statements to the stencils.

### Preprocessor directives

The ICON-Liskov DSL Preprocessor supports the following directives:

#### `!$DSL IMPORTS()`

This directive generates the necessary `USE` statements to import the Fortran to C interfaces.

#### `!$DSL START CREATE()`

This directive generates an OpenACC `DATA CREATE` statement for all output fields used in each DSL (icon4py) stencil. The directive also takes an **optional** keyword argument to specify extra fields to include in the `DATA CREATE` statement called `extra_fields`. Here you can specify a comma-separated list of strings which should be added to the `DATA CREATE` statement as follows `extra_fields=foo,bar`.

#### `!$DSL END CREATE()`

This directive generates an OpenACC `END DATA` statement which is neccessary to close the OpenACC data region.

#### `!$DSL DECLARE()`

This directive is used to declare all DSL input/output fields. The required arguments are the field name and its associated dimensions. For example:

```fortran
!$DSL DECLARE(vn=(nproma, p_patch%nlev, p_patch%nblks_e))
```

will generate the following code:

```fortran
! DSL INPUT / OUTPUT FIELDS
REAL(wp), DIMENSION((nproma, p_patch%nlev, p_patch%nblks_e)) :: vn_before
```

Furthermore, this directive also takes two optional keyword arguments. `type` takes a string which will be used to fill in the type of the declared field, for example `type=LOGICAL`. `suffix` takes a string which will be used as the suffix of the field e.g. `suffix=dsl`, by default the suffix is `before`.

#### `!$DSL START STENCIL()`

This directive denotes the start of a stencil. Required arguments are `name`, `vertical_lower`, `vertical_upper`, `horizontal_lower`, `horizontal_upper`. The value for `name` must correspond to a stencil found in one of the stencil modules inside `icon4py`, and all fields defined in the directive must correspond to the fields defined in the respective icon4py stencil. Optionally, absolute and relative tolerances for the output fields can also be set using the `_tol` or `_abs` suffixes respectively. An example call looks like this:

```fortran
!$DSL START STENCIL(name=mo_nh_diffusion_stencil_06; &
!$DSL       z_nabla2_e=z_nabla2_e(:,:,1); area_edge=p_patch%edges%area_edge(:,1); &
!$DSL       fac_bdydiff_v=fac_bdydiff_v; vn=p_nh_prog%vn(:,:,1); vn_abs_tol=1e-21_wp; &
!$DSL       vertical_lower=1; vertical_upper=nlev; &
!$DSL       horizontal_lower=i_startidx; horizontal_upper=i_endidx)
```

In addition, other optional keyword arguments are the following:

- `accpresent`: Takes a boolean string input, and controls the default data-sharing behavior for variables used in the OpenACC parallel region. Setting the flag to true will cause all variables to be assumed present on the device by default (`DEFAULT(PRESENT)`), and no explicit data-sharing attributes need to be specified. Setting it to false will require explicit data-sharing attributes for every variable used in the parallel region (`DEFAULT(NONE)`). By default it is set to false.<br><br>

- `mergecopy`: Takes a boolean string input. When set to True consecutive before field copy regions of stencils that have the mergecopy flag set to True are combined into a single before field copy region with a new name created by concatenating the names of the merged stencil regions. This is useful when there are consecutive stencils. By default it is set to false.<br><br>

- `copies`: Takes a boolean string input, and controls whether before field copies should be made or not. If set to False only the `#ifdef __DSL_VERIFY` directive is generated. Defaults to true.<br><br>

#### `!$DSL END STENCIL()`

This directive denotes the end of a stencil. The required argument is `name`, which must match the name of the preceding `START STENCIL` directive.

Together, the `START STENCIL` and `END STENCIL` directives result in the following generated code at the start and end of a stencil respectively.

```fortran
#ifdef __DSL_VERIFY
!$ACC PARALLEL IF( i_am_accel_node .AND. acc_on ) DEFAULT(NONE) ASYNC(1)
vn_before(:, :, :) = vn(:, :, :)
!$ACC END PARALLEL
```

```fortran
call nvtxEndRange()
#endif
call wrap_run_mo_nh_diffusion_stencil_06( &
z_nabla2_e=z_nabla2_e(:, :, 1), &
area_edge=p_patch%edges%area_edge(:, 1), &
fac_bdydiff_v=fac_bdydiff_v, &
vn=p_nh_prog%vn(:, :, 1), &
vn_before=vn_before(:, :, 1), &
vn_abs_tol=1e-21_wp, &
vertical_lower=1, &
vertical_upper=nlev, &
horizontal_lower=i_startidx, &
horizontal_upper=i_endidx
)
```

Additionally, there are the following keyword arguments:

- `noendif`: Takes a boolean string input and controls whether an `#endif` is generated or not. Defaults to false.<br><br>

- `noprofile`: Takes a boolean string input and controls whether a nvtx end profile directive is generated or not. Defaults to false.<br><br>

#### `!$DSL INSERT()`

This directive allows the user to generate any text that is placed between the parentheses. This is useful for situations where custom code generation is necessary.

#### `!$DSL START PROFILE()`

This directive allows generating an nvtx start profile data statement, and takes the stencil `name` as an argument.

#### `!$DSL END PROFILE()`

This directive allows generating an nvtx end profile statement.

#### `!$DSL ENDIF()`

This directive generates an `#endif` statement.

### `f2ser`

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
