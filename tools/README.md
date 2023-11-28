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

A preprocessor that facilitates integration of GT4Py code into the ICON model. `icon_liskov` is a CLI tool which takes a fortran file as input and processes it with the ICON-Liskov DSL Preprocessor, generating code and inserting that into the target output file.

`icon_liskov` can either operate in **integration** or **serialisation** mode. In **integration** mode liskov generates calls to Fortran wrapper functions which enable calling icon4py DSL code inside of ICON from Fortran. In **serialisation** mode ppser serialbox statements are generated allowing the serialisation of all variables in all stencils decorated with liskov directives.

### Usage

#### Integration mode

```bash
icon_liskov integrate [--profile] [--metadatagen] <input_filepath> <output_filepath>
```

Options:

- `profile`: adds nvtx profile statements to the stencils (optional).
- `metadatagen`: generates a metadata header at the top of the file which includes information on icon_liskov such as the version used.

#### Serialisation mode

```bash
icon_liskov serialise [--multinode] <input_filepath> <output_filepath>
```

Options:

- `multinode`: ppser init contains the rank of the MPI process to facilitate writing files in a multinode context.

**Note**: By default the data will be saved at the default folder location of the currently run experiment and will have a prefix of `liskov-serialisation`.

### Preprocessor directives

The ICON-Liskov DSL Preprocessor supports the following directives:

#### `!$DSL IMPORTS()`

This directive generates the necessary `USE` statements to import the Fortran to C interfaces.

#### `!$DSL START CREATE()`

This directive generates an OpenACC `DATA CREATE` statement. The directive requires an **optional** keyword argument to specify extra fields to include in the `DATA CREATE` statement called `extra_fields`. Here you can specify a comma-separated list of strings which should be added to the `DATA CREATE` statement as follows `extra_fields=foo,bar`.

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

This directive denotes the start of a stencil. Required arguments are `name`, `vertical_lower`, `vertical_upper`, `horizontal_lower`, `horizontal_upper`. The value for `name` must correspond to a stencil found in one of the stencil modules inside `icon4py`, and all fields defined in the directive must correspond to the fields defined in the respective icon4py stencil. Optionally, absolute and relative tolerances for the output fields can also be set using the `_tol` or `_abs` suffixes respectively. For each stencil, an ACC DATA region will be created. This ACC DATA region contains the before fileds of the according stencil. An example call looks like this:

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
!$ACC DATA CREATE( &
!$ACC   vn_before, &
!$ACC      IF ( i_am_accel_node )

#ifdef __DSL_VERIFY
!$ACC KERNELS IF( i_am_accel_node .AND. acc_on ) DEFAULT(NONE) ASYNC(1)
vn_before(:, :, :) = vn(:, :, :)
!$ACC END KERNELS
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

!$ACC END DATA
```

Additionally, there are the following keyword arguments:

- `noendif`: Takes a boolean string input and controls whether an `#endif` is generated or not. Defaults to false.<br><br>

- `noprofile`: Takes a boolean string input and controls whether a nvtx end profile directive is generated or not. Defaults to false.<br><br>

- `noaccenddata`: Takes a boolean string input and controls whether a `!$ACC END DATA` directive is generated or not. Defaults to false.<br><br>

#### `!$DSL FUSED START STENCIL()`

This directive denotes the start of a fused stencil. Required arguments are `name`, `vertical_lower`, `vertical_upper`, `horizontal_lower`, `horizontal_upper`. The value for `name` must correspond to a stencil found in one of the stencil modules inside `icon4py`, and all fields defined in the directive must correspond to the fields defined in the respective icon4py stencil. Optionally, absolute and relative tolerances for the output fields can also be set using the `_tol` or `_abs` suffixes respectively. For each stencil, an ACC ENTER/EXIT DATA statements will be created. This ACC ENTER/EXIT DATA region contains the before fileds of the according stencil. An example call looks like this:

```fortran
        !$DSL START FUSED STENCIL(name=calculate_diagnostic_quantities_for_turbulence; &
        !$DSL  kh_smag_ec=kh_smag_ec(:,:,1); vn=p_nh_prog%vn(:,:,1); e_bln_c_s=p_int%e_bln_c_s(:,:,1); &
        !$DSL  geofac_div=p_int%geofac_div(:,:,1); diff_multfac_smag=diff_multfac_smag(:); &
        !$DSL  wgtfac_c=p_nh_metrics%wgtfac_c(:,:,1); div_ic=p_nh_diag%div_ic(:,:,1); &
        !$DSL  hdef_ic=p_nh_diag%hdef_ic(:,:,1); &
        !$DSL  div_ic_abs_tol=1e-18_wp; vertical_lower=2; &
        !$DSL  vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx)
```

#### `!$DSL END FUSED STENCIL()`

This directive denotes the end of a fused stencil. The required argument is `name`, which must match the name of the preceding `START STENCIL` directive.

Note that each `START STENCIL` and `END STENCIL` will be transformed into a `DELETE` section, when using the `--fused` mode.
Together, the `START FUSED STENCIL` and `END FUSED STENCIL` directives result in the following generated code at the start and end of a stencil respectively.

```fortran
        !$ACC DATA CREATE( &
        !$ACC   kh_smag_e_before, &
        !$ACC   kh_smag_ec_before, &
        !$ACC   z_nabla2_e_before ) &
        !$ACC      IF ( i_am_accel_node )

#ifdef __DSL_VERIFY
        !$ACC KERNELS IF( i_am_accel_node ) DEFAULT(PRESENT) ASYNC(1)
        kh_smag_e_before(:, :, :) = kh_smag_e(:, :, :)
        kh_smag_ec_before(:, :, :) = kh_smag_ec(:, :, :)
        z_nabla2_e_before(:, :, :) = z_nabla2_e(:, :, :)
        !$ACC END KERNELS
```

```fortran
call wrap_run_calculate_diagnostic_quantities_for_turbulence( &
    kh_smag_ec=kh_smag_ec(:, :, 1), &
    vn=p_nh_prog%vn(:, :, 1), &
    e_bln_c_s=p_int%e_bln_c_s(:, :, 1), &
    geofac_div=p_int%geofac_div(:, :, 1), &
    diff_multfac_smag=diff_multfac_smag(:), &
    wgtfac_c=p_nh_metrics%wgtfac_c(:, :, 1), &
    div_ic=p_nh_diag%div_ic(:, :, 1), &
    div_ic_before=div_ic_before(:, :, 1), &
    hdef_ic=p_nh_diag%hdef_ic(:, :, 1), &
    hdef_ic_before=hdef_ic_before(:, :, 1), &
    div_ic_abs_tol=1e-18_wp, &
    vertical_lower=2, &
    vertical_upper=nlev, &
    horizontal_lower=i_startidx, &
    horizontal_upper=i_endidx)

!$ACC EXIT DATA DELETE( &
!$ACC   div_ic_before, &
!$ACC   hdef_ic_before ) &
!$ACC      IF ( i_am_accel_node )
```

#### `!$DSL INSERT()`

This directive allows the user to generate any text that is placed between the parentheses. This is useful for situations where custom code generation is necessary.

#### `!$DSL START PROFILE()`

This directive allows generating an nvtx start profile data statement, and takes the stencil `name` as an argument.

#### `!$DSL END PROFILE()`

This directive allows generating an nvtx end profile statement.

#### `!$DSL START DELETE`

This directive allows to disable code. The code is only disabled if both the fused mode and the substition mode are enabled.
The `START DELETE` indicates the starting line from which on code is deleted.

#### `!$DSL END DELETE`

This directive allows to disable code. The code is only disabled if both the fused mode and the substition mode are enabled.
The `END DELETE` indicates the ending line from which on code is deleted.

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

## `py2f`

Python utility for generating a C library and Fortran interface to call Python icon4py modules. The library [embeds python via CFFI ](https://cffi.readthedocs.io/en/latest/embedding.)

This is **highly experimental** and has not been tested from within Fortran code!

### usage

`py2fgen`: Generates a C header file and a Fortran interface and compiles python functions into a C library embedding python. The functions need to be decorated with `CffiMethod.register` and have a signature with scalar arguments or `GT4Py` fields, for example:

```
@CffiMethod.register
def foo(i:int, param:float, field1: Field[[VertexDim, KDim], float], field2: Field[CellDim, KDim], float])
```

see `src/icon4pytools/py2f/wrappers` for examples.

```bash
py2fgen icon4pytools.py2f.wrappers.diffusion_wrapper  py2f_build
```

where the first argument is the python module to parse and the second a build directory. The call above will generate the following in `py2f_build`:

```bash
 ls py2f_build/
total 204K
drwxrwxr-x 2 magdalena magdalena 4.0K Aug 24 17:01 .
drwxrwxr-x 9 magdalena magdalena 4.0K Aug 24 17:01 ..
-rw-rw-r-- 1 magdalena magdalena  74K Aug 24 16:58 diffusion_wrapper.c
-rw-rw-r-- 1 magdalena magdalena 3.5K Aug 24 17:01 diffusion_wrapper.f90
-rw-rw-r-- 1 magdalena magdalena  955 Aug 24 17:01 diffusion_wrapper.h
-rw-rw-r-- 1 magdalena magdalena  58K Aug 24 17:01 diffusion_wrapper.o
-rwxrwxr-x 1 magdalena magdalena  49K Aug 24 17:01 libdiffusion_wrapper.so

```
