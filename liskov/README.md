# icon4py-liskov

A preprocessor that facilitates integration of gt4py code into the ICON model.

## Installation

To install the icon4py-liskov package, follow the instructions in the `README.md` file located in the root of the repository.

## Description

The icon4py-liskov package includes the `icon_liskov` CLI tool which takes a fortran file as input and processes it with the ICON-Liskov DSL Preprocessor. This preprocessor adds the necessary `USE` statements and generates OpenACC `DATA CREATE` statements and declares DSL input/output fields based on directives in the input file. The preprocessor also processes stencils defined in the input file using the `START` and `END` directives, inserting the necessary code to run the stencils and adding nvtx profile statements if specified with the `--profile` flag.

### Usage

To use the `icon_liskov` tool, run the following command:

```bash
icon_liskov <filepath> [--profile]
```

Where filepath is the path to the input file to be processed, and the --profile flag (optional) adds nvtx profile statements to the stencils.

### Preprocessor directives

The ICON-Liskov DSL Preprocessor supports the following directives:

#### `!$DSL IMPORT()`

This directive generates the necessary `USE` statements to import the Fortran to C interfaces.

#### `!$DSL CREATE()`

This directive generates an OpenACC `DATA CREATE` statement for all output fields used in each DSL (icon4py) stencil.

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

#### `!$DSL START()`

This directive denotes the start of a stencil. Required arguments are `name`, `vertical_lower`, `vertical_upper`, `horizontal_lower`, `horizontal_upper`. The value for `name` must correspond to a stencil found in one of the stencil modules inside `icon4py`, and all fields defined in the directive must correspond to the fields defined in the respective icon4py stencil. Optionally, absolute and relative tolerances for the output fields can also be set using the `_tol` or `_abs` suffixes respectively. An example call looks like this:

```fortran
!$DSL START(name=mo_nh_diffusion_stencil_06; &
!$DSL       z_nabla2_e=z_nabla2_e(:,:,1); area_edge=p_patch%edges%area_edge(:,1); &
!$DSL       fac_bdydiff_v=fac_bdydiff_v; vn=p_nh_prog%vn(:,:,1); vn_abs_tol=1e-21_wp; &
!$DSL       vertical_lower=1; vertical_upper=nlev; &
!$DSL       horizontal_lower=i_startidx; horizontal_upper=i_endidx)
```

#### `!$DSL END()`

This directive denotes the end of a stencil. The required argument is `name`, which must match the name of the preceding `START` directive.

Together, the `START` and `END` directives result in the following generated code at the start and end of a stencil respectively.

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
