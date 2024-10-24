# icon4py-atmosphere-dycore

## Description

Contains code ported from ICON `src/atm_dyn_iconam`, which is the dynamical core of the ICON model.

## Attention

Variables in ICON can either sit on full levels or half levels. Let's denote the number of full levels is nlev.
Full-level variables have nlev levels in vertical dimension, while half-level variables have nlev+1 in vertical dimension.
Do not output full-level and half-level variables together when running a gt4py program if you need to access the nlev+1 level because it will cause random behavior (most probably illegal memory access).
The following code is an example:

```python
@field_operator
def foo(
    input_var,
    k_field,
    nlev,
):
    (half_level_var, full_level_var) =  where(
        (k_field >= 0) & (k_field < nlev),
        _compute_var(input_var),
        (half_level_var, full_level_var),
    )
    half_level_var = where(k_field == nlev, _init_cell_kdim_field_with_zero_vp(), half_level_var)
    return half_level_var, full_level_var

@program
def foo(
    input_var,
    k_field,
    nlev,
    full_level_var,
    half_level_var,
    horizontal_start,
    horizontal_end,
    vertical_start,
   vertical_end,
):
    _foo(
        input_var,
        k_field,
        nlev,
        out=(full_level_var, half_level_var),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        }
    )
)

foo(
    input_var=input_var,
    full_level_var=full_level_var,
    half_level_var=half_level_var,
    k_field=k_field,
    nlev=nlev,
    horizontal_start=start_cell,
    horizontal_end=end_cell,
    vertical_start=0,
    vertical_end=nlev + 1,
    offset_provider={},
)
```

## Installation instructions

Check the `README.md` at the root of the `model` folder for installation instructions.
