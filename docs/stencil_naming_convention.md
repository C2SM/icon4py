# Naming convection of stencils in ICON4Py

This naming convention should help to have a common style for stencil names.
It should help to read and understand stencil code.

## Multiple-word identifier format

The stencils must follow snake case.

## Length of name

The program name of the stencil needs to have less then 70 characters.

## Structure of name

The `field_operator` name should start with an underscore character.
The `program` name should be the same as for the field_operator without the underscore.
The `program` name should start with a verb.
Commonly used verbs are accumulate, add, apply, compute, copy, correct, extrapolate, interpolate, return, set, and
solve.
The verb can be followed by more describing words.

The describing words must state what the stencil computes and on which grid entities, and must not repeat the names of the variables in the signature.
In particular, a stencil name must not contain:

- the caller's argument or output variable names, for example `calculate_nabla2_for_z`, named after the Fortran temporary `z_nabla2_e`;
- Fortran temporary prefixes (`z_`, `p_`, `opt_`) or Fortran module names, for example `mo_intp_rbf_rbf_vec_interpol_cell`;
- the floating point precision (`_wp`, `_vp`), which is already part of the signature;
- the physical meaning of an operand that the mathematics does not depend on.

To check a name, replace an input with an unrelated field of the same type: if the stencil still works but the name no longer reads correctly, the name is too specific.

A stencil whose name passes this check is usually generic enough to belong in `model/common`.
See the section on shared code and generic naming in [CODING_GUIDELINES.md](../CODING_GUIDELINES.md) for where to place it.

## Example

A good example for a stencil name is for the `field_operator name`:

```
_interpolate_to_cell_center
```

and for the program

```
interpolate_to_cell_center
```
