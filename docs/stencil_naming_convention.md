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
Prefer the most specific verb that describes the operation; `compute` is the catch-all for operations that have no more specific verb.
The verb can be followed by more describing words.

The describing words must state what the stencil computes and on which grid entities, and must not repeat the names of the variables in the signature.
In particular, a stencil name must not contain:

- the caller's argument or output variable names, for example `calculate_nabla2_for_z`, named after the Fortran temporary `z_nabla2_e`; unless the mathematics genuinely depends on the quantity itself (then the swap test below does not pass);
- Fortran temporary prefixes (`z_`, `p_`, `opt_`) or Fortran module names, for example `mo_intp_rbf_rbf_vec_interpol_cell`;
- the physical meaning of an operand that the mathematics does not depend on.

The floating point precision (`_wp`, `_vp`) is an exception to the list above: GT4Py does not yet have dtype generics, so the same operator may exist in both precisions, for example `interpolate_edge_field_to_half_levels_vp` and `interpolate_edge_field_to_half_levels_wp`. The suffix is then required and stays in the name; drop it once the precision is part of the operator's type.

To check a name, replace an input with an unrelated field of the same type: if the stencil still works but the name no longer reads correctly, the name is too specific. If the computation genuinely depends on the quantity, naming after it is fine: the swap test will not pass.

Generic does not mean meaningless: name the operation, not placeholders. `add_fields` is a generic name, `compute_a_plus_b` is not.

These naming rules apply to every stencil, wherever it lives. Whether a stencil belongs in `model/common` is a separate question, answered by the section on shared code and generic naming in [CODING_GUIDELINES.md](../CODING_GUIDELINES.md).

## Example

A good example for a stencil name is for the `field_operator name`:

```
_interpolate_to_cell_center
```

and for the program

```
interpolate_to_cell_center
```
