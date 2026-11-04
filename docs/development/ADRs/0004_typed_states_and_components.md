---
tags: [components, model state, typing]
---

# Typed states and components

- **Status**: valid
- **Authors**: Mikael Simberg (@msimberg), Jacopo Canton (@jcanton)
- **Created**: 2026-10-16
- **Updated**: 2026-10-28

ICON4Py has the dycore, the diffusion, the tracer advection, the physics with
muphys and tmx, and the IO, driven by driver and physics_driver.
Nothing reliably checks that the correct fields and variables are passed at the
interfaces, so we decided to introduce typed states and components.

## Context

The dycore and the diffusion took "states" as input: rigid state dataclasses of
`gtx.Field`. To mypy a field is its dimensions and dtype, so passing `theta_v`
where `exner` is expected was not a type error.
A physics component received a `dict[str, DataField]` and returned one (ADR
0001). The keys it needed and produced were listed in tables of
`FieldMetaData`, and the driver built the dicts with per-process adapters
(`as_component_input`). Nothing checked that the two agreed, so a missing or
misspelled key was a `KeyError` at run time.

**Concerns**

- What a component reads and writes should be visible in its signature and
  **checked statically**.
- gt4py and the stencils should see the same arrays as before, at no run-time
  cost.
- The metadata of a quantity (dimensions, units, CF names, precision) should
  live in one single place.
- Allocation, copies and IO should work on any state without a per-state list
  of fields.
- Who allocates and owns the buffers should be as unimportant as possible, and
  as close as possible to functional programming.

## Decision

### Definitions and pieces

- A `Quantity` is anything that can be measured, i.e. that can be expressed as
  a pair of numerical value and unit of measurement. In ICON4Py a `Quantity` is
  a type-level tag. One subclass per quantity, never instantiated, e.g.
  `VnOnEdgeK(Quantity, units="m s-1", standard_name="normal_velocity",
long_name="velocity normal to edge", dims=EDGE_K): ...`. The metadata is on
  the class: e.g. `units`, the CF `standard_name`, `long_name`, `precision` and
  `dims`.
  In the present Part01 of states and components `dims`, i.e. grid location,
  are attached to `Quantity` to keep the diff small with current ICON4Py.
  Part03 will separate the two.
- A `Tendency` is the rate of change of a `Quantity` with respect to time. It
  is a derived quantity, i.e. its definition is based on another quantity.
  In ICON4Py they are identified with the `Tendency` marker, which replaces
  `FieldKind.TENDENCY` used by the physics driver.
  In the present Part01 of states and components nothing links a `Tendency` to
  its `Quantity`, e.g. `TendencyOfTemperature` to `Temperature`, Part07 will
  add this with `TendencyOf[Q]`: a quantity derived from its parent, e.g.
  `Tendency[Temperature, ...]`.
- A `Field` is a physical quantity, represented by a scalar, vector or tensor,
  that has a value for each point in space and time. In ICON4Py a `Field` is a
  GT4Py field with the additional `Quantity` tag, e.g. `Field(VnOnEdgeK,
data)`. It subclasses GT4Py's `NdArrayField`, so the stencils and program
  calls do not need to change. The tag is a phantom, invariant type parameter:
  `Field[VnOnEdgeK]` and `Field[ThetaVOnCellK]` are different types to mypy, so
  they cannot be swapped, but they stay the same array to GT4Py.
- A `State` is a named, typed record of the values that a `Component` reads or
  writes. In ICON4Py it is a frozen, keyword-only dataclass whose leaves are
  quantity-tagged fields (`Field[Q]`, optionally `| None`) or plain values
  (e.g. `at_first_substep: bool`, `ndyn_substeps_var: int`).
- A `Component` is a `Protocol` of `Input`, `Output` and `run(Input, Output) ->
  Output`. It is a computation (`run`) with a typed interface: it reads one
  `Input` `State` and writes one `Output` `State`. A component that writes
  nothing declares `Output = fw.Empty`. The caller specifies where `run`
  writes, similarly to numpy's `out=`. Whether leaves of `Output` may alias
  leaves of `Input` is up to the `Component`.

### Writing a component

Example with a trimmed `Diffusion`:

```python
class Diffusion(Component):
    class Input(State):
        vn: Field[VnOnEdgeK]
        theta_v: Field[ThetaVOnCellK]
        dtime: float

    class Output(State):
        vn: Field[VnOnEdgeK]
        theta_v: Field[ThetaVOnCellK]

    def run(self, inputs: Input, outputs: Output) -> Output:
        ...  # stencils read inputs.vn and write outputs.vn
        return outputs
```

Things to note:
The diffusion and the physics driver update in place. The dycore needs distinct
`now` and `next`. Static fields (metrics, interpolation coefficients) stay
constructor arguments for now.

At the moment of writing, the components are `SolveNonhydro`, `Diffusion`,
`Advection`, `PhysicsDriver`, `Muphys` and `Tmx` (run by the physics driver)
and `IOMonitor`.

#### PhysicsProcess

`PhysicsProcess` is a Component that gates its wrapped component by time
control (with `ProcessTimeControl`). This is used for all processes called by
the physics_driver.

### Adding a quantity

One tag per quantity at one place on the grid, named `<Quantity>On<Place>`
(`ThetaVOnCellK`, `ThetaVOnCellKHalf`), in `quantities.py`. A 0-d quantity has
no place suffix (e.g. `MaxVerticalCflNumber`). A quantity has a `standard_name`
only where the [CF standard name
table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
has one, otherwise `long_name` describes what the field is, with the ICON
variable name in parentheses. A time-averaged or accumulated variant of a
quantity at the same place reuses the tag.

### Type checking

The framework (things introduced by this PR), the physics components and the
driver are checked with the strict mypy flags (see `[[tool.mypy.overrides]]` in
`pyproject.toml`). `SolveNonhydro`, `Diffusion` and `Advection` use the default
flags, but the driver's calls to them are strict. The Fortran bindings are not
type-checked, so a swapped tag there is not reported as error.

## Consequences

Positive:

- mypy reports a field of the wrong quantity, a missing or unknown leaf, and a
  process whose `collect_input` builds another component's `Input`, directly in
  the editor.
- The metadata of a field is on its quantity tag. The IO reads its CF
  attributes from there, and the state CF tables and `FieldKind` are gone.
- Allocation, copies, the physics driver's accumulation and the IO work on any
  state without hardcoded mechanisms.

Negative:

- Every `Input` and `Output` view is spelled out by keyword at the call site
  (part02 Collect fixes this).
- A quantity and its place are one tag, so `ThetaVOnCellK` and
  `ThetaVOnCellKHalf` are unrelated types. Only a unit test checks that a tag's
  name matches its `dims` (part03 Dims fixes this).
- The static field factories (geometry, metrics, interpolation) still key their
  fields by string and describe them with `FieldMetaData`, and the static
  fields of the components are untagged `gtx.Field`s (there is no written plan
  for this but should probably be part01b).
- `Field` builds on gt4py internals. It subclasses `NdArrayField` (gtfn's
  argument conversion accepts only that class), relies on gt4py building every
  derived field through `from_array`, and needs one subclass per array
  namespace (NumPy, CuPy).
- Whether `outputs` may alias `inputs` is not checked (but that's by design).

## Alternatives Considered

### String-keyed dictionaries with metadata tables (ADR 0001)

- Good, because the set of fields can be chosen at run time.
- Bad, because nothing checks the keys against the tables. A missing or
  misspelled key is a `KeyError` at run time, and the value of a key is a
  `gtx.Field` to mypy.

### gt4py's dimension-typed fields alone (`fa.CellKField[wpfloat]`)

- Good, because it is gt4py's own type, with nothing on top.
- Bad, because the dimensions are run-time objects. `CellKField` and
  `CellKHalfField` are one type to mypy, and two quantities on the same
  dimensions (`theta_v`, `exner`) are never distinguished.

### A wrapper holding the gt4py field (`Field(quantity, data)` with `.data`)

- Good, because it does not depend on gt4py internals.
- Bad, because every stencil call unwraps `.data`, or a call wrapper does it at
  a cost on every call.

### `typing.Annotated[gtx.Field, metadata]`

- Good, because the metadata sits next to the annotation.
- Bad, because the type checker ignores the metadata, so two quantities are the
  same type and can only be checked at runtime.

### One `typing.NewType` per quantity

- Good, because the checker tells the quantities apart.
- Bad, because each `NewType` is isolated. There is no metadata on the type and
  no marker base for the tendencies, and generic code over any quantity
  (`zeros`, `copy`) needs one version per type or a cast.

## References

- C2SM/icon4py-knowledge `mwe/components/DESIGN.md`, section "01 Typed states
  and components": the minimal working example the design was developed on.
- [ADR 0001](0001_model_physics_components_interface.md), superseded by this one.
