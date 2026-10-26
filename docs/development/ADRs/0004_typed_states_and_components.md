---
tags:
  - components
  - model state
  - typing
---

# Typed states and components

- **Status**: valid
- **Authors**: Mikael Simberg (@msimberg), Jacopo Canton (@jcanton)
- **Created**: 2026-10-16
- **Updated**: 2026-10-16

In the context of a driver that composes the dycore, the diffusion, the tracer advection, the
physics and the IO, facing component interfaces whose fields nothing checks, we decided for typed
states and components and neglected string-keyed dictionaries with metadata tables,
`typing.Annotated` and `typing.NewType`, to achieve that mypy reports a missing, extra or swapped
field where a component is called, accepting that every view is spelled out by keyword and that
the tagged field subclasses gt4py's embedded `NdArrayField`.

## Context

A physics component received a `dict[str, DataField]` and returned one (ADR 0001). The keys it
needed and produced were listed in side tables of `FieldMetaData`, and the driver built the dicts
with per-process adapters (`as_component_input`). Nothing checked that the two agreed, so a
missing or misspelled key was a `KeyError` at run time. Tendencies were marked by
`FieldKind.TENDENCY` in the metadata.

The dycore and the diffusion took state dataclasses of `gtx.Field`. To mypy a field is its
dimensions and dtype, so passing `theta_v` where `exner` is expected is not a type error.

**Concerns**

- What a component reads and writes should be visible in its signature and checked statically.
- gt4py and the stencils should see the same arrays as before, at no run-time cost.
- Allocation, copies and IO should work on any state without a per-state list of fields.
- The caller should choose the output buffers (in-place updates, double buffering) without an
  allocation per call.
- The metadata of a quantity (dimensions, units, CF names, precision) should live in one place.

## Decision

### The pieces (`icon4py.model.common.framework`)

- `Quantity` is a type-level tag, one subclass per quantity at one place on the grid, never
  instantiated. The metadata is on the class: `dims`, `units`, the CF `standard_name`,
  `long_name` and `precision` (`"wp"` or `"vp"`). Tendencies derive from the `Tendency` marker
  base, which replaces `FieldKind.TENDENCY`. The physics driver accumulates exactly those.
- `Field[Q]` is a gt4py field tagged by its quantity, `Field(qty.VnOnEdgeK, data)`, sharing the
  buffer of `data`. It subclasses gt4py's `NdArrayField`, so it goes into a stencil as it is. A
  field gt4py derives from it (arithmetic, slicing) is a plain, untagged one. The tag is a
  phantom, invariant type parameter: `Field[VnOnEdgeK]` and `Field[ThetaVOnCellK]` are different
  types to mypy and the same array to gt4py.
- `State` is a frozen, keyword-only dataclass of typed leaves. Plain leaves (`dtime: float`,
  datetimes) are ordinary fields, and a leaf annotated `Field[Q] | None` is optional.
- `allocate`, `zeros` and `copy` make or copy fields for any state, from its annotations.
- `Component` is a `Protocol` of `Input`, `Output` and `run(inputs, outputs) -> Output`, the only
  method a component implements. A component that writes nothing declares `Output = fw.Empty`.

The quantity tags are in `framework/quantities.py`, the states passed between the components in
`framework/states.py`.

### Writing a component

A trimmed `Diffusion`:

```python
class Diffusion(fw.Component):
    class Input(fw.State):
        vn: fw.Field[qty.VnOnEdgeK]
        theta_v: fw.Field[qty.ThetaVOnCellK]
        dtime: float

    class Output(fw.State):
        vn: fw.Field[qty.VnOnEdgeK]
        theta_v: fw.Field[qty.ThetaVOnCellK]

    def run(self, inputs: Input, outputs: Output) -> Output:
        ...  # stencils read inputs.vn and write outputs.vn
        return outputs
```

`run` writes where the caller says, like numpy's `out=`: the caller owns the buffers. Whether
`outputs` may alias `inputs` is up to the component. The diffusion and the physics driver update
in place, the dycore needs distinct `now` and `next`. Static fields (metrics, interpolation
coefficients) stay constructor arguments.

The components are `SolveNonhydro`, `Diffusion`, `Advection`, `PhysicsDriver`, `Muphys` (run by
the physics driver) and the driver's `IOMonitor`.

### Calling the components

The driver owns the states. It builds each component's `Input` and `Output` from them by keyword
and calls `run`, passing the same buffers on both sides for an in-place update and `now` and
`next` to the dycore.

The tracers are optional leaves (`qv` to `qg` declared as `Field[Q] | None`), and the driver's
`TracerConfig` decides which are present.

The physics driver runs each physics component inside a `PhysicsProcess`, which holds the
component, its `collect_input` (builds the component's `Input` from the physics driver's
`EntryState`), a `ProcessTimeControl` and the output buffers kept between two runs. Outside its
window a process returns None. Inside it, the process runs its component when the time control
fires and on its first step in the window, and returns its last output otherwise.
`PhysicsProcess` is generic over the component's `Input` and `Output`, so mypy checks that
`collect_input` builds the `Input` of its component.

### Adding a quantity

One tag per quantity at one place on the grid, named `<Quantity>On<Place>` (`ThetaVOnCellK`,
`ThetaVOnCellKHalf`), in `quantities.py`. A 0-d quantity has no place suffix
(`MaxVerticalCflNumber`). A tag has a `standard_name` only where the
[CF standard name table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
has one, otherwise `long_name` says what the field is, with the ICON variable name in
parentheses. Some existing standard names are icon4py's own (`normal_velocity`,
`virtual_potential_temperature`, ...), and the output files name their variables after them. A
time-averaged or accumulated variant of a quantity at the same place reuses the tag.

### Type checking

The framework, the physics components and the driver are checked with the strict mypy flags (see
`[[tool.mypy.overrides]]` in `pyproject.toml`). `SolveNonhydro`, `Diffusion` and `Advection` use
the default flags, but the driver's calls to them are strict. The Fortran bindings are not
type-checked, so a swapped tag there is not reported. `test_framework.py` holds deliberate
quantity mismatches under `type: ignore`, and `warn_unused_ignores` fails the check when one stops
being an error.

## Consequences

On the positive side:

- mypy reports a field of the wrong quantity, a missing or unknown leaf, and a process whose
  `collect_input` builds another component's `Input`, where the code is written.
- The metadata of a field is on its quantity tag. The IO reads its CF attributes from there, and
  the state CF tables and `FieldKind` are gone.
- Allocation, copies, the physics driver's accumulation and the IO work on any state.

On the negative side:

- Every `Input` and `Output` view is spelled out by keyword at the call site.
- `Field` builds on gt4py internals. It subclasses `NdArrayField` (gtfn's argument conversion
  accepts only that class), relies on gt4py building every derived field through `from_array`,
  and needs one subclass per array namespace (NumPy, CuPy).
- Calls through the `fw.Component` protocol are not checked, because `run` is `Any -> Any` there.
  Only calls on a concrete class are.
- A quantity and its place are one tag, so `ThetaVOnCellK` and `ThetaVOnCellKHalf` are unrelated
  types. Only a unit test checks that a tag's name matches its `dims`.
- Whether `outputs` may alias `inputs` is not checked.
- The field factories (geometry, metrics, interpolation) still key their fields by string and
  describe them with `FieldMetaData`, and the static fields of the components are untagged
  `gtx.Field`s.

## Alternatives Considered

### String-keyed dictionaries with metadata tables (ADR 0001)

- Good, because the set of fields can be chosen at run time.
- Bad, because nothing checks the keys against the tables. A missing or misspelled key is a
  `KeyError` at run time, and the value of a key is a `gtx.Field` to mypy.

### A wrapper holding the gt4py field (`Field(quantity, data)` with `.data`)

- Good, because it depends on no gt4py internals.
- Bad, because every stencil call unwraps `.data`, or a call wrapper does it at a cost on every
  call.

### gt4py's dimension-typed fields alone (`fa.CellKField[wpfloat]`)

- Good, because it is gt4py's own type, with nothing on top.
- Bad, because the dimensions are run-time objects. `CellKField` and `CellKHalfField` are one
  type to mypy, and two quantities on the same dimensions (`theta_v`, `exner`) are never
  distinguished.

### `typing.Annotated[gtx.Field, metadata]`

- Good, because the metadata sits next to the annotation.
- Bad, because the type checker ignores the metadata, so two quantities are the same type.

### One `typing.NewType` per quantity

- Good, because the checker tells the quantities apart.
- Bad, because each `NewType` is isolated. There is no metadata on the type and no marker base
  for the tendencies, and generic code over any quantity (`zeros`, `copy`) needs one version per
  type or a cast.

## References

- C2SM/icon4py-knowledge `mwe/components/DESIGN.md`, section "01 Typed states and components":
  the minimal working example the design was developed on, with its trade-offs part by part.
- [ADR 0001](0001_model_physics_components_interface.md), superseded by this one.
