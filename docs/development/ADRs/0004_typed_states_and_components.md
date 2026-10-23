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
states and components (a component declares its `Input` and `Output` as frozen dataclasses of
fields tagged by their quantity) and neglected string-keyed dictionaries with metadata tables,
`typing.Annotated` and `typing.NewType`, to achieve that mypy reports a missing, extra or swapped
field where a component is called, accepting that every view is spelled out by keyword and that
the tagged field subclasses gt4py's embedded `NdArrayField`.

## Context

A physics component received a `dict[str, DataField]` and returned one (ADR 0001). The keys it
needed and produced were listed in side tables of `FieldMetaData`, the driver built the dicts
with per-process adapters (`as_component_input`), and nothing checked that the two agreed: a
missing or misspelled key was a `KeyError` at run time. The dycore and the diffusion took state
dataclasses whose fields are `gtx.Field`; to mypy a field is its dimensions and dtype, so
passing `theta_v` where `exner` is expected is not a type error. Tendencies were marked by
`FieldKind.TENDENCY` in the metadata.

**Concerns**

- What a component reads and writes should be visible in its signature and checked statically.
- gt4py and the stencils should see the same arrays as before, at no run-time cost.
- Allocation, copies and IO should work on any state without a per-state list of fields.
- The caller should be able to choose the output buffers (in-place updates, double buffering)
  without an allocation per call.
- The metadata of a quantity (dimensions, units, CF names, precision) should live in one place.

## Decision

### The pieces (`icon4py.model.common.framework`)

- `Quantity`: a type-level tag, one subclass per quantity at one place on the grid, never
  instantiated. The metadata is on the class: `dims`, `units`, the CF `standard_name`,
  `long_name` and `precision` (`"wp"` or `"vp"`, resolved through `type_alias` when a field is
  allocated). Tendencies derive from the `Tendency` marker base, which replaces
  `FieldKind.TENDENCY`: the physics driver accumulates exactly those.
- `Field[Q]`: a gt4py field tagged by its quantity, `Field(qty.VnOnEdgeK, data)`, sharing the
  buffer of `data`. It subclasses gt4py's `NdArrayField`, so it goes into a stencil as it is; a
  field gt4py derives from it (arithmetic, slicing) is a plain, untagged one. The tag is a
  phantom, invariant type parameter: `Field[VnOnEdgeK]` and `Field[ThetaVOnCellK]` are different
  types to mypy and the same array to gt4py. Invariance means that a signature
  `(Field[Q], Field[Q])` takes the same quantity twice.
- `State`: a frozen, keyword-only dataclass of typed leaves. The `Field` leaves are read back
  from the annotations; plain leaves (`dtime: float`, datetimes) are ordinary fields. A leaf
  annotated `Field[Q] | None` is optional.
- `allocate(State, grid, allocator, only=...)` makes a zero field per declaration (`only` names
  the optional leaves to allocate, the others stay None); `zeros(Q, grid, allocator)` makes one
  field; `copy(state, allocator)` copies the present leaves.
- `Component`: a `Protocol` of `Input`, `Output` and `run(inputs, outputs) -> Output`, the
  only method a component implements. A component declares its two states as nested classes; a
  component that writes nothing declares `Output = fw.Empty`.
- `TimeStepPair` / `PredictorCorrectorPair`: icon4py's pairs (`common.utils`), re-exported.

The quantity tags are in `framework/quantities.py`. The states passed between the components
(`PrognosticState`, `TracerState`, `Diagnostics`, `PrepAdvection`, `DycoreForcing`,
`DycoreDiagnostics`, `DiffusionDiagnostics`, `AdvectionDiagnostics`) are in
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

`component.run(inputs, outputs)` writes where the caller says, numpy `out=` style: the caller
owns the buffers. Whether `outputs` may alias `inputs` is the component's business: the
diffusion and the physics driver update in place, the dycore needs distinct `now` and `next`.
Static fields (metrics, interpolation coefficients) stay constructor arguments.

The components are `SolveNonhydro` (dycore), `Diffusion`, `Advection` (tracer advection),
`PhysicsDriver`, `Muphys` (microphysics, run by the physics driver) and the driver's `IOMonitor`
(`driver_io.py`).

### Calling the components

The driver (`icon4py.model.driver.driver`) owns the states, builds each component's `Input` and
`Output` views from them by keyword and calls `run`. For an in-place update it passes the same
buffers on both sides (`Diffusion`, `PhysicsDriver`: the prognostics are input and output); for
the dycore it passes `now` in and `next` out.

The tracers are optional leaves: `TracerState`, `Advection.Input`/`Output` and
`PhysicsDriver.Input`/`Output` declare `qv`...`qg` as `Field[Q] | None`, and the driver's
`TracerConfig` decides which are present; the driver's views hold the active ones only.

The physics components (`Muphys`) are plain components. The physics driver runs each in a
`PhysicsProcess` (`physics_driver.py`): the component, its `collect_input`, which picks the
component's `Input` from the physics driver's `EntryState`, a `ProcessTimeControl` and the
output buffers the process keeps between two runs. The physics driver calls
`process.step(entry, step_start)`: inside its window the process runs its component on the
steps its time control fires and on its first step in the window, and returns its last output
otherwise. `PhysicsProcess` is generic over the component's `Input` and `Output`, so mypy checks
`collect_input` against the component it is paired with.

### Adding a quantity

One tag per quantity at one place on the grid, named `<Quantity>On<Place>` (`ThetaVOnCellK`,
`ThetaVOnCellKHalf`), in `quantities.py`; a 0-d quantity (`dims=()`) has no place suffix
(`MaxVerticalCflNumber`). A tag has a `standard_name` only where the
[CF standard name table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
has one; otherwise `long_name` says what the field is, with the ICON variable name in
parentheses. Some existing standard names are not in the CF table (`normal_velocity`,
`specific_cloud_content`, `virtual_potential_temperature`, ...): they are icon4py's attribute
values, and the output files name their variables after them. A field allocated in variable
precision has `precision="vp"` on its tag. A time-averaged or accumulated variant of a quantity
at the same place reuses the tag. The IO names an output variable after its quantity's
`standard_name`; four variables keep their file names (`driver_io._FILE_NAMES`).

### Type checking

The framework, the physics components and the driver with its `IOMonitor` are checked with the
strict mypy flags (the `[[tool.mypy.overrides]]` entry in `pyproject.toml` for
`icon4py.model.common.framework.*`, `physics_driver.*`, `muphys.component` and
`icon4py.model.driver.*`, and for the framework, physics driver and muphys unit tests and the
driver's `test_driver_io` and `test_driver_io_output`). `SolveNonhydro`, `Diffusion` and
`Advection` are checked with the project's default flags; the driver's calls to them are strict.
The Fortran bindings, which build `SolveNonhydro` and `Diffusion` views too, are not
type-checked (`ignore_errors`), so a swapped tag there is not reported.

`test_framework.py` holds a block of quantity mismatches (`static_checks`), each under a
`type: ignore`: with `warn_unused_ignores`, a mismatch that stops being an error fails the
type check.

## Consequences

- mypy reports a field of the wrong quantity, a missing or unknown leaf and a process whose
  `collect_input` builds another component's `Input`, where the code is written.
- The metadata of a field is on its quantity tag; the IO reads its CF attributes from there, and
  the state CF tables and `FieldKind` are gone.
- `allocate` and `copy` work on any state through the annotations; the physics driver's
  accumulation and the IO walk the present leaves through `fields()`.
- Every `Input`/`Output` view is spelled out by keyword at the call site.
- `Field` builds on gt4py internals: it subclasses `NdArrayField` (gtfn's argument conversion
  accepts only that class), relies on gt4py building every derived field through `from_array`,
  and has one subclass per array namespace (NumPy, CuPy).
- Calls through the `fw.Component` protocol are not checked: `run` is `Any -> Any` there,
  because each component narrows `inputs` and `outputs` to its own `Input` and `Output`. Only
  calls on a concrete class are checked.
- A quantity and its place are one tag: `ThetaVOnCellK` and `ThetaVOnCellKHalf` are unrelated
  types, and only a unit test checks that a tag's name matches its `dims`.
- Whether `outputs` may alias `inputs` is not checked.
- The field factories (geometry, metrics, interpolation) still key their fields by string and
  describe them with `FieldMetaData`; `PhysicsDriverStaticState` and the static fields of the
  other components are untagged `gtx.Field`s.

## Alternatives Considered

### String-keyed dictionaries with metadata tables (ADR 0001)

- Good, because the set of fields can be chosen at run time.
- Bad, because nothing checks the keys against the tables: a missing or misspelled key is a
  `KeyError` at run time, and the value of a key is a `gtx.Field` to mypy.

### A wrapper holding the gt4py field (`Field(quantity, data)` with `.data`)

- Good, because it depends on no gt4py internals.
- Bad, because every stencil call unwraps `.data`, or a call wrapper does it at a cost on every
  call.

### gt4py's dimension-typed fields alone (`fa.CellKField[wpfloat]`)

- Good, because it is gt4py's own type, with nothing on top.
- Bad, because the dimensions are run-time objects: `CellKField` and `CellKHalfField` are one
  type to mypy, and two quantities on the same dimensions (`theta_v`, `exner`) are never
  distinguished.

### `typing.Annotated[gtx.Field, metadata]`

- Good, because the metadata sits next to the annotation.
- Bad, because the type checker ignores the metadata: two quantities are the same type.

### One `typing.NewType` per quantity

- Good, because the checker tells the quantities apart.
- Bad, because each `NewType` is isolated: no metadata on the type, no marker base for the
  tendencies, and generic code over any quantity (`zeros`, `copy`) needs one version per type or
  a cast.

## References

- C2SM/icon4py-knowledge `mwe/components/DESIGN.md`, section "01 Typed states and components":
  the minimal working example the design was developed on, with its trade-offs part by part.
- [ADR 0001](0001_model_physics_components_interface.md), superseded by this one.
