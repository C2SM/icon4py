# Typed states and components

The model is a set of components the driver composes. Each component declares what it reads
and what it writes as typed states; mypy and pyright check that the driver passes the right
fields. The design and its trade-offs are in C2SM/icon4py-knowledge `mwe/components`
(`DESIGN.md`, section "01 Typed states and components"); this README is the short version.

## The pieces (`framework.py`)

- `Quantity`: a type-level tag, one subclass per quantity at one place on the grid, never
  instantiated. The metadata is on the class: `dims`, `units`, the CF `standard_name`,
  `long_name` and `precision` (`"wp"` or `"vp"`, resolved through `type_alias` when a field
  is allocated). Tendencies derive from the `Tendency` marker base: the physics driver
  accumulates exactly those.
- `Field[Q]`: a gt4py field tagged by its quantity, `Field(qty.VnOnEdgeK, data)`. The tag is
  a phantom, invariant type parameter: `Field[VnOnEdgeK]` and `Field[ThetaVOnCellK]` are
  different types to the checkers and the same array to gt4py. Stencils read `.data`.
- `State`: a frozen, keyword-only dataclass of typed leaves. `declarations()` lists the
  `Field` leaves (name, quantity, optional); plain leaves (`dtime: float`, datetimes) are
  ordinary fields. A leaf annotated `Field[Q] | None` is optional.
- `allocate(State, grid, allocator, only=...)` makes a zero field per declaration (`only`
  names the optional leaves to allocate, the others stay None); `zeros(Q, grid, allocator)`
  makes one field; `copy(state, allocator)` copies the present leaves.
- `Component`: `Input`, `Output` and `run(inputs, out=None) -> Output`, the only method a
  component implements. A component that writes nothing declares `Output = fw.Empty`.
- `TimeStepPair` / `PredictorCorrectorPair`: icon4py's pairs (`common.utils`), re-exported.

The quantity tags are in `quantities.py`, the states the driver passes between the
components (`PrognosticState`, `TracerState`, `Diagnostics`, `PrepAdvection`,
`DycoreForcing`, `DycoreDiagnostics`, `DiffusionDiagnostics`, `AdvectionDiagnostics`) in
`states.py`.

## Writing a component

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

    def run(self, inputs: Input, out: Output | None = None) -> Output:
        out = self.buffers(out)
        ...  # stencils read inputs.vn.data and write out.vn.data
        return out
```

`component.run(inputs)` writes into the component's own buffers (`Component.output`,
allocated once on first use); `component.run(inputs, out=view)` writes where the caller says,
numpy `out=` style. Whether `out` may alias the input is the component's business: the
diffusion and the physics driver continue in place, the dycore needs distinct now/next.
Static fields (metrics, interpolation coefficients) stay constructor arguments.

The components today: `SolveNonhydro` (dycore), `Diffusion`, `Advection` (tracer advection),
`PhysicsDriver` with `MuphysComponent` as its process, and the driver's
`IOMonitor` (`driver_io.py`).

## Composing

The driver (`icon4py.model.driver.driver`) is the composer. It owns the states, builds each
component's `Input` and `Output` views by keyword from them and calls `run`. For an in-place
update it passes the same buffers on both sides (`Diffusion`, `PhysicsDriver`: the
prognostics are input and output); for the dycore it passes `now` in and `next` out.

Optional leaves carry the tracers: `TracerState`, `Advection.Input`/`Output` and
`PhysicsDriver.Input`/`Output` declare `qv`...`qg` as `Field[Q] | None`, and the driver's
`TracerConfig` decides which are present. Since `Component.output` would allocate every
optional leaf, the composer always passes `out=` to these components. The physics processes
pick their `Input` from the physics driver's `EntryState` with a `collect_input`; `bind`
pairs it with the component's `run`, so pairing one process's `collect_input` with another's
`run` is a type error.

## Adding a quantity

One tag per quantity at one place on the grid, named `<Quantity>On<Place>` (`ThetaVOnCellK`,
`ThetaVOnCellKHalf`), in `quantities.py`. Give a new tag a `standard_name` only where the
[CF standard name table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
has one, otherwise leave it out and say in `long_name` what the field is (with the ICON
variable name in parentheses). Some existing tags keep names that are not in the CF table
(`normal_velocity`, `specific_cloud_content`, `virtual_potential_temperature`, ...): they are
icon4py's historical attribute values, and the output files name their variables after them. Put `precision="vp"` on the tag when the field is allocated in
variable precision. A time-averaged or accumulated variant of a quantity at the same place
reuses the tag. The IO names an output variable after its quantity's `standard_name` (a few
keep their historical file names, `driver_io._FILE_NAMES`).

## Typing gate

The framework, the physics components and the driver with its `IOMonitor` are checked
strictly by both mypy and pyright (1.1.414, run through `npx` by pre-commit;
`reportUnnecessaryTypeIgnoreComment` on). The strict mypy override and the pyright `include`
in `pyproject.toml` cover `icon4py.model.common.components.*`, `physics_driver.*`,
`muphys.component` and `icon4py.model.driver.*`, and of the tests the components, physics
driver and muphys unit tests and the driver's `test_driver_io` and `test_driver_io_output`. `SolveNonhydro`, `Diffusion` and `Advection`
are not strict: mypy checks them with the project's default flags and pyright not at all; the
driver's calls to them are strict. The Fortran bindings, which build `SolveNonhydro` and
`Diffusion` views too, are not type-checked (mypy `ignore_errors`, no pyright), so a swapped
tag there is not reported.
`test_framework.py` holds a block of quantity mismatches (`static_checks`), each under a
`type: ignore` that both checkers require: if a mismatch stops being an error, the ignore is
reported as unused.
