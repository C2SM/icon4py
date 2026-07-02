# GT4Py pattern notes from the tmx port (milestone M1)

Findings from implementing the "risky" tmx patterns in isolation, verified on the
`roundtrip` (default) and `gtfn_cpu` backends. Kept here so later milestones (and other
ports) do not rediscover them.

## Proven patterns (no workaround needed)

- **KDim `scan_operator` with EdgeDim as horizontal dimension.** Scan operators are
  horizontal-dimension agnostic; the identical scan callables are reused by the cell- and
  edge-based tridiagonal solvers (`solve_vertical_diffusion_cells.py` /
  `solve_vertical_diffusion_edges.py`).
- **Scan with restricted vertical domain.** With program domain `KDim: (1, nlev)` the scan
  init is applied at the domain start (k=1) and rows above are left untouched — required by
  the w-diffusion solve (Fortran `minlvl=2`).
- **Fixed absolute-K rows inside `concat_where` branches** (top/bottom boundary rows of
  half-level interpolation and the Smagorinsky boundary copies). Each branch is evaluated
  only on its own K region, and domain inference chains through *nested* `concat_where`,
  so per-region `KDim ± n` shifts need only be valid inside their region.
- **Runtime scalars in K conditions**: `dims.KDim == nlev` with `nlev: gtx.int32` as a
  program argument.
- **Compile-time variant selection**: scalar `bool` program arguments driving Python `if`
  inside a field operator, optionally inlined at compile time via `program.compile(...)`
  (`StencilTest.STATIC_PARAMS`). Used for `use_louis` / `use_louis_land` / `use_louis_ice`.
- **In-place read-modify-write**: passing `tend` as input and as member of the tuple output
  (`out=(new_var, tend)`) works on both backends (tendency accumulation in the solvers).
- **Chained unstructured + vertical offsets**: `w(E2C[0])(KDim + 1)`; sparse-dimension
  element selection via `field[E2C2VDim(i)]` after a `field(E2C2V)` shift.
- **Dimension promotion**: 2D `CellField` conditions/factors broadcast against 3D
  `CellKField` expressions without explicit `broadcast`.
- **Half-level (nlev+1) fields** are plain KDim fields (tests allocate with
  `extend={dims.KDim: 1}`); no separate half-level dimension needed.
- Fortran `(edge, 3)` extrapolation coefficient arrays (`wgtfacq_e`, `wgtfacq1_e`) are
  passed as three separate 2D edge fields, matching dycore usage.
- A fused 3x3 velocity-gradient tensor + shear contraction (9 tensor locals in one field
  operator, mixed E2C2V/E2C/E2V offsets) compiles and verifies on gtfn_cpu without
  temporary-extraction issues.

## Additional patterns from milestone M3

- **`PhysicsConstants` enum members work inline on gtfn** (unlike plain module-level `Final`
  floats): `icon4py.model.common.constants` wpfloat-subclass enum members can be used directly
  inside field operators, including as `power()` arguments and in scalar/field division.
  Prefer this over the inline-local workaround when the constant exists in the enum.
- **Cross-package reuse of private common field operators** (e.g.
  `_interpolate_cell_field_to_half_levels_wp`) composes inside tmx field operators on both
  backends, also nested in `concat_where` branches.
- **Fixed absolute-K row inputs** (Fortran `field(:,nlevp1,:)`) must be passed as pre-sliced
  2D fields — GT4Py offsets are relative only; 2D→3D broadcast handles the rest.
- **Zone-table gap**: Fortran `min_rledge_int-3` has no `h_grid.Zone` equivalent; use the
  closest more-inclusive zone (`END`) — identical on a single node, revisit for MPI.
- The common `interpolate_to_cell_center` is vp-typed; usable from wp-only tmx because
  `vpfloat == wpfloat` in double precision. Breaks under `--enable-mixed-precision`.

## Limitations and workarounds

- **Module-level constants fail on gtfn**: symbols referenced inside a field operator must
  be locals (`x = wpfloat("...")`) — module-level `Final` constants raise
  `EveValueError: Symbols ... not found` at gtfn lowering (matches muphys practice).
- **`abs` is not the Python builtin** inside field operators: import it with
  `from gt4py.next import abs  # noqa: A004`.
- **No `and`/`or`/`not` on scalar bools in FOAST**: compose config-flag logic with nested
  `if`/`else` inside the field operator instead of boolean expressions.
- **Literal-zero tuple-output members** need
  `broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))`; a bare scalar cannot be a tuple
  output member.
- Geometry structs (`primal_normal_vert%v1/%v2`) must be flattened into separate x/y sparse
  fields — same as the diffusion precedent.
- Ruff `PLR0917` (too many positional args) is per-file-ignored only for `stencils/*.py`;
  numpy reference helpers in tests need keyword-only arguments.

## Still to prove in later milestones

- DaCe backends (`dace_cpu`/`dace_gpu`): scan-heavy programs likely need per-program
  options in `model_options.get_dace_options` (deferred to M7).
- Halo-exchange placement with `ExchangeRuntime` in the granule (M3).
