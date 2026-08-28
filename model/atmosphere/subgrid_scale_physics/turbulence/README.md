# ICON NWP 1D turbulence

GT4Py port of the COSMO/Raschendorfer 1D turbulence scheme selected by `inwp_turb = 1` in
ICON-NWP: surface-layer transfer (`turbtran`), the atmospheric TKE closure (`turbdiff`) and the
implicit vertical diffusion of first-order variables and tracers (`vertdiff`).

## Fortran provenance

| Fortran source                                        | ported to                                        |
| ----------------------------------------------------- | ------------------------------------------------ |
| `src/atm_phy_schemes/turb_transfer.f90` (`turbtran`)  | `stencils/`, driven by `Turbulence.run_turbtran` |
| `src/atm_phy_schemes/turb_diffusion.f90` (`turbdiff`) | `stencils/`, driven by `Turbulence.run_turbdiff` |
| `src/atm_phy_schemes/turb_vertdiff.f90` (`vertdiff`)  | `stencils/`, driven by `Turbulence.run_vertdiff` |
| `src/atm_phy_schemes/turb_utilities.f90`              | shared kernels used by all three                 |

The ICON-side interfaces (`mo_nwp_turbdiff_interface.f90`, `mo_nwp_turbtrans_interface.f90`) stay in
Fortran; they are out of scope. Scientific commentary in the Fortran sources is by Matthias
Raschendorfer (DWD); each stencil cites the module, subroutine and line range it was translated
from.

Only the operational configuration space is implemented. `TurbulenceConfig` rejects namelist
switches whose non-default values are not ported — see `turbulence.py`.

## Testing

Unit tests need no data:

```bash
uv run --group test --frozen pytest model/atmosphere/subgrid_scale_physics/turbulence/tests/turbulence/unit_tests/
```

Stencil and integration tests validate against serialized ICON reference data from
`exp.mch_icon-ch2_small`. That data is **not** downloadable while the port is in progress; place a
local capture under `$ICON4PY_TEST_DATA_PATH/` and `touch .extraction_complete` in it.

## Boundary rows: when to use `concat_where`

Nearly every section of `turbdiff` treats the surface half level `ke1` differently from the
half levels above it, and Raschendorfer writes the two cases as two separate ACC loops. That
split is a Fortran artefact — the loops alias one array, so the boundary block has to run first
— and it does not have to survive translation. The rule for the port:

**Merge into one `@gtx.program` with `concat_where` when the boundary row is a different
*coefficient or expression* for the *same output field*. Keep separate programs when the
boundary row writes a *different field*, or writes nothing.**

Worked out on section 1a) and 1b), which is where each half of the rule was decided:

| Fortran                                 | ported as                                                                 | why                                                                                                                                                                                                                                                                                     |
| --------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `zvari(:,ke1,n)` and `zvari(:,k,n)`     | ONE program, `concat_where(dims.KDim == nlev, …)` on the reciprocal depth | Identical difference quotient `(zvari(k-1) - zvari(k)) * scale`; only `scale` differs (`lays` vs `hlp`). One field, one program.                                                                                                                                                        |
| `hlp`/`dicke`, `DO k=ke,2,-1`           | one program, no `concat_where`                                            | Rows 0 and `ke1` are not written at all in this section, so there is no second case to select. Its own vertical domain says that.                                                                                                                                                       |
| `frh` (`k=2,ke1`) and `frm` (`k=2,kem`) | TWO programs                                                              | Different fields, different formulas, disjoint inputs. Fusing them would need `concat_where(KDim < nlev, shear, frm)`, i.e. a read-modify-write turning "`frm(:,ke1)` is never written" into "`frm(:,ke1)` is rewritten with its old value". That is a semantic change, not a refactor. |

### What merging buys, and what it costs

Buys, in order of weight:

1. **The vertical boundary is stated once, inside the stencil, next to the Fortran it came
   from.** With two programs it is stated in each caller's `vertical_start`/`vertical_end`, and
   `turbulence.py` would carry one hand-written index pair per boundary across ~13 sections.
   Each is an off-by-one that a section datatest hard-coding the same numbers cannot see.
2. One kernel launch instead of two. The boundary kernel is one row deep, so it is nearly all
   launch overhead — this matters for `dace_gpu`, the primary target.
3. The difference expression is written once rather than duplicated across two files.

Costs, measured here, not assumed:

- **The embedded backend cannot run `concat_where` at all.** In gt4py 1.1.10,
  `gt4py/next/embedded/nd_array_field.py::_concat_where` still implements the *old*
  boolean-mask API and is handed a `common.Domain` by the new frontend, so
  `concat_where(dims.KDim == nlev, …)` dies with
  `AttributeError: 'Domain' object has no attribute 'domain'`
  (the file's own comment: *"this is still the 'old' concat_where, needs to be replaced in a
  next PR"*). Mark the affected tests `@pytest.mark.uses_concat_where`; `model/testing/filters.py`
  turns that into an xfail on embedded. This is what the rest of the repo already does — the
  dycore implicit solver, velocity advection, the diffusion and muphys integration tests and
  `metric_fields` all xfail on embedded for the same reason.
  The narrow rule above is what bounds the damage: only genuine per-row-coefficient stencils lose
  embedded coverage, and a section with no such case keeps all three backends.
- `nlev` becomes a program argument *in addition to* `vertical_end`, and nothing checks that they
  agree. The datatest is what pins it.

### Mechanics

Adopt the dycore idiom (`model/common/.../metrics/metric_fields.py::_compute_ddqz_z_half`,
`dycore/stencils/compute_cell_diagnostics_for_dycore.py`): `nlev: gtx.int32` as a field-operator
argument, `concat_where(<K predicate>, <boundary value>, <interior value>)`, result assigned back
to the same name when several overrides chain. A 2D `CellField` is accepted as a branch and
broadcast over `KDim`.

Two things verified on `gtfn_cpu` and `dace_cpu` rather than assumed:

- `dims.KDim == nlev` is correct and bit-exact. Do **not** write `dims.KDim == nlev - 1`;
  GT4Py's domain inference miscompiles it (GridTools/gt4py#2205, and there is a `TODO(havogt)`
  in `dycore/stencils/compute_advection_in_vertical_momentum_equation.py` saying so).
- Only the branch a row selects is evaluated. Poisoning `hlp(:,ke1)` with NaN — a row section 1a)
  never defines — leaves the surface gradients bit-exact on both backends.
  `test_gradients_at_the_surface_do_not_read_the_geometric_depth` keeps measuring that, so a
  future GT4Py that evaluates both branches everywhere fails loudly instead of silently reading
  undefined rows.
