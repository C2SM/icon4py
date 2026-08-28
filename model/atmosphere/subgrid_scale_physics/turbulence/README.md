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

### Shared plumbing for a section test

`tests/turbulence/utils.py` holds what every section datatest needs — the serialized dates, the
experiment marker, the output-buffer allocators and the gate-consulting comparison — and states the
two conventions all of them follow: every comparison is masked to `ivstart:ivend`, and every output
field is allocated as a copy of its entry state so that the rows the section does not write are
asserted untouched rather than ignored. Import it as a module (`from .. import utils`) and read its
docstring before writing a new section test; `tests/turbulence/gate_registry.py` needs an entry for
each stencil before it can be compared against anything.

### Bit-exactness on the GPU backends

The gates in `tests/turbulence/gate_registry.py` are `Exact()`, and they hold on `embedded`,
`gtfn_cpu`, `dace_cpu`, `gtfn_gpu` and `dace_gpu`. Three things had to be true for that, and only
the first is obvious:

- **No multiply-add contraction on either side.** The reference is built with `-Kieee -Mnofma -gpu=nofma`; the port sets `CXXFLAGS=-ffp-contract=off` and `CUDAFLAGS=--fmad=false` in
  `tests/turbulence/conftest.py`. Both reach the compiler: GT4Py's CMake toolchain picks them up
  for `gtfn_*`, and for `dace_*` GT4Py reads them itself and writes `compiler.cpu.args` /
  `compiler.cuda.args` (`gt4py/next/program_processors/runners/dace/workflow/common.py`), which
  also displaces DaCe's default `--use_fast_math`. Verified on `dace_gpu`: `compute_thermal_forcing`
  (`a*b + c*d`, the expression that is sensitive to it) is bit-exact.
- **Write a square as a product, never as `x**2`.** Fortran's integer-exponent `**` is a
  multiplication; GT4Py's `**` becomes `math.pow`, and CUDA's `pow` carries up to 2 ulp of error.
  See the docstring of `_compute_mechanical_forcing`, which is where it was measured.
- **One persistent GT4Py build cache directory per backend.** GT4Py's cache key is the program,
  the offset provider and the column axis -- not the backend and not the compiler flags -- so a
  shared `GT4PY_BUILD_CACHE_DIR` serves a CPU-compiled program to a GPU run. `pytest_configure`
  in `tests/turbulence/conftest.py` adds the `--backend` subdirectory; the flag set is in the
  directory name the run scripts choose.

## Boundary rows: when to use `concat_where`

Nearly every section of `turbdiff` treats one end of the column differently from the rest — most
often the surface half level `ke1`, in section 11) the model top — and Raschendorfer writes the
two cases as two separate ACC loops. That split is a Fortran artefact — the loops alias one array,
so the boundary block has to run first — and it does not have to survive translation. The rule for
the port:

**Merge into one `@gtx.program` with `concat_where` when the boundary row is a different
*coefficient or expression* for the *same output field*. Keep separate programs when the
boundary row writes a *different field*, or writes nothing.**

Worked out on section 1a) and 1b), which is where each half of the rule was decided; section 11)
is the first application at the other end of the column:

| Fortran                                                            | ported as                                                                 | why                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `zvari(:,ke1,n)` and `zvari(:,k,n)`                                | ONE program, `concat_where(dims.KDim == nlev, …)` on the reciprocal depth | Identical difference quotient `(zvari(k-1) - zvari(k)) * scale`; only `scale` differs (`lays` vs `hlp`). One field, one program.                                                                                                                                                                                           |
| `hlp`/`dicke`, `DO k=ke,2,-1`                                      | one program, no `concat_where`                                            | Rows 0 and `ke1` are not written at all in this section, so there is no second case to select. Its own vertical domain says that.                                                                                                                                                                                          |
| `frh` (`k=2,ke1`) and `frm` (`k=2,kem`)                            | TWO programs                                                              | Different fields, different formulas, disjoint inputs. Fusing them would need `concat_where(KDim < nlev, shear, frm)`, i.e. a read-modify-write turning "`frm(:,ke1)` is never written" into "`frm(:,ke1)` is rewritten with its old value". That is a semantic change, not a refactor.                                    |
| `rcld(:,1)=rcld(:,2)` and `rcld(:,k)=(rcld(:,k)+rcld(:,k+1))*z1d2` | ONE program, `concat_where(dims.KDim == 0, …)` on the result              | Same two half levels read on every row, weighted `(0, 1)` at the model top and `(1/2, 1/2)` below. Only the coefficients differ, so one field, one program — the rule reads the same at `KDim == 0` as at `KDim == nlev`. The two rows the section does not write are left to the vertical domain, as `hlp`/`dicke` above. |

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

## Vertical recurrences: how to tell whether a `k`-loop is one

`turbdiff` has roughly ten genuine sequential recurrences and around a hundred and seventy `k`
loops, so the question "does this section need a `scan_operator`?" comes up in almost every
section and is answered wrongly by the obvious signal.

**A `!$ACC LOOP SEQ` is not evidence of a recurrence.** Raschendorfer marks a loop sequential for
several reasons — scratch reuse, `k`-dependent branching, register pressure — and the port must
not inherit the serialisation those bring. `solve_turb_budgets`' main loop
(`turb_utilities.f90:1384`) is `LOOP SEQ` with no `k±1` access at all and is deliberately kept
wide; section 6)'s only `LOOP SEQ` (`turb_diffusion.f90:2283`) is likewise offset-free, and is
dead in the operational configuration besides. Conversely the two real recurrences of section 8)
and section 9) are written as `LOOP SEQ` — but so is a lot else.

**The test that does decide it.** A `k` iteration must *read what a previous `k` iteration of the
same loop wrote*. Applied mechanically:

1. Take the array the loop writes.
2. Look for a read of **that same array** at `k±1` **inside the same loop nest**.
3. If the `k±1` read is of a *different* array — even one the previous `!$ACC PARALLEL` region
   just filled — there is no recurrence. A separate ACC region is a barrier, so what it produced
   is an input, and the port has it as its own field: an ordinary `Koff[±1]` stencil.
4. In-place writes at the *same* `k` (`dicke(i,k) = dicke(i,k)*tke(i,k)`) are pointwise, not
   sequential.

Section 6) fails the test at every one of its eight loops — five live, three dead — and is
therefore plain field operators throughout. Three of the loops read a neighbouring half level:
`expl_mom` reads the diffusion coefficient at `k-1`, `frm` reads `frh` at `k-1`, and the dead
circulation source reads `frm` at `k+1`. In each case the array read is not the array written.
That the four resulting stencils are bit-exact `Exact()` on `embedded`, `gtfn_cpu` and `dace_cpu`
is the confirmation.

**Aliasing that looks like a dependence.** `turbdiff` reuses storage aggressively, and the reuse
imposes an ordering on the Fortran that has no counterpart here. Section 6) writes the TKE
diffusion coefficient into `zaux(:,:,2)`, averages it onto the flux levels as `expl_mom`, and
then overwrites the same slot with the saved TKE profile: the second loop *must* run before the
third. In the port the coefficient is an intermediate inside
`compute_explicit_tke_diffusion_momentum`, so the two programs are independent and
`test_the_two_zaux_programs_do_not_constrain_each_others_order` asserts that they stay so. Expect
this pattern; it is the same one section 1a) documents for `zvari`.

**A consequence worth planning for.** An intermediate that the Fortran writes into a storage it
later reuses does not reach a savepoint and has no oracle. The coefficient `c_diff*l*q` above is
one, and the only way to validate it is through the quantity that consumes it. Do not build a
stencil whose sole output is such an intermediate; fold it into its consumer, where the reference
data can see it.
