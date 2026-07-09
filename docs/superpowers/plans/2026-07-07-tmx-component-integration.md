# TMX-as-Component Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register the GT4Py TMX turbulence granule (PR #1359) as a second process in the `PhysicsDriver`, with zero surface fluxes, full momentum coupling, savepoint-verified wrappers, and an APE_aes driver smoke test.

**Architecture:** Muphys-pattern L4 adapter: new files only (`data.py`, `state_stencils.py`, `state.py`, `component.py`, `static_fields.py`) inside the tmx package; `TmxState` implements the `PhysicsState` protocol (gather diagnoses T/p/u/v/air_mass/cv_air from the prognostic state; scatter applies tracer/exner/vn/w updates); `TmxComponent` implements the `Component` protocol around `Tmx.run()`. No PR-owned file is edited.

**Tech Stack:** Python 3.10+, GT4Py (`gt4py.next`), icon4py common (field factories, stencils, `PhysicsState`/`Component` protocols), pytest with icon4py serialbox datatests, `uv` workspace.

**Spec:** `docs/superpowers/specs/2026-07-07-tmx-component-integration-design.md`

## Global Constraints

- Branch: `physics_driver_tmx`. Never edit files that PR #1359 owns (everything currently under `model/atmosphere/subgrid_scale_physics/tmx/` — we only ADD files there).
- All work happens in the git worktree `/Users/chenyilu/Desktop/EXCLAIM/icon4py/.worktrees/physics_driver_tmx`; all commands run from that worktree root with `uv run`. Never touch the main checkout (pinned to `main`) or the sibling l2 worktree.
- **2026-07-09 amendment:** `physics_driver_l2` (at `9524a283b`) was merged in (`4598b4ecf`). Interface change that binds Tasks 4/5/9/10: `PhysicsState.scatter_to_prognostic` and `PhysicsDriver.run` now take `dtime: datetime.timedelta` (not `dt: float`); convert with `dtime.total_seconds()` only at the GT4Py stencil boundary (see the updated `muphys/state.py` for the pattern).
- Known pre-existing failure: `uv run tach check` exits 1 on this branch AND on untouched `physics_driver_l2` (spurious "does not depend on icon4py.model.common" for every module). Do not try to fix it; only ensure the ❌ count does not grow beyond the baseline (11 modules here).
- Datatests use experiment `exclaim_ape_aesPhys` v06 (`test_defs.Experiments.EXCLAIM_APE_AES`); data auto-downloads on first datatest run.
- Copyright header: every new `.py` file starts with the 7-line ICON4Py BSD-3-Clause header (copy from any existing file).
- Verification tolerance for savepoint comparisons: reuse `utils.assert_scaled_allclose` (rtol=1e-11, atol_scale=1e-9) from the tmx integration tests.
- Path shorthands used below:
  - `TMXPKG` = `model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx`
  - `TMXTESTS` = `model/atmosphere/subgrid_scale_physics/tmx/tests/tmx`
- Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: Baseline — verify the PR's TMX datatest runs locally

No code. Proves the environment works and fetches the v06 archive before any of our work depends on it.

**Files:** none.

- [ ] **Step 1: Run the PR's full-run TMX datatest**

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/integration_tests/test_tmx_run.py -v`
Expected: PASS (first run downloads `mpitask1_exclaim_ape_aesPhys_v06` into `testdata/ser_icondata/`; allow several minutes). If it fails with a download/permission error, stop and report — everything downstream needs this data.

- [ ] **Step 2: Confirm the archive landed**

Run: `ls testdata/ser_icondata/ | grep aesPhys`
Expected output contains: `mpitask1_exclaim_ape_aesPhys_v06` (v05 may also still be there; fine).

---

### Task 2: `data.py` — Component I/O metadata

**Files:**
- Create: `TMXPKG/data.py`
- Test: `TMXTESTS/unit_tests/__init__.py` (empty file with copyright header), `TMXTESTS/unit_tests/test_data.py`

**Interfaces:**
- Produces: `INPUTS_PROPERTIES: dict[str, model.FieldMetaData]` with exactly the 21 keys: `temperature, virtual_temperature, pressure, pressure_ifc, u, v, w, qv, qc, qi, qr, qs, qg, rho, air_mass, cv_air, evapotranspiration, sensible_heat_flux, u_stress, v_stress, q_snocpymlt`.
- Produces: `OUTPUTS_PROPERTIES: dict[str, model.FieldMetaData]` with exactly the 15 keys: `ddt_temperature, ddt_qv, ddt_qc, ddt_qi, ddt_u, ddt_v, ddt_w` (each with `kind="tendency"` semantics via `data.tendency_of`) and `km, kh, heating, dissip_ke, cptgz_vi, dissip_ke_vi, int_energy_vi, int_energy_vi_tend` (diagnostics).

- [ ] **Step 1: Write the failing test**

```python
# TMXTESTS/unit_tests/test_data.py  (add copyright header)
import dataclasses

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import data as tmx_data, tmx_states


def test_inputs_cover_input_and_surface_flux_states():
    state_keys = {f.name for f in dataclasses.fields(tmx_states.TmxInputState)} | {
        f.name for f in dataclasses.fields(tmx_states.TmxSurfaceFluxState)
    }
    assert set(tmx_data.INPUTS_PROPERTIES) == state_keys


def test_outputs_cover_tendencies_and_diagnostics():
    tendency_keys = {f.name for f in dataclasses.fields(tmx_states.TmxTendencyState)}
    diagnostic_keys = {
        "km", "kh", "heating", "dissip_ke",
        "cptgz_vi", "dissip_ke_vi", "int_energy_vi", "int_energy_vi_tend",
    }
    assert set(tmx_data.OUTPUTS_PROPERTIES) == tendency_keys | diagnostic_keys
    for key in tendency_keys:
        assert "tendency" in tmx_data.OUTPUTS_PROPERTIES[key]["standard_name"] or (
            tmx_data.OUTPUTS_PROPERTIES[key].get("long_name", "").startswith("tendency of")
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (no `tmx.data` yet).

- [ ] **Step 3: Write `TMXPKG/data.py`**

Follow the muphys template (`model/atmosphere/subgrid_scale_physics/muphys/src/.../muphys/data.py`). Reuse `icon4py.model.common.states.data` entries where they exist (`DIAGNOSTIC_CF_ATTRIBUTES`: `temperature`, `virtual_temperature`, `pressure`, `eastward_wind` (icon_var_name `u`), `northward_wind` (icon_var_name `v`), `upward_air_velocity`; `COMMON_TRACER_CF_ATTRIBUTES["q{v,c,i,r,s,g}"]`; `PROGNOSTIC_CF_ATTRIBUTES["air_density"]`; `TENDENCY_CF_ATTRIBUTES` for temperature/qv/qc/qi; `data.tendency_of(...)` for the u/v/w tendencies). Define new `model.FieldMetaData` entries inline for fields without a common identity, copying units from the `tmx_states.py` docstrings:

```python
# TMXPKG/data.py  (add copyright header)
"""Field metadata for the TmxComponent input/output contract."""

from __future__ import annotations

from icon4py.model.common.states import data, model


_TMX_ONLY_INPUTS: dict[str, model.FieldMetaData] = {
    "pressure_ifc": model.FieldMetaData(
        standard_name="air_pressure_on_interface_levels", units="Pa"
    ),
    "air_mass": model.FieldMetaData(standard_name="air_mass_per_unit_area", units="kg m-2"),
    "cv_air": model.FieldMetaData(
        standard_name="isometric_heat_capacity_of_moist_air_per_unit_area", units="J m-2 K-1"
    ),
    "evapotranspiration": model.FieldMetaData(
        standard_name="surface_evapotranspiration_flux", units="kg m-2 s-1"
    ),
    "sensible_heat_flux": model.FieldMetaData(
        standard_name="surface_upward_sensible_heat_flux", units="W m-2"
    ),
    "u_stress": model.FieldMetaData(standard_name="surface_downward_eastward_stress", units="N m-2"),
    "v_stress": model.FieldMetaData(standard_name="surface_downward_northward_stress", units="N m-2"),
    "q_snocpymlt": model.FieldMetaData(
        standard_name="heating_used_to_melt_snow_on_canopy", units="W m-2"
    ),
}

INPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "temperature": data.DIAGNOSTIC_CF_ATTRIBUTES["temperature"],
    "virtual_temperature": data.DIAGNOSTIC_CF_ATTRIBUTES["virtual_temperature"],
    "pressure": data.DIAGNOSTIC_CF_ATTRIBUTES["pressure"],
    "u": data.DIAGNOSTIC_CF_ATTRIBUTES["eastward_wind"],
    "v": data.DIAGNOSTIC_CF_ATTRIBUTES["northward_wind"],
    "w": data.DIAGNOSTIC_CF_ATTRIBUTES["upward_air_velocity"],
    "rho": data.PROGNOSTIC_CF_ATTRIBUTES["air_density"],
    **{f"q{s}": data.COMMON_TRACER_CF_ATTRIBUTES[f"q{s}"] for s in "vcirsg"},
    **_TMX_ONLY_INPUTS,
}

OUTPUTS_PROPERTIES: dict[str, model.FieldMetaData] = {
    "ddt_temperature": data.TENDENCY_CF_ATTRIBUTES["temperature"],
    "ddt_qv": data.TENDENCY_CF_ATTRIBUTES["qv"],
    "ddt_qc": data.TENDENCY_CF_ATTRIBUTES["qc"],
    "ddt_qi": data.TENDENCY_CF_ATTRIBUTES["qi"],
    "ddt_u": data.tendency_of(data.DIAGNOSTIC_CF_ATTRIBUTES["eastward_wind"]),
    "ddt_v": data.tendency_of(data.DIAGNOSTIC_CF_ATTRIBUTES["northward_wind"]),
    "ddt_w": data.tendency_of(data.DIAGNOSTIC_CF_ATTRIBUTES["upward_air_velocity"]),
    "km": model.FieldMetaData(
        standard_name="mass_weighted_turbulent_viscosity", units="kg m-1 s-1"
    ),
    "kh": model.FieldMetaData(
        standard_name="mass_weighted_turbulent_diffusivity", units="kg m-1 s-1"
    ),
    "heating": model.FieldMetaData(standard_name="turbulent_heating_rate", units="W m-2"),
    "dissip_ke": model.FieldMetaData(
        standard_name="kinetic_energy_dissipation_rate", units="W m-2"
    ),
    "cptgz_vi": model.FieldMetaData(
        standard_name="vertically_integrated_dry_static_energy", units="J m-2"
    ),
    "dissip_ke_vi": model.FieldMetaData(
        standard_name="vertically_integrated_kinetic_energy_dissipation_rate", units="W m-2"
    ),
    "int_energy_vi": model.FieldMetaData(
        standard_name="vertically_integrated_internal_energy", units="J m-2"
    ),
    "int_energy_vi_tend": model.FieldMetaData(
        standard_name="tendency_of_vertically_integrated_internal_energy", units="W m-2"
    ),
}
```

Adjust the exact `model.FieldMetaData` construction to how the class is defined (it may be a `TypedDict` — check `model/common/src/icon4py/model/common/states/model.py` and mirror how muphys `data.py` constructs entries; if it's a TypedDict use `dict(standard_name=..., units=...)`). The `test_outputs_cover_tendencies_and_diagnostics` metadata-access syntax (`[...]["standard_name"]`) must match too.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_data.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx/data.py model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/
git commit -m "feat(tmx): add Component I/O metadata contract (data.py)"
```

---

### Task 3: `state_stencils.py` — air_mass and cv_air

**Files:**
- Create: `TMXPKG/state_stencils.py`
- Test: `TMXTESTS/unit_tests/test_state_stencils.py`

**Interfaces:**
- Produces GT4Py programs:
  - `compute_air_mass(rho, ddqz_z_full, air_mass, horizontal_start, horizontal_end, vertical_start, vertical_end)` — `air_mass = rho * ddqz_z_full` (all `fa.CellKField[ta.wpfloat]`).
  - `compute_cv_air(qv, qc, qi, qr, qs, qg, air_mass, cv_air, horizontal_start, horizontal_end, vertical_start, vertical_end)` — port of `get_cvair` in `icon-mpim/src/atm_phy_aes/mo_aes_phy_diag.f90:215-252`:
    `cv = cvd*(1 - qtot) + cvv*qv + clw*(qc+qr) + ci*(qi+qs+qg)`, `cv_air = cv * air_mass`, with `qtot = qv+qc+qr+qi+qs+qg`.

- [ ] **Step 1: Write the failing test**

```python
# TMXTESTS/unit_tests/test_state_stencils.py  (add copyright header)
import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import state_stencils
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import test_utils


def _run(program, grid, **fields):
    program.with_grid_type(gtx.GridType.UNSTRUCTURED)(
        **fields,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(grid.num_cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(grid.num_levels),
        offset_provider={},
    )


def test_compute_air_mass_is_rho_dz():
    grid = simple.simple_grid()
    rho = data_alloc.constant_field(grid, 1.2, dims.CellDim, dims.KDim)
    dz = data_alloc.constant_field(grid, 250.0, dims.CellDim, dims.KDim)
    air_mass = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
    _run(state_stencils.compute_air_mass, grid, rho=rho, ddqz_z_full=dz, air_mass=air_mass)
    test_utils.assert_dallclose(air_mass.asnumpy(), 1.2 * 250.0)


def test_compute_cv_air_matches_fortran_formula():
    grid = simple.simple_grid()
    q = dict(qv=1e-3, qc=2e-4, qi=1e-4, qr=5e-5, qs=3e-5, qg=1e-5)
    fields = {k: data_alloc.constant_field(grid, val, dims.CellDim, dims.KDim) for k, val in q.items()}
    air_mass = data_alloc.constant_field(grid, 300.0, dims.CellDim, dims.KDim)
    cv_air = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
    _run(state_stencils.compute_cv_air, grid, **fields, air_mass=air_mass, cv_air=cv_air)

    qtot = sum(q.values())
    cv = (
        constants.CVD * (1.0 - qtot)
        + constants.CVV * q["qv"]
        + constants.CPL * (q["qc"] + q["qr"])
        + constants.SPECIFIC_HEAT_CAPACITY_ICE * (q["qi"] + q["qs"] + q["qg"])
    )
    test_utils.assert_dallclose(cv_air.asnumpy(), cv * 300.0)
```

Note: if `simple.simple_grid` is spelled differently (e.g. `simple.SimpleGrid()`), match how `muphys/tests/muphys/unit_tests/test_state.py` constructs its grid, and reuse its program-invocation helper if the raw `.with_grid_type(...)` call above doesn't match how other unit tests invoke programs (check `test_utils` / `model_options.setup_program` usage there first and copy that pattern).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_state_stencils.py -v`
Expected: FAIL with `ImportError` (no `state_stencils`).

- [ ] **Step 3: Write `TMXPKG/state_stencils.py`**

```python
# TMXPKG/state_stencils.py  (add copyright header)
"""Adapter stencils of the TmxState (dycore -> tmx input translation).

``compute_air_mass``: mair = rho * dz (``diag%airmass_new`` bound to
``field%mair`` in mo_interface_iconam_aes.f90; shallow atmosphere).
``compute_cv_air``: port of ``get_cvair`` in mo_aes_phy_diag.f90.
"""

import gt4py.next as gtx

from icon4py.model.common import constants, dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_air_mass(
    rho: fa.CellKField[wpfloat],
    ddqz_z_full: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    return rho * ddqz_z_full


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_air_mass(
    rho: fa.CellKField[wpfloat],
    ddqz_z_full: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_air_mass(
        rho=rho,
        ddqz_z_full=ddqz_z_full,
        out=air_mass,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_cv_air(
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    # locals, not module globals: see tmx/docs/gt4py_patterns.md (gtfn backend)
    cvd = constants.PhysicsConstants.cvd
    cvv = constants.PhysicsConstants.cvv
    clw = constants.PhysicsConstants.cpl
    ci = wpfloat(2108.0)  # SPECIFIC_HEAT_CAPACITY_ICE; use the enum member if one exists
    qliq = qc + qr
    qice = qi + qs + qg
    qtot = qv + qliq + qice
    cv = cvd * (wpfloat(1.0) - qtot) + cvv * qv + clw * qliq + ci * qice
    return cv * air_mass


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cv_air(
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    cv_air: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_cv_air(
        qv=qv, qc=qc, qi=qi, qr=qr, qs=qs, qg=qg,
        air_mass=air_mass,
        out=cv_air,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
```

Check `constants.PhysicsConstants` for an ice-heat-capacity member (grep `2108` / `ice` in `constants.py`); if present use it instead of the literal.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_state_stencils.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx/state_stencils.py model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_state_stencils.py
git commit -m "feat(tmx): add air_mass and cv_air adapter stencils"
```

---

### Task 4: `state.py` — TmxState gather + as_component_input

**Files:**
- Create: `TMXPKG/state.py`
- Test: `TMXTESTS/unit_tests/test_state.py`

**Interfaces:**
- Consumes: `state_stencils.compute_air_mass`, `state_stencils.compute_cv_air` (Task 3).
- Produces: `class TmxState(PhysicsState)` with:
  - `__init__(self, *, grid, ddqz_z_full, rbf_coeff_c1, rbf_coeff_c2, c_lin_e, primal_normal_cell_x, primal_normal_cell_y, backend=None)` — the last five are for gather (u,v from vn) and scatter (vn projection, Task 5); field types as in `tmx_states.TmxInterpolationState` / `EdgeParams`.
  - `gather_from_prognostic(prognostic, tracers) -> None`
  - `as_component_input() -> dict[str, ...]` returning exactly the 21 `INPUTS_PROPERTIES` keys (Task 2).
  - surface-flux buffers owned, zero-filled once at construction, returned by reference (phase-2 seam).

- [ ] **Step 1: Write the failing test**

Follow the muphys pattern (`muphys/tests/muphys/unit_tests/test_state.py` — reuse its `_uniform_prognostic` / tracer-state helpers by copying them in; grids come from `icon4py.model.common.grid.simple`). Note the RBF/edge stencils need connectivity offset providers — the simple grid provides them (`grid.connectivities` / offset provider protocol); mirror how existing unit tests of `edge_2_cell_vector_rbf_interpolation` set that up (see `model/common/tests/.../interpolation` stencil tests) if plain construction fails.

```python
# TMXTESTS/unit_tests/test_state.py  (add copyright header)
import dataclasses

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import data as tmx_data
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.state import TmxState
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.states import prognostic_state as prognostics, tracer_state
from icon4py.model.common.utils import data_allocation as data_alloc


def _tmx_state(grid):
    e2c_shape = (grid.num_edges, 2)
    return TmxState(
        grid=grid,
        ddqz_z_full=data_alloc.constant_field(grid, 100.0, dims.CellDim, dims.KDim),
        rbf_coeff_c1=data_alloc.zero_field(grid, dims.CellDim, dims.C2E2C2EDim),
        rbf_coeff_c2=data_alloc.zero_field(grid, dims.CellDim, dims.C2E2C2EDim),
        c_lin_e=data_alloc.constant_field(grid, 0.5, dims.EdgeDim, dims.E2CDim),
        primal_normal_cell_x=data_alloc.constant_field(grid, 1.0, dims.EdgeDim, dims.E2CDim),
        primal_normal_cell_y=data_alloc.zero_field(grid, dims.EdgeDim, dims.E2CDim),
        backend=None,
    )


def test_as_component_input_matches_contract():
    grid = simple.simple_grid()
    state = _tmx_state(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)  # copy helper from muphys test_state
    tracers = _tracer_state(grid, qv=1e-3)                             # copy helper from muphys test_state
    state.gather_from_prognostic(prognostic, tracers)
    inp = state.as_component_input()
    assert set(inp) == set(tmx_data.INPUTS_PROPERTIES)


def test_gather_computes_air_mass_and_zero_surface_fluxes():
    grid = simple.simple_grid()
    state = _tmx_state(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    state.gather_from_prognostic(prognostic, tracers)
    inp = state.as_component_input()
    np.testing.assert_allclose(
        inp["air_mass"].asnumpy(), prognostic.rho.asnumpy() * 100.0, rtol=1e-14
    )
    for key in ("evapotranspiration", "sensible_heat_flux", "u_stress", "v_stress", "q_snocpymlt"):
        assert (inp[key].asnumpy() == 0.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_state.py -v`
Expected: FAIL with `ImportError` (no `tmx.state`).

- [ ] **Step 3: Write `TMXPKG/state.py` (gather half)**

Model directly on `muphys/src/.../muphys/state.py` (its `State` class shows the exact `setup_program` calls, the `_require` tracer guard, and the temperature/pressure diagnosis sequence — reuse all of it):

- Constructor: store grid sizes/backend; `setup_program` the same three diagnosis programs muphys uses (`diagnose_virtual_temperature_and_temperature`, `diagnose_surface_pressure`, `diagnose_pressure`) plus:
  - `edge_2_cell_vector_rbf_interpolation` (from `icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation`) with offset provider from the grid for `C2E2C2E` (mirror how another consumer sets its `offset_provider=` — grep usages in `model/`),
  - `state_stencils.compute_air_mass`, `state_stencils.compute_cv_air` (Task 3),
  - `compute_vn_from_uv` (used in Task 5; wire the program now, offset provider `E2C`).
- Owned buffers (`data_alloc.zero_field`): `temperature`, `virtual_temperature`, `pressure`, `pressure_ifc` (KDim+1), `u`, `v`, `air_mass`, `cv_air`; the five 2-D surface-flux buffers (`dims.CellDim` only); scratch for scatter (Task 5): `_new_te`, `_tv_tendency`, `_exner_tendency`, `_ddt_vn` (EdgeKField).
- References set by gather: `self._rho`, `self._w`, `self._vn`, `self._tracers`.
- `gather_from_prognostic`: bind refs; run temperature/pressure diagnosis exactly as muphys does; run `edge_2_cell_vector_rbf_interpolation(p_e_in=prognostic.vn, ptr_coeff_1=self._rbf_coeff_c1, ptr_coeff_2=self._rbf_coeff_c2, p_u_out=self.u, p_v_out=self.v, ...)`; run `compute_air_mass(rho=..., ddqz_z_full=..., air_mass=...)` then `compute_cv_air(...)`.
- `as_component_input`: dict of the 21 keys → owned buffers / references (tracers via `_require`, `w=self._w`, `rho=self._rho`).
- `scatter_to_prognostic`: for THIS task, `raise NotImplementedError` — implemented in Task 5.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_state.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx/state.py model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_state.py
git commit -m "feat(tmx): add TmxState gather / component-input adapter"
```

---

### Task 5: `state.py` — scatter_to_prognostic

**Files:**
- Modify: `TMXPKG/state.py` (replace the `NotImplementedError`)
- Test: `TMXTESTS/unit_tests/test_state.py` (append)

**Interfaces:**
- Consumes: `compute_vn_from_uv` from `TMXPKG/stencils/compute_vn_from_uv.py` (PR-owned, imported not edited); muphys' exner-update stencils (`calculate_virtual_temperature_tendency`, `calculate_exner_tendency`, `compute_field_a_plus_coeff_times_field_b_on_cell_k` — same modules muphys `state.py` imports).
- Produces: `scatter_to_prognostic(prognostic, outputs, dtime: datetime.timedelta)` where `outputs` has the 15 `OUTPUTS_PROPERTIES` keys — **the protocol takes a `datetime.timedelta` since the l2 merge**; first line `dt = dtime.total_seconds()` (scalar for stencils, mirror post-merge `muphys/state.py`). Apply order: (1) `qv/qc/qi += ddt*dt`; (2) `ddt_temperature` → exner via the muphys path (verbatim structure from muphys `scatter_to_prognostic` steps 2); (3) `compute_vn_from_uv` on `(ddt_u, ddt_v)` → `self._ddt_vn`, then `vn += dt*_ddt_vn`; (4) `w += dt*ddt_w`; (5) store diagnostics as attributes (`self.km`, `self.kh`, `self.heating`, `self.dissip_ke`, `self.cptgz_vi`, `self.dissip_ke_vi`, `self.int_energy_vi`, `self.int_energy_vi_tend`).

- [ ] **Step 1: Write the failing test (append to test_state.py)**

```python
def _tmx_outputs(grid, *, ddt_u=0.0, ddt_v=0.0, ddt_w=0.0, ddt_qv=0.0):
    def ck(value, **kw):
        return data_alloc.constant_field(grid, value, dims.CellDim, dims.KDim, **kw)

    out = {
        "ddt_temperature": ck(0.0),
        "ddt_qv": ck(ddt_qv), "ddt_qc": ck(0.0), "ddt_qi": ck(0.0),
        "ddt_u": ck(ddt_u), "ddt_v": ck(ddt_v),
        "ddt_w": data_alloc.constant_field(grid, ddt_w, dims.CellDim, dims.KDim, extend={dims.KDim: 1}),
        "km": ck(0.0), "kh": ck(0.0), "heating": ck(0.0), "dissip_ke": ck(0.0),
    }
    for key in ("cptgz_vi", "dissip_ke_vi", "int_energy_vi", "int_energy_vi_tend"):
        out[key] = data_alloc.constant_field(grid, 0.0, dims.CellDim)
    return out


def test_scatter_applies_qv_and_w_tendencies():
    grid = simple.simple_grid()
    state = _tmx_state(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    state.gather_from_prognostic(prognostic, tracers)
    dt = 300.0
    state.scatter_to_prognostic(
        prognostic, _tmx_outputs(grid, ddt_qv=1e-7, ddt_w=1e-4), datetime.timedelta(seconds=dt)
    )
    np.testing.assert_allclose(tracers.qv.asnumpy(), 1e-3 + 1e-7 * dt, rtol=1e-12)
    np.testing.assert_allclose(prognostic.w.asnumpy(), 1e-4 * dt, rtol=1e-12)


def test_scatter_projects_wind_tendency_to_vn():
    # uniform ddt_u = 1e-4, ddt_v = 0; primal_normal_cell_x = 1, c_lin_e = 0.5 (two neighbors)
    # => ddt_vn = 2 * 0.5 * 1e-4 * 1.0 = 1e-4 on interior edges
    grid = simple.simple_grid()
    state = _tmx_state(grid)
    prognostic = _uniform_prognostic(grid, exner=0.95, theta_v=300.0)
    tracers = _tracer_state(grid, qv=1e-3)
    state.gather_from_prognostic(prognostic, tracers)
    dt = 300.0
    state.scatter_to_prognostic(
        prognostic, _tmx_outputs(grid, ddt_u=1e-4), datetime.timedelta(seconds=dt)
    )
    np.testing.assert_allclose(prognostic.vn.asnumpy(), 1e-4 * dt, rtol=1e-12)
```

(The simple grid is periodic, all edges have two neighbors, so the uniform-field expectation holds everywhere; if boundary rows differ, restrict the assertion with the grid's interior edge slice.)

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_state.py -v`
Expected: the two new tests FAIL with `NotImplementedError`; the Task-4 tests still PASS.

- [ ] **Step 3: Implement scatter (order and stencils as in the Interfaces block above)**

Copy the muphys `scatter_to_prognostic` structure for steps 1–2 (tracers, then tend_T → Tv tendency → exner tendency → exner update), but only for qv/qc/qi. For momentum: run the pre-wired `compute_vn_from_uv` program with `u=outputs["ddt_u"], v=outputs["ddt_v"], primal_normal_cell_x=..., primal_normal_cell_y=..., c_lin_e=..., vn=self._ddt_vn`, then apply. The generic apply program (`compute_field_a_plus_coeff_times_field_b_on_cell_k`) is cell-based: for `w` (KDim+1) reuse it but set up the program instance with `vertical_end=gtx.int32(nlev + 1)`; for `vn` (edges), first grep `generic_math_operations` for an edge variant — if none exists, add this to `state_stencils.py`:

```python
@gtx.field_operator
def _apply_tendency_on_edge_k(
    field_a: fa.EdgeKField[wpfloat],
    coeff: wpfloat,
    field_b: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    return field_a + coeff * field_b


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_tendency_on_edge_k(
    field_a: fa.EdgeKField[wpfloat],
    coeff: wpfloat,
    field_b: fa.EdgeKField[wpfloat],
    output_field: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_tendency_on_edge_k(
        field_a=field_a, coeff=coeff, field_b=field_b,
        out=output_field,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_state.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx/state.py model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx/state_stencils.py model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_state.py
git commit -m "feat(tmx): TmxState scatter with exner, tracer, vn and w updates"
```

---

### Task 6: `component.py` — TmxComponent

**Files:**
- Create: `TMXPKG/component.py`
- Test: `TMXTESTS/unit_tests/test_component.py`

**Interfaces:**
- Consumes: `tmx.Tmx`, `tmx.TmxConfig`, `tmx.TmxParams`, `tmx_states.*` (PR-owned, imported); `data.py` (Task 2).
- Produces:

```python
class TmxComponent:
    inputs_properties = tmx_data.INPUTS_PROPERTIES
    outputs_properties = tmx_data.OUTPUTS_PROPERTIES

    def __init__(
        self, *,
        grid, config: tmx.TmxConfig,
        metric_state: tmx_states.TmxMetricState,
        interpolation_state: tmx_states.TmxInterpolationState,
        edge_params, cell_params,
        dtime: datetime.timedelta,
        backend=None,
        exchange=decomposition.single_node_exchange,
        granule=None,  # test seam: inject a fake instead of building Tmx
    ) -> None: ...

    def __call__(self, state: dict, time_step: datetime.datetime) -> dict: ...
```

  - Constructor builds `Tmx(grid=..., config=..., params=tmx.TmxParams(config), vertical_grid=None, metric_state=..., interpolation_state=..., edge_params=..., cell_params=..., backend=..., exchange=...)` when `granule is None`; allocates `TmxDiagnosticState.allocate(grid, allocator)`, `TmxTendencyState.allocate(...)`, `TmxNewState.allocate(...)` once (`model_backends.get_allocator(backend)`).
  - `__call__` packs `tmx_states.TmxInputState(**{f: state[f] for f in input fields})` and `TmxSurfaceFluxState(**...)` (frozen dataclasses of references — build fresh each call, no copy), calls `self._granule.run(input_state=..., surface_flux_state=..., diagnostic_state=..., tendency_state=..., new_state=..., dtime=self._dt_seconds)`, returns the 15-key outputs dict referencing `self._tendency_state.*` and `self._diagnostic_state.{km,kh,heating,dissip_ke,cptgz_vi,dissip_ke_vi,int_energy_vi,int_energy_vi_tend}`.

- [ ] **Step 1: Write the failing test**

```python
# TMXTESTS/unit_tests/test_component.py  (add copyright header)
import datetime

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import (
    component as tmx_component,
    data as tmx_data,
    tmx_states,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc


class _FakeGranule:
    def __init__(self):
        self.calls = []

    def run(self, *, input_state, surface_flux_state, diagnostic_state, tendency_state, new_state, dtime):
        self.calls.append(dtime)


def _input_dict(grid):
    ck = lambda: data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
    half = lambda: data_alloc.zero_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
    cell = lambda: data_alloc.zero_field(grid, dims.CellDim)
    d = {k: ck() for k in tmx_data.INPUTS_PROPERTIES}
    d["w"] = half()
    d["pressure_ifc"] = half()
    for k in ("evapotranspiration", "sensible_heat_flux", "u_stress", "v_stress", "q_snocpymlt"):
        d[k] = cell()
    return d


def test_call_runs_granule_and_returns_output_contract():
    grid = simple.simple_grid()
    fake = _FakeGranule()
    comp = tmx_component.TmxComponent(
        grid=grid, config=None, metric_state=None, interpolation_state=None,
        edge_params=None, cell_params=None,
        dtime=datetime.timedelta(seconds=300), backend=None, granule=fake,
    )
    out = comp(_input_dict(grid), datetime.datetime(2008, 9, 1))
    assert fake.calls == [300.0]
    assert set(out) == set(tmx_data.OUTPUTS_PROPERTIES)
    assert out["ddt_qv"] is comp._tendency_state.ddt_qv
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_component.py -v`
Expected: FAIL with `ImportError` (no `tmx.component`).

- [ ] **Step 3: Implement `TMXPKG/component.py` per the Interfaces block**

Guard: `config`/`metric_state`/etc. may be `None` only when `granule` is injected — assert that. Mirror muphys `component.py` for structure, docstring style and the `inputs_properties`/`outputs_properties` class attributes.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_component.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx/component.py model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_component.py
git commit -m "feat(tmx): add TmxComponent (Component protocol around Tmx.run)"
```

---

### Task 7: `static_fields.py` — TMX static states from field factories

**Files:**
- Create: `TMXPKG/static_fields.py`
- Test: `TMXTESTS/integration_tests/test_static_fields.py` (datatest)

**Interfaces:**
- Consumes: geometry/interpolation/metrics field sources (the driver's `StaticFieldFactories` members), attribute constants:
  - metrics: `DDQZ_Z_FULL`, `INV_DDQZ_Z_FULL`, `DDQZ_Z_HALF`, `DDQZ_Z_FULL_E`, `WGTFAC_C`, `WGTFAC_E`, `WGTFACQ_C`, `WGTFACQ_E`, `Z_MC`, `Z_IFC` (`metrics_attributes`)
  - interpolation: `C_LIN_E`, `E_BLN_C_S`, `GEOFAC_DIV`, `CELL_AW_VERTS`, `RBF_VEC_COEFF_V1/V2/E/C1/C2` (`interpolation_attributes`)
  - geometry: `EDGE_CELL_DISTANCE` (`geometry_attributes`)
- Produces: `build_tmx_static_states(*, grid, geometry_source, interpolation_source, metrics_source, backend) -> tuple[tmx_states.TmxMetricState, tmx_states.TmxInterpolationState]`

Derivations (numpy at init time — one-shot, then `gtx.as_field(..., allocator=...)`):
- `inv_ddqz_z_half = 1 / ddqz_z_half` (mo_vertical_grid.f90:2159)
- `inv_ddqz_z_full_e = 1 / ddqz_z_full_e` (mo_vertical_grid.f90:~2140)
- `inv_ddqz_z_half_e` = cells→edges of `inv_ddqz_z_half` with `c_lin_e` over `E2C` (mo_vertical_grid.f90:2184, `cells2edges_scalar`)
- `inv_ddqz_z_half_v` = cells→vertices of `inv_ddqz_z_half` with `cells_aw_verts` over `V2C` (mo_vertical_grid.f90:2174, `cells2verts_scalar`)
- `geopot_agl_ifc = grav * (z_ifc - z_ifc[:, -1:])` (telescoped form of mo_vertical_grid.f90:241-248)
- `wgtfacq_c` / `wgtfacq_e`: factory fields are in DSL (bottom-up) order with 3 used rows — convert to the Fortran coefficient order TMX expects by mirroring `flip_back` in `TMXTESTS/integration_tests/utils.py` (verify the factory field's row convention against `compute_weight_factors.compute_wgtfacq_c_dsl`, which returns `[:, -3:]` in DSL order).
- `wgtfacq1_c` (top-boundary quadratic coefficients, mo_vertical_grid.f90:953-967, 0-based numpy):

```python
z1 = 0.5 * (z_ifc[:, 1] - z_ifc[:, 0])
z2 = 0.5 * (z_ifc[:, 1] + z_ifc[:, 2]) - z_ifc[:, 0]
z3 = 0.5 * (z_ifc[:, 2] + z_ifc[:, 3]) - z_ifc[:, 0]
w3 = z1 * z2 / ((z2 - z3) * (z1 - z3))
w2 = (z1 - w3 * (z1 - z3)) / (z1 - z2)
w1 = 1.0 - (w2 + w3)
wgtfacq1_c = np.stack([w1, w2, w3], axis=1)  # Fortran order: row k multiplies full level k
```

- `wgtfacq1_e`: mirror `compute_weight_factors.compute_wgtfacq_e_dsl` (which builds edge coefficients from the cell aux coefficients interpolated with `c_lin_e`), substituting the top-boundary `z1/z2/z3` above; cross-check against `mo_vertical_grid.f90` around lines 989-1014 (`z_aux_c(:,4:6,:) = wgtfacq1_c` then edge interpolation). The datatest in Step 1 is the arbiter — iterate until it passes.
- `edge_cell_length = geometry_source.get(geometry_attributes.EDGE_CELL_DISTANCE)`

- [ ] **Step 1: Write the failing datatest**

Compare every factory-built field against the savepoint-built states of the PR's own test helpers. Reuse the PR's fixtures (`TMXTESTS/fixtures.py` re-exports the icon4py datatest fixtures: `grid_savepoint`, `metrics_savepoint`, `interpolation_savepoint`, `init_savepoint`, `data_provider`, `icon_grid`, `backend` — copy the fixture-import block from `TMXTESTS/integration_tests/test_tmx_run.py` verbatim, including the `experiment_description` parametrization over `EXCLAIM_APE_AES`).

```python
# TMXTESTS/integration_tests/test_static_fields.py  (header + fixture imports as in test_tmx_run.py)
import dataclasses

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import static_fields
from ..integration_tests import utils

# The factory sources are built exactly like the standalone driver builds them:
# reuse driver_utils.create_static_field_factories with the savepoint grid manager
# IF a ready-made datatest fixture for factories exists (grep model/testing and
# model/common tests for `MetricsFieldsFactory(` usage in datatests and copy that
# construction); otherwise build the three factories directly as in
# driver_utils.create_static_field_factories.


@pytest.mark.datatest
def test_factory_static_states_match_savepoints(
    icon_grid, grid_savepoint, metrics_savepoint, interpolation_savepoint, init_savepoint, backend
):
    allocator = model_backends.get_allocator(backend)
    metric_ref = utils.construct_metric_state(
        metrics_savepoint=metrics_savepoint, init_savepoint=init_savepoint,
        grid_savepoint=grid_savepoint, allocator=allocator,
    )
    interp_ref = utils.construct_interpolation_state(interpolation_savepoint)

    metric_actual, interp_actual = static_fields.build_tmx_static_states(
        grid=icon_grid,
        geometry_source=geometry_source,          # from the factory construction above
        interpolation_source=interpolation_source,
        metrics_source=metrics_source,
        backend=backend,
    )
    for f in dataclasses.fields(metric_ref):
        utils.assert_scaled_allclose(
            getattr(metric_actual, f.name).asnumpy(),
            getattr(metric_ref, f.name).asnumpy(),
            err_msg=f.name,
        )
    for f in dataclasses.fields(interp_ref):
        utils.assert_scaled_allclose(
            getattr(interp_actual, f.name).asnumpy(),
            getattr(interp_ref, f.name).asnumpy(),
            err_msg=f.name,
        )
```

- [ ] **Step 2: Run to verify it fails** (`ImportError`)

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/integration_tests/test_static_fields.py -v`

- [ ] **Step 3: Implement `TMXPKG/static_fields.py` per the Interfaces block**

- [ ] **Step 4: Run the datatest until every field matches**

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/integration_tests/test_static_fields.py -v`
Expected: PASS with all ~26 field comparisons green. The likely iteration spots: `wgtfacq*` row order, `wgtfacq1_e`, and halo rows of the vertex/edge interpolations (if only halo rows differ, restrict the comparison to owned entries the way the PR's MPI test does, and note it in the test docstring).

- [ ] **Step 5: Commit**

```bash
git add model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx/static_fields.py model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/integration_tests/test_static_fields.py
git commit -m "feat(tmx): build TmxMetricState/TmxInterpolationState from field factories"
```

---

### Task 8: Component end-to-end datatest (entry → exit through the wrapper)

The design's primary DoD: `TmxComponent` driven through the dict interface reproduces the PR's `tmx-exit` savepoint.

**2026-07-09 amendment:** the l2 merge added `muphys/tests/muphys/integration_tests/test_muphys_datatest.py` — read it as the PRIMARY structural template for a component-level datatest (alongside `test_tmx_run.py` for the tmx savepoint accessors).

**Files:**
- Test: `TMXTESTS/integration_tests/test_component_datatest.py`

**Interfaces:**
- Consumes: `TmxComponent` (Task 6), savepoint constructors from `TMXTESTS/integration_tests/utils.py`, PR fixtures (`tmx_config`, `tmx_dtime`, savepoints — from `TMXTESTS/fixtures.py`, same imports as `test_tmx_run.py`).

- [ ] **Step 1: Write the failing test**

Mirror `test_tmx_run.py` construction, but route through the Component contract:

```python
@pytest.mark.datatest
@pytest.mark.parametrize("date", utils.TMX_DATES)
def test_tmx_component_reproduces_exit_savepoint(
    date, icon_grid, grid_savepoint, metrics_savepoint, interpolation_savepoint,
    init_savepoint, data_provider, tmx_config, tmx_dtime, backend,
):
    allocator = model_backends.get_allocator(backend)
    comp = tmx_component.TmxComponent(
        grid=icon_grid,
        config=tmx_config,
        metric_state=utils.construct_metric_state(
            metrics_savepoint=metrics_savepoint, init_savepoint=init_savepoint,
            grid_savepoint=grid_savepoint, allocator=allocator,
        ),
        interpolation_state=utils.construct_interpolation_state(interpolation_savepoint),
        edge_params=grid_savepoint.construct_edge_geometry(),
        cell_params=grid_savepoint.construct_cell_geometry(),
        dtime=datetime.timedelta(seconds=tmx_dtime),
        backend=backend,
    )
    entry = data_provider.from_savepoint_tmx_entry(date=date)        # match test_tmx_run.py's
    fluxes = data_provider.from_savepoint_tmx_surface_fluxes(date=date)  # savepoint accessors exactly
    exit_sp = data_provider.from_savepoint_tmx_exit(date=date)

    input_state = utils.construct_input_state(entry)
    flux_state = utils.construct_surface_flux_state(fluxes)
    state_dict = {
        **{f.name: getattr(input_state, f.name) for f in dataclasses.fields(input_state)},
        **{f.name: getattr(flux_state, f.name) for f in dataclasses.fields(flux_state)},
    }

    outputs = comp(state_dict, datetime.datetime.fromisoformat(date.rstrip("0").rstrip(".")))

    utils.verify_full_run_fields(
        diagnostic_state=comp._diagnostic_state,
        tendency_state=comp._tendency_state,
        exit_savepoint=exit_sp,
        num_levels=icon_grid.num_levels,
    )
    # and the dict view exposes the same objects
    assert outputs["ddt_temperature"] is comp._tendency_state.ddt_temperature
```

(Copy the exact savepoint accessor spellings and the `date=` handling from `test_tmx_run.py` — do not guess them.)

- [ ] **Step 2: Run to verify it fails** (ImportError / missing test file initially)

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/integration_tests/test_component_datatest.py -v`

- [ ] **Step 3: Fix until it passes**

Expected: 2 PASS (both dates), tolerances identical to the PR's own full-run test. Any mismatch here is a wrapper bug (packing order, dtime seconds, buffer reuse) — the granule itself is already verified by `test_tmx_run.py`.

- [ ] **Step 4: Also confirm a second consecutive `__call__` is clean, and verify the gather thermodynamics against the savepoints**

Add (inside the same test, after the first verification): call `comp(state_dict, ...)` a second time with the same inputs and verify `verify_full_run_fields` still passes — this catches state leaking between steps in the reused output buffers.

Then add a second test in the same file — the spec's gather-thermodynamics layer. The entry savepoint carries both the raw state (rho, q*) and the derived `mair`/`cvair`, so our stencils can be checked against Fortran on real data:

```python
@pytest.mark.datatest
@pytest.mark.parametrize("date", utils.TMX_DATES)
def test_gather_air_mass_and_cv_air_match_savepoint(
    date, icon_grid, metrics_savepoint, data_provider, backend
):
    entry = data_provider.from_savepoint_tmx_entry(date=date)  # accessor spelling from test_tmx_run.py
    grid = icon_grid
    air_mass = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
    cv_air = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
    # invoke the Task-3 programs directly (same setup_program pattern as in state.py)
    run_air_mass(rho=entry.rho(), ddqz_z_full=metrics_savepoint.ddqz_z_full(), air_mass=air_mass)
    run_cv_air(
        qv=entry.qv(), qc=entry.qc(), qi=entry.qi(),
        qr=entry.qr(), qs=entry.qs(), qg=entry.qg(),
        air_mass=air_mass, cv_air=cv_air,
    )
    utils.assert_scaled_allclose(air_mass.asnumpy(), entry.mair().asnumpy(), err_msg="mair")
    utils.assert_scaled_allclose(cv_air.asnumpy(), entry.cvair().asnumpy(), err_msg="cvair")
```

If `mair` disagrees only by a deep-atmosphere factor, the dycore's airmass includes `deepatmo` weights — take `ddqz_z_full` times the `deepatmo_divzL/U`-consistent thickness instead (check `mo_nh_diagnose_pres_temp`/airmass computation in icon-mpim) and record the finding in the spec's open-items table. If `cvair` disagrees, diff the constants (icon4py `SPECIFIC_HEAT_CAPACITY_ICE=2108.0` vs ICON's `ci`) before touching the formula.

Run: `uv run pytest model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/integration_tests/test_component_datatest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/integration_tests/test_component_datatest.py
git commit -m "test(tmx): component-level datatest against tmx entry/exit savepoints"
```

---

### Task 9: Driver config and registration

**2026-07-09 amendments:** (a) the l2 merge moved things around — before coding, confirm where the `muphys` config member lives now (`ExperimentConfig` in `model/testing/.../definitions.py` vs `standalone_driver/config.py`) and put the `tmx` member NEXT TO IT, wherever that is; (b) the muphys registration block in `driver_utils.initialize_granules` now passes `scheme=config.muphys.scheme` — leave it untouched; (c) `create_static_field_factories` gained `process_props` and `geometry_config` params — we only consume its outputs, no change needed; (d) the `tach check` step below is amended: it fails pre-existing on this branch (spurious "does not depend on common", 11 modules) — the gate is "no NEW ❌ lines beyond that baseline" after you add the standalone_driver→tmx dependency to `tach.toml`.

**Files:**
- Modify: `model/standalone_driver/src/icon4py/model/standalone_driver/config.py` (add `tmx` member to `ExperimentConfig`, default `None`; no auto-enable)
- Modify: `model/standalone_driver/src/icon4py/model/standalone_driver/driver_utils.py` (`initialize_granules`: register TMX process after muphys)
- Modify: `model/standalone_driver/pyproject.toml` (add `icon4py-atmosphere-tmx` dependency), `tach.toml` (add tmx to `icon4py.model.standalone_driver` `depends_on`)
- Test: `model/standalone_driver/tests/standalone_driver/unit_tests/test_config.py` (or nearest existing config test file — check what exists and append there)

**Interfaces:**
- Consumes: `TmxComponent` (Task 6), `TmxState` (Tasks 4-5), `build_tmx_static_states` (Task 7), `tmx.TmxConfig`.
- Produces: `ExperimentConfig.tmx: tmx.TmxConfig | None = None`. In `initialize_granules`, inside the existing `if config.muphys is not None:` block gains a sibling:

```python
if config.tmx is not None:
    tmx_metric_state, tmx_interpolation_state = tmx_static_fields.build_tmx_static_states(
        grid=grid,
        geometry_source=geometry_field_source,
        interpolation_source=interpolation_field_source,
        metrics_source=metrics_field_source,
        backend=backend,
    )
    tmx_process = physics_driver.PhysicsProcess(
        name="tmx",
        component=tmx_component.TmxComponent(
            grid=grid,
            config=config.tmx,
            metric_state=tmx_metric_state,
            interpolation_state=tmx_interpolation_state,
            edge_params=edge_geometry,
            cell_params=cell_geometry,
            dtime=config.driver.dtime,
            backend=backend,
            exchange=exchange,
        ),
        state=tmx_state.TmxState(
            grid=grid,
            ddqz_z_full=metrics_field_source.get(metrics_attributes.DDQZ_Z_FULL),
            rbf_coeff_c1=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_C1),
            rbf_coeff_c2=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_C2),
            c_lin_e=interpolation_field_source.get(interpolation_attributes.C_LIN_E),
            primal_normal_cell_x=edge_geometry.primal_normal_cell[0],
            primal_normal_cell_y=edge_geometry.primal_normal_cell[1],
            backend=backend,
        ),
        time_control=physics_driver.ProcessTimeControl(
            interval=config.driver.dtime,
            start_date=config.driver.start_of_simulation,
            end_date=model_time_variables.simulation_end_datetime,
            enable_process=True,
        ),
    )
    processes.append(tmx_process)
```

Restructure the tail of `initialize_granules` minimally: build `processes: list = []`, append muphys then tmx, and `physics_granule = physics_driver.PhysicsDriver(processes) if processes else None`. Check how `EdgeParams` exposes the primal-normal-cell pair (`primal_normal_cell_x`/`_y` attributes vs a tuple — match the actual `grid_states.EdgeParams` definition and the keyword names used at its construction site at `driver_utils.py:242-267`).

**Deliberate choice (documented in the spec):** `tmx` stays `None` unless a caller sets it (`dataclasses.replace(config, tmx=tmx.TmxConfig())`). The existing APE_aes datatest `test_standalone_driver_moist_physics` asserts physics leaves `vn`/`w` untouched — TMX with momentum coupling would break it by design, so TMX is opt-in and that test stays muphys-only.

- [ ] **Step 1: Write the failing test**

```python
# appended to the existing standalone_driver config/unit test module
def test_experiment_config_tmx_defaults_to_none(...existing config fixture...):
    assert config.tmx is None
```

Plus a granule-registration test if a unit-level harness for `initialize_granules` exists (it needs a grid + factories, so if only datatest-level coverage is practical, fold registration coverage into Task 10 and keep only the config-default test here).

- [ ] **Step 2: Run to verify it fails** (`AttributeError: tmx`)

- [ ] **Step 3: Implement the config field, registration block, pyproject + tach.toml edits**

After editing `model/standalone_driver/pyproject.toml`, run `uv lock` (workspace resolve) and `uv run tach check` (dependency rules; add the tmx module to `depends_on` of `icon4py.model.standalone_driver` in `tach.toml`).

- [ ] **Step 4: Run the tests + linters**

Run: `uv run pytest model/standalone_driver/tests/standalone_driver/unit_tests -v` (or the module chosen in Step 1)
Expected: PASS.
Run: `uv run tach check`
Expected: no violations.

- [ ] **Step 5: Commit**

```bash
git add model/standalone_driver/src/icon4py/model/standalone_driver/config.py model/standalone_driver/src/icon4py/model/standalone_driver/driver_utils.py model/standalone_driver/pyproject.toml tach.toml uv.lock model/standalone_driver/tests/
git commit -m "feat(driver): register TMX as an opt-in second physics process"
```

---

### Task 10: Driver smoke test — APE_aes with muphys + TMX

**2026-07-09 amendment:** `test_standalone_driver.py` was substantially rewritten by the l2 merge (validation test from #1303, new fixtures incl. geometry config). RE-READ the current file before copying anything; where this task's earlier description of that file conflicts with what you find, the file wins — keep only this task's REQUIREMENTS (new opt-in TMX smoke test; don't touch the existing moist-physics test; finite-field assertions; process-list assertion).

**Files:**
- Test: `model/standalone_driver/tests/standalone_driver/integration_tests/test_standalone_driver.py` (append a new test; do NOT touch `test_standalone_driver_moist_physics`)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the failing test**

Name it `test_standalone_driver_moist_physics_with_tmx`. Copy the structure of `test_standalone_driver_moist_physics` (same `EXCLAIM_APE_AES` parametrization and driver construction), with these differences:
- after loading the experiment config, enable TMX: `config = dataclasses.replace(config, tmx=tmx.TmxConfig.from_fortran_dict(atm_dict))` if the namelist dicts are reachable at that point, else `tmx=tmx.TmxConfig()` (defaults match the aquaplanet namelist for everything the smoke test cares about);
- run the same number of steps as the moist test (1 driver step, 00:00 → 00:05);
- do NOT assert savepoint equality for `vn`/`w` (TMX writes them by design); instead assert:

```python
    assert granules.physics is not None
    assert [p.name for p in granules.physics._processes] == ["muphys", "tmx"]
    for name, field in (
        ("vn", prognostic.vn), ("w", prognostic.w), ("exner", prognostic.exner),
        ("theta_v", prognostic.theta_v), ("rho", prognostic.rho),
        ("qv", tracers.qv), ("qc", tracers.qc), ("qi", tracers.qi),
    ):
        arr = field.asnumpy()
        assert np.isfinite(arr).all(), f"{name} has non-finite entries after muphys+tmx step"
```

(match the local variable names of the copied test for `prognostic`/`tracers`/`granules`; if `_processes` is considered private, add a tiny public accessor or assert via the recycle-cache keys after the run: `set(granules.physics._recycle_cache) == {"muphys", "tmx"}`).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest "model/standalone_driver/tests/standalone_driver/integration_tests/test_standalone_driver.py::test_standalone_driver_moist_physics_with_tmx" -v`
Expected: FAIL before the implementation is wired (e.g. on the process-list assertion) or ERROR if config plumbing is missing; after Task 9 it may already pass — in that case verify it fails when `tmx=None` (sanity that the assertion bites) by temporarily asserting `== ["muphys"]`, watch it fail, revert.

- [ ] **Step 3: Make it pass; then run the neighbouring tests too**

Run: `uv run pytest model/standalone_driver/tests/standalone_driver/integration_tests/test_standalone_driver.py -v`
Expected: ALL PASS — including the untouched `test_standalone_driver_moist_physics` (proves TMX opt-in leaves the existing contract intact).

- [ ] **Step 4: Commit**

```bash
git add model/standalone_driver/tests/standalone_driver/integration_tests/test_standalone_driver.py
git commit -m "test(driver): APE_aes smoke test running muphys + tmx in the PhysicsDriver"
```

---

### Task 11: Full verification sweep and wrap-up

**Files:**
- Modify: `docs/superpowers/specs/2026-07-07-tmx-component-integration-design.md` (mark open items resolved: record the verified `cv_air` formula source, the `ddt_w` boundary finding, and the TMX opt-in decision from Task 9)

- [ ] **Step 1: Run the complete affected test set**

```bash
uv run pytest model/atmosphere/subgrid_scale_physics/tmx -v
uv run pytest model/atmosphere/subgrid_scale_physics/muphys model/atmosphere/subgrid_scale_physics/physics_interface -v
uv run pytest model/standalone_driver/tests -v
```

Expected: ALL PASS (PR-owned tmx tests must be untouched and green — proves we didn't disturb the port).

- [ ] **Step 2: Lint/typing gates used by the repo**

Run: `uv run pre-commit run --files $(git diff --name-only physics_driver_l2...HEAD | tr '\n' ' ')` (or the repo's configured linter — check `.pre-commit-config.yaml`; run at least `ruff` on the new files).
Expected: clean.

- [ ] **Step 3: Update the spec's Open items table and commit**

```bash
git add docs/superpowers/specs/2026-07-07-tmx-component-integration-design.md docs/superpowers/plans/2026-07-07-tmx-component-integration.md
git commit -m "docs: close out phase-1 open items in the TMX integration spec"
```

- [ ] **Step 4: Report**

Summarize: component datatest tolerance achieved, smoke test steps/duration, any deviations from this plan (esp. `wgtfacq1_e` derivation and the `ddt_w` boundary-row finding), and the re-merge status of `pr-1359-tmx`.

---

## Verification checklist (maps to the spec's DoD)

| Spec requirement | Task |
| --- | --- |
| TMX registered as second process after muphys | 9, 10 |
| Zero surface fluxes in driver (phase-2 seam) | 4 |
| Momentum coupling ddt_u/v → vn, ddt_w → w | 5 |
| Component datatest reproduces tmx-exit at rtol 1e-11 | 8 |
| Static states from factories verified vs savepoints | 7 |
| air_mass/cv_air verified vs savepoint mair/cvair | 3 (formula unit tests), 8 Step 4 (datatest vs entry-savepoint mair/cvair) |
| APE_aes driver smoke, finite fields | 10 |
| No PR-owned file edited | all (checked in 11 via `git diff pr-1359-tmx -- model/.../tmx/src/.../tmx/{tmx.py,tmx_states.py,stencils}` = empty) |
