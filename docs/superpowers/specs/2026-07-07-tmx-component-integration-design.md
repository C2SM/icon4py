# TMX Turbulent Mixing as a PhysicsDriver Component — Design

**Date:** 2026-07-07
**Branch:** `physics_driver_tmx` (= `physics_driver_l2` + merge of C2SM/icon4py PR #1359 `port_turbulence`; merge commit `f35794229`)
**Status:** Approved design, pre-implementation

## Context

The `physics_driver_l2` branch provides an L2 physics orchestrator (`PhysicsDriver`,
`model/atmosphere/subgrid_scale_physics/physics_interface/`) that runs physics processes as
`Component`s with per-process `PhysicsState` adapters and `ProcessTimeControl` gating. Muphys
(microphysics) is the first integrated process and the structural template.

C2SM PR #1359 ports TMX turbulent mixing (Smagorinsky closure, implicit vertical diffusion of
hydrometeors/temperature/wind, horizontal diffusion, dissipative heating) natively to GT4Py as a
`Tmx` granule (`model/atmosphere/subgrid_scale_physics/tmx/`). This supersedes the earlier
pybind11 wrap plan in `docs/physics_driver_design.md`: no grid-layout translation, patch-array
derivation, or Kokkos coexistence is needed anymore.

In Fortran ICON (`mo_aes_phy_main.f90`), vdf/TMX runs after microphysics and radiation, before
optional chemistry, with sequential in-place state updates. The aqua-planet suite
(`run/exp.aes_aquaplanet_r02b04`) is exactly: graupel microphysics + radiation + vdf, sea surface
only, prescribed SST.

## Goals (phase 1, this cycle)

1. `TmxComponent` registered in `PhysicsDriver` as a second process, after muphys.
2. Surface fluxes prescribed as zero (the regime the PR's savepoint tests cover).
3. Momentum coupling included: `ddt_u/ddt_v` projected to `vn`, `ddt_w` applied to `w`.
4. Definition of done:
   - Component-level datatest: `TmxComponent` driven through the Component/PhysicsState wrappers
     reproduces the PR's `tmx-exit` savepoint tendencies at the PR's own tolerance (rtol 1e-11).
   - Driver smoke: APE_aes standalone run with muphys + TMX completes with finite fields.

## Non-goals (phase 1)

- Ocean bulk-flux scheme from prescribed SST (phase 2; the `TmxSurfaceFluxState` fill in
  `TmxState.gather` is the designed seam for it).
- Radiation (next component after TMX; without it the APE run is a numerical smoke test, not
  climate).
- Distributed halo exchange (`exchange=single_node_exchange` in phase 1; the `Tmx` granule already
  takes an exchange runtime, so this is a constructor argument later, not a redesign).
- CO2 tracer diffusion, surface tiles/JSBACH/sea-ice, 2m/10m diagnostics (excluded from the GT4Py
  port itself).
- Any edit to PR-owned files (keeps re-merges of the moving draft clean).

## Decisions already taken

| Decision             | Choice                                                                                                     |
| -------------------- | ---------------------------------------------------------------------------------------------------------- |
| Code baseline        | New branch `physics_driver_tmx`; source branches untouched; re-merge `pr-1359-tmx` as the draft PR evolves |
| Surface fluxes       | Staged: zero in phase 1, bulk-flux provider in phase 2                                                     |
| Momentum             | Applied in phase 1 (cells→edges projection, mirroring `mo_interface_iconam_aes`)                           |
| Wrapper architecture | Muphys pattern: new files inside the tmx package; `PhysicsDriver` unchanged                                |
| Process order        | muphys → tmx (Fortran: mig → rad → vdf; rad not yet ported)                                                |

## Architecture

New files, all in `model/atmosphere/subgrid_scale_physics/tmx/src/.../tmx/`:

```
component.py      TmxComponent  — Component protocol (L4 adapter)
state.py          TmxState      — PhysicsState protocol (L1 dycore↔physics adapter)
data.py           inputs_properties / outputs_properties metadata
static_fields.py  build TmxMetricState + TmxInterpolationState from field factories
```

Registration (in `driver_utils.py`) mirrors muphys:

```python
tmx_process = physics_driver.PhysicsProcess(
    name="tmx",
    component=TmxComponent(...),   # owns Tmx granule + persistent output buffers
    state=TmxState(...),           # gather/scatter vs PrognosticState
    time_control=ProcessTimeControl(interval=dt_vdf, ...),
)
physics_granule = physics_driver.PhysicsDriver([muphys_process, tmx_process])
```

### TmxComponent (component.py)

- Constructor: grid, `TmxConfig`, `TmxParams`, static states (from `static_fields.py`),
  `EdgeParams`/`CellParams`, backend. Instantiates the `Tmx` granule and allocates
  `TmxDiagnosticState`, `TmxTendencyState`, `TmxNewState` once; buffers are reused every step.
- `__call__(state: dict, time_step)`: packs the dict into `TmxInputState` +
  `TmxSurfaceFluxState` views (no copies where aliasing is safe), calls `Tmx.run(..., dtime)`,
  returns:
  - tendencies (`kind: "tendency"`): `ddt_temperature`, `ddt_qv`, `ddt_qc`, `ddt_qi`, `ddt_u`,
    `ddt_v`, `ddt_w`
  - diagnostics (`kind: "diagnostic"`): `km`, `kh`, `heating`, `dissip_ke`, `cptgz_vi`,
    `dissip_ke_vi`, `int_energy_vi`, `int_energy_vi_tend`
- Unlike muphys, no `(new − old)/dt` conversion: the granule returns tendencies directly.

### TmxState (state.py)

**gather_from_prognostic** (prognostic → `TmxInputState` fields):

| Field                                                            | Source                                                                                        |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `rho`, `w`, `qv…qg`                                              | direct references (levels already match; `w` is KDim+1)                                       |
| `temperature`, `virtual_temperature`, `pressure`, `pressure_ifc` | diagnosed from `theta_v`/`exner` with the same stencils muphys' `State` uses                  |
| `u`, `v`                                                         | RBF edge→cell from `vn` (existing common stencil), inside gather so TMX sees post-muphys wind |
| `air_mass`                                                       | new small stencil, ρ·Δz (`ddqz_z_full`), formula per `mo_interface_aes_tmx`                   |
| `cv_air`                                                         | new small stencil, moist heat capacity per `mo_interface_aes_tmx`                             |
| surface fluxes                                                   | zero-filled `TmxSurfaceFluxState` buffers (phase-2 seam)                                      |

`air_mass` and `cv_air` appear in the `tmx-entry` savepoint (`mair`, `cvair`), so both stencils
are datatest-verifiable.

**scatter_to_prognostic** (component outputs → prognostic), same order as muphys (moisture before
temperature so the Tv/exner conversion is consistent):

1. `qv/qc/qi += ddt_q* · dt` (qr/qs/qg are not diffused — untouched).
2. `ddt_temperature` → ΔTv → Δexner/θv via the existing muphys tend_T→exner path.
3. `(ddt_u, ddt_v)` → `ddt_vn` using the PR's own `compute_vn_from_uv` stencil applied to the
   tendency pair; `vn += ddt_vn · dt`.
4. `w += ddt_w · dt` (verify boundary rows carry zero tendency; the implicit w-solve starts at
   level 2).
5. Diagnostics stored as references on the adapter (like muphys precip fluxes), never applied.

### Static fields (static_fields.py)

One function building `TmxMetricState` + `TmxInterpolationState` from the driver's existing
geometry/interpolation/metrics factories. Already available (some under long names):
`rbf_vec_coeff_{c1,c2,e,v1,v2}`, `cells_aw_verts`, `c_lin_e`, `e_bln_c_s`, `geofac_div`,
`edge_cell_length`, `wgtfac_c/e`, `wgtfacq_c/e`
(`weighting_factor_for_quadratic_interpolation_to_*`), `ddqz_z_full`, `inv_ddqz_z_full`,
`ddqz_z_half`, `ddqz_z_full_e`, `z_mc`, `z_ifc`.

To derive with small GT4Py programs:

- `inv_ddqz_z_half`, `inv_ddqz_z_full_e`, `inv_ddqz_z_half_e`, `inv_ddqz_z_half_v` (reciprocals /
  interpolations to edges and vertices),
- `wgtfacq1_c`, `wgtfacq1_e` (top-boundary quadratic extrapolation coefficients, formulas ported
  from `mo_vertical_grid`),
- `geopot_agl_ifc = g · (z_ifc − z_sfc)`.

All derivations are verified against savepoint-built states (see Testing).

## Driver config

- New `tmx` section in the standalone-driver config: enable flag + `TmxConfig` overrides
  (defaults from the PR's dataclass). Gated on the APE_aes experiment like muphys
  (`config.tmx is not None`).
- `dt_vdf` defaults to the driver time step (matches `exp.aes_aquaplanet_r02b04`, where
  `dt_vdf` equals the dynamics step); `ProcessTimeControl` handles any longer interval.
- Backend shared with the driver.

## Testing

Cheapest first; datatests use experiment `exclaim_ape_aesPhys` **v06** (the PR bumped v05 → v06 to
add `tmx-entry`/`tmx-surface-fluxes`/`tmx-exit` savepoints; same experiment as the muphys tests,
auto-downloaded by the datatest infra).

| Layer       | Test                                                                                                                                                        | Proves                                                    |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Unit        | adapter round-trip on synthetic fields; momentum projection sanity (uniform u,v → expected vn)                                                              | gather/scatter bookkeeping                                |
| Datatest    | factory-built static states vs savepoint-built                                                                                                              | derived metric/interpolation fields correct               |
| Datatest    | gather-computed `air_mass`/`cv_air` vs savepoint `mair`/`cvair`                                                                                             | thermodynamic formulas match Fortran                      |
| Datatest    | `TmxComponent.__call__` fed from `tmx-entry` vs `tmx-exit`, rtol 1e-11                                                                                      | wrapper adds no error on top of the PR's own verification |
| Integration | APE_aes standalone run, muphys + TMX, same step count as the existing muphys smoke test, finite-field checks (extend `test_standalone_driver_runs_ape_aes`) | end-to-end wiring                                         |

## Open items

| Item                   | Resolution path                                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cv_air` exact formula | pinned during implementation from `mo_interface_aes_tmx`; gather datatest enforces it                                                              |
| `ddt_w` boundary rows  | one-line check during scatter implementation                                                                                                       |
| PR #1359 churn         | re-merge `pr-1359-tmx` periodically; only workspace files (`pyproject.toml`, `tach.toml`, `uv.lock`, CI) can conflict since our files are new-only |
| Testdata v06           | closed as design risk; download on first datatest run                                                                                              |

## Phase 2 sketch (not in this cycle)

Ocean bulk fluxes from prescribed SST: a flux provider fills `TmxSurfaceFluxState` in
`TmxState.gather` from SST + lowest-level state (drag-coefficient/Louis formulas, sea tile only,
`z0m_oce` roughness). Candidate shapes: a standalone `Component` ordered before TMX, or a
callable injected into `TmxState` — decided in its own design. After that, radiation is the
remaining aqua-planet component.

## References

- Driver/protocols: `physics_interface/.../physics_driver.py`,
  `common/.../components/components.py`, `common/.../components/physics_state.py`
- Muphys template: `muphys/.../component.py`, `muphys/.../state.py`, `muphys/.../data.py`
- TMX granule: `tmx/.../tmx.py` (constructor ~L483, `run` ~L2462), `tmx/.../tmx_states.py`,
  `tmx/README.md`, `tmx/docs/gt4py_patterns.md`
- Test template: `tmx/tests/tmx/integration_tests/test_tmx_run.py`, `.../utils.py`
- Fortran reference: `icon-mpim/src/atm_phy_aes/mo_aes_phy_main.f90` (sequence),
  `mo_interface_aes_tmx.f90` (contract), `run/exp.aes_aquaplanet_r02b04` (APE suite)
- Prior docs: `docs/physics_driver_design.md` (pybind11 wrap plan — superseded for TMX),
  `docs/2026-05-20-physics-driver-shapeup.md` (muphys cycle)
