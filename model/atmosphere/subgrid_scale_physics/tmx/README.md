# icon4py-atmosphere-tmx

Python port of the ICON AES turbulent mixing scheme "tmx" (Smagorinsky-based vertical and
horizontal turbulent diffusion), implemented with GT4Py.

The Fortran reference is `src/atm_phy_aes/tmx/` in the ICON model, entered through
`interface_aes_tmx` (`mo_interface_aes_tmx.f90`). The granule boundary is `vdf%Compute`
(`mo_vdf.f90`): tendencies and diagnostics out; the application of the tendencies to the
model state and the implicit longwave correction stay with the caller.

## Scope

This package ports the atmospheric part of tmx:

- Smagorinsky exchange coefficients (classic Lilly and Louis stability functions),
- implicit/explicit vertical diffusion (tridiagonal solves) of hydrometeors (qv, qc, qi),
  temperature/energy, horizontal wind (on edges) and vertical wind,
- horizontal conservative diffusion of scalars,
- kinetic-energy dissipation heating and vertically integrated energy diagnostics.

The surface (tiles, exchange coefficients, JSBACH land, sea ice, 2m/10m diagnostics) is
*not* part of this package: the scheme takes prescribed grid-mean surface fluxes
(sensible heat flux, evapotranspiration, momentum stress, snow-on-canopy melt heating)
as inputs. This matches the `isrfc_type == 1` idealized-surface path of the Fortran
reference. The CO2 tracer diffusion (`l_co2`) is also out of scope.

## Structure

- `tmx.py`: configuration (`TmxConfig`, namelist `aes_vdf_nml`), derived parameters
  (`TmxParams`) and the `Tmx` granule class. `Tmx.run` executes one time step in the
  Fortran stage order of `Compute` (`mo_vdf.f90`), with halo exchanges at the Fortran
  sync points; each stage is also callable on its own:

  | Stage | Granule method                  | Fortran                                             |
  | ----- | ------------------------------- | --------------------------------------------------- |
  | A     | `run_diagnostics`               | `Compute_diagnostics` (mo_vdf_atmo.f90)             |
  | B     | `run_hydrometeor_diffusion`     | `Compute_diffusion_hydrometeors` (mo_vdf.f90)       |
  | C     | `run_temperature_diffusion`     | `Compute_diffusion_temperature` (mo_vdf.f90)        |
  | D     | `run_horizontal_wind_diffusion` | `Compute_diffusion_hor_wind` (mo_vdf.f90)           |
  | E     | `run_vertical_wind_diffusion`   | `Compute_diffusion_vert_wind` (mo_vdf.f90)          |
  | F     | `run_energy_update`             | `Update_energy_tendencies` (mo_vdf.f90)             |
  | G     | `run_update_diagnostics`        | `Update_diagnostics` (mo_vdf_atmo.f90 / mo_vdf.f90) |

- `tmx_states.py`: frozen state dataclasses (metric, interpolation, input, surface flux,
  diagnostic, tendency and new-state containers). The Fortran (state, new_state,
  tendency) triples map to `TmxInputState` / `TmxNewState` / `TmxTendencyState`.

- `stencils/`: one GT4Py program per file, each documenting its Fortran provenance
  (module, subroutine, loop bounds) and the horizontal/vertical domains.

- `docs/gt4py_patterns.md`: GT4Py findings and workarounds collected during the port.

## Tests

```bash
# stencil and unit tests (no data required):
uv run --group test --frozen pytest --datatest-skip model/atmosphere/subgrid_scale_physics/tmx/

# on a compiled backend:
uv run --group test --frozen pytest --datatest-skip --backend gtfn_cpu model/atmosphere/subgrid_scale_physics/tmx/

# integration datatests against serialized ICON reference data
# (exp.exclaim_ape_aesPhys archive, auto-downloaded):
uv run --group test --frozen pytest --datatest-only model/atmosphere/subgrid_scale_physics/tmx/
```

The integration tests verify the granule per stage against the tmx savepoints
(`tmx-entry`, `tmx-diagnostics-exit`, `tmx-hydro-exit`, `tmx-temperature-exit`,
`tmx-hor-wind-exit`, `tmx-vert-wind-exit`) and end-to-end (`test_tmx_run.py`,
`tmx-entry` -> `tmx-exit`).

## Installation

Part of the icon4py uv workspace; installed by `uv sync` from the repository root.
