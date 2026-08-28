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
