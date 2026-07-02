# icon4py-atmosphere-tmx

Python port of the ICON AES turbulent mixing scheme "tmx" (Smagorinsky-based vertical and
horizontal turbulent diffusion), implemented with GT4Py.

The Fortran reference is `src/atm_phy_aes/tmx/` in the ICON model, entered through
`interface_aes_tmx` (`mo_interface_aes_tmx.f90`).

## Scope

This package ports the atmospheric part of tmx:

- Smagorinsky exchange coefficients (classic Lilly and Louis stability functions),
- implicit/explicit vertical diffusion (tridiagonal solves) of hydrometeors (qv, qc, qi),
  temperature/energy, horizontal wind (on edges) and vertical wind,
- horizontal conservative diffusion of scalars,
- kinetic-energy dissipation heating and vertically integrated energy diagnostics.

The surface (tiles, exchange coefficients, JSBACH land, sea ice) is *not* part of this
package: the scheme takes prescribed grid-mean surface fluxes (sensible heat flux,
evapotranspiration, momentum stress) as inputs. This matches the `isrfc_type == 1`
idealized-surface path of the Fortran reference.

## Installation

Part of the icon4py uv workspace; installed by `uv sync` from the repository root.
