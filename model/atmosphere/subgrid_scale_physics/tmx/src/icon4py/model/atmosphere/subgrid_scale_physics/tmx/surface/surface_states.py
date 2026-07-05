# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from collections.abc import Callable

    import gt4py.next as gtx
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


"""
State dataclasses of the tmx surface scheme.

Ported from ICON's ``mo_vdf_sfc_memory.f90`` (see ``mo_vdf_sfc.f90`` /
``mo_tmx_surface.f90``). The surface scheme runs before the atmospheric
diffusion and produces a ``TmxSurfaceFluxState``. All fields are 2D cell
fields; tile-resolved quantities are held per tile (ocean/ice/land) where
needed, but only the ocean/ice/land grid-mean fluxes leave the scheme.
"""


@dataclasses.dataclass(frozen=True)
class SurfaceInputState:
    """Prescribed inputs of the surface scheme: lowest-level atmosphere, forcings and tile fractions.

    Atmospheric fields are the lowest full level (``k = nlev - 1``) values.
    """

    ta: fa.CellField[ta.wpfloat]
    """Air temperature at the lowest full level [K]."""
    qa: fa.CellField[ta.wpfloat]
    """Specific humidity at the lowest full level [kg/kg]."""
    ua: fa.CellField[ta.wpfloat]
    """Zonal wind at the lowest full level [m/s]."""
    va: fa.CellField[ta.wpfloat]
    """Meridional wind at the lowest full level [m/s]."""
    pa: fa.CellField[ta.wpfloat]
    """Air pressure at the lowest full level [Pa]."""
    rho_atm: fa.CellField[ta.wpfloat]
    """Air density at the lowest full level [kg/m^3]."""
    psfc: fa.CellField[ta.wpfloat]
    """Surface pressure [Pa]."""
    dz: fa.CellField[ta.wpfloat]
    """Surface-layer reference height (``ddqz_z_half`` at the surface); the solver uses 0.5*dz [m]."""
    # radiation (net at the surface; ice-model forcing)
    lwflx_net: fa.CellField[ta.wpfloat]
    """Net longwave radiation at the surface [W/m^2]."""
    swflx_net: fa.CellField[ta.wpfloat]
    """Net shortwave radiation at the surface [W/m^2]."""
    emissivity: fa.CellField[ta.wpfloat]
    """Surface longwave emissivity [-]."""
    # prescribed ocean
    sst: fa.CellField[ta.wpfloat]
    """Prescribed ocean surface temperature (``ts_tile`` water) [K]."""
    ocean_u: fa.CellField[ta.wpfloat]
    """Zonal ocean surface current [m/s]."""
    ocean_v: fa.CellField[ta.wpfloat]
    """Meridional ocean surface current [m/s]."""
    # prescribed sea ice (thickness treated as quasi-static; skin T is prognostic)
    ice_u: fa.CellField[ta.wpfloat]
    """Zonal sea-ice drift velocity [m/s]."""
    ice_v: fa.CellField[ta.wpfloat]
    """Meridional sea-ice drift velocity [m/s]."""
    ice_thickness: fa.CellField[ta.wpfloat]
    """Sea-ice thickness (``hi``) [m]."""
    snowfall: fa.CellField[ta.wpfloat]
    """Solid precipitation onto the ice (``ssfl``) [kg/(m^2 s)]."""
    # prescribed land fluxes (JSBACH cut line: taken as inputs, not modelled)
    land_evapotrans: fa.CellField[ta.wpfloat]
    """Prescribed land evapotranspiration flux [kg/(m^2 s)]."""
    land_latent_hflx: fa.CellField[ta.wpfloat]
    """Prescribed land latent heat flux [W/m^2]."""
    land_sensible_hflx: fa.CellField[ta.wpfloat]
    """Prescribed land sensible heat flux [W/m^2]."""
    land_tskin: fa.CellField[ta.wpfloat]
    """Prescribed land skin temperature [K]."""
    land_rough_m: fa.CellField[ta.wpfloat]
    """Prescribed land momentum roughness length (``turb_rough_m``) [m]."""
    land_qsat_star: fa.CellField[ta.wpfloat]
    """Prescribed land surface saturation specific humidity (JSBACH ``seb_qsat_star``) [kg/kg]."""
    land_q_snocpymlt: fa.CellField[ta.wpfloat]
    """Prescribed land canopy snow-melt heating [W/m^2]."""
    # tile fractions; grid-mean = sum_tile frac_tile * X_tile
    frac_oce: fa.CellField[ta.wpfloat]
    """Ocean tile area fraction [-]."""
    frac_ice: fa.CellField[ta.wpfloat]
    """Sea-ice tile area fraction [-]."""
    frac_lnd: fa.CellField[ta.wpfloat]
    """Land tile area fraction [-]."""


@dataclasses.dataclass
class SurfaceState:
    """Persistent surface state carried across time steps (``states``/``new_states``/``tendencies``).

    Holds only quantities that must survive between steps; per-tile scratch
    (roughness, exchange coefficients, per-tile fluxes) is granule-local.
    """

    # prognostic sea-ice skin temperature (integrated by ice_fast)
    tsurf_ice_old: fa.CellField[ta.wpfloat]
    """Sea-ice skin temperature at the current step [K]."""
    tsurf_ice_new: fa.CellField[ta.wpfloat]
    """Sea-ice skin temperature after the step [K]; the driver swaps old <- new."""
    tsurf_ice_tend: fa.CellField[ta.wpfloat]
    """Sea-ice skin temperature tendency (new - old)/dt [K/s]."""
    winton_t1: fa.CellField[ta.wpfloat]
    """Upper Winton-layer temperature [degC]; carried for parity (i_ice_therm=2 disabled)."""
    winton_t2: fa.CellField[ta.wpfloat]
    """Lower Winton-layer temperature [degC]; carried for parity (i_ice_therm=2 disabled)."""
    snow_thickness: fa.CellField[ta.wpfloat]
    """Snow thickness on the ice (``hs``) [m]; updated in place across steps."""
    # ocean Charnock roughness uses the momentum coefficient from the previous step
    ocean_km: fa.CellField[ta.wpfloat]
    """Ocean momentum transfer coefficient from the previous step (Charnock lag) [-]."""
    # ice_fast is forced by the ice-tile fluxes of the previous atmospheric step
    lagged_ice_lhflx: fa.CellField[ta.wpfloat]
    """Ice-tile latent heat flux lagged one step (ice_fast forcing) [W/m^2]."""
    lagged_ice_shflx: fa.CellField[ta.wpfloat]
    """Ice-tile sensible heat flux lagged one step (ice_fast forcing) [W/m^2]."""
    lagged_ice_lwflx_net: fa.CellField[ta.wpfloat]
    """Ice-tile net longwave lagged one step [W/m^2]."""
    lagged_ice_swflx_net: fa.CellField[ta.wpfloat]
    """Ice-tile net shortwave lagged one step [W/m^2]."""

    @classmethod
    def allocate(
        cls, grid: base_grid.Grid, allocator: gtx_typing.Allocator | None = None
    ) -> SurfaceState:
        """Allocate a surface state with all fields initialized to zero."""
        surface = _surface_field_allocator(grid, allocator)
        return cls(
            tsurf_ice_old=surface(dims.CellDim),
            tsurf_ice_new=surface(dims.CellDim),
            tsurf_ice_tend=surface(dims.CellDim),
            winton_t1=surface(dims.CellDim),
            winton_t2=surface(dims.CellDim),
            snow_thickness=surface(dims.CellDim),
            ocean_km=surface(dims.CellDim),
            lagged_ice_lhflx=surface(dims.CellDim),
            lagged_ice_shflx=surface(dims.CellDim),
            lagged_ice_lwflx_net=surface(dims.CellDim),
            lagged_ice_swflx_net=surface(dims.CellDim),
        )


def _surface_field_allocator(
    grid: base_grid.Grid, allocator: gtx_typing.Allocator | None
) -> Callable[[gtx.Dimension], gtx.Field]:
    """Return a zero-field factory for 2D surface (cell) fields."""

    def surface(horizontal_dim: gtx.Dimension) -> gtx.Field:
        return data_alloc.zero_field(grid, horizontal_dim, dtype=ta.wpfloat, allocator=allocator)

    return surface
