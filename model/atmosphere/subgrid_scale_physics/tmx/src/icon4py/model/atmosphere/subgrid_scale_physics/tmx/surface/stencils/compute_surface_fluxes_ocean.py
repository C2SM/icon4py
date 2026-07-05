# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.physics.thermodynamics import ThermodynamicConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_fluxes_ocean(
    rho_sfc: fa.CellField[wpfloat],
    kh: fa.CellField[wpfloat],
    km: fa.CellField[wpfloat],
    wind_rel: fa.CellField[wpfloat],
    qa: fa.CellField[wpfloat],
    qsat_sfc: fa.CellField[wpfloat],
    ta: fa.CellField[wpfloat],
    temperature_sfc: fa.CellField[wpfloat],
    ua: fa.CellField[wpfloat],
    va: fa.CellField[wpfloat],
    ocean_u: fa.CellField[wpfloat],
    ocean_v: fa.CellField[wpfloat],
    wind_gustiness: wpfloat,
) -> tuple[
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
]:
    """
    Compute the ocean surface bulk fluxes.

    Port of the ocean branch of 'compute_sfc_fluxes' (mo_tmx_surface.f90:837-845).
    The scalar (heat/moisture) fluxes use a gustiness-enhanced effective wind
    ``sqrt(wind_g^2 + wind_rel^2)``; the momentum stress uses ``wind_rel`` and
    the ocean-current-relative wind:

        evap     =  rho * kh * gust * (qa - qsat_sfc)
        latent   =  evap * (lvc + (cvv - clw) * T_sfc)
        sensible =  cvd * rho * kh * gust * (ta - T_sfc)
        ustress  =  rho * km * wind_rel * (ua - u_oce)
        vstress  =  rho * km * wind_rel * (va - v_oce)

    Args:
        rho_sfc: surface air density [kg/m^3]
        kh: heat transfer coefficient [-]
        km: momentum transfer coefficient [-]
        wind_rel: surface-relative wind speed [m/s]
        qa: specific humidity at the lowest full level [kg/kg]
        qsat_sfc: surface saturation specific humidity [kg/kg]
        ta: air temperature at the lowest full level [K]
        temperature_sfc: surface temperature [K]
        ua: zonal wind at the lowest full level [m/s]
        va: meridional wind at the lowest full level [m/s]
        ocean_u: zonal ocean surface current [m/s]
        ocean_v: meridional ocean surface current [m/s]
        wind_gustiness: gustiness parameter (wind_g) [m/s]

    Returns:
        (evapotranspiration, latent_hflx, sensible_hflx, u_stress, v_stress)
    """
    gust = sqrt(wind_gustiness * wind_gustiness + wind_rel * wind_rel)
    evapotranspiration = rho_sfc * kh * gust * (qa - qsat_sfc)
    latent_hflx = evapotranspiration * (
        ThermodynamicConstants.lvc
        + (ThermodynamicConstants.cvv - ThermodynamicConstants.clw) * temperature_sfc
    )
    sensible_hflx = ThermodynamicConstants.cvd * rho_sfc * kh * gust * (ta - temperature_sfc)
    u_stress = rho_sfc * km * wind_rel * (ua - ocean_u)
    v_stress = rho_sfc * km * wind_rel * (va - ocean_v)
    return evapotranspiration, latent_hflx, sensible_hflx, u_stress, v_stress


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_fluxes_ocean(
    rho_sfc: fa.CellField[wpfloat],
    kh: fa.CellField[wpfloat],
    km: fa.CellField[wpfloat],
    wind_rel: fa.CellField[wpfloat],
    qa: fa.CellField[wpfloat],
    qsat_sfc: fa.CellField[wpfloat],
    ta: fa.CellField[wpfloat],
    temperature_sfc: fa.CellField[wpfloat],
    ua: fa.CellField[wpfloat],
    va: fa.CellField[wpfloat],
    ocean_u: fa.CellField[wpfloat],
    ocean_v: fa.CellField[wpfloat],
    evapotranspiration: fa.CellField[wpfloat],
    latent_hflx: fa.CellField[wpfloat],
    sensible_hflx: fa.CellField[wpfloat],
    u_stress: fa.CellField[wpfloat],
    v_stress: fa.CellField[wpfloat],
    wind_gustiness: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_surface_fluxes_ocean(
        rho_sfc=rho_sfc,
        kh=kh,
        km=km,
        wind_rel=wind_rel,
        qa=qa,
        qsat_sfc=qsat_sfc,
        ta=ta,
        temperature_sfc=temperature_sfc,
        ua=ua,
        va=va,
        ocean_u=ocean_u,
        ocean_v=ocean_v,
        wind_gustiness=wind_gustiness,
        out=(evapotranspiration, latent_hflx, sensible_hflx, u_stress, v_stress),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )
