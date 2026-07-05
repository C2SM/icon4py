# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.physics.thermodynamics import ThermodynamicConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_fluxes_ice(
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
    ice_u: fa.CellField[wpfloat],
    ice_v: fa.CellField[wpfloat],
) -> tuple[
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
]:
    """
    Compute the sea-ice surface bulk fluxes.

    Port of the ice branch of 'compute_sfc_fluxes' (mo_tmx_surface.f90:846-851).
    Unlike the ocean branch there is no gustiness enhancement; the latent heat
    uses the sublimation coefficients (``lsc``, ``ci``):

        evap     =  rho * kh * wind_rel * (qa - qsat_sfc)
        latent   =  evap * (lsc + (cvv - ci) * T_sfc)
        sensible =  cvd * rho * kh * wind_rel * (ta - T_sfc)
        ustress  =  rho * km * wind_rel * (ua - u_ice)
        vstress  =  rho * km * wind_rel * (va - v_ice)

    Args:
        rho_sfc: surface air density [kg/m^3]
        kh: heat transfer coefficient [-]
        km: momentum transfer coefficient [-]
        wind_rel: surface-relative wind speed [m/s]
        qa: specific humidity at the lowest full level [kg/kg]
        qsat_sfc: surface saturation specific humidity [kg/kg]
        ta: air temperature at the lowest full level [K]
        temperature_sfc: ice surface temperature [K]
        ua: zonal wind at the lowest full level [m/s]
        va: meridional wind at the lowest full level [m/s]
        ice_u: zonal sea-ice drift velocity [m/s]
        ice_v: meridional sea-ice drift velocity [m/s]

    Returns:
        (evapotranspiration, latent_hflx, sensible_hflx, u_stress, v_stress)
    """
    evapotranspiration = rho_sfc * kh * wind_rel * (qa - qsat_sfc)
    latent_hflx = evapotranspiration * (
        ThermodynamicConstants.lsc
        + (ThermodynamicConstants.cvv - ThermodynamicConstants.ci) * temperature_sfc
    )
    sensible_hflx = ThermodynamicConstants.cvd * rho_sfc * kh * wind_rel * (ta - temperature_sfc)
    u_stress = rho_sfc * km * wind_rel * (ua - ice_u)
    v_stress = rho_sfc * km * wind_rel * (va - ice_v)
    return evapotranspiration, latent_hflx, sensible_hflx, u_stress, v_stress


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_fluxes_ice(
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
    ice_u: fa.CellField[wpfloat],
    ice_v: fa.CellField[wpfloat],
    evapotranspiration: fa.CellField[wpfloat],
    latent_hflx: fa.CellField[wpfloat],
    sensible_hflx: fa.CellField[wpfloat],
    u_stress: fa.CellField[wpfloat],
    v_stress: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_surface_fluxes_ice(
        rho_sfc,
        kh,
        km,
        wind_rel,
        qa,
        qsat_sfc,
        ta,
        temperature_sfc,
        ua,
        va,
        ice_u,
        ice_v,
        out=(evapotranspiration, latent_hflx, sensible_hflx, u_stress, v_stress),
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )
