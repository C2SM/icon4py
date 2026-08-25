# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _update_temperature_with_dissipation_heating(
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    new_u: fa.CellKField[wpfloat],
    new_v: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    cv_air: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    tend_temperature: fa.CellKField[wpfloat],
    q_snocpymlt: fa.CellField[wpfloat],
    dissipation_factor: wpfloat,
    dtime: wpfloat,
    nlev: gtx.int32,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """
    Add the kinetic-energy dissipation heating (and the snow-on-canopy melt
    cooling at the lowest level) to the temperature tendency and update the
    temperature.

    Port of 'Update_energy_tendencies' (mo_vdf.f90):

        dissip_ke = 0.5 * mair * dissipation_factor / dtime
                    * (u**2 - new_u**2 + v**2 - new_v**2)
        heating   = dissip_ke - (q_snocpymlt at jk = nlev, else 0)
        tend_ta   = tend_ta + heating / cvair
        new_ta    = ta + tend_ta * dtime

    ``q_snocpymlt`` (the heating used to melt snow on the canopy) is non-zero
    only over land; the Fortran zero fill of ``heating`` outside the computed
    domain ('CALL init(heating)') is not part of this stencil and must be done
    by the caller. ``tend_temperature`` holds the heat-diffusion temperature
    tendency of 'Compute_diffusion_temperature' on entry (read-modify-write,
    'out=(..., tend_temperature)') and the final tmx temperature tendency on
    exit; ``new_u`` / ``new_v`` are the winds updated by the horizontal wind
    diffusion.

    The bottom row is selected with 'dims.KDim < nlev - 1' because
    'concat_where(dims.KDim == nlev - 1, ...)' is currently broken in GT4Py
    1.1.11 (GridTools/gt4py#2205).

    Domains (Fortran): jk = 1..nlev; the tmx ``t_domain`` cell range
    (``grf_bdywidth_c + 1`` to ``min_rlcell_int``), which maps to the
    horizontal domain ``(h_grid.Zone.NUDGING, h_grid.Zone.LOCAL)``.

    Args:
        u: old zonal wind [m/s]
        v: old meridional wind [m/s]
        new_u: zonal wind after the horizontal wind diffusion [m/s]
        new_v: meridional wind after the horizontal wind diffusion [m/s]
        air_mass: air mass per unit area (``mair``) [kg/m^2]
        cv_air: isometric specific heat of moist air (``cvair``) [J/(kg K)]
        temperature: old air temperature [K]
        tend_temperature: temperature tendency of the heat diffusion [K/s]
        q_snocpymlt: heating used to melt snow on the canopy (2D) [W/m^2]
        dissipation_factor: scaling factor of the dissipation heating
        dtime: time step [s]
        nlev: number of full levels

    Returns:
        kinetic energy dissipation per layer [W/m^2], turbulent heating per
        layer [W/m^2], updated temperature [K] and total temperature
        tendency [K/s]
    """
    dissip_ke = (
        wpfloat("0.5")
        * air_mass
        * dissipation_factor
        / dtime
        * (u * u - new_u * new_u + v * v - new_v * new_v)
    )
    heating = concat_where(dims.KDim < nlev - 1, dissip_ke, dissip_ke - q_snocpymlt)
    new_tend = tend_temperature + heating / cv_air
    new_temperature = temperature + new_tend * dtime
    return dissip_ke, heating, new_temperature, new_tend


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_temperature_with_dissipation_heating(
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    new_u: fa.CellKField[wpfloat],
    new_v: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    cv_air: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    tend_temperature: fa.CellKField[wpfloat],
    q_snocpymlt: fa.CellField[wpfloat],
    dissip_ke: fa.CellKField[wpfloat],
    heating: fa.CellKField[wpfloat],
    new_temperature: fa.CellKField[wpfloat],
    dissipation_factor: wpfloat,
    dtime: wpfloat,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_temperature_with_dissipation_heating(
        u=u,
        v=v,
        new_u=new_u,
        new_v=new_v,
        air_mass=air_mass,
        cv_air=cv_air,
        temperature=temperature,
        tend_temperature=tend_temperature,
        q_snocpymlt=q_snocpymlt,
        dissipation_factor=dissipation_factor,
        dtime=dtime,
        nlev=nlev,
        out=(dissip_ke, heating, new_temperature, tend_temperature),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
