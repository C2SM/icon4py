# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import maximum, minimum, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_roughness_ocean(
    wind_rel: fa.CellField[wpfloat],
    km: fa.CellField[wpfloat],
    charnock: wpfloat,
    viscous_coeff: wpfloat,
    kinematic_viscosity: wpfloat,
    z0m_min: wpfloat,
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """
    Compute the ocean surface roughness length (Charnock formula).

    Port of the ocean branch of 'compute_sfc_roughness' (mo_tmx_surface.f90:600-609),
    steady-state form (the ``linit`` first guess ``z0m_oce`` is applied by the
    granule on the first step):

        rough = wind_rel^2 * km * cchar / g + viscous_coeff * min(0.01, nu / (sqrt(km) * wind_rel))
        rough = max(z0m_min, rough)

    ``km`` is the ocean momentum transfer coefficient of the previous step.
    Momentum and heat roughness are equal over ocean.

    Args:
        wind_rel: surface-relative wind speed [m/s]
        km: ocean momentum transfer coefficient of the previous step [-]
        charnock: Charnock constant [-]
        viscous_coeff: viscous roughness coefficient [-]
        kinematic_viscosity: kinematic viscosity of air [m^2/s]
        z0m_min: minimum roughness length [m]

    Returns:
        (rough_m, rough_h): momentum and heat roughness lengths [m]
    """
    rough = wind_rel * wind_rel * km * charnock / PhysicsConstants.grav + viscous_coeff * minimum(
        wpfloat(0.01), kinematic_viscosity / (sqrt(km) * wind_rel)
    )
    rough = maximum(z0m_min, rough)
    return rough, rough


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_roughness_ocean(
    wind_rel: fa.CellField[wpfloat],
    km: fa.CellField[wpfloat],
    rough_m: fa.CellField[wpfloat],
    rough_h: fa.CellField[wpfloat],
    charnock: wpfloat,
    viscous_coeff: wpfloat,
    kinematic_viscosity: wpfloat,
    z0m_min: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_surface_roughness_ocean(
        wind_rel=wind_rel,
        km=km,
        charnock=charnock,
        viscous_coeff=viscous_coeff,
        kinematic_viscosity=kinematic_viscosity,
        z0m_min=z0m_min,
        out=(rough_m, rough_h),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )
