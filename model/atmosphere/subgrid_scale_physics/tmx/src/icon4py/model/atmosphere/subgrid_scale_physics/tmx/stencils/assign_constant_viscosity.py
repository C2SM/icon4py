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
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _assign_constant_viscosity(
    rho_ic: fa.CellKField[wpfloat],
    km_const: wpfloat,
    rturb_prandtl: wpfloat,
    nlev: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Assign a constant eddy viscosity and diffusivity (for turbulence model validation).

    Port of ``Assign_constant_eddy_viscosity`` in ICON's ``mo_vdf_atmo.f90``:
    - interior half levels (0 < k < nlev):
        km_ic = rho_ic * km_const
        kh_ic = km_ic * rturb_prandtl
    - boundary half levels are copies of the adjacent interior rows:
        k = 0 copies k = 1, k = nlev copies k = nlev - 1
      (Fortran 1-based: k = 1 <- k = 2, k = nlevp1 <- k = nlev).

    Args:
        rho_ic: air density at half-level cell centers (nlev + 1 levels)
        km_const: constant kinematic eddy viscosity
        rturb_prandtl: reciprocal turbulent Prandtl number
        nlev: number of full levels

    Returns:
        eddy viscosity km_ic and eddy diffusivity kh_ic at half levels
    """
    km = rho_ic * km_const
    km_ic = concat_where(dims.KDim == 0, km(KDim + 1), km)
    km_ic = concat_where(dims.KDim == nlev, km(KDim - 1), km_ic)
    kh_ic = km_ic * rturb_prandtl
    return km_ic, kh_ic


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def assign_constant_viscosity(
    rho_ic: fa.CellKField[wpfloat],
    km_ic: fa.CellKField[wpfloat],
    kh_ic: fa.CellKField[wpfloat],
    km_const: wpfloat,
    rturb_prandtl: wpfloat,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _assign_constant_viscosity(
        rho_ic=rho_ic,
        km_const=km_const,
        rturb_prandtl=rturb_prandtl,
        nlev=nlev,
        out=(km_ic, kh_ic),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
