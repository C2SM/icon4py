# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_inverse_air_mass(
    air_mass: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Compute the inverse air mass per unit area.

    Port of the ``inv_mair(jc,jk,jb) = 1._wp / mair(jc,jk,jb)`` loops at the
    top of 'Compute_diffusion_hydrometeors' and 'Compute_diffusion_temperature'
    (mo_vdf.f90). ``inv_mair`` scales the rows of the vertical diffusion matrix
    ('prepare_diffusion_matrix') and the surface flux right-hand side.

    The Fortran loops run over the tmx ``t_domain`` cell range
    (``grf_bdywidth_c + 1`` to ``min_rlcell_int``), which maps to the horizontal
    domain ``(h_grid.Zone.NUDGING, h_grid.Zone.LOCAL)``, and over all full
    levels.

    Args:
        air_mass: air mass per unit area (``mair``) at full levels [kg/m^2]

    Returns:
        inverse air mass per unit area at full levels [m^2/kg]
    """
    return wpfloat("1.0") / air_mass


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_inverse_air_mass(
    air_mass: fa.CellKField[wpfloat],
    inv_air_mass: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_inverse_air_mass(
        air_mass=air_mass,
        out=inv_air_mass,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
