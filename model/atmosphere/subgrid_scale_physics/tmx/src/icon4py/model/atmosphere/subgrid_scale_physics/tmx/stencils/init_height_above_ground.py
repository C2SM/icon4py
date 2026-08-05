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
def _init_height_above_ground(
    z_mc: fa.CellKField[wpfloat],
    z_ifc_sfc: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Compute the geometric height of the full levels above the surface.

    Port of ``compute_geopotential_height_above_ground`` in ICON's
    ``mo_vdf_atmo.f90`` (despite its name it computes the geometric height in
    meters; gravity is only applied later, e.g. in ``compute_static_energy``):

        ghf(k) = zf(k) - zh(nlevp1)

    with zf the height of the full levels (``z_mc``) and zh(nlevp1) the height
    of the bottom half level, i.e. the surface (Fortran 1-based jk = nlevp1 ->
    0-based k = nlev). The surface row is passed as the 2D field ``z_ifc_sfc``
    (slice ``z_ifc[:, nlev]``), since GT4Py offsets are relative and cannot
    address a fixed absolute K row.

    This is a time-independent (init-time) computation. The Fortran subroutine
    loops over the tmx ``t_domain`` cell range (``grf_bdywidth_c + 1`` to
    ``min_rlcell_int``), which maps to the horizontal domain
    ``(h_grid.Zone.NUDGING, h_grid.Zone.LOCAL)``, and over all full levels
    (Fortran jk = 1..nlev -> k = 0..nlev-1).

    Args:
        z_mc: geometric height of the full levels [m]
        z_ifc_sfc: geometric height of the surface (bottom half level) [m]

    Returns:
        height of the full levels above the surface [m]
    """
    return z_mc - z_ifc_sfc


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def init_height_above_ground(
    z_mc: fa.CellKField[wpfloat],
    z_ifc_sfc: fa.CellField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _init_height_above_ground(
        z_mc=z_mc,
        z_ifc_sfc=z_ifc_sfc,
        out=height_above_ground,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
