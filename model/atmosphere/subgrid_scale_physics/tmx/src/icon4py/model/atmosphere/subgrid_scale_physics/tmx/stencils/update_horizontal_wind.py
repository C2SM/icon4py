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
def _update_horizontal_wind(
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    tend_u: fa.CellKField[wpfloat],
    tend_v: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Apply the momentum-diffusion tendencies to the horizontal wind.

    Port of the final update loop of 'Compute_diffusion_hor_wind' (mo_vdf.f90):

        new_u = u + tend_u * dtime
        new_v = v + tend_v * dtime

    tend_u/tend_v are the RBF cell interpolations of the total (horizontal +
    vertical) vn diffusion tendency ('rbf_vec_interpol_cell' of tot_tend,
    ported separately via the common edge-to-cell RBF interpolation).

    Domains (Fortran caller, the tmx 'domain' cell loop bounds set in
    mo_tmx_field_class.f90): jk = 1..nlev; cells from
    rl_start = grf_bdywidth_c + 1 -> 'h_grid.Zone.NUDGING' to
    rl_end = min_rlcell_int -> 'h_grid.Zone.LOCAL'.
    """
    new_u = u + tend_u * dtime
    new_v = v + tend_v * dtime
    return new_u, new_v


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_horizontal_wind(
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    tend_u: fa.CellKField[wpfloat],
    tend_v: fa.CellKField[wpfloat],
    new_u: fa.CellKField[wpfloat],
    new_v: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_horizontal_wind(
        u=u,
        v=v,
        tend_u=tend_u,
        tend_v=tend_v,
        dtime=dtime,
        out=(new_u, new_v),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
