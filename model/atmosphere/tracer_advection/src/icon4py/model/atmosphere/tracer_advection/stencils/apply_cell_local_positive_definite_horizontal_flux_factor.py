# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Cell-local positive-definite limiter of Jocksch's FFSL-WENO schemes, edge part.

The clamp and the scaling ``p_out_e = r_m * z_b`` of mo_advection_hflux.f90 3021 and
3028-3032, written on the edges: the upwind cell is E2C[0] for vn >= 0 and E2C[1]
otherwise (the backtrajectory rule), and the normal points out of E2C[0] (ICON's
convention, which the upwind rule itself relies on), so the upwind cell's outflow is
``+flux`` for vn >= 0 and ``-flux`` for vn < 0. See the cell part for the orientation
discussion.
"""

import gt4py.next as gtx
from gt4py.next import maximum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C


@gtx.field_operator
def _apply_cell_local_positive_definite_horizontal_flux_factor(
    r_m: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    lvn_pos = p_vn >= 0.0
    # orientation of the normal relative to the upwind cell
    orientation = where(lvn_pos, 1.0, -1.0)
    # f90 3021: z_b = MAX(z_b, 0) on the upwind cell's outflow
    z_b = orientation * maximum(orientation * p_mflx_tracer_h, 0.0)
    # f90 3030: p_out_e = r_m * z_b with the upwind cell's r_m
    return where(lvn_pos, r_m(E2C[0]), r_m(E2C[1])) * z_b


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_cell_local_positive_definite_horizontal_flux_factor(
    r_m: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_cell_local_positive_definite_horizontal_flux_factor(
        r_m=r_m,
        p_mflx_tracer_h=p_mflx_tracer_h,
        p_vn=p_vn,
        out=p_mflx_tracer_h,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
