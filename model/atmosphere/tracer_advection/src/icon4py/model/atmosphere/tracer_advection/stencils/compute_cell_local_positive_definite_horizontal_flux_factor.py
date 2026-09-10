# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Cell-local positive-definite limiter of Jocksch's FFSL-WENO schemes, cell part.

Port of the ``p_itype_hlimit == ifluxl_sm`` branch inside his reconstruction-and-flux
kernels (mo_advection_hflux.f90 3013-3040 in upwind_hflux_miura3_weno; the same lines
in 102/132/202/203), paper Algorithm 1: per cell, the reconstructed fluxes of the edges
whose upwind cell it is are clamped to non-negative outflow (the reconstruction limiter)
and summed into ``flux_out``, and the outflow scaling ``r_m = min(1, q rho / (flux_out
+ eps))`` follows. The edge part, 'apply_cell_local_positive_definite_horizontal_flux_
factor', redoes the clamp and applies ``r_m``. Nothing crosses a cell boundary except
the edge fluxes themselves, so no halo exchange is needed.

His clamp is ``z_b = MAX(z_b, 0)``, which takes the edge normal to point out of the
upwind cell; on grids where the normal of an edge points into its upwind cell (every
E2C[1] cell, i.e. every vn < 0 edge under ICON's cell-1-to-cell-2 normal convention) that
zeroes the flux. Here the outflow is ``flux * orientation`` with the orientation of the
normal relative to the cell, ``sign(geofac_div)``, so the port coincides with his code
where the normals point his way and limits correctly elsewhere.
"""

import gt4py.next as gtx
from gt4py.next import maximum, minimum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E


@gtx.field_operator
def _clamped_outflow_contribution(
    p_mflx_tracer_h: fa.CellKField[ta.wpfloat],
    p_vn: fa.CellKField[ta.wpfloat],
    geofac_div: fa.CellField[ta.wpfloat],
    p_dtime: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    # the normal of this edge relative to the cell: +1 outward, -1 inward
    orientation = where(geofac_div > 0.0, 1.0, -1.0)
    # the cell is the edge's upwind cell iff the wind leaves it through the edge, with the
    # backtrajectory's tie rule vn >= 0 -> E2C[0] (the outward-normal cell)
    is_upwind = where(geofac_div > 0.0, p_vn >= 0.0, p_vn < 0.0)
    # f90 3021: z_b = MAX(z_b, 0), applied to the outflow
    z_b = orientation * maximum(orientation * p_mflx_tracer_h, 0.0)
    # f90 3022-3023: flux_out = flux_out + geofac_div * p_dtime * z_b
    return where(is_upwind, geofac_div * p_dtime * z_b, 0.0)


@gtx.field_operator
def _compute_cell_local_positive_definite_horizontal_flux_factor(
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    p_rhodz_now: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    p_dtime: ta.wpfloat,
    dbl_eps: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    # f90 3016-3025: the cell's three edges in C2E order, only the upwind ones contribute
    flux_out = (
        _clamped_outflow_contribution(
            p_mflx_tracer_h(C2E[0]), p_vn(C2E[0]), geofac_div[dims.C2EDim(0)], p_dtime
        )
        + _clamped_outflow_contribution(
            p_mflx_tracer_h(C2E[1]), p_vn(C2E[1]), geofac_div[dims.C2EDim(1)], p_dtime
        )
        + _clamped_outflow_contribution(
            p_mflx_tracer_h(C2E[2]), p_vn(C2E[2]), geofac_div[dims.C2EDim(2)], p_dtime
        )
    )
    # f90 3026-3027: r_m = MIN(1, (p_cc * rhodz_now) / (flux_out + dbl_eps))
    r_m = minimum(1.0, (p_cc * p_rhodz_now) / (flux_out + dbl_eps))
    return r_m


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cell_local_positive_definite_horizontal_flux_factor(
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    p_rhodz_now: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    p_dtime: ta.wpfloat,
    dbl_eps: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_cell_local_positive_definite_horizontal_flux_factor(
        geofac_div=geofac_div,
        p_cc=p_cc,
        p_rhodz_now=p_rhodz_now,
        p_mflx_tracer_h=p_mflx_tracer_h,
        p_vn=p_vn,
        p_dtime=p_dtime,
        dbl_eps=dbl_eps,
        out=r_m,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
