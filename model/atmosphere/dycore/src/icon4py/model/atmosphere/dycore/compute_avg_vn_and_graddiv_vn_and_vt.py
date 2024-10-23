# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, neighbor_sum

from icon4py.model.atmosphere.dycore.compute_avg_vn import _compute_avg_vn
from icon4py.model.atmosphere.dycore.compute_tangential_wind import _compute_tangential_wind
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2EO, E2C2EODim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_avg_vn_and_graddiv_vn_and_vt(
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2EODim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
) -> tuple[
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
]:
    """Formerly known as _mo_solve_nonhydro_stencil_30."""
    z_vn_avg_wp = _compute_avg_vn(e_flx_avg=e_flx_avg, vn=vn)
    z_graddiv_vn_vp = astype(neighbor_sum(geofac_grdiv * vn(E2C2EO), axis=E2C2EODim), vpfloat)
    vt_vp = _compute_tangential_wind(vn=vn, rbf_vec_coeff_e=rbf_vec_coeff_e)
    return z_vn_avg_wp, z_graddiv_vn_vp, vt_vp


@program(grid_type=GridType.UNSTRUCTURED)
def compute_avg_vn_and_graddiv_vn_and_vt(
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    z_vn_avg: fa.EdgeKField[wpfloat],
    z_graddiv_vn: fa.EdgeKField[vpfloat],
    vt: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_avg_vn_and_graddiv_vn_and_vt(
        e_flx_avg,
        vn,
        geofac_grdiv,
        rbf_vec_coeff_e,
        out=(z_vn_avg, z_graddiv_vn, vt),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
