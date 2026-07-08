# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2E, E2C2EDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_tangential_wind_wp(
    vn: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """
    Reconstruct the tangential velocity component at edge midpoints from the
    normal velocity components of the four surrounding edges (RBF vector
    interpolation).

    Working-precision port of ``rbf_vec_interpol_edge`` in ICON's
    ``mo_intp_rbf.f90`` (``rbf_vec_interpol_edge_lib`` in iconmath's
    ``mo_lib_intp_rbf.F90``):

        vt(e, k) = sum over the four E2C2E neighbor edges e' of
                   rbf_vec_coeff_e(e, e') * vn(e', k)

    The stencil is agnostic of the vertical staggering: it can be applied to
    half-level input (e.g. vn_ie -> vt_ie) as well as full-level input
    (vn -> vt); the K extent is just a domain argument.
    """
    return neighbor_sum(rbf_vec_coeff_e * vn(E2C2E), axis=E2C2EDim)


@gtx.field_operator
def _compute_tangential_wind(
    vn: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
) -> fa.EdgeKField[vpfloat]:
    """
    Variable-precision variant of ``_compute_tangential_wind_wp``.

    Formerly known as _mo_velocity_advection_stencil_01.
    """
    return astype(_compute_tangential_wind_wp(vn, rbf_vec_coeff_e), vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tangential_wind_wp(
    vn: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    vt: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_tangential_wind_wp(
        vn=vn,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        out=vt,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tangential_wind(
    vn: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    vt: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_tangential_wind(
        vn=vn,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        out=vt,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
