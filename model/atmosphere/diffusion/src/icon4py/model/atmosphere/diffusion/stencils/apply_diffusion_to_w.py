# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, broadcast
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.math.differential_operators import laplacian_on_cell_khalf
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _apply_nabla2_to_w(
    area: fa.CellField[wpfloat],
    z_nabla2_c: fa.CellKHalfField[vpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    w: fa.CellKHalfField[wpfloat],
    diff_multfac_w: wpfloat,
) -> fa.CellKHalfField[wpfloat]:
    z_nabla2_c_wp = astype(z_nabla2_c, wpfloat)

    w_wp = w - diff_multfac_w * (area * area) * laplacian_on_cell_khalf(z_nabla2_c_wp, geofac_n2s)
    return w_wp


@gtx.field_operator
def _apply_nabla2_to_w_in_upper_damping_layer(
    w: fa.CellKHalfField[wpfloat],
    diff_multfac_n2w: fa.KHalfField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    z_nabla2_c: fa.CellKHalfField[vpfloat],
) -> fa.CellKHalfField[wpfloat]:
    z_nabla2_c_wp = astype(z_nabla2_c, wpfloat)
    cell_area_tmp = broadcast(cell_area, (dims.CellDim, dims.KHalfDim))

    w_wp = w + diff_multfac_n2w * cell_area_tmp * z_nabla2_c_wp
    return w_wp


@gtx.field_operator
def _apply_diffusion_to_w(
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    w_old: fa.CellKHalfField[wpfloat],
    diff_multfac_w: wpfloat,
    diff_multfac_n2w: fa.KHalfField[wpfloat],
    nrdmax: gtx.int32,
    interior_idx: gtx.int32,
    halo_idx: gtx.int32,
) -> fa.CellKHalfField[wpfloat]:
    z_nabla2_c = astype(laplacian_on_cell_khalf(w_old, geofac_n2s), vpfloat)

    w = _apply_nabla2_to_w(area, z_nabla2_c, geofac_n2s, w_old, diff_multfac_w)

    w = concat_where(
        (0 < dims.KHalfDim)
        & (dims.KHalfDim < nrdmax)
        & (interior_idx <= dims.CellDim)
        & (dims.CellDim < halo_idx),
        _apply_nabla2_to_w_in_upper_damping_layer(w, diff_multfac_n2w, area, z_nabla2_c),
        w,
    )

    return w
