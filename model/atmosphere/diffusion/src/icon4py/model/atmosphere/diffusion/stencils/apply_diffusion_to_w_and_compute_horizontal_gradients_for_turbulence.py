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
from gt4py.next.ffront.fbuiltins import broadcast, where

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w import _apply_nabla2_to_w
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w_in_upper_damping_layer import (
    _apply_nabla2_to_w_in_upper_damping_layer,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_horizontal_gradients_for_turbulence import (
    _calculate_horizontal_gradients_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_w import (
    _calculate_nabla2_for_w,
)
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    w_old: fa.CellKField[wpfloat],
    type_shear: gtx.int32,
    dwdx: fa.CellKField[vpfloat],
    dwdy: fa.CellKField[vpfloat],
    diff_multfac_w: wpfloat,
    diff_multfac_n2w: fa.KField[wpfloat],
    k: fa.KField[gtx.int32],
    cell: fa.CellField[gtx.int32],
    nrdmax: gtx.int32,
    interior_idx: gtx.int32,
    halo_idx: gtx.int32,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    k = broadcast(k, (CellDim, KDim))
    dwdx, dwdy = (
        where(
            0 < k,
            _calculate_horizontal_gradients_for_turbulence(w_old, geofac_grg_x, geofac_grg_y),
            (dwdx, dwdy),
        )
        if type_shear == 2
        else (dwdx, dwdy)
    )

    z_nabla2_c = _calculate_nabla2_for_w(w_old, geofac_n2s)

    w = where(
        (interior_idx <= cell) & (cell < halo_idx),
        _apply_nabla2_to_w(area, z_nabla2_c, geofac_n2s, w_old, diff_multfac_w),
        w_old,
    )

    w = where(
        (0 < k) & (k < nrdmax) & (interior_idx <= cell) & (cell < halo_idx),
        _apply_nabla2_to_w_in_upper_damping_layer(w, diff_multfac_n2w, area, z_nabla2_c),
        w,
    )

    return w, dwdx, dwdy


@program(grid_type=GridType.UNSTRUCTURED)
def apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    w_old: fa.CellKField[wpfloat],
    w: fa.CellKField[wpfloat],
    type_shear: gtx.int32,
    dwdx: fa.CellKField[vpfloat],
    dwdy: fa.CellKField[vpfloat],
    diff_multfac_w: wpfloat,
    diff_multfac_n2w: fa.KField[wpfloat],
    k: fa.KField[gtx.int32],
    cell: fa.CellField[gtx.int32],
    nrdmax: gtx.int32,
    interior_idx: gtx.int32,
    halo_idx: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
        area,
        geofac_n2s,
        geofac_grg_x,
        geofac_grg_y,
        w_old,
        type_shear,
        dwdx,
        dwdy,
        diff_multfac_w,
        diff_multfac_n2w,
        k,
        cell,
        nrdmax,
        interior_idx,
        halo_idx,
        out=(w, dwdx, dwdy),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
