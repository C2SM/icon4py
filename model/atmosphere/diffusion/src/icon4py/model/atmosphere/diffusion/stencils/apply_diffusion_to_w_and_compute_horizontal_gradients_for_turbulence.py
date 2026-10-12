# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.experimental import concat_where

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
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    w_old: fa.CellKHalfField[wpfloat],
    type_shear: gtx.int32,
    dwdx: fa.CellKHalfField[vpfloat],
    dwdy: fa.CellKHalfField[vpfloat],
    diff_multfac_w: wpfloat,
    diff_multfac_n2w: fa.KHalfField[wpfloat],
    nrdmax: gtx.int32,
    interior_idx: gtx.int32,
    halo_idx: gtx.int32,
) -> tuple[
    fa.CellKHalfField[wpfloat],
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
]:
    dwdx, dwdy = (
        _calculate_horizontal_gradients_for_turbulence(w_old, geofac_grg_x, geofac_grg_y)
        if type_shear == 2
        else (dwdx, dwdy)
    )

    z_nabla2_c = _calculate_nabla2_for_w(w_old, geofac_n2s)

    w = _apply_nabla2_to_w(area, z_nabla2_c, geofac_n2s, w_old, diff_multfac_w)

    w = concat_where(
        (0 < dims.KHalfDim)
        & (dims.KHalfDim < nrdmax)
        & (interior_idx <= dims.CellDim)
        & (dims.CellDim < halo_idx),
        _apply_nabla2_to_w_in_upper_damping_layer(w, diff_multfac_n2w, area, z_nabla2_c),
        w,
    )

    return w, dwdx, dwdy
