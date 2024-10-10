# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import broadcast, maximum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend


# TODO (dastrm): this stencil has no test


@gtx.field_operator
def _apply_density_increment(
    rhodz_in: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    p_dtime: ta.wpfloat,
    even_timestep: bool,
) -> fa.CellKField[ta.wpfloat]:
    even = broadcast(even_timestep, (dims.CellDim, dims.KDim))
    rhodz_incr = p_dtime * (
        p_mflx_contra_v(Koff[1]) * deepatmo_divzl - p_mflx_contra_v * deepatmo_divzu
    )
    rhodz_out = where(even, rhodz_in + rhodz_incr, maximum(0.1 * rhodz_in, rhodz_in) - rhodz_incr)
    return rhodz_out


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def apply_density_increment(
    rhodz_in: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    p_dtime: ta.wpfloat,
    even_timestep: bool,
    rhodz_out: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_density_increment(
        rhodz_in,
        p_mflx_contra_v,
        deepatmo_divzl,
        deepatmo_divzu,
        p_dtime,
        even_timestep,
        out=rhodz_out,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
