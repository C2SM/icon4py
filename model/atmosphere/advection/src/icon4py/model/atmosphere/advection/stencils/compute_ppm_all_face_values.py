# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import where

from icon4py.model.atmosphere.advection.stencils.compute_ppm_quadratic_face_values import (
    _compute_ppm_quadratic_face_values,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend


# TODO (dastrm): this stencil is imported but never called
# TODO (dastrm): slev/elev and vertical_start/end are redundant


@gtx.field_operator
def _compute_ppm_all_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    p_face_in: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slev: gtx.int32,
    elev: gtx.int32,
    slevp1: gtx.int32,
    elevp1: gtx.int32,
) -> fa.CellKField[ta.wpfloat]:
    p_face = where(
        (k == slevp1) | (k == elev),
        _compute_ppm_quadratic_face_values(p_cc, p_cellhgt_mc_now),
        p_face_in,
    )

    p_face = where((k == slev), p_cc, p_face)

    p_face = where((k == elevp1), p_cc(Koff[-1]), p_face)

    return p_face


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def compute_ppm_all_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    p_face_in: fa.CellKField[ta.wpfloat],
    p_face: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slev: gtx.int32,
    elev: gtx.int32,
    slevp1: gtx.int32,
    elevp1: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_ppm_all_face_values(
        p_cc,
        p_cellhgt_mc_now,
        p_face_in,
        k,
        slev,
        elev,
        slevp1,
        elevp1,
        out=p_face,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
