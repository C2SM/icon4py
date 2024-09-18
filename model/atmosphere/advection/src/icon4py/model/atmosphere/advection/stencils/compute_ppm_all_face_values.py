# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast, int32, where

from icon4py.model.atmosphere.advection.stencils.compute_ppm_quadratic_face_values import (
    _compute_ppm_quadratic_face_values,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_ppm_all_face_values(
    p_cc: fa.CellKField[wpfloat],
    p_cellhgt_mc_now: fa.CellKField[wpfloat],
    p_face_in: fa.CellKField[wpfloat],
    k: fa.KField[int32],
    slev: int32,
    elev: int32,
    slevp1: int32,
    elevp1: int32,
) -> fa.CellKField[wpfloat]:
    k = broadcast(k, (dims.CellDim, dims.KDim))

    p_face = where(
        (k == slevp1) | (k == elev),
        _compute_ppm_quadratic_face_values(p_cc, p_cellhgt_mc_now),
        p_face_in,
    )

    p_face = where((k == slev), p_cc, p_face)

    p_face = where((k == elevp1), p_cc(Koff[-1]), p_face)

    return p_face


# TODO (dastrm): slev/elev and vertical_start/end are redundant


@program(grid_type=GridType.UNSTRUCTURED)
def compute_ppm_all_face_values(
    p_cc: fa.CellKField[wpfloat],
    p_cellhgt_mc_now: fa.CellKField[wpfloat],
    p_face_in: fa.CellKField[wpfloat],
    k: fa.KField[int32],
    slev: int32,
    elev: int32,
    slevp1: int32,
    elevp1: int32,
    p_face: fa.CellKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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