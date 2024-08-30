# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@gtx.field_operator
def _face_val_ppm_stencil_05(
    p_cc: fa.CellKField[float],
    p_cellhgt_mc_now: fa.CellKField[float],
    z_slope: fa.CellKField[float],
) -> fa.CellKField[float]:
    zgeo1 = p_cellhgt_mc_now(Koff[-1]) / (p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now)
    zgeo2 = 1.0 / (
        p_cellhgt_mc_now(Koff[-2])
        + p_cellhgt_mc_now(Koff[-1])
        + p_cellhgt_mc_now
        + p_cellhgt_mc_now(Koff[1])
    )
    zgeo3 = (p_cellhgt_mc_now(Koff[-2]) + p_cellhgt_mc_now(Koff[-1])) / (
        2.0 * p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now
    )
    zgeo4 = (p_cellhgt_mc_now(Koff[1]) + p_cellhgt_mc_now) / (
        2.0 * p_cellhgt_mc_now + p_cellhgt_mc_now(Koff[-1])
    )

    p_face = (
        p_cc(Koff[-1])
        + zgeo1 * (p_cc - p_cc(Koff[-1]))
        + zgeo2
        * (
            (2.0 * p_cellhgt_mc_now * zgeo1) * (zgeo3 - zgeo4) * (p_cc - p_cc(Koff[-1]))
            - zgeo3 * p_cellhgt_mc_now(Koff[-1]) * z_slope
            + zgeo4 * p_cellhgt_mc_now * z_slope(Koff[-1])
        )
    )

    return p_face


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def face_val_ppm_stencil_05(
    p_cc: fa.CellKField[float],
    p_cellhgt_mc_now: fa.CellKField[float],
    z_slope: fa.CellKField[float],
    p_face: fa.CellKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _face_val_ppm_stencil_05(
        p_cc,
        p_cellhgt_mc_now,
        z_slope,
        out=p_face,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
