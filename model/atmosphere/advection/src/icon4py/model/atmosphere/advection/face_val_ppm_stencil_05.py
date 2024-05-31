# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import Koff


@field_operator
def _face_val_ppm_stencil_05(
    p_cc: fa.CKfloatField,
    p_cellhgt_mc_now: fa.CKfloatField,
    z_slope: fa.CKfloatField,
) -> fa.CKfloatField:
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


@program(grid_type=GridType.UNSTRUCTURED)
def face_val_ppm_stencil_05(
    p_cc: fa.CKfloatField,
    p_cellhgt_mc_now: fa.CKfloatField,
    z_slope: fa.CKfloatField,
    p_face: fa.CKfloatField,
):
    _face_val_ppm_stencil_05(
        p_cc,
        p_cellhgt_mc_now,
        z_slope,
        out=p_face,
    )
