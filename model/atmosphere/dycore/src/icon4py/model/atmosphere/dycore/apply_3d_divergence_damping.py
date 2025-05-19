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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, broadcast, int32

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_3d_divergence_damping_to_w(
    scal_divdamp_half: Field[[KDim], wpfloat],
    graddiv_vertical: Field[[CellDim, KDim], vpfloat],
    z_w_divdamp: Field[[CellDim, KDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:
    graddiv_vertical_wp = astype(graddiv_vertical, wpfloat)
    scal_divdamp_half = broadcast(scal_divdamp_half, (CellDim, KDim))
    z_w_divdamp_wp = z_w_divdamp + scal_divdamp_half * graddiv_vertical_wp
    return z_w_divdamp_wp


@field_operator
def _apply_3d_divergence_damping_to_vn(
    scal_divdamp: Field[[KDim], wpfloat],
    graddiv_normal: Field[[EdgeDim, KDim], vpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
) -> Field[[EdgeDim, KDim], wpfloat]:
    graddiv_normal_wp = astype(graddiv_normal, wpfloat)
    scal_divdamp = broadcast(scal_divdamp, (EdgeDim, KDim))
    vn_wp = vn + (scal_divdamp * graddiv_normal_wp)
    return vn_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_3d_divergence_damping(
    scal_divdamp: Field[[KDim], wpfloat],
    scal_divdamp_half: Field[[KDim], wpfloat],
    graddiv_normal: Field[[EdgeDim, KDim], vpfloat],
    graddiv_vertical: Field[[CellDim, KDim], vpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    z_w_divdamp: Field[[CellDim, KDim], wpfloat],
    edge_horizontal_start: int32,
    edge_horizontal_end: int32,
    cell_horizontal_start: int32,
    cell_horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_3d_divergence_damping_to_vn(
        scal_divdamp,
        graddiv_normal,
        vn,
        out=vn,
        domain={
            EdgeDim: (edge_horizontal_start, edge_horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
    _apply_3d_divergence_damping_to_w(
        scal_divdamp_half,
        graddiv_vertical,
        z_w_divdamp,
        out=z_w_divdamp,
        domain={
            CellDim: (cell_horizontal_start, cell_horizontal_end),
            KDim: (vertical_start + 1, vertical_end),
        },
    )


@field_operator
def _apply_3d_divergence_damping_only_to_w(
    w: Field[[CellDim, KDim], wpfloat],
    z_w_divdamp: Field[[CellDim, KDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:
    return w + z_w_divdamp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_3d_divergence_damping_only_to_w(
    w: Field[[CellDim, KDim], wpfloat],
    z_w_divdamp: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_3d_divergence_damping_only_to_w(
        w,
        z_w_divdamp,
        out=w,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
