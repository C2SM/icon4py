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
from gt4py.next.ffront.fbuiltins import Field, astype, broadcast, int32, neighbor_sum, where

from icon4py.model.atmosphere.dycore.interpolate_to_cell_center import _interpolate_to_cell_center
from icon4py.model.atmosphere.dycore.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp,
)
from icon4py.model.common.dimension import E2C2E, CEDim, CellDim, E2C2EDim, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_tangential_wind_and_contravariant_at_edge(
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], wpfloat],
    ddxn_z_full: Field[[EdgeDim, KDim], vpfloat],
    ddxt_z_full: Field[[EdgeDim, KDim], vpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
) -> tuple[Field[[EdgeDim, KDim], vpfloat], Field[[EdgeDim, KDim], vpfloat]]:
    vt_wp = neighbor_sum(rbf_vec_coeff_e * vn(E2C2E), axis=E2C2EDim)

    ddxn_z_full_wp = astype(ddxn_z_full, wpfloat)
    z_w_concorr_me_wp = vn * ddxn_z_full_wp + astype(vt_wp * ddxt_z_full, wpfloat)
    return astype(vt_wp, vpfloat), astype(z_w_concorr_me_wp, vpfloat)


@field_operator
def _compute_contravariant_at_cell(
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CEDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nflatlev_startindex: int32,
    nlev: int32,
) -> Field[[CellDim, KDim], float]:
    local_z_w_concorr_mc = where(
        (k_field >= nflatlev_startindex) & (k_field < nlev),
        _interpolate_to_cell_center(z_w_concorr_me, e_bln_c_s),
        broadcast(0.0, (CellDim, KDim)),
    )

    w_concorr_c = where(
        (k_field >= nflatlev_startindex + int32(1)) & (k_field < nlev),
        _interpolate_to_half_levels_vp(interpolant=local_z_w_concorr_mc, wgtfac_c=wgtfac_c),
        w_concorr_c,
    )

    return w_concorr_c


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_tangential_wind_and_contravariant(
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], wpfloat],
    ddxn_z_full: Field[[EdgeDim, KDim], vpfloat],
    ddxt_z_full: Field[[EdgeDim, KDim], vpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    e_bln_c_s: Field[[CEDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    vt: Field[[EdgeDim, KDim], wpfloat],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nflatlev_startindex: int32,
    nlev: int32,
    edge_horizontal_start: int32,
    edge_horizontal_end: int32,
    cell_horizontal_start: int32,
    cell_horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_tangential_wind_and_contravariant_at_edge(
        rbf_vec_coeff_e,
        ddxn_z_full,
        ddxt_z_full,
        vn,
        out=(vt, z_w_concorr_me),
        domain={
            EdgeDim: (edge_horizontal_start, edge_horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
    _compute_contravariant_at_cell(
        z_w_concorr_me,
        e_bln_c_s,
        wgtfac_c,
        w_concorr_c,
        k_field,
        nflatlev_startindex,
        nlev,
        out=w_concorr_c,
        domain={
            CellDim: (cell_horizontal_start, cell_horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
